"""Core GM Agent loop."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Iterator

from .config import PARALLEL_TOOL_CALLS
from .context import build_context
from .mcp.base import ToolDef, ToolResult
from .mcp.client import MCPClient
from .models.base import LLMBackend, Message, StreamChunk, ToolCall
from .models.factory import get_backend
from .storage.campaign import campaign_store
from .storage.session import session_store
from .storage.schemas import Campaign, Session, ToolCallRecord, TurnMetadata
from .summarizer import RollingSummarizer, should_update_summary

if TYPE_CHECKING:
    from .mcp.foundry_vtt import FoundryVTTServer


class GMAgent:
    """The core GM agent that processes player input and generates responses.

    Uses MCPClient for tool execution, which supports both local and remote modes.
    Available tools depend on the MCP servers configured:
    - PF2eRAGServer: Rules, creatures, spells, items lookup
    - CampaignStateServer: Scene management, event logging, history search
    - CharacterRunnerServer: NPC, monster, and player embodiment
    - FoundryVTTServer: Foundry VTT integration (when connected)
    """

    def __init__(
        self,
        campaign_id: str,
        llm: LLMBackend | None = None,
        verbose: bool = False,
        enable_rag: bool = True,
        enable_campaign_state: bool = True,
        enable_character_runner: bool = True,
        auto_summarize: bool = True,
        foundry_server: "FoundryVTTServer | None" = None,
    ):
        self.campaign = campaign_store.get(campaign_id)
        if not self.campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found")

        self.session = session_store.get_or_start(campaign_id)
        self.llm = llm or get_backend()
        self.verbose = verbose
        self.auto_summarize = auto_summarize

        # Store server enable flags for reference
        self._enable_rag = enable_rag
        self._enable_campaign_state = enable_campaign_state
        self._enable_character_runner = enable_character_runner

        # Initialize MCP client with context
        self._mcp = MCPClient(
            context={
                "campaign_id": campaign_id,
                "llm": self.llm,
            },
            foundry_server=foundry_server,
        )

        # Store Foundry server reference for external access
        self.foundry_server = foundry_server

        # Convenience accessors for servers (backward compatibility)
        self.rag_server = self._mcp.get_server("pf2e-rag") if enable_rag else None
        self.campaign_state_server = (
            self._mcp.get_server("campaign-state") if enable_campaign_state else None
        )
        self.character_runner_server = (
            self._mcp.get_server("character-runner") if enable_character_runner else None
        )

        # Initialize summarizer
        if auto_summarize:
            self._summarizer = RollingSummarizer(self.llm)
        else:
            self._summarizer = None

    def set_foundry_server(self, foundry_server: "FoundryVTTServer | None") -> None:
        """Set or update the Foundry VTT server.

        This allows updating the server after agent creation, e.g., when
        Foundry connects after the agent was already created.

        Args:
            foundry_server: The FoundryVTTServer instance, or None to remove
        """
        self.foundry_server = foundry_server
        self._mcp.set_foundry_server(foundry_server)

    def _get_all_tools(self) -> list[ToolDef]:
        """Get combined tools from all servers."""
        return self._mcp.list_tools()

    def process_turn(self, player_input: str, metadata: TurnMetadata | None = None) -> str:
        """
        Process a player turn and return the GM response.

        This handles:
        1. Building context from campaign and session
        2. Sending to LLM with available tools
        3. Executing any tool calls
        4. Getting final response
        5. Recording the turn with metadata
        6. Updating rolling summary if needed

        Args:
            player_input: The player's input text
            metadata: Optional metadata for analytics/fine-tuning
                (source, player info, etc.). Processing time will
                be automatically recorded.

        Returns:
            The GM's response text
        """
        start_time = time.time()

        # Build context messages
        context_messages = build_context(self.campaign, self.session)

        # Add the new player input
        messages = context_messages + [Message(role="user", content=player_input)]

        # Get available tools from all servers
        tools = self._get_all_tools()

        # Track tool calls for this turn
        tool_call_records: list[ToolCallRecord] = []
        tool_usage: dict[str, int] = {}  # Track usage counts per tool
        tool_failures: list[str] = []  # Track failed tool calls

        # Main agent loop - handle tool calls
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get LLM response
            response = self.llm.chat(messages, tools=tools)

            # If no tool calls, we're done
            if not response.tool_calls:
                break

            # Process tool calls
            if self.verbose:
                print(f"\n[Tool calls: {[tc.name for tc in response.tool_calls]}]")

            # Add assistant message with tool calls
            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute tool calls (parallel or sequential based on config)
            if PARALLEL_TOOL_CALLS and len(response.tool_calls) > 1:
                # Parallel execution
                results = self._execute_tools_parallel(response.tool_calls)
            else:
                # Sequential execution (default)
                results = [
                    (tool_call, self._execute_tool(tool_call)) for tool_call in response.tool_calls
                ]

            # Process results in order
            for tool_call, result in results:
                if self.verbose:
                    print(f"  {tool_call.name}({tool_call.args}) -> {result.success}")

                # Track tool usage
                tool_usage[tool_call.name] = tool_usage.get(tool_call.name, 0) + 1

                # Track failures
                if not result.success:
                    tool_failures.append(tool_call.name)

                # Record the tool call
                tool_call_records.append(
                    ToolCallRecord(
                        name=tool_call.name,
                        args=tool_call.args,
                        result=result.to_string(),
                    )
                )

                # Add tool result to messages
                messages.append(
                    Message(
                        role="tool",
                        content=result.to_string(),
                        tool_call_id=tool_call.id,
                    )
                )

        # Get the final response text
        final_response = response.text

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build metadata with timing info and tool analytics
        turn_metadata = metadata or TurnMetadata()
        turn_metadata.processing_time_ms = processing_time_ms
        turn_metadata.tool_count = len(tool_call_records)
        turn_metadata.tool_usage = tool_usage
        turn_metadata.tool_failures = tool_failures
        turn_metadata.model = self.llm.get_model_name()

        # Record the turn
        session_store.add_turn(
            campaign_id=self.campaign.id,
            player_input=player_input,
            gm_response=final_response,
            tool_calls=tool_call_records if tool_call_records else None,
            metadata=turn_metadata,
        )

        # Refresh session state
        self.session = session_store.get_current(self.campaign.id)

        # Update rolling summary if needed
        if self._summarizer and should_update_summary(self.session):
            self._update_rolling_summary()

        return final_response

    def process_turn_stream(
        self, player_input: str, metadata: TurnMetadata | None = None
    ) -> Iterator[StreamChunk]:
        """
        Process a player turn and yield streaming response chunks.

        This handles:
        1. Building context from campaign and session
        2. Streaming from LLM with available tools
        3. Buffering chunks until tool calls are detected
        4. Executing tool calls synchronously when detected
        5. Continuing streaming with tool results
        6. Recording the complete turn with metadata

        Args:
            player_input: The player's input text
            metadata: Optional metadata for analytics/fine-tuning

        Yields:
            StreamChunk objects with incremental content

        Note:
            Tool calls will pause the stream while they execute,
            then streaming continues with the tool results in context.
        """
        start_time = time.time()

        # Build context messages
        context_messages = build_context(self.campaign, self.session)

        # Add the new player input
        messages = context_messages + [Message(role="user", content=player_input)]

        # Get available tools from all servers
        tools = self._get_all_tools()

        # Track tool calls and analytics for this turn
        tool_call_records: list[ToolCallRecord] = []
        tool_usage: dict[str, int] = {}
        tool_failures: list[str] = []
        accumulated_text = ""

        # Main agent loop - handle tool calls
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Stream LLM response
            chunks_buffer = []
            current_tool_calls = []
            current_finish_reason = "stop"
            current_usage = {}

            for chunk in self.llm.chat_stream(messages, tools=tools):
                # Buffer text chunks
                if chunk.delta:
                    accumulated_text += chunk.delta
                    chunks_buffer.append(chunk)

                # Collect tool calls
                if chunk.tool_calls:
                    current_tool_calls.extend(chunk.tool_calls)

                # Track finish reason and usage from final chunk
                if chunk.finish_reason:
                    current_finish_reason = chunk.finish_reason
                if chunk.usage:
                    current_usage = chunk.usage

            # If no tool calls, yield all buffered chunks and we're done
            if not current_tool_calls:
                for buffered_chunk in chunks_buffer:
                    yield buffered_chunk
                break

            # Tool calls detected - yield buffered text, then pause for execution
            for buffered_chunk in chunks_buffer:
                yield buffered_chunk

            if self.verbose:
                print(f"\n[Tool calls: {[tc.name for tc in current_tool_calls]}]")

            # Add assistant message with tool calls
            messages.append(
                Message(
                    role="assistant",
                    content=accumulated_text,
                    tool_calls=current_tool_calls,
                )
            )

            # Execute tool calls synchronously
            if PARALLEL_TOOL_CALLS and len(current_tool_calls) > 1:
                results = self._execute_tools_parallel(current_tool_calls)
            else:
                results = [
                    (tool_call, self._execute_tool(tool_call)) for tool_call in current_tool_calls
                ]

            # Process results and add to messages
            for tool_call, result in results:
                if self.verbose:
                    print(f"  {tool_call.name}({tool_call.args}) -> {result.success}")

                # Track tool usage
                tool_usage[tool_call.name] = tool_usage.get(tool_call.name, 0) + 1

                # Track failures
                if not result.success:
                    tool_failures.append(tool_call.name)

                # Record the tool call
                tool_call_records.append(
                    ToolCallRecord(
                        name=tool_call.name,
                        args=tool_call.args,
                        result=result.to_string(),
                    )
                )

                # Add tool result to messages
                messages.append(
                    Message(
                        role="tool",
                        content=result.to_string(),
                        tool_call_id=tool_call.id,
                    )
                )

            # Reset accumulated text for next iteration
            accumulated_text = ""

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build metadata with timing info and tool analytics
        turn_metadata = metadata or TurnMetadata()
        turn_metadata.processing_time_ms = processing_time_ms
        turn_metadata.tool_count = len(tool_call_records)
        turn_metadata.tool_usage = tool_usage
        turn_metadata.tool_failures = tool_failures
        turn_metadata.model = self.llm.get_model_name()

        # Record the turn (use accumulated_text as final response)
        session_store.add_turn(
            campaign_id=self.campaign.id,
            player_input=player_input,
            gm_response=accumulated_text,
            tool_calls=tool_call_records if tool_call_records else None,
            metadata=turn_metadata,
        )

        # Refresh session state
        self.session = session_store.get_current(self.campaign.id)

        # Update rolling summary if needed
        if self._summarizer and should_update_summary(self.session):
            self._update_rolling_summary()

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        Routes to the appropriate MCP server via MCPClient.
        """
        return self._mcp.call_tool(tool_call.name, tool_call.args)

    def _execute_tools_parallel(
        self, tool_calls: list[ToolCall]
    ) -> list[tuple[ToolCall, ToolResult]]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of (tool_call, result) tuples in the same order as input
        """
        results: dict[str, tuple[ToolCall, ToolResult]] = {}

        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            # Submit all tool calls
            future_to_call = {executor.submit(self._execute_tool, tc): tc for tc in tool_calls}

            # Collect results as they complete
            for future in as_completed(future_to_call):
                tool_call = future_to_call[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ToolResult(success=False, error=f"Tool execution failed: {e}")
                results[tool_call.id] = (tool_call, result)

        # Return in original order
        return [results[tc.id] for tc in tool_calls]

    def _update_rolling_summary(self) -> None:
        """Update the session's rolling summary."""
        if not self._summarizer:
            return

        try:
            new_summary = self._summarizer.maybe_update(self.session, force=False)
            if new_summary:
                self.session.summary = new_summary
                session_store._save_current(self.session)
                if self.verbose:
                    print(f"\n[Rolling summary updated]")
        except Exception as e:
            if self.verbose:
                print(f"\n[Summary update failed: {e}]")

    def end_session(self, summary: str = "", auto_generate: bool = False) -> Session | None:
        """End the current session.

        Args:
            summary: Optional summary text. If empty and auto_generate is True,
                    will generate a summary using the LLM.
            auto_generate: If True and no summary provided, generate one.

        Returns:
            The ended session, or None if no active session.
        """
        if not summary and auto_generate and self._summarizer:
            try:
                summary = self._summarizer.maybe_update(self.session, force=True) or ""
            except Exception:
                pass  # Use empty summary if generation fails

        return session_store.end(self.campaign.id, summary)

    def generate_summary(self) -> str:
        """Force-generate a summary of the current session.

        Returns:
            The generated summary text.
        """
        if not self._summarizer:
            from .summarizer import generate_summary

            return generate_summary(self.llm, self.session)

        return self._summarizer.maybe_update(self.session, force=True) or ""

    def update_scene(self, **kwargs) -> None:
        """Update the current scene state."""
        from .storage.schemas import SceneState

        current_scene = self.session.scene_state
        updated = SceneState(
            location=kwargs.get("location", current_scene.location),
            npcs_present=kwargs.get("npcs_present", current_scene.npcs_present),
            time_of_day=kwargs.get("time_of_day", current_scene.time_of_day),
            conditions=kwargs.get("conditions", current_scene.conditions),
            notes=kwargs.get("notes", current_scene.notes),
        )
        session_store.update_scene(self.campaign.id, updated)
        self.session = session_store.get_current(self.campaign.id)

    def close(self) -> None:
        """Clean up resources."""
        self._mcp.close()
