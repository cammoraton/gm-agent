"""Lightweight chat agent for GM assistance without campaign state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .mcp.base import ToolResult
from .mcp.client import MCPClient
from .models.base import LLMBackend, Message, ToolCall
from .models.factory import get_backend

if TYPE_CHECKING:
    from .mcp.foundry_vtt import FoundryVTTServer

# System prompt for chat mode (no campaign context)
CHAT_SYSTEM_PROMPT = """You are an assistant for Pathfinder 2nd Edition (Remaster) - NOT D&D or other systems.

CRITICAL RULES:
1. ONLY state facts that come directly from your search tool results
2. If a search returns limited or no results, say "I couldn't find detailed information about X in the database"
3. NEVER fabricate details, stats, locations, NPCs, or lore not in the search results
4. NEVER reference D&D content (no Lolth, Forgotten Realms, etc.) - this is Pathfinder/Golarion only
5. When uncertain, say so clearly rather than guessing

Your tools:
- lookup_creature, lookup_spell, lookup_item: Look up specific game content by name
- search_rules: Search for rules and mechanics
- search_lore: Search for world lore, locations, history, nations (USE THIS for places like Absalom, Cheliax, etc.)
- search_content: General search with optional type filters
- evaluate_encounter, suggest_encounter: Encounter building and evaluation
- roll_dice, roll_check: Dice rolling with PF2e degree of success
- add_note, search_notes: Record and recall session notes

When answering questions about Pathfinder content:
1. Use the search tools to find official information
2. Quote or summarize ONLY what the tools return
3. If results are sparse, acknowledge the limitation
4. For creative suggestions (encounters, plot hooks), clearly label them as your suggestions vs. official content

For encounter evaluation, use evaluate_encounter with creature levels.
For dice, use roll_dice (e.g., "2d6+4") or roll_check (modifier vs DC).

Be concise and accurate. Accuracy is more important than completeness.
"""


class ChatAgent:
    """A lightweight chat agent with RAG, encounter, dice, and notes tools.

    Uses MCPClient for tool execution, which supports both local and remote modes.
    Available tools:
    - Quick rules lookups
    - GM prep assistance
    - Encounter evaluation and building
    - Dice rolling and check resolution
    - Session note-taking
    - General Pathfinder 2e questions
    - Foundry VTT integration (when connected)
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        verbose: bool = False,
        system_prompt: str | None = None,
        enable_rag: bool = True,
        enable_encounters: bool = True,
        enable_dice: bool = True,
        enable_notes: bool = True,
        foundry_server: "FoundryVTTServer | None" = None,
    ):
        self.llm = llm or get_backend()
        self.verbose = verbose
        self.system_prompt = system_prompt or CHAT_SYSTEM_PROMPT
        self.foundry_server = foundry_server

        # Store enable flags for reference
        self._enable_rag = enable_rag
        self._enable_encounters = enable_encounters
        self._enable_dice = enable_dice
        self._enable_notes = enable_notes

        # Conversation history
        self._messages: list[Message] = [Message(role="system", content=self.system_prompt)]

        # Initialize MCP client (no campaign context for chat)
        self._mcp = MCPClient(
            context={},
            foundry_server=foundry_server,
        )

        # Convenience accessors for servers (backward compatibility)
        self.rag_server = self._mcp.get_server("pf2e-rag") if enable_rag else None
        self.encounter_server = self._mcp.get_server("encounter") if enable_encounters else None
        self.dice_server = self._mcp.get_server("dice") if enable_dice else None
        self.notes_server = self._mcp.get_server("notes") if enable_notes else None

    def get_tools(self) -> list:
        """Get all available tools from MCPClient."""
        return self._mcp.list_tools()

    def chat(self, user_input: str) -> str:
        """Process user input and return a response.

        Args:
            user_input: The user's message

        Returns:
            The assistant's response
        """
        # Add user message to history
        self._messages.append(Message(role="user", content=user_input))

        # Get available tools from all servers
        tools = self.get_tools()

        # Agent loop - handle tool calls
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get LLM response
            response = self.llm.chat(self._messages, tools=tools)

            # If no tool calls, we're done
            if not response.tool_calls:
                break

            # Process tool calls
            if self.verbose:
                print(f"\n[Tool calls: {[tc.name for tc in response.tool_calls]}]")

            # Add assistant message with tool calls
            self._messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute each tool call
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)

                if self.verbose:
                    preview = result.to_string()[:100]
                    print(f"  {tool_call.name}({tool_call.args}) -> {preview}...")

                # Add tool result
                self._messages.append(
                    Message(
                        role="tool",
                        content=result.to_string(),
                        tool_call_id=tool_call.id,
                    )
                )

        # Add final response to history
        self._messages.append(Message(role="assistant", content=response.text))

        return response.text

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call via MCPClient."""
        return self._mcp.call_tool(tool_call.name, tool_call.args)

    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self._messages = [Message(role="system", content=self.system_prompt)]

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self._messages.copy()

    def close(self) -> None:
        """Clean up resources."""
        self._mcp.close()
