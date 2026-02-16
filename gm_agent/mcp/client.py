"""Unified MCP client with dual-mode support.

Supports two modes:
- local: Direct MCP execution (standalone, in-process servers)
- remote: CLI -> API (MCP facade) -> Celery -> MCP Workers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from ..config import MCP_MODE, MCP_API_URL
from .base import MCPServer, ToolDef, ToolResult
from .registry import (
    SERVERS,
    TOOL_TO_SERVER,
    get_server_for_tool,
    get_server_info,
    register_foundry_tools,
)

if TYPE_CHECKING:
    from .foundry_vtt import FoundryVTTServer

logger = logging.getLogger(__name__)


class MCPClient:
    """Unified MCP client with dual-mode support.

    In local mode, servers are instantiated in-process.
    In remote mode, tool calls are routed to the API server.
    """

    def __init__(
        self,
        mode: str | None = None,
        context: dict[str, Any] | None = None,
        foundry_server: "FoundryVTTServer | None" = None,
    ):
        """Initialize the MCP client.

        Args:
            mode: "local" or "remote". Defaults to MCP_MODE config.
            context: Context dict with campaign_id, session_id, etc.
            foundry_server: Optional Foundry VTT server instance (local mode only)
        """
        self.mode = mode or MCP_MODE
        self.context = context or {}
        self._foundry_server = foundry_server

        # Local mode: instantiated servers
        self._local_servers: dict[str, MCPServer] = {}
        self._tool_to_server: dict[str, MCPServer] = {}

        # Remote mode: API session
        self._session: requests.Session | None = None

        if self.mode == "local":
            self._init_local_servers()
        else:
            self._init_remote_session()

    def _init_local_servers(self) -> None:
        """Initialize local MCP servers based on context."""
        # Import here to avoid circular imports
        from .pf2e_rag import PF2eRAGServer
        from .dice import DiceServer
        from .encounter import EncounterServer
        from .notes import NotesServer
        from .campaign_state import CampaignStateServer
        from .character_runner import CharacterRunnerServer
        from .npc_builder import NPCBuilderServer
        from .npc_knowledge import NPCKnowledgeServer
        from .creature_modifier import CreatureModifierServer
        from .subsystem import SubsystemServer

        campaign_id = self.context.get("campaign_id")
        llm = self.context.get("llm")

        # Load campaign books for scoped searches
        campaign_books: list[str] = []
        if campaign_id:
            from ..storage.campaign import CampaignStore
            try:
                campaign = CampaignStore().get(campaign_id)
                if campaign:
                    campaign_books = campaign.books
            except Exception:
                pass

        # Always create stateless servers
        self._local_servers["pf2e-rag"] = PF2eRAGServer(campaign_books=campaign_books)
        self._local_servers["dice"] = DiceServer()
        self._local_servers["encounter"] = EncounterServer()
        self._local_servers["notes"] = NotesServer()
        self._local_servers["creature-modifier"] = CreatureModifierServer()

        # Create campaign-dependent servers if campaign_id provided
        if campaign_id:
            self._local_servers["campaign-state"] = CampaignStateServer(campaign_id)
            self._local_servers["character-runner"] = CharacterRunnerServer(campaign_id, llm=llm)
            self._local_servers["npc-builder"] = NPCBuilderServer(campaign_id, llm=llm)
            self._local_servers["npc-knowledge"] = NPCKnowledgeServer(campaign_id)
            self._local_servers["subsystem"] = SubsystemServer(campaign_id)

        # Add Foundry server if provided
        if self._foundry_server:
            self._local_servers["foundry-vtt"] = self._foundry_server

        # Build tool routing map
        self._build_tool_map()

    def _build_tool_map(self) -> None:
        """Build mapping from tool names to their servers."""
        self._tool_to_server.clear()
        for server in self._local_servers.values():
            for tool in server.list_tools():
                self._tool_to_server[tool.name] = server

    def _init_remote_session(self) -> None:
        """Initialize remote API session."""
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
            }
        )

    def set_foundry_server(self, foundry_server: "FoundryVTTServer | None") -> None:
        """Set or update the Foundry VTT server (local mode only).

        Args:
            foundry_server: The FoundryVTTServer instance, or None to remove
        """
        if self.mode != "local":
            return

        # Remove old server if present
        if "foundry-vtt" in self._local_servers:
            del self._local_servers["foundry-vtt"]

        # Add new server
        if foundry_server:
            self._foundry_server = foundry_server
            self._local_servers["foundry-vtt"] = foundry_server
        else:
            self._foundry_server = None

        # Rebuild tool map
        self._build_tool_map()

    def list_tools(self) -> list[ToolDef]:
        """List all available tools.

        Returns:
            List of tool definitions
        """
        if self.mode == "local":
            return self._list_tools_local()
        else:
            return self._list_tools_remote()

    def _list_tools_local(self) -> list[ToolDef]:
        """List tools from local servers."""
        tools = []
        for server in self._local_servers.values():
            tools.extend(server.list_tools())
        return tools

    def _list_tools_remote(self) -> list[ToolDef]:
        """List tools from remote API."""
        try:
            response = self._session.get(
                f"{MCP_API_URL}/api/mcp/tools",
                params={"context": self.context} if self.context else None,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            tools = []
            for tool_data in data.get("tools", []):
                tools.append(ToolDef(**tool_data))
            return tools
        except Exception as e:
            logger.error(f"Failed to list remote tools: {e}")
            return []

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            ToolResult with success status and data/error
        """
        if self.mode == "local":
            return self._call_tool_local(name, args)
        else:
            return self._call_tool_remote(name, args)

    def _call_tool_local(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute tool locally via in-process server."""
        server = self._tool_to_server.get(name)
        if server:
            return server.call_tool(name, args)
        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _call_tool_remote(
        self, name: str, args: dict[str, Any], async_mode: bool = False
    ) -> ToolResult:
        """Execute tool via remote API.

        Args:
            name: Tool name
            args: Tool arguments
            async_mode: If True, return task_id for async execution

        Returns:
            ToolResult with data or task_id for async
        """
        try:
            payload = {
                "tool": name,
                "args": args,
                "context": self.context,
                "async": async_mode,
            }

            response = self._session.post(
                f"{MCP_API_URL}/api/mcp/call",
                json=payload,
                timeout=120,  # Tool execution can take time
            )
            response.raise_for_status()
            data = response.json()

            if async_mode:
                return ToolResult(
                    success=True,
                    data={"task_id": data.get("task_id")},
                )

            if data.get("success"):
                return ToolResult(success=True, data=data.get("data"))
            else:
                return ToolResult(success=False, error=data.get("error"))

        except requests.exceptions.Timeout:
            return ToolResult(success=False, error="Tool execution timed out")
        except requests.exceptions.RequestException as e:
            return ToolResult(success=False, error=f"API request failed: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {e}")

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Check status of an async task (remote mode only).

        Args:
            task_id: The Celery task ID

        Returns:
            Dict with status, result, and error fields
        """
        if self.mode == "local":
            return {
                "status": "error",
                "error": "Async tasks not supported in local mode",
            }

        try:
            response = self._session.get(
                f"{MCP_API_URL}/api/mcp/task/{task_id}",
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_server(self, name: str) -> MCPServer | None:
        """Get a specific server instance (local mode only).

        Args:
            name: Server name

        Returns:
            MCPServer instance or None
        """
        if self.mode != "local":
            return None
        return self._local_servers.get(name)

    def close(self) -> None:
        """Clean up resources."""
        if self.mode == "local":
            # Don't close Foundry server (managed externally)
            for name, server in self._local_servers.items():
                if name != "foundry-vtt":
                    server.close()
            self._local_servers.clear()
            self._tool_to_server.clear()
        else:
            if self._session:
                self._session.close()
                self._session = None


class MCPClientContextManager:
    """Context manager for MCPClient to ensure proper cleanup."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._client: MCPClient | None = None

    def __enter__(self) -> MCPClient:
        self._client = MCPClient(*self._args, **self._kwargs)
        return self._client

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()
        return False
