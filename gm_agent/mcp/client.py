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

# Map class names to conventional registry keys used by get_server()
_CLASS_TO_SERVER_NAME: dict[str, str] = {
    "PF2eRAGServer": "pf2e-rag",
    "EncounterServer": "encounter",
    "CreatureModifierServer": "creature-modifier",
    "CampaignStateServer": "campaign-state",
    "CharacterRunnerServer": "character-runner",
    "NPCBuilderServer": "npc-builder",
    "NPCKnowledgeServer": "npc-knowledge",
    "SubsystemServer": "subsystem",
    "FictionTreeServer": "fiction-tree",
    "MicroscopeSessionServer": "microscope-session",
    "SettlementServer": "settlement",
    "DelveServer": "delve",
    "DungeonServer": "dungeon",
    "GroundingServer": "grounding",
}


def _server_registry_name(server: "MCPServer") -> str:
    """Get the conventional registry name for a server instance."""
    class_name = type(server).__name__
    return _CLASS_TO_SERVER_NAME.get(class_name, class_name)


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
        from .dice import DiceServer
        from .notes import NotesServer

        campaign_id = self.context.get("campaign_id")
        llm = self.context.get("llm")

        # Load campaign for system-aware init
        campaign = None
        campaign_books: list[str] = []
        if campaign_id:
            from ..storage.campaign import CampaignStore
            try:
                campaign = CampaignStore().get(campaign_id)
                if campaign:
                    campaign_books = campaign.books
            except Exception:
                pass

        # Always create system-agnostic servers
        self._local_servers["dice"] = DiceServer()
        self._local_servers["notes"] = NotesServer()

        # Core campaign servers (system-agnostic) — available to all game systems
        game_systems = campaign.game_systems if campaign else ["pf2e"]
        if campaign_id:
            try:
                from .campaign_state import CampaignStateServer
                from .npc_knowledge import NPCKnowledgeServer
                from ..systems import get_system as _get_sys_for_plugins

                # Collect plugins from all game systems
                system_plugins = []
                for sys_name in game_systems:
                    try:
                        system = _get_sys_for_plugins(sys_name)
                        system_plugins.extend(system.campaign_plugins(campaign_id))
                    except KeyError:
                        pass

                self._local_servers["campaign-state"] = CampaignStateServer(
                    campaign_id, system_plugins=system_plugins,
                )
                self._local_servers["npc-knowledge"] = NPCKnowledgeServer(campaign_id)
            except ImportError:
                pass

        # System-driven server initialization
        server_context = {
            "campaign_id": campaign_id,
            "campaign_books": campaign_books,
            "llm": llm,
        }

        system_servers_added = False
        try:
            from ..systems import get_system
            for sys_name in game_systems:
                try:
                    system = get_system(sys_name)
                    for server in system.servers(server_context):
                        server_key = _server_registry_name(server)
                        self._local_servers[server_key] = server
                        system_servers_added = True
                except KeyError:
                    pass
        except ImportError:
            pass

        # Fallback: if no system-driven servers were added (e.g. registry empty),
        # use the hardcoded PF2e path for backward compatibility
        if not system_servers_added:
            self._init_pf2e_servers_fallback(
                campaign_id, campaign_books, llm,
            )

        # Fiction tree server (available to all systems when campaign exists)
        if campaign_id:
            try:
                from .fiction_tree import FictionTreeServer
                self._local_servers["fiction-tree"] = FictionTreeServer(campaign_id)
            except ImportError:
                pass

        # Grounding server — available when PF2e + generation games coexist
        if campaign_id and "pf2e-rag" in self._local_servers:
            has_gen_game = any(
                k in self._local_servers
                for k in ("microscope-session", "settlement", "delve", "dungeon")
            )
            if has_gen_game:
                try:
                    from .grounding import GroundingServer
                    from ..storage.fiction_tree import FictionTreeStore
                    from ..rag import PathfinderSearch

                    fiction_store = FictionTreeStore(campaign_id)
                    search = PathfinderSearch()
                    self._local_servers["grounding"] = GroundingServer(
                        campaign_id, fiction_store, search, llm=llm,
                    )
                except (ImportError, Exception):
                    pass

        # Add Foundry server if provided
        if self._foundry_server:
            self._local_servers["foundry-vtt"] = self._foundry_server

        # Build tool routing map
        self._build_tool_map()

    def _init_pf2e_servers_fallback(
        self,
        campaign_id: str | None,
        campaign_books: list[str],
        llm: Any,
    ) -> None:
        """Fallback: hardcoded PF2e server init (backward compatibility).

        Core servers (campaign-state, npc-knowledge) are already created above,
        so this only adds PF2e-specific servers.
        """
        from .pf2e_rag import PF2eRAGServer
        from .encounter import EncounterServer
        from .creature_modifier import CreatureModifierServer
        from .character_runner import CharacterRunnerServer
        from .npc_builder import NPCBuilderServer
        from .subsystem import SubsystemServer

        self._local_servers["pf2e-rag"] = PF2eRAGServer(campaign_books=campaign_books)
        self._local_servers["encounter"] = EncounterServer()
        self._local_servers["creature-modifier"] = CreatureModifierServer()

        if campaign_id:
            # Only add core servers if not already created
            if "campaign-state" not in self._local_servers:
                from .campaign_state import CampaignStateServer
                self._local_servers["campaign-state"] = CampaignStateServer(campaign_id)
            if "npc-knowledge" not in self._local_servers:
                from .npc_knowledge import NPCKnowledgeServer
                self._local_servers["npc-knowledge"] = NPCKnowledgeServer(campaign_id)
            self._local_servers["character-runner"] = CharacterRunnerServer(campaign_id, llm=llm)
            self._local_servers["npc-builder"] = NPCBuilderServer(campaign_id, llm=llm)
            self._local_servers["subsystem"] = SubsystemServer(campaign_id)

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
        # Sanitize corrupted tool names from tokenizer artifacts
        # e.g., "search_content<|channel|>json" → "search_content"
        if "<|" in name:
            name = name.split("<|")[0]
        if "?" in name:
            name = name.split("?")[0]

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
