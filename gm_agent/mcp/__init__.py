"""MCP server implementations."""

from .base import MCPServer, ToolDef, ToolParameter, ToolResult
from .pf2e_rag import PF2eRAGServer
from .campaign_state import CampaignStateServer
from .creature_modifier import CreatureModifierServer
from .subsystem import SubsystemServer
from .foundry_vtt import FoundryBridge, FoundryVTTServer
from .client import MCPClient, MCPClientContextManager
from .registry import (
    SERVERS,
    TOOL_TO_SERVER,
    ServerInfo,
    get_server_for_tool,
    get_server_info,
    is_celery_eligible,
    register_foundry_tools,
    unregister_foundry_tools,
)

# CharacterRunnerServer imported separately to avoid circular imports
# (it imports from models.base which imports from mcp.base)

__all__ = [
    # Base classes
    "MCPServer",
    "ToolDef",
    "ToolParameter",
    "ToolResult",
    # Server implementations
    "PF2eRAGServer",
    "CampaignStateServer",
    "CreatureModifierServer",
    "SubsystemServer",
    "FoundryBridge",
    "FoundryVTTServer",
    # Client
    "MCPClient",
    "MCPClientContextManager",
    # Registry
    "SERVERS",
    "TOOL_TO_SERVER",
    "ServerInfo",
    "get_server_for_tool",
    "get_server_info",
    "is_celery_eligible",
    "register_foundry_tools",
    "unregister_foundry_tools",
]


def __getattr__(name: str):
    """Lazy import for CharacterRunnerServer to avoid circular imports."""
    if name == "CharacterRunnerServer":
        from .character_runner import CharacterRunnerServer

        return CharacterRunnerServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
