"""MCP server registry with tool-to-server mapping.

This module defines the available MCP servers and their metadata,
enabling dynamic server creation and tool routing.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import MCPServer


@dataclass
class ServerInfo:
    """Metadata about an MCP server."""

    # Server properties
    stateless: bool = True
    celery_eligible: bool = True

    # Context requirements
    requires_campaign: bool = False
    requires_llm: bool = False
    requires_websocket: bool = False

    # Server-specific config
    config: dict = field(default_factory=dict)


# Server registry with metadata
SERVERS: dict[str, ServerInfo] = {
    "pf2e-rag": ServerInfo(
        stateless=True,
        celery_eligible=True,
    ),
    "dice": ServerInfo(
        stateless=True,
        celery_eligible=True,
    ),
    "encounter": ServerInfo(
        stateless=True,
        celery_eligible=True,
    ),
    "notes": ServerInfo(
        stateless=False,  # Needs session state
        celery_eligible=True,
    ),
    "campaign-state": ServerInfo(
        stateless=False,
        celery_eligible=True,
        requires_campaign=True,
    ),
    "character-runner": ServerInfo(
        stateless=False,
        celery_eligible=True,
        requires_campaign=True,
        requires_llm=True,
    ),
    "npc-builder": ServerInfo(
        stateless=False,
        celery_eligible=True,
        requires_campaign=True,
        requires_llm=True,
    ),
    "npc-knowledge": ServerInfo(
        stateless=False,
        celery_eligible=True,
        requires_campaign=True,
    ),
    "rumors": ServerInfo(
        stateless=False,
        celery_eligible=True,
        requires_campaign=True,
    ),
    "creature-modifier": ServerInfo(
        stateless=True,
        celery_eligible=True,
    ),
    "subsystem": ServerInfo(
        stateless=False,
        celery_eligible=True,
        requires_campaign=True,
    ),
    "foundry-vtt": ServerInfo(
        stateless=False,
        celery_eligible=False,  # Must stay in API (WebSocket)
        requires_websocket=True,
    ),
}

# Tool name -> server name mapping (built at import time)
TOOL_TO_SERVER: dict[str, str] = {}


def _build_tool_mapping() -> None:
    """Build the tool-to-server mapping by instantiating each server."""
    # Import here to avoid circular imports
    from .pf2e_rag import PF2eRAGServer
    from .dice import DiceServer
    from .encounter import EncounterServer
    from .notes import NotesServer
    from .creature_modifier import CreatureModifierServer

    # Map tools from stateless servers (safe to instantiate)
    server_classes = {
        "pf2e-rag": PF2eRAGServer,
        "dice": DiceServer,
        "encounter": EncounterServer,
        "notes": NotesServer,
        "creature-modifier": CreatureModifierServer,
    }

    for server_name, server_class in server_classes.items():
        try:
            server = server_class()
            for tool in server.list_tools():
                TOOL_TO_SERVER[tool.name] = server_name
            server.close()
        except Exception:
            # Server may fail to initialize (e.g., missing DB)
            pass

    # Manually add campaign-state tools (requires campaign_id)
    campaign_state_tools = [
        "update_scene",
        "advance_time",
        "log_event",
        "search_history",
        "update_summary",
        "get_campaign_info",
        "add_relationship",
        "get_relationships",
        "query_relationships",
        "remove_relationship",
        "search_dialogue",
        "flag_dialogue",
        "create_faction",
        "get_faction_info",
        "list_factions",
        "add_npc_to_faction",
        "get_faction_members",
        "update_faction_reputation",
        "create_location",
        "get_location_info",
        "list_locations",
        "add_location_knowledge",
        "add_location_event",
        "set_scene_location",
        "connect_locations",
        "create_secret",
        "reveal_secret",
        "list_secrets",
        "get_revelation_history",
    ]
    for tool in campaign_state_tools:
        TOOL_TO_SERVER[tool] = "campaign-state"

    # Manually add character-runner tools (requires campaign_id + llm)
    character_runner_tools = [
        "run_npc",
        "run_monster",
        "run_player_character",
    ]
    for tool in character_runner_tools:
        TOOL_TO_SERVER[tool] = "character-runner"

    # Manually add npc-builder tools (requires campaign_id + llm)
    npc_builder_tools = [
        "build_npc_profile",
    ]
    for tool in npc_builder_tools:
        TOOL_TO_SERVER[tool] = "npc-builder"

    # Manually add npc-knowledge tools (requires campaign_id)
    npc_knowledge_tools = [
        "add_npc_knowledge",
        "query_npc_knowledge",
        "what_will_npc_share",
        "npc_learns",
        "add_party_knowledge",
        "query_party_knowledge",
        "has_party_learned",
    ]
    for tool in npc_knowledge_tools:
        TOOL_TO_SERVER[tool] = "npc-knowledge"

    # Manually add rumors tools (requires campaign_id)
    rumors_tools = [
        "seed_rumor",
        "get_rumors_at_location",
        "get_character_rumors",
        "propagate_rumors",
        "list_all_rumors",
    ]
    for tool in rumors_tools:
        TOOL_TO_SERVER[tool] = "rumors"

    # Manually add subsystem tools (requires campaign_id)
    subsystem_tools = [
        "start_subsystem",
        "subsystem_action",
        "get_subsystem_state",
        "end_subsystem",
        "list_subsystems",
    ]
    for tool in subsystem_tools:
        TOOL_TO_SERVER[tool] = "subsystem"

    # Foundry VTT tools will be added dynamically when server connects


def get_server_for_tool(tool_name: str) -> str | None:
    """Get the server name that handles a given tool.

    Args:
        tool_name: The name of the tool

    Returns:
        Server name or None if tool is unknown
    """
    return TOOL_TO_SERVER.get(tool_name)


def get_server_info(server_name: str) -> ServerInfo | None:
    """Get metadata for a server.

    Args:
        server_name: The name of the server

    Returns:
        ServerInfo or None if server is unknown
    """
    return SERVERS.get(server_name)


def is_celery_eligible(server_name: str) -> bool:
    """Check if a server can be executed via Celery.

    Args:
        server_name: The name of the server

    Returns:
        True if the server can run in a Celery worker
    """
    info = SERVERS.get(server_name)
    return info.celery_eligible if info else False


def register_foundry_tools(tools: list[str]) -> None:
    """Register Foundry VTT tools when server connects.

    Args:
        tools: List of tool names from Foundry server
    """
    for tool in tools:
        TOOL_TO_SERVER[tool] = "foundry-vtt"


def unregister_foundry_tools() -> None:
    """Unregister Foundry VTT tools when server disconnects."""
    tools_to_remove = [tool for tool, server in TOOL_TO_SERVER.items() if server == "foundry-vtt"]
    for tool in tools_to_remove:
        del TOOL_TO_SERVER[tool]


# Build mapping on import
_build_tool_mapping()
