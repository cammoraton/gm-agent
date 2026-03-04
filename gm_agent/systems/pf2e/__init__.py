"""Pathfinder 2e (Remaster) game system.

Self-contained system with servers, storage, and prompts under
gm_agent/systems/pf2e/. Re-export stubs at the old paths ensure
backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import GameSystem, register_system

from .prompts import GM_SYSTEM_PROMPT

if TYPE_CHECKING:
    from ...mcp.base import MCPServer


@register_system
class PF2eSystem(GameSystem):
    """Pathfinder 2e (Remaster) game system."""

    name = "pf2e"
    display_name = "Pathfinder 2e (Remaster)"
    category = "rpg"

    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        """Create PF2e-specific MCP servers based on context.

        Core campaign servers (CampaignStateServer, NPCKnowledgeServer) are
        now created by MCPClient directly so they're available to all systems.
        """
        from .servers.pf2e_rag import PF2eRAGServer
        from .servers.encounter import EncounterServer
        from .servers.creature_modifier import CreatureModifierServer
        from .servers.subsystem import SubsystemServer
        from ...mcp.character_runner import CharacterRunnerServer
        from ...mcp.npc_builder import NPCBuilderServer

        campaign_id = context.get("campaign_id")
        campaign_books = context.get("campaign_books", [])
        llm = context.get("llm")

        servers: list[MCPServer] = [
            PF2eRAGServer(campaign_books=campaign_books),
            EncounterServer(),
            CreatureModifierServer(backend=llm),
        ]

        if campaign_id:
            servers.extend([
                CharacterRunnerServer(campaign_id, llm=llm),
                NPCBuilderServer(campaign_id, llm=llm),
                SubsystemServer(campaign_id),
            ])

        return servers

    def campaign_plugins(self, campaign_id: str, **kwargs: Any) -> list:
        """Return PF2e campaign tool plugins for CampaignStateServer."""
        from .campaign_tools import PF2eCampaignToolPlugin
        return [PF2eCampaignToolPlugin(campaign_id)]

    def system_prompt(self) -> str:
        """Return the PF2e GM system prompt."""
        return GM_SYSTEM_PROMPT

    def stores(self, campaign_id: str) -> dict[str, Any]:
        """Return PF2e-specific stores."""
        from .storage.ap_progress import APProgressStore
        from .storage.treasure import TreasureStore

        return {
            "ap_progress": APProgressStore(campaign_id),
            "treasure": TreasureStore(campaign_id),
        }
