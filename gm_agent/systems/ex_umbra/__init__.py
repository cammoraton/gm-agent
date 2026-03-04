"""Ex Umbra dungeon-building game system."""

from __future__ import annotations

from typing import Any

from .. import GameSystem, register_system
from ...mcp.base import MCPServer


@register_system
class ExUmbraSystem(GameSystem):
    """Ex Umbra — collaborative dungeon building from the shadows up."""

    name = "ex_umbra"
    display_name = "Ex Umbra"
    category = "generation"

    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        from .servers import DungeonServer

        campaign_id = context.get("campaign_id")
        if not campaign_id:
            return []
        return [DungeonServer(campaign_id, llm=context.get("llm"))]

    def system_prompt(self) -> str:
        from .prompts import EX_UMBRA_SYSTEM_PROMPT
        return EX_UMBRA_SYSTEM_PROMPT
