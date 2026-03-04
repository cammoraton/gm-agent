"""Ex Novo settlement-building game system."""

from __future__ import annotations

from typing import Any

from .. import GameSystem, register_system
from ...mcp.base import MCPServer


@register_system
class ExNovoSystem(GameSystem):
    """Ex Novo — collaborative settlement building."""

    name = "ex_novo"
    display_name = "Ex Novo"
    category = "generation"

    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        from .servers import SettlementServer

        campaign_id = context.get("campaign_id")
        if not campaign_id:
            return []
        return [SettlementServer(campaign_id, llm=context.get("llm"))]

    def system_prompt(self) -> str:
        from .prompts import EX_NOVO_SYSTEM_PROMPT
        return EX_NOVO_SYSTEM_PROMPT
