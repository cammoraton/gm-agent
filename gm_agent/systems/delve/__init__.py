"""Delve hold-building game system."""

from __future__ import annotations

from typing import Any

from .. import GameSystem, register_system
from ...mcp.base import MCPServer


@register_system
class DelveSystem(GameSystem):
    """Delve — solo/cooperative underground hold building."""

    name = "delve"
    display_name = "Delve"
    category = "generation"

    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        from .servers import DelveServer

        campaign_id = context.get("campaign_id")
        if not campaign_id:
            return []
        return [DelveServer(campaign_id, llm=context.get("llm"))]

    def system_prompt(self) -> str:
        from .prompts import DELVE_SYSTEM_PROMPT
        return DELVE_SYSTEM_PROMPT
