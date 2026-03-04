"""Microscope worldbuilding game system."""

from __future__ import annotations

from typing import Any

from .. import GameSystem, register_system
from ...mcp.base import MCPServer


@register_system
class MicroscopeSystem(GameSystem):
    """Microscope — collaborative fractal worldbuilding."""

    name = "microscope"
    display_name = "Microscope"
    category = "generation"

    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        from .servers import MicroscopeSessionServer

        campaign_id = context.get("campaign_id")
        if not campaign_id:
            return []
        return [MicroscopeSessionServer(campaign_id, llm=context.get("llm"))]

    def system_prompt(self) -> str:
        from .prompts import MICROSCOPE_SYSTEM_PROMPT
        return MICROSCOPE_SYSTEM_PROMPT
