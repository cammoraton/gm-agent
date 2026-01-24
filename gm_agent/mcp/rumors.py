"""Rumors MCP server for rumor propagation and management."""

from typing import Any

from ..config import CAMPAIGNS_DIR
from ..rumor_mill import RumorEngine
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class RumorsServer(MCPServer):
    """MCP server for rumor propagation management.

    Provides tools for:
    - Seeding rumors
    - Propagating rumors over time
    - Querying rumors by location or character
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self._rumor_engine: RumorEngine | None = None
        self._tools = self._build_tools()

    @property
    def rumors(self) -> RumorEngine:
        """Lazy-load rumor engine."""
        if self._rumor_engine is None:
            self._rumor_engine = RumorEngine(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._rumor_engine

    def _build_tools(self) -> list[ToolDef]:
        """Build the tool definitions."""
        return [
            ToolDef(
                name="seed_rumor",
                description="Seed a new rumor that will spread through the campaign world.",
                parameters=[
                    ToolParameter(
                        name="content",
                        type="string",
                        description="The rumor content/text",
                    ),
                    ToolParameter(
                        name="starting_locations",
                        type="string",
                        description="Comma-separated location names where rumor starts (optional)",
                        required=False,
                    ),
                    ToolParameter(
                        name="spread_rate",
                        type="string",
                        description="How fast rumor spreads: slow, medium, or fast",
                        required=False,
                    ),
                    ToolParameter(
                        name="source_type",
                        type="string",
                        description="Source: pc_seeded, event, or npc_created",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="get_rumors_at_location",
                description="Get all rumors currently known at a location.",
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="Name of the location",
                    ),
                ],
            ),
            ToolDef(
                name="get_character_rumors",
                description="Get all rumors known by a character.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character",
                    ),
                ],
            ),
            ToolDef(
                name="propagate_rumors",
                description="Propagate all rumors based on time passed.",
                parameters=[
                    ToolParameter(
                        name="days",
                        type="integer",
                        description="Number of days to simulate (default: 1)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="list_all_rumors",
                description="List all rumors in the campaign.",
                parameters=[],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        """List all available tools."""
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments."""
        try:
            if name == "seed_rumor":
                return self._seed_rumor(args)
            elif name == "get_rumors_at_location":
                return self._get_rumors_at_location(args)
            elif name == "get_character_rumors":
                return self._get_character_rumors(args)
            elif name == "propagate_rumors":
                return self._propagate_rumors(args)
            elif name == "list_all_rumors":
                return self._list_all_rumors()
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _seed_rumor(self, args: dict[str, Any]) -> ToolResult:
        """Seed a new rumor."""
        content = args.get("content")
        if not content:
            return ToolResult(success=False, error="content is required")

        # Parse starting locations
        starting_locations = None
        if args.get("starting_locations"):
            location_names = [loc.strip() for loc in args["starting_locations"].split(",")]
            starting_locations = []
            for name in location_names:
                location = self.rumors.locations.get_by_name(name)
                if location:
                    starting_locations.append(location.id)

        spread_rate = args.get("spread_rate", "medium")
        source_type = args.get("source_type", "pc_seeded")

        rumor = self.rumors.seed_rumor(
            content=content,
            starting_locations=starting_locations,
            spread_rate=spread_rate,
            source_type=source_type,
        )

        location_count = len(rumor.current_locations)
        return ToolResult(
            success=True,
            data=f"Seeded rumor: \"{content}\" at {location_count} location(s). "
                 f"Spread rate: {spread_rate}, ID: {rumor.id}"
        )

    def _get_rumors_at_location(self, args: dict[str, Any]) -> ToolResult:
        """Get rumors at a location."""
        location_name = args.get("location_name")
        if not location_name:
            return ToolResult(success=False, error="location_name is required")

        location = self.rumors.locations.get_by_name(location_name)
        if not location:
            return ToolResult(
                success=False,
                error=f"Location '{location_name}' not found"
            )

        rumors = self.rumors.get_rumors_at_location(location.id)

        if not rumors:
            return ToolResult(
                success=True,
                data=f"No rumors at {location.name}"
            )

        lines = [f"Rumors at {location.name}:"]
        for rumor in rumors:
            accuracy_pct = int(rumor.accuracy * 100)
            lines.append(f"  - {rumor.content} (accuracy: {accuracy_pct}%, spread: {rumor.spread_rate})")

        return ToolResult(success=True, data="\n".join(lines))

    def _get_character_rumors(self, args: dict[str, Any]) -> ToolResult:
        """Get rumors known by a character."""
        character_name = args.get("character_name")
        if not character_name:
            return ToolResult(success=False, error="character_name is required")

        character = self.rumors.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        rumors = self.rumors.get_character_rumors(character.id)

        if not rumors:
            return ToolResult(
                success=True,
                data=f"{character.name} knows no rumors"
            )

        lines = [f"Rumors known by {character.name}:"]
        for rumor in rumors:
            accuracy_pct = int(rumor.accuracy * 100)
            lines.append(f"  - {rumor.content} (accuracy: {accuracy_pct}%)")

        return ToolResult(success=True, data="\n".join(lines))

    def _propagate_rumors(self, args: dict[str, Any]) -> ToolResult:
        """Propagate all rumors."""
        days = args.get("days", 1)

        if not isinstance(days, int) or days < 1:
            return ToolResult(success=False, error="days must be a positive integer")

        stats = self.rumors.propagate_rumors(time_delta_days=days)

        return ToolResult(
            success=True,
            data=f"Propagated rumors for {days} day(s). "
                 f"{stats['rumors_spread']} rumors spread to {stats['new_locations']} new locations."
        )

    def _list_all_rumors(self) -> ToolResult:
        """List all rumors."""
        rumors = self.rumors.list_all()

        if not rumors:
            return ToolResult(success=True, data="No rumors in campaign")

        lines = ["All rumors:"]
        for rumor in rumors:
            accuracy_pct = int(rumor.accuracy * 100)
            location_count = len(rumor.current_locations)
            lines.append(
                f"  - {rumor.content} "
                f"(accuracy: {accuracy_pct}%, locations: {location_count}, spread: {rumor.spread_rate})"
            )

        return ToolResult(success=True, data="\n".join(lines))

    def close(self) -> None:
        """Close resources."""
        pass
