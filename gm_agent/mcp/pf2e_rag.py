"""PF2e RAG MCP server wrapping PathfinderSearch."""

from typing import Any

from ..config import RAG_DB_PATH
from ..rag import PathfinderSearch
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class PF2eRAGServer(MCPServer):
    """MCP server providing Pathfinder 2e RAG search tools."""

    def __init__(self, db_path: str | None = None):
        self.search = PathfinderSearch(db_path=str(db_path or RAG_DB_PATH))
        self._tools = self._build_tools()

    def _build_tools(self) -> list[ToolDef]:
        """Build the tool definitions."""
        return [
            ToolDef(
                name="lookup_creature",
                description="Look up a specific creature/monster by name. Returns detailed stats and abilities.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The creature name to look up (e.g., 'goblin', 'red dragon')",
                    ),
                ],
            ),
            ToolDef(
                name="lookup_spell",
                description="Look up a specific spell, cantrip, or focus spell by name.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The spell name to look up (e.g., 'fireball', 'shield')",
                    ),
                ],
            ),
            ToolDef(
                name="lookup_item",
                description="Look up a specific item or piece of equipment by name.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The item name to look up (e.g., 'longsword', 'healing potion')",
                    ),
                ],
            ),
            ToolDef(
                name="lookup_location",
                description="Look up a location, city, nation, or place in Golarion by name. Returns lore and setting information.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The location name to look up (e.g., 'Absalom', 'Cheliax', 'Sandpoint')",
                    ),
                ],
            ),
            ToolDef(
                name="search_rules",
                description="Search for rules, conditions, or game mechanics. Use for questions about how the game works.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="The rules query (e.g., 'flanking', 'dying condition', 'cover')",
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of results",
                        required=False,
                        default=5,
                    ),
                ],
            ),
            ToolDef(
                name="search_content",
                description="General search across all Pathfinder 2e content with optional type filters.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="The search query",
                    ),
                    ToolParameter(
                        name="types",
                        type="string",
                        description="Comma-separated content types to include (e.g., 'feat,class_feature'). Available: spell, cantrip, focus_spell, creature, npc, feat, class_feature, equipment, item, ancestry, class, archetype, condition, action, trait, page_text, location, settlement, deity, background",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of results",
                        required=False,
                        default=10,
                    ),
                ],
            ),
            ToolDef(
                name="search_lore",
                description="Search for world lore, setting information, locations, history, nations, and organizations in Golarion. Use this for questions about places, cultures, deities, or world history.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="The lore query (e.g., 'Absalom', 'Cheliax', 'Aroden', 'Inner Sea')",
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of results",
                        required=False,
                        default=5,
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        """List all available tools."""
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments."""
        try:
            if name == "lookup_creature":
                return self._lookup_creature(args["name"])
            elif name == "lookup_spell":
                return self._lookup_spell(args["name"])
            elif name == "lookup_item":
                return self._lookup_item(args["name"])
            elif name == "lookup_location":
                return self._lookup_location(args["name"])
            elif name == "search_rules":
                return self._search_rules(args["query"], args.get("limit", 5))
            elif name == "search_content":
                types = args.get("types")
                if types and isinstance(types, str):
                    types = [t.strip() for t in types.split(",")]
                return self._search_content(args["query"], types, args.get("limit", 10))
            elif name == "search_lore":
                return self._search_lore(args["query"], args.get("limit", 5))
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _lookup_creature(self, name: str) -> ToolResult:
        """Look up a creature by name."""
        results = self.search.search(
            name,
            doc_type="creature",
            top_k=1,
        )
        if not results:
            # Try broader search
            results = self.search.search(
                name,
                include_types=["creature", "npc"],
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No creature found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_spell(self, name: str) -> ToolResult:
        """Look up a spell by name."""
        results = self.search.search(
            name,
            include_types=["spell", "cantrip", "focus_spell"],
            top_k=1,
        )
        if not results:
            # Try broader search
            results = self.search.search(
                name,
                include_types=["spell", "cantrip", "focus_spell"],
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No spell found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_item(self, name: str) -> ToolResult:
        """Look up an item by name."""
        results = self.search.search(
            name,
            include_types=["equipment", "item", "weapon", "armor", "shield"],
            top_k=1,
        )
        if not results:
            results = self.search.search(
                name,
                include_types=["equipment", "item", "weapon", "armor", "shield"],
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No item found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_location(self, name: str) -> ToolResult:
        """Look up a specific location by name. Prefers structured data."""
        # Search structured location data first (exact match preferred)
        results = self.search.search(
            name,
            include_types=["location", "settlement", "region"],
            top_k=5,
        )

        if not results:
            return ToolResult(success=True, data=f"No location found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_rules(self, query: str, limit: int) -> ToolResult:
        """Search for rules content."""
        results = self.search.search(
            query,
            source_categories=["core", "supplements"],
            exclude_types=["creature", "npc", "spell", "cantrip", "focus_spell"],
            top_k=limit,
        )

        if not results:
            return ToolResult(success=True, data=f"No rules found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_content(self, query: str, types: list[str] | None, limit: int) -> ToolResult:
        """General content search."""
        kwargs = {"top_k": limit}
        if types:
            kwargs["include_types"] = types

        results = self.search.search(query, **kwargs)

        if not results:
            return ToolResult(success=True, data=f"No content found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_lore(self, query: str, limit: int) -> ToolResult:
        """Search for world lore and setting information using semantic search."""
        # Use semantic search for lore - better for narrative/conceptual queries
        # Don't demote page_text since that's where lore lives
        results = self.search.search(
            query,
            include_types=[
                "page_text",
                "location",
                "settlement",
                "deity",
                "background",
            ],
            top_k=limit,
            use_semantic=True,
            demote_page_text=False,
        )

        if not results:
            # Fall back to FTS if semantic returns nothing
            results = self.search.search(
                query,
                include_types=[
                    "page_text",
                    "location",
                    "settlement",
                    "deity",
                    "background",
                ],
                top_k=limit,
                demote_page_text=False,
            )

        if not results:
            return ToolResult(success=True, data=f"No lore found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _format_results(self, results: list[dict]) -> str:
        """Format search results for display."""
        formatted = []
        for r in results:
            header = f"**{r['name']}** ({r['type']}) - {r['source']}"
            content = r.get("content", "")
            # Truncate very long content
            if len(content) > 2000:
                content = content[:2000] + "..."
            formatted.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(formatted)

    def close(self) -> None:
        """Close the search connection."""
        self.search.close()
