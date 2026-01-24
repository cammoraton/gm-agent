"""NPC Knowledge MCP server for managing what NPCs know and share."""

from typing import Any

from ..config import CAMPAIGNS_DIR
from ..storage.knowledge import KnowledgeStore
from ..storage.characters import CharacterStore
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class NPCKnowledgeServer(MCPServer):
    """MCP server for NPC knowledge management.

    Provides tools for:
    - Adding knowledge to NPCs
    - Querying what NPCs know
    - Checking what NPCs will share based on conditions
    - Teaching NPCs new information
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self._knowledge_store: KnowledgeStore | None = None
        self._character_store: CharacterStore | None = None
        self._tools = self._build_tools()

    @property
    def knowledge(self) -> KnowledgeStore:
        """Lazy-load knowledge store."""
        if self._knowledge_store is None:
            self._knowledge_store = KnowledgeStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._knowledge_store

    @property
    def characters(self) -> CharacterStore:
        """Lazy-load character store."""
        if self._character_store is None:
            self._character_store = CharacterStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._character_store

    def _build_tools(self) -> list[ToolDef]:
        """Build the tool definitions."""
        return [
            ToolDef(
                name="add_npc_knowledge",
                description="Add a piece of knowledge to an NPC's memory. Use this when an NPC learns something new.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character who knows this",
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="The knowledge content (what the NPC knows)",
                    ),
                    ToolParameter(
                        name="knowledge_type",
                        type="string",
                        description="Type: 'fact', 'rumor', 'secret', 'witnessed_event', 'conversation'",
                        required=False,
                        default="fact",
                    ),
                    ToolParameter(
                        name="sharing_condition",
                        type="string",
                        description="When will they share? 'free', 'trust', 'persuasion_dc_15', 'duress', 'never'",
                        required=False,
                        default="free",
                    ),
                    ToolParameter(
                        name="source",
                        type="string",
                        description="Where did they learn this? (e.g., 'witnessed', 'told_by_Voz')",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="importance",
                        type="integer",
                        description="How important is this knowledge? (1-10)",
                        required=False,
                        default=5,
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="Comma-separated tags for categorization (e.g., 'mayor,corruption,politics')",
                        required=False,
                        default="",
                    ),
                ],
            ),
            ToolDef(
                name="query_npc_knowledge",
                description="Query what an NPC knows. Use this to check an NPC's knowledge before they speak.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character to query",
                    ),
                    ToolParameter(
                        name="knowledge_type",
                        type="string",
                        description="Filter by type (fact, rumor, secret, etc.)",
                        required=False,
                    ),
                    ToolParameter(
                        name="min_importance",
                        type="integer",
                        description="Minimum importance level (1-10)",
                        required=False,
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="Comma-separated tags to filter by",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of results",
                        required=False,
                        default=20,
                    ),
                ],
            ),
            ToolDef(
                name="what_will_npc_share",
                description="Check what an NPC will share given relationship and persuasion conditions. Use before revealing NPC knowledge.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character",
                    ),
                    ToolParameter(
                        name="trust_level",
                        type="integer",
                        description="Trust level with the requester (-5 to +5)",
                        required=False,
                        default=0,
                    ),
                    ToolParameter(
                        name="persuasion_dc_met",
                        type="integer",
                        description="Highest persuasion DC the requester has met",
                        required=False,
                        default=0,
                    ),
                    ToolParameter(
                        name="under_duress",
                        type="boolean",
                        description="Is the character under duress/threat?",
                        required=False,
                        default=False,
                    ),
                    ToolParameter(
                        name="knowledge_type",
                        type="string",
                        description="Filter by knowledge type",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results",
                        required=False,
                        default=20,
                    ),
                ],
            ),
            ToolDef(
                name="npc_learns",
                description="Teach an NPC something new. Alias for add_npc_knowledge with simplified parameters.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character learning",
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="What they learned",
                    ),
                    ToolParameter(
                        name="source",
                        type="string",
                        description="Who/what taught them (e.g., 'the party', 'Voz Lirayne')",
                        required=False,
                        default="",
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
            if name == "add_npc_knowledge":
                return self._add_npc_knowledge(args)
            elif name == "query_npc_knowledge":
                return self._query_npc_knowledge(args)
            elif name == "what_will_npc_share":
                return self._what_will_npc_share(args)
            elif name == "npc_learns":
                return self._npc_learns(args)
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _add_npc_knowledge(self, args: dict[str, Any]) -> ToolResult:
        """Add knowledge to an NPC."""
        character_name = args.get("character_name")
        content = args.get("content")

        if not character_name or not content:
            return ToolResult(
                success=False,
                error="Both character_name and content are required"
            )

        # Get the character
        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Parse tags
        tags_str = args.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        # Add knowledge
        entry = self.knowledge.add_knowledge(
            character_id=character.id,
            character_name=character.name,
            content=content,
            knowledge_type=args.get("knowledge_type", "fact"),
            sharing_condition=args.get("sharing_condition", "free"),
            source=args.get("source", ""),
            importance=args.get("importance", 5),
            tags=tags,
        )

        return ToolResult(
            success=True,
            data=f"Knowledge added to {character.name}: \"{content}\" "
                 f"(type: {entry.knowledge_type}, sharing: {entry.sharing_condition}, "
                 f"importance: {entry.importance})"
        )

    def _query_npc_knowledge(self, args: dict[str, Any]) -> ToolResult:
        """Query what an NPC knows."""
        character_name = args.get("character_name")

        if not character_name:
            return ToolResult(success=False, error="character_name is required")

        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Parse tags
        tags_str = args.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None

        # Query knowledge
        knowledge = self.knowledge.query_knowledge(
            character_id=character.id,
            knowledge_type=args.get("knowledge_type"),
            min_importance=args.get("min_importance"),
            tags=tags,
            limit=args.get("limit", 20),
        )

        if not knowledge:
            filters = []
            if args.get("knowledge_type"):
                filters.append(f"type={args['knowledge_type']}")
            if args.get("min_importance"):
                filters.append(f"importance>={args['min_importance']}")
            if tags:
                filters.append(f"tags={tags_str}")
            filter_str = ", ".join(filters) if filters else "no filters"

            return ToolResult(
                success=True,
                data=f"{character.name} has no knowledge matching {filter_str}"
            )

        # Format results
        lines = [f"{character.name} knows {len(knowledge)} things:"]
        for k in knowledge:
            tags_display = f" [{', '.join(k.tags)}]" if k.tags else ""
            sharing_marker = "🔒" if k.sharing_condition != "free" else ""
            lines.append(
                f"\n{sharing_marker}[{k.id}] {k.knowledge_type.upper()} (importance: {k.importance}){tags_display}"
            )
            lines.append(f"  \"{k.content}\"")
            if k.source:
                lines.append(f"  Source: {k.source}")
            if k.sharing_condition != "free":
                lines.append(f"  Sharing: {k.sharing_condition}")

        return ToolResult(success=True, data="\n".join(lines))

    def _what_will_npc_share(self, args: dict[str, Any]) -> ToolResult:
        """Check what an NPC will share given conditions."""
        character_name = args.get("character_name")

        if not character_name:
            return ToolResult(success=False, error="character_name is required")

        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Get shareable knowledge
        shareable = self.knowledge.get_shareable_knowledge(
            character_id=character.id,
            trust_level=args.get("trust_level", 0),
            persuasion_dc_met=args.get("persuasion_dc_met", 0),
            under_duress=args.get("under_duress", False),
            knowledge_type=args.get("knowledge_type"),
            limit=args.get("limit", 20),
        )

        if not shareable:
            return ToolResult(
                success=True,
                data=f"{character.name} will not share any knowledge under these conditions."
            )

        # Format results
        conditions = []
        if args.get("trust_level", 0) > 0:
            conditions.append(f"trust: {args['trust_level']}")
        if args.get("persuasion_dc_met", 0) > 0:
            conditions.append(f"persuasion DC: {args['persuasion_dc_met']}")
        if args.get("under_duress"):
            conditions.append("under duress")
        condition_str = " (" + ", ".join(conditions) + ")" if conditions else ""

        lines = [f"{character.name} will share {len(shareable)} things{condition_str}:"]
        for k in shareable:
            tags_display = f" [{', '.join(k.tags)}]" if k.tags else ""
            lines.append(
                f"\n[{k.id}] {k.knowledge_type.upper()}{tags_display}"
            )
            lines.append(f"  \"{k.content}\"")

        return ToolResult(success=True, data="\n".join(lines))

    def _npc_learns(self, args: dict[str, Any]) -> ToolResult:
        """Simplified version of add_npc_knowledge."""
        return self._add_npc_knowledge({
            "character_name": args.get("character_name"),
            "content": args.get("content"),
            "knowledge_type": "conversation",
            "sharing_condition": "free",
            "source": args.get("source", ""),
            "importance": 5,
        })

    def close(self) -> None:
        """Close resources."""
        if self._knowledge_store:
            self._knowledge_store.close()
            self._knowledge_store = None
