"""PF2e RAG MCP server wrapping PathfinderSearch."""

from typing import Any

from ..config import RAG_DB_PATH
from ..rag import PathfinderSearch
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class PF2eRAGServer(MCPServer):
    """MCP server providing Pathfinder 2e RAG search tools."""

    def __init__(self, db_path: str | None = None, campaign_books: list[str] | None = None):
        self.search = PathfinderSearch(db_path=str(db_path or RAG_DB_PATH))
        self._campaign_books = campaign_books or []
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
                description="General search across all Pathfinder 2e content with optional type and book filters.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="The search query",
                    ),
                    ToolParameter(
                        name="types",
                        type="string",
                        description=(
                            "Comma-separated content types to include. "
                            "Places: landmark (districts, buildings, named places in setting books), "
                            "settlement (cities/towns), region (countries/areas), "
                            "location (adventure encounter rooms/areas). "
                            "Creatures: creature, creature_family, creature_template, npc. "
                            "Rules: rule, variant_rule, condition, action, game_mechanic, guidance, table. "
                            "Character: ancestry, heritage, background, class, class_feature, archetype, feat. "
                            "Magic: spell, cantrip, focus_spell. "
                            "Other: equipment, item, hazard, deity, organization, historical_event"
                        ),
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Filter results to a specific book (e.g., 'Absalom', 'Mwangi Expanse', 'Player Core'). Supports fuzzy matching.",
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
            ToolDef(
                name="lookup_hazard",
                description="Look up a specific hazard or haunt by name. Returns disabling information and stats.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The hazard name to look up (e.g., 'poison dart trap', 'poltergeist')",
                    ),
                ],
            ),
            ToolDef(
                name="lookup_encounter",
                description="Look up an encounter area by code (A1, B3) or name. Returns room description, creatures, hazards, and read-aloud text. When running an adventure path, pass the book name to scope results.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Area code (e.g., 'A1', 'B3') or location name",
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Adventure path or book name to scope results (e.g., 'Abomination Vaults')",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="lookup_npc",
                description="Look up an NPC by name. Returns combined roleplay info, stat block, and narrative context by merging NPC, creature, and page text entries.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The NPC name to look up (e.g., 'Nyrissa', 'Oleg Leveton', 'The Stag Lord')",
                    ),
                ],
            ),
            ToolDef(
                name="search_guidance",
                description="Search for GM advice, tips on running the game, and how-to-run guidance for creatures, encounters, or situations.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Topic to get guidance on (e.g., 'running goblins', 'chase scenes', 'social encounters')",
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
                name="get_read_aloud",
                description="Get read-aloud/boxed text for a location or encounter area. Use this to find descriptive text to read to players.",
                parameters=[
                    ToolParameter(
                        name="location",
                        type="string",
                        description="Location name or area code (e.g., 'A1', 'mine entrance')",
                    ),
                ],
            ),
            ToolDef(
                name="search_pages",
                description="Search full page text from books. Use this for narrative content, flavor text, or when entity search doesn't find what you need.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="The search query",
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Optional book name to restrict search to",
                        required=False,
                        default=None,
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
                name="get_db_stats",
                description="Get statistics about the Pathfinder search database: entity counts, page counts, category/book breakdowns.",
                parameters=[],
            ),
            ToolDef(
                name="find_page",
                description="Find which page(s) a term, spell, creature, or item is on. Returns book, page number, and type.",
                parameters=[
                    ToolParameter(
                        name="term",
                        type="string",
                        description="The term to find (e.g., 'Fireball', 'goblin warrior', 'longsword')",
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Optional book name to restrict search to",
                        required=False,
                        default=None,
                    ),
                ],
            ),
            ToolDef(
                name="browse_book",
                description="Hierarchical book browser. No args: list all books (optionally filtered by book_type). Book only: book summary + chapter TOC. Book + chapter: chapter summary + page-by-page drill-down. Accepts user-friendly book names (e.g., 'Player Core' instead of '[PZO12001E] Player Core').",
                parameters=[
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Optional book name (user-friendly names accepted, e.g., 'Player Core')",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="chapter",
                        type="string",
                        description="Optional chapter name (requires book). Returns chapter summary + page summaries.",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="book_type",
                        type="string",
                        description="Filter book list by type: rulebook, bestiary, adventure, setting, npc, players_guide",
                        required=False,
                        default=None,
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
                return self._search_content(
                    args["query"], types, args.get("book"), args.get("limit", 10),
                )
            elif name == "search_lore":
                return self._search_lore(args["query"], args.get("limit", 5))
            elif name == "lookup_hazard":
                return self._lookup_hazard(args["name"])
            elif name == "lookup_encounter":
                return self._lookup_encounter(args["query"], args.get("book"))
            elif name == "lookup_npc":
                return self._lookup_npc(args["name"])
            elif name == "search_guidance":
                return self._search_guidance(args["query"], args.get("limit", 5))
            elif name == "get_read_aloud":
                return self._get_read_aloud(args["location"])
            elif name == "search_pages":
                return self._search_pages(
                    args["query"], args.get("book"), args.get("limit", 5)
                )
            elif name == "get_db_stats":
                return self._get_db_stats()
            elif name == "find_page":
                return self._find_page(args["term"], args.get("book"))
            elif name == "browse_book":
                return self._browse_book(
                    args.get("book"), args.get("chapter"), args.get("book_type"),
                )
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
            results = self.search.search(
                name,
                category="creature",
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No creature found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_spell(self, name: str) -> ToolResult:
        """Look up a spell by name."""
        results = self.search.search(
            name,
            category="spell",
            top_k=1,
        )
        if not results:
            results = self.search.search(
                name,
                category="spell",
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No spell found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_item(self, name: str) -> ToolResult:
        """Look up an item by name."""
        results = self.search.search(
            name,
            category="equipment",
            top_k=1,
        )
        if not results:
            results = self.search.search(
                name,
                category="equipment",
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No item found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_location(self, name: str) -> ToolResult:
        """Look up a specific location by name."""
        results = self.search.search(
            name,
            category="location",
            top_k=5,
        )

        if not results:
            return ToolResult(success=True, data=f"No location found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_rules(self, query: str, limit: int) -> ToolResult:
        """Search for rules content."""
        results = self.search.search(
            query,
            book_type=["rulebook"],
            exclude_types=["creature", "npc", "spell", "cantrip", "focus_spell"],
            top_k=limit,
        )

        if not results:
            return ToolResult(success=True, data=f"No rules found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_content(
        self, query: str, types: list[str] | None, book: str | None, limit: int,
    ) -> ToolResult:
        """General content search."""
        kwargs = {"top_k": limit}
        if types:
            kwargs["include_types"] = types
        if book:
            kwargs["book"] = book

        results = self.search.search(query, **kwargs)

        if not results:
            return ToolResult(success=True, data=f"No content found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_lore(self, query: str, limit: int) -> ToolResult:
        """Search for world lore and setting information."""
        # Use category filter for lore-related content
        results = self.search.search(
            query,
            category=["location", "lore", "deity", "organization"],
            top_k=limit,
            use_semantic=True,
            demote_page_text=False,
        )

        if not results:
            # Fall back to FTS
            results = self.search.search(
                query,
                category=["location", "lore", "deity", "organization"],
                top_k=limit,
                demote_page_text=False,
            )

        if not results:
            # Try page search as last resort
            page_results = self.search.search_pages(query, top_k=limit)
            if page_results:
                return ToolResult(success=True, data=self._format_page_results(page_results))

        if not results:
            return ToolResult(success=True, data=f"No lore found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_hazard(self, name: str) -> ToolResult:
        """Look up a hazard or haunt by name."""
        results = self.search.search(
            name,
            category="hazard",
            top_k=1,
        )
        if not results:
            results = self.search.search(
                name,
                category="hazard",
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No hazard found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_encounter(self, query: str, book: str | None = None) -> ToolResult:
        """Look up an encounter area by code or name."""
        # Use explicit book param, fall back to first campaign book
        search_book = book or (self._campaign_books[0] if self._campaign_books else None)

        if search_book:
            # Try scoped to the specific book first
            results = self.search.search(
                query,
                category="location",
                book=search_book,
                top_k=5,
            )
            if results:
                return ToolResult(success=True, data=self._format_results(results))

        # Unscoped fallback
        results = self.search.search(
            query,
            category="location",
            top_k=3,
        )

        if not results:
            # Fallback: try page search for adventure content
            page_results = self.search.search_pages(
                query, book=search_book, top_k=3
            )
            if page_results:
                return ToolResult(success=True, data=self._format_page_results(page_results))
            return ToolResult(success=True, data=f"No encounter area found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _lookup_npc(self, name: str) -> ToolResult:
        """Look up an NPC by name, merging NPC entity, creature stats, and page text."""
        sections = []

        # 1. Search for NPC-typed entities (roleplay info, personality)
        npc_results = self.search.search(
            name, category="npc", top_k=3, preprocess=False,
        )

        # 2. Search for creature-typed entities (stat blocks)
        creature_results = self.search.search(
            name, category="creature", top_k=3, preprocess=False,
        )

        # 3. Search page text for narrative context
        page_results = self.search.search_pages(name, top_k=3)

        # Filter to best matches (name appears in result name, case-insensitive)
        name_lower = name.lower()

        def is_name_match(result_name: str) -> bool:
            rn = result_name.lower()
            return name_lower in rn or rn in name_lower

        npc_matches = [r for r in npc_results if is_name_match(r["name"])]
        creature_matches = [r for r in creature_results if is_name_match(r["name"])]

        # Build merged output
        if npc_matches:
            for r in npc_matches:
                content = r.get("content", "").strip()
                if content:
                    page_info = f", p.{r['page']}" if r.get("page") else ""
                    sections.append(
                        f"### Roleplay Info\n"
                        f"**{r['name']}** (NPC) - {r.get('book', '')}{page_info}\n\n"
                        f"{content}"
                    )

        if creature_matches:
            for r in creature_matches:
                content = r.get("content", "").strip()
                if content:
                    page_info = f", p.{r['page']}" if r.get("page") else ""
                    sections.append(
                        f"### Stat Block\n"
                        f"**{r['name']}** (creature) - {r.get('book', '')}{page_info}\n\n"
                        f"{content}"
                    )

        if page_results:
            page_sections = []
            for r in page_results:
                chapter = f" ({r['chapter']})" if r.get("chapter") else ""
                page_sections.append(
                    f"**{r['book']}** p.{r['page_number']}{chapter}\n{r.get('snippet', '')}"
                )
            if page_sections:
                sections.append(
                    "### Narrative Context\n" + "\n\n".join(page_sections)
                )

        if not sections:
            # Nothing found with name matching — fall back to showing whatever we got
            all_results = npc_results + creature_results
            if all_results:
                return ToolResult(success=True, data=self._format_results(all_results))
            if page_results:
                return ToolResult(success=True, data=self._format_page_results(page_results))
            return ToolResult(success=True, data=f"No NPC found matching '{name}'")

        return ToolResult(success=True, data="\n\n---\n\n".join(sections))

    def _search_guidance(self, query: str, limit: int) -> ToolResult:
        """Search for GM advice and guidance."""
        results = self.search.search(
            query,
            category=["guidance", "rule"],
            top_k=limit,
            use_semantic=True,
        )

        if not results:
            results = self.search.search(
                query,
                category=["guidance", "rule"],
                top_k=limit,
            )

        if not results:
            return ToolResult(success=True, data=f"No guidance found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _get_read_aloud(self, location: str) -> ToolResult:
        """Get read-aloud text for a location."""
        results = self.search.search(
            location,
            category="location",
            top_k=5,
        )

        if not results:
            return ToolResult(success=True, data=f"No read-aloud text found for '{location}'")

        # Extract read_aloud from metadata where available
        lines = []
        for r in results:
            md = r.get("metadata", {})
            read_aloud = md.get("read_aloud", "")
            name = r.get("name", "Unknown")
            book = r.get("book", "")
            if read_aloud:
                lines.append(f"## {name}\n**Source:** {book}\n\n> {read_aloud}")
            else:
                # Fall back to content snippet
                content = r.get("content", "")[:300]
                if content:
                    lines.append(f"## {name}\n**Source:** {book}\n\n{content}")

        if not lines:
            return ToolResult(
                success=True,
                data=f"Found locations matching '{location}' but none have read-aloud text.",
            )

        return ToolResult(success=True, data="\n\n---\n\n".join(lines))

    def _search_pages(self, query: str, book: str | None, limit: int) -> ToolResult:
        """Search full page text from books."""
        results = self.search.search_pages(query, top_k=limit, book=book)

        if not results:
            return ToolResult(success=True, data=f"No pages found matching '{query}'")

        return ToolResult(success=True, data=self._format_page_results(results))

    def _get_db_stats(self) -> ToolResult:
        """Get database statistics."""
        stats = self.search.get_stats()

        lines = [
            f"**Database Statistics** (schema v{stats.get('schema_version', '?')})",
            f"- Entities: {stats['total_entities']:,}",
            f"- Pages: {stats['total_pages']:,}",
            f"- Embeddings: {stats['total_embeddings']:,}",
            "",
            "**By Category:**",
        ]
        for cat, count in stats["by_category"].items():
            lines.append(f"- {cat}: {count:,}")

        lines.append("")
        lines.append("**By Book Type:**")
        for bt, count in stats["by_book_type"].items():
            lines.append(f"- {bt}: {count:,}")

        return ToolResult(success=True, data="\n".join(lines))

    def _find_page(self, term: str, book: str | None) -> ToolResult:
        """Find which page(s) a term is on."""
        results = self.search.find_page_for_term(term, book=book)

        if not results:
            return ToolResult(success=True, data=f"No page found for '{term}'")

        lines = []
        for r in results[:15]:
            chapter = f" ({r['chapter']})" if r.get("chapter") else ""
            type_info = f" ({r['type']})" if r.get("type") and r["type"] != "page_reference" else ""
            lines.append(
                f"**{r.get('name', term)}**{type_info} - {r['book']}, p.{r['page_number']}{chapter}"
            )

        return ToolResult(success=True, data="\n".join(lines))

    def _browse_book(
        self,
        book: str | None,
        chapter: str | None,
        book_type: str | None = None,
    ) -> ToolResult:
        """Hierarchical book browser with book name resolution.

        No args:           List books, optionally filtered by book_type.
        Book only:         Book summary + chapter TOC.
        Book + chapter:    Chapter summary + page-by-page drill-down.
        """
        if not book:
            # List all books with summaries, optionally filtered
            books = self.search.list_books_with_summaries(book_type=book_type)
            if not books:
                if book_type:
                    return ToolResult(
                        success=True,
                        data=f"No books found with type '{book_type}'.",
                    )
                return ToolResult(
                    success=True,
                    data="No book summaries available. The database may use an older schema.",
                )

            type_label = f" ({book_type})" if book_type else ""
            lines = [f"**Available Books{type_label}** ({len(books)} with summaries)", ""]
            for b in books:
                lines.append(
                    f"- **{b['book']}** ({b['total_pages']} pages, {b['chapter_count']} chapters)"
                )

            return ToolResult(success=True, data="\n".join(lines))

        # Resolve user-friendly book name
        resolved = self.search.resolve_book_name(book)
        if resolved:
            book = resolved

        if not chapter:
            # Book overview + chapter TOC
            chapters = self.search.list_chapters(book)
            book_sum = self.search.get_book_summary(book)

            if not chapters and not book_sum:
                return ToolResult(
                    success=True,
                    data=f"No summary data available for '{book}'. Try browse_book with no args to see available books.",
                )

            lines = [f"**{book}**"]

            if book_sum:
                lines.append(
                    f"{book_sum.get('total_pages', '?')} pages, "
                    f"{book_sum.get('chapter_count', '?')} chapters"
                )
                lines.append("")
                summary_text = book_sum.get("summary", "")
                if summary_text:
                    if len(summary_text) > 800:
                        summary_text = summary_text[:800] + "..."
                    lines.append(summary_text)
                lines.append("")

            if chapters:
                lines.append("**Table of Contents:**")
                for ch in chapters:
                    lines.append(
                        f"- **{ch['chapter']}** (pp. {ch['page_start']}-{ch['page_end']}, {ch['page_count']} pages)"
                    )

            return ToolResult(success=True, data="\n".join(lines))

        # Book + chapter: chapter summary + page-by-page summaries
        lines = []

        # Get chapter-level summary
        ch = self.search.get_chapter_summary(book, chapter)
        if ch:
            lines.append(f"**{ch['chapter']}** - {book}")
            lines.append(
                f"Pages {ch['page_start']}-{ch['page_end']} ({ch['page_count']} pages)"
            )
            lines.append("")
            if ch.get("summary"):
                lines.append(ch["summary"])
            if ch.get("keywords"):
                lines.append(f"\n**Keywords:** {', '.join(ch['keywords'][:20])}")
            if ch.get("entities"):
                lines.append(f"**Key entities:** {', '.join(ch['entities'][:20])}")
            lines.append("")

        # Get page-by-page summaries for that chapter
        page_summaries = self.search.get_page_summaries_for_chapter(book, chapter)

        if not ch and not page_summaries:
            return ToolResult(
                success=True,
                data=f"No chapter matching '{chapter}' found in {book}",
            )

        if page_summaries:
            lines.append(f"**Page Summaries** ({len(page_summaries)} pages)")
            lines.append("")
            for ps in page_summaries:
                page_type = f" [{ps.get('page_type', '')}]" if ps.get("page_type") else ""
                lines.append(f"**p.{ps['page_number']}**{page_type}: {ps.get('summary', '')}")
                if ps.get("entities_on_page"):
                    entities = ps["entities_on_page"]
                    if isinstance(entities, list) and entities:
                        lines.append(f"  Entities: {', '.join(entities[:10])}")
                lines.append("")

        return ToolResult(success=True, data="\n".join(lines))

    def _format_results(self, results: list[dict]) -> str:
        """Format search results for display."""
        formatted = []
        for r in results:
            page_info = f", p.{r['page']}" if r.get("page") else ""
            header = f"**{r['name']}** ({r['type']}) - {r.get('book', r.get('source', ''))}{page_info}"
            content = r.get("content", "")
            if len(content) > 2000:
                content = content[:2000] + "..."
            formatted.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(formatted)

    def _format_page_results(self, results: list[dict]) -> str:
        """Format page search results for display."""
        formatted = []
        for r in results:
            chapter = f" ({r['chapter']})" if r.get("chapter") else ""
            header = f"**{r['book']}** p.{r['page_number']}{chapter}"
            snippet = r.get("snippet", "")
            formatted.append(f"{header}\n{snippet}")

        return "\n\n---\n\n".join(formatted)

    def close(self) -> None:
        """Close the search connection."""
        self.search.close()
