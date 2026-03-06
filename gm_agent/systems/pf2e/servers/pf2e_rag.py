"""PF2e RAG MCP server wrapping PathfinderSearch."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

from gm_agent.config import RAG_DB_PATH
from gm_agent.rag import PathfinderSearch
from gm_agent.mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult


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
                name="lookup",
                description=(
                    "Look up a named entity and return its full description. "
                    "type must be one of: creature, spell, item, location, hazard, npc, encounter, class. "
                    "npc vs creature: named characters (villains, allies, quest-givers) → npc. "
                    "Generic monsters (goblin warrior, red dragon) → creature. "
                    "The boundary is fuzzy — npc merges creature stat blocks automatically. "
                    "type=location returns encounter room descriptions and read-aloud/boxed text — "
                    "ALWAYS specify book= for dungeon locations to avoid cross-AP matches. "
                    "type=class returns a class entry (key ability, roles, overview). "
                    "For specific class mechanics use search_rules(query='classname ability'). "
                    "book= scopes npc, location, and encounter searches to a specific AP or sourcebook."
                ),
                parameters=[
                    ToolParameter(
                        name="type",
                        type="string",
                        description="Entity type: creature, spell, item, location, hazard, npc, encounter, class",
                    ),
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The entity name or search term (e.g., 'goblin', 'fireball', 'Nyrissa', 'A1')",
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Optional: scope to a specific book or AP (e.g., 'Kingmaker', 'Abomination Vaults')",
                        required=False,
                        default=None,
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
                            "Comma-separated content types to filter. "
                            "Valid types — Places: landmark, settlement, region, location | "
                            "Creatures: creature, creature_family, npc | "
                            "Rules: rule, variant_rule, condition, action, game_mechanic, guidance, subsystem, table | "
                            "Character: ancestry, heritage, background, class, class_feature, archetype, feat | "
                            "Magic: spell, cantrip, focus_spell | "
                            "Other: equipment, item, hazard, deity, organization, historical_event. "
                            "Do NOT use 'adventure' or 'book' — these are not valid content types"
                        ),
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Filter to a specific book or adventure path. Use for AP-scoped questions (e.g., 'Season of Ghosts', 'Kingmaker') and location/setting queries. Do NOT use for creatures, spells, or rules — they span many books and searching unscoped gives better results. Supports fuzzy matching.",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="chapter",
                        type="string",
                        description="Filter to a specific chapter within a book (requires book). Use for chapter-scoped queries like 'Kingmaker Chapter 2'. Resolves chapter name to page range.",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="level",
                        type="integer",
                        description="Filter to entities at this level (creatures, hazards). Use with creature/hazard type filters.",
                        required=False,
                    ),
                    ToolParameter(
                        name="level_range",
                        type="string",
                        description="Filter to level range, e.g. '3-7'. Use with creature/hazard type filters.",
                        required=False,
                    ),
                    ToolParameter(
                        name="traits",
                        type="string",
                        description="Comma-separated traits to filter by (e.g. 'undead,humanoid'). All traits must match.",
                        required=False,
                    ),
                    ToolParameter(
                        name="remaster_only",
                        type="boolean",
                        description="If true, only return results from Remaster-era books (Player Core, Monster Core, etc.). Use to exclude legacy content (old Bestiary, Core Rulebook).",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="scope",
                        type="string",
                        description=(
                            "Shortcut scope filter. 'lore': world lore, locations, deities, regions, "
                            "organizations, history (use for setting/Golarion questions). "
                            "'guidance': GM advice and tips on running the game."
                        ),
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
                        name="chapter",
                        type="string",
                        description="Optional chapter name to restrict search to (requires book).",
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
                name="get_player_advice",
                description=(
                    "Look up rules and guidance to advise a player on their situation. "
                    "Returns relevant rules, conditions, and guidance sections organized by query type. "
                    "Use this to quickly surface rules that help a player understand their options."
                ),
                parameters=[
                    ToolParameter(
                        name="situation",
                        type="string",
                        description="What the player is facing or asking about (e.g., 'fighting a flying creature', 'persuading the duke')",
                    ),
                    ToolParameter(
                        name="query_type",
                        type="string",
                        description=(
                            "Type of advice needed: "
                            "'combat' (flanking, positioning, action economy), "
                            "'exploration' (activities, travel, searching), "
                            "'social' (diplomacy, attitudes, deception), "
                            "'mechanical' (rules clarifications, conditions), "
                            "'character_build' (feats, class features, ancestry)"
                        ),
                    ),
                    ToolParameter(
                        name="character_class",
                        type="string",
                        description="Character's class for class-specific advice (e.g., 'ranger', 'wizard')",
                        required=False,
                        default=None,
                    ),
                    ToolParameter(
                        name="level",
                        type="integer",
                        description="Character level for level-appropriate content",
                        required=False,
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
            ToolDef(
                name="list_entities",
                description=(
                    "Browse and list entities of a type (up to the limit, default 50) — use this instead of search_content "
                    "when the user wants a list, table, or inventory. "
                    "Examples: all deities, all dragons, landmarks in Absalom, creature families. "
                    "Returns a compact table with key metadata (level, weapon, domains, etc.), "
                    "sorted alphabetically. Supports name_filter for substring matching "
                    "(e.g., name_filter='dragon' to list all dragons)."
                ),
                parameters=[
                    ToolParameter(
                        name="type",
                        type="string",
                        description=(
                            "Entity type to list (e.g., 'creature', 'deity', 'landmark', "
                            "'settlement', 'creature_family', 'spell', 'feat', 'region')"
                        ),
                    ),
                    ToolParameter(
                        name="book",
                        type="string",
                        description="Optional book name to scope results (e.g., 'Absalom', 'Monster Core')",
                        required=False,
                    ),
                    ToolParameter(
                        name="name_filter",
                        type="string",
                        description="Optional substring to filter names (e.g., 'dragon' to list all dragons)",
                        required=False,
                    ),
                    ToolParameter(
                        name="trait",
                        type="string",
                        description=(
                            "Filter creatures by trait (case-insensitive). Common traits: "
                            "dragon, undead, humanoid, beast, construct, fiend, fey, elemental, "
                            "celestial, aberration, plant, animal, swarm, incorporeal, fire, cold, etc."
                        ),
                        required=False,
                    ),
                    ToolParameter(
                        name="min_level",
                        type="integer",
                        description="Minimum level (inclusive). Filters by metadata level field.",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_level",
                        type="integer",
                        description="Maximum level (inclusive). Filters by metadata level field.",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results (default 50)",
                        required=False,
                        default=50,
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
            # Strip injected context prefixes from search queries
            if "query" in args and isinstance(args["query"], str):
                args = {**args, "query": self._clean_query(args["query"])}

            # Normalize book name: strip non-breaking spaces and truncation ellipsis
            if "book" in args and isinstance(args["book"], str):
                _book_val = args["book"].replace("\xa0", " ").replace("…", "").strip()
                args = {**args, "book": _book_val}
            if "name" in args and isinstance(args["name"], str):
                _name_val = args["name"].replace("\xa0", " ").replace("…", "").replace("...", "").strip()
                args = {**args, "name": _name_val}

            if name == "lookup":
                _type = (args.get("type") or "").lower().strip()
                _name = args.get("name") or args.get("query", "")
                _book = args.get("book")
                if _type == "creature":
                    return self._lookup_creature(_name)
                elif _type == "spell":
                    return self._lookup_spell(_name)
                elif _type == "item":
                    return self._lookup_item(_name)
                elif _type == "location":
                    return self._lookup_location(_name, book=_book)
                elif _type == "hazard":
                    return self._lookup_hazard(_name)
                elif _type == "npc":
                    return self._lookup_npc(_name, book=_book)
                elif _type == "encounter":
                    return self._lookup_encounter(_name, _book)
                elif _type == "class":
                    return self._lookup_class(_name)
                else:
                    # creature_family is a valid list_entities type but not a lookup type —
                    # redirect rather than returning an error or doing a generic search.
                    if _type == "creature_family":
                        return self._list_entities("creature_family", None, _name or None, 20)
                    # Unknown type — fall back to general content search rather than
                    # returning an error that wastes a turn and derails the agent.
                    logger.debug(
                        "lookup: unknown type %r — falling back to search_content(%r)", _type, _name
                    )
                    return self._search_content(
                        query=_name, types=None, book=_book, limit=5,
                    )
            elif name == "search_rules":
                return self._search_rules(args["query"], args.get("limit", 5))
            elif name == "search_content":
                # Missing query + book/chapter → redirect to browse_book rather than erroring
                if not args.get("query") and not args.get("scope"):
                    _bq = args.get("book")
                    _cq = args.get("chapter")
                    if _bq or _cq:
                        return self._browse_book(_bq, _cq, args.get("book_type"))
                    return ToolResult(
                        success=False,
                        error=(
                            "search_content requires a 'query' parameter. "
                            "To browse a book's structure, use browse_book(book=..., chapter=...) instead."
                        ),
                    )
                # scope shortcut: 'lore' or 'guidance' redirect to specialized search
                # BUT only when types is not also specified — types is more specific and wins.
                _scope = (args.get("scope") or "").lower().strip()
                _scope_types_arg = args.get("types")
                if _scope and _scope_types_arg:
                    # scope + types are contradictory; types is explicit so ignore scope
                    _scope = ""
                if _scope == "lore":
                    return self._search_lore(
                        args["query"], args.get("limit", 5),
                        book=args.get("book"), chapter=args.get("chapter"),
                    )
                elif _scope == "guidance":
                    return self._search_guidance(
                        args["query"], args.get("limit", 5),
                        book=args.get("book"), chapter=args.get("chapter"),
                    )

                types = args.get("types")
                # Redirect search_content(types="book") → browse_book.
                # The model uses this when it wants a book/sourcebook listing.
                if (
                    isinstance(types, str)
                    and types.strip().lower() in ("book", "sourcebook", "source_book", "module")
                ):
                    _book_query = args.get("book") or args.get("query", "")
                    return self._browse_book(_book_query, args.get("chapter"), None)
                if types and isinstance(types, str):
                    types = [t.strip() for t in types.split(",")]
                # Strip out truly invalid types that have no alias expansion.
                _INVALID_TYPES = {"content", "chapter", "article"}
                _invalid_used = []
                if types:
                    _invalid_used = [t for t in types if t.lower() in _INVALID_TYPES]
                    types = [t for t in types if t.lower() not in _INVALID_TYPES] or None
                traits = args.get("traits")
                if traits and isinstance(traits, str):
                    traits = [t.strip() for t in traits.split(",")]
                level_range = None
                level_range_str = args.get("level_range")
                if level_range_str and isinstance(level_range_str, str) and "-" in level_range_str:
                    parts = level_range_str.split("-", 1)
                    try:
                        level_range = (int(parts[0].strip()), int(parts[1].strip()))
                    except (ValueError, IndexError):
                        pass
                remaster_only = args.get("remaster_only")

                # Interceptor: search_content(types="npc") is often the wrong call.
                # - Looks like an NPC name → try lookup_npc first (proper names ≤6 words)
                # - Explicit inventory query + book scope → list_entities is more reliable
                #   (concept searches like "bandits" should fall through to _search_content
                #    so FTS returns actual NPC content, not a bare name list)
                _npc_only = types and set(types) <= {"npc", "npc_group"}
                if _npc_only:
                    _query = args["query"]
                    _book = args.get("book")
                    _words = _query.split()
                    _has_proper_noun = any(
                        w[0].isupper() for w in _words if w and len(w) > 1
                    )
                    if _has_proper_noun and len(_words) <= 6:
                        # Looks like an NPC name — try direct lookup first
                        _hit = self._lookup_npc(_query, book=_book)
                        if (
                            _hit.success
                            and _hit.data
                            and not _hit.data.startswith(("No NPC found", "No NPC entity found"))
                        ):
                            return _hit
                        # lookup_npc failed (likely a place/concept name, not a person).
                        # If book is scoped, list all NPCs in that book — that's the
                        # most useful thing we can return (e.g., "Willowshore" → SoG NPC list).
                        if _book:
                            return self._list_entities(
                                "npc", _book, None, args.get("limit", 50)
                            )
                        # No book scope — fall through to _search_content FTS
                    elif _book:
                        # Only redirect to list_entities for explicit inventory queries
                        # ("npcs", "characters", "who are", "list all") — NOT concept
                        # searches ("bandits", "merchants") which need FTS content results.
                        _INVENTORY_SIGNALS = {
                            "npc", "npcs", "character", "characters",
                            "list", "all", "who", "what",
                        }
                        if any(w.lower() in _INVENTORY_SIGNALS for w in _words):
                            return self._list_entities(
                                "npc", _book, None, args.get("limit", 50)
                            )

                result = self._search_content(
                    args["query"], types, args.get("book"), args.get("limit", 10),
                    level=args.get("level"), level_range=level_range, traits=traits,
                    remaster_only=remaster_only, chapter=args.get("chapter"),
                )

                # Relationship/faction supplement: when the query signals social
                # dynamics and the caller didn't already request relationship/faction
                # types, automatically append any matching entries. This removes
                # dependence on the model remembering to use these types explicitly.
                _rel_signals = {
                    "relationship", "relation", "faction", "ally", "allies",
                    "rival", "enemy", "enemies", "conflict", "alliance",
                    "social", "politics", "power", "tension", "connected",
                    "dynamic", "map", "ecosystem", "who controls",
                }
                _requested_types = set(types or [])
                _already_has_rel = bool(
                    _requested_types & {"relationship", "faction"}
                )
                _query_words = set((args.get("query") or "").lower().split())
                if (
                    not _already_has_rel
                    and _query_words & _rel_signals
                    and args.get("book")
                ):
                    _book_for_rel = args["book"]
                    _rel_result = self._search_content(
                        args["query"], ["relationship"], _book_for_rel, 5,
                    )
                    _fac_result = self._search_content(
                        args["query"], ["faction"], _book_for_rel, 5,
                    )
                    _supplements = []
                    if _rel_result.success and _rel_result.data and "No results" not in _rel_result.data:
                        _supplements.append("**NPC Relationships:**\n" + _rel_result.data)
                    if _fac_result.success and _fac_result.data and "No results" not in _fac_result.data:
                        _supplements.append("**Factions:**\n" + _fac_result.data)
                    if _supplements and result.success:
                        result = ToolResult(
                            success=True,
                            data=(result.data or "") + "\n\n" + "\n\n".join(_supplements),
                        )

                # Warn about invalid types that were silently dropped.
                if _invalid_used and result.success and result.data:
                    bad = ", ".join(f"'{t}'" for t in _invalid_used)
                    result = ToolResult(
                        success=True,
                        data=(
                            f"[Note: type(s) {bad} are not valid content types and were ignored. "
                            f"Use browse_book() for sourcebook/chapter listings.]\n\n"
                            + result.data
                        ),
                    )
                return result
            elif name == "search_book":
                # Hallucinated tool — redirect to browse_book.
                _book_query = args.get("book") or args.get("query", "")
                return self._browse_book(_book_query, args.get("chapter"), None)
            elif name == "search_npc":
                # Common hallucinated tool — redirect to NPC search
                query = args.get("name") or args.get("query", "")
                return self._search_content(
                    query, ["npc"], args.get("book"), args.get("limit", 10),
                )
            elif name == "lookup_class":
                # Hallucinated tool — redirect to class search
                query = args.get("name") or args.get("query", "")
                return self._search_content(
                    query, ["class", "class_feature"], None, args.get("limit", 5),
                )
            elif name == "lookup_rule":
                # Hallucinated tool — redirect to rules search
                query = args.get("name") or args.get("query", "")
                return self._search_rules(query, args.get("limit", 5))
            elif name == "get_read_aloud":
                # Redirect to lookup(type="location") with book scoping
                return self._lookup_location(args.get("location", ""), book=args.get("book"))
            elif name == "search_pages":
                return self._search_pages(
                    args["query"], args.get("book"), args.get("limit", 5),
                    chapter=args.get("chapter"),
                )
            elif name == "get_db_stats":
                return self._get_db_stats()
            elif name == "find_page":
                return self._find_page(args["term"], args.get("book"))
            elif name == "browse_book":
                return self._browse_book(
                    args.get("book"), args.get("chapter"), args.get("book_type"),
                )
            elif name == "get_player_advice":
                return self._get_player_advice(
                    args["situation"],
                    args["query_type"],
                    args.get("character_class"),
                    args.get("level"),
                )
            elif name == "list_entities":
                return self._list_entities(
                    args["type"],
                    args.get("book"),
                    args.get("name_filter"),
                    args.get("limit", 50),
                    trait=args.get("trait"),
                    min_level=args.get("min_level"),
                    max_level=args.get("max_level"),
                )
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except KeyError as e:
            return ToolResult(
                success=False,
                error=f"Missing required parameter {e} for tool '{name}'. "
                      f"Please retry with all required arguments.",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _lookup_creature(self, name: str) -> ToolResult:
        """Look up a creature by name, including creature family lore."""
        # Search for the creature itself
        results = self.search.search(
            name,
            doc_type="creature",
            top_k=3,
        )
        # Also search creature_family — family entries contain ecology,
        # tactics, lore, and taxonomy that individual creature entries lack
        family_results = self.search.search(
            name,
            doc_type="creature_family",
            top_k=3,
        )
        if family_results:
            # Only append families whose name actually matches a query word
            # (prevents e.g. "stag beetle" family appearing for "Stag Lord")
            # Stat blocks come first — they have combat mechanics; family entries are supplementary lore
            query_words = set(name.lower().split())
            relevant_families = [
                f for f in family_results
                if any(w in f["name"].lower() for w in query_words)
            ]
            if relevant_families:
                results = results + relevant_families

        if not results:
            # Broader fallback across the whole creature category
            results = self.search.search(
                name,
                category="creature",
                top_k=3,
            )

        if not results:
            return ToolResult(success=True, data=f"No creature found matching '{name}'")

        return ToolResult(success=True, data=self._format_results(results))

    # Canonical rulebooks that contain class chapters (checked in order)
    _CLASS_BOOKS = (
        "Player Core 2",
        "Player Core",
        "Secrets of Magic",
        "Dark Archive",
        "Guns & Gears",
        "Rage of Elements",
        "War of Immortals",
        "Battlecry!",
    )

    def _lookup_class(self, name: str) -> ToolResult:
        """Look up a class by name, supplementing thin entity entries with page text."""
        results = self.search.search(name, doc_type="class", top_k=5)
        if not results:
            results = self.search.search(name, category="class", top_k=5)

        parts: list[str] = []
        if results:
            parts.append(self._format_results(results))

        # Always supplement with page text — class entity entries are often thin
        # (e.g. Thaumaturge only has the Shield Implement section in Battlecry!)
        page_results = self.search.search_pages(name, top_k=3)
        # Keep only pages from canonical class books
        page_results = [
            p for p in page_results
            if any(b in p.get("book", "") for b in self._CLASS_BOOKS)
        ]
        if page_results:
            parts.append(self._format_page_results(page_results))

        if not parts:
            return ToolResult(
                success=True,
                data=f"No class entry found for '{name}'. Try search_rules for specific mechanics.",
            )
        return ToolResult(success=True, data="\n\n---\n\n".join(parts))

    # Known PF2e remaster spell renames (legacy → remaster).
    # Used as fallback when a spell can't be found under its pre-remaster name.
    _SPELL_RENAMES: dict[str, str] = {
        "magic missile": "force barrage",
        "charm person": "befriend",
        "charm": "befriend",
        "sleep": "slumber",
        "entangle": "entangling flora",
        "color spray": "dizzying colors",
        "glitterdust": "revealing light",
        "hideous laughter": "hideous laughter",  # unchanged
    }

    @staticmethod
    def _clean_query(query: str) -> str:
        """Strip injected GM context prefix from search queries.

        Models sometimes pass the full context-prefixed question as the query:
            (The GM is running Season of Ghosts.)\n\nActual question text
        Strip everything before the last \\n\\n if the prefix starts with '('.
        """
        if "\n\n" in query:
            parts = [p.strip() for p in query.split("\n\n") if p.strip()]
            if len(parts) >= 2 and parts[0].startswith("("):
                return parts[-1]
        return query

    @staticmethod
    def _spell_name_matches(query: str, result_name: str) -> bool:
        """Return True if result_name is a reasonable match for the spell query."""
        def _norm(s: str) -> str:
            return " ".join(
                "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()
            )
        qn = _norm(query)
        rn = _norm(result_name)
        return qn in rn or rn in qn

    def _lookup_spell(self, name: str) -> ToolResult:
        """Look up a spell by name.

        Filters FTS results to only name-matching entries to avoid returning
        the wrong spell (e.g. Detect Magic for 'Magic Missile'). Falls back
        to known remaster renames when the original name isn't indexed.
        """
        def _search_name_match(query: str) -> list:
            results = self.search.search(query, category="spell", top_k=5)
            return [r for r in results if self._spell_name_matches(query, r.get("name", ""))]

        results = _search_name_match(name)

        if not results:
            # Try known remaster rename
            renamed = self._SPELL_RENAMES.get(name.lower())
            if renamed:
                results = _search_name_match(renamed)
                if results:
                    note = f"('{name}' was renamed to '{renamed.title()}' in the PF2e remaster)\n\n"
                    return ToolResult(success=True, data=note + self._format_results(results))

        if not results:
            return ToolResult(
                success=True,
                data=(
                    f"No spell found matching '{name}'. "
                    "Note: many spells were renamed in the PF2e remaster — "
                    "try the remaster name if known (e.g. 'Force Barrage' instead of 'Magic Missile')."
                ),
            )

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

    def _lookup_location(self, name: str, book: str | None = None) -> ToolResult:
        """Look up a specific location by name, optionally scoped to a book."""
        results = self.search.search(
            name,
            category="location",
            book=book,
            top_k=5,
        )

        if not results and book:
            # Fall back to unscoped search if book-scoped returns nothing
            results = self.search.search(name, category="location", top_k=5)

        if not results:
            # Last resort: word-split name_filter to surface locations whose name
            # contains a query word — catches "Gauntlight entrance" → "Damp Entrance"
            # when the agent knows the concept but not the room's proper name.
            if book:
                candidates = []
                seen_ids: set = set()
                for word in name.lower().split():
                    if len(word) >= 4:
                        matches = self.search.list_entities(
                            include_types=["location"],
                            book=book,
                            name_filter=word,
                            limit=10,
                        )
                        for m in matches:
                            if m.get("id") not in seen_ids:
                                seen_ids.add(m.get("id"))
                                candidates.append(m)
                if candidates:
                    # Show matches prominently — agent can pick the right one and retry
                    hint_names = ", ".join(
                        f'"{c["name"]}" (p.{c["page"]})' if c.get("page") else f'"{c["name"]}"'
                        for c in candidates[:10]
                    )
                    return ToolResult(
                        success=True,
                        data=(
                            f"No location named '{name}' found. "
                            f"Locations in this book matching parts of that name: {hint_names}. "
                            f"Retry lookup with the exact name, e.g. lookup(type='location', name='{candidates[0]['name']}')."
                        ),
                    )
            return ToolResult(success=True, data=f"No location found matching '{name}'")

        # Always supplement with a name-filter pass to surface locations whose name
        # contains a query word but whose content text doesn't contain other query words.
        # Example: "Gauntlight Entrance" → FTS finds "Security Checkpoint" (has both words
        # in content) but misses "Damp Entrance" (only "Entrance" in name, not "Gauntlight").
        # Name-filter catches it regardless of the FTS mix.
        if book:
            existing_ids = {r.get("id") for r in results}
            name_extras = []
            for word in name.lower().split():
                if len(word) >= 4:
                    extras = self.search.list_entities(
                        include_types=["location"],
                        book=book,
                        name_filter=word,
                        limit=10,
                    )
                    for e in extras:
                        if e.get("id") not in existing_ids:
                            existing_ids.add(e.get("id"))
                            name_extras.append(e)
            if name_extras:
                # Prepend name-matched extras — they're likely more relevant
                # than FTS semantic matches when searching by location name
                results = name_extras + results

        # Format results, surfacing read_aloud prominently as boxed text
        lines = []
        for r in results:
            page_info = f", p.{r['page']}" if r.get("page") else ""
            header = f"**{r['name']}** (location) - {r.get('book', '')}{page_info}"
            md = r.get("metadata", {})
            read_aloud = md.get("read_aloud", "")
            content = r.get("content", "")
            if len(content) > 1500:
                content = content[:1500] + "..."
            entry = header
            if read_aloud:
                entry += f"\n\n*Read-aloud text:*\n> {read_aloud}"
            if content:
                entry += f"\n\n{content}"
            lines.append(entry)

        return ToolResult(success=True, data=f"[{len(results)} results]\n\n" + "\n\n---\n\n".join(lines))

    # Core rulebooks that should be authoritative for rules text.
    _CORE_RULEBOOKS = frozenset({
        "Player Core", "Player Core 2", "GM Core", "Monster Core",
    })

    def _search_rules(self, query: str, limit: int) -> ToolResult:
        """Search for rules content.

        When the top result is from a non-core rulebook (e.g. Book of the Dead)
        with short content, supplements with page text from core rulebooks to
        ensure the canonical rules text is included.
        """
        results = self.search.search(
            query,
            book_type=["rulebook"],
            exclude_types=["creature", "npc", "spell", "cantrip", "focus_spell"],
            top_k=limit,
        )

        if not results:
            return ToolResult(success=True, data=f"No rules found matching '{query}'")

        # Supplement thin non-core results with core rulebook page text.
        # This catches cases where consolidation picked a supplement's content
        # (e.g. Book of the Dead's Treat Wounds variant) over the core rules.
        top = results[0]
        top_content_len = len(top.get("content", ""))
        top_book = top.get("book", "")
        if top_content_len < 600 and top_book not in self._CORE_RULEBOOKS:
            page_results = self.search.search_pages(
                query, top_k=2, book=None, preprocess=True,
            )
            core_pages = [
                p for p in page_results
                if p.get("book", "") in self._CORE_RULEBOOKS
            ]
            if core_pages:
                page = core_pages[0]
                snippet = page.get("snippet", page.get("content", ""))
                if len(snippet) > 2000:
                    snippet = snippet[:2000] + "..."
                supplement = (
                    f"**{top['name']}** (core rules) - "
                    f"{page['book']}, p.{page.get('page_number', '?')}\n{snippet}"
                )
                formatted = self._format_results(results)
                return ToolResult(
                    success=True,
                    data=formatted + "\n\n---\n\n" + supplement,
                )

        return ToolResult(success=True, data=self._format_results(results))

    # Type aliases: agents often request "rule" but game content uses
    # more specific types. Expand to avoid zero-result searches.
    _TYPE_EXPANSIONS = {
        # --- rules / mechanics ---
        "rule": ["rule", "game_mechanic", "subsystem", "guidance", "variant_rule", "action"],
        "rules": ["rule", "game_mechanic", "subsystem", "guidance", "variant_rule", "action"],
        "mechanic": ["rule", "game_mechanic", "subsystem", "action"],
        "mechanics": ["rule", "game_mechanic", "subsystem", "action"],
        "game_rule": ["rule", "game_mechanic"],
        "game rule": ["rule", "game_mechanic"],
        "variant_rule": ["rule", "game_mechanic"],
        "subsystem_rule": ["subsystem", "rule", "game_mechanic"],
        "ability": ["feat", "action", "rule", "game_mechanic"],
        "abilities": ["feat", "action", "rule", "game_mechanic"],
        "activity": ["action"],
        "activities": ["action"],
        "class_feature": ["class", "feat", "action"],
        "class_features": ["class", "feat", "action"],
        "talent": ["feat"],
        "talents": ["feat"],
        "power": ["spell", "focus_spell", "feat", "action"],
        "powers": ["spell", "focus_spell", "feat", "action"],
        # --- classes / character options ---
        "class": ["class", "class_feature"],
        "classes": ["class", "class_feature"],
        "character_class": ["class"],
        "character class": ["class"],
        "multiclass": ["archetype"],
        "multiclass_archetype": ["archetype"],
        "prestige_class": ["archetype"],
        # --- spells / magic ---
        "magic": ["spell", "focus_spell", "cantrip", "item", "equipment"],
        "spells": ["spell", "cantrip", "focus_spell"],
        "cantrips": ["cantrip"],
        "focus_spells": ["focus_spell"],
        "incantation": ["spell"],
        # --- equipment / items ---
        "weapon": ["equipment", "item"],
        "weapons": ["equipment", "item"],
        "armor": ["equipment", "item"],
        "armour": ["equipment", "item"],
        "gear": ["equipment", "item"],
        "loot": ["equipment", "item"],
        "treasure": ["equipment", "item"],
        "magic_item": ["item", "equipment"],
        "magic item": ["item", "equipment"],
        "magic_items": ["item", "equipment"],
        "rune": ["equipment", "item"],
        "runes": ["equipment", "item"],
        "consumable": ["item", "equipment"],
        "consumables": ["item", "equipment"],
        "potion": ["item", "equipment"],
        "potions": ["item", "equipment"],
        "scroll": ["item", "equipment"],
        "wand": ["item", "equipment"],
        # --- creatures / monsters ---
        "monster": ["creature"],
        "monsters": ["creature"],
        "beast": ["creature"],
        "beasts": ["creature"],
        "enemy": ["creature", "npc"],
        "enemies": ["creature", "npc"],
        "foe": ["creature", "npc"],
        "foes": ["creature", "npc"],
        "stat_block": ["creature"],
        "statblock": ["creature"],
        "creature_stat_block": ["creature"],
        "creature_template": ["creature"],
        "creature_type": ["creature_family"],
        "monster_type": ["creature_family"],
        "dragon_family": ["creature_family"],
        "creature_families": ["creature_family"],
        # --- npcs / characters ---
        "character": ["npc"],
        "characters": ["npc"],
        "person": ["npc"],
        "people": ["npc"],
        "villager": ["npc"],
        "villain": ["npc"],
        "ally": ["npc"],
        "contact": ["npc"],
        "recurring_character": ["npc"],
        # --- conditions / effects ---
        "conditions": ["condition"],
        "status": ["condition"],
        "status_effect": ["condition"],
        "status effect": ["condition"],
        "effect": ["condition", "rule"],
        "effects": ["condition", "rule"],
        # --- hazards / traps ---
        "trap": ["hazard"],
        "traps": ["hazard"],
        "hazards": ["hazard"],
        "obstacle": ["hazard"],
        "environmental_hazard": ["hazard"],
        "haunt": ["hazard"],
        # --- locations / encounters ---
        "encounter": ["location", "hazard"],
        "encounters": ["location", "hazard"],
        "encounter_area": ["location", "hazard"],
        "room": ["location"],
        "rooms": ["location"],
        "area": ["location", "landmark"],
        "areas": ["location", "landmark"],
        "dungeon_room": ["location"],
        "chamber": ["location"],
        "lair": ["location"],
        # --- places / geography ---
        "district": ["landmark"],
        "districts": ["landmark"],
        "neighborhood": ["landmark"],
        "quarter": ["landmark"],
        "building": ["landmark"],
        "poi": ["landmark"],
        "city": ["settlement"],
        "town": ["settlement"],
        "village": ["settlement"],
        "settlements": ["settlement"],
        "country": ["region"],
        "nation": ["region"],
        "province": ["region"],
        "territory": ["region"],
        "regions": ["region"],
        "place": ["settlement", "region", "landmark"],
        "places": ["settlement", "region", "landmark"],
        "location_npc": ["npc", "location"],
        # --- deities / religion ---
        "god": ["deity"],
        "gods": ["deity"],
        "goddess": ["deity"],
        "deities": ["deity"],
        "divine": ["deity"],
        "religion": ["deity", "organization"],
        "faith": ["deity"],
        "pantheon": ["deity"],
        # --- organizations / factions ---
        "faction": ["organization"],
        "factions": ["organization"],
        "guild": ["organization"],
        "guilds": ["organization"],
        "group": ["organization"],
        "groups": ["organization"],
        "organization": ["organization"],
        "organizations": ["organization"],
        # --- ancestry / backgrounds ---
        "race": ["ancestry"],
        "races": ["ancestry"],
        "species": ["ancestry"],
        "ancestries": ["ancestry"],
        "heritage": ["heritage"],
        "heritages": ["heritage"],
        "backgrounds": ["background"],
        "archetypes": ["archetype"],
        # --- history / lore ---
        "event": ["historical_event"],
        "events": ["historical_event"],
        "history": ["historical_event", "region", "organization"],
        "lore": ["historical_event", "region", "organization", "npc"],
        "lore_event": ["historical_event"],
        # --- guidance / advice ---
        "advice": ["guidance"],
        "gm_advice": ["guidance"],
        "gm advice": ["guidance"],
        "tip": ["guidance"],
        "tips": ["guidance"],
        # --- book-level / invalid (drop filter) ---
        "adventure": ["location", "landmark", "npc", "hazard", "encounter_area", "guidance"],
        "adventure_path": [],      # drop — AP discovery uses browse_book
        "ap": [],
        "sourcebook": [],
        "source_book": [],
        "source book": [],
        "setting": [],             # setting content spans all types
        "setting_book": [],
        "book": [],
        "module": [],
        "chapter": [],
        "article": [],
        "other": [],
    }

    def _search_content(
        self, query: str, types: list[str] | None, book: str | None, limit: int,
        level: int | None = None,
        level_range: tuple[int, int] | None = None,
        traits: list[str] | None = None,
        remaster_only: bool | None = None,
        chapter: str | None = None,
    ) -> ToolResult:
        """General content search."""
        kwargs = {"top_k": limit}
        if types:
            # Expand type aliases (e.g., "rule" → rule + game_mechanic + subsystem + ...)
            expanded = []
            for t in types:
                expanded.extend(self._TYPE_EXPANSIONS.get(t, [t]))
            expanded = list(dict.fromkeys(expanded))  # dedup, preserve order
            # Empty expansion (e.g., "book" → []) means drop the type filter entirely
            if expanded:
                kwargs["include_types"] = expanded
        if book:
            kwargs["book"] = book
        if chapter:
            kwargs["chapter"] = chapter
        if level is not None:
            kwargs["level"] = level
        if level_range is not None:
            kwargs["level_range"] = level_range
        if traits:
            kwargs["traits"] = traits
        if remaster_only:
            kwargs["is_remaster"] = True

        results = self.search.search(query, **kwargs)

        # Relationship/faction FTS often fails because content format (members/goals/territory)
        # doesn't match natural-language queries well. Fall back to list_entities (direct SQL).
        _rel_types = {"relationship", "faction"}
        if not results and book and types and set(types) <= _rel_types:
            rel_entities = self.search.list_entities(
                include_types=list(types), book=book, limit=limit,
            )
            if rel_entities:
                return ToolResult(success=True, data=self._format_results(rel_entities))

        # If still nothing, retry without book filter.
        # Models sometimes hallucinate which AP content is from.
        if not results and book:
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "book"}
            results = self.search.search(query, **fallback_kwargs)
            if results:
                actual_book = results[0].get("book", "unknown")
                note = f"(No results in '{book}' — showing results from all books. Top match is from: {actual_book})\n\n"
                return ToolResult(success=True, data=note + self._format_results(results))

        if not results:
            return ToolResult(success=True, data=f"No content found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _search_lore(
        self, query: str, limit: int,
        book: str | None = None, chapter: str | None = None,
    ) -> ToolResult:
        """Search for world lore and setting information."""
        lore_cats = ["location", "lore", "deity", "organization", "region", "settlement", "landmark", "historical_event"]
        scope_kwargs: dict = {}
        if book:
            scope_kwargs["book"] = book
        if chapter:
            scope_kwargs["chapter"] = chapter

        # FTS first — best for named entity lookups (Absalom, Mana Wastes, etc.)
        results = self.search.search(
            query,
            category=lore_cats,
            top_k=limit,
            demote_page_text=False,
            **scope_kwargs,
        )

        if not results:
            # Semantic fallback for conceptual queries with no FTS matches
            results = self.search.search(
                query,
                category=lore_cats,
                top_k=limit,
                use_semantic=True,
                demote_page_text=False,
                **scope_kwargs,
            )

        # If book-scoped search returned nothing, retry without book filter
        if not results and book:
            unscoped = {k: v for k, v in scope_kwargs.items() if k != "book"}
            results = self.search.search(
                query, category=lore_cats, top_k=limit,
                demote_page_text=False, **unscoped,
            )
            if results:
                actual_book = results[0].get("book", "unknown")
                note = f"(No results in '{book}' — showing results from all books. Top match is from: {actual_book})\n\n"
                return ToolResult(success=True, data=note + self._format_results(results))

        if not results:
            # Try page search as last resort
            page_results = self.search.search_pages(
                query, top_k=limit, book=book, chapter=chapter,
            )
            if page_results:
                return ToolResult(success=True, data=self._format_page_results(page_results))

        if not results:
            return ToolResult(success=True, data=f"No lore found matching '{query}'")

        return ToolResult(success=True, data=self._format_results(results))

    def _list_entities(
        self, entity_type: str, book: str | None, name_filter: str | None, limit: int,
        trait: str | None = None, min_level: int | None = None, max_level: int | None = None,
    ) -> ToolResult:
        """List entities by type, optionally filtered by book, name, trait, and level."""
        results = self.search.list_entities(
            include_types=[entity_type],
            book=book,
            name_filter=name_filter,
            limit=limit,
            trait=trait,
            min_level=min_level,
            max_level=max_level,
        )

        if not results:
            msg = f"No {entity_type} entities found"
            if book:
                msg += f" in '{book}'"
            if name_filter:
                msg += f" matching '{name_filter}'"
            if trait:
                msg += f" with trait '{trait}'"
            if min_level is not None or max_level is not None:
                lo = min_level if min_level is not None else "?"
                hi = max_level if max_level is not None else "?"
                msg += f" at level {lo}-{hi}"
            return ToolResult(success=True, data=msg)

        lines = [f"Found {len(results)} {entity_type} entries:\n"]
        for r in results:
            meta = r.get("metadata", {})
            meta_parts = self._format_metadata(entity_type, meta)
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            book_str = r.get("book", "")
            # Shorten book name for compact display
            if "]" in book_str:
                book_str = book_str.split("]", 1)[-1].strip()
            lines.append(f"- **{r['name']}**{meta_str} — {book_str}")

        if len(results) >= limit:
            lines.append(f"\n[Showing first {limit} results. Use name_filter to narrow or increase limit for more.]")

        result_data = "\n".join(lines)

        # When listing NPCs in an AP book, automatically append relationship/faction data.
        # This surfaces social dynamics without requiring an extra tool call.
        if entity_type in ("npc", "npc_group") and book:
            rel_entities = self.search.list_entities(
                include_types=["relationship"], book=book, limit=20,
            )
            fac_entities = self.search.list_entities(
                include_types=["faction"], book=book, limit=15,
            )
            supplements = []
            if rel_entities:
                rel_lines = [f"- **{e['name']}**: {e.get('content','')[:150]}" for e in rel_entities]
                supplements.append("**NPC Relationships:**\n" + "\n".join(rel_lines))
            if fac_entities:
                fac_lines = [f"- **{e['name']}**: {e.get('content','')[:150]}" for e in fac_entities]
                supplements.append("**Factions:**\n" + "\n".join(fac_lines))
            if supplements:
                result_data += "\n\n" + "\n\n".join(supplements)

        return ToolResult(success=True, data=result_data)

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

    # Title prefixes and nickname mappings used for NPC name variant generation.
    _NPC_TITLE_PREFIXES = frozenset(
        {"mayor", "captain", "doctor", "father", "mother", "elder",
         "governor", "lord", "lady", "chief", "commander", "sergeant",
         "sheriff", "marshal", "warden", "baron", "baroness", "count",
         "countess", "duke", "duchess", "prince", "princess", "king", "queen"}
    )
    _NPC_NICKNAME_MAP = {
        "grandmother": "granny",
        "granny": "grandmother",
        "grandma": "granny",
    }

    @staticmethod
    def _npc_name_variants(name: str) -> list[str]:
        """Return alternative name forms to try when the primary search fails.

        Generates: title-stripped form, nickname substitution, surname-only.
        """
        variants: list[str] = []
        words = name.split()
        if not words:
            return variants

        lower_first = words[0].lower()

        # 1. Strip leading title (Mayor Ren → Ren; Captain Oleg → Oleg)
        if lower_first in PF2eRAGServer._NPC_TITLE_PREFIXES and len(words) > 1:
            variants.append(" ".join(words[1:]))

        # 2. Nickname substitution on first word (Grandmother Hu → Granny Hu)
        if lower_first in PF2eRAGServer._NPC_NICKNAME_MAP and len(words) > 1:
            replacement = PF2eRAGServer._NPC_NICKNAME_MAP[lower_first]
            variants.append(replacement.capitalize() + " " + " ".join(words[1:]))

        # 3. Surname only — useful when first name is a title or ambiguous
        if len(words) >= 2:
            variants.append(words[-1])

        # Deduplicate while preserving order
        seen: set[str] = {name.lower()}
        deduped: list[str] = []
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                deduped.append(v)
        return deduped

    def _lookup_npc(self, name: str, book: str | None = None) -> ToolResult:
        """Look up an NPC by name, merging NPC entity, creature stats, and page text.

        Also finds NPCs that are typed as 'creature' in the DB (common for named
        NPCs with stat blocks in APs). Creature-typed entries with unique rarity
        or from adventure/npc book types are treated as potential NPC matches.

        Tries name variants (nickname substitutions, title stripping) automatically
        when the primary search returns no name-matching results.
        """
        scope_kwargs: dict = {}
        if book:
            resolved = self.search.resolve_book_name(book)
            if resolved:
                scope_kwargs["book"] = resolved

        sections = []

        # 1. Search for NPC-typed entities (roleplay info, personality)
        npc_results = self.search.search(
            name, category="npc", top_k=3, preprocess=False, **scope_kwargs,
        )

        # 2. Search for creature-typed entities (stat blocks — many named NPCs
        #    are typed as 'creature' because they have stat blocks)
        creature_results = self.search.search(
            name, category="creature", top_k=3, preprocess=False, **scope_kwargs,
        )

        # 3. Search page text for narrative context
        page_results = self.search.search_pages(name, top_k=3)

        # Filter to best matches (name appears in result name, case-insensitive)
        # Prefer exact matches over substring matches to avoid e.g.
        # "Stag Lord" returning "Stag Lord Monarch" instead of "The Stag Lord"
        name_lower = name.lower()
        # Strip articles for matching
        _ARTICLES = {"the ", "a ", "an "}

        def _strip_article(s: str) -> str:
            for art in _ARTICLES:
                if s.startswith(art):
                    return s[len(art):]
            return s

        def _norm_name(s: str) -> str:
            """Strip punctuation and collapse whitespace for name comparison.

            Handles stored names with embedded quote chars like
            '"granny" Hu Ban-niang' so they match the query "granny hu".
            """
            return " ".join(
                "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s.lower()).split()
            )

        def is_name_match(result_name: str) -> bool:
            rn = _norm_name(result_name)
            qn = _norm_name(name)
            return qn in rn or rn in qn

        def _name_match_quality(result_name: str) -> int:
            """Higher = better match. Exact > prefix > substring.

            Multi-word queries matching longer names are demoted to avoid
            e.g. "Stag Lord" matching "Stag Lord Monarch" (different character).
            """
            rn = _strip_article(_norm_name(result_name))
            qn = _strip_article(_norm_name(name))
            if rn == qn:
                return 3  # exact match (ignoring articles)
            if rn.startswith(qn) or qn.startswith(rn):
                # Multi-word query matching a longer name = likely different entity
                rn_words = rn.split()
                qn_words = qn.split()
                if len(qn_words) >= 2 and len(rn_words) > len(qn_words):
                    return 1  # e.g., "Stag Lord" → "Stag Lord Monarch"
                return 2  # e.g., "Oleg" → "Oleg Leveton"
            return 1  # substring match

        npc_matches = [r for r in npc_results if is_name_match(r["name"])]
        # Sort by match quality so exact matches come first
        npc_matches.sort(key=lambda r: _name_match_quality(r["name"]), reverse=True)

        # For creature results, also include unique-rarity entries from AP books
        # (these are almost always named NPCs with stat blocks)
        creature_matches = []
        for r in creature_results:
            if is_name_match(r["name"]):
                creature_matches.append(r)
            elif r.get("book_type") in ("adventure", "npc"):
                meta = r.get("metadata", {})
                if isinstance(meta, str):
                    import json as _json
                    try:
                        meta = _json.loads(meta)
                    except (ValueError, TypeError):
                        meta = {}
                rarity = str(meta.get("rarity", "")).lower()
                if rarity == "unique":
                    creature_matches.append(r)
        creature_matches.sort(key=lambda r: _name_match_quality(r["name"]), reverse=True)

        # Filter to good name matches only (exact or prefix).
        # Substring-only matches are likely different entities
        # (e.g. "Stag Lord Monarch" when searching for "Stag Lord").
        # Other sections (creature stats, page text) provide safety net.
        def _filter_best(matches: list) -> list:
            return [r for r in matches if _name_match_quality(r["name"]) >= 2]

        npc_matches = _filter_best(npc_matches)
        creature_matches = _filter_best(creature_matches)

        # If no name matches yet, try automatic name variants (e.g. "Grandmother Hu"
        # → "Granny Hu"; "Mayor Ren" → "Ren") before falling through to page text.
        if not npc_matches and not creature_matches:
            for variant in self._npc_name_variants(name):
                v_lower = variant.lower()
                v_npc = self.search.search(
                    variant, category="npc", top_k=3, preprocess=False, **scope_kwargs,
                )
                v_cre = self.search.search(
                    variant, category="creature", top_k=3, preprocess=False, **scope_kwargs,
                )
                # Apply same match logic using the variant name
                v_norm = _norm_name(variant)

                def _is_match_v(rname: str) -> bool:  # noqa: E731
                    rn = _norm_name(rname)
                    return v_norm in rn or rn in v_norm

                def _quality_v(rname: str) -> int:  # noqa: E731
                    rn = _strip_article(_norm_name(rname))
                    qn = _strip_article(v_norm)
                    if rn == qn:
                        return 3
                    if rn.startswith(qn) or qn.startswith(rn):
                        rn_w = rn.split()
                        qn_w = qn.split()
                        if len(qn_w) >= 2 and len(rn_w) > len(qn_w):
                            return 1
                        return 2
                    return 1

                v_npc_m = [r for r in v_npc if _quality_v(r["name"]) >= 2]
                v_cre_m = [r for r in v_cre if _quality_v(r["name"]) >= 2]
                if v_npc_m or v_cre_m:
                    npc_matches = sorted(v_npc_m, key=lambda r: _quality_v(r["name"]), reverse=True)
                    creature_matches = sorted(v_cre_m, key=lambda r: _quality_v(r["name"]), reverse=True)
                    # Re-run page search with the variant that matched
                    page_results = self.search.search_pages(variant, top_k=3)
                    name_lower = v_lower  # update for page snippet filtering below
                    break

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
                        f"**{r['name']}** ({r.get('type', 'creature')}) - {r.get('book', '')}{page_info}\n\n"
                        f"{content}"
                    )

        # Only add page context when we actually found the NPC — otherwise page
        # text is too noisy (e.g. OR-fallback returns deity pages for a missing NPC).
        if page_results and (npc_matches or creature_matches):
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
            # Nothing found — apply name-quality filter to prevent false positives
            # (e.g. Grandmother Spider deity entry when searching for Grandmother Hu).
            # Legitimate deity-NPCs (Arazni, Tar-Baphon) are already typed as "npc"
            # or "creature" in the DB and will have been found above.
            all_results = [
                r for r in npc_results + creature_results
                if _name_match_quality(r["name"]) >= 2
            ]
            if all_results:
                return ToolResult(success=True, data=self._format_results(all_results))
            # Still nothing — fall back to page text. The entity may be stored under
            # a different name (e.g., "Governor Heh" vs "Heh Shan-Bao") but the pages
            # will contain the searched name verbatim.
            if page_results:
                page_sections = []
                for r in page_results:
                    ch_label = f" ({r['chapter']})" if r.get("chapter") else ""
                    page_sections.append(
                        f"**{r['book']}** p.{r['page_number']}{ch_label}\n{r.get('snippet', '')}"
                    )
                return ToolResult(
                    success=True,
                    data=(
                        f"No NPC entity found for '{name}'. Related page content:\n\n"
                        + "\n\n".join(page_sections)
                    ),
                )
            return ToolResult(success=True, data=f"No NPC found matching '{name}'")

        return ToolResult(success=True, data="\n\n---\n\n".join(sections))

    def _search_guidance(
        self, query: str, limit: int,
        book: str | None = None, chapter: str | None = None,
    ) -> ToolResult:
        """Search for GM advice and guidance."""
        scope_kwargs: dict = {}
        if book:
            scope_kwargs["book"] = book
        if chapter:
            scope_kwargs["chapter"] = chapter

        results = self.search.search(
            query,
            category=["guidance", "rule"],
            top_k=limit,
            use_semantic=True,
            **scope_kwargs,
        )

        if not results:
            results = self.search.search(
                query,
                category=["guidance", "rule"],
                top_k=limit,
                **scope_kwargs,
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

    def _search_pages(
        self, query: str, book: str | None, limit: int,
        chapter: str | None = None,
    ) -> ToolResult:
        """Search full page text from books."""
        results = self.search.search_pages(
            query, top_k=limit, book=book, chapter=chapter,
        )

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

        # Resolve user-friendly book name — may be multi-volume AP
        resolved_books = self.search.resolve_book_names(book)
        if len(resolved_books) > 1 and not chapter:
            # Multi-volume AP series — show all volumes
            lines = [f"**{book}** ({len(resolved_books)} volumes)", ""]
            for b in resolved_books:
                book_sum = self.search.get_book_summary(b)
                if book_sum:
                    summary = book_sum.get("summary", "")[:200]
                    lines.append(f"- **{b}** ({book_sum.get('total_pages', '?')} pages): {summary}...")
                else:
                    lines.append(f"- **{b}**")
            return ToolResult(success=True, data="\n".join(lines))
        elif resolved_books:
            book = resolved_books[0]
        else:
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
                    ch_line = (
                        f"- **{ch['chapter']}** (pp. {ch['page_start']}-{ch['page_end']}, {ch['page_count']} pages)"
                    )
                    snippet = (ch.get("summary") or "").strip()
                    if snippet:
                        # First sentence or 160 chars, whichever is shorter
                        dot = snippet.find(". ")
                        snippet = snippet[: dot + 1] if 0 < dot < 160 else snippet[:160]
                        ch_line += f" — {snippet}"
                    lines.append(ch_line)

            # Consolidated open threads across all chapters
            all_threads = []
            for ch in chapters:
                threads = ch.get("open_threads") or []
                if threads:
                    all_threads.append((ch["chapter"], threads))
            if all_threads:
                lines.append("")
                lines.append("**Open/Unresolved Threads (by chapter):**")
                for ch_name, threads in all_threads:
                    lines.append(f"\n*{ch_name}:*")
                    for thread in threads:
                        lines.append(f"- {thread}")

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
            if ch.get("open_threads"):
                lines.append(f"\n**Open threads (unresolved at end of chapter):**")
                for thread in ch["open_threads"]:
                    lines.append(f"- {thread}")
            lines.append("")

        # Get page-by-page summaries for that chapter
        page_summaries = self.search.get_page_summaries_for_chapter(book, chapter)

        if not ch and not page_summaries:
            available = self.search.list_chapters(book)
            # Try numeric positional match: "Chapter 2", "Part 2", "2" → 2nd chapter in TOC.
            # Skip front-matter entries (Introduction, Preface, Appendix, etc.) so that
            # "Chapter 2" means the 2nd narrative chapter, not the 3rd TOC entry overall.
            if available and chapter:
                import re as _re
                m = _re.search(r'\b(\d+)\b', chapter)
                if m:
                    _frontmatter = ("introduction", "preface", "appendix", "glossary", "index", "foreword")
                    _chapter_entries = [
                        c for c in available
                        if not c["chapter"].lower().startswith(_frontmatter)
                    ]
                    idx = int(m.group(1)) - 1  # convert to 0-indexed
                    if 0 <= idx < len(_chapter_entries):
                        return self._browse_book(book, _chapter_entries[idx]["chapter"])
            # List available chapters to help the agent correct the name
            if available:
                ch_names = ", ".join(f'"{c["chapter"]}"' for c in available[:15])
                hint = f" Available chapters: {ch_names}"
            else:
                hint = " Try browse_book(book='{book}') with no chapter to see the table of contents."
            return ToolResult(
                success=True,
                data=f"No chapter matching '{chapter}' found in {book}.{hint}",
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

    def _get_player_advice(
        self,
        situation: str,
        query_type: str,
        character_class: str | None,
        level: int | None,
    ) -> ToolResult:
        """Return structured rules + guidance to advise a player on their situation."""
        valid_types = ("combat", "exploration", "social", "mechanical", "character_build")
        if query_type not in valid_types:
            return ToolResult(
                success=False,
                error=f"Invalid query_type '{query_type}'. Valid: {', '.join(valid_types)}",
            )

        # Build query list and content types by query_type
        if query_type == "combat":
            queries = [situation, "flanking cover three-action economy", "conditions in combat"]
            include_types = ["rule", "action", "condition", "game_mechanic"]
        elif query_type == "exploration":
            queries = [situation, "exploration activities", "scouting searching traveling"]
            include_types = ["rule", "action", "game_mechanic"]
        elif query_type == "social":
            queries = [situation, "diplomacy deception intimidation", "attitude thresholds"]
            include_types = ["rule", "action", "condition"]
        elif query_type == "mechanical":
            queries = [situation]
            include_types = ["rule", "action", "condition", "game_mechanic"]
        else:  # character_build
            queries = [character_class or situation]
            include_types = ["class", "feat", "ancestry", "heritage"]

        # Level kwargs for content search
        level_kwargs: dict = {}
        if level is not None and query_type in ("character_build", "combat", "mechanical"):
            level_kwargs["level"] = level

        seen_ids: set[str] = set()
        rules_context: list[dict] = []
        guidance_context: list[dict] = []

        for query in queries:
            results = self.search.search(
                query, include_types=include_types, top_k=3, **level_kwargs
            )
            for r in results:
                rid = str(r.get("id") or r.get("name", ""))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    content = r.get("content", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    rules_context.append({
                        "name": r.get("name"),
                        "type": r.get("type"),
                        "book": r.get("book", ""),
                        "page": r.get("page"),
                        "content": content,
                    })

        # Guidance search for non-character-build queries
        if query_type != "character_build":
            guidance_results = self.search.search(
                queries[0], category=["guidance", "rule"], top_k=3
            )
            for r in guidance_results:
                rid = str(r.get("id") or r.get("name", ""))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    content = r.get("content", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    guidance_context.append({
                        "name": r.get("name"),
                        "type": r.get("type"),
                        "content": content,
                    })

        # Format output
        lines = [
            f"**Player Advice: {situation}**",
            f"*Query type: {query_type}*",
        ]

        if rules_context:
            lines.append(f"\n**Rules & Mechanics** ({len(rules_context)} results)")
            for entry in rules_context:
                page_info = f", p.{entry['page']}" if entry.get("page") else ""
                lines.append(
                    f"\n**{entry['name']}** ({entry['type']}) — {entry['book']}{page_info}"
                )
                if entry["content"]:
                    lines.append(entry["content"])
        else:
            lines.append("\n*No specific rules found — try a broader situation description.*")

        if guidance_context:
            lines.append(f"\n**GM/Player Guidance** ({len(guidance_context)} results)")
            for entry in guidance_context:
                lines.append(f"\n**{entry['name']}** ({entry['type']})")
                if entry["content"]:
                    lines.append(entry["content"])

        lines.append(f"\n*Use these rules to advise the player on: {situation}*")

        return ToolResult(success=True, data="\n".join(lines))

    @staticmethod
    def _format_creature_abilities(meta: dict) -> str:
        """Format creature special abilities as a separate block."""
        abilities = meta.get("abilities")
        if not abilities or not isinstance(abilities, list):
            return ""
        ab_lines = []
        for ab in abilities:
            if not isinstance(ab, dict):
                continue
            ab_name = ab.get("name", "")
            ab_desc = ab.get("description", "").strip().replace("\n", " ")
            cost = ab.get("action_cost", "")
            if not ab_name:
                continue
            cost_str = f" [{cost}]" if cost and cost != "passive" else ""
            desc_str = f" — {ab_desc[:300]}" if ab_desc else ""
            ab_lines.append(f"  - **{ab_name}**{cost_str}{desc_str}")
        if not ab_lines:
            return ""
        return "**Special Abilities:**\n" + "\n".join(ab_lines)

    def _format_results(self, results: list[dict]) -> str:
        """Format search results for display, including key metadata."""
        if not results:
            return ""
        formatted = []
        for r in results:
            page_info = f", p.{r['page']}" if r.get("page") else ""
            header = f"**{r['name']}** ({r['type']}) - {r.get('book', r.get('source', ''))}{page_info}"

            # Surface key metadata inline
            meta = r.get("metadata", {})
            meta_parts = self._format_metadata(r.get("type", ""), meta)
            if meta_parts:
                header += "\n" + " | ".join(meta_parts)

            # Special abilities for creatures — separate block after metadata line
            abilities_block = ""
            if r.get("type") in ("creature", "creature_family"):
                abilities_block = self._format_creature_abilities(meta)

            content = r.get("content", "")
            if len(content) > 2000:
                content = content[:2000] + "..."

            parts = [f"{header}\n{content}"]
            if abilities_block:
                parts.append(abilities_block)
            formatted.append("\n\n".join(parts))

        return f"[{len(results)} results]\n\n" + "\n\n---\n\n".join(formatted)

    # Trait values that are actually alignment abbreviations, sizes, or parse artifacts.
    # These are already captured in dedicated metadata fields or are noise.
    _NOISE_TRAITS = frozenset({
        "ce", "ne", "le", "cn", "ln", "ng", "lg", "cg", "n",
        "tiny", "small", "medium", "large", "huge", "gargantuan",
        "cc", "cd", "cb", "nn",
        "common", "uncommon", "rare", "unique",  # rarity already a field
    })

    @staticmethod
    def _clean_traits(raw_traits: list) -> list[str]:
        """Normalize and filter creature traits, removing noise."""
        cleaned = []
        seen = set()
        for t in raw_traits:
            t_lower = str(t).strip().lower()
            if (
                t_lower
                and t_lower not in PF2eRAGServer._NOISE_TRAITS
                and not t_lower.startswith("level ")
                and t_lower not in seen
            ):
                cleaned.append(t_lower)
                seen.add(t_lower)
        return cleaned

    @staticmethod
    def _format_metadata(entity_type: str, meta: dict) -> list[str]:
        """Extract key metadata fields for display based on entity type."""
        parts = []
        if entity_type in ("creature", "creature_family"):
            for key in ("level", "hp", "ac", "rarity", "size"):
                val = meta.get(key)
                if val is not None and val != "":
                    parts.append(f"{key.upper() if key in ('hp','ac') else key.title()}: {val}")
            # Saves (Fort/Ref/Will)
            saves = meta.get("saves")
            if saves and isinstance(saves, dict):
                saves_str = "/".join(
                    f"{k[0]}{'+' if int(v) >= 0 else ''}{v}"
                    for k, v in saves.items()
                    if v is not None
                )
                if saves_str:
                    parts.append(f"Saves: {saves_str}")
            # Immunities
            immunities = meta.get("immunities")
            if immunities:
                imm_list = immunities if isinstance(immunities, list) else [immunities]
                parts.append(f"Immunities: {', '.join(str(i) for i in imm_list)}")
            # Resistances
            resistances = meta.get("resistances")
            if resistances:
                if isinstance(resistances, list):
                    res_list = [f"{r.get('type','?')} {r.get('value','')}" if isinstance(r, dict) else str(r) for r in resistances]
                    parts.append(f"Resistances: {', '.join(res_list)}")
                else:
                    parts.append(f"Resistances: {resistances}")
            # Weaknesses
            weaknesses = meta.get("weaknesses")
            if weaknesses:
                if isinstance(weaknesses, list):
                    wk_list = [f"{w.get('type','?')} {w.get('value','')}" if isinstance(w, dict) else str(w) for w in weaknesses]
                    parts.append(f"Weaknesses: {', '.join(wk_list)}")
                else:
                    parts.append(f"Weaknesses: {weaknesses}")
            # Creature family
            family = meta.get("creature_family")
            if family:
                parts.append(f"Family: {family}")
            # Traits (cleaned)
            raw_traits = meta.get("traits")
            if raw_traits and isinstance(raw_traits, list):
                cleaned = PF2eRAGServer._clean_traits(raw_traits)
                if cleaned:
                    parts.append(f"Traits: {', '.join(cleaned)}")
            # NOTE: abilities are rendered as a separate block in _format_results, not here
        elif entity_type in ("deity", "religion"):
            for key, label in [
                ("favored_weapon", "Weapon"),
                ("divine_font", "Font"),
                ("divine_skill", "Skill"),
            ]:
                val = meta.get(key)
                if val:
                    parts.append(f"{label}: {val}")
            domains = meta.get("domains")
            if domains:
                if isinstance(domains, list):
                    parts.append(f"Domains: {', '.join(str(d) for d in domains)}")
                else:
                    parts.append(f"Domains: {domains}")
        elif entity_type in ("settlement", "region"):
            for key in ("population", "government"):
                val = meta.get(key)
                if val:
                    parts.append(f"{key.title()}: {val}")
        elif entity_type == "hazard":
            for key in ("level", "rarity"):
                val = meta.get(key)
                if val is not None and val != "":
                    parts.append(f"{key.title()}: {val}")
        return parts

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
