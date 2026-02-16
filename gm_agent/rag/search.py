#!/usr/bin/env python3
"""
Read-only Pathfinder search over pre-built search.db (from pf2e-extraction).

Primary: SQLite FTS5 keyword search with BM25 scoring
Secondary: Pre-computed embeddings for semantic/vector search
"""

import json
import math
import os
import re
import sqlite3
import struct
from typing import Optional

import numpy as np

# Local embeddings via sentence-transformers (lazy loaded for query encoding)
_embedding_model = None
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dims, matches pre-built embeddings

# Stop words for query preprocessing
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "how",
        "when",
        "where",
        "why",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "down",
        "during",
        "except",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "through",
        "to",
        "toward",
        "under",
        "until",
        "up",
        "upon",
        "with",
        "within",
        "tell",
        "explain",
        "describe",
        "give",
        "show",
        "find",
        "get",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "so",
        "because",
        "although",
        "please",
        "thanks",
        "help",
        "need",
        "want",
        "like",
        "know",
        # Game term stop words - indicate type but shouldn't be required in results
        "spell",
        "spells",
        "feat",
        "feats",
        "action",
        "actions",
        "item",
        "items",
        "creature",
        "creatures",
        "monster",
        "monsters",
        "npc",
        "npcs",
        "equipment",
        "rule",
        "rules",
        "condition",
        "conditions",
        "trait",
        "traits",
        "ability",
        "abilities",
        "class",
        "classes",
        "ancestry",
        "ancestries",
        "background",
        "backgrounds",
        "heritage",
        "heritages",
    }
)



def preprocess_query(query: str) -> str:
    """
    Preprocess natural language query into keywords for FTS5.

    Handles questions like "What is a goblin?" -> "goblin"
    """
    query = query.lower()
    # Remove punctuation except hyphens (for terms like "off-guard")
    query = re.sub(r"[^\w\s\-]", " ", query)
    words = query.split()
    keywords = [w for w in words if w not in STOP_WORDS and len(w) >= 2]
    if not keywords and words:
        keywords = [w for w in words if len(w) >= 3]
    if not keywords:
        return query.strip()
    return " ".join(keywords)


def _decompose_complex_query(
    query: str,
    condition_names: set[str] | None = None,
    class_names: set[str] | None = None,
) -> list[dict]:
    """
    Decompose a complex question into simpler sub-queries.

    Args:
        query: Natural language query
        condition_names: Set of known condition names (lowercased). If None,
            condition detection is skipped.
        class_names: Set of known class names (lowercased). If None,
            class detection is skipped.

    Returns list of {query, type_hint, description} dicts.
    """
    query_lower = query.lower()
    sub_queries = []

    # Pattern: weapon damage questions
    # NOTE: weapons/runes remain hardcoded until pf2e-extraction adds
    # fine-grained equipment sub-typing (see pf2e-extraction TODO).
    if any(w in query_lower for w in ["damage", "hit", "attack"]) and any(
        w in query_lower for w in ["sword", "axe", "weapon", "bow", "spear", "dagger", "mace"]
    ):
        weapons = [
            "longsword",
            "shortsword",
            "greatsword",
            "battleaxe",
            "greataxe",
            "longbow",
            "shortbow",
            "dagger",
            "rapier",
            "mace",
            "warhammer",
            "spear",
            "glaive",
            "halberd",
            "flail",
            "scimitar",
        ]
        for weapon in weapons:
            if weapon in query_lower:
                sub_queries.append(
                    {
                        "query": weapon,
                        "type_hint": "equipment",
                        "description": f"{weapon} base stats",
                    }
                )
                break

        runes = [
            "striking",
            "greater striking",
            "major striking",
            "potency",
            "flaming",
            "frost",
            "shock",
            "corrosive",
            "holy",
            "unholy",
        ]
        for rune in runes:
            if rune in query_lower:
                sub_queries.append(
                    {
                        "query": f"{rune} rune",
                        "type_hint": "equipment",
                        "description": f"{rune} rune effect",
                    }
                )

        if "damage" in query_lower:
            sub_queries.append(
                {
                    "query": "weapon damage dice",
                    "type_hint": "rule",
                    "description": "how weapon damage works",
                }
            )

    # Pattern: spell questions
    elif any(w in query_lower for w in ["cast", "spell", "magic", "cantrip"]):
        spell_patterns = [
            (r"cast\s+(\w+(?:\s+\w+)?)", "spell"),
            (r"(\w+(?:\s+\w+)?)\s+spell", "spell"),
        ]
        for pattern, type_hint in spell_patterns:
            match = re.search(pattern, query_lower)
            if match:
                spell_name = match.group(1)
                if spell_name not in STOP_WORDS:
                    sub_queries.append(
                        {
                            "query": spell_name,
                            "type_hint": "spell",
                            "description": f"{spell_name} spell details",
                        }
                    )
                    break

    # Pattern: class/level questions (classes loaded from DB)
    elif class_names and any(w in query_lower for w in ["level", "class", "feature", "ability"]):
        for cls in class_names:
            if cls in query_lower:
                sub_queries.append(
                    {
                        "query": cls,
                        "type_hint": "class",
                        "description": f"{cls} class features",
                    }
                )
                level_match = re.search(r"level\s*(\d+)", query_lower)
                if level_match:
                    level = level_match.group(1)
                    sub_queries.append(
                        {
                            "query": f"{cls} level {level}",
                            "type_hint": "class_feature",
                            "description": f"{cls} level {level} features",
                        }
                    )
                break

    # Pattern: DC/difficulty questions
    if "dc" in query_lower or "difficulty" in query_lower:
        sub_queries.append(
            {
                "query": "difficulty class",
                "type_hint": "rule",
                "description": "DC rules",
            }
        )
        level_match = re.search(r"level\s*(\d+)", query_lower)
        if level_match or "level" in query_lower:
            sub_queries.append(
                {
                    "query": "DC by level table",
                    "type_hint": "table",
                    "description": "DC table by level",
                }
            )

    # Pattern: condition questions (conditions loaded from DB)
    if condition_names:
        found_conditions = [cond for cond in condition_names if cond in query_lower]
        if found_conditions:
            for cond in found_conditions:
                sub_queries.append(
                    {
                        "query": cond,
                        "type_hint": "condition",
                        "description": f"{cond} condition",
                    }
                )
            if "happen" in query_lower or "when" in query_lower or "how" in query_lower:
                if "dying" in found_conditions:
                    sub_queries.append(
                        {
                            "query": "death and dying",
                            "type_hint": "rule",
                            "description": "death rules",
                        }
                    )
                    sub_queries.append(
                        {
                            "query": "recovery check",
                            "type_hint": "rule",
                            "description": "recovery from dying",
                        }
                    )
        elif any(w in query_lower for w in ["condition", "status", "effect"]):
            sub_queries.append(
                {
                    "query": preprocess_query(query),
                    "type_hint": "condition",
                    "description": "condition search",
                }
            )

    if not sub_queries:
        sub_queries.append(
            {
                "query": preprocess_query(query),
                "type_hint": None,
                "description": "general search",
            }
        )

    return sub_queries


def get_embedding_model():
    """Lazy load the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return None
    return _embedding_model


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding as numpy array using local sentence-transformers model."""
    model = get_embedding_model()
    if model is None:
        return None
    try:
        return model.encode(text, convert_to_numpy=True)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def get_embeddings_batch(texts: list[str]) -> Optional[np.ndarray]:
    """Get embeddings for multiple texts as numpy array."""
    model = get_embedding_model()
    if model is None:
        return None
    try:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100)
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def pack_embedding(embedding: list[float]) -> bytes:
    """Pack embedding as binary for efficient storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def unpack_embedding(data: bytes) -> list[float]:
    """Unpack embedding from binary storage."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


# Load remaster aliases for legacy->remaster term expansion
REMASTER_ALIASES = {}
REMASTER_ALIASES_PATH = os.path.join(os.path.dirname(__file__), "remaster_aliases.json")
if os.path.exists(REMASTER_ALIASES_PATH):
    try:
        with open(REMASTER_ALIASES_PATH) as f:
            _alias_data = json.load(f)
            for category, mappings in _alias_data.items():
                if category == "meta" or not isinstance(mappings, dict):
                    continue
                for legacy, remaster in mappings.items():
                    if remaster:
                        REMASTER_ALIASES[legacy.lower()] = remaster.lower()
    except Exception as e:
        print(f"Warning: Could not load remaster aliases: {e}")


def expand_query_aliases(query: str) -> list[str]:
    """Expand query with remaster aliases. Returns list of queries to try."""
    query_lower = query.lower()
    queries = [query]

    if query_lower in REMASTER_ALIASES:
        queries.append(REMASTER_ALIASES[query_lower])

    for legacy, remaster in REMASTER_ALIASES.items():
        if query_lower == remaster and legacy not in queries:
            queries.append(legacy)

    words = query_lower.split()
    if len(words) > 1:
        reverse_aliases = {v: k for k, v in REMASTER_ALIASES.items()}
        expanded_words = []
        has_expansion = False
        for word in words:
            if word in REMASTER_ALIASES:
                expanded_words.append(REMASTER_ALIASES[word])
                has_expansion = True
            elif word in reverse_aliases:
                expanded_words.append(reverse_aliases[word])
                has_expansion = True
            else:
                expanded_words.append(word)

        if has_expansion:
            expanded_query = " ".join(expanded_words)
            if expanded_query not in queries:
                queries.append(expanded_query)

    return queries


# Mapping from deprecated filter params to new schema equivalents
CATEGORY_ALIASES = {
    # Old source_categories -> new book_type
    "core": {"book_type": "rulebook"},
    "supplements": {"book_type": "rulebook"},
    "rules": {"book_type": "rulebook"},
    "bestiaries": {"book_type": "bestiary"},
    "monsters": {"book_type": "bestiary"},
    "creatures": {"book_type": "bestiary"},
    "setting": {"book_type": "setting"},
    "world": {"book_type": "setting"},
    "lost_omens": {"book_type": "setting"},
    "players_guides": {"book_type": "players_guide"},
    "adventures": {"book_type": "adventure"},
    "aps": {"book_type": "adventure"},
}


class PathfinderSearch:
    """Read-only search engine over pre-built search.db from pf2e-extraction."""

    # Type rankings for GM agent use (higher = more preferred)
    DEFAULT_TYPE_BOOST = {
        # Core game content (highest priority)
        "spell": 20,
        "cantrip": 20,
        "focus_spell": 20,
        "ritual": 15,
        "creature": 25,
        "creature_family": 15,
        "creature_template": 12,
        "npc": 22,
        "condition": 25,
        "rule": 22,
        "variant_rule": 18,
        "action": 20,
        "feat": 18,
        "class_feature": 15,
        "equipment": 15,
        "item": 15,
        # Player character options — lower priority for GM tool
        "ancestry": 8,
        "heritage": 6,
        "background": 6,
        "class": 10,
        "archetype": 8,
        "trait": 10,
        "deity": 12,
        "hazard": 20,
        "haunt": 20,
        "affliction": 18,
        # Setting/lore content
        "location": 10,
        "settlement": 10,
        "region": 8,
        "landmark": 10,
        "organization": 10,
        "historical_event": 6,
        "npc_group": 8,
        "faction": 8,
        # GM content (high priority for GM agent)
        "game_mechanic": 20,
        "guidance": 18,
        "read_aloud": 15,
        "template": 8,
        # Reference content (lower priority)
        "table": 5,
        "subsystem": 8,
        "example_of_play": 3,
    }

    # Book type rankings — authoritative sources ranked higher for dedup
    # when multiple entries share the same name (e.g., "Recall Knowledge"
    # action in Player Core vs AP hazard with the same name).
    BOOK_TYPE_BOOST = {
        "rulebook": 15,
        "bestiary": 10,
        "players_guide": 7,
        "setting": 5,
        "npc": 3,
        "adventure": 0,
    }

    REQUIRED_TABLES = {"content", "content_fts", "pages", "pages_fts", "embeddings"}

    def __init__(self, db_path: str = "pathfinder_search.db"):
        self.db_path = db_path
        self.conn = None
        self._open_readonly()
        self._condition_names, self._class_names = self._load_term_sets()

    def _open_readonly(self):
        """Open database in read-only mode, validate schema."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Search database not found: {self.db_path}\n"
                "Build it with: cd pf2e-extraction && uv run python cli.py index"
            )

        uri = f"file:{self.db_path}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = sqlite3.Row

        # Validate required tables exist
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
        tables = {row["name"] for row in cursor}
        missing = self.REQUIRED_TABLES - tables
        if missing:
            raise RuntimeError(
                f"Database missing required tables: {missing}\n"
                f"Found: {tables}\n"
                "This database may use an older schema. Rebuild with pf2e-extraction."
            )

    def _load_term_sets(self) -> tuple[set[str], set[str]]:
        """Load condition and class names from the DB for search enhancement."""
        condition_names = set()
        class_names = set()
        try:
            cursor = self.conn.execute(
                "SELECT LOWER(name) FROM content WHERE type = 'condition'"
            )
            condition_names = {row[0] for row in cursor}
            cursor = self.conn.execute(
                "SELECT LOWER(name) FROM content WHERE type = 'class'"
            )
            class_names = {row[0] for row in cursor}
        except Exception:
            pass  # Graceful degradation — search still works without term sets
        return condition_names, class_names

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        use_semantic: bool = False,
        preprocess: bool = True,
        boost_types: dict = None,
        demote_page_text: bool = True,
        include_types: list[str] = None,
        exclude_types: list[str] = None,
        # New schema filters
        category: str | list[str] = None,
        book_type: str | list[str] = None,
        book: str = None,
        # Deprecated params (mapped to new equivalents)
        source: str = None,
        source_categories: list[str] = None,
        edition: str | list = None,
        exclude_summaries: bool = False,
        exclude_raw: bool = False,
    ) -> list[dict]:
        """
        Search for entities in the content table.

        Args:
            query: Search query (natural language supported with preprocessing)
            top_k: Number of results to return
            doc_type: Filter by document type (single type)
            use_semantic: Use semantic/vector search instead of FTS5
            preprocess: Apply query preprocessing (default True)
            boost_types: Dict of {type: boost_value} to adjust ranking
            demote_page_text: If True, apply default type boosts (default True)
            include_types: Only include these types (whitelist)
            exclude_types: Exclude these types (blacklist)
            category: Filter by broad category (spell, feat, creature, etc.)
            book_type: Filter by book type (rulebook, bestiary, adventure, setting)
            book: Filter by exact book name
            source: Deprecated alias for book
            source_categories: Deprecated - mapped to book_type via CATEGORY_ALIASES
            edition: Deprecated - ignored (all content is remaster)
            exclude_summaries: Deprecated - no summary types in new schema
            exclude_raw: Deprecated - no raw types in new schema
        """
        # Map deprecated source_categories to book_type
        if source_categories and not book_type:
            book_types = set()
            for cat in source_categories:
                alias = CATEGORY_ALIASES.get(cat.lower())
                if alias and "book_type" in alias:
                    book_types.add(alias["book_type"])
            if book_types:
                book_type = list(book_types)

        # Map deprecated source to book
        if source and not book:
            book = source

        # Resolve fuzzy book name to exact DB name
        if book:
            resolved = self.resolve_book_name(book)
            if resolved:
                book = resolved

        # Normalize category/book_type to lists
        if isinstance(category, str):
            category = [category]
        if isinstance(book_type, str):
            book_type = [book_type]

        # Build type exclusion set
        type_exclusions = set()
        if exclude_types:
            type_exclusions.update(exclude_types)

        # Preprocess query
        original_query = query
        if preprocess:
            query = preprocess_query(query)
            if not query.strip():
                query = original_query

        # Determine type boosts
        type_boosts = self.DEFAULT_TYPE_BOOST.copy() if demote_page_text else {}
        if boost_types:
            type_boosts.update(boost_types)

        if use_semantic:
            results = self._search_semantic(
                original_query,
                top_k * 2,
                doc_type,
                category,
                book_type,
                book,
                include_types,
                type_exclusions,
            )
        else:
            results = self._search_fts(
                query,
                top_k * 2,
                doc_type,
                category,
                book_type,
                book,
                original_query,
                include_types,
                type_exclusions,
            )

        # Apply type-based score adjustments
        if type_boosts and results:
            for r in results:
                type_boost = type_boosts.get(r["type"], 0)
                r["score"] = r["score"] + type_boost
                r["type_boost"] = type_boost

            results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _search_fts(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        category: list[str] = None,
        book_type: list[str] = None,
        book: str = None,
        original_query: str = None,
        include_types: list = None,
        type_exclusions: set = None,
    ) -> list[dict]:
        """Full-text search using FTS5 with exact match boosting and alias expansion."""
        boost_query = original_query or query

        # Expand with remaster aliases
        queries = set(expand_query_aliases(query))
        if original_query and original_query != query:
            queries.update(expand_query_aliases(original_query))
        queries = list(queries)

        all_results = {}

        for q in queries:
            results = self._search_fts_single(
                q, top_k * 2, doc_type, category, book_type, book,
                include_types, type_exclusions,
            )
            for r in results:
                key = (r["name"], r["source"])
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # Boost conditions/rules when query contains known terms
        if doc_type is None and self._condition_names:
            query_lower = boost_query.lower()
            for term in self._condition_names:
                if term in query_lower:
                    term_results = self._search_fts_single(
                        term, 5, doc_type=None, category=category,
                        book_type=book_type, book=book,
                        include_types=include_types, type_exclusions=type_exclusions,
                    )
                    for r in term_results:
                        if r["type"] in ("condition", "rule", "trait"):
                            key = (r["name"], r["source"])
                            r["score"] = r["score"] + 50
                            if key not in all_results or r["score"] > all_results[key]["score"]:
                                all_results[key] = r

        # Boost results from a book whose title matches the query.
        # E.g., searching "Absalom" boosts entries from [PZO9304E] Absalom.
        if doc_type is None and not book:
            resolved_book = self.resolve_book_name(boost_query)
            if resolved_book:
                for r in all_results.values():
                    if r.get("book") == resolved_book:
                        r["score"] += 30
                        r["book_title_boost"] = 30

        # Apply book_type boost before dedup so authoritative sources
        # (rulebooks) win when multiple entries share the same name.
        # Also boost remaster content (PZO12xxx/13xxx/15xxx) over legacy
        # (PZO2xxx/9xxx) so Monster Core beats Bestiary for duplicate names.
        for r in all_results.values():
            boost = self.BOOK_TYPE_BOOST.get(r.get("book_type", ""), 0)
            book_name = r.get("book", "")
            if book_name.startswith(("[PZO12", "[PZO13", "[PZO15")):
                boost += 8  # Remaster content boost
            r["score"] += boost
            r["book_type_boost"] = boost

        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        # Dedupe by name (case-insensitive)
        seen_names = set()
        deduped = []
        for r in sorted_results:
            name_lower = r["name"].lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                deduped.append(r)

        return deduped[:top_k]

    def _search_fts_single(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        category: list[str] = None,
        book_type: list[str] = None,
        book: str = None,
        include_types: list = None,
        type_exclusions: set = None,
    ) -> list[dict]:
        """Single FTS5 search without alias expansion."""
        # content has TEXT PK (id), FTS5 uses implicit integer rowid
        sql = """
            SELECT c.*,
                   bm25(content_fts) - (CASE WHEN LOWER(c.name) = LOWER(?) THEN 100 ELSE 0 END) as score
            FROM content c
            JOIN content_fts ON c.rowid = content_fts.rowid
            WHERE content_fts MATCH ?
        """
        params: list = [query, query]

        if doc_type:
            sql += " AND c.type = ?"
            params.append(doc_type)

        if category:
            placeholders = ",".join("?" * len(category))
            sql += f" AND c.category IN ({placeholders})"
            params.extend(category)

        if book_type:
            placeholders = ",".join("?" * len(book_type))
            sql += f" AND c.book_type IN ({placeholders})"
            params.extend(book_type)

        if book:
            sql += " AND c.book = ?"
            params.append(book)

        if include_types:
            placeholders = ",".join("?" * len(include_types))
            sql += f" AND c.type IN ({placeholders})"
            params.extend(include_types)

        if type_exclusions:
            placeholders = ",".join("?" * len(type_exclusions))
            sql += f" AND c.type NOT IN ({placeholders})"
            params.extend(type_exclusions)

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        try:
            cursor = self.conn.execute(sql, params)
            return [self._row_to_result(row) for row in cursor]
        except sqlite3.OperationalError:
            # FTS5 query syntax error - try quoted query
            safe_query = '"' + query.replace('"', '""') + '"'
            params[1] = safe_query
            try:
                cursor = self.conn.execute(sql, params)
                return [self._row_to_result(row) for row in cursor]
            except Exception:
                return []

    def _row_to_result(self, row: sqlite3.Row) -> dict:
        """Convert a content table row to a result dict."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return {
            "name": row["name"],
            "type": row["type"],
            "category": row["category"],
            "source": row["book"],  # backward compat alias
            "book": row["book"],
            "book_type": row["book_type"],
            "page": row["page"],
            "content": row["content"],
            "metadata": metadata,
            "score": -row["score"],  # BM25 returns negative (more negative = better)
        }

    def _search_semantic(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        category: list[str] = None,
        book_type: list[str] = None,
        book: str = None,
        include_types: list = None,
        type_exclusions: set = None,
    ) -> list[dict]:
        """Semantic search using pre-computed embeddings with numpy vectorized dot product."""
        query_embedding = get_embedding(query)
        if query_embedding is None:
            print("Failed to get query embedding, falling back to FTS")
            return self._search_fts(
                query, top_k, doc_type, category, book_type, book,
                None, include_types, type_exclusions,
            )

        # Build filter SQL for embeddings table
        sql = """
            SELECT e.source_id, e.chunk_text, e.embedding, e.book, e.page_number, e.source_type
            FROM embeddings e
            WHERE e.source_type = 'entity'
        """
        params = []

        if book:
            sql += " AND e.book = ?"
            params.append(book)

        if book_type:
            # Join with content to filter by book_type
            sql = """
                SELECT e.source_id, e.chunk_text, e.embedding, e.book, e.page_number, e.source_type
                FROM embeddings e
                JOIN content c ON e.source_id = c.id
                WHERE e.source_type = 'entity'
            """
            placeholders = ",".join("?" * len(book_type))
            sql += f" AND c.book_type IN ({placeholders})"
            params.extend(book_type)
            if book:
                sql += " AND e.book = ?"
                params.append(book)
            if category:
                placeholders = ",".join("?" * len(category))
                sql += f" AND c.category IN ({placeholders})"
                params.extend(category)
            if include_types:
                placeholders = ",".join("?" * len(include_types))
                sql += f" AND c.type IN ({placeholders})"
                params.extend(include_types)
            if type_exclusions:
                placeholders = ",".join("?" * len(type_exclusions))
                sql += f" AND c.type NOT IN ({placeholders})"
                params.extend(type_exclusions)
            if doc_type:
                sql += " AND c.type = ?"
                params.append(doc_type)
        else:
            # No book_type filter - use simpler query, filter post-hoc if needed
            if book:
                # Already added above
                pass

        try:
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()

            if not rows:
                return self._search_fts(
                    query, top_k, doc_type, category, book_type, book,
                    None, include_types, type_exclusions,
                )

            # Vectorized dot product (embeddings are L2-normalized)
            source_ids = []
            chunk_texts = []
            books = []
            pages = []
            embeddings_list = []

            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                embeddings_list.append(emb)
                source_ids.append(row["source_id"])
                chunk_texts.append(row["chunk_text"])
                books.append(row["book"])
                pages.append(row["page_number"])

            embedding_matrix = np.stack(embeddings_list)
            # Dot product = cosine similarity for L2-normalized vectors
            similarities = embedding_matrix @ query_embedding

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Look up entity details from content table
            results = []
            for idx in top_indices:
                source_id = source_ids[idx]
                similarity = float(similarities[idx])

                # Look up the full entity
                entity_row = self.conn.execute(
                    "SELECT * FROM content WHERE id = ?", (source_id,)
                ).fetchone()

                if entity_row:
                    # Apply type/category filters that weren't applied in SQL
                    if not book_type:
                        if doc_type and entity_row["type"] != doc_type:
                            continue
                        if category and entity_row["category"] not in category:
                            continue
                        if include_types and entity_row["type"] not in include_types:
                            continue
                        if type_exclusions and entity_row["type"] in type_exclusions:
                            continue

                    metadata = json.loads(entity_row["metadata"]) if entity_row["metadata"] else {}
                    results.append(
                        {
                            "name": entity_row["name"],
                            "type": entity_row["type"],
                            "category": entity_row["category"],
                            "source": entity_row["book"],
                            "book": entity_row["book"],
                            "book_type": entity_row["book_type"],
                            "page": entity_row["page"],
                            "content": entity_row["content"],
                            "metadata": metadata,
                            "score": similarity,
                        }
                    )

            return results

        except Exception as e:
            print(f"Semantic search error: {e}, falling back to FTS")
            return self._search_fts(
                query, top_k, doc_type, category, book_type, book,
                None, include_types, type_exclusions,
            )

    def search_pages(
        self,
        query: str,
        top_k: int = 10,
        book: str = None,
        preprocess: bool = True,
    ) -> list[dict]:
        """
        Search full page text via pages/pages_fts tables.

        Returns list of dicts with book, page_number, chapter, snippet.
        """
        # Resolve fuzzy book name to exact DB name
        if book:
            resolved = self.resolve_book_name(book)
            if resolved:
                book = resolved

        original_query = query
        if preprocess:
            query = preprocess_query(query)
            if not query.strip():
                query = original_query

        # Expand with remaster aliases
        queries = set(expand_query_aliases(query))
        if original_query != query:
            queries.update(expand_query_aliases(original_query))
        queries = list(queries)

        all_results = {}

        for q in queries:
            sql = """
                SELECT p.book, p.page_number, p.chapter,
                       snippet(pages_fts, 2, '>>>', '<<<', '...', 40) as snippet,
                       bm25(pages_fts) as score
                FROM pages p
                JOIN pages_fts ON p.id = pages_fts.rowid
                WHERE pages_fts MATCH ?
            """
            params: list = [q]

            if book:
                sql += " AND p.book = ?"
                params.append(book)

            sql += " ORDER BY score LIMIT ?"
            params.append(top_k * 2)

            try:
                cursor = self.conn.execute(sql, params)
                for row in cursor:
                    key = (row["book"], row["page_number"])
                    score = -row["score"]
                    if key not in all_results or score > all_results[key]["score"]:
                        all_results[key] = {
                            "book": row["book"],
                            "page_number": row["page_number"],
                            "chapter": row["chapter"],
                            "snippet": row["snippet"],
                            "score": score,
                        }
            except sqlite3.OperationalError:
                safe_query = '"' + q.replace('"', '""') + '"'
                try:
                    params[0] = safe_query
                    cursor = self.conn.execute(sql, params)
                    for row in cursor:
                        key = (row["book"], row["page_number"])
                        score = -row["score"]
                        if key not in all_results or score > all_results[key]["score"]:
                            all_results[key] = {
                                "book": row["book"],
                                "page_number": row["page_number"],
                                "chapter": row["chapter"],
                                "snippet": row["snippet"],
                                "score": score,
                            }
                except Exception:
                    pass

        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

    def search_complex(self, query: str, top_k: int = 10, **kwargs) -> dict:
        """Handle complex questions by decomposing into sub-queries."""
        sub_queries = _decompose_complex_query(
            query,
            condition_names=self._condition_names,
            class_names=self._class_names,
        )

        if len(sub_queries) == 1 and sub_queries[0]["type_hint"] is None:
            results = self.search(query, top_k=top_k, **kwargs)
            return {
                "decomposition": sub_queries,
                "results": {query: results},
                "combined": results,
            }

        all_results = {}
        combined = {}

        for sq in sub_queries:
            sub_query = sq["query"]
            type_hint = sq["type_hint"]

            results = self.search(sub_query, top_k=top_k, doc_type=type_hint, **kwargs)
            if not results and type_hint:
                results = self.search(sub_query, top_k=top_k, **kwargs)

            all_results[sub_query] = results

            for r in results:
                key = (r["name"], r["source"])
                if key not in combined or r["score"] > combined[key]["score"]:
                    combined[key] = r

        combined_list = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        return {
            "decomposition": sub_queries,
            "results": all_results,
            "combined": combined_list[:top_k],
        }

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) as total FROM content")
        total_entities = cursor.fetchone()["total"]

        cursor = self.conn.execute("SELECT COUNT(*) as total FROM pages")
        total_pages = cursor.fetchone()["total"]

        cursor = self.conn.execute("SELECT COUNT(*) as total FROM embeddings")
        total_embeddings = cursor.fetchone()["total"]

        cursor = self.conn.execute("""
            SELECT category, COUNT(*) as count
            FROM content
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = {row["category"]: row["count"] for row in cursor}

        cursor = self.conn.execute("""
            SELECT book_type, COUNT(*) as count
            FROM content
            GROUP BY book_type
            ORDER BY count DESC
        """)
        by_book_type = {row["book_type"]: row["count"] for row in cursor}

        cursor = self.conn.execute("""
            SELECT book, COUNT(*) as count
            FROM content
            GROUP BY book
            ORDER BY count DESC
        """)
        by_book = {row["book"]: row["count"] for row in cursor}

        # Schema version
        schema_version = None
        try:
            cursor = self.conn.execute("SELECT version FROM schema_meta ORDER BY version DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                schema_version = row["version"]
        except Exception:
            pass

        return {
            "total_entities": total_entities,
            "total_pages": total_pages,
            "total_embeddings": total_embeddings,
            "by_category": by_category,
            "by_book_type": by_book_type,
            "by_book": by_book,
            "schema_version": schema_version,
        }

    # ------------------------------------------------------------------
    # Book name resolution
    # ------------------------------------------------------------------

    def resolve_book_name(self, query: str) -> str | None:
        """Resolve a user-friendly book name to the exact name in the database.

        Tries: exact match, suffix match (after ']'), contains match on
        book_summaries, then fallback to content table.
        """
        # 1. Exact match
        row = self.conn.execute(
            "SELECT book FROM book_summaries WHERE book = ?", (query,)
        ).fetchone()
        if row:
            return row["book"]
        # 2. Suffix match: "Player Core" matches "[PZO12001E] Player Core"
        row = self.conn.execute(
            "SELECT book FROM book_summaries WHERE book LIKE '%] ' || ?",
            (query,),
        ).fetchone()
        if row:
            return row["book"]
        # 3. Contains match (case-insensitive via LIKE)
        row = self.conn.execute(
            "SELECT book FROM book_summaries WHERE book LIKE ?",
            (f"%{query}%",),
        ).fetchone()
        if row:
            return row["book"]
        # 4. Fallback: try content table (covers books without summaries)
        row = self.conn.execute(
            "SELECT DISTINCT book FROM content WHERE book LIKE ?",
            (f"%{query}%",),
        ).fetchone()
        if row:
            return row["book"]
        return None

    # ------------------------------------------------------------------
    # Summary queries (v4 tables — graceful degradation for older DBs)
    # ------------------------------------------------------------------

    def get_book_summary(self, book: str) -> dict | None:
        """Get book-level summary.

        Returns dict with book, total_pages, chapter_count, summary, chapters
        or None if not found / table missing.
        """
        resolved = self.resolve_book_name(book)
        if resolved:
            book = resolved
        try:
            cursor = self.conn.execute(
                "SELECT * FROM book_summaries WHERE book = ?", (book,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("chapters"):
                result["chapters"] = json.loads(result["chapters"])
            return result
        except sqlite3.OperationalError:
            return None

    def list_books_with_summaries(self, book_type: str = None) -> list[dict]:
        """List all books that have book-level summaries.

        Args:
            book_type: Optional filter by book type (rulebook, bestiary, etc.)

        Returns list of dicts with book, total_pages, chapter_count, summary.
        """
        try:
            if book_type:
                cursor = self.conn.execute(
                    "SELECT book, total_pages, chapter_count, summary FROM book_summaries WHERE book_type = ? ORDER BY book",
                    (book_type,),
                )
            else:
                cursor = self.conn.execute(
                    "SELECT book, total_pages, chapter_count, summary FROM book_summaries ORDER BY book"
                )
            return [dict(row) for row in cursor]
        except sqlite3.OperationalError:
            return []

    def get_chapter_summary(self, book: str, chapter: str) -> dict | None:
        """Get chapter-level summary using LIKE match on chapter name.

        Returns dict with chapter, page_start, page_end, page_count, summary,
        keywords, entities or None if not found.
        """
        resolved = self.resolve_book_name(book)
        if resolved:
            book = resolved
        try:
            cursor = self.conn.execute(
                "SELECT * FROM chapter_summaries WHERE book = ? AND chapter LIKE ?",
                (book, f"%{chapter}%"),
            )
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            for field in ("keywords", "entities"):
                if result.get(field):
                    result[field] = json.loads(result[field])
            return result
        except sqlite3.OperationalError:
            return None

    def list_chapters(self, book: str) -> list[dict]:
        """List chapters with page ranges for a book.

        Returns list of dicts with chapter, page_start, page_end, page_count.
        """
        resolved = self.resolve_book_name(book)
        if resolved:
            book = resolved
        try:
            cursor = self.conn.execute(
                """SELECT chapter, page_start, page_end, page_count
                   FROM chapter_summaries
                   WHERE book = ?
                   ORDER BY page_start""",
                (book,),
            )
            return [dict(row) for row in cursor]
        except sqlite3.OperationalError:
            return []

    def get_page_summary(self, book: str, page_number: int) -> dict | None:
        """Get single page summary.

        Returns dict with page_number, chapter, page_type, summary, keywords,
        rules_referenced, entities_on_page, gm_notes or None.
        """
        resolved = self.resolve_book_name(book)
        if resolved:
            book = resolved
        try:
            cursor = self.conn.execute(
                "SELECT * FROM page_summaries WHERE book = ? AND page_number = ?",
                (book, page_number),
            )
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            for field in ("keywords", "rules_referenced", "entities_on_page", "gm_notes"):
                if result.get(field):
                    result[field] = json.loads(result[field])
            return result
        except sqlite3.OperationalError:
            return None

    def get_page_summaries_for_chapter(self, book: str, chapter: str) -> list[dict]:
        """Get all page summaries for a chapter.

        Returns list of dicts, ordered by page_number.
        """
        resolved = self.resolve_book_name(book)
        if resolved:
            book = resolved
        try:
            cursor = self.conn.execute(
                """SELECT * FROM page_summaries
                   WHERE book = ? AND chapter LIKE ?
                   ORDER BY page_number""",
                (book, f"%{chapter}%"),
            )
            results = []
            for row in cursor:
                d = dict(row)
                for field in ("keywords", "rules_referenced", "entities_on_page", "gm_notes"):
                    if d.get(field):
                        d[field] = json.loads(d[field])
                results.append(d)
            return results
        except sqlite3.OperationalError:
            return []

    def find_page_for_term(self, term: str, book: str = None) -> list[dict]:
        """Find page(s) where a term is defined or mentioned.

        Searches content table (entity names), page_summaries entities_on_page,
        and pages_fts. Returns deduplicated list of {name, type, book, page_number, source}.
        """
        if book:
            resolved = self.resolve_book_name(book)
            if resolved:
                book = resolved
        results_by_key: dict[tuple, dict] = {}

        # 1. Search content table for name matches
        try:
            sql = "SELECT name, type, book, page FROM content WHERE LOWER(name) LIKE ?"
            params: list = [f"%{term.lower()}%"]
            if book:
                sql += " AND book = ?"
                params.append(book)
            sql += " LIMIT 20"

            cursor = self.conn.execute(sql, params)
            for row in cursor:
                key = (row["book"], row["page"])
                if key not in results_by_key and row["page"] is not None:
                    results_by_key[key] = {
                        "name": row["name"],
                        "type": row["type"],
                        "book": row["book"],
                        "page_number": row["page"],
                        "source": "entity",
                    }
        except sqlite3.OperationalError:
            pass

        # 2. Search page_summaries.entities_on_page via JSON contains
        try:
            sql = """SELECT book, page_number, chapter, entities_on_page
                     FROM page_summaries
                     WHERE LOWER(entities_on_page) LIKE ?"""
            params = [f"%{term.lower()}%"]
            if book:
                sql += " AND book = ?"
                params.append(book)
            sql += " LIMIT 20"

            cursor = self.conn.execute(sql, params)
            for row in cursor:
                key = (row["book"], row["page_number"])
                if key not in results_by_key:
                    results_by_key[key] = {
                        "name": term,
                        "type": "page_reference",
                        "book": row["book"],
                        "page_number": row["page_number"],
                        "chapter": row["chapter"],
                        "source": "page_summary",
                    }
        except sqlite3.OperationalError:
            pass

        # 3. FTS search on pages for additional hits
        try:
            fts_query = f'"{term}"'
            sql = """SELECT p.book, p.page_number, p.chapter,
                            bm25(pages_fts) as score
                     FROM pages p
                     JOIN pages_fts ON p.id = pages_fts.rowid
                     WHERE pages_fts MATCH ?"""
            params = [fts_query]
            if book:
                sql += " AND p.book = ?"
                params.append(book)
            sql += " ORDER BY score LIMIT 10"

            cursor = self.conn.execute(sql, params)
            for row in cursor:
                key = (row["book"], row["page_number"])
                if key not in results_by_key:
                    results_by_key[key] = {
                        "name": term,
                        "type": "page_reference",
                        "book": row["book"],
                        "page_number": row["page_number"],
                        "chapter": row["chapter"],
                        "source": "page_fts",
                    }
        except sqlite3.OperationalError:
            pass

        # Sort: entities first, then by page number
        results = sorted(
            results_by_key.values(),
            key=lambda r: (0 if r["source"] == "entity" else 1, r["page_number"]),
        )
        return results

    def list_entities(
        self,
        book: str | None = None,
        book_type: str | None = None,
        category: str | list[str] | None = None,
        include_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """List entities by filters without requiring a search query.

        Unlike search(), this doesn't use FTS5 — it does a plain SQL query
        with WHERE filters. Use for bulk retrieval by book/type.

        Returns same dict structure as search() but without scores.
        """
        if book:
            resolved = self.resolve_book_name(book)
            if resolved:
                book = resolved

        if isinstance(category, str):
            category = [category]

        sql = "SELECT * FROM content WHERE 1=1"
        params: list = []

        if book:
            sql += " AND book = ?"
            params.append(book)

        if book_type:
            sql += " AND book_type = ?"
            params.append(book_type)

        if category:
            placeholders = ",".join("?" * len(category))
            sql += f" AND category IN ({placeholders})"
            params.extend(category)

        if include_types:
            placeholders = ",".join("?" * len(include_types))
            sql += f" AND type IN ({placeholders})"
            params.extend(include_types)

        if exclude_types:
            placeholders = ",".join("?" * len(exclude_types))
            sql += f" AND type NOT IN ({placeholders})"
            params.extend(exclude_types)

        sql += " ORDER BY name LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        results = []
        for row in cursor:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            results.append({
                "name": row["name"],
                "type": row["type"],
                "category": row["category"],
                "source": row["book"],
                "book": row["book"],
                "book_type": row["book_type"],
                "page": row["page"],
                "content": row["content"],
                "metadata": metadata,
            })
        return results

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
