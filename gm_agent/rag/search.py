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
        # Question-filler words — carry no domain meaning for rules lookup
        "work",
        "works",
        "working",
        "happen",
        "happens",
        "happened",
        "use",
        "used",
        "using",
        "mean",
        "means",
        "meaning",
        "called",
        "make",
        "makes",
        "making",
        "go",
        "goes",
        "going",
        "look",
        "thing",
        "things",
        "way",
        "let",
        "say",
        "also",
        "just",
        "much",
        "many",
        "some",
        "any",
        "other",
        "another",
        "same",
        "different",
        "each",
        "every",
        "all",
        "both",
        "well",
        "very",
        "really",
        "exactly",
        # Game term stop words - indicate type but shouldn't be required in results
        "spell",
        "spells",
        "feat",
        "feats",
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
        # Edition/meta words — describe query intent but never appear in entity content
        "remaster",
        "remastered",
        "remasters",
        "edition",
        "editions",
    }
)



# Lightweight stemming: known game-relevant suffixes.
# Maps inflected forms back to stems so FTS5 can match.
# Only covers forms that actually appear in our DB content.
_STEM_OVERRIDES: dict[str, str] = {
    # Game terms — keep as-is (they ARE the indexed form)
    "dying": "dying",
    "flanking": "flanking",
    "flying": "flying",
    # -ing stems that the heuristic gets wrong
    "casting": "cast",
    "falling": "falling",
    "calling": "call",
    "killing": "kill",
    "rolling": "roll",
    "pulling": "pull",
    "spelling": "spell",
    "stalling": "stall",
    "climbing": "climb",
    "tracking": "track",
    "shoving": "shove",
    "hiding": "hide",
    "riding": "ride",
    "striking": "strike",
    "slashing": "slash",
    "piercing": "piercing",  # game term (damage type)
    "bludgeoning": "bludgeoning",  # game term (damage type)
}


_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _strip_leading_articles(text: str) -> str:
    """Strip leading articles ('The', 'A', 'An') from a name or query.

    This ensures 'The Stag Lord' matches query 'Stag Lord' as an exact match.
    """
    return _LEADING_ARTICLE_RE.sub("", text).strip()


def _simple_stem(word: str) -> str:
    """Lightweight suffix stripping for FTS5 query expansion.

    Returns the stemmed form, or the original word if no rule applies.
    Not a full Porter stemmer — only handles common English inflections
    that cause FTS5 AND mismatches in our domain.
    """
    if word in _STEM_OVERRIDES:
        return _STEM_OVERRIDES[word]
    # -ing: grappling→grapple, swimming→swim, healing→heal
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        # doubled consonant at end: swimming→swimm→swim, running→runn→run
        if base.endswith(("mm", "nn", "tt", "dd", "pp", "bb", "gg", "ll", "rr", "ss")):
            return base[:-1]
        # consonant cluster likely from dropped 'e': grappl→grapple, castl→castle
        vowels = set("aeiou")
        if len(base) >= 2 and base[-1] not in vowels and base[-2] not in vowels:
            return base + "e"
        return base
    # -ed: happened→happen (but keep "undead", short words)
    if word.endswith("ed") and len(word) > 5:
        base = word[:-2]
        if base.endswith(("pp", "nn", "tt", "dd", "bb", "gg", "ll", "rr", "ss")):
            return base[:-1]
        return base
    # -s: happens→happen, works→work (but not "class", short words)
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def preprocess_query(query: str) -> str:
    """
    Preprocess natural language query into keywords for FTS5.

    Handles questions like "What is a goblin?" -> "goblin"
    """
    query = query.lower()
    # Remove punctuation (hyphens become spaces for FTS5 compatibility)
    query = re.sub(r"[^\w\s]", " ", query)
    words = query.split()
    keywords = [w for w in words if w not in STOP_WORDS and len(w) >= 2]
    if not keywords and words:
        keywords = [w for w in words if len(w) >= 3]
    if not keywords:
        return query.strip()
    return " ".join(keywords)


# --- Auto-detection of book/chapter references in query text ---

_CHAPTER_DETECT_RE = re.compile(r'\b(?:chapter|ch\.?)\s*(\d+)\b', re.IGNORECASE)
_BOOK_N_DETECT_RE = re.compile(r'\bbook\s+(\d+)\b', re.IGNORECASE)
_BOOK_N_SUFFIX_RE = re.compile(r'^[\s,]*book\s+(\d+)\b', re.IGNORECASE)


def detect_book_in_query(
    query: str,
    book_name_index: list[tuple[str, str | None]],
) -> tuple[str | None, str]:
    """Auto-detect a book name in query text and extract it.

    Args:
        query: The raw query string.
        book_name_index: List of (lowercase_name, full_db_name_or_None).
            Sorted by name length descending (longest match wins).
            None for full_db_name means this is a series prefix
            (caller should pass to resolve_book_names).

    Returns:
        (book_ref, cleaned_query) where book_ref is suitable for
        resolve_book_names(), or (None, original_query) if no book
        detected or book IS the entire query.
    """
    query_lower = query.lower()

    for name_lower, full_name in book_name_index:
        idx = query_lower.find(name_lower)
        if idx == -1:
            continue

        # Word boundary checks
        if idx > 0 and query_lower[idx - 1].isalnum():
            continue
        end_idx = idx + len(name_lower)
        if end_idx < len(query_lower) and query_lower[end_idx].isalnum():
            continue

        # Check if "Book N" immediately follows (for AP series)
        book_ref = name_lower
        remainder = query[end_idx:]
        book_n = _BOOK_N_SUFFIX_RE.match(remainder)
        if book_n:
            book_ref = name_lower + " book " + book_n.group(1)
            end_idx += book_n.end()

        # Strip the book reference from the query
        cleaned = (query[:idx] + query[end_idx:]).strip()
        cleaned = ' '.join(cleaned.split())

        if not cleaned:
            # Book name IS the entire query — don't auto-filter
            return None, query

        # Return resolved name or series prefix for resolve_book_names
        if full_name and not book_n:
            return full_name, cleaned
        return book_ref, cleaned

    return None, query


def detect_chapter_in_query(query: str) -> tuple[str | None, str]:
    """Detect and strip chapter/book-N references from query text.

    Returns:
        (chapter_name_or_None, cleaned_query).
        Always strips "Book N" noise even without a chapter match.
    """
    chapter = None
    cleaned = query

    # Detect "Chapter N" / "Ch N" / "Ch. N"
    m = _CHAPTER_DETECT_RE.search(cleaned)
    if m:
        chapter = f"Chapter {m.group(1)}"
        cleaned = cleaned[:m.start()] + cleaned[m.end():]

    # Strip standalone "Book N" references (noise for FTS)
    cleaned = _BOOK_N_DETECT_RE.sub('', cleaned)

    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())

    if chapter or cleaned != query.strip():
        return chapter, cleaned
    return None, query


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
# Creature family aliases loaded separately for substring-based expansion
# (e.g., "adult red dragon" → "adult cinder dragon")
CREATURE_FAMILY_ALIASES: dict[str, str] = {}
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
                if category == "creature_families":
                    for legacy, remaster in mappings.items():
                        if remaster:
                            CREATURE_FAMILY_ALIASES[legacy.lower()] = remaster.lower()
    except Exception as e:
        print(f"Warning: Could not load remaster aliases: {e}")


def _build_sorted_alias_index(
    aliases: dict[str, str],
) -> dict[str, str]:
    """Build a sorted-word index for order-insensitive alias matching.

    E.g., "attack of opportunity" → sorted key "attack of opportunity"
    and "opportunity attack" would be normalized to "attack opportunity"
    which doesn't match. Instead we store frozenset-based keys.
    """
    index: dict[str, str] = {}
    for legacy, remaster in aliases.items():
        words = legacy.split()
        if len(words) >= 2:
            sorted_key = " ".join(sorted(words))
            index[sorted_key] = remaster
    return index


# Pre-build sorted alias index for order-insensitive matching
_SORTED_ALIAS_INDEX = _build_sorted_alias_index(REMASTER_ALIASES)


def expand_query_aliases(query: str) -> list[str]:
    """Expand query with remaster aliases. Returns list of queries to try."""
    query_lower = query.lower()
    queries = [query]

    # Exact match
    if query_lower in REMASTER_ALIASES:
        queries.append(REMASTER_ALIASES[query_lower])

    # Reverse match (remaster → legacy)
    for legacy, remaster in REMASTER_ALIASES.items():
        if query_lower == remaster and legacy not in queries:
            queries.append(legacy)

    # Order-insensitive match for multi-word queries
    # e.g., "opportunity attack" matches alias "attack of opportunity"
    query_words = query_lower.split()
    if len(query_words) >= 2:
        sorted_query = " ".join(sorted(query_words))
        if sorted_query in _SORTED_ALIAS_INDEX:
            alias_target = _SORTED_ALIAS_INDEX[sorted_query]
            if alias_target not in queries:
                queries.append(alias_target)
        # Also check with stop words stripped from alias keys
        query_content_words = sorted(w for w in query_words if w not in {"of", "the", "a", "an"})
        sorted_content = " ".join(query_content_words)
        for sorted_key, target in _SORTED_ALIAS_INDEX.items():
            key_content_words = " ".join(sorted(w for w in sorted_key.split() if w not in {"of", "the", "a", "an"}))
            if sorted_content == key_content_words and target not in queries:
                queries.append(target)

    # Per-word expansion for multi-word queries
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

    # Substring-based creature family expansion for compound queries.
    # "adult red dragon" → also search "adult cinder dragon".
    # Sorted longest-first so "copper dragon" matches before "dragon".
    if CREATURE_FAMILY_ALIASES:
        for legacy, remaster in sorted(
            CREATURE_FAMILY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if legacy in query_lower:
                replaced = query_lower.replace(legacy, remaster)
                if replaced not in queries:
                    queries.append(replaced)
            # Reverse: "adult cinder dragon" → also search "adult red dragon"
            if remaster in query_lower:
                replaced = query_lower.replace(remaster, legacy)
                if replaced not in queries:
                    queries.append(replaced)

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


def _infer_entity_types(
    query: str,
    creature_family_names: set[str] | None = None,
) -> list[str] | None:
    """Auto-detect include_types for entity-list queries.

    Returns include_types list or None if no inference possible.
    """
    query_lower = query.lower()

    # Detect list-building intent
    list_patterns = [
        "types of", "kinds of", "variants of", "list of",
        "different types", "all the", "what are the",
    ]
    if not any(p in query_lower for p in list_patterns):
        return None

    # Creature-related keywords
    creature_keywords = {
        "dragon", "undead", "beast", "elemental", "demon", "devil",
        "giant", "fey", "construct", "aberration", "fiend", "celestial",
        "humanoid", "animal", "ooze", "plant", "fungus", "monitor",
    }
    if (any(k in query_lower for k in creature_keywords)
            or (creature_family_names
                and any(name in query_lower for name in creature_family_names))):
        return ["creature", "creature_family", "creature_template"]

    return None


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
        # GM content — lower than core entities to avoid reference rules
        # outranking the actual things GMs search for
        "game_mechanic": 10,
        "guidance": 12,
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

    # Class-level default — overridden in _load_creation_tables() after DB init.
    # Ensures get_creation_table() works even when __init__ is bypassed (e.g., mocks).
    _creation_tables: dict = {}

    def __init__(self, db_path: str = "pathfinder_search.db"):
        self.db_path = db_path
        self.conn = None
        self._open_readonly()
        self._condition_names, self._class_names = self._load_term_sets()
        self._creature_family_names = self._load_creature_family_names()
        self._book_name_index = self._build_book_name_index()
        # Load aliases and creation tables from DB (v7+); graceful no-op on older DB
        self._load_aliases_from_db()
        self._load_creation_tables()

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

    def _load_creature_family_names(self) -> set[str]:
        """Load creature family names from the DB for entity-list type inference."""
        try:
            cursor = self.conn.execute(
                "SELECT DISTINCT LOWER(name) FROM content WHERE type = 'creature_family'"
            )
            return {row[0] for row in cursor}
        except Exception:
            return set()

    def _build_book_name_index(self) -> list[tuple[str, str | None]]:
        """Build index of book names for auto-detection in queries.

        Returns list of (lowercase_name, full_db_name_or_None) sorted by
        name length descending (longest match wins).
        None for full_db_name means this is a series prefix (resolve dynamically).
        """
        index: list[tuple[str, str | None]] = []
        try:
            rows = self.conn.execute(
                "SELECT book FROM book_summaries ORDER BY book"
            ).fetchall()
            seen: set[str] = set()

            for row in rows:
                full = row["book"]
                lower = full.lower()

                # Add full book name (2+ words to avoid false positives)
                if lower not in seen and len(full.split()) >= 2:
                    index.append((lower, full))
                    seen.add(lower)

                # Extract AP series name from "X N of M" pattern
                m = re.match(r'^(.+?)\s+\d+\s+of\s+\d+', full)
                if m:
                    series = m.group(1).strip()
                    series_lower = series.lower()
                    if series_lower not in seen and len(series.split()) >= 2:
                        index.append((series_lower, None))
                        seen.add(series_lower)

                # Add single-word book names (e.g., "Kingmaker", "Gatewalkers")
                if len(full.split()) == 1 and lower not in seen:
                    index.append((lower, full))
                    seen.add(lower)

            # Sort by length descending: longer names match first
            index.sort(key=lambda x: len(x[0]), reverse=True)
        except Exception:
            pass
        return index

    def _load_aliases_from_db(self) -> None:
        """Load remaster aliases from term_aliases table (v7+) into REMASTER_ALIASES
        and CREATURE_FAMILY_ALIASES.

        Gracefully no-ops if the table doesn't exist (v6 or older DB).
        Updates module-level dicts so all module-level functions
        (expand_query_aliases, etc.) pick up the DB-loaded aliases.
        """
        try:
            cursor = self.conn.execute(
                "SELECT legacy_term, remaster_term, category FROM term_aliases"
                " WHERE alias_type = 'remaster'"
            )
            rows = list(cursor)
            if not rows:
                return
            for legacy, remaster, category in rows:
                REMASTER_ALIASES[legacy] = remaster
                if category == "creature_families":
                    CREATURE_FAMILY_ALIASES[legacy] = remaster
            # Rebuild sorted alias index after loading new aliases
            global _SORTED_ALIAS_INDEX
            _SORTED_ALIAS_INDEX = _build_sorted_alias_index(REMASTER_ALIASES)
        except sqlite3.OperationalError:
            pass  # term_aliases table absent in v6 DB — use file-loaded aliases

    def _load_creation_tables(self) -> None:
        """Load PF2e creation tables from DB (v7+) into instance dicts.

        Gracefully no-ops if no pf2e_creation_table rows exist.
        Tables keyed by their 'table_key' metadata field:
          creature_stats, hazard_stats, elite_hp, role_adjustments,
          creature_xp, simple_hazard_xp, threat_thresholds, xp_per_extra_player, treasure
        """
        self._creation_tables: dict[str, dict] = {}
        try:
            cursor = self.conn.execute(
                "SELECT metadata FROM content WHERE type = 'pf2e_creation_table'"
            )
            for row in cursor:
                if not row[0]:
                    continue
                try:
                    meta = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    continue
                table_key = meta.get("table_key")
                if not table_key:
                    continue
                # Merge level-keyed rows into a single dict per table_key
                if table_key == "creature_stats":
                    level = meta.get("level")
                    if level is not None:
                        self._creation_tables.setdefault("creature_stats", {})[level] = meta.get("stats", {})
                elif table_key == "hazard_stats":
                    level = meta.get("level")
                    if level is not None:
                        self._creation_tables.setdefault("hazard_stats", {})[level] = meta.get("stats", {})
                elif table_key == "elite_hp":
                    # Convert string keys back to int for compatibility
                    raw = meta.get("adjustments", {})
                    self._creation_tables["elite_hp"] = {int(k): v for k, v in raw.items()}
                elif table_key == "role_adjustments":
                    self._creation_tables["role_adjustments"] = meta.get("adjustments", {})
                elif table_key == "creature_xp":
                    raw = meta.get("xp_by_diff", {})
                    self._creation_tables["creature_xp"] = {int(k): v for k, v in raw.items()}
                elif table_key == "simple_hazard_xp":
                    raw = meta.get("xp_by_diff", {})
                    self._creation_tables["simple_hazard_xp"] = {int(k): v for k, v in raw.items()}
                elif table_key == "threat_thresholds":
                    raw = meta.get("thresholds", {})
                    self._creation_tables["threat_thresholds"] = {k: tuple(v) for k, v in raw.items()}
                elif table_key == "xp_per_extra_player":
                    self._creation_tables["xp_per_extra_player"] = meta.get("xp_by_threat", {})
                elif table_key == "treasure":
                    raw = meta.get("by_level", {})
                    self._creation_tables["treasure"] = {int(k): v for k, v in raw.items()}
        except sqlite3.OperationalError:
            pass  # DB has no pf2e_creation_table rows — fallback to hardcoded constants

    def get_creation_table(self, key: str) -> dict:
        """Return a DB-loaded creation table dict, or empty dict if not available.

        Args:
            key: Table key (e.g., 'creature_stats', 'hazard_stats', 'elite_hp',
                 'role_adjustments', 'creature_xp', 'threat_thresholds',
                 'xp_per_extra_player', 'treasure').

        Returns:
            Dict of table data, or empty dict (fallback to module-level constant).
        """
        return self._creation_tables.get(key, {})

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
        is_remaster: bool | None = None,
        # Metadata filters
        level: int | None = None,
        level_range: tuple[int, int] | None = None,
        traits: list[str] | None = None,
        # Chapter scoping
        chapter: str | None = None,
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
            level: Filter to entities at this exact level (creatures, hazards)
            level_range: Filter to level range (inclusive), e.g. (3, 7)
            traits: Filter to entities with ALL of these traits
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

        # Auto-detect book name in query text when not explicitly provided
        if not book and not source and preprocess:
            auto_book, cleaned = detect_book_in_query(query, self._book_name_index)
            if auto_book:
                book = auto_book
                if cleaned.strip():
                    query = cleaned

        # Auto-detect chapter/book-N references in query text
        if not chapter and preprocess:
            auto_chapter, cleaned = detect_chapter_in_query(query)
            if auto_chapter:
                chapter = auto_chapter
                if cleaned.strip():
                    query = cleaned
            elif cleaned != query:
                # "Book N" was stripped even without chapter detection
                if cleaned.strip():
                    query = cleaned

        # Resolve fuzzy book name(s) to exact DB names
        books = None
        if book:
            books = self.resolve_book_names(book)
            if not books:
                books = [book]  # Fallback: pass raw name

        # Resolve chapter to page range
        chapter_page_range: tuple[int, int] | None = None
        if chapter and books:
            ch_summary = self.get_chapter_summary(books[0], chapter)
            if ch_summary:
                chapter_page_range = (ch_summary["page_start"], ch_summary["page_end"])

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

        # Auto-detect entity types for list queries (e.g., "types of dragon")
        if not include_types and not doc_type and not category:
            inferred = _infer_entity_types(original_query, self._creature_family_names)
            if inferred:
                include_types = inferred

        if use_semantic:
            results = self._search_semantic(
                original_query,
                top_k * 2,
                doc_type,
                category,
                book_type,
                books,
                include_types,
                type_exclusions,
                level=level,
                level_range=level_range,
                traits=traits,
                is_remaster=is_remaster,
                chapter_page_range=chapter_page_range,
            )
        else:
            results = self._search_fts(
                query,
                top_k * 2,
                doc_type,
                category,
                book_type,
                books,
                original_query,
                include_types,
                type_exclusions,
                level=level,
                level_range=level_range,
                traits=traits,
                is_remaster=is_remaster,
                chapter_page_range=chapter_page_range,
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
        books: list[str] | None = None,
        original_query: str = None,
        include_types: list = None,
        type_exclusions: set = None,
        level: int | None = None,
        level_range: tuple[int, int] | None = None,
        traits: list[str] | None = None,
        is_remaster: bool | None = None,
        chapter_page_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Full-text search using FTS5 with exact match boosting and alias expansion."""
        boost_query = original_query or query

        # Expand with remaster aliases
        queries = set(expand_query_aliases(query))
        if original_query and original_query != query:
            queries.update(expand_query_aliases(original_query))

        # Add stemmed variants (both single and multi-word)
        query_words = query.split()
        stemmed = [_simple_stem(w) for w in query_words]
        stemmed_query = " ".join(stemmed)
        if stemmed_query != query:
            queries.add(stemmed_query)
            queries.update(expand_query_aliases(stemmed_query))

        queries = list(queries)

        all_results = {}
        search_kwargs = dict(
            doc_type=doc_type, category=category, book_type=book_type,
            books=books, include_types=include_types,
            type_exclusions=type_exclusions,
            level=level, level_range=level_range, traits=traits,
            is_remaster=is_remaster,
            chapter_page_range=chapter_page_range,
        )

        for q in queries:
            results = self._search_fts_single(
                q, top_k * 2, **search_kwargs,
            )
            for r in results:
                key = (r["name"], r["source"])
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # OR fallback: if AND query returned few results on a multi-word query,
        # try each word individually to broaden recall.
        # Uses a significant penalty (-30) so single-word matches don't
        # outrank good multi-word alias matches (e.g., "holy" rune vs
        # "Chalice of Justice" found via alias expansion).
        if len(all_results) < 3 and len(query_words) >= 2:
            for word in query_words:
                stemmed_word = _simple_stem(word)
                for w in {word, stemmed_word}:
                    if len(w) < 3:
                        continue
                    word_results = self._search_fts_single(
                        w, top_k, **search_kwargs,
                    )
                    for r in word_results:
                        key = (r["name"], r["source"])
                        r["score"] = r["score"] - 30
                        if key not in all_results or r["score"] > all_results[key]["score"]:
                            all_results[key] = r

        # Boost conditions/rules when query contains known terms.
        # Also check alias-expanded forms (e.g., "flanking" aliases to
        # "off-guard" which IS a condition name).
        if doc_type is None and self._condition_names:
            query_lower = boost_query.lower()
            # Build expanded term set: original query words + alias targets
            check_terms = set(query_lower.split())
            for word in list(check_terms):
                if word in REMASTER_ALIASES:
                    check_terms.add(REMASTER_ALIASES[word])
                for legacy, remaster in REMASTER_ALIASES.items():
                    if word == remaster:
                        check_terms.add(legacy)

            for term in self._condition_names:
                if term in query_lower or term in check_terms:
                    term_results = self._search_fts_single(
                        term, 5, doc_type=None, category=category,
                        book_type=book_type, books=books,
                        include_types=include_types, type_exclusions=type_exclusions,
                        level=level, level_range=level_range, traits=traits,
                        is_remaster=is_remaster,
                    )
                    for r in term_results:
                        if r["type"] in ("condition", "rule", "trait"):
                            key = (r["name"], r["source"])
                            r["score"] = r["score"] + 50
                            if key not in all_results or r["score"] > all_results[key]["score"]:
                                all_results[key] = r

        # Boost results from a book whose title matches the query.
        # E.g., searching "Absalom" boosts entries from [PZO9304E] Absalom.
        if doc_type is None and not books:
            resolved_books = self.resolve_book_names(boost_query)
            if resolved_books:
                book_set = set(resolved_books)
                for r in all_results.values():
                    if r.get("book") in book_set:
                        r["score"] += 30
                        r["book_title_boost"] = 30

        # Apply book_type boost before dedup so authoritative sources
        # (rulebooks) win when multiple entries share the same name.
        # Also boost remaster content over legacy so Monster Core beats
        # Bestiary for duplicate names.
        for r in all_results.values():
            boost = self.BOOK_TYPE_BOOST.get(r.get("book_type", ""), 0)
            if r.get("is_remaster"):
                boost += 8  # Remaster content boost
            if r.get("edition") == "pf1e":
                boost -= 15  # PF1E lore is tertiary; PF2E content supersedes it
            r["score"] += boost
            r["book_type_boost"] = boost

        # Boost merged/consolidated entries — they contain rich, authoritative
        # content synthesized from all books. Without this, BM25 penalizes
        # their length and thin per-book entries win the name dedup.
        for r in all_results.values():
            meta = r.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta) if meta else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            if meta.get("merged") and len(r.get("content", "")) > 200:
                r["score"] += 15
                r["merged_boost"] = 15

        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        return sorted_results[:top_k]

    def _search_fts_single(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        category: list[str] = None,
        book_type: list[str] = None,
        books: list[str] | None = None,
        include_types: list = None,
        type_exclusions: set = None,
        level: int | None = None,
        level_range: tuple[int, int] | None = None,
        traits: list[str] | None = None,
        is_remaster: bool | None = None,
        chapter_page_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Single FTS5 search without alias expansion."""
        # content has TEXT PK (id), FTS5 uses implicit integer rowid
        # Score: BM25 (negative = better) minus name-match boosts (exact=100, partial=30)
        # For exact match, also compare article-stripped versions so
        # "The Stag Lord" matches query "Stag Lord" as exact (+100).
        stripped_query = _strip_leading_articles(query)
        sql = """
            SELECT c.*,
                   bm25(content_fts)
                   - (CASE WHEN LOWER(c.name) = LOWER(?) THEN 100
                           WHEN LOWER(TRIM(
                                CASE WHEN LOWER(c.name) LIKE 'the %' THEN SUBSTR(c.name, 5)
                                     WHEN LOWER(c.name) LIKE 'a %' THEN SUBSTR(c.name, 3)
                                     WHEN LOWER(c.name) LIKE 'an %' THEN SUBSTR(c.name, 4)
                                     ELSE c.name END
                           )) = LOWER(?) THEN 100
                           WHEN LOWER(c.name) LIKE '%' || LOWER(?) || '%' THEN 30
                           ELSE 0 END) as score
            FROM content c
            JOIN content_fts ON c.rowid = content_fts.rowid
            WHERE content_fts MATCH ?
        """
        params: list = [query, stripped_query, query, query]

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

        if books:
            placeholders = ",".join("?" * len(books))
            sql += f" AND c.book IN ({placeholders})"
            params.extend(books)

        if include_types:
            placeholders = ",".join("?" * len(include_types))
            sql += f" AND c.type IN ({placeholders})"
            params.extend(include_types)

        if type_exclusions:
            placeholders = ",".join("?" * len(type_exclusions))
            sql += f" AND c.type NOT IN ({placeholders})"
            params.extend(type_exclusions)

        if is_remaster is not None:
            sql += " AND c.is_remaster = ?"
            params.append(1 if is_remaster else 0)

        # Metadata filters
        if level is not None:
            sql += " AND CAST(json_extract(c.metadata, '$.level') AS INTEGER) = ?"
            params.append(level)

        if level_range is not None:
            sql += " AND CAST(json_extract(c.metadata, '$.level') AS INTEGER) BETWEEN ? AND ?"
            params.extend([level_range[0], level_range[1]])

        if traits:
            for trait in traits:
                sql += ' AND LOWER(c.metadata) LIKE ?'
                params.append(f'%"{trait.lower()}"%')

        if chapter_page_range is not None:
            sql += " AND c.page BETWEEN ? AND ?"
            params.extend([chapter_page_range[0], chapter_page_range[1]])

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        try:
            cursor = self.conn.execute(sql, params)
            return [self._row_to_result(row) for row in cursor]
        except sqlite3.OperationalError:
            # FTS5 query syntax error - try quoted query
            safe_query = '"' + query.replace('"', '""') + '"'
            params[3] = safe_query
            try:
                cursor = self.conn.execute(sql, params)
                return [self._row_to_result(row) for row in cursor]
            except Exception:
                return []

    def _row_to_result(self, row: sqlite3.Row) -> dict:
        """Convert a content table row to a result dict."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        result = {
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
        # is_remaster column (schema v6+) — graceful fallback for older DBs
        try:
            result["is_remaster"] = bool(row["is_remaster"])
        except (IndexError, KeyError):
            result["is_remaster"] = False
        # edition column (schema v10+) — graceful fallback for older DBs
        try:
            result["edition"] = row["edition"] or "pf2e"
        except (IndexError, KeyError):
            result["edition"] = "pf2e_remaster" if result["is_remaster"] else "pf2e"
        return result

    def _search_semantic(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        category: list[str] = None,
        book_type: list[str] = None,
        books: list[str] | None = None,
        include_types: list = None,
        type_exclusions: set = None,
        level: int | None = None,
        level_range: tuple[int, int] | None = None,
        traits: list[str] | None = None,
        is_remaster: bool | None = None,
        chapter_page_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Semantic search using pre-computed embeddings with numpy vectorized dot product."""
        query_embedding = get_embedding(query)
        if query_embedding is None:
            print("Failed to get query embedding, falling back to FTS")
            return self._search_fts(
                query, top_k, doc_type, category, book_type, books,
                None, include_types, type_exclusions,
                level=level, level_range=level_range, traits=traits,
            )

        # Build filter SQL for embeddings table
        sql = """
            SELECT e.source_id, e.chunk_text, e.embedding, e.book, e.page_number, e.source_type
            FROM embeddings e
            WHERE e.source_type = 'entity'
        """
        params = []

        if books:
            placeholders = ",".join("?" * len(books))
            sql += f" AND e.book IN ({placeholders})"
            params.extend(books)

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
            if books:
                bk_placeholders = ",".join("?" * len(books))
                sql += f" AND e.book IN ({bk_placeholders})"
                params.extend(books)
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
            if books:
                # Already added above
                pass

        try:
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()

            if not rows:
                return self._search_fts(
                    query, top_k, doc_type, category, book_type, books,
                    None, include_types, type_exclusions,
                    level=level, level_range=level_range, traits=traits,
                )

            # Vectorized dot product (embeddings are L2-normalized)
            source_ids = []
            embeddings_list = []

            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                embeddings_list.append(emb)
                source_ids.append(row["source_id"])

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

                    # Apply metadata filters
                    if level is not None:
                        entity_level = metadata.get("level")
                        if entity_level is None or int(entity_level) != level:
                            continue
                    if level_range is not None:
                        entity_level = metadata.get("level")
                        if entity_level is None or not (level_range[0] <= int(entity_level) <= level_range[1]):
                            continue
                    if traits:
                        entity_traits = [t.lower() for t in metadata.get("traits", [])]
                        if not all(t.lower() in entity_traits for t in traits):
                            continue
                    if chapter_page_range is not None:
                        page = entity_row["page"]
                        if page is None or not (chapter_page_range[0] <= page <= chapter_page_range[1]):
                            continue

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
                query, top_k, doc_type, category, book_type, books,
                None, include_types, type_exclusions,
                level=level, level_range=level_range, traits=traits,
            )

    def search_pages(
        self,
        query: str,
        top_k: int = 10,
        book: str = None,
        preprocess: bool = True,
        chapter: str | None = None,
    ) -> list[dict]:
        """
        Search full page text via pages/pages_fts tables.

        Returns list of dicts with book, page_number, chapter, snippet.
        """
        # Auto-detect book/chapter from query text when not explicitly set
        if not book and preprocess:
            auto_book, cleaned = detect_book_in_query(query, self._book_name_index)
            if auto_book:
                book = auto_book
                if cleaned.strip():
                    query = cleaned
        if not chapter and preprocess:
            auto_chapter, cleaned = detect_chapter_in_query(query)
            if auto_chapter:
                chapter = auto_chapter
                if cleaned.strip():
                    query = cleaned
            elif cleaned != query and cleaned.strip():
                query = cleaned

        # Resolve fuzzy book name(s) to exact DB names
        books = None
        if book:
            books = self.resolve_book_names(book)
            if not books:
                books = [book]

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
            self._search_pages_fts(q, books, chapter, top_k * 2, all_results)

        # OR fallback: if AND query returned no results on a multi-word query,
        # try each word individually to broaden recall (same pattern as _search_fts).
        if not all_results and len(original_query.split()) >= 2:
            words = [w for w in original_query.split() if len(w) >= 3]
            for word in words:
                for w in {word, _simple_stem(word)}:
                    if len(w) < 3:
                        continue
                    self._search_pages_fts(
                        w, books, chapter, top_k, all_results,
                        score_penalty=30,
                    )

        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

    def _search_pages_fts(
        self, query: str, books: list[str] | None, chapter: str | None,
        limit: int, results: dict, score_penalty: float = 0,
    ) -> None:
        """Execute a single pages_fts query, merging results into *results* dict."""
        sql = """
            SELECT p.book, p.page_number, p.chapter,
                   snippet(pages_fts, 2, '>>>', '<<<', '...', 40) as snippet,
                   bm25(pages_fts) as score
            FROM pages p
            JOIN pages_fts ON p.id = pages_fts.rowid
            WHERE pages_fts MATCH ?
        """
        params: list = [query]

        if books:
            placeholders = ",".join("?" * len(books))
            sql += f" AND p.book IN ({placeholders})"
            params.extend(books)

        if chapter:
            sql += " AND p.chapter LIKE ?"
            params.append(f"%{chapter}%")

        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        try:
            cursor = self.conn.execute(sql, params)
            for row in cursor:
                key = (row["book"], row["page_number"])
                score = -row["score"] - score_penalty
                if key not in results or score > results[key]["score"]:
                    results[key] = {
                        "book": row["book"],
                        "page_number": row["page_number"],
                        "chapter": row["chapter"],
                        "snippet": row["snippet"],
                        "score": score,
                    }
        except sqlite3.OperationalError:
            safe_query = '"' + query.replace('"', '""') + '"'
            try:
                params[0] = safe_query
                cursor = self.conn.execute(sql, params)
                for row in cursor:
                    key = (row["book"], row["page_number"])
                    score = -row["score"] - score_penalty
                    if key not in results or score > results[key]["score"]:
                        results[key] = {
                            "book": row["book"],
                            "page_number": row["page_number"],
                            "chapter": row["chapter"],
                            "snippet": row["snippet"],
                            "score": score,
                        }
            except Exception:
                pass

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

        Tries: exact match, suffix match (after ']'), scored contains match
        on book_summaries, then fallback to content table.

        Contains matches are scored by how prominent the query is in the
        book name: setting/rulebook books are preferred over adventures,
        and books where the query forms a larger fraction of the name rank
        higher.  This prevents "Absalom" from matching "Little Trouble in
        Big Absalom" when "Lost Omens Absalom, City of Lost Omens" exists.
        """
        # Normalize whitespace — LLMs sometimes emit non-breaking spaces (\xa0)
        query = " ".join(query.split())
        # Normalize colons used as subtitle separators
        # e.g. "Lost Omens: The Mwangi Expanse" → "Lost Omens The Mwangi Expanse"
        if ":" in query:
            query = " ".join(query.replace(":", " ").split())
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
        # 3. Scored contains match — prefer the most relevant book
        rows = self.conn.execute(
            "SELECT book, book_type FROM book_summaries WHERE book LIKE ?",
            (f"%{query}%",),
        ).fetchall()
        if rows:
            return self._best_book_match(query, rows)
        # 4. Fallback: try content table (covers books without summaries)
        rows = self.conn.execute(
            "SELECT DISTINCT book FROM content WHERE book LIKE ?",
            (f"%{query}%",),
        ).fetchall()
        if rows:
            # No book_type available — just use query prominence
            return self._best_book_match(query, rows)
        return None

    _VOL_N_OF_M_RE = re.compile(r'\b(\d+)\s+of\s+\d+\b', re.IGNORECASE)

    @staticmethod
    def _best_book_match(
        query: str, rows: list[sqlite3.Row]
    ) -> str:
        """Pick the best book match from multiple LIKE results.

        Scoring: book_type preference + query prominence in book name.

        Special case: when all matching books are volumes of the same AP series
        (all share the query as a prefix and use "N of M" numbering), prefer
        the earliest volume rather than the shortest-named one.  This prevents
        "Season of Ghosts" from resolving to Book 3 just because it has a
        shorter title than Book 1.
        """
        _TYPE_SCORE = {
            "setting": 3,
            "rulebook": 2,
            "bestiary": 1,
            "adventure": 0,
            "players_guide": 0,
        }
        q_lower = query.lower()
        q_len = len(q_lower)

        # Series-volume detection: all books share the query as a common prefix
        # AND at least one uses "N of M" volume numbering.
        if len(rows) > 1:
            all_prefixed = all(row["book"].lower().startswith(q_lower) for row in rows)
            has_vol_numbering = any(
                PathfinderSearch._VOL_N_OF_M_RE.search(row["book"]) for row in rows
            )
            if all_prefixed and has_vol_numbering:
                def _vol_num(book: str) -> int:
                    m = PathfinderSearch._VOL_N_OF_M_RE.search(book)
                    return int(m.group(1)) if m else 999
                return min((row["book"] for row in rows), key=_vol_num)

        best_book = rows[0]["book"]
        best_score = -1.0

        for row in rows:
            book = row["book"]
            book_type = row["book_type"] if "book_type" in row.keys() else ""
            # Prominence: query length / book name length
            prominence = q_len / max(len(book), 1)
            # Book type bonus
            type_bonus = _TYPE_SCORE.get(book_type, 0) * 0.1
            score = prominence + type_bonus
            if score > best_score:
                best_score = score
                best_book = book

        return best_book

    _BOOK_N_RE = re.compile(r'^(.+?)\s+book\s+(\d+)$', re.IGNORECASE)

    def resolve_book_names(self, query: str) -> list[str]:
        """Resolve a book query to one or more exact DB names.

        Like resolve_book_name() but returns ALL matching books for AP series.
        "Season of Ghosts" -> all 5 SoG books.
        "Season of Ghosts Book 1" -> just the 1st book.
        "Player Core" -> single-element list.
        Returns empty list if nothing matched.
        """
        # Handle "X Book N" pattern -> "X N of"
        m = self._BOOK_N_RE.match(query)
        if m:
            series, num = m.group(1), m.group(2)
            pattern = f"{num} of"
            rows = self.conn.execute(
                "SELECT book FROM book_summaries WHERE book LIKE ? AND book LIKE ?",
                (f"%{series}%", f"%{pattern}%"),
            ).fetchall()
            if rows:
                return [r["book"] for r in rows]
            # Fall through to normal resolution

        # Try single exact/suffix resolution first
        single = self.resolve_book_name(query)
        if single:
            # Check if this is an AP series match (multiple books contain query).
            # Query both book_summaries (main volumes) and content (catches player
            # guides and other companion books not in book_summaries).
            rows = self.conn.execute(
                "SELECT book FROM book_summaries WHERE book LIKE ?",
                (f"%{query}%",),
            ).fetchall()
            # If multiple books match AND the query is substantially shorter
            # than the resolved name, it's likely a series prefix.
            is_series_prefix = len(query) < len(single) * 0.8
            if len(rows) > 1 and is_series_prefix:
                matched = [r["book"] for r in rows]
                # Also include companion books (player guides, etc.) in content
                # that match the series prefix but aren't in book_summaries.
                extra = self.conn.execute(
                    "SELECT DISTINCT book FROM content WHERE book LIKE ? AND book NOT IN ("
                    + ",".join("?" * len(matched))
                    + ")",
                    (f"%{query}%", *matched),
                ).fetchall()
                return matched + [r["book"] for r in extra]
            if not rows and is_series_prefix:
                # book_summaries has no entries for this series (e.g. Season of Ghosts).
                # Fall back to content table for series expansion.
                all_books = self.conn.execute(
                    "SELECT DISTINCT book FROM content WHERE book LIKE ?",
                    (f"%{query}%",),
                ).fetchall()
                if len(all_books) > 1:
                    return [r["book"] for r in all_books]
            return [single]

        return []

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
            for field in ("keywords", "entities", "open_threads"):
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        result[field] = []
            return result
        except sqlite3.OperationalError:
            return None

    def list_chapters(self, book: str) -> list[dict]:
        """List chapters with page ranges for a book.

        Returns list of dicts with chapter, page_start, page_end, page_count, open_threads.
        """
        resolved = self.resolve_book_name(book)
        if resolved:
            book = resolved
        try:
            cursor = self.conn.execute(
                """SELECT chapter, page_start, page_end, page_count, open_threads, summary
                   FROM chapter_summaries
                   WHERE book = ?
                   ORDER BY page_start""",
                (book,),
            )
            rows = []
            for row in cursor:
                d = dict(row)
                if d.get("open_threads"):
                    try:
                        d["open_threads"] = json.loads(d["open_threads"])
                    except (json.JSONDecodeError, TypeError):
                        d["open_threads"] = []
                rows.append(d)
            return rows
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
        books = None
        if book:
            books = self.resolve_book_names(book)
            if not books:
                books = [book]
        results_by_key: dict[tuple, dict] = {}

        # 1. Search content table for name matches
        try:
            sql = "SELECT name, type, book, page FROM content WHERE LOWER(name) LIKE ?"
            params: list = [f"%{term.lower()}%"]
            if books:
                placeholders = ",".join("?" * len(books))
                sql += f" AND book IN ({placeholders})"
                params.extend(books)
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
            if books:
                placeholders = ",".join("?" * len(books))
                sql += f" AND book IN ({placeholders})"
                params.extend(books)
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
            if books:
                placeholders = ",".join("?" * len(books))
                sql += f" AND p.book IN ({placeholders})"
                params.extend(books)
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
        name_filter: str | None = None,
        trait: str | None = None,
        min_level: int | None = None,
        max_level: int | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """List entities by filters without requiring a search query.

        Unlike search(), this doesn't use FTS5 — it does a plain SQL query
        with WHERE filters. Use for bulk retrieval by book/type.

        Args:
            trait: Filter by trait in metadata JSON (case-insensitive).
                   Uses json_each() to search the traits array.
            min_level: Minimum level (inclusive) from metadata.level.
            max_level: Maximum level (inclusive) from metadata.level.

        Returns same dict structure as search() but without scores.
        """
        books = None
        if book:
            books = self.resolve_book_names(book)
            if not books:
                books = [book]

        if isinstance(category, str):
            category = [category]

        # Use a JOIN with json_each for trait filtering at the SQL level
        if trait:
            sql = (
                "SELECT DISTINCT c.* FROM content c, "
                "json_each(c.metadata, '$.traits') t "
                "WHERE lower(t.value) = ? "
            )
            params: list = [trait.strip().lower()]
        else:
            sql = "SELECT * FROM content WHERE 1=1"
            params = []

        if books:
            placeholders = ",".join("?" * len(books))
            tbl = "c" if trait else "content"
            sql += f" AND {tbl}.book IN ({placeholders})"
            params.extend(books)

        if book_type:
            tbl = "c" if trait else "content"
            sql += f" AND {tbl}.book_type = ?"
            params.append(book_type)

        if category:
            placeholders = ",".join("?" * len(category))
            tbl = "c" if trait else "content"
            sql += f" AND {tbl}.category IN ({placeholders})"
            params.extend(category)

        if include_types:
            placeholders = ",".join("?" * len(include_types))
            tbl = "c" if trait else "content"
            sql += f" AND {tbl}.type IN ({placeholders})"
            params.extend(include_types)

        if exclude_types:
            placeholders = ",".join("?" * len(exclude_types))
            tbl = "c" if trait else "content"
            sql += f" AND {tbl}.type NOT IN ({placeholders})"
            params.extend(exclude_types)

        if name_filter:
            tbl = "c" if trait else "content"
            sql += f" AND {tbl}.name LIKE ?"
            params.append(f"%{name_filter}%")

        if min_level is not None:
            tbl = "c" if trait else "content"
            sql += f" AND CAST(json_extract({tbl}.metadata, '$.level') AS INTEGER) >= ?"
            params.append(min_level)

        if max_level is not None:
            tbl = "c" if trait else "content"
            sql += f" AND CAST(json_extract({tbl}.metadata, '$.level') AS INTEGER) <= ?"
            params.append(max_level)

        tbl = "c" if trait else "content"
        sql += f" ORDER BY {tbl}.name LIMIT ?"
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
