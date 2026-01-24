#!/usr/bin/env python3
"""
Fast Pathfinder search using SQLite FTS5 + local embeddings for semantic search.

Primary: SQLite FTS5 (instant indexing, fast keyword search with query preprocessing)
Secondary: SQLite embeddings + sentence-transformers (semantic/vector search)
"""

import glob
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional
import time
import struct
import math

# Local embeddings via sentence-transformers (lazy loaded)
_embedding_model = None
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dims

# Source categories for filtering
# Note: Sources may have multiple name formats (extracted vs SRD), so we include both
SOURCE_CATEGORIES = {
    "core": [
        "[PZO12001E] Player Core",
        "[PZO12002E] GM Core",
        "[PZO12004E] Player Core 2 ",
        "Pathfinder Player Core",
        "Pathfinder GM Core",
        "Pathfinder Player Core 2",
    ],
    "bestiaries": [
        "[PZO12003E] Monster Core",
        "[PZO12007E] NPC Core",
        "[PZO12009E] Monster Core 2",
        "[PZO9311E] Monster of Myth",
        "Pathfinder Monster Core",
        "Pathfinder Monster Core 2",
        "Pathfinder NPC Core",
        "Pathfinder Bestiary",
        "Pathfinder Bestiary 2",
        "Pathfinder Bestiary 3",
    ],
    "supplements": [
        # Rules expansion books (extracted format)
        "[PZO12005e] Howl of the Wild",
        "[PZO12006E] War of Immortals",
        "[PZO12008e] Battlecry!",
        "[PZO12010e] Guns & Gears",
        "[PZO12011E] Treasure Vault",
        "[PZO2108E] Secrets of Magic",
        "[PZO2110E] Book of the Dead",
        "[PZO2111E] Dark Archive",
        "[PZO2113] Rage of Elements",
        "[PZO13003e] Divine Mysteries",
        "[PZO13004E] Rival Academies",
        "[PZO13007E] Draconic Codex",
        "[PZO9308E] Ancestry Guide",
        "[PZO9310E] Grand Bazaar",
        # SRD format
        "Pathfinder Howl of the Wild",
        "Pathfinder War of Immortals",
        "Pathfinder Guns & Gears",
        "Pathfinder Treasure Vault",
        "Pathfinder Secrets of Magic",
        "Pathfinder Book of the Dead",
        "Pathfinder Dark Archive",
        "Pathfinder Rage of Elements",
        "Pathfinder Lost Omens Ancestry Guide",
        "Pathfinder Lost Omens Grand Bazaar",
        "Pathfinder Lost Omens Draconic Codex",
    ],
    "setting": [
        # Lost Omens / Setting books (extracted format)
        "[PZO13001E] Tian Xia World Guide",
        "[PZO13002E] Tian Xia Character Guide",
        "[PZO13005E] Shining Kingdoms",
        "[PZO9301E] World Guide",
        "[PZO9303E] Gods & Magic",
        "[PZO9304E] Absalom",
        "[PZO9307E] Pathfinder Society Guide",
        "[PZO9309E] Mwangi Expanse",
        "[PZO9312E] Knights of Lastwall",
        "[PZO9313E] Travel Guide",
        "[PZO9314E] Impossible Lands",
        "[PZO9315E] Firebrands",
        "[PZO9316E] Highhelm",
        "[PZOPZO9302E] Character Guide",
        # SRD format
        "Pathfinder Lost Omens World Guide",
        "Pathfinder Lost Omens Gods & Magic",
        "Pathfinder Lost Omens Absalom",
        "Pathfinder Lost Omens Mwangi Expanse",
        "Pathfinder Lost Omens Knights of Lastwall",
        "Pathfinder Lost Omens Travel Guide",
        "Pathfinder Lost Omens Impossible Lands",
        "Pathfinder Lost Omens Firebrands",
        "Pathfinder Lost Omens Highhelm",
        "Pathfinder Lost Omens Character Guide",
        "Pathfinder Lost Omens Tian Xia World Guide",
        "Pathfinder Lost Omens Tian Xia Character Guide",
    ],
    "adventures": [
        "[PZO2020E] Kingmaker",
        "Pathfinder Kingmaker",
    ],
}

# Helpful aliases for source categories
SOURCE_CATEGORY_ALIASES = {
    "rules": ["core", "supplements"],  # All rules content (no setting/adventures)
    "monsters": ["bestiaries"],
    "creatures": ["bestiaries"],
    "world": ["setting"],
    "lost_omens": ["setting"],
    "aps": ["adventures"],
}

# Content types that are summaries/aggregated content
SUMMARY_TYPES = frozenset(
    {
        "page_summary",
        "chapter_summary",
        "book_summary",
        "rollup_summary",
        "summary",
    }
)

# Raw/fallback content types (lowest quality for structured queries)
RAW_TYPES = frozenset(
    {
        "page_text",
        "bestiary_page",
        "raw_text",
    }
)


def get_sources_for_categories(categories: list[str]) -> list[str]:
    """
    Get list of source names for given category names.

    Args:
        categories: List of category names like ['core', 'bestiaries']
                   Can also use aliases like 'rules' or 'monsters'

    Returns:
        List of source book names
    """
    sources = set()
    for cat in categories:
        cat_lower = cat.lower()
        # Check if it's an alias
        if cat_lower in SOURCE_CATEGORY_ALIASES:
            for sub_cat in SOURCE_CATEGORY_ALIASES[cat_lower]:
                sources.update(SOURCE_CATEGORIES.get(sub_cat, []))
        # Check if it's a direct category
        elif cat_lower in SOURCE_CATEGORIES:
            sources.update(SOURCE_CATEGORIES[cat_lower])
    return list(sources)


def list_filter_options() -> dict:
    """
    List all available filter options for search.

    Returns dict with:
        - source_categories: Available category names and their source counts
        - source_aliases: Aliases that expand to multiple categories
        - summary_types: Content types considered summaries
        - raw_types: Content types considered raw/fallback
    """
    return {
        "source_categories": {cat: len(sources) for cat, sources in SOURCE_CATEGORIES.items()},
        "source_aliases": {alias: cats for alias, cats in SOURCE_CATEGORY_ALIASES.items()},
        "summary_types": list(SUMMARY_TYPES),
        "raw_types": list(RAW_TYPES),
    }


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
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
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
        "what",
        "which",
        "who",
        "whom",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "will",
        "would",
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
        "me",
        "about",
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
        # Game term stop words - these indicate type but shouldn't be required in results
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
    # Lowercase
    query = query.lower()

    # Remove punctuation except hyphens (for terms like "off-guard")
    query = re.sub(r"[^\w\s\-]", " ", query)

    # Split into words
    words = query.split()

    # Remove stop words, keep words >= 2 chars
    keywords = [w for w in words if w not in STOP_WORDS and len(w) >= 2]

    # If we removed everything, try keeping more words
    if not keywords and words:
        keywords = [w for w in words if len(w) >= 3]

    # If still nothing, return original (stripped)
    if not keywords:
        return query.strip()

    return " ".join(keywords)


def decompose_complex_query(query: str) -> list[dict]:
    """
    Decompose a complex question into simpler sub-queries.

    Returns list of {query, type_hint, description} dicts.

    Examples:
        "How much damage does a +2 striking longsword do?"
        -> [
            {query: "striking rune", type_hint: "rule", description: "striking rune rules"},
            {query: "longsword", type_hint: "equipment", description: "longsword stats"},
            {query: "weapon damage", type_hint: "rule", description: "damage calculation"}
        ]
    """
    query_lower = query.lower()
    sub_queries = []

    # Pattern: weapon damage questions
    if any(w in query_lower for w in ["damage", "hit", "attack"]) and any(
        w in query_lower for w in ["sword", "axe", "weapon", "bow", "spear", "dagger", "mace"]
    ):
        # Extract weapon name
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

        # Check for runes
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

        # Add general damage rules
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
        # Extract spell name if present
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

    # Pattern: class/level questions
    elif any(w in query_lower for w in ["level", "class", "feature", "ability"]):
        classes = [
            "fighter",
            "wizard",
            "cleric",
            "rogue",
            "ranger",
            "barbarian",
            "bard",
            "champion",
            "druid",
            "monk",
            "sorcerer",
            "witch",
            "oracle",
            "swashbuckler",
            "investigator",
            "inventor",
            "gunslinger",
            "magus",
            "summoner",
            "thaumaturge",
            "psychic",
            "kineticist",
        ]
        for cls in classes:
            if cls in query_lower:
                sub_queries.append(
                    {
                        "query": cls,
                        "type_hint": "class",
                        "description": f"{cls} class features",
                    }
                )
                # Check for specific level
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
        # Check for level-based DC
        level_match = re.search(r"level\s*(\d+)", query_lower)
        if level_match or "level" in query_lower:
            sub_queries.append(
                {
                    "query": "DC by level table",
                    "type_hint": "table",
                    "description": "DC table by level",
                }
            )
        # Check for creature DC
        if "creature" in query_lower:
            sub_queries.append(
                {
                    "query": "creature difficulty class",
                    "type_hint": "rule",
                    "description": "creature DC rules",
                }
            )

    # Pattern: condition questions - check for condition names even without "condition" keyword
    conditions = [
        "dying",
        "wounded",
        "unconscious",
        "frightened",
        "sickened",
        "stunned",
        "paralyzed",
        "petrified",
        "blinded",
        "deafened",
        "fatigued",
        "drained",
        "doomed",
        "clumsy",
        "enfeebled",
        "stupefied",
        "off-guard",
        "flat-footed",
        "prone",
        "grabbed",
        "restrained",
        "immobilized",
        "hidden",
        "concealed",
        "invisible",
        "quickened",
        "slowed",
        "persistent damage",
    ]
    found_conditions = [cond for cond in conditions if cond in query_lower]
    if found_conditions:
        for cond in found_conditions:
            sub_queries.append(
                {
                    "query": cond,
                    "type_hint": "condition",
                    "description": f"{cond} condition",
                }
            )
        # Add related rules if asking "what happens"
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

    # Pattern: condition questions with "condition" keyword but no specific condition named
    elif any(w in query_lower for w in ["condition", "status", "effect"]):
        sub_queries.append(
            {
                "query": preprocess_query(query),
                "type_hint": "condition",
                "description": "condition search",
            }
        )

    # If no patterns matched, return the original query
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
            print(f"Loaded embedding model: {EMBED_MODEL_NAME}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return None
    return _embedding_model


def get_embedding(text: str) -> list[float]:
    """Get embedding using local sentence-transformers model."""
    model = get_embedding_model()
    if model is None:
        return []
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


# Load remaster aliases for legacy->remaster term expansion
REMASTER_ALIASES = {}
REMASTER_ALIASES_PATH = os.path.join(os.path.dirname(__file__), "remaster_aliases.json")
if os.path.exists(REMASTER_ALIASES_PATH):
    try:
        with open(REMASTER_ALIASES_PATH) as f:
            _alias_data = json.load(f)
            # Flatten all categories into a single lookup dict (legacy -> remaster)
            for category, mappings in _alias_data.items():
                if category == "meta" or not isinstance(mappings, dict):
                    continue
                for legacy, remaster in mappings.items():
                    if remaster:  # Skip null values
                        REMASTER_ALIASES[legacy.lower()] = remaster.lower()
    except Exception as e:
        print(f"Warning: Could not load remaster aliases: {e}")


def expand_query_aliases(query: str) -> list[str]:
    """Expand query with remaster aliases. Returns list of queries to try."""
    query_lower = query.lower()
    queries = [query]

    # Check if the exact query is a legacy term
    if query_lower in REMASTER_ALIASES:
        queries.append(REMASTER_ALIASES[query_lower])

    # Also check reverse (remaster -> legacy) for flexibility
    for legacy, remaster in REMASTER_ALIASES.items():
        if query_lower == remaster and legacy not in queries:
            queries.append(legacy)

    # For multi-word queries, also try expanding individual terms
    words = query_lower.split()
    if len(words) > 1:
        # Build reverse lookup (remaster -> legacy)
        reverse_aliases = {v: k for k, v in REMASTER_ALIASES.items()}

        # Try replacing each word with its alias
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


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts using local sentence-transformers."""
    model = get_embedding_model()
    if model is None:
        return []
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100)
        return embeddings.tolist()
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return []


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
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", data))


class PathfinderSearch:
    """Fast search engine for Pathfinder content."""

    def __init__(self, db_path: str = "pathfinder_search.db"):
        self.db_path = db_path
        self.conn = None
        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite with FTS5 and embeddings table."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Create main documents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_key TEXT UNIQUE,
                name TEXT,
                type TEXT,
                source TEXT,
                edition TEXT DEFAULT 'pf2e_remaster',
                content TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add edition column if missing (migration for existing databases)
        try:
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN edition TEXT DEFAULT 'pf2e_remaster'"
            )
            # Migrate: populate edition from metadata where available
            self.conn.execute("""
                UPDATE documents SET edition = json_extract(metadata, '$.edition')
                WHERE json_extract(metadata, '$.edition') IS NOT NULL
            """)
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create embeddings table (separate for optional semantic search)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id INTEGER PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """)

        # Create FTS5 virtual table for full-text search
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                name,
                type,
                source,
                content,
                content='documents',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS in sync
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, name, type, source, content)
                VALUES (new.id, new.name, new.type, new.source, new.content);
            END
        """)

        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, name, type, source, content)
                VALUES ('delete', old.id, old.name, old.type, old.source, old.content);
            END
        """)

        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, name, type, source, content)
                VALUES ('delete', old.id, old.name, old.type, old.source, old.content);
                INSERT INTO documents_fts(rowid, name, type, source, content)
                VALUES (new.id, new.name, new.type, new.source, new.content);
            END
        """)

        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_key ON documents(doc_key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_edition ON documents(edition)")

        self.conn.commit()

    def add_documents_batch(
        self,
        documents: list[dict],
        add_embeddings: bool = False,
        show_progress: bool = True,
    ) -> int:
        """Add multiple documents efficiently."""
        start = time.time()

        # SQLite batch insert
        batch_data = []
        for doc in documents:
            doc_key = f"{doc['source']}::{doc['name']}"
            # Determine edition: from doc, metadata, or default
            edition = (
                doc.get("edition") or doc.get("metadata", {}).get("edition") or "pf2e_remaster"
            )
            batch_data.append(
                (
                    doc_key,
                    doc["name"],
                    doc["type"],
                    doc["source"],
                    edition,
                    doc["content"],
                    json.dumps(doc.get("metadata", {})),
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO documents (doc_key, name, type, source, edition, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            batch_data,
        )
        self.conn.commit()
        added = len(batch_data)

        elapsed = time.time() - start
        if show_progress:
            print(
                f"SQLite: Added {added} documents in {elapsed:.2f}s ({added/elapsed:.0f} docs/sec)"
            )

        # Optionally add embeddings
        if add_embeddings:
            self._add_embeddings_batch(documents, show_progress)

        return added

    def _add_embeddings_batch(self, documents: list[dict], show_progress: bool = True):
        """Add embeddings for documents to SQLite using local sentence-transformers."""
        print("Generating embeddings via sentence-transformers and storing in SQLite...")
        embed_start = time.time()

        # Get all doc_ids for the documents we're embedding
        doc_keys = [f"{doc['source']}::{doc['name']}" for doc in documents]

        batch_size = 100  # sentence-transformers batch size
        total_embedded = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_keys = doc_keys[i : i + batch_size]

            # Get contents for embedding (truncate for model token limit)
            contents = [doc["content"][:2000] for doc in batch]

            # Generate embeddings
            embeddings = get_embeddings_batch(contents)
            if not embeddings:
                print("Failed to generate embeddings, skipping batch")
                continue

            # Get doc_ids from database
            placeholders = ",".join("?" * len(batch_keys))
            cursor = self.conn.execute(
                f"SELECT id, doc_key FROM documents WHERE doc_key IN ({placeholders})",
                batch_keys,
            )
            doc_id_map = {row["doc_key"]: row["id"] for row in cursor}

            # Insert embeddings into SQLite
            embed_data = []
            for j, doc in enumerate(batch):
                doc_key = batch_keys[j]
                if doc_key in doc_id_map and j < len(embeddings):
                    embed_data.append((doc_id_map[doc_key], pack_embedding(embeddings[j])))

            if embed_data:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (doc_id, embedding) VALUES (?, ?)",
                    embed_data,
                )
                self.conn.commit()
                total_embedded += len(embed_data)

            if show_progress:
                print(f"  Embeddings: {min(i+batch_size, len(documents))}/{len(documents)}")

        embed_elapsed = time.time() - embed_start
        if show_progress:
            print(f"SQLite: Indexed {total_embedded} embeddings in {embed_elapsed:.2f}s")

    def index_all_embeddings(self, show_progress: bool = True):
        """Generate embeddings for all documents in the database."""
        # Get count of documents without embeddings
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM documents d
            LEFT JOIN embeddings e ON d.id = e.doc_id
            WHERE e.doc_id IS NULL
        """)
        missing_count = cursor.fetchone()[0]

        if missing_count == 0:
            print("All documents already have embeddings")
            return 0

        print(f"Generating embeddings for {missing_count} documents...")

        batch_size = 100
        total_embedded = 0
        offset = 0

        while True:
            # Get batch of documents without embeddings
            cursor = self.conn.execute(
                """
                SELECT d.id, d.doc_key, d.content FROM documents d
                LEFT JOIN embeddings e ON d.id = e.doc_id
                WHERE e.doc_id IS NULL
                LIMIT ? OFFSET ?
            """,
                (batch_size, offset),
            )

            rows = cursor.fetchall()
            if not rows:
                break

            # Generate embeddings
            contents = [row["content"][:2000] for row in rows]
            embeddings = get_embeddings_batch(contents)

            if not embeddings:
                print("Failed to generate embeddings")
                break

            # Store in database
            embed_data = [
                (row["id"], pack_embedding(embeddings[i]))
                for i, row in enumerate(rows)
                if i < len(embeddings)
            ]

            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (doc_id, embedding) VALUES (?, ?)",
                embed_data,
            )
            self.conn.commit()
            total_embedded += len(embed_data)

            if show_progress:
                print(f"  Embeddings: {total_embedded}/{missing_count}")

            offset += batch_size

        print(f"Generated {total_embedded} embeddings")
        return total_embedded

    # Default type rankings for GM agent use
    # Higher values = more preferred in results
    DEFAULT_TYPE_BOOST = {
        # Core game content (highest priority)
        "spell": 20,
        "cantrip": 20,
        "focus_spell": 20,
        "ritual": 15,
        "creature": 20,
        "npc": 18,
        "condition": 25,
        "rule": 22,
        "action": 20,
        "feat": 18,
        "class_feature": 18,
        "equipment": 15,
        "item": 15,
        "ancestry": 15,
        "heritage": 15,
        "background": 12,
        "class": 15,
        "archetype": 12,
        "trait": 10,
        "deity": 12,
        "hazard": 15,
        "haunt": 15,
        # Setting content
        "location": 10,
        "settlement": 10,
        "region": 8,
        "npc_group": 8,
        "faction": 8,
        # Reference content (lower priority)
        "guidance": 5,
        "table": 5,
        "subsystem": 8,
        "example_of_play": 3,
        # Raw page text (lowest - use as fallback)
        "page_text": -10,
        "bestiary_page": -5,
    }

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        source: str = None,
        edition: str | list = None,
        use_semantic: bool = False,
        preprocess: bool = True,
        boost_types: dict = None,
        demote_page_text: bool = True,
        source_categories: list[str] = None,
        include_types: list[str] = None,
        exclude_types: list[str] = None,
        exclude_summaries: bool = False,
        exclude_raw: bool = False,
    ) -> list[dict]:
        """
        Search for documents.

        Args:
            query: Search query (natural language supported with preprocessing)
            top_k: Number of results to return
            doc_type: Filter by document type (single type)
            source: Filter by source (single source name)
            edition: Filter by edition - "pf1e", "pf2e", "pf2e_remaster", or list of editions
                    None or "all" returns all editions
            use_semantic: Use semantic/vector search (embeddings) instead of FTS5
            preprocess: Apply query preprocessing to extract keywords (default True)
            boost_types: Dict of {type: boost_value} to adjust ranking (default uses DEFAULT_TYPE_BOOST)
            demote_page_text: If True, deprioritize page_text results (default True for GM use)
            source_categories: Filter by source categories like ['core', 'bestiaries', 'supplements']
                             Can also use aliases like 'rules' (core + supplements) or 'monsters'
            include_types: Only include these document types (whitelist)
            exclude_types: Exclude these document types (blacklist)
            exclude_summaries: If True, exclude summary types (page_summary, chapter_summary, etc.)
            exclude_raw: If True, exclude raw text types (page_text, bestiary_page, etc.)
        """
        # Normalize edition parameter
        if edition == "all" or edition is None:
            edition_filter = None
        elif isinstance(edition, str):
            edition_filter = [edition]
        else:
            edition_filter = edition

        # Build source filter from categories
        source_filter = None
        if source_categories:
            source_filter = get_sources_for_categories(source_categories)
        elif source:
            source_filter = [source]

        # Build type exclusion set
        type_exclusions = set()
        if exclude_summaries:
            type_exclusions.update(SUMMARY_TYPES)
        if exclude_raw:
            type_exclusions.update(RAW_TYPES)
        if exclude_types:
            type_exclusions.update(exclude_types)

        # Preprocess query to extract keywords (handles natural language questions)
        original_query = query
        if preprocess:
            query = preprocess_query(query)
            # If preprocessing removed everything, use original
            if not query.strip():
                query = original_query

        # Determine type boosts to apply
        type_boosts = self.DEFAULT_TYPE_BOOST.copy() if demote_page_text else {}
        if boost_types:
            type_boosts.update(boost_types)

        if use_semantic:
            results = self._search_semantic(
                original_query,
                top_k * 2,
                doc_type,
                source_filter,
                edition_filter,
                include_types,
                type_exclusions,
            )
        else:
            results = self._search_fts(
                query,
                top_k * 2,
                doc_type,
                source_filter,
                edition_filter,
                original_query,
                include_types,
                type_exclusions,
            )

        # Apply type-based score adjustments
        if type_boosts and results:
            for r in results:
                type_boost = type_boosts.get(r["type"], 0)
                r["score"] = r["score"] + type_boost
                r["type_boost"] = type_boost  # For debugging

            # Re-sort by adjusted score
            results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    # Known condition/rule names that should be boosted when mentioned in queries
    BOOSTABLE_TERMS = {
        "persistent damage",
        "dying",
        "wounded",
        "doomed",
        "drained",
        "fatigued",
        "frightened",
        "sickened",
        "stunned",
        "slowed",
        "paralyzed",
        "petrified",
        "blinded",
        "deafened",
        "invisible",
        "hidden",
        "concealed",
        "cover",
        "flanking",
        "off-guard",
        "flat-footed",
        "prone",
        "grabbed",
        "restrained",
        "immobilized",
        "quickened",
        "clumsy",
        "enfeebled",
        "stupefied",
    }

    def _search_fts(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        source_filter: list = None,
        edition_filter: list = None,
        original_query: str = None,
        include_types: list = None,
        type_exclusions: set = None,
    ) -> list[dict]:
        """Full-text search using FTS5 with exact match boosting and alias expansion."""
        # Use original query for boosting logic if available
        boost_query = original_query or query

        # Expand query with remaster aliases (e.g., "magic missile" -> ["magic missile", "force barrage"])
        # IMPORTANT: Expand BOTH the preprocessed query AND the original query
        # This handles cases like "bag of holding" -> "spacious pouch" where preprocessing
        # removes stop words ("of") that are part of the alias key
        queries = set(expand_query_aliases(query))
        if original_query and original_query != query:
            queries.update(expand_query_aliases(original_query))
        queries = list(queries)

        # Collect results from all query variants
        all_results = {}  # keyed by (name, source) to dedupe

        for q in queries:
            results = self._search_fts_single(
                q,
                top_k * 2,
                doc_type,
                source_filter,
                edition_filter,
                include_types,
                type_exclusions,
            )
            for r in results:
                key = (r["name"], r["source"])
                # Keep the result with the best score
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # Boost conditions/rules when query contains known terms
        # This ensures the core rule is included even if other terms dilute BM25 score
        if doc_type is None:  # Only when not filtering by type
            query_lower = boost_query.lower()
            for term in self.BOOSTABLE_TERMS:
                if term in query_lower:
                    # Search for just the condition/rule term
                    term_results = self._search_fts_single(
                        term,
                        5,
                        doc_type=None,
                        source_filter=source_filter,
                        edition_filter=edition_filter,
                        include_types=include_types,
                        type_exclusions=type_exclusions,
                    )
                    for r in term_results:
                        if r["type"] in ("condition", "rule", "trait"):
                            key = (r["name"], r["source"])
                            # Boost score significantly for core rule matches
                            r["score"] = r["score"] + 50  # Boost to ensure visibility
                            if key not in all_results or r["score"] > all_results[key]["score"]:
                                all_results[key] = r

        # Sort by score and return top_k
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        # Dedupe by name (case-insensitive) - keep highest scoring version
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
        source_filter: list = None,
        edition_filter: list = None,
        include_types: list = None,
        type_exclusions: set = None,
    ) -> list[dict]:
        """Single FTS5 search without alias expansion."""
        # Boost exact name matches by adding a CASE expression
        # bm25 returns negative scores (more negative = better match)
        # We subtract 100 for exact matches to boost them significantly
        sql = """
            SELECT d.*,
                   bm25(documents_fts) - (CASE WHEN LOWER(d.name) = LOWER(?) THEN 100 ELSE 0 END) as score
            FROM documents d
            JOIN documents_fts ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
        """
        params = [query, query]

        if doc_type:
            sql += " AND d.type = ?"
            params.append(doc_type)

        # Source filtering (supports list from source_categories)
        if source_filter:
            placeholders = ",".join("?" * len(source_filter))
            sql += f" AND d.source IN ({placeholders})"
            params.extend(source_filter)

        if edition_filter:
            placeholders = ",".join("?" * len(edition_filter))
            sql += f" AND d.edition IN ({placeholders})"
            params.extend(edition_filter)

        # Type whitelist (only include specific types)
        if include_types:
            placeholders = ",".join("?" * len(include_types))
            sql += f" AND d.type IN ({placeholders})"
            params.extend(include_types)

        # Type blacklist (exclude specific types)
        if type_exclusions:
            placeholders = ",".join("?" * len(type_exclusions))
            sql += f" AND d.type NOT IN ({placeholders})"
            params.extend(type_exclusions)

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        try:
            cursor = self.conn.execute(sql, params)
            results = []
            for row in cursor:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                results.append(
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "source": row["source"],
                        "edition": row["edition"],
                        "content": row["content"],
                        "metadata": metadata,
                        "score": -row["score"],
                        # Conversion tagging
                        "is_converted": metadata.get("converted_from") is not None,
                        "converted_from": metadata.get("converted_from"),
                        "conversion_confidence": metadata.get("conversion_confidence"),
                    }
                )
            return results
        except sqlite3.OperationalError:
            # FTS5 query syntax error - try quoted query
            safe_query = '"' + query.replace('"', '""') + '"'
            params[1] = safe_query
            try:
                cursor = self.conn.execute(sql, params)
                results = []
                for row in cursor:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    results.append(
                        {
                            "name": row["name"],
                            "type": row["type"],
                            "source": row["source"],
                            "edition": row["edition"],
                            "content": row["content"],
                            "metadata": metadata,
                            "score": -row["score"],
                            "is_converted": metadata.get("converted_from") is not None,
                            "converted_from": metadata.get("converted_from"),
                            "conversion_confidence": metadata.get("conversion_confidence"),
                        }
                    )
                return results
            except:
                return []

    def _search_semantic(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str = None,
        source_filter: list = None,
        edition_filter: list = None,
        include_types: list = None,
        type_exclusions: set = None,
    ) -> list[dict]:
        """Semantic search using SQLite embeddings with local sentence-transformers."""
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            print("Failed to get query embedding, falling back to FTS")
            return self._search_fts(
                query,
                top_k,
                doc_type,
                source_filter,
                edition_filter,
                None,
                include_types,
                type_exclusions,
            )

        # Check if we have embeddings
        cursor = self.conn.execute("SELECT COUNT(*) FROM embeddings")
        embed_count = cursor.fetchone()[0]
        if embed_count == 0:
            print("No embeddings in database. Run index_all_embeddings() first.")
            return self._search_fts(
                query,
                top_k,
                doc_type,
                source_filter,
                edition_filter,
                None,
                include_types,
                type_exclusions,
            )

        # Build SQL query with filters
        sql = """
            SELECT d.*, e.embedding
            FROM documents d
            JOIN embeddings e ON d.id = e.doc_id
            WHERE 1=1
        """
        params = []

        if doc_type:
            sql += " AND d.type = ?"
            params.append(doc_type)

        if source_filter:
            placeholders = ",".join("?" * len(source_filter))
            sql += f" AND d.source IN ({placeholders})"
            params.extend(source_filter)

        if edition_filter:
            placeholders = ",".join("?" * len(edition_filter))
            sql += f" AND d.edition IN ({placeholders})"
            params.extend(edition_filter)

        if include_types:
            placeholders = ",".join("?" * len(include_types))
            sql += f" AND d.type IN ({placeholders})"
            params.extend(include_types)

        if type_exclusions:
            placeholders = ",".join("?" * len(type_exclusions))
            sql += f" AND d.type NOT IN ({placeholders})"
            params.extend(type_exclusions)

        try:
            cursor = self.conn.execute(sql, params)

            # Calculate cosine similarity for each result
            scored_results = []
            for row in cursor:
                doc_embedding = unpack_embedding(row["embedding"])
                similarity = cosine_similarity(query_embedding, doc_embedding)

                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                scored_results.append(
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "source": row["source"],
                        "edition": row["edition"],
                        "content": row["content"],
                        "metadata": metadata,
                        "score": similarity,
                        "is_converted": metadata.get("converted_from") is not None,
                        "converted_from": metadata.get("converted_from"),
                        "conversion_confidence": metadata.get("conversion_confidence"),
                    }
                )

            # Sort by similarity (highest first) and return top_k
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            return scored_results[:top_k]

        except Exception as e:
            print(f"Semantic search error: {e}, falling back to FTS")
            return self._search_fts(
                query,
                top_k,
                doc_type,
                source_filter,
                edition_filter,
                None,
                include_types,
                type_exclusions,
            )

    def search_complex(self, query: str, top_k: int = 10, **kwargs) -> dict:
        """
        Handle complex questions by decomposing into sub-queries.

        Returns a dict with:
            - decomposition: list of sub-queries generated
            - results: dict of {sub_query: results}
            - combined: merged and ranked results

        Example:
            >>> search.search_complex("How much damage does a +2 striking longsword do?")
            {
                'decomposition': [
                    {'query': 'longsword', 'type_hint': 'equipment', ...},
                    {'query': 'striking rune', 'type_hint': 'equipment', ...},
                    {'query': 'weapon damage dice', 'type_hint': 'rule', ...}
                ],
                'results': {
                    'longsword': [...],
                    'striking rune': [...],
                    'weapon damage dice': [...]
                },
                'combined': [...]  # Top results across all sub-queries
            }
        """
        # Decompose the query
        sub_queries = decompose_complex_query(query)

        # If no decomposition happened, just do regular search
        if len(sub_queries) == 1 and sub_queries[0]["type_hint"] is None:
            results = self.search(query, top_k=top_k, **kwargs)
            return {
                "decomposition": sub_queries,
                "results": {query: results},
                "combined": results,
            }

        # Execute each sub-query
        all_results = {}
        combined = {}

        for sq in sub_queries:
            sub_query = sq["query"]
            type_hint = sq["type_hint"]

            # Search with type hint as filter if provided
            results = self.search(sub_query, top_k=top_k, doc_type=type_hint, **kwargs)

            # If no results with type filter, try without
            if not results and type_hint:
                results = self.search(sub_query, top_k=top_k, **kwargs)

            all_results[sub_query] = results

            # Add to combined, tracking by (name, source) to dedupe
            for r in results:
                key = (r["name"], r["source"])
                if key not in combined or r["score"] > combined[key]["score"]:
                    combined[key] = r

        # Sort combined by score
        combined_list = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        return {
            "decomposition": sub_queries,
            "results": all_results,
            "combined": combined_list[:top_k],
        }

    def get_stats(self) -> dict:
        """Get index statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) as total FROM documents")
        total = cursor.fetchone()["total"]

        cursor = self.conn.execute("SELECT COUNT(*) as total FROM embeddings")
        embed_count = cursor.fetchone()["total"]

        cursor = self.conn.execute("""
            SELECT type, COUNT(*) as count
            FROM documents
            GROUP BY type
            ORDER BY count DESC
        """)
        by_type = {row["type"]: row["count"] for row in cursor}

        cursor = self.conn.execute("""
            SELECT source, COUNT(*) as count
            FROM documents
            GROUP BY source
            ORDER BY count DESC
            LIMIT 20
        """)
        by_source = {row["source"]: row["count"] for row in cursor}

        cursor = self.conn.execute("""
            SELECT edition, COUNT(*) as count
            FROM documents
            GROUP BY edition
            ORDER BY count DESC
        """)
        by_edition = {row["edition"] or "unknown": row["count"] for row in cursor}

        return {
            "total_documents": total,
            "embeddings_count": embed_count,
            "by_type": by_type,
            "by_source": by_source,
            "by_edition": by_edition,
        }

    def clear(self):
        """Clear all data."""
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM documents")
        self.conn.execute("DELETE FROM documents_fts")
        self.conn.commit()

    def close(self):
        """Close connections."""
        if self.conn:
            self.conn.close()


def load_srd_documents(srd_dir: str = "srd_data") -> list[dict]:
    """Load SRD data as documents."""
    docs = []

    if not os.path.exists(srd_dir):
        return docs

    srd_files = [
        ("actions.json", "action"),
        ("conditions.json", "condition"),
        ("spells.json", "spell"),
        ("feats.json", "feat"),
        ("equipments.json", "equipment"),
        ("classs.json", "class"),
        ("ancestrys.json", "ancestry"),
        ("deitys.json", "deity"),
        ("creatures.json", "creature"),
        ("npcs.json", "npc"),
        ("vehicles.json", "vehicle"),
        ("hazards.json", "hazard"),
        ("heritages.json", "heritage"),
        ("backgrounds.json", "background"),
        ("familiar_abilitys.json", "familiar_ability"),
    ]

    for filename, doc_type in srd_files:
        filepath = os.path.join(srd_dir, filename)
        if not os.path.exists(filepath):
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)

            for item in data:
                name = item.get("name", "")
                if not name:
                    continue

                # Build content string
                content_parts = [f"name: {name}"]

                for key in ["description", "level", "hp", "ac", "traits", "source"]:
                    value = item.get(key)
                    if value:
                        if isinstance(value, list):
                            value = ", ".join(str(v) for v in value)
                        content_parts.append(f"{key}: {value}")

                # Add abilities for creatures
                if doc_type in ("creature", "npc"):
                    abilities = item.get("abilities", [])
                    if abilities:
                        ability_names = [a.get("name", "") for a in abilities if a.get("name")]
                        if ability_names:
                            content_parts.append(f"abilities: {', '.join(ability_names)}")

                    attacks = item.get("attacks", [])
                    if attacks:
                        attack_info = []
                        for a in attacks:
                            if a.get("name"):
                                attack_str = a["name"]
                                if a.get("bonus"):
                                    attack_str += f" +{a['bonus']}"
                                if a.get("damage"):
                                    attack_str += f" ({a['damage']})"
                                attack_info.append(attack_str)
                        if attack_info:
                            content_parts.append(f"attacks: {', '.join(attack_info)}")

                content = " | ".join(content_parts)

                # Handle source that might be a dict
                item_source = item.get("source", "SRD")
                if isinstance(item_source, dict):
                    item_source = "SRD"
                elif isinstance(item_source, list):
                    item_source = item_source[0] if item_source else "SRD"

                docs.append(
                    {
                        "name": name,
                        "type": item.get("type", doc_type),
                        "source": str(item_source),
                        "content": content,
                        "metadata": {
                            k: v for k, v in item.items() if isinstance(v, (str, int, float, bool))
                        },
                    }
                )

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return docs


def load_extracted_documents(extracted_dir: str = "extracted") -> list[dict]:
    """Load extracted PDF documents."""
    docs = []

    if not os.path.exists(extracted_dir):
        return docs

    for book_dir in os.listdir(extracted_dir):
        book_path = os.path.join(extracted_dir, book_dir)
        if not os.path.isdir(book_path):
            continue

        entity_files = [
            ("creatures.json", "creature"),
            ("items.json", "item"),
            ("npcs.json", "npc"),
            ("locations.json", "location"),
            ("hazards.json", "hazard"),
            ("haunts.json", "haunt"),
            ("read_aloud.json", "read_aloud"),
            ("rules.json", "rule"),
            ("tables.json", "table"),
            ("actions.json", "action"),
            ("conditions.json", "condition"),
            ("subsystems.json", "subsystem"),
            ("skill_checks.json", "skill_check"),
            # Spell types
            ("spells.json", "spell"),
            ("cantrips.json", "cantrip"),
            ("focus_spells.json", "focus_spell"),
            # Character options
            ("feats.json", "feat"),
            ("classes.json", "class"),
            ("ancestries.json", "ancestry"),
            ("heritages.json", "heritage"),
            ("backgrounds.json", "background"),
            # Guidance and examples
            ("guidance.json", "guidance"),
            ("example_of_play.json", "example_of_play"),
            ("rulebook_content.json", None),  # Use item's 'type' field
        ]

        for filename, doc_type in entity_files:
            filepath = os.path.join(book_path, filename)
            if not os.path.exists(filepath):
                continue

            try:
                with open(filepath) as f:
                    data = json.load(f)

                for item in data:
                    name = item.get("name", "")
                    if not name:
                        continue

                    content_parts = [f"name: {name}"]
                    for key, value in item.items():
                        if key != "name" and value:
                            if isinstance(value, list):
                                value = ", ".join(str(v) for v in value)
                            elif isinstance(value, dict):
                                continue
                            content_parts.append(f"{key}: {value}")

                    # Use item's type field if doc_type is None (e.g., rulebook_content.json)
                    item_type = doc_type if doc_type else item.get("type", "rule")
                    # Handle case where type is a list or dict
                    if isinstance(item_type, list):
                        item_type = item_type[0] if item_type else "rule"
                    elif isinstance(item_type, dict):
                        # Extract first key or value as type
                        item_type = list(item_type.keys())[0] if item_type else "rule"
                    docs.append(
                        {
                            "name": name,
                            "type": str(item_type),  # Ensure string
                            "source": book_dir,  # Always use book_dir, not item's source
                            "content": " | ".join(content_parts),
                            "metadata": {
                                k: v
                                for k, v in item.items()
                                if isinstance(v, (str, int, float, bool))
                            },
                        }
                    )

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    return docs


def load_full_text_documents(extracted_dir: str = "extracted") -> list[dict]:
    """Load full_text_cleaned.md files as searchable page chunks.

    This captures content that doesn't fit neatly into structured extraction,
    like example-of-play dialogues, sidebar prose, and narrative content.
    """
    import re

    docs = []

    for book_dir in sorted(os.listdir(extracted_dir)):
        full_text_path = os.path.join(extracted_dir, book_dir, "full_text_cleaned.md")
        if not os.path.exists(full_text_path):
            continue

        try:
            with open(full_text_path) as f:
                content = f.read()

            # Split by page headers (## Page N)
            pages = re.split(r"^## Page (\d+)\s*$", content, flags=re.MULTILINE)

            # pages[0] is content before first page header (usually empty)
            # then alternating: page_num, content, page_num, content...
            for i in range(1, len(pages), 2):
                if i + 1 >= len(pages):
                    break
                page_num = pages[i]
                page_content = pages[i + 1].strip()

                if not page_content or len(page_content) < 100:
                    continue  # Skip nearly empty pages

                # Truncate very long pages for indexing
                if len(page_content) > 5000:
                    page_content = page_content[:5000] + "..."

                docs.append(
                    {
                        "name": f"{book_dir} - Page {page_num}",
                        "type": "page_text",
                        "source": book_dir,
                        "content": page_content,
                        "metadata": {"page": int(page_num), "source": book_dir},
                    }
                )

        except Exception as e:
            print(f"Error loading {full_text_path}: {e}")

    return docs


def load_bestiary_pages(extracted_dir: str = "extracted") -> list[dict]:
    """Load bestiary pages as searchable documents for flavor/lore content."""
    docs = []

    # Bestiary directories to index
    bestiary_dirs = {
        "PZO12003E Monster Core": "Monster Core",
        "PZO12009E": "Monster Core 2",
    }

    for book_dir, book_name in bestiary_dirs.items():
        pages_dir = os.path.join(extracted_dir, book_dir, "pages")
        if not os.path.exists(pages_dir):
            continue

        for page_file in sorted(glob.glob(os.path.join(pages_dir, "page_*.json"))):
            try:
                with open(page_file) as f:
                    data = json.load(f)

                text = data.get("text", "")
                page_num = data.get("page", 0)

                # Skip very short pages (likely just images/headers)
                if len(text) < 200:
                    continue

                # Extract creature/section name from first header
                lines = text.split("\n")
                section_name = None
                for line in lines[:10]:
                    if line.startswith("# ") or line.startswith("## "):
                        section_name = line.lstrip("#").strip()
                        break

                if not section_name:
                    section_name = f"Page {page_num}"

                docs.append(
                    {
                        "name": section_name,
                        "type": "bestiary_page",
                        "source": book_name,
                        "content": text[:8000],  # Limit content size for indexing
                        "metadata": {"page": page_num, "book": book_dir},
                    }
                )

            except Exception as e:
                print(f"Error loading {page_file}: {e}")

    return docs


def load_rollup_summaries(extracted_dir: str = "extracted") -> list[dict]:
    """Load rollup summaries (book and chapter level) for semantic search.

    These summaries help semantic search find relevant sections, then
    the page_range metadata allows drilling down to actual page content.
    """
    docs = []

    for book_dir in sorted(os.listdir(extracted_dir)):
        summary_path = os.path.join(extracted_dir, book_dir, "rollup_summaries.json")
        if not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path) as f:
                data = json.load(f)

            book_name = data.get("book", book_dir)

            # Index book-level summary
            book_summary = data.get("book_summary", {})
            if book_summary.get("summary"):
                docs.append(
                    {
                        "name": f"{book_name} - Overview",
                        "type": "book_summary",
                        "source": book_dir,
                        "content": book_summary["summary"],
                        "metadata": {
                            "book": book_name,
                            "chapter_count": book_summary.get("chapter_count", 0),
                            "total_pages": book_summary.get("total_pages", 0),
                        },
                    }
                )

            # Index chapter-level summaries
            for i, chapter in enumerate(data.get("chapter_summaries", [])):
                chapter_name = chapter.get("name", "").strip()
                page_range = chapter.get("page_range", [])

                # Generate name if empty
                if not chapter_name:
                    if page_range:
                        chapter_name = f"Pages {page_range[0]}-{page_range[1] if len(page_range) > 1 else page_range[0]}"
                    else:
                        chapter_name = f"Section {i + 1}"

                summary = chapter.get("summary", "")
                if not summary:
                    continue

                docs.append(
                    {
                        "name": f"{book_name} - {chapter_name}",
                        "type": "chapter_summary",
                        "source": book_dir,
                        "content": summary,
                        "metadata": {
                            "book": book_name,
                            "chapter": chapter_name,
                            "page_start": page_range[0] if page_range else None,
                            "page_end": page_range[1] if len(page_range) > 1 else None,
                        },
                    }
                )

        except Exception as e:
            print(f"Error loading {summary_path}: {e}")

    return docs


def main():
    """CLI entry point for search testing and index building."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast Pathfinder Search (SQLite FTS5 + Ollama embeddings)"
    )
    parser.add_argument("--index", action="store_true", help="Build/rebuild the search index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic search (requires embeddings)",
    )
    parser.add_argument("--type", type=str, help="Filter by document type")
    parser.add_argument("--source", type=str, help="Filter by source")
    parser.add_argument(
        "--edition",
        type=str,
        help="Filter by edition (pf1e, pf2e, pf2e_remaster, or 'all')",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument(
        "--with-embeddings", action="store_true", help="Also generate embeddings (slow)"
    )
    parser.add_argument("--clear", action="store_true", help="Clear all indexed data")
    args = parser.parse_args()

    search = PathfinderSearch()

    if args.clear:
        print("Clearing index...")
        search.clear()
        print("Done.")
        return

    if args.index:
        print("Building search index...")
        start = time.time()

        print("\nLoading SRD documents...")
        srd_docs = load_srd_documents()
        print(f"  Loaded {len(srd_docs)} SRD documents")

        print("\nLoading extracted documents...")
        extracted_docs = load_extracted_documents()
        print(f"  Loaded {len(extracted_docs)} extracted documents")

        print("\nLoading full text documents...")
        full_text_docs = load_full_text_documents()
        print(f"  Loaded {len(full_text_docs)} full text pages")

        print("\nLoading rollup summaries...")
        summary_docs = load_rollup_summaries()
        print(f"  Loaded {len(summary_docs)} summary documents")

        all_docs = srd_docs + extracted_docs + full_text_docs + summary_docs
        print(f"\nTotal: {len(all_docs)} documents")

        print("\nIndexing...")
        search.add_documents_batch(all_docs, add_embeddings=args.with_embeddings)

        elapsed = time.time() - start
        print(f"\nIndexing complete in {elapsed:.2f}s")

        stats = search.get_stats()
        print(f"\nIndex statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Embeddings: {stats['embeddings_count']}")
        return

    if args.stats:
        stats = search.get_stats()
        print("Index Statistics")
        print("=" * 40)
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embeddings: {stats['embeddings_count']}")
        print(f"\nBy edition:")
        for e, count in stats.get("by_edition", {}).items():
            print(f"  {e}: {count}")
        print(f"\nBy type:")
        for t, count in stats["by_type"].items():
            print(f"  {t}: {count}")
        print(f"\nTop sources:")
        for s, count in list(stats["by_source"].items())[:10]:
            print(f"  {s}: {count}")
        return

    if args.search:
        results = search.search(
            args.search,
            top_k=args.top_k,
            doc_type=args.type,
            source=args.source,
            edition=args.edition,
            use_semantic=args.semantic,
        )

        mode = "semantic" if args.semantic else "FTS5"
        print(f"\nSearch ({mode}): {args.search}")
        if args.type:
            print(f"Filter: type={args.type}")
        if args.source:
            print(f"Filter: source={args.source}")
        if args.edition:
            print(f"Filter: edition={args.edition}")
        print("-" * 40)

        if not results:
            print("No results found.")
        else:
            for i, r in enumerate(results, 1):
                edition_tag = f" [{r.get('edition', 'unknown')}]" if r.get("edition") else ""
                print(f"\n[{i}] {r['name']}{edition_tag} (score: {r['score']:.4f})")
                print(f"    Type: {r['type']}")
                print(f"    Source: {r['source']}")
                content = r["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"    {content}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
