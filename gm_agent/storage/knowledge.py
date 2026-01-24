"""SQLite-based knowledge store for NPC memory and information."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


class KnowledgeEntry(BaseModel):
    """A piece of knowledge that a character possesses."""

    id: int | None = None
    character_id: str
    character_name: str  # For display/search
    content: str
    knowledge_type: str = "fact"  # fact, rumor, secret, witnessed_event, conversation
    sharing_condition: str = "free"  # free, trust, persuasion_dc_X, duress, never
    source: str = ""  # witnessed, told_by_{name}, learned_from_pc, etc.
    importance: int = 5  # 1-10 scale
    decay_rate: float = 0.0  # 0.0-1.0, how quickly importance decreases
    learned_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "character_id": self.character_id,
            "character_name": self.character_name,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "sharing_condition": self.sharing_condition,
            "source": self.source,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "learned_at": self.learned_at.isoformat(),
            "tags": ",".join(self.tags) if self.tags else "",
        }


class KnowledgeStore:
    """SQLite-based knowledge storage for a campaign.

    Tracks what NPCs know and under what conditions they'll share it:
    - Knowledge types: facts, rumors, secrets, witnessed events
    - Sharing conditions: free, trust-based, persuasion DC, duress, never
    - Importance and decay for memory simulation
    - Tagging for efficient queries
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.db_path = self.base_dir / campaign_id / "knowledge.db"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            -- Main knowledge table
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                content TEXT NOT NULL,
                knowledge_type TEXT NOT NULL DEFAULT 'fact',
                sharing_condition TEXT NOT NULL DEFAULT 'free',
                source TEXT,
                importance INTEGER NOT NULL DEFAULT 5,
                decay_rate REAL NOT NULL DEFAULT 0.0,
                learned_at TEXT NOT NULL,
                tags TEXT
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_knowledge_character ON knowledge(character_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(knowledge_type);
            CREATE INDEX IF NOT EXISTS idx_knowledge_importance ON knowledge(importance);
            CREATE INDEX IF NOT EXISTS idx_knowledge_sharing ON knowledge(sharing_condition);
        """)
        conn.commit()

    def add_knowledge(
        self,
        character_id: str,
        character_name: str,
        content: str,
        knowledge_type: str = "fact",
        sharing_condition: str = "free",
        source: str = "",
        importance: int = 5,
        decay_rate: float = 0.0,
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        """Add a knowledge entry for a character.

        Args:
            character_id: ID of the character who knows this
            character_name: Name of the character (for display)
            content: The knowledge content
            knowledge_type: Type of knowledge (fact, rumor, secret, etc.)
            sharing_condition: Condition for sharing (free, trust, persuasion_dc_X, never)
            source: Where the knowledge came from
            importance: 1-10 importance scale
            decay_rate: How quickly importance decreases (0.0-1.0)
            tags: List of tags for categorization

        Returns:
            The created KnowledgeEntry
        """
        conn = self._get_conn()
        learned_at = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""

        cursor = conn.execute(
            """
            INSERT INTO knowledge (character_id, character_name, content,
                                 knowledge_type, sharing_condition, source,
                                 importance, decay_rate, learned_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                character_id,
                character_name,
                content,
                knowledge_type,
                sharing_condition,
                source,
                importance,
                decay_rate,
                learned_at,
                tags_str,
            ),
        )
        conn.commit()

        return KnowledgeEntry(
            id=cursor.lastrowid,
            character_id=character_id,
            character_name=character_name,
            content=content,
            knowledge_type=knowledge_type,
            sharing_condition=sharing_condition,
            source=source,
            importance=importance,
            decay_rate=decay_rate,
            learned_at=datetime.fromisoformat(learned_at),
            tags=tags or [],
        )

    def query_knowledge(
        self,
        character_id: str | None = None,
        character_name: str | None = None,
        knowledge_type: str | None = None,
        min_importance: int | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[KnowledgeEntry]:
        """Query knowledge entries with filters.

        Args:
            character_id: Filter by character ID
            character_name: Filter by character name
            knowledge_type: Filter by knowledge type
            min_importance: Minimum importance level
            tags: Filter by tags (matches if entry has any of these tags)
            limit: Maximum results

        Returns:
            List of matching KnowledgeEntry objects
        """
        conn = self._get_conn()
        params: list[Any] = []

        sql = """
            SELECT id, character_id, character_name, content,
                   knowledge_type, sharing_condition, source,
                   importance, decay_rate, learned_at, tags
            FROM knowledge
            WHERE 1=1
        """

        if character_id:
            sql += " AND character_id = ?"
            params.append(character_id)

        if character_name:
            sql += " AND LOWER(character_name) = LOWER(?)"
            params.append(character_name)

        if knowledge_type:
            sql += " AND knowledge_type = ?"
            params.append(knowledge_type)

        if min_importance is not None:
            sql += " AND importance >= ?"
            params.append(min_importance)

        if tags:
            # Match if entry has any of the specified tags
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            sql += " AND (" + " OR ".join(tag_conditions) + ")"

        # Order by importance descending, then learned_at descending
        sql += " ORDER BY importance DESC, learned_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_by_id(self, knowledge_id: int) -> KnowledgeEntry | None:
        """Get a knowledge entry by ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, character_id, character_name, content,
                   knowledge_type, sharing_condition, source,
                   importance, decay_rate, learned_at, tags
            FROM knowledge
            WHERE id = ?
            """,
            (knowledge_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_entry(row)

    def can_share(
        self,
        knowledge_id: int,
        trust_level: int = 0,
        persuasion_dc_met: int = 0,
        under_duress: bool = False,
    ) -> bool:
        """Check if knowledge can be shared given conditions.

        Args:
            knowledge_id: ID of the knowledge entry
            trust_level: Trust level of the requester (-5 to +5)
            persuasion_dc_met: Highest persuasion DC the requester has met
            under_duress: Whether the character is under duress

        Returns:
            True if the knowledge can be shared, False otherwise
        """
        entry = self.get_by_id(knowledge_id)
        if not entry:
            return False

        condition = entry.sharing_condition

        # Free knowledge - always share
        if condition == "free":
            return True

        # Never share
        if condition == "never":
            return False

        # Trust-based sharing
        if condition == "trust":
            # Require trust level >= 2
            return trust_level >= 2

        # Persuasion DC check
        if condition.startswith("persuasion_dc_"):
            try:
                required_dc = int(condition.split("_")[-1])
                return persuasion_dc_met >= required_dc
            except (ValueError, IndexError):
                return False

        # Duress - share secrets under pressure
        if condition == "duress":
            return under_duress

        # Unknown condition - default to not sharing
        return False

    def get_shareable_knowledge(
        self,
        character_id: str,
        trust_level: int = 0,
        persuasion_dc_met: int = 0,
        under_duress: bool = False,
        knowledge_type: str | None = None,
        limit: int = 20,
    ) -> list[KnowledgeEntry]:
        """Get knowledge that can be shared given the conditions.

        Args:
            character_id: Character whose knowledge to query
            trust_level: Trust level of the requester
            persuasion_dc_met: Highest persuasion DC met
            under_duress: Whether under duress
            knowledge_type: Optional filter by knowledge type
            limit: Maximum results

        Returns:
            List of shareable KnowledgeEntry objects
        """
        # Get all knowledge for the character
        all_knowledge = self.query_knowledge(
            character_id=character_id,
            knowledge_type=knowledge_type,
            limit=1000,  # Get more initially to filter
        )

        # Filter by sharing conditions
        shareable = []
        for entry in all_knowledge:
            if self.can_share(
                entry.id,
                trust_level=trust_level,
                persuasion_dc_met=persuasion_dc_met,
                under_duress=under_duress,
            ):
                shareable.append(entry)
                if len(shareable) >= limit:
                    break

        return shareable

    def update_importance(self, knowledge_id: int, new_importance: int) -> bool:
        """Update the importance of a knowledge entry.

        Args:
            knowledge_id: ID of the knowledge entry
            new_importance: New importance value (1-10)

        Returns:
            True if successful, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE knowledge SET importance = ? WHERE id = ?",
            (max(1, min(10, new_importance)), knowledge_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def apply_decay(self, days_passed: int = 1) -> int:
        """Apply memory decay to all knowledge entries.

        Reduces importance based on decay_rate.

        Args:
            days_passed: Number of days that have passed

        Returns:
            Number of entries that decayed to importance 0 and were removed
        """
        conn = self._get_conn()

        # Reduce importance based on decay_rate
        conn.execute(
            """
            UPDATE knowledge
            SET importance = MAX(0, importance - (decay_rate * ?))
            WHERE decay_rate > 0
            """,
            (days_passed,),
        )

        # Remove entries that decayed to 0
        cursor = conn.execute("DELETE FROM knowledge WHERE importance <= 0")
        removed = cursor.rowcount

        conn.commit()
        return removed

    def delete(self, knowledge_id: int) -> bool:
        """Delete a knowledge entry.

        Args:
            knowledge_id: ID of the knowledge entry

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        conn.commit()
        return cursor.rowcount > 0

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert a database row to a KnowledgeEntry."""
        tags_str = row["tags"] or ""
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        return KnowledgeEntry(
            id=row["id"],
            character_id=row["character_id"],
            character_name=row["character_name"],
            content=row["content"],
            knowledge_type=row["knowledge_type"],
            sharing_condition=row["sharing_condition"],
            source=row["source"],
            importance=row["importance"],
            decay_rate=row["decay_rate"],
            learned_at=datetime.fromisoformat(row["learned_at"]),
            tags=tags,
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
