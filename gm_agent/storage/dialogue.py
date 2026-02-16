"""FTS5 dialogue index for NPC conversation tracking."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


class DialogueEntry(BaseModel):
    """A dialogue entry from an NPC conversation."""

    id: int | None = None
    character_id: str
    character_name: str
    session_id: str
    turn_number: int | None = None
    content: str
    dialogue_type: str = "statement"  # statement, promise, threat, lie, rumor, secret
    flagged: bool = False  # Flag important dialogue
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_searchable_text(self) -> str:
        """Convert to text for FTS indexing."""
        return f"{self.character_name} {self.content}"


class DialogueStore:
    """FTS5-based dialogue search for a campaign.

    Tracks NPC dialogue for:
    - Consistency checking (what did they say before?)
    - Contradiction detection
    - Promise/threat tracking
    - Rumor spreading
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.db_path = self.base_dir / campaign_id / "dialogue.db"
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
            -- Main dialogue table
            CREATE TABLE IF NOT EXISTS dialogue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_number INTEGER,
                content TEXT NOT NULL,
                dialogue_type TEXT NOT NULL DEFAULT 'statement',
                flagged INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL
            );

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS dialogue_fts USING fts5(
                content,
                character_name,
                content='dialogue',
                content_rowid='id'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS dialogue_ai AFTER INSERT ON dialogue BEGIN
                INSERT INTO dialogue_fts(rowid, content, character_name)
                VALUES (new.id, new.content, new.character_name);
            END;

            CREATE TRIGGER IF NOT EXISTS dialogue_ad AFTER DELETE ON dialogue BEGIN
                INSERT INTO dialogue_fts(dialogue_fts, rowid, content, character_name)
                VALUES ('delete', old.id, old.content, old.character_name);
            END;

            CREATE TRIGGER IF NOT EXISTS dialogue_au AFTER UPDATE ON dialogue BEGIN
                INSERT INTO dialogue_fts(dialogue_fts, rowid, content, character_name)
                VALUES ('delete', old.id, old.content, old.character_name);
                INSERT INTO dialogue_fts(rowid, content, character_name)
                VALUES (new.id, new.content, new.character_name);
            END;

            -- Indexes for filtering
            CREATE INDEX IF NOT EXISTS idx_dialogue_character ON dialogue(character_id);
            CREATE INDEX IF NOT EXISTS idx_dialogue_session ON dialogue(session_id);
            CREATE INDEX IF NOT EXISTS idx_dialogue_type ON dialogue(dialogue_type);
            CREATE INDEX IF NOT EXISTS idx_dialogue_flagged ON dialogue(flagged);
            CREATE INDEX IF NOT EXISTS idx_dialogue_character_name ON dialogue(character_name);
        """)
        conn.commit()

    def log_dialogue(
        self,
        character_id: str,
        character_name: str,
        session_id: str,
        content: str,
        dialogue_type: str = "statement",
        flagged: bool = False,
        turn_number: int | None = None,
    ) -> DialogueEntry:
        """Log a dialogue entry."""
        conn = self._get_conn()
        timestamp = datetime.now().isoformat()

        cursor = conn.execute(
            """
            INSERT INTO dialogue (character_id, character_name, session_id,
                                turn_number, content, dialogue_type, flagged, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                character_id,
                character_name,
                session_id,
                turn_number,
                content,
                dialogue_type,
                1 if flagged else 0,
                timestamp,
            ),
        )
        conn.commit()

        return DialogueEntry(
            id=cursor.lastrowid,
            character_id=character_id,
            character_name=character_name,
            session_id=session_id,
            turn_number=turn_number,
            content=content,
            dialogue_type=dialogue_type,
            flagged=flagged,
            timestamp=datetime.fromisoformat(timestamp),
        )

    def search(
        self,
        query: str = "",
        character_name: str | None = None,
        character_id: str | None = None,
        dialogue_type: str | None = None,
        flagged_only: bool = False,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[DialogueEntry]:
        """Search dialogue using FTS5 and filters.

        Args:
            query: FTS5 search query (searches content and character_name)
            character_name: Filter by character name (case-insensitive)
            character_id: Filter by character ID
            dialogue_type: Filter by dialogue type
            flagged_only: Only return flagged dialogue
            session_id: Filter by session ID
            limit: Maximum results

        Returns:
            List of matching DialogueEntry objects
        """
        conn = self._get_conn()
        params: list[Any] = []

        # Build query based on filters
        if query.strip():
            # Use FTS5 search
            sql = """
                SELECT d.id, d.character_id, d.character_name, d.session_id,
                       d.turn_number, d.content, d.dialogue_type, d.flagged, d.timestamp
                FROM dialogue d
                JOIN dialogue_fts f ON d.id = f.rowid
                WHERE dialogue_fts MATCH ?
            """
            params.append(query)
        else:
            # No FTS search, just filters
            sql = """
                SELECT id, character_id, character_name, session_id,
                       turn_number, content, dialogue_type, flagged, timestamp
                FROM dialogue
                WHERE 1=1
            """

        # Add filters
        if character_name:
            sql += " AND LOWER(character_name) = LOWER(?)"
            params.append(character_name)

        if character_id:
            sql += " AND character_id = ?"
            params.append(character_id)

        if dialogue_type:
            sql += " AND dialogue_type = ?"
            params.append(dialogue_type)

        if flagged_only:
            sql += " AND flagged = 1"

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        # Order by timestamp descending (most recent first)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        return [
            DialogueEntry(
                id=row["id"],
                character_id=row["character_id"],
                character_name=row["character_name"],
                session_id=row["session_id"],
                turn_number=row["turn_number"],
                content=row["content"],
                dialogue_type=row["dialogue_type"],
                flagged=bool(row["flagged"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    def flag_dialogue(self, dialogue_id: int, flagged: bool = True) -> bool:
        """Flag or unflag a dialogue entry.

        Args:
            dialogue_id: ID of the dialogue entry
            flagged: True to flag, False to unflag

        Returns:
            True if successful, False if dialogue not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE dialogue SET flagged = ? WHERE id = ?",
            (1 if flagged else 0, dialogue_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_by_id(self, dialogue_id: int) -> DialogueEntry | None:
        """Get a dialogue entry by ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, character_id, character_name, session_id,
                   turn_number, content, dialogue_type, flagged, timestamp
            FROM dialogue
            WHERE id = ?
            """,
            (dialogue_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return DialogueEntry(
            id=row["id"],
            character_id=row["character_id"],
            character_name=row["character_name"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            content=row["content"],
            dialogue_type=row["dialogue_type"],
            flagged=bool(row["flagged"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def get_character_dialogue(
        self,
        character_name: str,
        limit: int = 20,
        dialogue_type: str | None = None,
    ) -> list[DialogueEntry]:
        """Get recent dialogue for a character.

        Convenience method for common use case.

        Args:
            character_name: Name of the character
            limit: Maximum results
            dialogue_type: Optional filter by dialogue type

        Returns:
            List of DialogueEntry objects
        """
        return self.search(
            character_name=character_name,
            dialogue_type=dialogue_type,
            limit=limit,
        )

    def delete_session_dialogues(self, session_id: str) -> int:
        """Delete all dialogue entries for a session. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM dialogue WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount

    def delete(self, dialogue_id: int) -> bool:
        """Delete a dialogue entry.

        Args:
            dialogue_id: ID of the dialogue entry

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM dialogue WHERE id = ?", (dialogue_id,))
        conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
