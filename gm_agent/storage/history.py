"""FTS5 history index for campaign event search."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


class HistoryEvent(BaseModel):
    """An event logged in campaign history."""

    id: int | None = None
    session_id: str
    turn_number: int | None = None
    event: str
    importance: str = "session"  # session, arc, campaign
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_searchable_text(self) -> str:
        """Convert to text for FTS indexing."""
        parts = [self.event]
        if self.tags:
            parts.append(" ".join(self.tags))
        return " ".join(parts)


class HistoryIndex:
    """FTS5-based history search for a campaign."""

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.db_path = self.base_dir / campaign_id / "history.db"
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
            -- Main events table
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_number INTEGER,
                event TEXT NOT NULL,
                importance TEXT NOT NULL DEFAULT 'session',
                tags TEXT,
                timestamp TEXT NOT NULL
            );

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                event,
                tags,
                content='events',
                content_rowid='id'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                INSERT INTO events_fts(rowid, event, tags)
                VALUES (new.id, new.event, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
                INSERT INTO events_fts(events_fts, rowid, event, tags)
                VALUES ('delete', old.id, old.event, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
                INSERT INTO events_fts(events_fts, rowid, event, tags)
                VALUES ('delete', old.id, old.event, old.tags);
                INSERT INTO events_fts(rowid, event, tags)
                VALUES (new.id, new.event, new.tags);
            END;

            -- Index for importance filtering
            CREATE INDEX IF NOT EXISTS idx_events_importance ON events(importance);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            """)
        conn.commit()

    def log_event(
        self,
        session_id: str,
        event: str,
        importance: str = "session",
        tags: list[str] | None = None,
        turn_number: int | None = None,
    ) -> HistoryEvent:
        """Log an event to history."""
        conn = self._get_conn()
        tags_str = ",".join(tags) if tags else ""
        timestamp = datetime.now().isoformat()

        cursor = conn.execute(
            """
            INSERT INTO events (session_id, turn_number, event, importance, tags, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_number, event, importance, tags_str, timestamp),
        )
        conn.commit()

        return HistoryEvent(
            id=cursor.lastrowid,
            session_id=session_id,
            turn_number=turn_number,
            event=event,
            importance=importance,
            tags=tags or [],
            timestamp=datetime.fromisoformat(timestamp),
        )

    def search(
        self,
        query: str,
        importance: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[HistoryEvent]:
        """Search history using FTS5."""
        conn = self._get_conn()

        # Build query based on filters
        if query.strip():
            # Use FTS5 search
            sql = """
                SELECT e.id, e.session_id, e.turn_number, e.event,
                       e.importance, e.tags, e.timestamp
                FROM events e
                JOIN events_fts f ON e.id = f.rowid
                WHERE events_fts MATCH ?
            """
            params: list[Any] = [query]
        else:
            # No search term, just filter
            sql = """
                SELECT id, session_id, turn_number, event,
                       importance, tags, timestamp
                FROM events
                WHERE 1=1
            """
            params = []

        if importance:
            sql += " AND importance = ?"
            params.append(importance)

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            tags = row["tags"].split(",") if row["tags"] else []
            results.append(
                HistoryEvent(
                    id=row["id"],
                    session_id=row["session_id"],
                    turn_number=row["turn_number"],
                    event=row["event"],
                    importance=row["importance"],
                    tags=tags,
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
            )

        return results

    def get_recent(
        self,
        limit: int = 10,
        importance: str | None = None,
    ) -> list[HistoryEvent]:
        """Get most recent events."""
        return self.search("", importance=importance, limit=limit)

    def get_session_events(self, session_id: str) -> list[HistoryEvent]:
        """Get all events for a specific session."""
        return self.search("", session_id=session_id, limit=1000)

    def delete_session_events(self, session_id: str) -> int:
        """Delete all events for a session. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_importance = {}
        for row in conn.execute(
            "SELECT importance, COUNT(*) as count FROM events GROUP BY importance"
        ):
            by_importance[row["importance"]] = row["count"]

        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM events").fetchone()[0]

        return {
            "total_events": total,
            "by_importance": by_importance,
            "sessions_with_events": sessions,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "HistoryIndex":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
