"""SQLite-based AP (Adventure Path) progression tracking.

Tracks completed encounters, explored areas, milestones, and treasure
for an Adventure Path campaign to help the GM know where the party is.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


class APProgressEntry(BaseModel):
    """A progress entry for an Adventure Path."""

    id: int | None = None
    campaign_id: str
    book: str  # e.g. "Abomination Vaults Book 1"
    entry_type: str  # encounter, area, milestone, treasure
    name: str
    completed: bool = False
    session_id: str = ""
    xp_awarded: int = 0
    notes: str = ""
    completed_at: datetime | None = None


class APProgressStore:
    """SQLite-based AP progress tracking for a campaign.

    Tracks what encounters, areas, milestones, and treasure the party
    has completed in an Adventure Path.
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.db_path = self.base_dir / campaign_id / "ap_progress.db"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ap_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                book TEXT NOT NULL,
                entry_type TEXT NOT NULL DEFAULT 'encounter',
                name TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                session_id TEXT DEFAULT '',
                xp_awarded INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ap_progress_book
                ON ap_progress(campaign_id, book);
            CREATE INDEX IF NOT EXISTS idx_ap_progress_type
                ON ap_progress(campaign_id, entry_type);
        """)
        conn.commit()

    def mark_complete(
        self,
        name: str,
        entry_type: str = "encounter",
        book: str = "",
        xp_awarded: int = 0,
        session_id: str = "",
        notes: str = "",
    ) -> APProgressEntry:
        """Mark an AP entry as completed."""
        conn = self._get_conn()
        now = datetime.now().isoformat()
        cursor = conn.execute(
            """INSERT INTO ap_progress
               (campaign_id, book, entry_type, name, completed, session_id, xp_awarded, notes, completed_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (self.campaign_id, book, entry_type, name, session_id, xp_awarded, notes, now),
        )
        conn.commit()

        return APProgressEntry(
            id=cursor.lastrowid,
            campaign_id=self.campaign_id,
            book=book,
            entry_type=entry_type,
            name=name,
            completed=True,
            session_id=session_id,
            xp_awarded=xp_awarded,
            notes=notes,
            completed_at=datetime.fromisoformat(now),
        )

    def get_progress(self, book: str = "") -> list[dict[str, Any]]:
        """Get all progress entries, optionally filtered by book."""
        conn = self._get_conn()
        if book:
            rows = conn.execute(
                "SELECT * FROM ap_progress WHERE campaign_id = ? AND book = ? ORDER BY id",
                (self.campaign_id, book),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ap_progress WHERE campaign_id = ? ORDER BY id",
                (self.campaign_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    def get_book_progress(self, book: str) -> dict[str, Any]:
        """Get summary of progress for a specific book."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT entry_type, COUNT(*) as count, SUM(xp_awarded) as total_xp "
            "FROM ap_progress WHERE campaign_id = ? AND book = ? "
            "GROUP BY entry_type",
            (self.campaign_id, book),
        ).fetchall()

        summary: dict[str, Any] = {"book": book, "types": {}, "total_entries": 0, "total_xp": 0}
        for row in rows:
            summary["types"][row["entry_type"]] = {
                "count": row["count"],
                "xp": row["total_xp"] or 0,
            }
            summary["total_entries"] += row["count"]
            summary["total_xp"] += row["total_xp"] or 0

        return summary

    def list_incomplete(self, book: str = "") -> list[dict[str, Any]]:
        """List entries that are NOT yet completed (pre-registered but not done).

        Note: most entries are only created when completed, so this is for
        entries explicitly registered as upcoming.
        """
        conn = self._get_conn()
        if book:
            rows = conn.execute(
                "SELECT * FROM ap_progress WHERE campaign_id = ? AND book = ? AND completed = 0 ORDER BY id",
                (self.campaign_id, book),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ap_progress WHERE campaign_id = ? AND completed = 0 ORDER BY id",
                (self.campaign_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    def total_xp(self) -> int:
        """Get total XP awarded across all entries."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(xp_awarded), 0) as total FROM ap_progress WHERE campaign_id = ?",
            (self.campaign_id,),
        ).fetchone()
        return row["total"]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
