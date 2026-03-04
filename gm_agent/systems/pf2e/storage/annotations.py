"""SQLite-based data quality annotation store.

Records issues the agent encounters with search.db content during gameplay
(truncated descriptions, wrong types, missing entities, etc.) for feedback
to the pf2e-extraction pipeline.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gm_agent.config import CAMPAIGNS_DIR

VALID_ISSUE_TYPES = {"truncated", "mistyped", "missing", "wrong_info", "search_miss"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class AnnotationEntry(BaseModel):
    """A data quality annotation."""

    id: int | None = None
    campaign_id: str
    entity_name: str
    entity_type: str = ""
    entity_id: str = ""
    book: str = ""
    page: int | None = None
    issue_type: str  # truncated|mistyped|missing|wrong_info|search_miss
    severity: str = "medium"  # low|medium|high|critical
    description: str = ""
    search_query: str = ""
    session_id: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class AnnotationStore:
    """SQLite-based data quality annotation tracking for a campaign."""

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.db_path = self.base_dir / campaign_id / "annotations.db"
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
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT '',
                entity_id TEXT DEFAULT '',
                book TEXT DEFAULT '',
                page INTEGER,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                description TEXT NOT NULL DEFAULT '',
                search_query TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_entity
                ON annotations(campaign_id, entity_name, entity_type);
            CREATE INDEX IF NOT EXISTS idx_annotations_issue
                ON annotations(campaign_id, issue_type);
        """)
        conn.commit()

    def add_annotation(
        self,
        entity_name: str,
        issue_type: str,
        entity_type: str = "",
        entity_id: str = "",
        book: str = "",
        page: int | None = None,
        severity: str = "medium",
        description: str = "",
        search_query: str = "",
        session_id: str = "",
    ) -> AnnotationEntry:
        """Add a data quality annotation with dedup.

        If an annotation with the same (entity_name, entity_type, issue_type)
        already exists, bumps severity upward and appends description.
        """
        conn = self._get_conn()

        # Check for existing annotation with same key
        existing = conn.execute(
            "SELECT id, severity, description FROM annotations "
            "WHERE campaign_id = ? AND entity_name = ? AND entity_type = ? AND issue_type = ?",
            (self.campaign_id, entity_name, entity_type, issue_type),
        ).fetchone()

        if existing:
            # Bump severity to the higher of old and new
            old_sev = SEVERITY_ORDER.get(existing["severity"], 1)
            new_sev = SEVERITY_ORDER.get(severity, 1)
            merged_severity = severity if new_sev > old_sev else existing["severity"]

            # Append description
            old_desc = existing["description"]
            if description and description not in old_desc:
                merged_desc = f"{old_desc}; {description}" if old_desc else description
            else:
                merged_desc = old_desc

            conn.execute(
                "UPDATE annotations SET severity = ?, description = ? WHERE id = ?",
                (merged_severity, merged_desc, existing["id"]),
            )
            conn.commit()

            return AnnotationEntry(
                id=existing["id"],
                campaign_id=self.campaign_id,
                entity_name=entity_name,
                entity_type=entity_type,
                entity_id=entity_id,
                book=book,
                page=page,
                issue_type=issue_type,
                severity=merged_severity,
                description=merged_desc,
                search_query=search_query,
                session_id=session_id,
            )

        # New annotation
        now = datetime.now().isoformat()
        cursor = conn.execute(
            """INSERT INTO annotations
               (campaign_id, entity_name, entity_type, entity_id, book, page,
                issue_type, severity, description, search_query, session_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.campaign_id, entity_name, entity_type, entity_id, book, page,
                issue_type, severity, description, search_query, session_id, now,
            ),
        )
        conn.commit()

        return AnnotationEntry(
            id=cursor.lastrowid,
            campaign_id=self.campaign_id,
            entity_name=entity_name,
            entity_type=entity_type,
            entity_id=entity_id,
            book=book,
            page=page,
            issue_type=issue_type,
            severity=severity,
            description=description,
            search_query=search_query,
            session_id=session_id,
            created_at=datetime.fromisoformat(now),
        )

    def list_annotations(
        self,
        issue_type: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """List annotations, optionally filtered."""
        conn = self._get_conn()
        conditions = ["campaign_id = ?"]
        params: list[Any] = [self.campaign_id]

        if issue_type:
            conditions.append("issue_type = ?")
            params.append(issue_type)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM annotations WHERE {where} ORDER BY id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Return total annotation count for this campaign."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM annotations WHERE campaign_id = ?",
            (self.campaign_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
