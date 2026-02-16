"""SQLite-based treasure and loot tracking.

Tracks items acquired, distributed, and sold, plus party wealth by level.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


# PF2e expected permanent item + currency wealth per level (Table 10-10)
TREASURE_BY_LEVEL: dict[int, dict[str, float]] = {
    1: {"total": 15.0, "currency": 0.0},
    2: {"total": 30.0, "currency": 3.0},
    3: {"total": 75.0, "currency": 12.0},
    4: {"total": 140.0, "currency": 25.0},
    5: {"total": 270.0, "currency": 45.0},
    6: {"total": 450.0, "currency": 75.0},
    7: {"total": 720.0, "currency": 100.0},
    8: {"total": 1_000.0, "currency": 135.0},
    9: {"total": 1_500.0, "currency": 200.0},
    10: {"total": 2_000.0, "currency": 275.0},
    11: {"total": 2_800.0, "currency": 375.0},
    12: {"total": 4_000.0, "currency": 500.0},
    13: {"total": 6_000.0, "currency": 750.0},
    14: {"total": 9_000.0, "currency": 1_000.0},
    15: {"total": 13_000.0, "currency": 1_500.0},
    16: {"total": 20_000.0, "currency": 2_250.0},
    17: {"total": 30_000.0, "currency": 3_250.0},
    18: {"total": 45_000.0, "currency": 5_000.0},
    19: {"total": 68_000.0, "currency": 8_000.0},
    20: {"total": 100_000.0, "currency": 12_000.0},
}


class TreasureEntry(BaseModel):
    """A treasure/loot entry."""

    id: int | None = None
    campaign_id: str
    item_name: str
    value_gp: float = 0.0
    item_level: int = 0
    holder: str = "party"  # party, character name, or "sold"
    source: str = ""  # Where it came from
    session_id: str = ""
    acquired_at: datetime = Field(default_factory=datetime.now)


class TreasureStore:
    """SQLite-based treasure tracking for a campaign."""

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.db_path = self.base_dir / campaign_id / "treasure.db"
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
            CREATE TABLE IF NOT EXISTS treasure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                item_name TEXT NOT NULL,
                value_gp REAL DEFAULT 0.0,
                item_level INTEGER DEFAULT 0,
                holder TEXT DEFAULT 'party',
                source TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                acquired_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_treasure_holder
                ON treasure(campaign_id, holder);
        """)
        conn.commit()

    def add_item(
        self,
        item_name: str,
        value_gp: float = 0.0,
        item_level: int = 0,
        source: str = "",
        session_id: str = "",
    ) -> TreasureEntry:
        """Add a new treasure item to the party loot."""
        conn = self._get_conn()
        now = datetime.now().isoformat()
        cursor = conn.execute(
            """INSERT INTO treasure
               (campaign_id, item_name, value_gp, item_level, holder, source, session_id, acquired_at)
               VALUES (?, ?, ?, ?, 'party', ?, ?, ?)""",
            (self.campaign_id, item_name, value_gp, item_level, source, session_id, now),
        )
        conn.commit()

        return TreasureEntry(
            id=cursor.lastrowid,
            campaign_id=self.campaign_id,
            item_name=item_name,
            value_gp=value_gp,
            item_level=item_level,
            holder="party",
            source=source,
            session_id=session_id,
            acquired_at=datetime.fromisoformat(now),
        )

    def distribute_item(self, item_id: int, character: str) -> bool:
        """Distribute a party item to a specific character."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE treasure SET holder = ? WHERE id = ? AND campaign_id = ?",
            (character, item_id, self.campaign_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def sell_item(self, item_id: int) -> float:
        """Sell an item (mark as sold, return half value)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value_gp FROM treasure WHERE id = ? AND campaign_id = ?",
            (item_id, self.campaign_id),
        ).fetchone()

        if not row:
            return 0.0

        sale_value = row["value_gp"] / 2
        conn.execute(
            "UPDATE treasure SET holder = 'sold' WHERE id = ? AND campaign_id = ?",
            (item_id, self.campaign_id),
        )
        conn.commit()
        return sale_value

    def get_party_wealth(self) -> dict[str, Any]:
        """Get total party wealth summary."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT holder, COUNT(*) as count, COALESCE(SUM(value_gp), 0) as total_gp "
            "FROM treasure WHERE campaign_id = ? AND holder != 'sold' "
            "GROUP BY holder",
            (self.campaign_id,),
        ).fetchall()

        holders: dict[str, dict] = {}
        total_gp = 0.0
        total_items = 0
        for row in rows:
            holders[row["holder"]] = {
                "count": row["count"],
                "value_gp": row["total_gp"],
            }
            total_gp += row["total_gp"]
            total_items += row["count"]

        # Sold items
        sold_row = conn.execute(
            "SELECT COUNT(*) as count, COALESCE(SUM(value_gp), 0) as total_gp "
            "FROM treasure WHERE campaign_id = ? AND holder = 'sold'",
            (self.campaign_id,),
        ).fetchone()
        sold_value = (sold_row["total_gp"] / 2) if sold_row else 0.0

        return {
            "total_items": total_items,
            "total_value_gp": total_gp,
            "sold_income_gp": sold_value,
            "effective_wealth_gp": total_gp + sold_value,
            "holders": holders,
        }

    def get_wealth_by_level(self, party_level: int, party_size: int = 4) -> dict[str, Any]:
        """Compare party wealth to expected wealth for their level."""
        wealth = self.get_party_wealth()
        expected = TREASURE_BY_LEVEL.get(party_level, {"total": 0, "currency": 0})
        expected_total = expected["total"] * party_size

        return {
            "party_level": party_level,
            "party_size": party_size,
            "current_wealth_gp": wealth["effective_wealth_gp"],
            "expected_wealth_gp": expected_total,
            "difference_gp": wealth["effective_wealth_gp"] - expected_total,
            "percentage": (
                round(100 * wealth["effective_wealth_gp"] / expected_total, 1)
                if expected_total > 0
                else 0
            ),
        }

    def list_items(self, holder: str = "") -> list[dict[str, Any]]:
        """List all items, optionally filtered by holder."""
        conn = self._get_conn()
        if holder:
            rows = conn.execute(
                "SELECT * FROM treasure WHERE campaign_id = ? AND holder = ? ORDER BY id",
                (self.campaign_id, holder),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM treasure WHERE campaign_id = ? ORDER BY id",
                (self.campaign_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
