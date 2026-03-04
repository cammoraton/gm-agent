"""SQLite-backed store for the Fiction Tree.

The Fiction Tree is a hierarchical narrative index that stores
structured fiction (timelines, settlements, dungeons) from generation
games. It uses a fractal zoom model inspired by Microscope, with nodes
at different zoom levels connected in a tree structure.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


class FictionNode(BaseModel):
    """A single node in the fiction tree."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    campaign_id: str = ""
    parent_id: str | None = None  # null = root
    zoom_level: str = "era"  # era, period, event, scene, detail
    tone: str | None = None  # light, dark, ambiguous
    title: str = ""
    summary: str = ""
    content: str = ""  # Full description
    tags: list[str] = Field(default_factory=list)
    source_system: str = "manual"  # microscope, ex_novo, delve, ex_umbra, manual

    # Cross-references (IDs into other stores)
    linked_entities: dict[str, list[str]] = Field(default_factory=dict)

    # Microscope palette (inherited down tree, overridable)
    palette_yes: list[str] = Field(default_factory=list)
    palette_no: list[str] = Field(default_factory=list)

    # Ordering
    sort_order: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


VALID_ZOOM_LEVELS = ("era", "period", "event", "scene", "detail")
VALID_TONES = ("light", "dark", "ambiguous", None)


class FictionTreeStore:
    """SQLite-backed store for fiction tree nodes.

    Database layout:
        campaigns/{campaign_id}/fiction_tree.db
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        db_dir = self.base_dir / campaign_id
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "fiction_tree.db"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS fiction_nodes (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                parent_id TEXT,
                zoom_level TEXT NOT NULL DEFAULT 'era',
                tone TEXT,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                source_system TEXT NOT NULL DEFAULT 'manual',
                linked_entities TEXT NOT NULL DEFAULT '{}',
                palette_yes TEXT NOT NULL DEFAULT '[]',
                palette_no TEXT NOT NULL DEFAULT '[]',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES fiction_nodes(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_fn_parent ON fiction_nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_fn_campaign ON fiction_nodes(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_fn_zoom ON fiction_nodes(zoom_level);
            CREATE INDEX IF NOT EXISTS idx_fn_source ON fiction_nodes(source_system);
        """)

        # Standalone FTS5 virtual table (not content-linked, simpler management)
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fiction_nodes_fts
                USING fts5(node_id, title, summary, content);
            """)
        except sqlite3.OperationalError:
            pass  # FTS5 already exists

        self.conn.commit()

    def _row_to_node(self, row: sqlite3.Row) -> FictionNode:
        """Convert a database row to a FictionNode."""
        d = dict(row)
        d["tags"] = json.loads(d["tags"])
        d["linked_entities"] = json.loads(d["linked_entities"])
        d["palette_yes"] = json.loads(d["palette_yes"])
        d["palette_no"] = json.loads(d["palette_no"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return FictionNode(**d)

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add_node(self, node: FictionNode) -> FictionNode:
        """Add a new node to the fiction tree."""
        if not node.campaign_id:
            node.campaign_id = self.campaign_id
        now = datetime.now()
        node.created_at = now
        node.updated_at = now

        self.conn.execute(
            """INSERT INTO fiction_nodes
               (id, campaign_id, parent_id, zoom_level, tone, title, summary,
                content, tags, source_system, linked_entities, palette_yes,
                palette_no, sort_order, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id, node.campaign_id, node.parent_id,
                node.zoom_level, node.tone, node.title, node.summary,
                node.content, json.dumps(node.tags), node.source_system,
                json.dumps(node.linked_entities),
                json.dumps(node.palette_yes), json.dumps(node.palette_no),
                node.sort_order, now.isoformat(), now.isoformat(),
            ),
        )

        # Update FTS
        self._update_fts(node.id, node.title, node.summary, node.content)
        self.conn.commit()
        return node

    def get_node(self, node_id: str) -> FictionNode | None:
        """Get a node by ID."""
        row = self.conn.execute(
            "SELECT * FROM fiction_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(self, node_id: str, **kwargs: Any) -> FictionNode | None:
        """Update specific fields on a node."""
        node = self.get_node(node_id)
        if node is None:
            return None

        json_fields = {"tags", "linked_entities", "palette_yes", "palette_no"}
        sets = []
        params = []

        for key, value in kwargs.items():
            if not hasattr(node, key) or key in ("id", "campaign_id", "created_at"):
                continue
            if key in json_fields:
                sets.append(f"{key} = ?")
                params.append(json.dumps(value))
            else:
                sets.append(f"{key} = ?")
                params.append(value)
            setattr(node, key, value)

        if not sets:
            return node

        now = datetime.now()
        sets.append("updated_at = ?")
        params.append(now.isoformat())
        node.updated_at = now

        params.append(node_id)
        self.conn.execute(
            f"UPDATE fiction_nodes SET {', '.join(sets)} WHERE id = ?",
            params,
        )

        # Update FTS if text fields changed
        if any(k in kwargs for k in ("title", "summary", "content")):
            self._update_fts(node_id, node.title, node.summary, node.content)

        self.conn.commit()
        return node

    def delete_node(self, node_id: str, cascade: bool = False) -> bool:
        """Delete a node. If cascade, also delete all descendants."""
        node = self.get_node(node_id)
        if node is None:
            return False

        if cascade:
            self._delete_subtree(node_id)
        else:
            # Re-parent children to this node's parent
            self.conn.execute(
                "UPDATE fiction_nodes SET parent_id = ? WHERE parent_id = ?",
                (node.parent_id, node_id),
            )
            self.conn.execute("DELETE FROM fiction_nodes WHERE id = ?", (node_id,))
            self._delete_fts(node_id)

        self.conn.commit()
        return True

    def _delete_subtree(self, node_id: str) -> None:
        """Recursively delete a node and all descendants."""
        children = self.conn.execute(
            "SELECT id FROM fiction_nodes WHERE parent_id = ?", (node_id,)
        ).fetchall()
        for child in children:
            self._delete_subtree(child["id"])
        self.conn.execute("DELETE FROM fiction_nodes WHERE id = ?", (node_id,))
        self._delete_fts(node_id)

    # -------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------

    def get_children(self, parent_id: str | None = None) -> list[FictionNode]:
        """Get children of a node (or roots if parent_id is None)."""
        if parent_id is None:
            rows = self.conn.execute(
                "SELECT * FROM fiction_nodes WHERE parent_id IS NULL AND campaign_id = ? ORDER BY sort_order, created_at",
                (self.campaign_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM fiction_nodes WHERE parent_id = ? ORDER BY sort_order, created_at",
                (parent_id,),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_roots(self) -> list[FictionNode]:
        """Get all root nodes for this campaign."""
        return self.get_children(None)

    def get_ancestors(self, node_id: str) -> list[FictionNode]:
        """Get all ancestors of a node, from root to parent."""
        ancestors = []
        current = self.get_node(node_id)
        if current is None:
            return ancestors

        while current and current.parent_id:
            parent = self.get_node(current.parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent

        ancestors.reverse()
        return ancestors

    def get_siblings(self, node_id: str) -> list[FictionNode]:
        """Get siblings (nodes with same parent), including the node itself."""
        node = self.get_node(node_id)
        if node is None:
            return []
        return self.get_children(node.parent_id)

    def reorder_siblings(self, node_ids: list[str]) -> None:
        """Set sort_order for the given nodes based on list position."""
        for i, nid in enumerate(node_ids):
            self.conn.execute(
                "UPDATE fiction_nodes SET sort_order = ? WHERE id = ?",
                (i, nid),
            )
        self.conn.commit()

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    def search(self, query: str, zoom_level: str | None = None, limit: int = 20) -> list[FictionNode]:
        """Full-text search across fiction tree."""
        # Search standalone FTS table, join back to main table via node_id
        if zoom_level:
            rows = self.conn.execute(
                """SELECT fn.* FROM fiction_nodes fn
                   JOIN fiction_nodes_fts fts ON fn.id = fts.node_id
                   WHERE fiction_nodes_fts MATCH ? AND fn.zoom_level = ? AND fn.campaign_id = ?
                   ORDER BY fts.rank
                   LIMIT ?""",
                (query, zoom_level, self.campaign_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT fn.* FROM fiction_nodes fn
                   JOIN fiction_nodes_fts fts ON fn.id = fts.node_id
                   WHERE fiction_nodes_fts MATCH ? AND fn.campaign_id = ?
                   ORDER BY fts.rank
                   LIMIT ?""",
                (query, self.campaign_id, limit),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def by_tag(self, tag: str) -> list[FictionNode]:
        """Find nodes that contain a specific tag."""
        # JSON array search via LIKE
        rows = self.conn.execute(
            """SELECT * FROM fiction_nodes
               WHERE campaign_id = ? AND tags LIKE ?
               ORDER BY sort_order, created_at""",
            (self.campaign_id, f'%"{tag}"%'),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def by_source_system(self, source_system: str) -> list[FictionNode]:
        """Find nodes from a specific source system."""
        rows = self.conn.execute(
            """SELECT * FROM fiction_nodes
               WHERE campaign_id = ? AND source_system = ?
               ORDER BY sort_order, created_at""",
            (self.campaign_id, source_system),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # -------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------

    def export_subtree(self, root_id: str) -> list[dict]:
        """Export a subtree as a list of node dicts."""
        result = []

        def _collect(node_id: str):
            node = self.get_node(node_id)
            if node:
                result.append(node.model_dump(mode="json"))
                for child in self.get_children(node_id):
                    _collect(child.id)

        _collect(root_id)
        return result

    def import_subtree(self, nodes: list[dict], new_parent_id: str | None = None) -> list[FictionNode]:
        """Import a list of node dicts. Reassigns IDs and re-parents under new_parent_id."""
        id_map: dict[str, str] = {}
        imported: list[FictionNode] = []

        for node_data in nodes:
            old_id = node_data.get("id", "")
            new_id = uuid.uuid4().hex[:16]
            id_map[old_id] = new_id

        for node_data in nodes:
            old_id = node_data.get("id", "")
            old_parent = node_data.get("parent_id")

            node_data["id"] = id_map[old_id]
            node_data["campaign_id"] = self.campaign_id

            if old_parent in id_map:
                node_data["parent_id"] = id_map[old_parent]
            elif old_parent is None and new_parent_id:
                # Root of exported subtree gets re-parented
                if old_id == nodes[0].get("id", ""):
                    node_data["parent_id"] = new_parent_id
            elif new_parent_id and old_parent not in id_map:
                node_data["parent_id"] = new_parent_id

            node = FictionNode(**node_data)
            imported.append(self.add_node(node))

        return imported

    def get_tree_summary(self, root_id: str | None = None, max_depth: int = 3) -> str:
        """Generate a compact outline of the tree for context injection."""
        lines: list[str] = []

        def _render(node_id: str | None, depth: int, indent: int):
            if depth > max_depth:
                return

            children = self.get_children(node_id)
            for child in children:
                tone_marker = ""
                if child.tone == "light":
                    tone_marker = " [Light]"
                elif child.tone == "dark":
                    tone_marker = " [Dark]"

                prefix = "  " * indent
                lines.append(f"{prefix}- {child.title}{tone_marker} ({child.zoom_level})")
                if child.summary and depth < max_depth:
                    lines.append(f"{prefix}  {child.summary[:100]}")
                _render(child.id, depth + 1, indent + 1)

        if root_id:
            root = self.get_node(root_id)
            if root:
                lines.append(f"# {root.title} ({root.zoom_level})")
                if root.summary:
                    lines.append(root.summary[:200])
                _render(root_id, 1, 1)
        else:
            _render(None, 0, 0)

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # FTS helpers
    # -------------------------------------------------------------------

    def _update_fts(self, node_id: str, title: str, summary: str, content: str) -> None:
        """Update the FTS index for a node."""
        # Delete old entry and insert new (standalone FTS table keyed by node_id)
        self.conn.execute(
            "DELETE FROM fiction_nodes_fts WHERE node_id = ?", (node_id,)
        )
        self.conn.execute(
            "INSERT INTO fiction_nodes_fts(node_id, title, summary, content) VALUES(?, ?, ?, ?)",
            (node_id, title, summary, content),
        )

    def _delete_fts(self, node_id: str) -> None:
        """Delete FTS entry for a node."""
        self.conn.execute(
            "DELETE FROM fiction_nodes_fts WHERE node_id = ?", (node_id,)
        )

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
