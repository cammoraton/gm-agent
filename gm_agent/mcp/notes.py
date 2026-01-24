"""GM Notes MCP server.

Provides tools for managing session notes:
- Add notes with optional tags
- List and search notes
- Delete notes

Notes are stored in memory during a session but can be persisted
to campaign storage.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class Note(BaseModel):
    """A single note entry."""

    id: str
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    importance: str = "normal"  # low, normal, high, critical

    def matches(self, query: str) -> bool:
        """Check if note matches a search query."""
        query = query.lower()
        return query in self.content.lower() or any(query in tag.lower() for tag in self.tags)


class NotesStore:
    """In-memory store for notes with optional persistence."""

    def __init__(self):
        self._notes: dict[str, Note] = {}
        self._counter = 0

    def add(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: str = "normal",
    ) -> Note:
        """Add a new note."""
        self._counter += 1
        note_id = f"note-{self._counter}"

        note = Note(
            id=note_id,
            content=content,
            tags=tags or [],
            importance=importance,
        )
        self._notes[note_id] = note
        return note

    def get(self, note_id: str) -> Note | None:
        """Get a note by ID."""
        return self._notes.get(note_id)

    def list(
        self,
        tag: str | None = None,
        importance: str | None = None,
    ) -> list[Note]:
        """List notes, optionally filtered by tag or importance."""
        notes = list(self._notes.values())

        if tag:
            tag = tag.lower()
            notes = [n for n in notes if any(tag in t.lower() for t in n.tags)]

        if importance:
            notes = [n for n in notes if n.importance == importance]

        return sorted(notes, key=lambda n: n.created_at, reverse=True)

    def search(self, query: str) -> list[Note]:
        """Search notes by content or tags."""
        return [n for n in self._notes.values() if n.matches(query)]

    def delete(self, note_id: str) -> bool:
        """Delete a note by ID."""
        if note_id in self._notes:
            del self._notes[note_id]
            return True
        return False

    def clear(self) -> int:
        """Clear all notes. Returns count of deleted notes."""
        count = len(self._notes)
        self._notes.clear()
        return count

    def update(
        self, note_id: str, content: str | None = None, tags: list[str] | None = None
    ) -> Note | None:
        """Update an existing note."""
        note = self._notes.get(note_id)
        if not note:
            return None

        if content is not None:
            note.content = content
        if tags is not None:
            note.tags = tags

        return note

    def to_dict(self) -> list[dict]:
        """Export notes as list of dicts for persistence."""
        return [n.model_dump(mode="json") for n in self._notes.values()]

    def from_dict(self, data: list[dict]) -> None:
        """Import notes from list of dicts."""
        for note_data in data:
            note = Note(**note_data)
            self._notes[note.id] = note
            # Update counter to avoid ID collisions
            if note.id.startswith("note-"):
                try:
                    num = int(note.id.split("-")[1])
                    self._counter = max(self._counter, num)
                except ValueError:
                    pass


class NotesServer(MCPServer):
    """MCP server for GM notes tools."""

    def __init__(self, store: NotesStore | None = None):
        self._store = store or NotesStore()
        self._tools = [
            ToolDef(
                name="add_note",
                description="Add a note to remember important information. "
                "Use tags to categorize (e.g., 'npc', 'plot', 'location', 'combat'). "
                "Set importance to 'high' or 'critical' for key information.",
                parameters=[
                    ToolParameter(
                        name="content",
                        type="string",
                        description="The note content",
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="Comma-separated tags (e.g., 'npc,sandpoint,quest')",
                        required=False,
                    ),
                    ToolParameter(
                        name="importance",
                        type="string",
                        description="Importance level: low, normal, high, or critical",
                        required=False,
                        default="normal",
                    ),
                ],
            ),
            ToolDef(
                name="list_notes",
                description="List all notes, optionally filtered by tag or importance.",
                parameters=[
                    ToolParameter(
                        name="tag",
                        type="string",
                        description="Filter by tag",
                        required=False,
                    ),
                    ToolParameter(
                        name="importance",
                        type="string",
                        description="Filter by importance level",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="search_notes",
                description="Search notes by content or tags.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search query",
                    ),
                ],
            ),
            ToolDef(
                name="delete_note",
                description="Delete a note by its ID.",
                parameters=[
                    ToolParameter(
                        name="note_id",
                        type="string",
                        description="The ID of the note to delete",
                    ),
                ],
            ),
            ToolDef(
                name="update_note",
                description="Update an existing note's content or tags.",
                parameters=[
                    ToolParameter(
                        name="note_id",
                        type="string",
                        description="The ID of the note to update",
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="New content (leave empty to keep current)",
                        required=False,
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="New comma-separated tags (leave empty to keep current)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="get_important_notes",
                description="Get all high and critical importance notes. "
                "Use this to recall key information.",
                parameters=[],
            ),
        ]

    @property
    def store(self) -> NotesStore:
        """Access the notes store."""
        return self._store

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict) -> ToolResult:
        if name == "add_note":
            return self._add_note(args)
        elif name == "list_notes":
            return self._list_notes(args)
        elif name == "search_notes":
            return self._search_notes(args)
        elif name == "delete_note":
            return self._delete_note(args)
        elif name == "update_note":
            return self._update_note(args)
        elif name == "get_important_notes":
            return self._get_important_notes(args)
        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _add_note(self, args: dict) -> ToolResult:
        try:
            content = args["content"]
            tags = []
            if args.get("tags"):
                tags = [t.strip() for t in args["tags"].split(",") if t.strip()]
            importance = args.get("importance", "normal")

            if importance not in ("low", "normal", "high", "critical"):
                importance = "normal"

            note = self._store.add(content, tags, importance)

            return ToolResult(
                success=True,
                data={
                    "id": note.id,
                    "content": note.content,
                    "tags": note.tags,
                    "importance": note.importance,
                    "message": f"Note added with ID {note.id}",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _list_notes(self, args: dict) -> ToolResult:
        try:
            tag = args.get("tag")
            importance = args.get("importance")

            notes = self._store.list(tag=tag, importance=importance)

            return ToolResult(
                success=True,
                data={
                    "count": len(notes),
                    "notes": [
                        {
                            "id": n.id,
                            "content": (
                                n.content[:100] + "..." if len(n.content) > 100 else n.content
                            ),
                            "tags": n.tags,
                            "importance": n.importance,
                            "created_at": n.created_at.isoformat(),
                        }
                        for n in notes
                    ],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _search_notes(self, args: dict) -> ToolResult:
        try:
            query = args["query"]
            notes = self._store.search(query)

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "count": len(notes),
                    "notes": [
                        {
                            "id": n.id,
                            "content": n.content,
                            "tags": n.tags,
                            "importance": n.importance,
                        }
                        for n in notes
                    ],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _delete_note(self, args: dict) -> ToolResult:
        try:
            note_id = args["note_id"]
            if self._store.delete(note_id):
                return ToolResult(
                    success=True,
                    data={
                        "message": f"Note {note_id} deleted",
                    },
                )
            else:
                return ToolResult(success=False, error=f"Note {note_id} not found")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _update_note(self, args: dict) -> ToolResult:
        try:
            note_id = args["note_id"]
            content = args.get("content")
            tags = None
            if args.get("tags"):
                tags = [t.strip() for t in args["tags"].split(",") if t.strip()]

            note = self._store.update(note_id, content=content, tags=tags)
            if note:
                return ToolResult(
                    success=True,
                    data={
                        "id": note.id,
                        "content": note.content,
                        "tags": note.tags,
                        "message": f"Note {note_id} updated",
                    },
                )
            else:
                return ToolResult(success=False, error=f"Note {note_id} not found")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_important_notes(self, args: dict) -> ToolResult:
        try:
            high_notes = self._store.list(importance="high")
            critical_notes = self._store.list(importance="critical")

            all_important = critical_notes + high_notes

            return ToolResult(
                success=True,
                data={
                    "critical_count": len(critical_notes),
                    "high_count": len(high_notes),
                    "notes": [
                        {
                            "id": n.id,
                            "content": n.content,
                            "tags": n.tags,
                            "importance": n.importance,
                        }
                        for n in all_important
                    ],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def close(self) -> None:
        """Clean up resources. No-op for in-memory notes store."""
        pass
