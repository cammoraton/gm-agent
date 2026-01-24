"""Redis-backed notes storage for distributed access.

This module provides a Redis-backed implementation of NotesStore
that enables notes to be shared across multiple workers and processes.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import redis
from pydantic import BaseModel, Field

from .base import MCPServer, ToolDef, ToolParameter, ToolResult
from .notes import Note, NotesServer

logger = logging.getLogger(__name__)


class RedisNotesStore:
    """Redis-backed store for notes with distributed access.

    Notes are stored in Redis with keys prefixed by session_id,
    enabling multiple workers to share the same notes.
    """

    def __init__(
        self,
        session_id: str,
        redis_url: str | None = None,
        key_prefix: str = "gm_notes",
    ):
        """Initialize Redis notes store.

        Args:
            session_id: Unique session identifier for note isolation
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            key_prefix: Prefix for Redis keys
        """
        self.session_id = session_id
        self.key_prefix = key_prefix
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client: redis.Redis | None = None

    @property
    def _redis(self) -> redis.Redis:
        """Get Redis client (lazy initialization)."""
        if self._client is None:
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, suffix: str = "") -> str:
        """Generate Redis key for this session."""
        base = f"{self.key_prefix}:{self.session_id}"
        return f"{base}:{suffix}" if suffix else base

    def _note_key(self, note_id: str) -> str:
        """Generate key for a specific note."""
        return self._key(f"note:{note_id}")

    def _counter_key(self) -> str:
        """Generate key for the note counter."""
        return self._key("counter")

    def _index_key(self) -> str:
        """Generate key for the notes index."""
        return self._key("index")

    def add(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: str = "normal",
    ) -> Note:
        """Add a new note.

        Args:
            content: Note content
            tags: Optional list of tags
            importance: Importance level (low, normal, high, critical)

        Returns:
            The created Note
        """
        # Increment counter atomically
        counter = self._redis.incr(self._counter_key())
        note_id = f"note-{counter}"

        note = Note(
            id=note_id,
            content=content,
            tags=tags or [],
            created_at=datetime.now(),
            importance=importance,
        )

        # Store note as JSON
        note_data = note.model_dump(mode="json")
        self._redis.set(self._note_key(note_id), json.dumps(note_data))

        # Add to index
        self._redis.sadd(self._index_key(), note_id)

        return note

    def get(self, note_id: str) -> Note | None:
        """Get a note by ID.

        Args:
            note_id: The note ID

        Returns:
            Note or None if not found
        """
        data = self._redis.get(self._note_key(note_id))
        if data:
            return Note(**json.loads(data))
        return None

    def list(
        self,
        tag: str | None = None,
        importance: str | None = None,
    ) -> list[Note]:
        """List notes, optionally filtered.

        Args:
            tag: Filter by tag (case-insensitive)
            importance: Filter by importance level

        Returns:
            List of matching notes, sorted by creation time (newest first)
        """
        note_ids = self._redis.smembers(self._index_key())
        notes = []

        for note_id in note_ids:
            note = self.get(note_id)
            if note:
                # Apply filters
                if tag:
                    tag_lower = tag.lower()
                    if not any(tag_lower in t.lower() for t in note.tags):
                        continue

                if importance and note.importance != importance:
                    continue

                notes.append(note)

        # Sort by creation time (newest first)
        return sorted(notes, key=lambda n: n.created_at, reverse=True)

    def search(self, query: str) -> list[Note]:
        """Search notes by content or tags.

        Args:
            query: Search query

        Returns:
            List of matching notes
        """
        note_ids = self._redis.smembers(self._index_key())
        notes = []

        for note_id in note_ids:
            note = self.get(note_id)
            if note and note.matches(query):
                notes.append(note)

        return notes

    def delete(self, note_id: str) -> bool:
        """Delete a note by ID.

        Args:
            note_id: The note ID

        Returns:
            True if deleted, False if not found
        """
        key = self._note_key(note_id)
        if self._redis.exists(key):
            self._redis.delete(key)
            self._redis.srem(self._index_key(), note_id)
            return True
        return False

    def clear(self) -> int:
        """Clear all notes for this session.

        Returns:
            Number of notes deleted
        """
        note_ids = self._redis.smembers(self._index_key())
        count = len(note_ids)

        # Delete all note keys
        for note_id in note_ids:
            self._redis.delete(self._note_key(note_id))

        # Clear index and counter
        self._redis.delete(self._index_key())
        self._redis.delete(self._counter_key())

        return count

    def update(
        self,
        note_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> Note | None:
        """Update an existing note.

        Args:
            note_id: The note ID
            content: New content (None to keep existing)
            tags: New tags (None to keep existing)

        Returns:
            Updated Note or None if not found
        """
        note = self.get(note_id)
        if not note:
            return None

        if content is not None:
            note.content = content
        if tags is not None:
            note.tags = tags

        # Save updated note
        note_data = note.model_dump(mode="json")
        self._redis.set(self._note_key(note_id), json.dumps(note_data))

        return note

    def to_dict(self) -> list[dict]:
        """Export notes as list of dicts.

        Returns:
            List of note dicts
        """
        notes = self.list()
        return [n.model_dump(mode="json") for n in notes]

    def from_dict(self, data: list[dict]) -> None:
        """Import notes from list of dicts.

        Args:
            data: List of note dicts to import
        """
        for note_data in data:
            note = Note(**note_data)

            # Store note
            self._redis.set(self._note_key(note.id), json.dumps(note.model_dump(mode="json")))
            self._redis.sadd(self._index_key(), note.id)

            # Update counter if needed
            if note.id.startswith("note-"):
                try:
                    num = int(note.id.split("-")[1])
                    current = int(self._redis.get(self._counter_key()) or 0)
                    if num > current:
                        self._redis.set(self._counter_key(), num)
                except ValueError:
                    pass

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


class RedisNotesServer(NotesServer):
    """MCP server for notes with Redis-backed storage.

    This server uses RedisNotesStore instead of the in-memory NotesStore,
    enabling notes to be shared across workers.
    """

    def __init__(
        self,
        session_id: str,
        redis_url: str | None = None,
    ):
        """Initialize Redis notes server.

        Args:
            session_id: Session ID for note isolation
            redis_url: Optional Redis URL override
        """
        store = RedisNotesStore(session_id=session_id, redis_url=redis_url)
        super().__init__(store=store)
        self._session_id = session_id

    def close(self) -> None:
        """Clean up resources."""
        if isinstance(self._store, RedisNotesStore):
            self._store.close()
