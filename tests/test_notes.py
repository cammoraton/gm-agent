"""Tests for the GM notes MCP server."""

import pytest

from gm_agent.mcp.notes import NotesServer, NotesStore, Note


class TestNotesStore:
    """Tests for the NotesStore."""

    @pytest.fixture
    def store(self):
        return NotesStore()

    def test_add_note(self, store):
        """Should add a note."""
        note = store.add("Test note content")
        assert note.id.startswith("note-")
        assert note.content == "Test note content"

    def test_add_note_with_tags(self, store):
        """Should add note with tags."""
        note = store.add("Tagged note", tags=["npc", "sandpoint"])
        assert note.tags == ["npc", "sandpoint"]

    def test_add_note_with_importance(self, store):
        """Should set importance level."""
        note = store.add("Important note", importance="critical")
        assert note.importance == "critical"

    def test_get_note(self, store):
        """Should retrieve note by ID."""
        note = store.add("Get me")
        retrieved = store.get(note.id)
        assert retrieved == note

    def test_get_missing_note(self, store):
        """Should return None for missing note."""
        assert store.get("nonexistent") is None

    def test_list_notes(self, store):
        """Should list all notes."""
        store.add("Note 1")
        store.add("Note 2")
        store.add("Note 3")

        notes = store.list()
        assert len(notes) == 3

    def test_list_by_tag(self, store):
        """Should filter by tag."""
        store.add("NPC note", tags=["npc"])
        store.add("Location note", tags=["location"])
        store.add("Both", tags=["npc", "location"])

        npc_notes = store.list(tag="npc")
        assert len(npc_notes) == 2

    def test_list_by_importance(self, store):
        """Should filter by importance."""
        store.add("Normal note")
        store.add("High note", importance="high")
        store.add("Critical note", importance="critical")

        high_notes = store.list(importance="high")
        assert len(high_notes) == 1

    def test_search_notes(self, store):
        """Should search by content."""
        store.add("The goblin attacks")
        store.add("The dragon sleeps")
        store.add("More goblins appear")

        results = store.search("goblin")
        assert len(results) == 2

    def test_search_by_tag(self, store):
        """Should search by tag."""
        store.add("Note 1", tags=["combat"])
        store.add("Note 2", tags=["social"])

        results = store.search("combat")
        assert len(results) == 1

    def test_delete_note(self, store):
        """Should delete a note."""
        note = store.add("Delete me")
        assert store.delete(note.id)
        assert store.get(note.id) is None

    def test_delete_missing_note(self, store):
        """Should return False for missing note."""
        assert not store.delete("nonexistent")

    def test_clear_notes(self, store):
        """Should clear all notes."""
        store.add("Note 1")
        store.add("Note 2")

        count = store.clear()
        assert count == 2
        assert len(store.list()) == 0

    def test_update_note(self, store):
        """Should update note content."""
        note = store.add("Original")
        updated = store.update(note.id, content="Updated")

        assert updated.content == "Updated"

    def test_update_note_tags(self, store):
        """Should update note tags."""
        note = store.add("Note", tags=["old"])
        updated = store.update(note.id, tags=["new", "tags"])

        assert updated.tags == ["new", "tags"]

    def test_update_missing_note(self, store):
        """Should return None for missing note."""
        assert store.update("nonexistent", content="Test") is None

    def test_export_import(self, store):
        """Should export and import notes."""
        store.add("Note 1", tags=["tag1"])
        store.add("Note 2", importance="high")

        exported = store.to_dict()
        assert len(exported) == 2

        new_store = NotesStore()
        new_store.from_dict(exported)
        assert len(new_store.list()) == 2


class TestNotesServer:
    """Tests for the NotesServer MCP server."""

    @pytest.fixture
    def server(self):
        server = NotesServer()
        yield server
        server.close()

    def test_list_tools(self, server):
        """Should list all notes tools."""
        tools = server.list_tools()
        tool_names = [t.name for t in tools]

        assert "add_note" in tool_names
        assert "list_notes" in tool_names
        assert "search_notes" in tool_names
        assert "delete_note" in tool_names
        assert "update_note" in tool_names
        assert "get_important_notes" in tool_names

    def test_add_note_tool(self, server):
        """Should add note via tool call."""
        result = server.call_tool("add_note", {"content": "Test note"})

        assert result.success
        assert "id" in result.data
        assert result.data["content"] == "Test note"

    def test_add_note_with_tags(self, server):
        """Should parse comma-separated tags."""
        result = server.call_tool(
            "add_note", {"content": "Tagged note", "tags": "npc, sandpoint, quest"}
        )

        assert result.success
        assert result.data["tags"] == ["npc", "sandpoint", "quest"]

    def test_add_note_with_importance(self, server):
        """Should set importance."""
        result = server.call_tool(
            "add_note", {"content": "Critical info", "importance": "critical"}
        )

        assert result.success
        assert result.data["importance"] == "critical"

    def test_list_notes_tool(self, server):
        """Should list notes."""
        server.call_tool("add_note", {"content": "Note 1"})
        server.call_tool("add_note", {"content": "Note 2"})

        result = server.call_tool("list_notes", {})

        assert result.success
        assert result.data["count"] == 2

    def test_list_notes_by_tag(self, server):
        """Should filter by tag."""
        server.call_tool("add_note", {"content": "NPC", "tags": "npc"})
        server.call_tool("add_note", {"content": "Location", "tags": "location"})

        result = server.call_tool("list_notes", {"tag": "npc"})

        assert result.success
        assert result.data["count"] == 1

    def test_search_notes_tool(self, server):
        """Should search notes."""
        server.call_tool("add_note", {"content": "The dragon attacks"})
        server.call_tool("add_note", {"content": "The goblin hides"})

        result = server.call_tool("search_notes", {"query": "dragon"})

        assert result.success
        assert result.data["count"] == 1

    def test_delete_note_tool(self, server):
        """Should delete note."""
        add_result = server.call_tool("add_note", {"content": "Delete me"})
        note_id = add_result.data["id"]

        result = server.call_tool("delete_note", {"note_id": note_id})

        assert result.success
        assert "deleted" in result.data["message"]

    def test_delete_missing_note(self, server):
        """Should return error for missing note."""
        result = server.call_tool("delete_note", {"note_id": "nonexistent"})

        assert not result.success
        assert "not found" in result.error

    def test_update_note_tool(self, server):
        """Should update note."""
        add_result = server.call_tool("add_note", {"content": "Original"})
        note_id = add_result.data["id"]

        result = server.call_tool("update_note", {"note_id": note_id, "content": "Updated"})

        assert result.success
        assert result.data["content"] == "Updated"

    def test_get_important_notes_tool(self, server):
        """Should get high and critical notes."""
        server.call_tool("add_note", {"content": "Normal"})
        server.call_tool("add_note", {"content": "High", "importance": "high"})
        server.call_tool("add_note", {"content": "Critical", "importance": "critical"})

        result = server.call_tool("get_important_notes", {})

        assert result.success
        assert result.data["critical_count"] == 1
        assert result.data["high_count"] == 1
        assert len(result.data["notes"]) == 2

    def test_unknown_tool(self, server):
        """Should return error for unknown tool."""
        result = server.call_tool("unknown_tool", {})

        assert not result.success
        assert "Unknown tool" in result.error

    def test_store_access(self, server):
        """Should provide access to underlying store."""
        assert server.store is not None
        assert isinstance(server.store, NotesStore)
