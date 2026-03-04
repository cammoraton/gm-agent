"""Tests for NPCKnowledgeServer new tools: decay, update, mark_contested."""

from pathlib import Path

import pytest

from gm_agent.storage.knowledge import KnowledgeStore
from gm_agent.storage.characters import CharacterStore
from gm_agent.mcp.npc_knowledge import NPCKnowledgeServer


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    """Return an in-memory KnowledgeStore using a temp directory."""
    return KnowledgeStore("test-campaign", base_dir=tmp_path)


@pytest.fixture
def npc_server(tmp_path: Path):
    """NPCKnowledgeServer with patched CAMPAIGNS_DIR."""
    import gm_agent.mcp.npc_knowledge as nk_module
    original_dir = nk_module.CAMPAIGNS_DIR
    nk_module.CAMPAIGNS_DIR = tmp_path

    # Create a character so tests can use the server
    char_store = CharacterStore("test-campaign", base_dir=tmp_path)
    char_store.create(name="Voz Lirayne", character_type="npc")

    server = NPCKnowledgeServer("test-campaign")
    yield server, char_store

    nk_module.CAMPAIGNS_DIR = original_dir
    server.close()


# ---------------------------------------------------------------------------
# KnowledgeStore.update_knowledge() unit tests
# ---------------------------------------------------------------------------


class TestKnowledgeStoreUpdateKnowledge:
    """Direct tests for the new update_knowledge() method."""

    def test_update_importance(self, store: KnowledgeStore):
        entry = store.add_knowledge(
            character_id="char1", character_name="Alice",
            content="The mayor is corrupt.", importance=5,
        )
        success = store.update_knowledge(entry.id, importance=8)
        assert success
        updated = store.get_by_id(entry.id)
        assert updated.importance == 8

    def test_update_sharing_condition(self, store: KnowledgeStore):
        entry = store.add_knowledge(
            character_id="char1", character_name="Alice",
            content="Secret tunnel location.", sharing_condition="free",
        )
        store.update_knowledge(entry.id, sharing_condition="trust")
        updated = store.get_by_id(entry.id)
        assert updated.sharing_condition == "trust"

    def test_add_tags_merges_with_existing(self, store: KnowledgeStore):
        entry = store.add_knowledge(
            character_id="char1", character_name="Alice",
            content="Guild contact.", tags=["guild"],
        )
        store.update_knowledge(entry.id, add_tags=["important", "quest"])
        updated = store.get_by_id(entry.id)
        assert "guild" in updated.tags
        assert "important" in updated.tags
        assert "quest" in updated.tags

    def test_remove_tags(self, store: KnowledgeStore):
        entry = store.add_knowledge(
            character_id="char1", character_name="Alice",
            content="Old rumor.", tags=["rumor", "old"],
        )
        store.update_knowledge(entry.id, remove_tags=["old"])
        updated = store.get_by_id(entry.id)
        assert "old" not in updated.tags
        assert "rumor" in updated.tags

    def test_importance_clamped_to_1_10(self, store: KnowledgeStore):
        entry = store.add_knowledge(
            character_id="char1", character_name="Alice",
            content="Test.", importance=5,
        )
        store.update_knowledge(entry.id, importance=99)
        updated = store.get_by_id(entry.id)
        assert updated.importance == 10

    def test_not_found_returns_false(self, store: KnowledgeStore):
        result = store.update_knowledge(99999, importance=8)
        assert result is False

    def test_noop_update_returns_true(self, store: KnowledgeStore):
        entry = store.add_knowledge(
            character_id="char1", character_name="Alice", content="Content.",
        )
        result = store.update_knowledge(entry.id)  # No fields changed
        assert result is True


# ---------------------------------------------------------------------------
# NPCKnowledgeServer new tool tests
# ---------------------------------------------------------------------------


class TestApplyKnowledgeDecay:
    """Tests for apply_knowledge_decay tool."""

    def test_global_decay_removes_expired_entries(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")

        # Add high-decay entry (importance 2, decay_rate 1.0 → gone after 3 days)
        server.knowledge.add_knowledge(
            character_id=char.id, character_name=char.name,
            content="Old rumor about the tavern fire.",
            importance=2, decay_rate=1.0,
        )
        # Add stable entry
        server.knowledge.add_knowledge(
            character_id=char.id, character_name=char.name,
            content="The guard captain is bribable.",
            importance=8, decay_rate=0.0,
        )

        result = server.call_tool("apply_knowledge_decay", {"days_passed": 3})
        assert result.success
        assert "removed 1" in result.data.lower() or "Removed 1" in result.data

    def test_character_filtered_decay(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")

        # Create a second character
        char_store.create(name="Second NPC", character_type="npc")
        second = char_store.get_by_name("Second NPC")

        # Add decaying entry to Voz
        server.knowledge.add_knowledge(
            character_id=char.id, character_name=char.name,
            content="Something Voz will forget.", importance=2, decay_rate=1.0,
        )
        # Add stable entry to Second NPC
        server.knowledge.add_knowledge(
            character_id=second.id, character_name=second.name,
            content="Something Second NPC will keep.", importance=8, decay_rate=0.0,
        )

        result = server.call_tool("apply_knowledge_decay", {
            "days_passed": 3,
            "character_name": "Voz Lirayne",
        })
        assert result.success
        assert "Voz Lirayne" in result.data

        # Second NPC's entry should be untouched
        entries = server.knowledge.query_knowledge(character_id=second.id)
        assert len(entries) == 1

    def test_character_not_found_error(self, npc_server):
        server, _ = npc_server
        result = server.call_tool("apply_knowledge_decay", {
            "days_passed": 1,
            "character_name": "Nobody",
        })
        assert not result.success
        assert "not found" in result.error


class TestUpdateKnowledge:
    """Tests for update_knowledge MCP tool."""

    def _add_entry(self, server, char, content="Test content.", importance=5, tags=None):
        return server.knowledge.add_knowledge(
            character_id=char.id, character_name=char.name,
            content=content, importance=importance, tags=tags or [],
        )

    def test_update_importance_field(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")
        entry = self._add_entry(server, char)

        result = server.call_tool("update_knowledge", {
            "knowledge_id": entry.id,
            "importance": 9,
        })
        assert result.success
        assert "importance → 9" in result.data

    def test_update_sharing_condition(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")
        entry = self._add_entry(server, char)

        result = server.call_tool("update_knowledge", {
            "knowledge_id": entry.id,
            "sharing_condition": "trust",
        })
        assert result.success
        assert "sharing → trust" in result.data

    def test_add_tags_without_removing_existing(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")
        entry = self._add_entry(server, char, tags=["existing"])

        result = server.call_tool("update_knowledge", {
            "knowledge_id": entry.id,
            "add_tags": "new,extra",
        })
        assert result.success
        # Verify via direct DB
        updated = server.knowledge.get_by_id(entry.id)
        assert "existing" in updated.tags
        assert "new" in updated.tags

    def test_remove_tags(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")
        entry = self._add_entry(server, char, tags=["keep", "remove_me"])

        result = server.call_tool("update_knowledge", {
            "knowledge_id": entry.id,
            "remove_tags": "remove_me",
        })
        assert result.success
        updated = server.knowledge.get_by_id(entry.id)
        assert "remove_me" not in updated.tags
        assert "keep" in updated.tags

    def test_invalid_id_error(self, npc_server):
        server, _ = npc_server
        result = server.call_tool("update_knowledge", {
            "knowledge_id": 99999,
            "importance": 5,
        })
        assert not result.success
        assert "not found" in result.error


class TestMarkKnowledgeContested:
    """Tests for mark_knowledge_contested MCP tool."""

    def test_marks_importance_1_and_adds_tag(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")

        entry = server.knowledge.add_knowledge(
            character_id=char.id, character_name=char.name,
            content="The Duke was seen at the scene of the crime.",
            importance=7,
        )

        result = server.call_tool("mark_knowledge_contested", {
            "knowledge_id": entry.id,
            "reason": "New witness contradicts this account.",
        })

        assert result.success
        assert "contested" in result.data.lower()
        assert "importance → 1" in result.data

        # Verify in DB
        updated = server.knowledge.get_by_id(entry.id)
        assert updated.importance == 1
        assert "contested" in updated.tags

    def test_adds_contested_tag_to_existing_tags(self, npc_server):
        server, char_store = npc_server
        char = char_store.get_by_name("Voz Lirayne")

        entry = server.knowledge.add_knowledge(
            character_id=char.id, character_name=char.name,
            content="Rumor about the guild.", importance=6,
            tags=["guild", "rumor"],
        )

        server.call_tool("mark_knowledge_contested", {"knowledge_id": entry.id})

        updated = server.knowledge.get_by_id(entry.id)
        assert "contested" in updated.tags
        assert "guild" in updated.tags  # Existing tags preserved

    def test_invalid_id_error(self, npc_server):
        server, _ = npc_server
        result = server.call_tool("mark_knowledge_contested", {"knowledge_id": 99999})
        assert not result.success
        assert "not found" in result.error
