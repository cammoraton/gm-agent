"""Tests for NPC dialogue tracking."""

import pytest
from datetime import datetime
from unittest.mock import patch

from gm_agent.storage.dialogue import DialogueStore, DialogueEntry
from gm_agent.mcp.campaign_state import CampaignStateServer
from gm_agent.mcp.character_runner import CharacterRunnerServer
from gm_agent.storage.characters import CharacterStore
from gm_agent.storage.session import session_store
from gm_agent.models.base import Message, LLMResponse
from unittest.mock import MagicMock


class TestDialogueEntry:
    """Tests for DialogueEntry schema."""

    def test_dialogue_entry_creation(self):
        """Test creating a DialogueEntry."""
        entry = DialogueEntry(
            id=1,
            character_id="abc123",
            character_name="Voz Lirayne",
            session_id="session1",
            turn_number=5,
            content="I know about the secret passage.",
            dialogue_type="secret",
            flagged=True,
            timestamp=datetime.now()
        )

        assert entry.id == 1
        assert entry.character_id == "abc123"
        assert entry.character_name == "Voz Lirayne"
        assert entry.content == "I know about the secret passage."
        assert entry.dialogue_type == "secret"
        assert entry.flagged is True

    def test_dialogue_entry_defaults(self):
        """Test DialogueEntry default values."""
        entry = DialogueEntry(
            character_id="xyz789",
            character_name="Guard Captain",
            session_id="session1",
            content="Welcome travelers."
        )

        assert entry.dialogue_type == "statement"
        assert entry.flagged is False
        assert entry.turn_number is None
        assert isinstance(entry.timestamp, datetime)

    def test_to_searchable_text(self):
        """Test converting to searchable text."""
        entry = DialogueEntry(
            character_id="abc123",
            character_name="Voz Lirayne",
            session_id="session1",
            content="The mayor is corrupt."
        )

        text = entry.to_searchable_text()
        assert "Voz Lirayne" in text
        assert "The mayor is corrupt" in text


class TestDialogueStore:
    """Tests for DialogueStore."""

    def test_log_dialogue(self, tmp_path):
        """Test logging dialogue."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        entry = store.log_dialogue(
            character_id="npc1",
            character_name="Voz Lirayne",
            session_id="session1",
            content="I'll help you find the truth.",
            dialogue_type="promise",
            flagged=False,
            turn_number=3
        )

        assert entry.id is not None
        assert entry.character_name == "Voz Lirayne"
        assert entry.content == "I'll help you find the truth."
        assert entry.dialogue_type == "promise"

    def test_search_dialogue_by_content(self, tmp_path):
        """Test searching dialogue by content."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        store.log_dialogue(
            character_id="npc1",
            character_name="Voz Lirayne",
            session_id="session1",
            content="The mayor is hiding something."
        )
        store.log_dialogue(
            character_id="npc2",
            character_name="Guard Captain",
            session_id="session1",
            content="I trust the mayor completely."
        )

        # Search for "mayor"
        results = store.search(query="mayor", limit=10)
        assert len(results) == 2

        # Search for "hiding"
        results = store.search(query="hiding", limit=10)
        assert len(results) == 1
        assert results[0].character_name == "Voz Lirayne"

    def test_search_dialogue_by_character(self, tmp_path):
        """Test searching dialogue by character name."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        store.log_dialogue(
            character_id="npc1",
            character_name="Voz Lirayne",
            session_id="session1",
            content="First statement."
        )
        store.log_dialogue(
            character_id="npc1",
            character_name="Voz Lirayne",
            session_id="session1",
            content="Second statement."
        )
        store.log_dialogue(
            character_id="npc2",
            character_name="Guard Captain",
            session_id="session1",
            content="Captain's statement."
        )

        results = store.search(character_name="Voz Lirayne", limit=10)
        assert len(results) == 2
        for r in results:
            assert r.character_name == "Voz Lirayne"

    def test_search_dialogue_by_type(self, tmp_path):
        """Test searching dialogue by type."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="I promise to help.",
            dialogue_type="promise"
        )
        store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="I'll destroy you!",
            dialogue_type="threat"
        )
        store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="Just a normal statement.",
            dialogue_type="statement"
        )

        results = store.search(dialogue_type="threat", limit=10)
        assert len(results) == 1
        assert results[0].dialogue_type == "threat"

    def test_search_flagged_only(self, tmp_path):
        """Test searching for flagged dialogue only."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="Important revelation!",
            flagged=True
        )
        store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="Casual chat."
        )

        results = store.search(flagged_only=True, limit=10)
        assert len(results) == 1
        assert results[0].flagged is True

    def test_flag_dialogue(self, tmp_path):
        """Test flagging and unflagging dialogue."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        entry = store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="Test content."
        )

        # Flag it
        success = store.flag_dialogue(entry.id, flagged=True)
        assert success is True

        # Verify it's flagged
        loaded = store.get_by_id(entry.id)
        assert loaded.flagged is True

        # Unflag it
        success = store.flag_dialogue(entry.id, flagged=False)
        assert success is True

        # Verify it's unflagged
        loaded = store.get_by_id(entry.id)
        assert loaded.flagged is False

    def test_flag_nonexistent_dialogue(self, tmp_path):
        """Test flagging dialogue that doesn't exist."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        success = store.flag_dialogue(9999, flagged=True)
        assert success is False

    def test_get_character_dialogue(self, tmp_path):
        """Test getting dialogue for a specific character."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        store.log_dialogue(
            character_id="npc1",
            character_name="Voz Lirayne",
            session_id="session1",
            content="First."
        )
        store.log_dialogue(
            character_id="npc1",
            character_name="Voz Lirayne",
            session_id="session1",
            content="Second.",
            dialogue_type="promise"
        )
        store.log_dialogue(
            character_id="npc2",
            character_name="Other NPC",
            session_id="session1",
            content="Other."
        )

        # Get all Voz dialogue
        results = store.get_character_dialogue("Voz Lirayne", limit=10)
        assert len(results) == 2

        # Get only promises from Voz
        results = store.get_character_dialogue("Voz Lirayne", dialogue_type="promise", limit=10)
        assert len(results) == 1
        assert results[0].dialogue_type == "promise"

    def test_delete_dialogue(self, tmp_path):
        """Test deleting dialogue entries."""
        store = DialogueStore("test-campaign", base_dir=tmp_path)

        entry = store.log_dialogue(
            character_id="npc1",
            character_name="Voz",
            session_id="session1",
            content="To be deleted."
        )

        # Delete it
        success = store.delete(entry.id)
        assert success is True

        # Verify it's gone
        loaded = store.get_by_id(entry.id)
        assert loaded is None

        # Try deleting again
        success = store.delete(entry.id)
        assert success is False


class TestCampaignStateDialogueTools:
    """Tests for dialogue tools in CampaignStateServer."""

    @pytest.fixture
    def campaign_server(self, tmp_path):
        """Create a CampaignStateServer with test data."""
        with patch("gm_agent.mcp.campaign_state.CAMPAIGNS_DIR", tmp_path):
            # Create test characters
            store = CharacterStore("test-campaign", base_dir=tmp_path)
            char1 = store.create("Voz Lirayne", character_type="npc")
            char2 = store.create("Guard Captain", character_type="npc")

            # Create test session
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                session = session_store.start("test-campaign")

                server = CampaignStateServer("test-campaign")

                # Log some dialogue
                server.dialogue.log_dialogue(
                    character_id=char1.id,
                    character_name="Voz Lirayne",
                    session_id=session.id,
                    content="The mayor is corrupt.",
                    dialogue_type="rumor"
                )
                server.dialogue.log_dialogue(
                    character_id=char2.id,
                    character_name="Guard Captain",
                    session_id=session.id,
                    content="I trust the mayor.",
                    dialogue_type="statement"
                )

                yield server

    def test_search_dialogue(self, campaign_server):
        """Test searching dialogue via MCP tool."""
        result = campaign_server.call_tool(
            "search_dialogue",
            {"query": "mayor", "limit": 10}
        )

        assert result.success
        assert "Voz Lirayne" in result.data
        assert "Guard Captain" in result.data
        assert "corrupt" in result.data

    def test_search_dialogue_by_character(self, campaign_server):
        """Test searching dialogue filtered by character."""
        result = campaign_server.call_tool(
            "search_dialogue",
            {"character_name": "Voz Lirayne", "limit": 10}
        )

        assert result.success
        assert "Voz Lirayne" in result.data
        assert "Guard Captain" not in result.data

    def test_search_dialogue_by_type(self, campaign_server):
        """Test searching dialogue filtered by type."""
        result = campaign_server.call_tool(
            "search_dialogue",
            {"dialogue_type": "rumor", "limit": 10}
        )

        assert result.success
        assert "Voz Lirayne" in result.data
        assert "corrupt" in result.data
        assert "Guard Captain" not in result.data

    def test_search_dialogue_no_results(self, campaign_server):
        """Test searching with no matching results."""
        result = campaign_server.call_tool(
            "search_dialogue",
            {"query": "nonexistent", "limit": 10}
        )

        assert result.success
        assert "No dialogue found" in result.data

    def test_flag_dialogue_tool(self, campaign_server):
        """Test flagging dialogue via MCP tool."""
        # Get a dialogue ID first
        all_dialogue = campaign_server.dialogue.search(limit=1)
        dialogue_id = all_dialogue[0].id

        # Flag it
        result = campaign_server.call_tool(
            "flag_dialogue",
            {"dialogue_id": dialogue_id, "flagged": True}
        )

        assert result.success
        assert "flagged" in result.data

        # Verify it's flagged
        entry = campaign_server.dialogue.get_by_id(dialogue_id)
        assert entry.flagged is True

    def test_flag_dialogue_invalid_id(self, campaign_server):
        """Test flagging dialogue with invalid ID."""
        result = campaign_server.call_tool(
            "flag_dialogue",
            {"dialogue_id": 9999}
        )

        assert not result.success
        assert "not found" in result.error


class TestCharacterRunnerDialogueLogging:
    """Tests for automatic dialogue logging in CharacterRunnerServer."""

    @pytest.fixture
    def runner_server(self, tmp_path):
        """Create CharacterRunnerServer with mock LLM."""
        with patch("gm_agent.mcp.character_runner.CAMPAIGNS_DIR", tmp_path):
            # Create test character
            store = CharacterStore("test-campaign", base_dir=tmp_path)
            npc = store.create("Voz Lirayne", character_type="npc")
            npc.personality = "Sarcastic and witty"
            npc.speech_patterns = "Uses humor to deflect"
            npc.knowledge = ["The mayor is corrupt"]
            npc.goals = ["Expose the truth"]
            store.update(npc)

            # Create test session
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                session = session_store.start("test-campaign")

                # Mock LLM
                mock_llm = MagicMock()
                mock_llm.chat.return_value = LLMResponse(
                    text="Well, well, what do we have here?",
                    stop_reason="end_turn",
                    tool_calls=[]
                )

                server = CharacterRunnerServer("test-campaign", llm=mock_llm)

                yield server

    def test_run_npc_logs_dialogue(self, runner_server):
        """Test that running an NPC automatically logs dialogue."""
        # Run NPC
        result = runner_server.call_tool(
            "run_npc",
            {
                "npc_name": "Voz Lirayne",
                "player_input": "Hello there!"
            }
        )

        assert result.success

        # Verify dialogue was logged
        dialogue = runner_server.dialogue.search(
            character_name="Voz Lirayne",
            limit=10
        )

        assert len(dialogue) == 1
        assert dialogue[0].character_name == "Voz Lirayne"
        assert dialogue[0].content == "Well, well, what do we have here?"
        assert dialogue[0].dialogue_type == "statement"


class TestDialoguePersistence:
    """Tests for dialogue persistence across loads."""

    def test_dialogue_persists_across_loads(self, tmp_path):
        """Test that dialogue is saved and loaded correctly."""
        with patch("gm_agent.storage.dialogue.CAMPAIGNS_DIR", tmp_path):
            # Create and save dialogue
            store1 = DialogueStore("test-campaign", base_dir=tmp_path)
            entry1 = store1.log_dialogue(
                character_id="npc1",
                character_name="Voz Lirayne",
                session_id="session1",
                content="The secret is hidden in the library.",
                dialogue_type="secret",
                flagged=True
            )
            store1.close()

            # Load in new store instance
            store2 = DialogueStore("test-campaign", base_dir=tmp_path)
            loaded = store2.get_by_id(entry1.id)

            assert loaded is not None
            assert loaded.character_name == "Voz Lirayne"
            assert loaded.content == "The secret is hidden in the library."
            assert loaded.dialogue_type == "secret"
            assert loaded.flagged is True
            store2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
