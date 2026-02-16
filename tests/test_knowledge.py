"""Tests for NPC knowledge and memory management."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from gm_agent.storage.knowledge import KnowledgeStore, KnowledgeEntry
from gm_agent.mcp.npc_knowledge import NPCKnowledgeServer
from gm_agent.mcp.character_runner import CharacterRunnerServer
from gm_agent.storage.characters import CharacterStore
from gm_agent.storage.session import session_store
from gm_agent.models.base import Message, LLMResponse


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry schema."""

    def test_knowledge_entry_creation(self):
        """Test creating a KnowledgeEntry."""
        entry = KnowledgeEntry(
            id=1,
            character_id="npc1",
            character_name="Voz Lirayne",
            content="The mayor is corrupt.",
            knowledge_type="rumor",
            sharing_condition="trust",
            source="witnessed",
            importance=8,
            decay_rate=0.1,
            tags=["mayor", "corruption"]
        )

        assert entry.id == 1
        assert entry.character_id == "npc1"
        assert entry.content == "The mayor is corrupt."
        assert entry.knowledge_type == "rumor"
        assert entry.sharing_condition == "trust"
        assert entry.importance == 8
        assert "mayor" in entry.tags

    def test_knowledge_entry_defaults(self):
        """Test KnowledgeEntry default values."""
        entry = KnowledgeEntry(
            character_id="npc1",
            character_name="Voz",
            content="Test knowledge"
        )

        assert entry.knowledge_type == "fact"
        assert entry.sharing_condition == "free"
        assert entry.importance == 5
        assert entry.decay_rate == 0.0
        assert entry.tags == []


class TestKnowledgeStore:
    """Tests for KnowledgeStore."""

    def test_add_knowledge(self, tmp_path):
        """Test adding knowledge."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge(
            character_id="npc1",
            character_name="Voz Lirayne",
            content="The secret passage is behind the bookshelf.",
            knowledge_type="secret",
            sharing_condition="trust",
            source="witnessed",
            importance=9,
            tags=["secret", "passage"]
        )

        assert entry.id is not None
        assert entry.content == "The secret passage is behind the bookshelf."
        assert entry.knowledge_type == "secret"
        assert entry.sharing_condition == "trust"
        assert "passage" in entry.tags

    def test_query_knowledge_by_character(self, tmp_path):
        """Test querying knowledge by character."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        store.add_knowledge("npc1", "Voz", "Voz knows this.")
        store.add_knowledge("npc2", "Captain", "Captain knows this.")

        results = store.query_knowledge(character_id="npc1", limit=10)
        assert len(results) == 1
        assert results[0].character_name == "Voz"

    def test_query_knowledge_by_type(self, tmp_path):
        """Test querying knowledge by type."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        store.add_knowledge("npc1", "Voz", "A fact.", knowledge_type="fact")
        store.add_knowledge("npc1", "Voz", "A rumor.", knowledge_type="rumor")
        store.add_knowledge("npc1", "Voz", "A secret.", knowledge_type="secret")

        results = store.query_knowledge(character_id="npc1", knowledge_type="rumor", limit=10)
        assert len(results) == 1
        assert results[0].knowledge_type == "rumor"

    def test_query_knowledge_by_importance(self, tmp_path):
        """Test querying knowledge by minimum importance."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        store.add_knowledge("npc1", "Voz", "Low importance.", importance=2)
        store.add_knowledge("npc1", "Voz", "Medium importance.", importance=5)
        store.add_knowledge("npc1", "Voz", "High importance.", importance=9)

        results = store.query_knowledge(character_id="npc1", min_importance=6, limit=10)
        assert len(results) == 1
        assert results[0].importance == 9

    def test_query_knowledge_by_tags(self, tmp_path):
        """Test querying knowledge by tags."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        store.add_knowledge("npc1", "Voz", "About mayor.", tags=["mayor", "politics"])
        store.add_knowledge("npc1", "Voz", "About castle.", tags=["castle", "location"])

        results = store.query_knowledge(character_id="npc1", tags=["mayor"], limit=10)
        assert len(results) == 1
        assert "mayor" in results[0].tags

    def test_can_share_free_knowledge(self, tmp_path):
        """Test sharing free knowledge."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge(
            "npc1", "Voz", "Free knowledge.",
            sharing_condition="free"
        )

        # Free knowledge can always be shared
        assert store.can_share(entry.id, trust_level=0) is True
        assert store.can_share(entry.id, trust_level=-5) is True

    def test_can_share_never_knowledge(self, tmp_path):
        """Test never sharing knowledge."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge(
            "npc1", "Voz", "Never share this.",
            sharing_condition="never"
        )

        # Never knowledge cannot be shared
        assert store.can_share(entry.id, trust_level=5) is False
        assert store.can_share(entry.id, persuasion_dc_met=30) is False
        assert store.can_share(entry.id, under_duress=True) is False

    def test_can_share_trust_based(self, tmp_path):
        """Test trust-based sharing."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge(
            "npc1", "Voz", "Trust required.",
            sharing_condition="trust"
        )

        # Requires trust level >= 2
        assert store.can_share(entry.id, trust_level=1) is False
        assert store.can_share(entry.id, trust_level=2) is True
        assert store.can_share(entry.id, trust_level=5) is True

    def test_can_share_persuasion_dc(self, tmp_path):
        """Test persuasion DC based sharing."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge(
            "npc1", "Voz", "Persuasion needed.",
            sharing_condition="persuasion_dc_15"
        )

        # Requires persuasion DC >= 15
        assert store.can_share(entry.id, persuasion_dc_met=10) is False
        assert store.can_share(entry.id, persuasion_dc_met=15) is True
        assert store.can_share(entry.id, persuasion_dc_met=20) is True

    def test_get_shareable_knowledge(self, tmp_path):
        """Test getting shareable knowledge given conditions."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        # Add various knowledge with different sharing conditions
        store.add_knowledge("npc1", "Voz", "Free info.", sharing_condition="free")
        store.add_knowledge("npc1", "Voz", "Trusted info.", sharing_condition="trust")
        store.add_knowledge("npc1", "Voz", "Persuasion info.", sharing_condition="persuasion_dc_15")
        store.add_knowledge("npc1", "Voz", "Never share.", sharing_condition="never")

        # With no trust or persuasion, only free knowledge
        shareable = store.get_shareable_knowledge("npc1", trust_level=0, limit=10)
        assert len(shareable) == 1
        assert shareable[0].sharing_condition == "free"

        # With high trust, get free + trust
        shareable = store.get_shareable_knowledge("npc1", trust_level=3, limit=10)
        assert len(shareable) == 2

        # With persuasion DC, get free + persuasion
        shareable = store.get_shareable_knowledge("npc1", persuasion_dc_met=20, limit=10)
        assert len(shareable) == 2

        # With trust + persuasion, get all except never
        shareable = store.get_shareable_knowledge(
            "npc1", trust_level=3, persuasion_dc_met=20, limit=10
        )
        assert len(shareable) == 3

    def test_update_importance(self, tmp_path):
        """Test updating knowledge importance."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge("npc1", "Voz", "Test.", importance=5)

        # Update importance
        success = store.update_importance(entry.id, 8)
        assert success is True

        # Verify update
        loaded = store.get_by_id(entry.id)
        assert loaded.importance == 8

    def test_apply_decay(self, tmp_path):
        """Test memory decay."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        # Add knowledge with decay
        store.add_knowledge("npc1", "Voz", "Will decay.", importance=5, decay_rate=0.5)
        store.add_knowledge("npc1", "Voz", "No decay.", importance=5, decay_rate=0.0)

        # Apply 2 days of decay
        removed = store.apply_decay(days_passed=2)

        # One should be removed (5 - 0.5*2 - 0.5*2 = 3), second at (5 - 1.0*2 = 3)
        # Actually: importance = max(0, 5 - 0.5*2) = 4 for first
        # So none removed yet
        assert removed == 0

        # Apply more decay to remove it
        store.apply_decay(days_passed=10)
        remaining = store.query_knowledge(character_id="npc1", limit=10)
        # Only the non-decaying knowledge should remain
        assert len(remaining) == 1
        assert remaining[0].decay_rate == 0.0

    def test_delete_knowledge(self, tmp_path):
        """Test deleting knowledge."""
        store = KnowledgeStore("test-campaign", base_dir=tmp_path)

        entry = store.add_knowledge("npc1", "Voz", "To be deleted.")

        # Delete it
        success = store.delete(entry.id)
        assert success is True

        # Verify it's gone
        loaded = store.get_by_id(entry.id)
        assert loaded is None


class TestNPCKnowledgeServer:
    """Tests for NPCKnowledgeServer MCP tools."""

    @pytest.fixture
    def knowledge_server(self, tmp_path):
        """Create NPCKnowledgeServer with test data."""
        with patch("gm_agent.mcp.npc_knowledge.CAMPAIGNS_DIR", tmp_path):
            # Create test character
            store = CharacterStore("test-campaign", base_dir=tmp_path)
            char = store.create("Voz Lirayne", character_type="npc")

            server = NPCKnowledgeServer("test-campaign")

            # Add some knowledge
            server.knowledge.add_knowledge(
                character_id=char.id,
                character_name="Voz Lirayne",
                content="The mayor is corrupt.",
                knowledge_type="rumor",
                sharing_condition="trust",
                importance=7,
                tags=["mayor", "corruption"]
            )

            yield server

    def test_add_npc_knowledge(self, knowledge_server):
        """Test adding knowledge via MCP tool."""
        result = knowledge_server.call_tool(
            "add_npc_knowledge",
            {
                "character_name": "Voz Lirayne",
                "content": "The old church is haunted.",
                "knowledge_type": "rumor",
                "sharing_condition": "free",
                "importance": 6,
                "tags": "church,haunted"
            }
        )

        assert result.success
        assert "The old church is haunted" in result.data
        assert "rumor" in result.data

    def test_query_npc_knowledge(self, knowledge_server):
        """Test querying NPC knowledge via MCP tool."""
        result = knowledge_server.call_tool(
            "query_npc_knowledge",
            {"character_name": "Voz Lirayne", "limit": 10}
        )

        assert result.success
        assert "Voz Lirayne knows" in result.data
        assert "mayor is corrupt" in result.data

    def test_query_npc_knowledge_with_filters(self, knowledge_server):
        """Test querying with filters."""
        result = knowledge_server.call_tool(
            "query_npc_knowledge",
            {
                "character_name": "Voz Lirayne",
                "knowledge_type": "rumor",
                "min_importance": 5,
                "limit": 10
            }
        )

        assert result.success
        assert "rumor" in result.data.lower()

    def test_what_will_npc_share(self, knowledge_server):
        """Test checking what NPC will share."""
        # With no trust, trust-based knowledge won't be shared
        result = knowledge_server.call_tool(
            "what_will_npc_share",
            {
                "character_name": "Voz Lirayne",
                "trust_level": 0,
                "limit": 10
            }
        )

        assert result.success
        assert "will not share" in result.data

        # With high trust, should share
        result = knowledge_server.call_tool(
            "what_will_npc_share",
            {
                "character_name": "Voz Lirayne",
                "trust_level": 3,
                "limit": 10
            }
        )

        assert result.success
        assert "will share" in result.data
        assert "mayor is corrupt" in result.data

    def test_npc_learns(self, knowledge_server):
        """Test simplified npc_learns tool."""
        result = knowledge_server.call_tool(
            "npc_learns",
            {
                "character_name": "Voz Lirayne",
                "content": "The party helped the townsfolk.",
                "source": "the party"
            }
        )

        assert result.success
        assert "Knowledge added" in result.data


class TestCharacterRunnerKnowledgeIntegration:
    """Tests for knowledge integration in CharacterRunnerServer."""

    @pytest.fixture
    def runner_server(self, tmp_path):
        """Create CharacterRunnerServer with knowledge."""
        with patch("gm_agent.mcp.character_runner.CAMPAIGNS_DIR", tmp_path):
            # Create test character
            store = CharacterStore("test-campaign", base_dir=tmp_path)
            npc = store.create("Voz Lirayne", character_type="npc")
            npc.personality = "Witty and knowledgeable"
            npc.speech_patterns = "Direct and sarcastic"
            store.update(npc)

            # Add knowledge
            from gm_agent.storage.knowledge import KnowledgeStore
            knowledge_store = KnowledgeStore("test-campaign", base_dir=tmp_path)
            knowledge_store.add_knowledge(
                character_id=npc.id,
                character_name="Voz Lirayne",
                content="The mayor is hiding something.",
                knowledge_type="rumor",
                importance=8
            )

            # Create test session
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                session = session_store.start("test-campaign")

                # Mock LLM
                mock_llm = MagicMock()
                mock_llm.chat.return_value = LLMResponse(
                    text="Yes, the mayor is definitely suspicious.",
                    stop_reason="end_turn",
                    tool_calls=[]
                )

                server = CharacterRunnerServer("test-campaign", llm=mock_llm)

                yield server, mock_llm

    def test_npc_has_knowledge_in_context(self, runner_server):
        """Test that NPC knowledge is injected into system prompt."""
        server, mock_llm = runner_server

        # Run NPC
        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Voz Lirayne",
                "player_input": "What do you know about the mayor?"
            }
        )

        assert result.success

        # Check that LLM was called with knowledge in system prompt
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_message = messages[0].content

        # Knowledge should be in the system prompt
        assert "mayor is hiding something" in system_message


class TestKnowledgePersistence:
    """Tests for knowledge persistence."""

    def test_knowledge_persists_across_loads(self, tmp_path):
        """Test that knowledge is saved and loaded correctly."""
        with patch("gm_agent.storage.knowledge.CAMPAIGNS_DIR", tmp_path):
            # Create and save knowledge
            store1 = KnowledgeStore("test-campaign", base_dir=tmp_path)
            entry1 = store1.add_knowledge(
                character_id="npc1",
                character_name="Voz Lirayne",
                content="Secret knowledge.",
                knowledge_type="secret",
                sharing_condition="trust",
                importance=9,
                tags=["secret"]
            )
            store1.close()

            # Load in new store instance
            store2 = KnowledgeStore("test-campaign", base_dir=tmp_path)
            loaded = store2.get_by_id(entry1.id)

            assert loaded is not None
            assert loaded.content == "Secret knowledge."
            assert loaded.knowledge_type == "secret"
            assert loaded.sharing_condition == "trust"
            assert loaded.importance == 9
            assert "secret" in loaded.tags
            store2.close()


class TestPartyKnowledge:
    """Tests for party knowledge tracking tools (Phase 3)."""

    @pytest.fixture
    def party_server(self, tmp_path):
        """Create NPCKnowledgeServer for party knowledge testing."""
        with patch("gm_agent.mcp.npc_knowledge.CAMPAIGNS_DIR", tmp_path):
            server = NPCKnowledgeServer("test-campaign")
            yield server

    def test_add_party_knowledge(self, party_server):
        result = party_server.call_tool("add_party_knowledge", {
            "content": "The mayor is secretly working with the cult.",
            "source": "found letter in mayor's desk",
            "tags": "mayor,cult,main_plot",
            "importance": 8,
        })
        assert result.success
        assert "Party knowledge recorded" in result.data
        assert "mayor is secretly working" in result.data

    def test_query_party_knowledge(self, party_server):
        # Add some knowledge
        party_server.call_tool("add_party_knowledge", {
            "content": "The secret entrance is behind the waterfall.",
            "source": "told by hermit",
            "tags": "dungeon,entrance",
        })
        party_server.call_tool("add_party_knowledge", {
            "content": "Dragons are weak to cold iron.",
            "source": "library research",
            "tags": "monsters,weakness",
        })

        # Query all
        result = party_server.call_tool("query_party_knowledge", {})
        assert result.success
        assert "waterfall" in result.data
        assert "Dragons" in result.data

    def test_query_party_knowledge_with_text_filter(self, party_server):
        party_server.call_tool("add_party_knowledge", {
            "content": "The mayor has a secret room.",
            "tags": "mayor",
        })
        party_server.call_tool("add_party_knowledge", {
            "content": "The blacksmith forges magical weapons.",
            "tags": "shops",
        })

        result = party_server.call_tool("query_party_knowledge", {
            "query": "mayor",
        })
        assert result.success
        assert "mayor" in result.data
        assert "blacksmith" not in result.data

    def test_query_party_knowledge_with_tags(self, party_server):
        party_server.call_tool("add_party_knowledge", {
            "content": "Plot info.",
            "tags": "main_plot",
        })
        party_server.call_tool("add_party_knowledge", {
            "content": "Side info.",
            "tags": "side_quest",
        })

        result = party_server.call_tool("query_party_knowledge", {
            "tags": "main_plot",
        })
        assert result.success
        assert "Plot info" in result.data

    def test_query_party_knowledge_empty(self, party_server):
        result = party_server.call_tool("query_party_knowledge", {})
        assert result.success
        assert "no matching knowledge" in result.data

    def test_has_party_learned_yes(self, party_server):
        party_server.call_tool("add_party_knowledge", {
            "content": "The mayor is corrupt and takes bribes.",
            "source": "overheard conversation",
        })

        result = party_server.call_tool("has_party_learned", {
            "topic": "corrupt",
        })
        assert result.success
        assert "**Yes**" in result.data
        assert "corrupt" in result.data

    def test_has_party_learned_no(self, party_server):
        result = party_server.call_tool("has_party_learned", {
            "topic": "dragon lair location",
        })
        assert result.success
        assert "**No**" in result.data

    def test_party_knowledge_isolated_from_npc(self, party_server, tmp_path):
        """Party knowledge should not appear in NPC queries and vice versa."""
        # Add party knowledge
        party_server.call_tool("add_party_knowledge", {
            "content": "Party secret info.",
        })

        # Add NPC knowledge (need a character first)
        char_store = CharacterStore("test-campaign", base_dir=tmp_path)
        npc = char_store.create("Test NPC", character_type="npc")

        party_server.call_tool("add_npc_knowledge", {
            "character_name": "Test NPC",
            "content": "NPC secret info.",
        })

        # Query NPC knowledge - should not include party knowledge
        npc_result = party_server.call_tool("query_npc_knowledge", {
            "character_name": "Test NPC",
        })
        assert "NPC secret info" in npc_result.data
        assert "Party secret info" not in npc_result.data

        # Query party knowledge - should not include NPC knowledge
        party_result = party_server.call_tool("query_party_knowledge", {})
        assert "Party secret info" in party_result.data
        assert "NPC secret info" not in party_result.data

    def test_has_party_learned_missing_topic(self, party_server):
        result = party_server.call_tool("has_party_learned", {"topic": ""})
        assert not result.success
        assert "topic is required" in result.error

    def test_add_party_knowledge_missing_content(self, party_server):
        result = party_server.call_tool("add_party_knowledge", {})
        assert not result.success
        assert "content is required" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
