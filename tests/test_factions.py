"""Tests for faction and organization management."""

import pytest
from unittest.mock import patch, MagicMock

from gm_agent.storage.factions import FactionStore
from gm_agent.storage.schemas import Faction
from gm_agent.mcp.campaign_state import CampaignStateServer
from gm_agent.mcp.character_runner import CharacterRunnerServer
from gm_agent.storage.characters import CharacterStore
from gm_agent.storage.knowledge import KnowledgeStore
from gm_agent.storage.session import session_store
from gm_agent.models.base import Message, LLMResponse


class TestFactionSchema:
    """Tests for Faction schema."""

    def test_faction_creation(self):
        """Test creating a Faction."""
        faction = Faction(
            id="thieves-guild",
            campaign_id="test",
            name="Thieves Guild",
            description="Criminal organization",
            goals=["Control the underworld", "Steal the crown jewels"],
            resources=["Hideout", "Network of informants"],
            member_character_ids=["npc1", "npc2"],
            reputation_with_party=15,
        )

        assert faction.id == "thieves-guild"
        assert faction.name == "Thieves Guild"
        assert len(faction.goals) == 2
        assert len(faction.member_character_ids) == 2
        assert faction.reputation_with_party == 15

    def test_faction_defaults(self):
        """Test Faction default values."""
        faction = Faction(
            id="test-faction",
            campaign_id="test",
            name="Test Faction"
        )

        assert faction.description == ""
        assert faction.goals == []
        assert faction.resources == []
        assert faction.member_character_ids == []
        assert faction.reputation_with_party == 0
        assert faction.inter_faction_attitudes == {}


class TestFactionStore:
    """Tests for FactionStore."""

    def test_create_faction(self, tmp_path):
        """Test creating a faction."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create(
            name="Thieves Guild",
            description="Criminal organization",
            goals=["Control underworld"],
            resources=["Hideout", "Gold"]
        )

        assert faction.name == "Thieves Guild"
        assert faction.id == "thieves-guild"
        assert "Control underworld" in faction.goals
        assert "Hideout" in faction.resources

    def test_get_faction(self, tmp_path):
        """Test getting a faction by ID."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        created = store.create("Test Faction")
        loaded = store.get(created.id)

        assert loaded is not None
        assert loaded.name == "Test Faction"

    def test_get_by_name(self, tmp_path):
        """Test getting a faction by name."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        store.create("Thieves Guild")
        loaded = store.get_by_name("Thieves Guild")

        assert loaded is not None
        assert loaded.name == "Thieves Guild"

        # Case insensitive
        loaded2 = store.get_by_name("thieves guild")
        assert loaded2 is not None

    def test_update_faction(self, tmp_path):
        """Test updating a faction."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        faction.description = "Updated description"
        store.update(faction)

        loaded = store.get(faction.id)
        assert loaded.description == "Updated description"

    def test_delete_faction(self, tmp_path):
        """Test deleting a faction."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        success = store.delete(faction.id)
        assert success is True

        loaded = store.get(faction.id)
        assert loaded is None

    def test_list_all(self, tmp_path):
        """Test listing all factions."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        store.create("Faction One")
        store.create("Faction Two")

        factions = store.list_all()
        assert len(factions) == 2

    def test_add_member(self, tmp_path):
        """Test adding a member to a faction."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        success = store.add_member(faction.id, "npc1")
        assert success is True

        loaded = store.get(faction.id)
        assert "npc1" in loaded.member_character_ids

        # Adding again should be idempotent
        store.add_member(faction.id, "npc1")
        loaded = store.get(faction.id)
        assert loaded.member_character_ids.count("npc1") == 1

    def test_remove_member(self, tmp_path):
        """Test removing a member from a faction."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        store.add_member(faction.id, "npc1")

        success = store.remove_member(faction.id, "npc1")
        assert success is True

        loaded = store.get(faction.id)
        assert "npc1" not in loaded.member_character_ids

    def test_get_members(self, tmp_path):
        """Test getting faction members."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        store.add_member(faction.id, "npc1")
        store.add_member(faction.id, "npc2")

        members = store.get_members(faction.id)
        assert len(members) == 2
        assert "npc1" in members

    def test_update_reputation(self, tmp_path):
        """Test updating faction reputation."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        success = store.update_reputation(faction.id, 50)
        assert success is True

        loaded = store.get(faction.id)
        assert loaded.reputation_with_party == 50

        # Test clamping to -100 to +100
        store.update_reputation(faction.id, 150)
        loaded = store.get(faction.id)
        assert loaded.reputation_with_party == 100

        store.update_reputation(faction.id, -150)
        loaded = store.get(faction.id)
        assert loaded.reputation_with_party == -100

    def test_adjust_reputation(self, tmp_path):
        """Test adjusting faction reputation by delta."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        store.adjust_reputation(faction.id, 20)

        loaded = store.get(faction.id)
        assert loaded.reputation_with_party == 20

        store.adjust_reputation(faction.id, -10)
        loaded = store.get(faction.id)
        assert loaded.reputation_with_party == 10

    def test_set_inter_faction_attitude(self, tmp_path):
        """Test setting attitudes between factions."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction1 = store.create("Faction One")
        faction2 = store.create("Faction Two")

        success = store.set_inter_faction_attitude(faction1.id, faction2.id, "hostile")
        assert success is True

        loaded = store.get(faction1.id)
        assert loaded.inter_faction_attitudes[faction2.id] == "hostile"

    def test_add_shared_knowledge(self, tmp_path):
        """Test adding shared knowledge to a faction."""
        store = FactionStore("test-campaign", base_dir=tmp_path)

        faction = store.create("Test Faction")
        success = store.add_shared_knowledge(faction.id, "42")
        assert success is True

        loaded = store.get(faction.id)
        assert "42" in loaded.shared_knowledge


class TestCampaignStateFactionTools:
    """Tests for faction tools in CampaignStateServer."""

    @pytest.fixture
    def campaign_server(self, tmp_path):
        """Create CampaignStateServer with test data."""
        with patch("gm_agent.mcp.campaign_state.CAMPAIGNS_DIR", tmp_path):
            # Create test characters
            char_store = CharacterStore("test-campaign", base_dir=tmp_path)
            char1 = char_store.create("Rogue Leader", character_type="npc")
            char2 = char_store.create("Thief Member", character_type="npc")

            server = CampaignStateServer("test-campaign")

            yield server

    def test_create_faction(self, campaign_server):
        """Test creating a faction via MCP tool."""
        result = campaign_server.call_tool(
            "create_faction",
            {
                "name": "Thieves Guild",
                "description": "Criminal organization",
                "goals": "Control underworld,Steal artifacts",
                "resources": "Hideout,Gold"
            }
        )

        assert result.success
        assert "Thieves Guild" in result.data

    def test_get_faction_info(self, campaign_server):
        """Test getting faction info via MCP tool."""
        # Create faction first
        campaign_server.call_tool(
            "create_faction",
            {
                "name": "Thieves Guild",
                "goals": "Control underworld",
            }
        )

        result = campaign_server.call_tool(
            "get_faction_info",
            {"faction_name": "Thieves Guild"}
        )

        assert result.success
        assert "Thieves Guild" in result.data
        assert "Control underworld" in result.data

    def test_list_factions(self, campaign_server):
        """Test listing factions via MCP tool."""
        campaign_server.call_tool("create_faction", {"name": "Faction One"})
        campaign_server.call_tool("create_faction", {"name": "Faction Two"})

        result = campaign_server.call_tool("list_factions", {})

        assert result.success
        assert "Faction One" in result.data
        assert "Faction Two" in result.data

    def test_add_npc_to_faction(self, campaign_server):
        """Test adding NPC to faction via MCP tool."""
        campaign_server.call_tool("create_faction", {"name": "Thieves Guild"})

        result = campaign_server.call_tool(
            "add_npc_to_faction",
            {
                "character_name": "Rogue Leader",
                "faction_name": "Thieves Guild"
            }
        )

        assert result.success
        assert "Rogue Leader" in result.data
        assert "Thieves Guild" in result.data

    def test_get_faction_members(self, campaign_server):
        """Test getting faction members via MCP tool."""
        campaign_server.call_tool("create_faction", {"name": "Thieves Guild"})
        campaign_server.call_tool(
            "add_npc_to_faction",
            {"character_name": "Rogue Leader", "faction_name": "Thieves Guild"}
        )
        campaign_server.call_tool(
            "add_npc_to_faction",
            {"character_name": "Thief Member", "faction_name": "Thieves Guild"}
        )

        result = campaign_server.call_tool(
            "get_faction_members",
            {"faction_name": "Thieves Guild"}
        )

        assert result.success
        assert "Rogue Leader" in result.data
        assert "Thief Member" in result.data

    def test_update_faction_reputation(self, campaign_server):
        """Test updating faction reputation via MCP tool."""
        campaign_server.call_tool("create_faction", {"name": "Thieves Guild"})

        result = campaign_server.call_tool(
            "update_faction_reputation",
            {
                "faction_name": "Thieves Guild",
                "reputation": 50
            }
        )

        assert result.success
        assert "50" in result.data


class TestCharacterRunnerFactionIntegration:
    """Tests for faction integration in CharacterRunnerServer."""

    @pytest.fixture
    def runner_server(self, tmp_path):
        """Create CharacterRunnerServer with faction and knowledge."""
        with patch("gm_agent.mcp.character_runner.CAMPAIGNS_DIR", tmp_path):
            # Create test character
            char_store = CharacterStore("test-campaign", base_dir=tmp_path)
            npc = char_store.create("Rogue Leader", character_type="npc")
            npc.personality = "Cunning and ambitious"
            npc.speech_patterns = "Street slang"
            char_store.update(npc)

            # Create faction
            faction_store = FactionStore("test-campaign", base_dir=tmp_path)
            faction = faction_store.create(
                "Thieves Guild",
                goals=["Control the black market"]
            )

            # Add shared faction knowledge
            knowledge_store = KnowledgeStore("test-campaign", base_dir=tmp_path)
            knowledge_entry = knowledge_store.add_knowledge(
                character_id="faction",
                character_name="Thieves Guild",
                content="The guild hideout is beneath the old tavern.",
                knowledge_type="secret",
                importance=8
            )
            faction_store.add_shared_knowledge(faction.id, str(knowledge_entry.id))

            # Add NPC to faction
            faction_store.add_member(faction.id, npc.id)
            npc.faction_ids = [faction.id]
            char_store.update(npc)

            # Create test session
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                session = session_store.start("test-campaign")

                # Mock LLM
                mock_llm = MagicMock()
                mock_llm.chat.return_value = LLMResponse(
                    text="The guild has its ways...",
                    stop_reason="end_turn",
                    tool_calls=[]
                )

                server = CharacterRunnerServer("test-campaign", llm=mock_llm)

                yield server, mock_llm

    def test_npc_has_faction_knowledge_in_context(self, runner_server):
        """Test that NPC gets faction knowledge in system prompt."""
        server, mock_llm = runner_server

        # Run NPC
        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Rogue Leader",
                "player_input": "Where does the guild meet?"
            }
        )

        assert result.success

        # Check that LLM was called with faction knowledge in system prompt
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_message = messages[0].content

        # Faction knowledge should be in the system prompt
        assert "FACTION" in system_message
        assert "hideout" in system_message.lower()

    def test_npc_has_faction_goals_in_context(self, runner_server):
        """Test that NPC gets faction goals in system prompt."""
        server, mock_llm = runner_server

        # Run NPC
        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Rogue Leader",
                "player_input": "What are your plans?"
            }
        )

        assert result.success

        # Check system prompt has faction goals
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_message = messages[0].content

        assert "FACTION" in system_message
        assert "black market" in system_message.lower()


class TestFactionPersistence:
    """Tests for faction persistence."""

    def test_faction_persists_across_loads(self, tmp_path):
        """Test that factions are saved and loaded correctly."""
        with patch("gm_agent.storage.factions.CAMPAIGNS_DIR", tmp_path):
            # Create and save faction
            store1 = FactionStore("test-campaign", base_dir=tmp_path)
            faction1 = store1.create(
                "Thieves Guild",
                description="Criminal organization",
                goals=["Control underworld"],
                resources=["Hideout"]
            )
            store1.add_member(faction1.id, "npc1")
            store1.update_reputation(faction1.id, 25)

            # Load in new store instance
            store2 = FactionStore("test-campaign", base_dir=tmp_path)
            loaded = store2.get(faction1.id)

            assert loaded is not None
            assert loaded.name == "Thieves Guild"
            assert loaded.description == "Criminal organization"
            assert "Control underworld" in loaded.goals
            assert "Hideout" in loaded.resources
            assert "npc1" in loaded.member_character_ids
            assert loaded.reputation_with_party == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
