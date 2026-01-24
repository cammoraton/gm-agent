"""Tests for location management."""

import pytest
from unittest.mock import patch, MagicMock

from gm_agent.storage.locations import LocationStore
from gm_agent.storage.schemas import Location
from gm_agent.mcp.campaign_state import CampaignStateServer
from gm_agent.mcp.character_runner import CharacterRunnerServer
from gm_agent.storage.characters import CharacterStore
from gm_agent.storage.knowledge import KnowledgeStore
from gm_agent.storage.session import session_store
from gm_agent.models.base import Message, LLMResponse


class TestLocationSchema:
    """Tests for Location schema."""

    def test_location_creation(self):
        """Test creating a Location."""
        location = Location(
            id="old-tavern",
            campaign_id="test",
            name="The Old Tavern",
            description="A rundown establishment",
            common_knowledge=["42", "43"],
            recent_events=["Fight broke out last night"],
            isolation_level="connected",
            connected_locations=["market-square", "dock-district"],
        )

        assert location.id == "old-tavern"
        assert location.name == "The Old Tavern"
        assert len(location.common_knowledge) == 2
        assert len(location.recent_events) == 1
        assert location.isolation_level == "connected"
        assert len(location.connected_locations) == 2

    def test_location_defaults(self):
        """Test Location default values."""
        location = Location(
            id="test-location",
            campaign_id="test",
            name="Test Location"
        )

        assert location.description == ""
        assert location.common_knowledge == []
        assert location.recent_events == []
        assert location.isolation_level == "connected"
        assert location.connected_locations == []


class TestLocationStore:
    """Tests for LocationStore."""

    def test_create_location(self, tmp_path):
        """Test creating a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create(
            name="The Old Tavern",
            description="A rundown establishment",
            isolation_level="connected"
        )

        assert location.name == "The Old Tavern"
        assert location.id == "the-old-tavern"
        assert location.isolation_level == "connected"

    def test_get_location(self, tmp_path):
        """Test getting a location by ID."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        created = store.create("Test Location")
        loaded = store.get(created.id)

        assert loaded is not None
        assert loaded.name == "Test Location"

    def test_get_by_name(self, tmp_path):
        """Test getting a location by name."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        store.create("The Old Tavern")
        loaded = store.get_by_name("The Old Tavern")

        assert loaded is not None
        assert loaded.name == "The Old Tavern"

        # Case insensitive
        loaded2 = store.get_by_name("the old tavern")
        assert loaded2 is not None

    def test_update_location(self, tmp_path):
        """Test updating a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        location.description = "Updated description"
        store.update(location)

        loaded = store.get(location.id)
        assert loaded.description == "Updated description"

    def test_delete_location(self, tmp_path):
        """Test deleting a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        success = store.delete(location.id)
        assert success is True

        loaded = store.get(location.id)
        assert loaded is None

    def test_list_all(self, tmp_path):
        """Test listing all locations."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        store.create("Location One")
        store.create("Location Two")

        locations = store.list_all()
        assert len(locations) == 2

    def test_add_common_knowledge(self, tmp_path):
        """Test adding common knowledge to a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        success = store.add_common_knowledge(location.id, "42")
        assert success is True

        loaded = store.get(location.id)
        assert "42" in loaded.common_knowledge

        # Adding again should be idempotent
        store.add_common_knowledge(location.id, "42")
        loaded = store.get(location.id)
        assert loaded.common_knowledge.count("42") == 1

    def test_remove_common_knowledge(self, tmp_path):
        """Test removing common knowledge from a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        store.add_common_knowledge(location.id, "42")

        success = store.remove_common_knowledge(location.id, "42")
        assert success is True

        loaded = store.get(location.id)
        assert "42" not in loaded.common_knowledge

    def test_get_common_knowledge(self, tmp_path):
        """Test getting common knowledge for a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        store.add_common_knowledge(location.id, "42")
        store.add_common_knowledge(location.id, "43")

        knowledge = store.get_common_knowledge(location.id)
        assert len(knowledge) == 2
        assert "42" in knowledge

    def test_add_event(self, tmp_path):
        """Test adding a recent event to a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        success = store.add_event(location.id, "Fight broke out")
        assert success is True

        loaded = store.get(location.id)
        assert "Fight broke out" in loaded.recent_events

    def test_clear_events(self, tmp_path):
        """Test clearing recent events from a location."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        store.add_event(location.id, "Event 1")
        store.add_event(location.id, "Event 2")

        success = store.clear_events(location.id)
        assert success is True

        loaded = store.get(location.id)
        assert len(loaded.recent_events) == 0

    def test_connect_locations(self, tmp_path):
        """Test connecting two locations."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        loc1 = store.create("Location One")
        loc2 = store.create("Location Two")

        success = store.connect_locations(loc1.id, loc2.id)
        assert success is True

        # Check bidirectional connection
        loaded1 = store.get(loc1.id)
        loaded2 = store.get(loc2.id)

        assert loc2.id in loaded1.connected_locations
        assert loc1.id in loaded2.connected_locations

    def test_disconnect_locations(self, tmp_path):
        """Test disconnecting two locations."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        loc1 = store.create("Location One")
        loc2 = store.create("Location Two")
        store.connect_locations(loc1.id, loc2.id)

        success = store.disconnect_locations(loc1.id, loc2.id)
        assert success is True

        # Check bidirectional disconnection
        loaded1 = store.get(loc1.id)
        loaded2 = store.get(loc2.id)

        assert loc2.id not in loaded1.connected_locations
        assert loc1.id not in loaded2.connected_locations

    def test_set_isolation_level(self, tmp_path):
        """Test setting isolation level."""
        store = LocationStore("test-campaign", base_dir=tmp_path)

        location = store.create("Test Location")
        success = store.set_isolation_level(location.id, "remote")
        assert success is True

        loaded = store.get(location.id)
        assert loaded.isolation_level == "remote"


class TestCampaignStateLocationTools:
    """Tests for location tools in CampaignStateServer."""

    @pytest.fixture
    def campaign_server(self, tmp_path):
        """Create CampaignStateServer with test data."""
        with patch("gm_agent.mcp.campaign_state.CAMPAIGNS_DIR", tmp_path):
            server = CampaignStateServer("test-campaign")
            yield server

    def test_create_location(self, campaign_server):
        """Test creating a location via MCP tool."""
        result = campaign_server.call_tool(
            "create_location",
            {
                "name": "The Old Tavern",
                "description": "A rundown establishment",
                "isolation_level": "connected"
            }
        )

        assert result.success
        assert "The Old Tavern" in result.data

    def test_get_location_info(self, campaign_server):
        """Test getting location info via MCP tool."""
        # Create location first
        campaign_server.call_tool(
            "create_location",
            {
                "name": "The Old Tavern",
                "description": "A rundown establishment",
            }
        )

        result = campaign_server.call_tool(
            "get_location_info",
            {"location_name": "The Old Tavern"}
        )

        assert result.success
        assert "The Old Tavern" in result.data
        assert "rundown" in result.data

    def test_list_locations(self, campaign_server):
        """Test listing locations via MCP tool."""
        campaign_server.call_tool("create_location", {"name": "Location One"})
        campaign_server.call_tool("create_location", {"name": "Location Two"})

        result = campaign_server.call_tool("list_locations", {})

        assert result.success
        assert "Location One" in result.data
        assert "Location Two" in result.data

    def test_add_location_knowledge(self, campaign_server):
        """Test adding common knowledge to a location via MCP tool."""
        campaign_server.call_tool("create_location", {"name": "The Old Tavern"})

        result = campaign_server.call_tool(
            "add_location_knowledge",
            {
                "location_name": "The Old Tavern",
                "knowledge_id": "42"
            }
        )

        assert result.success
        assert "42" in result.data
        assert "The Old Tavern" in result.data

    def test_add_location_event(self, campaign_server):
        """Test adding a recent event to a location via MCP tool."""
        campaign_server.call_tool("create_location", {"name": "The Old Tavern"})

        result = campaign_server.call_tool(
            "add_location_event",
            {
                "location_name": "The Old Tavern",
                "event": "Fight broke out last night"
            }
        )

        assert result.success
        assert "Fight" in result.data

    def test_connect_locations(self, campaign_server):
        """Test connecting locations via MCP tool."""
        campaign_server.call_tool("create_location", {"name": "Tavern"})
        campaign_server.call_tool("create_location", {"name": "Market"})

        result = campaign_server.call_tool(
            "connect_locations",
            {
                "location1": "Tavern",
                "location2": "Market"
            }
        )

        assert result.success
        assert "Tavern" in result.data
        assert "Market" in result.data

    def test_set_scene_location(self, campaign_server, tmp_path):
        """Test setting scene location via MCP tool."""
        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            # Create a session first
            session = session_store.start("test-campaign")

            # Create location
            campaign_server.call_tool("create_location", {"name": "The Old Tavern"})

            # Set scene location
            result = campaign_server.call_tool(
                "set_scene_location",
                {"location_name": "The Old Tavern"}
            )

            assert result.success
            assert "The Old Tavern" in result.data

            # Verify scene state was updated
            updated_session = session_store.get_current("test-campaign")
            assert updated_session.scene_state.location == "The Old Tavern"
            assert updated_session.scene_state.location_id == "the-old-tavern"


class TestCharacterRunnerLocationIntegration:
    """Tests for location integration in CharacterRunnerServer."""

    @pytest.fixture
    def runner_server(self, tmp_path):
        """Create CharacterRunnerServer with location and knowledge."""
        with patch("gm_agent.mcp.character_runner.CAMPAIGNS_DIR", tmp_path):
            # Create test character
            char_store = CharacterStore("test-campaign", base_dir=tmp_path)
            npc = char_store.create("Tavern Keeper", character_type="npc")
            npc.personality = "Friendly and talkative"
            npc.speech_patterns = "Local accent"
            char_store.update(npc)

            # Create location
            location_store = LocationStore("test-campaign", base_dir=tmp_path)
            location = location_store.create(
                "The Old Tavern",
                description="A popular local establishment"
            )

            # Add location common knowledge
            knowledge_store = KnowledgeStore("test-campaign", base_dir=tmp_path)
            knowledge_entry = knowledge_store.add_knowledge(
                character_id="location",
                character_name="The Old Tavern",
                content="The mayor visits every Tuesday.",
                knowledge_type="rumor",
                importance=7
            )
            location_store.add_common_knowledge(location.id, str(knowledge_entry.id))

            # Add recent event
            location_store.add_event(location.id, "There was a fight last night")

            # Create test session with location
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                session = session_store.start("test-campaign")
                session.scene_state.location = "The Old Tavern"
                session.scene_state.location_id = "the-old-tavern"
                session_store.update_scene("test-campaign", session.scene_state)

                # Mock LLM
                mock_llm = MagicMock()
                mock_llm.chat.return_value = LLMResponse(
                    text="Aye, the mayor does come by regular-like...",
                    stop_reason="end_turn",
                    tool_calls=[]
                )

                server = CharacterRunnerServer("test-campaign", llm=mock_llm)

                yield server, mock_llm

    def test_npc_has_location_knowledge_in_context(self, runner_server):
        """Test that NPC gets location knowledge in system prompt."""
        server, mock_llm = runner_server

        # Run NPC
        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Tavern Keeper",
                "player_input": "When does the mayor visit?"
            }
        )

        assert result.success

        # Check that LLM was called with location knowledge in system prompt
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_message = messages[0].content

        # Location knowledge should be in the system prompt
        assert "LOCATION" in system_message
        assert "mayor" in system_message.lower()

    def test_npc_has_location_events_in_context(self, runner_server):
        """Test that NPC gets location events in system prompt."""
        server, mock_llm = runner_server

        # Run NPC
        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Tavern Keeper",
                "player_input": "Anything interesting happen recently?"
            }
        )

        assert result.success

        # Check system prompt has location events
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_message = messages[0].content

        assert "LOCATION" in system_message
        assert "fight" in system_message.lower()


class TestLocationPersistence:
    """Tests for location persistence."""

    def test_location_persists_across_loads(self, tmp_path):
        """Test that locations are saved and loaded correctly."""
        with patch("gm_agent.storage.locations.CAMPAIGNS_DIR", tmp_path):
            # Create and save location
            store1 = LocationStore("test-campaign", base_dir=tmp_path)
            loc1 = store1.create(
                "The Old Tavern",
                description="A popular establishment",
                isolation_level="connected"
            )
            store1.add_common_knowledge(loc1.id, "42")
            store1.add_event(loc1.id, "Fight broke out")

            # Load in new store instance
            store2 = LocationStore("test-campaign", base_dir=tmp_path)
            loaded = store2.get(loc1.id)

            assert loaded is not None
            assert loaded.name == "The Old Tavern"
            assert loaded.description == "A popular establishment"
            assert loaded.isolation_level == "connected"
            assert "42" in loaded.common_knowledge
            assert "Fight broke out" in loaded.recent_events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
