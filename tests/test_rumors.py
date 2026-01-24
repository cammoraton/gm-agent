"""Tests for rumor propagation system."""

import pytest
from unittest.mock import patch

from gm_agent.rumor_mill import RumorEngine
from gm_agent.storage.schemas import Rumor
from gm_agent.storage.locations import LocationStore
from gm_agent.storage.characters import CharacterStore
from gm_agent.mcp.rumors import RumorsServer


class TestRumorSchema:
    """Tests for Rumor schema."""

    def test_rumor_creation(self):
        """Test creating a Rumor."""
        rumor = Rumor(
            id="rumor-001",
            campaign_id="test",
            content="The mayor is corrupt",
            source_type="pc_seeded",
            accuracy=1.0,
            distortion_rate=0.05,
            spread_rate="medium",
            current_locations=["market-square", "tavern"],
            tags=["politics", "corruption"],
        )

        assert rumor.id == "rumor-001"
        assert rumor.content == "The mayor is corrupt"
        assert rumor.accuracy == 1.0
        assert rumor.spread_rate == "medium"
        assert len(rumor.current_locations) == 2

    def test_rumor_defaults(self):
        """Test Rumor default values."""
        rumor = Rumor(
            id="rumor-001",
            campaign_id="test",
            content="Test rumor"
        )

        assert rumor.source_type == "pc_seeded"
        assert rumor.accuracy == 1.0
        assert rumor.distortion_rate == 0.05
        assert rumor.spread_rate == "medium"
        assert rumor.current_locations == []
        assert rumor.known_by_characters == []
        assert rumor.spread_history == []
        assert rumor.tags == []


class TestRumorEngine:
    """Tests for RumorEngine."""

    @pytest.fixture
    def setup_world(self, tmp_path):
        """Create a test world with connected locations."""
        # Create locations
        location_store = LocationStore("test-campaign", base_dir=tmp_path)
        tavern = location_store.create("The Old Tavern", isolation_level="connected")
        market = location_store.create("Market Square", isolation_level="connected")
        dock = location_store.create("Dock District", isolation_level="remote")
        castle = location_store.create("Castle", isolation_level="isolated")

        # Connect locations: tavern <-> market <-> dock
        location_store.connect_locations(tavern.id, market.id)
        location_store.connect_locations(market.id, dock.id)
        # Castle is isolated, not connected

        # Create some NPCs
        char_store = CharacterStore("test-campaign", base_dir=tmp_path)
        innkeeper = char_store.create("Innkeeper", character_type="npc")
        innkeeper.chattiness = 0.8
        innkeeper.is_hub = True
        char_store.update(innkeeper)

        merchant = char_store.create("Merchant", character_type="npc")
        merchant.chattiness = 0.6
        merchant.is_hub = True
        char_store.update(merchant)

        return {
            "locations": location_store,
            "characters": char_store,
            "tavern": tavern,
            "market": market,
            "dock": dock,
            "castle": castle,
        }

    def test_seed_rumor(self, tmp_path, setup_world):
        """Test seeding a new rumor."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor = engine.seed_rumor(
            content="The mayor is corrupt",
            starting_locations=[setup_world["tavern"].id],
            spread_rate="medium",
        )

        assert rumor.content == "The mayor is corrupt"
        assert rumor.accuracy == 1.0
        assert setup_world["tavern"].id in rumor.current_locations
        assert rumor.spread_rate == "medium"

    def test_seed_rumor_all_locations(self, tmp_path, setup_world):
        """Test seeding rumor at all locations (default)."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor = engine.seed_rumor(content="Dragons are returning")

        # Should be at all 4 locations
        assert len(rumor.current_locations) == 4

    def test_get_rumors_at_location(self, tmp_path, setup_world):
        """Test getting rumors at a specific location."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor1 = engine.seed_rumor(
            content="Rumor at tavern",
            starting_locations=[setup_world["tavern"].id],
        )

        rumor2 = engine.seed_rumor(
            content="Rumor at market",
            starting_locations=[setup_world["market"].id],
        )

        tavern_rumors = engine.get_rumors_at_location(setup_world["tavern"].id)
        assert len(tavern_rumors) == 1
        assert tavern_rumors[0].content == "Rumor at tavern"

        market_rumors = engine.get_rumors_at_location(setup_world["market"].id)
        assert len(market_rumors) == 1
        assert market_rumors[0].content == "Rumor at market"

    def test_propagate_rumor_connected_locations(self, tmp_path, setup_world):
        """Test rumor propagation between connected locations."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        # Seed rumor at tavern only
        rumor = engine.seed_rumor(
            content="The guards are corrupt",
            starting_locations=[setup_world["tavern"].id],
            spread_rate="fast",  # Fast spread for testing
        )

        assert len(rumor.current_locations) == 1

        # Propagate for 1 day
        stats = engine.propagate_rumors(time_delta_days=1)

        # Should spread to market (connected to tavern)
        rumor = engine.get(rumor.id)
        assert len(rumor.current_locations) > 1
        assert setup_world["market"].id in rumor.current_locations

    def test_propagate_rumor_accuracy_degradation(self, tmp_path, setup_world):
        """Test that rumor accuracy degrades as it spreads."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor = engine.seed_rumor(
            content="Test rumor",
            starting_locations=[setup_world["tavern"].id],
            spread_rate="fast",
            distortion_rate=0.1,  # High distortion for testing
        )

        initial_accuracy = rumor.accuracy
        assert initial_accuracy == 1.0

        # Propagate multiple times
        for _ in range(3):
            engine.propagate_rumors(time_delta_days=1)

        # Reload rumor
        rumor = engine.get(rumor.id)

        # Accuracy should have decreased
        assert rumor.accuracy < initial_accuracy

    def test_propagate_rumor_respects_isolation(self, tmp_path, setup_world):
        """Test that isolation levels affect propagation."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        # Seed at market, which connects to dock (remote) but not castle (isolated)
        rumor = engine.seed_rumor(
            content="Test rumor",
            starting_locations=[setup_world["market"].id],
            spread_rate="fast",
        )

        # Propagate many times
        for _ in range(10):
            engine.propagate_rumors(time_delta_days=1)

        rumor = engine.get(rumor.id)

        # Castle is isolated and not connected, so should not have rumor
        assert setup_world["castle"].id not in rumor.current_locations

    def test_spread_rate_affects_propagation(self, tmp_path, setup_world):
        """Test that spread rate affects how fast rumors propagate."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        slow_rumor = engine.seed_rumor(
            content="Slow rumor",
            starting_locations=[setup_world["tavern"].id],
            spread_rate="slow",
        )

        fast_rumor = engine.seed_rumor(
            content="Fast rumor",
            starting_locations=[setup_world["tavern"].id],
            spread_rate="fast",
        )

        # Propagate once
        engine.propagate_rumors(time_delta_days=1)

        slow_rumor = engine.get(slow_rumor.id)
        fast_rumor = engine.get(fast_rumor.id)

        # Fast rumor should spread to more locations
        assert len(fast_rumor.current_locations) >= len(slow_rumor.current_locations)

    def test_mark_character_knows_rumor(self, tmp_path, setup_world):
        """Test marking that a character knows a rumor."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor = engine.seed_rumor(content="Test rumor")

        # Get a character
        chars = setup_world["characters"].list()
        character = chars[0]

        # Mark character knows rumor
        success = engine.mark_character_knows_rumor(rumor.id, character.id)
        assert success is True

        # Verify
        rumor = engine.get(rumor.id)
        assert character.id in rumor.known_by_characters

    def test_get_character_rumors(self, tmp_path, setup_world):
        """Test getting rumors known by a character."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor1 = engine.seed_rumor(content="Rumor 1")
        rumor2 = engine.seed_rumor(content="Rumor 2")

        chars = setup_world["characters"].list()
        character = chars[0]

        # Mark character knows rumor1
        engine.mark_character_knows_rumor(rumor1.id, character.id)

        # Get character's rumors
        char_rumors = engine.get_character_rumors(character.id)
        assert len(char_rumors) == 1
        assert char_rumors[0].content == "Rumor 1"

    def test_list_all_rumors(self, tmp_path, setup_world):
        """Test listing all rumors."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        engine.seed_rumor(content="Rumor 1")
        engine.seed_rumor(content="Rumor 2")
        engine.seed_rumor(content="Rumor 3")

        all_rumors = engine.list_all()
        assert len(all_rumors) == 3

    def test_delete_rumor(self, tmp_path, setup_world):
        """Test deleting a rumor."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor = engine.seed_rumor(content="Test rumor")
        success = engine.delete(rumor.id)
        assert success is True

        loaded = engine.get(rumor.id)
        assert loaded is None

    def test_spread_history_tracking(self, tmp_path, setup_world):
        """Test that spread history is tracked."""
        engine = RumorEngine("test-campaign", base_dir=tmp_path)

        rumor = engine.seed_rumor(
            content="Test rumor",
            starting_locations=[setup_world["tavern"].id],
            spread_rate="fast",
        )

        # Propagate
        engine.propagate_rumors(time_delta_days=1)

        rumor = engine.get(rumor.id)

        # Should have spread history
        if len(rumor.current_locations) > 1:
            assert len(rumor.spread_history) > 0
            # Each history entry should have required fields
            for entry in rumor.spread_history:
                assert "timestamp" in entry
                assert "from_location" in entry
                assert "to_location" in entry
                assert "accuracy" in entry


class TestRumorsServer:
    """Tests for RumorsServer MCP tools."""

    @pytest.fixture
    def setup_server(self, tmp_path):
        """Create RumorsServer with test data."""
        with patch("gm_agent.mcp.rumors.CAMPAIGNS_DIR", tmp_path):
            # Create locations
            location_store = LocationStore("test-campaign", base_dir=tmp_path)
            tavern = location_store.create("The Old Tavern")
            market = location_store.create("Market Square")
            location_store.connect_locations(tavern.id, market.id)

            # Create character
            char_store = CharacterStore("test-campaign", base_dir=tmp_path)
            char_store.create("Test NPC", character_type="npc")

            server = RumorsServer("test-campaign")

            yield server, tavern, market

    def test_seed_rumor_tool(self, setup_server):
        """Test seed_rumor MCP tool."""
        server, tavern, market = setup_server

        result = server.call_tool(
            "seed_rumor",
            {
                "content": "The mayor is corrupt",
                "spread_rate": "medium",
                "source_type": "pc_seeded",
            }
        )

        assert result.success
        assert "The mayor is corrupt" in result.data
        assert "medium" in result.data

    def test_seed_rumor_at_specific_locations(self, setup_server):
        """Test seeding rumor at specific locations."""
        server, tavern, market = setup_server

        result = server.call_tool(
            "seed_rumor",
            {
                "content": "Test rumor",
                "starting_locations": "The Old Tavern,Market Square",
            }
        )

        assert result.success

    def test_get_rumors_at_location(self, setup_server):
        """Test get_rumors_at_location MCP tool."""
        server, tavern, market = setup_server

        # Seed a rumor
        server.call_tool(
            "seed_rumor",
            {
                "content": "Tavern rumor",
                "starting_locations": "The Old Tavern",
            }
        )

        result = server.call_tool(
            "get_rumors_at_location",
            {"location_name": "The Old Tavern"}
        )

        assert result.success
        assert "Tavern rumor" in result.data

    def test_get_character_rumors(self, setup_server):
        """Test get_character_rumors MCP tool."""
        server, tavern, market = setup_server

        # This will return empty since we haven't marked character as knowing any rumors
        result = server.call_tool(
            "get_character_rumors",
            {"character_name": "Test NPC"}
        )

        assert result.success
        assert "knows no rumors" in result.data or "Test NPC" in result.data

    def test_propagate_rumors_tool(self, setup_server):
        """Test propagate_rumors MCP tool."""
        server, tavern, market = setup_server

        # Seed a rumor
        server.call_tool(
            "seed_rumor",
            {
                "content": "Test rumor",
                "starting_locations": "The Old Tavern",
                "spread_rate": "fast",
            }
        )

        # Propagate
        result = server.call_tool(
            "propagate_rumors",
            {"days": 2}
        )

        assert result.success
        assert "2 day" in result.data

    def test_list_all_rumors_tool(self, setup_server):
        """Test list_all_rumors MCP tool."""
        server, tavern, market = setup_server

        server.call_tool("seed_rumor", {"content": "Rumor 1"})
        server.call_tool("seed_rumor", {"content": "Rumor 2"})

        result = server.call_tool("list_all_rumors", {})

        assert result.success
        assert "Rumor 1" in result.data
        assert "Rumor 2" in result.data


class TestRumorPersistence:
    """Tests for rumor persistence."""

    def test_rumor_persists_across_loads(self, tmp_path):
        """Test that rumors are saved and loaded correctly."""
        # Create and save rumor
        engine1 = RumorEngine("test-campaign", base_dir=tmp_path)

        # Create a location first
        location_store = LocationStore("test-campaign", base_dir=tmp_path)
        location = location_store.create("Test Location")

        rumor1 = engine1.seed_rumor(
            content="The king is missing",
            starting_locations=[location.id],
            spread_rate="fast",
            distortion_rate=0.1,
            tags=["politics", "mystery"],
        )

        # Load in new engine instance
        engine2 = RumorEngine("test-campaign", base_dir=tmp_path)
        loaded = engine2.get(rumor1.id)

        assert loaded is not None
        assert loaded.content == "The king is missing"
        assert loaded.spread_rate == "fast"
        assert loaded.distortion_rate == 0.1
        assert "politics" in loaded.tags
        assert location.id in loaded.current_locations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
