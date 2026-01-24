"""Tests for NPC relationship management."""

import pytest
from unittest.mock import Mock, patch

from gm_agent.storage.schemas import Relationship, CharacterProfile
from gm_agent.mcp.campaign_state import CampaignStateServer
from gm_agent.storage.characters import CharacterStore


class TestRelationshipSchema:
    """Tests for the Relationship schema."""

    def test_relationship_creation(self):
        """Test creating a Relationship."""
        rel = Relationship(
            target_character_id="abc123",
            target_name="Voz Lirayne",
            relationship_type="ally",
            attitude="friendly",
            trust_level=3,
            history="Met during the ambush",
            notes="Reliable information broker"
        )

        assert rel.target_character_id == "abc123"
        assert rel.target_name == "Voz Lirayne"
        assert rel.relationship_type == "ally"
        assert rel.attitude == "friendly"
        assert rel.trust_level == 3
        assert rel.history == "Met during the ambush"
        assert rel.notes == "Reliable information broker"

    def test_relationship_defaults(self):
        """Test Relationship default values."""
        rel = Relationship(
            target_character_id="xyz789",
            target_name="Guard Captain"
        )

        assert rel.relationship_type == "acquaintance"
        assert rel.attitude == "neutral"
        assert rel.trust_level == 0
        assert rel.history == ""
        assert rel.notes == ""

    def test_relationship_serialization(self):
        """Test Relationship serialization."""
        rel = Relationship(
            target_character_id="abc123",
            target_name="Voz Lirayne",
            relationship_type="ally",
            attitude="friendly",
            trust_level=3
        )

        data = rel.model_dump()
        assert data["target_character_id"] == "abc123"
        assert data["target_name"] == "Voz Lirayne"
        assert data["relationship_type"] == "ally"
        assert data["trust_level"] == 3


class TestCharacterProfileRelationships:
    """Tests for CharacterProfile relationships field."""

    def test_character_with_relationships(self):
        """Test CharacterProfile with relationships."""
        rel1 = Relationship(
            target_character_id="npc1",
            target_name="Ally NPC",
            relationship_type="ally"
        )
        rel2 = Relationship(
            target_character_id="npc2",
            target_name="Enemy NPC",
            relationship_type="enemy",
            attitude="hostile",
            trust_level=-4
        )

        character = CharacterProfile(
            id="char123",
            campaign_id="test-campaign",
            name="Test Character",
            relationships=[rel1, rel2]
        )

        assert len(character.relationships) == 2
        assert character.relationships[0].target_name == "Ally NPC"
        assert character.relationships[1].target_name == "Enemy NPC"
        assert character.relationships[1].trust_level == -4

    def test_character_default_empty_relationships(self):
        """Test CharacterProfile has empty relationships by default."""
        character = CharacterProfile(
            id="char123",
            campaign_id="test-campaign",
            name="Test Character"
        )

        assert character.relationships == []


class TestCampaignStateRelationshipTools:
    """Tests for relationship management tools in CampaignStateServer."""

    @pytest.fixture
    def campaign_server(self, tmp_path):
        """Create a CampaignStateServer with test data."""
        with patch("gm_agent.mcp.campaign_state.CAMPAIGNS_DIR", tmp_path):
            # Create test characters
            store = CharacterStore("test-campaign", base_dir=tmp_path)
            char1 = store.create("Voz Lirayne", character_type="npc")
            char2 = store.create("Guard Captain", character_type="npc")
            char3 = store.create("Goblin Chief", character_type="monster")

            server = CampaignStateServer("test-campaign")
            # Inject the character store with our test data
            server._character_store = store

            yield server

    def test_add_relationship(self, campaign_server):
        """Test adding a relationship."""
        result = campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "relationship_type": "ally",
                "attitude": "friendly",
                "trust_level": 2,
                "history": "Worked together on investigation",
            }
        )

        assert result.success
        assert "added" in result.data
        assert "Voz Lirayne -> Guard Captain" in result.data
        assert "ally" in result.data

        # Verify relationship was saved
        voz = campaign_server.characters.get_by_name("Voz Lirayne")
        assert len(voz.relationships) == 1
        assert voz.relationships[0].target_name == "Guard Captain"
        assert voz.relationships[0].relationship_type == "ally"
        assert voz.relationships[0].trust_level == 2

    def test_update_existing_relationship(self, campaign_server):
        """Test updating an existing relationship."""
        # Add initial relationship
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "relationship_type": "acquaintance",
                "trust_level": 0,
            }
        )

        # Update it
        result = campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "relationship_type": "ally",
                "attitude": "friendly",
                "trust_level": 3,
            }
        )

        assert result.success
        assert "updated" in result.data

        # Verify only one relationship exists
        voz = campaign_server.characters.get_by_name("Voz Lirayne")
        assert len(voz.relationships) == 1
        assert voz.relationships[0].relationship_type == "ally"
        assert voz.relationships[0].trust_level == 3

    def test_add_relationship_character_not_found(self, campaign_server):
        """Test adding relationship when character doesn't exist."""
        result = campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Unknown NPC",
                "target_name": "Guard Captain",
            }
        )

        assert not result.success
        assert "not found" in result.error

    def test_add_relationship_target_not_found(self, campaign_server):
        """Test adding relationship when target doesn't exist."""
        result = campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Unknown Target",
            }
        )

        assert not result.success
        assert "not found" in result.error

    def test_get_relationships(self, campaign_server):
        """Test getting all relationships for a character."""
        # Add some relationships
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "relationship_type": "ally",
                "attitude": "friendly",
                "trust_level": 2,
                "history": "Worked together",
            }
        )
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Goblin Chief",
                "relationship_type": "enemy",
                "attitude": "hostile",
                "trust_level": -3,
            }
        )

        result = campaign_server.call_tool(
            "get_relationships",
            {"character_name": "Voz Lirayne"}
        )

        assert result.success
        assert "Voz Lirayne's relationships:" in result.data
        assert "Guard Captain" in result.data
        assert "ally" in result.data
        assert "friendly" in result.data
        assert "trust: +2" in result.data
        assert "Goblin Chief" in result.data
        assert "enemy" in result.data
        assert "hostile" in result.data
        assert "trust: -3" in result.data
        assert "Worked together" in result.data

    def test_get_relationships_no_relationships(self, campaign_server):
        """Test getting relationships when character has none."""
        result = campaign_server.call_tool(
            "get_relationships",
            {"character_name": "Voz Lirayne"}
        )

        assert result.success
        assert "no relationships" in result.data

    def test_query_relationships_by_type(self, campaign_server):
        """Test querying relationships by type."""
        # Add mixed relationships
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "relationship_type": "ally",
            }
        )
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Goblin Chief",
                "relationship_type": "enemy",
            }
        )

        # Query for allies
        result = campaign_server.call_tool(
            "query_relationships",
            {
                "character_name": "Voz Lirayne",
                "relationship_type": "ally"
            }
        )

        assert result.success
        assert "Guard Captain" in result.data
        assert "Goblin Chief" not in result.data

    def test_query_relationships_by_attitude(self, campaign_server):
        """Test querying relationships by attitude."""
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "attitude": "friendly",
            }
        )
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Goblin Chief",
                "attitude": "hostile",
            }
        )

        result = campaign_server.call_tool(
            "query_relationships",
            {
                "character_name": "Voz Lirayne",
                "attitude": "hostile"
            }
        )

        assert result.success
        assert "Goblin Chief" in result.data
        assert "Guard Captain" not in result.data

    def test_query_relationships_by_min_trust(self, campaign_server):
        """Test querying relationships by minimum trust level."""
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "trust_level": 3,
            }
        )
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Goblin Chief",
                "trust_level": -2,
            }
        )

        result = campaign_server.call_tool(
            "query_relationships",
            {
                "character_name": "Voz Lirayne",
                "min_trust": 2
            }
        )

        assert result.success
        assert "Guard Captain" in result.data
        assert "Goblin Chief" not in result.data

    def test_query_relationships_no_matches(self, campaign_server):
        """Test querying with no matching relationships."""
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
                "relationship_type": "ally",
            }
        )

        result = campaign_server.call_tool(
            "query_relationships",
            {
                "character_name": "Voz Lirayne",
                "relationship_type": "enemy"
            }
        )

        assert result.success
        assert "No relationships found" in result.data

    def test_remove_relationship(self, campaign_server):
        """Test removing a relationship."""
        # Add relationship
        campaign_server.call_tool(
            "add_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
            }
        )

        # Verify it exists
        voz = campaign_server.characters.get_by_name("Voz Lirayne")
        assert len(voz.relationships) == 1

        # Remove it
        result = campaign_server.call_tool(
            "remove_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
            }
        )

        assert result.success
        assert "removed" in result.data

        # Verify it's gone
        voz = campaign_server.characters.get_by_name("Voz Lirayne")
        assert len(voz.relationships) == 0

    def test_remove_nonexistent_relationship(self, campaign_server):
        """Test removing a relationship that doesn't exist."""
        result = campaign_server.call_tool(
            "remove_relationship",
            {
                "character_name": "Voz Lirayne",
                "target_name": "Guard Captain",
            }
        )

        assert not result.success
        assert "No relationship found" in result.error


class TestRelationshipPersistence:
    """Tests for relationship persistence."""

    def test_relationships_persist_across_loads(self, tmp_path):
        """Test that relationships are saved and loaded correctly."""
        with patch("gm_agent.storage.characters.CAMPAIGNS_DIR", tmp_path):
            # Create and save character with relationships
            store1 = CharacterStore("test-campaign", base_dir=tmp_path)
            char1 = store1.create("Voz Lirayne")
            char2 = store1.create("Guard Captain")

            rel = Relationship(
                target_character_id=char2.id,
                target_name="Guard Captain",
                relationship_type="ally",
                attitude="friendly",
                trust_level=3,
                history="Allies since the war"
            )
            char1.relationships.append(rel)
            store1.update(char1)

            # Load character in new store instance
            store2 = CharacterStore("test-campaign", base_dir=tmp_path)
            loaded = store2.get_by_name("Voz Lirayne")

            assert len(loaded.relationships) == 1
            assert loaded.relationships[0].target_name == "Guard Captain"
            assert loaded.relationships[0].relationship_type == "ally"
            assert loaded.relationships[0].trust_level == 3
            assert loaded.relationships[0].history == "Allies since the war"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
