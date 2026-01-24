"""Integration tests for storage layer (CampaignStore and SessionStore)."""

import json
from pathlib import Path

import pytest

from gm_agent.storage.campaign import CampaignStore, slugify
from gm_agent.storage.session import SessionStore
from gm_agent.storage.schemas import (
    Campaign,
    Session,
    Turn,
    SceneState,
    PartyMember,
    ToolCallRecord,
    TurnMetadata,
)


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_slugify(self):
        """slugify should convert name to lowercase with dashes."""
        assert slugify("Rise of the Runelords") == "rise-of-the-runelords"

    def test_removes_special_chars(self):
        """slugify should remove special characters."""
        assert slugify("Test!@#$%Campaign") == "testcampaign"

    def test_converts_spaces_to_dashes(self):
        """slugify should convert spaces to dashes."""
        assert slugify("My Test Campaign") == "my-test-campaign"

    def test_removes_underscores(self):
        """slugify should remove underscores (they're filtered as special chars)."""
        # Note: The regex [^a-z0-9\s-] removes underscores before the space conversion
        assert slugify("my_test_campaign") == "mytestcampaign"

    def test_collapses_multiple_dashes(self):
        """slugify should collapse multiple dashes."""
        assert slugify("test---campaign") == "test-campaign"
        assert slugify("test   campaign") == "test-campaign"

    def test_strips_leading_trailing_dashes(self):
        """slugify should strip leading and trailing dashes."""
        assert slugify("-test-campaign-") == "test-campaign"
        assert slugify("  test  ") == "test"

    def test_unicode_handling(self):
        """slugify should handle unicode characters."""
        # Unicode letters that aren't a-z get removed
        assert slugify("Tëst Cämpaign") == "tst-cmpaign"

    def test_numbers_preserved(self):
        """slugify should preserve numbers."""
        assert slugify("Campaign 2024") == "campaign-2024"
        assert slugify("2e Rules") == "2e-rules"

    def test_empty_string(self):
        """slugify should handle empty string."""
        assert slugify("") == ""

    def test_only_special_chars(self):
        """slugify should return empty for only special chars."""
        assert slugify("!@#$%") == ""


class TestCampaignStore:
    """Integration tests for CampaignStore."""

    def test_create_campaign(self, campaign_store: CampaignStore):
        """CampaignStore should create and return new campaign."""
        campaign = campaign_store.create(
            name="Test Campaign",
            background="A test background",
        )

        assert campaign.id == "test-campaign"
        assert campaign.name == "Test Campaign"
        assert campaign.background == "A test background"
        assert campaign.created_at is not None
        assert campaign.updated_at is not None

    def test_create_campaign_with_party(
        self, campaign_store: CampaignStore, sample_party: list[PartyMember]
    ):
        """CampaignStore should create campaign with party members."""
        campaign = campaign_store.create(
            name="Party Campaign",
            background="Test",
            party=sample_party,
        )

        assert len(campaign.party) == 3
        assert campaign.party[0].name == "Valeros"

    def test_create_campaign_directory_structure(
        self, campaign_store: CampaignStore, tmp_campaigns_dir: Path
    ):
        """CampaignStore should create proper directory structure."""
        campaign = campaign_store.create(name="Test Campaign")

        campaign_dir = tmp_campaigns_dir / "test-campaign"
        assert campaign_dir.exists()
        assert (campaign_dir / "campaign.json").exists()
        assert (campaign_dir / "sessions").exists()

    def test_create_duplicate_fails(self, campaign_store: CampaignStore):
        """CampaignStore should reject duplicate campaign IDs."""
        campaign_store.create(name="Test Campaign")

        with pytest.raises(ValueError, match="already exists"):
            campaign_store.create(name="Test Campaign")

    def test_get_campaign(self, campaign_store: CampaignStore):
        """CampaignStore should retrieve campaign by ID."""
        created = campaign_store.create(name="Retrieve Test")
        retrieved = campaign_store.get("retrieve-test")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name

    def test_get_nonexistent_returns_none(self, campaign_store: CampaignStore):
        """CampaignStore should return None for missing campaign."""
        result = campaign_store.get("nonexistent")
        assert result is None

    def test_list_campaigns(self, campaign_store: CampaignStore):
        """CampaignStore should list all campaigns sorted by name."""
        campaign_store.create(name="Zebra Campaign")
        campaign_store.create(name="Alpha Campaign")
        campaign_store.create(name="Middle Campaign")

        campaigns = campaign_store.list()

        assert len(campaigns) == 3
        assert campaigns[0].name == "Alpha Campaign"
        assert campaigns[1].name == "Middle Campaign"
        assert campaigns[2].name == "Zebra Campaign"

    def test_list_empty(self, campaign_store: CampaignStore):
        """CampaignStore should return empty list when no campaigns."""
        campaigns = campaign_store.list()
        assert campaigns == []

    def test_update_campaign(self, campaign_store: CampaignStore):
        """CampaignStore should update campaign and timestamp."""
        campaign = campaign_store.create(name="Update Test")
        original_updated = campaign.updated_at

        campaign.background = "Updated background"
        campaign.current_arc = "New arc"
        updated = campaign_store.update(campaign)

        assert updated.background == "Updated background"
        assert updated.current_arc == "New arc"
        assert updated.updated_at > original_updated

        # Verify persistence
        retrieved = campaign_store.get(campaign.id)
        assert retrieved.background == "Updated background"

    def test_delete_campaign(self, campaign_store: CampaignStore, tmp_campaigns_dir: Path):
        """CampaignStore should delete campaign and directory."""
        campaign = campaign_store.create(name="Delete Test")
        campaign_dir = tmp_campaigns_dir / "delete-test"
        assert campaign_dir.exists()

        result = campaign_store.delete("delete-test")

        assert result is True
        assert not campaign_dir.exists()
        assert campaign_store.get("delete-test") is None

    def test_delete_nonexistent_returns_false(self, campaign_store: CampaignStore):
        """CampaignStore should return False when deleting missing campaign."""
        result = campaign_store.delete("nonexistent")
        assert result is False

    def test_file_persistence(self, campaign_store: CampaignStore, tmp_campaigns_dir: Path):
        """CampaignStore should persist data to JSON file."""
        campaign = campaign_store.create(
            name="Persistence Test",
            background="Test background",
        )

        # Read file directly
        campaign_file = tmp_campaigns_dir / "persistence-test" / "campaign.json"
        with open(campaign_file) as f:
            data = json.load(f)

        assert data["id"] == "persistence-test"
        assert data["name"] == "Persistence Test"
        assert data["background"] == "Test background"

    def test_reload_campaign_from_disk(
        self, tmp_campaigns_dir: Path, sample_party: list[PartyMember]
    ):
        """Campaign should be loadable from a fresh CampaignStore."""
        # Create campaign with one store instance
        store1 = CampaignStore(base_dir=tmp_campaigns_dir)
        store1.create(name="Reload Test", background="Test", party=sample_party)

        # Load with fresh store instance
        store2 = CampaignStore(base_dir=tmp_campaigns_dir)
        campaign = store2.get("reload-test")

        assert campaign is not None
        assert campaign.name == "Reload Test"
        assert len(campaign.party) == 3


class TestSessionStore:
    """Integration tests for SessionStore."""

    def test_start_session(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should start new session for campaign."""
        session = session_store.start(persisted_campaign.id)

        assert session.id is not None
        assert len(session.id) == 8  # UUID[:8]
        assert session.campaign_id == persisted_campaign.id
        assert session.turns == []
        assert session.started_at is not None

    def test_get_current_session(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should retrieve current active session."""
        started = session_store.start(persisted_campaign.id)
        current = session_store.get_current(persisted_campaign.id)

        assert current is not None
        assert current.id == started.id

    def test_get_current_no_session(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should return None when no active session."""
        current = session_store.get_current(persisted_campaign.id)
        assert current is None

    def test_get_or_start_creates_new(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """get_or_start should create session if none exists."""
        session = session_store.get_or_start(persisted_campaign.id)
        assert session is not None
        assert session.campaign_id == persisted_campaign.id

    def test_get_or_start_returns_existing(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """get_or_start should return existing active session."""
        first = session_store.start(persisted_campaign.id)
        second = session_store.get_or_start(persisted_campaign.id)

        assert second.id == first.id

    def test_add_turn(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should add turn to current session."""
        session_store.start(persisted_campaign.id)

        turn = session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="I attack the goblin",
            gm_response="Roll for attack!",
        )

        assert turn.player_input == "I attack the goblin"
        assert turn.gm_response == "Roll for attack!"
        assert turn.timestamp is not None

        # Verify in session
        session = session_store.get_current(persisted_campaign.id)
        assert len(session.turns) == 1

    def test_add_turn_with_tool_calls(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should add turn with tool calls."""
        session_store.start(persisted_campaign.id)

        tool_calls = [
            ToolCallRecord(
                name="lookup_creature",
                args={"name": "goblin"},
                result="Goblin data",
            )
        ]

        turn = session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="What is a goblin?",
            gm_response="A goblin is...",
            tool_calls=tool_calls,
        )

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "lookup_creature"

    def test_add_turn_with_scene_state(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should update scene state from turn."""
        session_store.start(persisted_campaign.id)

        scene = SceneState(
            location="Tavern",
            npcs_present=["Bartender"],
            time_of_day="evening",
        )

        session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="We enter the tavern",
            gm_response="The tavern is warm and inviting",
            scene_state=scene,
        )

        session = session_store.get_current(persisted_campaign.id)
        assert session.scene_state.location == "Tavern"

    def test_add_turn_no_session_fails(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should fail when adding turn without session."""
        with pytest.raises(ValueError, match="No active session"):
            session_store.add_turn(
                campaign_id=persisted_campaign.id,
                player_input="test",
                gm_response="test",
            )

    def test_update_scene(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should update scene state."""
        session_store.start(persisted_campaign.id)

        scene = SceneState(
            location="Forest",
            conditions=["dark", "foggy"],
        )
        session_store.update_scene(persisted_campaign.id, scene)

        session = session_store.get_current(persisted_campaign.id)
        assert session.scene_state.location == "Forest"
        assert "foggy" in session.scene_state.conditions

    def test_update_scene_no_session_fails(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should fail when updating scene without session."""
        with pytest.raises(ValueError, match="No active session"):
            session_store.update_scene(persisted_campaign.id, SceneState(location="Test"))

    def test_end_session(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should end and archive current session."""
        session_store.start(persisted_campaign.id)
        session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="test",
            gm_response="test",
        )

        ended = session_store.end(persisted_campaign.id, summary="Great session!")

        assert ended is not None
        assert ended.ended_at is not None
        assert ended.summary == "Great session!"

        # No current session
        assert session_store.get_current(persisted_campaign.id) is None

    def test_end_session_no_session(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should return None when ending nonexistent session."""
        result = session_store.end(persisted_campaign.id)
        assert result is None

    def test_list_sessions(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should list archived sessions."""
        # Create and end multiple sessions
        session_store.start(persisted_campaign.id)
        session_store.end(persisted_campaign.id, "Session 1")

        session_store.start(persisted_campaign.id)
        session_store.end(persisted_campaign.id, "Session 2")

        sessions = session_store.list(persisted_campaign.id)

        assert len(sessions) == 2
        # Sorted by started_at
        assert sessions[0].summary == "Session 1"
        assert sessions[1].summary == "Session 2"

    def test_list_sessions_excludes_current(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore.list should not include current session."""
        session_store.start(persisted_campaign.id)

        sessions = session_store.list(persisted_campaign.id)
        assert len(sessions) == 0

    def test_get_session_by_id(self, session_store: SessionStore, persisted_campaign: Campaign):
        """SessionStore should retrieve archived session by ID."""
        session_store.start(persisted_campaign.id)
        started = session_store.get_current(persisted_campaign.id)
        session_store.end(persisted_campaign.id)

        retrieved = session_store.get(persisted_campaign.id, started.id)

        assert retrieved is not None
        assert retrieved.id == started.id

    def test_get_nonexistent_session(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should return None for missing session."""
        result = session_store.get(persisted_campaign.id, "nonexistent")
        assert result is None

    def test_start_ends_current_session(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """Starting new session should end current one."""
        first = session_store.start(persisted_campaign.id)
        session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="test",
            gm_response="test",
        )

        second = session_store.start(persisted_campaign.id)

        assert second.id != first.id

        # First session should be archived
        sessions = session_store.list(persisted_campaign.id)
        assert len(sessions) == 1
        assert sessions[0].id == first.id

    def test_session_file_persistence(
        self,
        session_store: SessionStore,
        persisted_campaign: Campaign,
        tmp_campaigns_dir: Path,
    ):
        """Session should persist to JSON file."""
        session = session_store.start(persisted_campaign.id)

        # Check current.json exists
        current_file = tmp_campaigns_dir / persisted_campaign.id / "sessions" / "current.json"
        assert current_file.exists()

        with open(current_file) as f:
            data = json.load(f)
        assert data["id"] == session.id

    def test_archived_session_persistence(
        self,
        session_store: SessionStore,
        persisted_campaign: Campaign,
        tmp_campaigns_dir: Path,
    ):
        """Archived session should persist to named file."""
        session = session_store.start(persisted_campaign.id)
        session_id = session.id
        session_store.end(persisted_campaign.id)

        # Check archive file exists
        archive_file = tmp_campaigns_dir / persisted_campaign.id / "sessions" / f"{session_id}.json"
        assert archive_file.exists()

        # current.json should be gone
        current_file = tmp_campaigns_dir / persisted_campaign.id / "sessions" / "current.json"
        assert not current_file.exists()

    def test_add_turn_with_metadata(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should record turn metadata."""
        session_store.start(persisted_campaign.id)

        metadata = TurnMetadata(
            source="automation",
            event_type="playerChat",
            player_id="player-123",
            actor_name="Valeros",
            processing_time_ms=150.5,
            model="qwen3:latest",
            tool_count=2,
        )

        turn = session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="I search the room",
            gm_response="You find a hidden door",
            metadata=metadata,
        )

        assert turn.metadata.source == "automation"
        assert turn.metadata.event_type == "playerChat"
        assert turn.metadata.player_id == "player-123"
        assert turn.metadata.actor_name == "Valeros"
        assert turn.metadata.processing_time_ms == 150.5
        assert turn.metadata.model == "qwen3:latest"
        assert turn.metadata.tool_count == 2

    def test_add_turn_default_metadata(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """SessionStore should use default metadata when not provided."""
        session_store.start(persisted_campaign.id)

        turn = session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="What do I see?",
            gm_response="You see a dark corridor",
        )

        # Default metadata values
        assert turn.metadata.source == "manual"
        assert turn.metadata.event_type is None
        assert turn.metadata.player_id is None

    def test_metadata_persists_in_session(
        self, session_store: SessionStore, persisted_campaign: Campaign
    ):
        """Metadata should be preserved when session is reloaded."""
        session_store.start(persisted_campaign.id)

        metadata = TurnMetadata(
            source="automation",
            event_type="combatTurn",
            actor_name="Goblin",
            processing_time_ms=250.0,
        )

        session_store.add_turn(
            campaign_id=persisted_campaign.id,
            player_input="Goblin's turn",
            gm_response="The goblin attacks!",
            metadata=metadata,
        )

        # Reload session from disk
        session = session_store.get_current(persisted_campaign.id)
        turn = session.turns[0]

        assert turn.metadata.source == "automation"
        assert turn.metadata.event_type == "combatTurn"
        assert turn.metadata.actor_name == "Goblin"
        assert turn.metadata.processing_time_ms == 250.0


class TestStorageIntegration:
    """Integration tests for CampaignStore and SessionStore together."""

    def test_delete_campaign_removes_sessions(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        tmp_campaigns_dir: Path,
    ):
        """Deleting campaign should remove all sessions."""
        campaign = campaign_store.create(name="Delete Sessions Test")
        session_store.start(campaign.id)
        session_store.end(campaign.id)
        session_store.start(campaign.id)
        session_store.end(campaign.id)

        campaign_store.delete(campaign.id)

        # Directory should be gone
        campaign_dir = tmp_campaigns_dir / campaign.id
        assert not campaign_dir.exists()

    def test_full_session_lifecycle(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        sample_party: list[PartyMember],
    ):
        """Test complete campaign and session workflow."""
        # Create campaign
        campaign = campaign_store.create(
            name="Full Lifecycle Test",
            background="Test campaign",
            party=sample_party,
        )

        # Start session
        session = session_store.start(campaign.id)
        assert session.turns == []

        # Add turns
        session_store.add_turn(
            campaign_id=campaign.id,
            player_input="We explore the dungeon",
            gm_response="You see a dark corridor ahead",
        )

        session_store.add_turn(
            campaign_id=campaign.id,
            player_input="I light a torch",
            gm_response="The torch illuminates ancient carvings",
            tool_calls=[
                ToolCallRecord(
                    name="search_rules",
                    args={"query": "light"},
                    result="Light rules info",
                )
            ],
        )

        # Update scene
        session_store.update_scene(
            campaign.id,
            SceneState(
                location="Ancient Dungeon",
                conditions=["dim light"],
            ),
        )

        # Get current state
        current = session_store.get_current(campaign.id)
        assert len(current.turns) == 2
        assert current.scene_state.location == "Ancient Dungeon"

        # End session
        ended = session_store.end(campaign.id, "Explored the dungeon entrance")
        assert ended.ended_at is not None

        # Verify archived
        sessions = session_store.list(campaign.id)
        assert len(sessions) == 1
        assert sessions[0].summary == "Explored the dungeon entrance"

        # Start new session
        new_session = session_store.start(campaign.id)
        assert new_session.id != ended.id
        assert new_session.turns == []
