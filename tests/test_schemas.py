"""Unit tests for Pydantic data models."""

import json
from datetime import datetime

import pytest

from gm_agent.storage.schemas import (
    Campaign,
    Session,
    Turn,
    SceneState,
    PartyMember,
    ToolCallRecord,
)


class TestSceneState:
    """Tests for SceneState model."""

    def test_default_values(self):
        """SceneState should have sensible defaults."""
        scene = SceneState()
        assert scene.location == "Unknown"
        assert scene.npcs_present == []
        assert scene.time_of_day == "day"
        assert scene.conditions == []
        assert scene.notes == ""

    def test_custom_values(self, sample_scene_state: SceneState):
        """SceneState should accept custom values."""
        assert sample_scene_state.location == "Sandpoint Market Square"
        assert "Sheriff Hemlock" in sample_scene_state.npcs_present
        assert sample_scene_state.time_of_day == "morning"
        assert "crowded" in sample_scene_state.conditions
        assert "Festival" in sample_scene_state.notes

    def test_json_roundtrip(self, sample_scene_state: SceneState):
        """SceneState should serialize and deserialize correctly."""
        json_str = sample_scene_state.model_dump_json()
        restored = SceneState.model_validate_json(json_str)
        assert restored == sample_scene_state

    def test_dict_roundtrip(self, sample_scene_state: SceneState):
        """SceneState should convert to/from dict correctly."""
        data = sample_scene_state.model_dump()
        restored = SceneState.model_validate(data)
        assert restored == sample_scene_state

    def test_mutable_default_isolation(self):
        """Multiple instances should not share mutable defaults."""
        scene1 = SceneState()
        scene2 = SceneState()
        scene1.npcs_present.append("Test NPC")
        assert "Test NPC" not in scene2.npcs_present


class TestToolCallRecord:
    """Tests for ToolCallRecord model."""

    def test_required_fields(self):
        """ToolCallRecord requires name, args, and result."""
        record = ToolCallRecord(
            name="lookup_creature",
            args={"name": "goblin"},
            result="Goblin info...",
        )
        assert record.name == "lookup_creature"
        assert record.args == {"name": "goblin"}
        assert record.result == "Goblin info..."

    def test_complex_args(self):
        """ToolCallRecord should handle complex args dicts."""
        record = ToolCallRecord(
            name="search_content",
            args={"query": "fire spells", "types": ["spell", "cantrip"], "limit": 5},
            result="Search results...",
        )
        assert record.args["types"] == ["spell", "cantrip"]
        assert record.args["limit"] == 5

    def test_json_roundtrip(self, sample_tool_call_record: ToolCallRecord):
        """ToolCallRecord should serialize and deserialize correctly."""
        json_str = sample_tool_call_record.model_dump_json()
        restored = ToolCallRecord.model_validate_json(json_str)
        assert restored == sample_tool_call_record


class TestTurn:
    """Tests for Turn model."""

    def test_required_fields(self):
        """Turn requires player_input and gm_response."""
        turn = Turn(
            player_input="I attack the goblin",
            gm_response="Roll for attack!",
        )
        assert turn.player_input == "I attack the goblin"
        assert turn.gm_response == "Roll for attack!"
        assert turn.tool_calls == []
        assert turn.scene_state is None
        assert isinstance(turn.timestamp, datetime)

    def test_with_tool_calls(self, sample_tool_call_record: ToolCallRecord):
        """Turn should store tool calls."""
        turn = Turn(
            player_input="What is a goblin?",
            gm_response="A goblin is...",
            tool_calls=[sample_tool_call_record],
        )
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "lookup_creature"

    def test_with_scene_state(self, sample_scene_state: SceneState):
        """Turn should store optional scene state."""
        turn = Turn(
            player_input="We enter the market",
            gm_response="The market is bustling",
            scene_state=sample_scene_state,
        )
        assert turn.scene_state is not None
        assert turn.scene_state.location == "Sandpoint Market Square"

    def test_timestamp_auto_generated(self):
        """Turn should auto-generate timestamp."""
        before = datetime.now()
        turn = Turn(player_input="test", gm_response="response")
        after = datetime.now()
        assert before <= turn.timestamp <= after

    def test_json_roundtrip(self, sample_turn: Turn):
        """Turn should serialize and deserialize correctly."""
        data = sample_turn.model_dump(mode="json")
        json_str = json.dumps(data, default=str)
        restored = Turn.model_validate_json(json_str)
        assert restored.player_input == sample_turn.player_input
        assert restored.gm_response == sample_turn.gm_response
        assert len(restored.tool_calls) == len(sample_turn.tool_calls)

    def test_datetime_coercion(self):
        """Turn should accept datetime strings."""
        data = {
            "player_input": "test",
            "gm_response": "response",
            "timestamp": "2024-01-15T14:30:00",
            "tool_calls": [],
        }
        turn = Turn.model_validate(data)
        assert turn.timestamp == datetime(2024, 1, 15, 14, 30, 0)


class TestPartyMember:
    """Tests for PartyMember model."""

    def test_required_name(self):
        """PartyMember requires name."""
        member = PartyMember(name="Valeros")
        assert member.name == "Valeros"
        assert member.ancestry == ""
        assert member.class_name == ""
        assert member.level == 1
        assert member.player_name == ""
        assert member.notes == ""

    def test_full_member(self, sample_party_member: PartyMember):
        """PartyMember should store all fields."""
        assert sample_party_member.name == "Valeros"
        assert sample_party_member.ancestry == "Human"
        assert sample_party_member.class_name == "Fighter"
        assert sample_party_member.level == 5
        assert sample_party_member.player_name == "Alice"
        assert "longsword" in sample_party_member.notes

    def test_json_roundtrip(self, sample_party_member: PartyMember):
        """PartyMember should serialize and deserialize correctly."""
        json_str = sample_party_member.model_dump_json()
        restored = PartyMember.model_validate_json(json_str)
        assert restored == sample_party_member


class TestCampaign:
    """Tests for Campaign model."""

    def test_required_fields(self):
        """Campaign requires id and name."""
        now = datetime.now()
        campaign = Campaign(
            id="test-campaign",
            name="Test Campaign",
            created_at=now,
            updated_at=now,
        )
        assert campaign.id == "test-campaign"
        assert campaign.name == "Test Campaign"
        assert campaign.background == ""
        assert campaign.current_arc == ""
        assert campaign.party == []
        assert campaign.preferences == {}

    def test_full_campaign(self, sample_campaign: Campaign):
        """Campaign should store all fields."""
        assert sample_campaign.id == "rise-of-the-runelords"
        assert sample_campaign.name == "Rise of the Runelords"
        assert "ancient evil" in sample_campaign.background
        assert "Burnt Offerings" in sample_campaign.current_arc
        assert len(sample_campaign.party) == 3
        assert sample_campaign.preferences.get("combat_style") == "theater_of_mind"

    def test_party_members(self, sample_campaign: Campaign):
        """Campaign party should contain PartyMember objects."""
        for member in sample_campaign.party:
            assert isinstance(member, PartyMember)
            assert member.name in ["Valeros", "Seoni", "Kyra"]

    def test_json_roundtrip(self, sample_campaign: Campaign):
        """Campaign should serialize and deserialize correctly."""
        data = sample_campaign.model_dump(mode="json")
        json_str = json.dumps(data, default=str)
        restored = Campaign.model_validate_json(json_str)
        assert restored.id == sample_campaign.id
        assert restored.name == sample_campaign.name
        assert len(restored.party) == len(sample_campaign.party)

    def test_datetime_coercion(self):
        """Campaign should accept datetime strings."""
        data = {
            "id": "test",
            "name": "Test",
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-15T14:00:00",
        }
        campaign = Campaign.model_validate(data)
        assert campaign.created_at == datetime(2024, 1, 1, 10, 0, 0)
        assert campaign.updated_at == datetime(2024, 1, 15, 14, 0, 0)

    def test_mutable_defaults_isolation(self):
        """Multiple campaigns should not share mutable defaults."""
        now = datetime.now()
        c1 = Campaign(id="c1", name="C1", created_at=now, updated_at=now)
        c2 = Campaign(id="c2", name="C2", created_at=now, updated_at=now)
        c1.party.append(PartyMember(name="Test"))
        assert len(c2.party) == 0


class TestSession:
    """Tests for Session model."""

    def test_required_fields(self):
        """Session requires id and campaign_id."""
        now = datetime.now()
        session = Session(
            id="test123",
            campaign_id="test-campaign",
            started_at=now,
        )
        assert session.id == "test123"
        assert session.campaign_id == "test-campaign"
        assert session.turns == []
        assert session.summary == ""
        assert isinstance(session.scene_state, SceneState)
        assert session.ended_at is None

    def test_full_session(self, sample_session: Session):
        """Session should store all fields."""
        assert sample_session.id == "abc12345"
        assert sample_session.campaign_id == "rise-of-the-runelords"
        assert len(sample_session.turns) == 1
        assert sample_session.scene_state.location == "Sandpoint Market Square"

    def test_default_scene_state(self):
        """Session should create default SceneState."""
        session = Session(
            id="test",
            campaign_id="test",
            started_at=datetime.now(),
        )
        assert session.scene_state.location == "Unknown"
        assert session.scene_state.time_of_day == "day"

    def test_json_roundtrip(self, sample_session: Session):
        """Session should serialize and deserialize correctly."""
        data = sample_session.model_dump(mode="json")
        json_str = json.dumps(data, default=str)
        restored = Session.model_validate_json(json_str)
        assert restored.id == sample_session.id
        assert restored.campaign_id == sample_session.campaign_id
        assert len(restored.turns) == len(sample_session.turns)

    def test_ended_at_optional(self):
        """Session.ended_at should be None for active sessions."""
        session = Session(
            id="test",
            campaign_id="test",
            started_at=datetime.now(),
        )
        assert session.ended_at is None

        # Can set ended_at
        session.ended_at = datetime.now()
        assert session.ended_at is not None

    def test_many_turns(self, sample_session_with_many_turns: Session):
        """Session should handle many turns."""
        assert len(sample_session_with_many_turns.turns) == 20
        assert sample_session_with_many_turns.turns[0].player_input == "Player action 1"
        assert sample_session_with_many_turns.turns[19].player_input == "Player action 20"

    def test_mutable_defaults_isolation(self):
        """Multiple sessions should not share mutable defaults."""
        now = datetime.now()
        s1 = Session(id="s1", campaign_id="c", started_at=now)
        s2 = Session(id="s2", campaign_id="c", started_at=now)
        s1.turns.append(Turn(player_input="test", gm_response="response"))
        assert len(s2.turns) == 0


class TestNestedSerialization:
    """Tests for nested model serialization."""

    def test_campaign_with_party_roundtrip(self, sample_campaign: Campaign):
        """Campaign with party should serialize nested PartyMembers."""
        data = sample_campaign.model_dump(mode="json")
        assert isinstance(data["party"], list)
        assert data["party"][0]["name"] == "Valeros"

        restored = Campaign.model_validate(data)
        assert restored.party[0].name == "Valeros"
        assert isinstance(restored.party[0], PartyMember)

    def test_session_with_turns_roundtrip(self, sample_session: Session):
        """Session with turns should serialize nested Turns."""
        data = sample_session.model_dump(mode="json")
        assert isinstance(data["turns"], list)
        assert data["turns"][0]["player_input"] == "What is a goblin?"

        restored = Session.model_validate(data)
        assert restored.turns[0].player_input == "What is a goblin?"
        assert isinstance(restored.turns[0], Turn)

    def test_turn_with_tool_calls_roundtrip(self, sample_turn: Turn):
        """Turn with tool calls should serialize nested ToolCallRecords."""
        data = sample_turn.model_dump(mode="json")
        assert isinstance(data["tool_calls"], list)
        assert data["tool_calls"][0]["name"] == "lookup_creature"

        restored = Turn.model_validate(data)
        assert restored.tool_calls[0].name == "lookup_creature"
        assert isinstance(restored.tool_calls[0], ToolCallRecord)


class TestValidation:
    """Tests for model validation."""

    def test_campaign_rejects_invalid_party(self):
        """Campaign should reject invalid party member data."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Campaign.model_validate(
                {
                    "id": "test",
                    "name": "Test",
                    "party": [{"invalid": "data"}],  # Missing required 'name'
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                }
            )

    def test_session_rejects_invalid_turns(self):
        """Session should reject invalid turn data."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Session.model_validate(
                {
                    "id": "test",
                    "campaign_id": "test",
                    "turns": [{"invalid": "data"}],  # Missing required fields
                    "started_at": "2024-01-01T00:00:00",
                }
            )

    def test_party_member_level_type(self):
        """PartyMember level should be coerced to int."""
        member = PartyMember.model_validate({"name": "Test", "level": "5"})  # String instead of int
        assert member.level == 5
        assert isinstance(member.level, int)
