"""Tests for Virtual Player system."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gm_agent.systems.shared.virtual_player import (
    VirtualPlayerConfig,
    VirtualPlayerDecision,
)
from gm_agent.systems.shared.virtual_player_engine import (
    VirtualPlayerEngine,
    _parse_json_object,
)
from gm_agent.systems.microscope.servers import MicroscopeSessionServer
from gm_agent.models.base import LLMResponse, Message


# ===========================================================================
# Fixtures
# ===========================================================================


class MockLLMForVP:
    """Minimal mock LLM that returns JSON decisions."""

    def __init__(self, response_text: str = ""):
        self.response_text = response_text or json.dumps({
            "decision_type": "create_period",
            "content": {
                "title": "The Age of Innovation",
                "tone": "light",
                "description": "Technology advances rapidly.",
            },
            "reasoning": "The timeline needs a period of progress.",
        })
        self.calls: list = []

    def chat(self, messages, tools=None, thinking=None, temperature=None):
        self.calls.append(messages)
        return LLMResponse(text=self.response_text)

    def get_model_name(self) -> str:
        return "mock-vp"

    def is_available(self) -> bool:
        return True


@pytest.fixture
def mock_llm():
    return MockLLMForVP()


@pytest.fixture
def campaigns_dir(tmp_path: Path) -> Path:
    d = tmp_path / "campaigns"
    d.mkdir()
    (d / "vp-test").mkdir()
    return d


@pytest.fixture
def server(campaigns_dir: Path, mock_llm) -> MicroscopeSessionServer:
    with patch("gm_agent.systems.microscope.servers.CAMPAIGNS_DIR", campaigns_dir):
        srv = MicroscopeSessionServer("vp-test", llm=mock_llm)
        yield srv
        srv.close()


@pytest.fixture
def server_no_llm(campaigns_dir: Path) -> MicroscopeSessionServer:
    with patch("gm_agent.systems.microscope.servers.CAMPAIGNS_DIR", campaigns_dir):
        srv = MicroscopeSessionServer("vp-test")
        yield srv
        srv.close()


def _setup_session(server):
    return server.call_tool("microscope_setup", {
        "big_picture": "The rise and fall of a galactic empire",
        "start_period": "The First Colony",
        "start_tone": "light",
        "end_period": "The Final Silence",
        "end_tone": "dark",
        "players": '["Alice", "Bob"]',
    })


# ===========================================================================
# Data Models
# ===========================================================================


class TestVirtualPlayerConfig:
    def test_basic(self):
        vp = VirtualPlayerConfig(name="Oracle")
        assert vp.name == "Oracle"
        assert vp.is_active
        assert vp.personality_profile == {}

    def test_with_profile(self):
        from gm_agent.systems.shared.personality import PersonalityProfile
        pp = PersonalityProfile.from_archetype("the_sage")
        vp = VirtualPlayerConfig(
            name="Sage",
            personality_profile=pp.to_dict(),
        )
        assert vp.personality_profile["archetype"] == "the_sage"

    def test_describe(self):
        vp = VirtualPlayerConfig(name="Test")
        assert "Test" in vp.describe()

    def test_describe_with_archetype(self):
        from gm_agent.systems.shared.personality import PersonalityProfile
        pp = PersonalityProfile.from_archetype("the_sage")
        vp = VirtualPlayerConfig(
            name="Sage",
            personality_profile=pp.to_dict(),
        )
        desc = vp.describe()
        assert "the_sage" in desc

    def test_serialization(self):
        vp = VirtualPlayerConfig(name="Test", creative_preferences={"genre": "dark"})
        d = vp.model_dump()
        vp2 = VirtualPlayerConfig(**d)
        assert vp2.name == "Test"
        assert vp2.creative_preferences["genre"] == "dark"


class TestVirtualPlayerDecision:
    def test_basic(self):
        d = VirtualPlayerDecision(
            player_name="Oracle",
            decision_type="create_period",
            content={"title": "X"},
            reasoning="Because.",
        )
        assert d.player_name == "Oracle"
        assert d.decision_type == "create_period"
        assert d.content["title"] == "X"


# ===========================================================================
# JSON parsing
# ===========================================================================


class TestParseJsonObject:
    def test_clean_json(self):
        result = _parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fenced(self):
        result = _parse_json_object('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_with_surrounding_text(self):
        result = _parse_json_object('Here is the decision:\n{"key": "value"}\nEnd.')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        result = _parse_json_object("not json at all")
        assert result == {}

    def test_array_not_object(self):
        result = _parse_json_object('[1, 2, 3]')
        assert result == {}


# ===========================================================================
# VirtualPlayerEngine
# ===========================================================================


class TestVirtualPlayerEngine:
    def test_generate_decision(self, mock_llm):
        engine = VirtualPlayerEngine(mock_llm)
        result = engine.generate_decision(
            player_name="Oracle",
            game_state={"big_picture": "Test history"},
            decision_type="create_period",
        )
        assert result["decision_type"] == "create_period"
        assert "title" in result["content"]
        assert len(mock_llm.calls) == 1

    def test_with_personality(self, mock_llm):
        from gm_agent.systems.shared.personality import PersonalityProfile
        pp = PersonalityProfile.from_archetype("the_sage")

        engine = VirtualPlayerEngine(mock_llm)
        result = engine.generate_decision(
            player_name="Sage",
            personality_profile=pp.to_dict(),
            decision_type="create_event",
        )
        assert result["decision_type"] in ("create_event", "create_period")

        # Check that personality was included in the system prompt
        system_msg = mock_llm.calls[0][0].content
        assert "personality" in system_msg.lower() or "trait" in system_msg.lower()

    def test_with_preferences(self, mock_llm):
        engine = VirtualPlayerEngine(mock_llm)
        result = engine.generate_decision(
            player_name="Artist",
            creative_preferences={"genre": "dark fantasy", "tone": "melancholy"},
            decision_type="create",
        )
        assert isinstance(result, dict)

    def test_with_options(self, mock_llm):
        engine = VirtualPlayerEngine(mock_llm)
        result = engine.generate_decision(
            player_name="Player",
            available_options=["create_period", "create_event"],
            decision_type="create",
        )
        assert isinstance(result, dict)

    def test_malformed_response(self):
        bad_llm = MockLLMForVP(response_text="I don't know what to do!")
        engine = VirtualPlayerEngine(bad_llm)
        result = engine.generate_decision(
            player_name="Confused",
            decision_type="create",
        )
        # Should still return a dict, even if content is empty
        assert isinstance(result, dict)
        assert "decision_type" in result


# ===========================================================================
# Microscope Integration
# ===========================================================================


class TestMicroscopeVirtualPlayer:
    def test_add_virtual_player_with_archetype(self, server):
        _setup_session(server)
        result = server.call_tool("add_virtual_player", {
            "name": "Oracle",
            "archetype": "the_sage",
        })
        assert result.success
        assert "Oracle" in result.data
        assert len(server.state["virtual_players"]) == 1

    def test_add_virtual_player_custom_traits(self, server):
        _setup_session(server)
        result = server.call_tool("add_virtual_player", {
            "name": "Custom",
            "traits": '{"openness": 0.9, "risk_appetite": 0.8}',
        })
        assert result.success
        assert len(server.state["virtual_players"]) == 1

    def test_add_duplicate_fails(self, server):
        _setup_session(server)
        server.call_tool("add_virtual_player", {"name": "Oracle"})
        result = server.call_tool("add_virtual_player", {"name": "Oracle"})
        assert not result.success
        assert "already exists" in (result.error or "")

    def test_add_unknown_archetype(self, server):
        _setup_session(server)
        result = server.call_tool("add_virtual_player", {
            "name": "Bad",
            "archetype": "nonexistent_archetype",
        })
        assert not result.success

    def test_add_no_session(self, server):
        result = server.call_tool("add_virtual_player", {"name": "X"})
        assert not result.success

    def test_virtual_player_turn(self, server):
        _setup_session(server)
        server.call_tool("add_virtual_player", {
            "name": "Oracle",
            "archetype": "the_sage",
        })
        result = server.call_tool("virtual_player_turn", {
            "player_name": "Oracle",
            "decision_type": "create_period",
        })
        assert result.success
        assert "Oracle" in result.data
        assert "Period Created" in result.data

    def test_virtual_player_turn_no_llm(self, server_no_llm):
        _setup_session(server_no_llm)
        server_no_llm.state["virtual_players"] = [{"name": "X", "is_active": True}]
        result = server_no_llm.call_tool("virtual_player_turn", {
            "player_name": "X",
        })
        assert not result.success
        assert "LLM" in (result.error or "")

    def test_virtual_player_turn_unknown_player(self, server):
        _setup_session(server)
        result = server.call_tool("virtual_player_turn", {
            "player_name": "Unknown",
        })
        assert not result.success
        assert "not found" in (result.error or "")

    def test_remove_virtual_player(self, server):
        _setup_session(server)
        server.call_tool("add_virtual_player", {"name": "Oracle"})
        result = server.call_tool("remove_virtual_player", {"player_name": "Oracle"})
        assert result.success
        assert len(server.state.get("virtual_players", [])) == 0

    def test_remove_unknown_player(self, server):
        _setup_session(server)
        result = server.call_tool("remove_virtual_player", {"player_name": "Unknown"})
        assert not result.success

    def test_remove_no_session(self, server):
        result = server.call_tool("remove_virtual_player", {"player_name": "X"})
        assert not result.success

    def test_tool_count_with_virtual_players(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "add_virtual_player" in names
        assert "virtual_player_turn" in names
        assert "remove_virtual_player" in names
        assert len(names) == 25  # 22 + 3 virtual player tools


# ===========================================================================
# Virtual Player Memory (WS4)
# ===========================================================================


class TestVirtualPlayerMemory:
    """Tests for VP decision history tracking."""

    def test_history_initialized_empty(self, server):
        """New VPs should have empty history."""
        _setup_session(server)
        server.call_tool("add_virtual_player", {"name": "Oracle"})
        vp = server.state["virtual_players"][0]
        assert vp["history"] == []

    def test_history_accumulates(self, server):
        """Each turn should append to VP history."""
        _setup_session(server)
        server.call_tool("add_virtual_player", {"name": "Oracle"})

        server.call_tool("virtual_player_turn", {
            "player_name": "Oracle",
            "decision_type": "create_period",
        })
        server.call_tool("virtual_player_turn", {
            "player_name": "Oracle",
            "decision_type": "create_event",
        })

        vp = server.state["virtual_players"][0]
        assert len(vp["history"]) == 2
        assert vp["history"][0]["decision_type"] == "create_period"

    def test_history_passed_to_engine(self, mock_llm):
        """Engine should receive decision_history and include it in prompt."""
        engine = VirtualPlayerEngine(mock_llm)

        history = [
            {"decision_type": "create_period", "content": {"title": "Rise"}, "reasoning": "first"},
            {"decision_type": "create_event", "content": {"title": "Battle"}, "reasoning": "second"},
        ]

        engine.generate_decision(
            player_name="Oracle",
            decision_type="create_event",
            decision_history=history,
        )

        # Check that history section was included in the user message
        user_msg = mock_llm.calls[0][1].content
        assert "previous decisions" in user_msg.lower()
        assert "create_period" in user_msg
        assert "create_event" in user_msg

    def test_history_not_in_prompt_when_empty(self, mock_llm):
        """Empty history should not add history section to prompt."""
        engine = VirtualPlayerEngine(mock_llm)

        engine.generate_decision(
            player_name="Oracle",
            decision_type="create",
            decision_history=[],
        )

        user_msg = mock_llm.calls[0][1].content
        assert "previous decisions" not in user_msg.lower()

    def test_history_none_no_section(self, mock_llm):
        """None history (default) should not add section."""
        engine = VirtualPlayerEngine(mock_llm)

        engine.generate_decision(
            player_name="Oracle",
            decision_type="create",
        )

        user_msg = mock_llm.calls[0][1].content
        assert "previous decisions" not in user_msg.lower()

    def test_history_truncation(self, mock_llm):
        """Only last MAX_DECISION_HISTORY entries should be included."""
        from gm_agent.systems.shared.virtual_player_engine import MAX_DECISION_HISTORY

        engine = VirtualPlayerEngine(mock_llm)

        history = [
            {"decision_type": f"type_{i}", "content": {"n": i}, "reasoning": f"r{i}"}
            for i in range(10)
        ]

        engine.generate_decision(
            player_name="Oracle",
            decision_type="create",
            decision_history=history,
        )

        user_msg = mock_llm.calls[0][1].content
        # Should only contain the last MAX_DECISION_HISTORY entries
        assert f"type_{10 - MAX_DECISION_HISTORY}" in user_msg
        assert "type_9" in user_msg
        # Earlier entries should be absent
        assert "type_0" not in user_msg

    def test_history_content_truncated(self, mock_llm):
        """Long content in history should be truncated in prompt."""
        engine = VirtualPlayerEngine(mock_llm)

        history = [
            {"decision_type": "create", "content": {"text": "x" * 500}, "reasoning": "long"},
        ]

        engine.generate_decision(
            player_name="Oracle",
            decision_type="create",
            decision_history=history,
        )

        user_msg = mock_llm.calls[0][1].content
        assert "previous decisions" in user_msg.lower()
        # Content should be truncated with ...
        assert "..." in user_msg

    def test_legacy_vp_without_history_key(self, server):
        """VPs loaded from old state (no history key) should work."""
        _setup_session(server)
        # Simulate a legacy VP without history key
        server.state["virtual_players"] = [
            {"name": "Legacy", "is_active": True, "personality_profile": {}, "creative_preferences": {}},
        ]
        server._save_state(server.state)

        result = server.call_tool("virtual_player_turn", {
            "player_name": "Legacy",
            "decision_type": "create_period",
        })
        assert result.success

        # History should now be populated
        vp = server.state["virtual_players"][0]
        assert len(vp.get("history", [])) == 1
