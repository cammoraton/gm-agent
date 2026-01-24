"""Tests for rolling summary generation."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gm_agent.models.base import LLMBackend, LLMResponse
from gm_agent.storage.schemas import Session, Turn, SceneState, ToolCallRecord
from gm_agent.summarizer import (
    should_update_summary,
    generate_summary_prompt,
    generate_summary,
    RollingSummarizer,
    SUMMARY_SYSTEM_PROMPT,
)


class MockLLM(LLMBackend):
    """Mock LLM for testing."""

    def __init__(self, response_text: str = "This is a test summary."):
        self.response_text = response_text
        self.calls: list[tuple] = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return LLMResponse(text=self.response_text, tool_calls=[])

    def get_model_name(self) -> str:
        return "mock-llm"

    def is_available(self) -> bool:
        return True


@pytest.fixture
def empty_session() -> Session:
    """Create an empty session for testing."""
    return Session(
        id="test-session",
        campaign_id="test-campaign",
        turns=[],
        started_at=datetime.now(),
        scene_state=SceneState(location="Tavern", time_of_day="evening"),
    )


@pytest.fixture
def session_with_turns() -> Session:
    """Create a session with multiple turns."""
    return Session(
        id="test-session",
        campaign_id="test-campaign",
        turns=[
            Turn(
                player_input="I enter the tavern",
                gm_response="You push open the heavy oak door...",
                timestamp=datetime.now(),
            ),
            Turn(
                player_input="I look around",
                gm_response="The tavern is dimly lit...",
                timestamp=datetime.now(),
            ),
            Turn(
                player_input="I approach the bartender",
                gm_response="The grizzled bartender eyes you warily...",
                timestamp=datetime.now(),
                tool_calls=[
                    ToolCallRecord(
                        name="lookup_creature",
                        args={"name": "commoner"},
                        result="Commoner - Level -1 creature...",
                    )
                ],
            ),
        ],
        started_at=datetime.now(),
        scene_state=SceneState(
            location="Rusty Nail Tavern",
            time_of_day="evening",
            npcs_present=["Bartender"],
            conditions=["dim", "crowded"],
        ),
    )


class TestShouldUpdateSummary:
    """Tests for should_update_summary function."""

    def test_empty_session_returns_false(self, empty_session: Session):
        """Empty session should not trigger summary update."""
        assert not should_update_summary(empty_session)

    def test_force_returns_true(self, empty_session: Session):
        """Forced update should return true even for empty session."""
        assert should_update_summary(empty_session, force=True)

    def test_interval_triggers_update(self, empty_session: Session):
        """Summary should be triggered at configured intervals."""
        from gm_agent.config import TURNS_BETWEEN_SUMMARIES

        # Add exactly TURNS_BETWEEN_SUMMARIES turns
        for i in range(TURNS_BETWEEN_SUMMARIES):
            empty_session.turns.append(
                Turn(
                    player_input=f"Turn {i}",
                    gm_response=f"Response {i}",
                    timestamp=datetime.now(),
                )
            )

        assert should_update_summary(empty_session)

    def test_non_interval_returns_false(self, empty_session: Session):
        """Summary should not trigger between intervals."""
        from gm_agent.config import TURNS_BETWEEN_SUMMARIES

        # Add one less than interval
        for i in range(TURNS_BETWEEN_SUMMARIES - 1):
            empty_session.turns.append(
                Turn(
                    player_input=f"Turn {i}",
                    gm_response=f"Response {i}",
                    timestamp=datetime.now(),
                )
            )

        assert not should_update_summary(empty_session)


class TestGenerateSummaryPrompt:
    """Tests for generate_summary_prompt function."""

    def test_returns_system_and_user_messages(self, session_with_turns: Session):
        """Should return system and user messages."""
        messages = generate_summary_prompt(session_with_turns)

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_system_prompt_content(self, session_with_turns: Session):
        """System message should contain summary instructions."""
        messages = generate_summary_prompt(session_with_turns)

        assert SUMMARY_SYSTEM_PROMPT in messages[0].content

    def test_user_prompt_contains_turns(self, session_with_turns: Session):
        """User message should contain turn content."""
        messages = generate_summary_prompt(session_with_turns)
        user_content = messages[1].content

        assert "enter the tavern" in user_content
        assert "look around" in user_content

    def test_user_prompt_contains_scene(self, session_with_turns: Session):
        """User message should contain scene information."""
        messages = generate_summary_prompt(session_with_turns)
        user_content = messages[1].content

        assert "Rusty Nail Tavern" in user_content
        assert "evening" in user_content

    def test_includes_tool_calls(self, session_with_turns: Session):
        """User message should mention tool usage."""
        messages = generate_summary_prompt(session_with_turns)
        user_content = messages[1].content

        assert "lookup_creature" in user_content

    def test_includes_previous_summary(self, session_with_turns: Session):
        """User message should include existing summary."""
        session_with_turns.summary = "Previously, the party arrived in town."
        messages = generate_summary_prompt(session_with_turns)
        user_content = messages[1].content

        assert "Previously, the party arrived in town." in user_content

    def test_since_turn_parameter(self, session_with_turns: Session):
        """since_turn should skip earlier turns."""
        messages = generate_summary_prompt(session_with_turns, since_turn=2)
        user_content = messages[1].content

        # Should not include first turns
        assert "enter the tavern" not in user_content
        # Should include later turn
        assert "bartender" in user_content

    def test_empty_turns_handled(self, empty_session: Session):
        """Should handle session with no turns."""
        messages = generate_summary_prompt(empty_session)
        user_content = messages[1].content

        assert "No turns yet" in user_content


class TestGenerateSummary:
    """Tests for generate_summary function."""

    def test_calls_llm_with_prompt(self, session_with_turns: Session):
        """Should call LLM with generated prompt."""
        mock_llm = MockLLM("Test summary output")

        result = generate_summary(mock_llm, session_with_turns)

        assert len(mock_llm.calls) == 1
        messages, tools = mock_llm.calls[0]
        assert len(messages) == 2
        assert tools is None  # No tools for summarization

    def test_returns_llm_response(self, session_with_turns: Session):
        """Should return the LLM's text response."""
        mock_llm = MockLLM("The party entered the Rusty Nail Tavern...")

        result = generate_summary(mock_llm, session_with_turns)

        assert result == "The party entered the Rusty Nail Tavern..."

    def test_strips_whitespace(self, session_with_turns: Session):
        """Should strip whitespace from response."""
        mock_llm = MockLLM("  Summary with whitespace  \n\n")

        result = generate_summary(mock_llm, session_with_turns)

        assert result == "Summary with whitespace"


class TestRollingSummarizer:
    """Tests for RollingSummarizer class."""

    def test_init_with_defaults(self):
        """Summarizer should use default interval."""
        from gm_agent.config import TURNS_BETWEEN_SUMMARIES

        mock_llm = MockLLM()
        summarizer = RollingSummarizer(mock_llm)

        assert summarizer.llm == mock_llm
        assert summarizer.interval == TURNS_BETWEEN_SUMMARIES
        assert summarizer._last_summarized_turn == 0

    def test_init_with_custom_interval(self):
        """Summarizer should accept custom interval."""
        mock_llm = MockLLM()
        summarizer = RollingSummarizer(mock_llm, interval=3)

        assert summarizer.interval == 3

    def test_maybe_update_empty_session(self, empty_session: Session):
        """Should return None for empty session."""
        mock_llm = MockLLM()
        summarizer = RollingSummarizer(mock_llm)

        result = summarizer.maybe_update(empty_session)

        assert result is None
        assert len(mock_llm.calls) == 0

    def test_maybe_update_force(self, session_with_turns: Session):
        """Force should trigger update regardless of turn count."""
        mock_llm = MockLLM("Forced summary")
        summarizer = RollingSummarizer(mock_llm, interval=100)

        result = summarizer.maybe_update(session_with_turns, force=True)

        assert result == "Forced summary"
        assert len(mock_llm.calls) == 1

    def test_maybe_update_at_interval(self, empty_session: Session):
        """Should update when turn count hits interval."""
        mock_llm = MockLLM("Interval summary")
        summarizer = RollingSummarizer(mock_llm, interval=3)

        # Add 3 turns to hit the interval
        for i in range(3):
            empty_session.turns.append(
                Turn(
                    player_input=f"Turn {i}",
                    gm_response=f"Response {i}",
                    timestamp=datetime.now(),
                )
            )

        result = summarizer.maybe_update(empty_session)

        assert result == "Interval summary"
        assert summarizer._last_summarized_turn == 3

    def test_maybe_update_tracks_position(self, empty_session: Session):
        """Should track last summarized position."""
        mock_llm = MockLLM("Summary")
        summarizer = RollingSummarizer(mock_llm, interval=2)

        # First batch
        for i in range(2):
            empty_session.turns.append(
                Turn(player_input=f"T{i}", gm_response=f"R{i}", timestamp=datetime.now())
            )

        summarizer.maybe_update(empty_session)
        assert summarizer._last_summarized_turn == 2

        # Second batch
        for i in range(2, 4):
            empty_session.turns.append(
                Turn(player_input=f"T{i}", gm_response=f"R{i}", timestamp=datetime.now())
            )

        summarizer.maybe_update(empty_session)
        assert summarizer._last_summarized_turn == 4

    def test_maybe_update_skips_between_intervals(self, empty_session: Session):
        """Should not update between intervals."""
        mock_llm = MockLLM("Summary")
        summarizer = RollingSummarizer(mock_llm, interval=5)

        # Add 3 turns (not at interval)
        for i in range(3):
            empty_session.turns.append(
                Turn(player_input=f"T{i}", gm_response=f"R{i}", timestamp=datetime.now())
            )

        result = summarizer.maybe_update(empty_session)

        assert result is None
        assert len(mock_llm.calls) == 0

    def test_reset_clears_state(self):
        """Reset should clear last summarized position."""
        mock_llm = MockLLM()
        summarizer = RollingSummarizer(mock_llm)
        summarizer._last_summarized_turn = 10

        summarizer.reset()

        assert summarizer._last_summarized_turn == 0
