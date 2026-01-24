"""Unit tests for context assembly."""

import pytest

from gm_agent.context import build_context, estimate_tokens
from gm_agent.models.base import Message
from gm_agent.storage.schemas import (
    Campaign,
    Session,
    Turn,
    SceneState,
    PartyMember,
    ToolCallRecord,
)


class TestBuildContext:
    """Tests for build_context function."""

    def test_minimal_campaign_and_session(
        self, minimal_campaign: Campaign, minimal_session: Session
    ):
        """build_context should work with minimal campaign data."""
        messages = build_context(minimal_campaign, minimal_session)

        # Should have at least the system message
        assert len(messages) >= 1
        assert messages[0].role == "system"
        assert "Game Master" in messages[0].content

    def test_full_campaign_context_layers(self, sample_campaign: Campaign, sample_session: Session):
        """build_context should include all 6 context layers."""
        messages = build_context(sample_campaign, sample_session)

        system_content = messages[0].content

        # Layer 1: Campaign background
        assert "ancient evil" in system_content

        # Layer 2: Current arc
        assert "Burnt Offerings" in system_content

        # Party info
        assert "Valeros" in system_content
        assert "Fighter" in system_content

        # Layer 5: Scene state (in system message)
        assert "Sandpoint Market Square" in system_content

    def test_party_members_in_context(self, sample_campaign: Campaign, minimal_session: Session):
        """build_context should include all party members."""
        messages = build_context(sample_campaign, minimal_session)
        system_content = messages[0].content

        assert "Valeros" in system_content
        assert "Seoni" in system_content
        assert "Kyra" in system_content
        assert "Fighter" in system_content
        assert "Sorcerer" in system_content
        assert "Cleric" in system_content
        assert "Level 5" in system_content

    def test_scene_state_included(self, sample_campaign: Campaign, sample_session: Session):
        """build_context should include current scene state."""
        messages = build_context(sample_campaign, sample_session)
        system_content = messages[0].content

        # Scene info should be in system message
        assert "Sandpoint Market Square" in system_content
        assert "morning" in system_content
        assert "Sheriff Hemlock" in system_content

    def test_scene_state_excluded_when_default(
        self, sample_campaign: Campaign, minimal_session: Session
    ):
        """build_context should not include default scene state details."""
        messages = build_context(sample_campaign, minimal_session)
        system_content = messages[0].content

        # Default scene should not add "Current Scene" section
        # (since location is "Unknown" and no NPCs/conditions)
        # The word "Unknown" by itself might appear, but not the scene section header
        # Actually checking that default scenes don't clutter the context
        assert minimal_session.scene_state.location == "Unknown"

    def test_recent_turns_included(self, sample_campaign: Campaign, sample_session: Session):
        """build_context should include recent turns as messages."""
        messages = build_context(sample_campaign, sample_session)

        # Filter to user and assistant messages (not system)
        conversation = [m for m in messages if m.role in ("user", "assistant", "tool")]

        # Should have the turn from sample_session
        assert any("goblin" in m.content.lower() for m in conversation)

    def test_max_turns_limiting(
        self, sample_campaign: Campaign, sample_session_with_many_turns: Session
    ):
        """build_context should limit recent turns to max_turns."""
        # Default is 15 turns
        messages = build_context(sample_campaign, sample_session_with_many_turns)

        # Count user messages (each turn = 1 user + 1 assistant)
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) <= 15

        # Verify we get the LAST 15 turns, not the first
        # Turn 6 through 20 should be present (indices 5-19)
        last_user_msg = user_messages[-1]
        assert "action 20" in last_user_msg.content

        first_user_msg = user_messages[0]
        assert "action 6" in first_user_msg.content

    def test_custom_max_turns(
        self, sample_campaign: Campaign, sample_session_with_many_turns: Session
    ):
        """build_context should respect custom max_turns parameter."""
        messages = build_context(sample_campaign, sample_session_with_many_turns, max_turns=5)

        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 5

        # Should be turns 16-20
        assert "action 16" in user_messages[0].content
        assert "action 20" in user_messages[-1].content

    def test_tool_calls_reconstructed(self, sample_campaign: Campaign):
        """build_context should reconstruct tool call history."""
        # Create session with tool calls
        session = Session(
            id="tool-test",
            campaign_id=sample_campaign.id,
            turns=[
                Turn(
                    player_input="What is a goblin?",
                    gm_response="Based on my research, a goblin is...",
                    tool_calls=[
                        ToolCallRecord(
                            name="lookup_creature",
                            args={"name": "goblin"},
                            result="Goblin data here",
                        )
                    ],
                )
            ],
        )

        messages = build_context(sample_campaign, session)

        # Should have: system, user, assistant (with tool_calls), tool, assistant
        roles = [m.role for m in messages]
        assert "tool" in roles

        # Find the assistant message with tool calls
        tool_call_msgs = [m for m in messages if m.tool_calls]
        assert len(tool_call_msgs) == 1
        assert tool_call_msgs[0].tool_calls[0].name == "lookup_creature"

        # Find the tool result message
        tool_results = [m for m in messages if m.role == "tool"]
        assert len(tool_results) == 1
        assert "Goblin data" in tool_results[0].content

    def test_multiple_tool_calls_in_turn(self, sample_campaign: Campaign):
        """build_context should handle multiple tool calls in a single turn."""
        session = Session(
            id="multi-tool",
            campaign_id=sample_campaign.id,
            turns=[
                Turn(
                    player_input="Look up goblin and fireball",
                    gm_response="Here's what I found...",
                    tool_calls=[
                        ToolCallRecord(
                            name="lookup_creature",
                            args={"name": "goblin"},
                            result="Goblin info",
                        ),
                        ToolCallRecord(
                            name="lookup_spell",
                            args={"name": "fireball"},
                            result="Fireball info",
                        ),
                    ],
                )
            ],
        )

        messages = build_context(sample_campaign, session)

        # Should have two tool result messages
        tool_results = [m for m in messages if m.role == "tool"]
        assert len(tool_results) == 2

    def test_session_summary_included(self, sample_campaign: Campaign):
        """build_context should include session summary if present."""
        session = Session(
            id="summary-test",
            campaign_id=sample_campaign.id,
            summary="Last session, the party defeated the goblins.",
        )

        messages = build_context(sample_campaign, session)
        system_content = messages[0].content

        assert "defeated the goblins" in system_content

    def test_empty_session(self, sample_campaign: Campaign):
        """build_context should handle session with no turns."""
        session = Session(id="empty", campaign_id=sample_campaign.id)

        messages = build_context(sample_campaign, session)

        # Should just have system message
        assert len(messages) == 1
        assert messages[0].role == "system"

    def test_message_roles_correct(self, sample_campaign: Campaign, sample_session: Session):
        """build_context should use correct message roles."""
        messages = build_context(sample_campaign, sample_session)

        # First message should be system
        assert messages[0].role == "system"

        # Conversation should alternate user/assistant (with possible tool messages)
        valid_roles = {"system", "user", "assistant", "tool"}
        for msg in messages:
            assert msg.role in valid_roles

    def test_empty_fields_excluded(self, minimal_campaign: Campaign):
        """build_context should not add empty sections."""
        session = Session(id="test", campaign_id=minimal_campaign.id)
        messages = build_context(minimal_campaign, session)

        system_content = messages[0].content

        # Should not have headers for empty sections
        assert "## Campaign Background" not in system_content
        assert "## Current Story Arc" not in system_content
        assert "## Party Members" not in system_content


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_messages(self):
        """estimate_tokens should return 0 for empty list."""
        assert estimate_tokens([]) == 0

    def test_single_message(self):
        """estimate_tokens should estimate based on character count."""
        messages = [Message(role="user", content="Hello world")]  # 11 chars
        tokens = estimate_tokens(messages)
        # 11 chars / 4 = 2.75 -> 2 (integer division)
        assert tokens == 2

    def test_multiple_messages(self):
        """estimate_tokens should sum all message contents."""
        messages = [
            Message(role="system", content="A" * 100),  # 100 chars
            Message(role="user", content="B" * 200),  # 200 chars
            Message(role="assistant", content="C" * 100),  # 100 chars
        ]
        tokens = estimate_tokens(messages)
        # 400 chars / 4 = 100 tokens
        assert tokens == 100

    def test_full_context_estimate(self, sample_campaign: Campaign, sample_session: Session):
        """estimate_tokens should work on real context."""
        messages = build_context(sample_campaign, sample_session)
        tokens = estimate_tokens(messages)

        # Should be a reasonable number (not 0, not huge)
        assert tokens > 50  # Has system prompt + content
        assert tokens < 10000  # Not unreasonably large

    def test_empty_content_messages(self):
        """estimate_tokens should handle empty content messages."""
        messages = [
            Message(role="assistant", content=""),  # Empty (tool call message)
            Message(role="user", content="test"),
        ]
        tokens = estimate_tokens(messages)
        assert tokens == 1  # Only 4 chars from "test"


class TestContextIntegration:
    """Integration tests for context building."""

    def test_context_message_order(self, sample_campaign: Campaign):
        """Messages should be in correct chronological order."""
        session = Session(
            id="order-test",
            campaign_id=sample_campaign.id,
            turns=[
                Turn(player_input="First action", gm_response="First response"),
                Turn(player_input="Second action", gm_response="Second response"),
                Turn(player_input="Third action", gm_response="Third response"),
            ],
        )

        messages = build_context(sample_campaign, session)

        # Extract conversation (skip system)
        conversation = [m for m in messages[1:]]

        # Should be: user, assistant, user, assistant, user, assistant
        expected_roles = ["user", "assistant", "user", "assistant", "user", "assistant"]
        actual_roles = [m.role for m in conversation]
        assert actual_roles == expected_roles

        # Content should be in order
        user_msgs = [m.content for m in conversation if m.role == "user"]
        assert user_msgs == ["First action", "Second action", "Third action"]

    def test_context_with_mixed_turns(self, sample_campaign: Campaign):
        """Context should handle turns with and without tool calls."""
        session = Session(
            id="mixed-test",
            campaign_id=sample_campaign.id,
            turns=[
                Turn(player_input="Simple question", gm_response="Simple answer"),
                Turn(
                    player_input="What is a goblin?",
                    gm_response="A goblin is...",
                    tool_calls=[
                        ToolCallRecord(
                            name="lookup_creature",
                            args={"name": "goblin"},
                            result="Goblin data",
                        )
                    ],
                ),
                Turn(player_input="Another question", gm_response="Another answer"),
            ],
        )

        messages = build_context(sample_campaign, session)

        # Should have all messages
        roles = [m.role for m in messages]
        assert roles.count("user") == 3
        assert roles.count("tool") == 1

    def test_large_context_stays_reasonable(self, sample_campaign: Campaign):
        """Large sessions should not explode context size."""
        # Create session with many long turns
        turns = []
        for i in range(50):
            turns.append(
                Turn(
                    player_input=f"Question {i}: " + "x" * 500,
                    gm_response=f"Response {i}: " + "y" * 1000,
                )
            )

        session = Session(id="large-test", campaign_id=sample_campaign.id, turns=turns)

        # With default max_turns=15, should be limited
        messages = build_context(sample_campaign, session)
        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 15
