"""Tests for the GameLoopController automation module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from gm_agent.game_loop import GameLoopController, PlayerMessageBatch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock GMAgent."""
    agent = MagicMock()
    agent.process_turn.return_value = "The GM responds dramatically."
    # Mock campaign with empty preferences (use defaults)
    agent.campaign.preferences = {}
    return agent


@pytest.fixture
def mock_bridge():
    """Create a mock FoundryBridge."""
    bridge = MagicMock()
    bridge.is_connected.return_value = True
    bridge.send_command.return_value = {"success": True}
    return bridge


@pytest.fixture
def controller(mock_agent, mock_bridge):
    """Create a GameLoopController with mocked dependencies.

    Uses batch_window_seconds=0 for immediate processing in tests.
    """
    return GameLoopController(
        campaign_id="test-campaign",
        agent=mock_agent,
        bridge=mock_bridge,
        batch_window_seconds=0,  # Immediate processing for tests
        cooldown_seconds=0,  # Disable NPC rate limiting for tests
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGameLoopControllerInit:
    """Tests for GameLoopController initialization."""

    def test_init(self, mock_agent, mock_bridge):
        """Test controller initialization."""
        controller = GameLoopController(
            campaign_id="my-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
        )

        assert controller.campaign_id == "my-campaign"
        assert controller.agent == mock_agent
        assert controller.bridge == mock_bridge
        assert not controller.enabled
        assert not controller._handlers_registered

    def test_start_enables_controller(self, controller, mock_bridge):
        """Test starting the controller."""
        controller.start()

        assert controller.enabled
        assert controller._handlers_registered
        # Verify event handlers were registered
        assert mock_bridge.on_event.call_count == 2
        mock_bridge.on_event.assert_any_call("playerChat", controller._handle_player_chat)
        mock_bridge.on_event.assert_any_call("combatTurn", controller._handle_combat_turn)

    def test_start_only_registers_handlers_once(self, controller, mock_bridge):
        """Test that handlers are only registered once."""
        controller.start()
        controller.stop()
        controller.start()

        # Should only register handlers once
        assert mock_bridge.on_event.call_count == 2

    def test_stop_disables_controller(self, controller):
        """Test stopping the controller."""
        controller.start()
        controller.stop()

        assert not controller.enabled

    def test_set_bridge_updates_reference(self, controller, mock_agent):
        """Test that set_bridge updates the bridge reference."""
        new_bridge = MagicMock()
        new_bridge.is_connected.return_value = True

        controller.set_bridge(new_bridge)

        assert controller.bridge == new_bridge

    def test_set_bridge_reregisters_handlers_if_enabled(self, controller, mock_agent):
        """Test that set_bridge re-registers handlers if controller was enabled."""
        controller.start()
        assert controller._handlers_registered

        new_bridge = MagicMock()
        new_bridge.is_connected.return_value = True

        controller.set_bridge(new_bridge)

        # Handlers should be registered on the new bridge
        assert new_bridge.on_event.call_count == 2
        new_bridge.on_event.assert_any_call("playerChat", controller._handle_player_chat)
        new_bridge.on_event.assert_any_call("combatTurn", controller._handle_combat_turn)

    def test_set_bridge_does_not_register_handlers_if_disabled(self, controller, mock_agent):
        """Test that set_bridge doesn't register handlers if controller was disabled."""
        # Controller never started, so disabled
        new_bridge = MagicMock()
        new_bridge.is_connected.return_value = True

        controller.set_bridge(new_bridge)

        # No handlers registered since controller is disabled
        new_bridge.on_event.assert_not_called()

    def test_reconnection_flow(self, mock_agent):
        """Test full reconnection flow: connect, disconnect, reconnect."""
        # Initial bridge
        bridge1 = MagicMock()
        bridge1.is_connected.return_value = True

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=bridge1,
            batch_window_seconds=0,
        )
        controller.start()

        # Verify initial setup
        assert controller.enabled
        assert bridge1.on_event.call_count == 2

        # Simulate disconnect
        controller.stop()
        assert not controller.enabled

        # Simulate reconnect with new bridge
        bridge2 = MagicMock()
        bridge2.is_connected.return_value = True
        bridge2.send_command.return_value = {"success": True}

        controller.set_bridge(bridge2)
        controller.start()

        # Verify handlers registered on new bridge
        assert controller.enabled
        assert bridge2.on_event.call_count == 2

        # Verify controller works with new bridge
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Hello",
            }
        )
        bridge2.send_command.assert_called()


# =============================================================================
# Player Chat Event Tests
# =============================================================================


class TestPlayerChatHandler:
    """Tests for player chat event handling."""

    def test_handle_player_chat_when_disabled(self, controller, mock_agent):
        """Test that player chat is ignored when disabled."""
        controller.enabled = False

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I search the room",
            }
        )

        mock_agent.process_turn.assert_not_called()

    def test_handle_player_chat_success(self, controller, mock_agent, mock_bridge):
        """Test successful player chat handling."""
        controller.enabled = True

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I search the bookshelf for hidden compartments",
                "playerId": "user-123",
            }
        )

        # Agent should be called with formatted prompt
        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Valeros" in call_args
        assert "search the bookshelf" in call_args

        # Response should be posted to chat
        mock_bridge.send_command.assert_called_with(
            "createChat",
            {"content": "The GM responds dramatically.", "speaker": "GM Agent"},
        )

    def test_handle_player_chat_empty_content(self, controller, mock_agent):
        """Test that empty content is ignored."""
        controller.enabled = True

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "",
            }
        )

        mock_agent.process_turn.assert_not_called()

    def test_handle_player_chat_missing_content(self, controller, mock_agent):
        """Test that missing content is ignored."""
        controller.enabled = True

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
            }
        )

        mock_agent.process_turn.assert_not_called()

    def test_handle_player_chat_agent_error(self, controller, mock_agent, mock_bridge):
        """Test error handling when agent fails."""
        controller.enabled = True
        mock_agent.process_turn.side_effect = RuntimeError("LLM error")

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I attack the dragon",
            }
        )

        # Should post error message
        mock_bridge.send_command.assert_called()
        call_args = mock_bridge.send_command.call_args[0]
        assert "error" in call_args[1]["content"].lower()

    def test_handle_player_chat_llm_unavailable(self, controller, mock_agent, mock_bridge):
        """Test error handling when LLM is unavailable."""
        from gm_agent.models import LLMUnavailableError

        controller.enabled = True
        mock_agent.process_turn.side_effect = LLMUnavailableError("Connection refused")

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I search the room",
            }
        )

        # Should post a user-friendly unavailable message
        mock_bridge.send_command.assert_called()
        call_args = mock_bridge.send_command.call_args[0]
        content = call_args[1]["content"]
        assert "unavailable" in content.lower()
        assert "LLM" in content or "offline" in content.lower()


# =============================================================================
# Combat Turn Event Tests
# =============================================================================


class TestCombatTurnHandler:
    """Tests for combat turn event handling."""

    def test_handle_combat_turn_when_disabled(self, controller, mock_agent):
        """Test that combat turns are ignored when disabled."""
        controller.enabled = False

        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 0,
                "combatant": {"name": "Goblin", "isNPC": True},
            }
        )

        mock_agent.process_turn.assert_not_called()

    def test_handle_combat_turn_player_turn(self, controller, mock_agent):
        """Test that player turns are ignored (wait for player input)."""
        controller.enabled = True

        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 0,
                "combatant": {"name": "Valeros", "isNPC": False},
            }
        )

        mock_agent.process_turn.assert_not_called()

    def test_handle_combat_turn_npc_full_control(self, controller, mock_agent, mock_bridge):
        """Test NPC turn with full control (NPC not designated for ACA)."""
        controller.enabled = True
        # ACA active but this NPC not designated
        mock_bridge.send_command.return_value = {
            "active": True,
            "designations": {"other-actor": "ai"},
        }

        controller._handle_combat_turn(
            {
                "round": 2,
                "turn": 1,
                "combatant": {
                    "name": "Goblin Boss",
                    "isNPC": True,
                    "actorId": "actor-123",
                },
            }
        )

        # Agent should be called with combat prompt
        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Goblin Boss" in call_args
        assert "turn" in call_args.lower()
        assert "combat" in call_args.lower()

    def test_handle_combat_turn_npc_aca_controlling(self, controller, mock_agent, mock_bridge):
        """Test NPC turn with AI Combat Assistant controlling this NPC."""
        controller.enabled = True
        mock_bridge.send_command.side_effect = [
            # getACAState - this NPC designated for AI
            {"active": True, "designations": {"actor-123": "ai"}},
            # getACATurnState
            {"actorName": "Goblin Boss", "actionsRemaining": 3},
        ]

        controller._handle_combat_turn(
            {
                "round": 2,
                "turn": 1,
                "combatant": {
                    "name": "Goblin Boss",
                    "isNPC": True,
                    "actorId": "actor-123",
                },
            }
        )

        # Agent should be called with narration prompt
        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Goblin Boss" in call_args
        assert "narrative" in call_args.lower() or "AI Combat Assistant" in call_args

    def test_handle_combat_turn_missing_combatant(self, controller, mock_agent):
        """Test handling turn event with no combatant info."""
        controller.enabled = True

        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 0,
            }
        )

        mock_agent.process_turn.assert_not_called()

    def test_handle_combat_turn_empty_combatant(self, controller, mock_agent):
        """Test handling turn event with empty combatant."""
        controller.enabled = True

        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 0,
                "combatant": {},
            }
        )

        mock_agent.process_turn.assert_not_called()


# =============================================================================
# AI Combat Assistant Detection Tests
# =============================================================================


class TestAICombatAssistantDetection:
    """Tests for AI Combat Assistant detection."""

    def test_aca_active_true(self, controller, mock_bridge):
        """Test detecting active AI Combat Assistant."""
        mock_bridge.send_command.return_value = {"active": True}

        result = controller._is_ai_combat_assistant_active()

        assert result is True
        mock_bridge.send_command.assert_called_with("getACAState", timeout=2.0)

    def test_aca_active_false(self, controller, mock_bridge):
        """Test detecting inactive AI Combat Assistant."""
        mock_bridge.send_command.return_value = {"active": False}

        result = controller._is_ai_combat_assistant_active()

        assert result is False

    def test_aca_missing_key(self, controller, mock_bridge):
        """Test handling missing active key."""
        mock_bridge.send_command.return_value = {"installed": True}

        result = controller._is_ai_combat_assistant_active()

        assert result is False

    def test_aca_bridge_error(self, controller, mock_bridge):
        """Test handling bridge error when checking ACA."""
        mock_bridge.send_command.side_effect = ConnectionError("Disconnected")

        result = controller._is_ai_combat_assistant_active()

        assert result is False

    def test_aca_timeout(self, controller, mock_bridge):
        """Test handling timeout when checking ACA."""
        mock_bridge.send_command.side_effect = TimeoutError("Timeout")

        result = controller._is_ai_combat_assistant_active()

        assert result is False


# =============================================================================
# ACA Per-NPC Status Tests
# =============================================================================


class TestACAPerNPCStatus:
    """Tests for per-NPC ACA status detection."""

    def test_get_aca_status_npc_controlled_by_aca(self, controller, mock_bridge):
        """Test detecting NPC controlled by ACA."""
        mock_bridge.send_command.side_effect = [
            # First call: getACAState
            {
                "active": True,
                "designations": {"actor-123": "ai", "actor-456": "friendly"},
            },
            # Second call: getACATurnState
            {
                "actorName": "Goblin Boss",
                "isProcessing": True,
                "actionsRemaining": 2,
                "actionHistory": ["Strike (hit)"],
            },
        ]

        result = controller._get_aca_status_for_npc("actor-123", "Goblin Boss")

        assert result["controls_this_npc"] is True
        assert result["aca_active"] is True
        assert result["turn_state"]["actionsRemaining"] == 2

    def test_get_aca_status_npc_not_controlled(self, controller, mock_bridge):
        """Test detecting NPC not controlled by ACA."""
        mock_bridge.send_command.return_value = {
            "active": True,
            "designations": {"actor-456": "ai"},
        }

        result = controller._get_aca_status_for_npc("actor-123", "Goblin")

        assert result["controls_this_npc"] is False
        assert result["aca_active"] is True
        assert result["turn_state"] is None

    def test_get_aca_status_aca_inactive(self, controller, mock_bridge):
        """Test when ACA is not active."""
        mock_bridge.send_command.return_value = {"active": False}

        result = controller._get_aca_status_for_npc("actor-123", "Goblin")

        assert result["controls_this_npc"] is False
        assert result["aca_active"] is False

    def test_get_aca_status_by_name_fallback(self, controller, mock_bridge):
        """Test fallback to name matching when actor ID not found."""
        mock_bridge.send_command.side_effect = [
            # First call: getACAState (designation by name)
            {
                "active": True,
                "designations": {"Goblin Boss": "ai"},
            },
            # Second call: getACATurnState
            {"actorName": "Goblin Boss", "isProcessing": False},
        ]

        # Actor ID not in designations, but name matches
        result = controller._get_aca_status_for_npc("actor-999", "Goblin Boss")

        assert result["controls_this_npc"] is True

    def test_get_aca_status_bridge_error(self, controller, mock_bridge):
        """Test graceful handling of bridge errors."""
        mock_bridge.send_command.side_effect = ConnectionError("Disconnected")

        result = controller._get_aca_status_for_npc("actor-123", "Goblin")

        # Should default to GM Agent control
        assert result["controls_this_npc"] is False
        assert result["aca_active"] is False

    def test_get_aca_status_turn_state_error(self, controller, mock_bridge):
        """Test handling turn state fetch error."""
        mock_bridge.send_command.side_effect = [
            # ACA state succeeds
            {"active": True, "designations": {"actor-123": "ai"}},
            # Turn state fails
            TimeoutError("Timeout"),
        ]

        result = controller._get_aca_status_for_npc("actor-123", "Goblin")

        # Should still know ACA controls, just no turn state
        assert result["controls_this_npc"] is True
        assert result["turn_state"] is None


# =============================================================================
# Response Posting Tests
# =============================================================================


class TestPostResponse:
    """Tests for response posting to Foundry."""

    def test_post_response_success(self, controller, mock_bridge):
        """Test successful response posting."""
        controller._post_response("The goblin snarls!")

        mock_bridge.send_command.assert_called_with(
            "createChat",
            {"content": "The goblin snarls!", "speaker": "GM Agent"},
        )

    def test_post_response_empty(self, controller, mock_bridge):
        """Test that empty responses are not posted."""
        controller._post_response("")

        mock_bridge.send_command.assert_not_called()

    def test_post_response_none(self, controller, mock_bridge):
        """Test that None responses are not posted."""
        controller._post_response(None)

        mock_bridge.send_command.assert_not_called()

    def test_post_response_bridge_error(self, controller, mock_bridge):
        """Test graceful handling of bridge errors."""
        mock_bridge.send_command.side_effect = ConnectionError("Disconnected")

        # Should not raise
        controller._post_response("Test message")


# =============================================================================
# NPC Turn Handling Tests
# =============================================================================


class TestNPCTurnHandling:
    """Tests for NPC turn execution and narration."""

    def test_run_npc_turn_success(self, controller, mock_agent, mock_bridge):
        """Test successful NPC turn execution."""
        mock_agent.process_turn.return_value = "The goblin lunges at the fighter!"

        controller._run_npc_turn("Goblin")

        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Goblin" in call_args
        assert "turn" in call_args.lower()

        mock_bridge.send_command.assert_called()

    def test_run_npc_turn_agent_error(self, controller, mock_agent, mock_bridge):
        """Test NPC turn with agent error."""
        mock_agent.process_turn.side_effect = RuntimeError("LLM unavailable")

        controller._run_npc_turn("Goblin")

        # Should post fallback message
        mock_bridge.send_command.assert_called()
        call_args = mock_bridge.send_command.call_args[0][1]
        assert "hesitates" in call_args["content"].lower()

    def test_narrate_npc_turn_success(self, controller, mock_agent, mock_bridge):
        """Test successful NPC turn narration."""
        mock_agent.process_turn.return_value = "The goblin acts with cunning speed..."

        controller._narrate_npc_turn("Goblin")

        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Goblin" in call_args
        assert "AI Combat Assistant" in call_args

        mock_bridge.send_command.assert_called()

    def test_narrate_npc_turn_with_turn_state(self, controller, mock_agent, mock_bridge):
        """Test NPC narration with ACA turn state context."""
        mock_agent.process_turn.return_value = "The goblin follows up with another strike..."

        turn_state = {
            "actionsRemaining": 1,
            "actionHistory": ["Strike (hit)", "Move"],
            "permanentNotes": "Aggressive flanker",
        }

        controller._narrate_npc_turn("Goblin", turn_state)

        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Strike (hit)" in call_args or "Actions so far" in call_args
        assert "Aggressive flanker" in call_args or "Tactical notes" in call_args
        assert "1" in call_args  # Actions remaining

    def test_narrate_npc_turn_agent_error(self, controller, mock_agent, mock_bridge):
        """Test NPC narration with agent error."""
        mock_agent.process_turn.side_effect = RuntimeError("LLM unavailable")

        controller._narrate_npc_turn("Goblin")

        # Should post fallback message
        mock_bridge.send_command.assert_called()
        call_args = mock_bridge.send_command.call_args[0][1]
        assert "acts swiftly" in call_args["content"].lower()


# =============================================================================
# Message Batching Tests
# =============================================================================


class TestMessageBatching:
    """Tests for player message batching functionality."""

    def test_immediate_mode_no_batching(self, mock_agent, mock_bridge):
        """Test that batch_window_seconds=0 processes messages immediately."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,  # Immediate processing
        )
        controller.enabled = True

        for i in range(5):
            controller._handle_player_chat(
                {
                    "actorName": "Valeros",
                    "content": f"Action {i}",
                    "playerId": "player-1",
                }
            )

        # Each message processed separately
        assert mock_agent.process_turn.call_count == 5

    def test_batching_combines_messages(self, mock_agent, mock_bridge):
        """Test that messages within window are combined into single prompt."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=1.0,  # 1 second window
        )
        controller.enabled = True

        # Send multiple messages (they'll be batched)
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I search the room",
                "playerId": "player-1",
            }
        )
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "And check for traps",
                "playerId": "player-1",
            }
        )

        # Not processed yet (waiting for window)
        assert mock_agent.process_turn.call_count == 0

        # Manually flush the batch
        with controller._batch_lock:
            controller._flush_player_batch("player-1")

        # Now processed as single combined prompt
        assert mock_agent.process_turn.call_count == 1
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "I search the room" in call_args
        assert "And check for traps" in call_args

    def test_max_batch_size_triggers_flush(self, mock_agent, mock_bridge):
        """Test that reaching max_batch_size triggers immediate flush."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=10.0,  # Long window
            max_batch_size=3,  # But small batch size
        )
        controller.enabled = True

        # Send messages up to max
        for i in range(3):
            controller._handle_player_chat(
                {
                    "actorName": "Valeros",
                    "content": f"Action {i + 1}",
                    "playerId": "player-1",
                }
            )

        # Should have flushed after 3rd message
        assert mock_agent.process_turn.call_count == 1

    def test_different_players_separate_batches(self, mock_agent, mock_bridge):
        """Test that different players have separate batches."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=1.0,
        )
        controller.enabled = True

        # Player 1 sends message
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I attack!",
                "playerId": "player-1",
            }
        )
        # Player 2 sends message
        controller._handle_player_chat(
            {
                "actorName": "Seoni",
                "content": "I cast a spell!",
                "playerId": "player-2",
            }
        )

        # Both should have pending batches
        assert len(controller._player_batches) == 2

        # Flush both
        with controller._batch_lock:
            controller._flush_player_batch("player-1")
            controller._flush_player_batch("player-2")

        # Both processed separately
        assert mock_agent.process_turn.call_count == 2

    def test_batch_timer_flushes_after_window(self, mock_agent, mock_bridge):
        """Test that batch is flushed after window expires."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0.1,  # Short window for test
        )
        controller.enabled = True

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action",
                "playerId": "player-1",
            }
        )

        # Wait for timer to fire
        time.sleep(0.2)

        # Should have been processed
        assert mock_agent.process_turn.call_count == 1

    def test_npc_rate_limit_separate_from_batching(self, mock_agent, mock_bridge):
        """Test that NPC rate limiting is separate from player batching."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,  # Immediate for players
            cooldown_seconds=1.0,  # Rate limit for NPCs
        )
        controller.enabled = True
        mock_bridge.send_command.return_value = {"active": False}

        # Player message processed immediately
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I attack!",
                "playerId": "player-1",
            }
        )
        assert mock_agent.process_turn.call_count == 1

        # First NPC turn goes through
        controller._handle_combat_turn(
            {
                "combatant": {"name": "Goblin", "isNPC": True, "actorId": "npc-1"},
            }
        )
        assert mock_agent.process_turn.call_count == 2

        # Second NPC turn blocked by rate limit
        controller._handle_combat_turn(
            {
                "combatant": {"name": "Orc", "isNPC": True, "actorId": "npc-2"},
            }
        )
        assert mock_agent.process_turn.call_count == 2  # Blocked

        # Player can still send (not rate limited)
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Another attack!",
                "playerId": "player-1",
            }
        )
        assert mock_agent.process_turn.call_count == 3

    def test_stop_flushes_pending_batches(self, mock_agent, mock_bridge):
        """Test that stop() flushes pending batches by default."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=10.0,  # Long window
        )
        controller.enabled = True

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Pending action",
                "playerId": "player-1",
            }
        )

        # Not processed yet
        assert mock_agent.process_turn.call_count == 0

        # Stop flushes pending
        controller.stop(flush_pending=True)

        # Now processed
        assert mock_agent.process_turn.call_count == 1
        assert not controller.enabled

    def test_stop_can_discard_pending(self, mock_agent, mock_bridge):
        """Test that stop(flush_pending=False) discards pending batches."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=10.0,
        )
        controller.enabled = True

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Pending action",
                "playerId": "player-1",
            }
        )

        # Stop without flushing
        controller.stop(flush_pending=False)

        # Not processed
        assert mock_agent.process_turn.call_count == 0
        assert len(controller._player_batches) == 0

    def test_batched_message_count_tracked(self, mock_agent, mock_bridge):
        """Test that batched message count is tracked in stats."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=1.0,
            max_batch_size=3,
        )
        controller.enabled = True

        # Send 3 messages (triggers flush at max)
        for i in range(3):
            controller._handle_player_chat(
                {
                    "actorName": "Valeros",
                    "content": f"Action {i + 1}",
                    "playerId": "player-1",
                }
            )

        stats = controller.get_stats()
        assert stats["batched_message_count"] == 3
        assert stats["player_chat_count"] == 1  # One batch response


# =============================================================================
# Stats Tracking Tests
# =============================================================================


class TestStatsTracking:
    """Tests for automation statistics tracking."""

    def test_get_stats_initial(self, mock_agent, mock_bridge):
        """Test initial stats are zero."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
        )

        stats = controller.get_stats()

        assert stats["enabled"] is False
        assert stats["response_count"] == 0
        assert stats["player_chat_count"] == 0
        assert stats["npc_turn_count"] == 0
        assert stats["batched_message_count"] == 0
        assert stats["average_batch_size"] is None  # No batches yet
        assert stats["pending_batches"] == 0
        assert stats["pending_messages"] == 0
        assert stats["oldest_batch_age_seconds"] is None
        assert stats["error_count"] == 0
        assert stats["total_processing_time_ms"] == 0.0
        assert stats["started_at"] is None
        assert stats["last_response_time"] is None

    def test_get_stats_after_start(self, mock_agent, mock_bridge):
        """Test stats after automation starts."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
        )
        controller.start()

        stats = controller.get_stats()

        assert stats["enabled"] is True
        assert stats["started_at"] is not None
        assert stats["uptime_seconds"] is not None
        assert stats["uptime_seconds"] >= 0

    def test_stats_track_player_chat(self, controller, mock_agent, mock_bridge):
        """Test stats track player chat responses."""
        controller.start()

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action",
                "playerId": "player-1",
            }
        )

        stats = controller.get_stats()

        assert stats["response_count"] == 1
        assert stats["player_chat_count"] == 1
        assert stats["npc_turn_count"] == 0
        assert stats["batched_message_count"] == 1
        assert stats["last_response_time"] is not None

    def test_stats_track_npc_turn(self, controller, mock_agent, mock_bridge):
        """Test stats track NPC turn responses."""
        controller.start()
        mock_bridge.send_command.return_value = {"active": False}

        controller._handle_combat_turn(
            {
                "combatant": {"name": "Goblin", "isNPC": True, "actorId": "npc-1"},
            }
        )

        stats = controller.get_stats()

        assert stats["response_count"] == 1
        assert stats["player_chat_count"] == 0
        assert stats["npc_turn_count"] == 1

    def test_stats_track_multiple_players(self, mock_agent, mock_bridge):
        """Test stats track multiple players' messages."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,  # Immediate processing
        )
        controller.start()

        # Multiple players send messages
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 1",
                "playerId": "player-1",
            }
        )
        controller._handle_player_chat(
            {
                "actorName": "Seoni",
                "content": "Action 2",
                "playerId": "player-2",
            }
        )
        controller._handle_player_chat(
            {
                "actorName": "Merisiel",
                "content": "Action 3",
                "playerId": "player-3",
            }
        )

        stats = controller.get_stats()

        assert stats["response_count"] == 3
        assert stats["player_chat_count"] == 3
        assert stats["batched_message_count"] == 3

    def test_stats_track_errors(self, mock_agent, mock_bridge):
        """Test stats track errors."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,  # Immediate processing
        )
        controller.start()
        mock_agent.process_turn.side_effect = RuntimeError("Test error")

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action",
                "playerId": "player-1",
            }
        )

        stats = controller.get_stats()

        assert stats["error_count"] == 1
        assert stats["response_count"] == 1  # Error responses still count

    def test_stats_processing_time_tracked(self, mock_agent, mock_bridge):
        """Test that processing time is tracked in stats."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Simulate a slow LLM response
        def slow_response(*args, **kwargs):
            time.sleep(0.05)  # 50ms
            return "Response"

        mock_agent.process_turn.side_effect = slow_response

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action",
                "playerId": "player-1",
            }
        )

        stats = controller.get_stats()

        # Should have tracked at least 50ms of processing time
        assert stats["total_processing_time_ms"] >= 50

    def test_stats_average_batch_size(self, mock_agent, mock_bridge):
        """Test average batch size calculation."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Send 3 messages from 3 players (1 message each = avg 1.0)
        for i in range(3):
            controller._handle_player_chat(
                {
                    "actorName": f"Player{i}",
                    "content": f"Action {i}",
                    "playerId": f"player-{i}",
                }
            )

        stats = controller.get_stats()

        assert stats["average_batch_size"] == 1.0
        assert stats["batched_message_count"] == 3
        assert stats["player_chat_count"] == 3

    def test_stats_oldest_batch_age(self, mock_agent, mock_bridge):
        """Test oldest pending batch age tracking."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=10.0,  # Long window to keep batch pending
            max_batch_size=10,  # High limit to avoid flush
        )
        controller.start()

        # Add a message to create a pending batch
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action",
                "playerId": "player-1",
            }
        )

        # Wait a bit
        time.sleep(0.05)

        stats = controller.get_stats()

        assert stats["pending_batches"] == 1
        assert stats["oldest_batch_age_seconds"] is not None
        assert stats["oldest_batch_age_seconds"] >= 0.05

        # Clean up
        controller.stop(flush_pending=False)

    def test_stats_oldest_batch_age_none_when_no_pending(self, mock_agent, mock_bridge):
        """Test oldest batch age is None when no pending batches."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,  # Immediate flush
        )
        controller.start()

        stats = controller.get_stats()

        assert stats["pending_batches"] == 0
        assert stats["oldest_batch_age_seconds"] is None

    def test_reset_stats(self, mock_agent, mock_bridge):
        """Test reset_stats clears all counters."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Accumulate some stats
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action",
                "playerId": "player-1",
            }
        )
        mock_bridge.send_command.return_value = {"active": False}
        controller._handle_combat_turn(
            {
                "combatant": {"name": "Goblin", "isNPC": True, "actorId": "npc-1"},
            }
        )

        # Verify stats accumulated
        stats = controller.get_stats()
        assert stats["response_count"] > 0
        assert stats["player_chat_count"] > 0
        assert stats["npc_turn_count"] > 0

        # Reset stats
        controller.reset_stats()

        stats = controller.get_stats()
        assert stats["response_count"] == 0
        assert stats["player_chat_count"] == 0
        assert stats["npc_turn_count"] == 0
        assert stats["error_count"] == 0
        assert stats["batched_message_count"] == 0
        assert stats["total_processing_time_ms"] == 0.0
        assert stats["last_response_time"] is None
        # started_at should NOT be reset
        assert stats["started_at"] is not None
        # enabled should NOT change
        assert stats["enabled"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestGameLoopIntegration:
    """Integration tests for the game loop flow."""

    def test_exploration_flow(self, controller, mock_agent, mock_bridge):
        """Test complete exploration flow."""
        controller.start()

        # Player describes action
        controller._handle_player_chat(
            {
                "actorName": "Ezren",
                "content": "I cast Detect Magic on the altar",
                "playerId": "player-1",
            }
        )

        # Verify agent was called
        mock_agent.process_turn.assert_called()

        # Verify response was posted
        mock_bridge.send_command.assert_called()

    def test_combat_flow_full_control(self, controller, mock_agent, mock_bridge):
        """Test complete combat flow with full NPC control."""
        controller.start()
        # ACA active but not controlling this NPC
        mock_bridge.send_command.return_value = {"active": False}

        # NPC's turn comes up
        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 2,
                "combatant": {"name": "Orc Warrior", "isNPC": True, "actorId": "orc-1"},
            }
        )

        # Verify agent was called with combat context
        mock_agent.process_turn.assert_called()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "Orc Warrior" in call_args

    def test_combat_flow_aca_coexistence(self, controller, mock_agent, mock_bridge):
        """Test combat flow with AI Combat Assistant handling tactics."""
        controller.start()
        mock_bridge.send_command.side_effect = [
            {"active": True, "designations": {"orc-1": "ai"}},
            {"actorName": "Orc Warrior", "actionsRemaining": 3},
        ]

        # NPC's turn comes up
        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 2,
                "combatant": {"name": "Orc Warrior", "isNPC": True, "actorId": "orc-1"},
            }
        )

        # Verify agent was called with narration focus
        mock_agent.process_turn.assert_called()
        call_args = mock_agent.process_turn.call_args[0][0]
        assert "narrative" in call_args.lower() or "AI Combat Assistant" in call_args

    def test_mixed_turns(self, controller, mock_agent, mock_bridge):
        """Test handling mixed player/NPC turns."""
        controller.start()
        # ACA not active
        mock_bridge.send_command.return_value = {"active": False}

        # Player turn - should be ignored
        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 0,
                "combatant": {"name": "Fighter", "isNPC": False},
            }
        )
        assert mock_agent.process_turn.call_count == 0

        # NPC turn - should be handled
        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 1,
                "combatant": {"name": "Goblin", "isNPC": True, "actorId": "goblin-1"},
            }
        )
        assert mock_agent.process_turn.call_count == 1

        # Player turn again - should be ignored
        controller._handle_combat_turn(
            {
                "round": 1,
                "turn": 2,
                "combatant": {"name": "Wizard", "isNPC": False},
            }
        )
        assert mock_agent.process_turn.call_count == 1

        # Player action via chat
        controller._handle_player_chat(
            {
                "actorName": "Wizard",
                "content": "I cast Magic Missile at the goblin",
            }
        )
        assert mock_agent.process_turn.call_count == 2


# =============================================================================
# Error Threshold Auto-Disable Tests
# =============================================================================


class TestErrorThresholdAutoDisable:
    """Tests for error threshold auto-disable functionality."""

    def test_consecutive_errors_reset_on_success(self, mock_agent, mock_bridge):
        """Test that consecutive errors are reset after a successful response."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=5,
        )
        controller.enabled = True
        controller._consecutive_errors = 3  # Simulate prior errors

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action",
                "playerId": "player-1",
            }
        )

        assert controller._consecutive_errors == 0
        assert controller.enabled is True

    def test_error_increments_consecutive_count(self, mock_agent, mock_bridge):
        """Test that errors increment the consecutive error counter."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=5,
        )
        controller.enabled = True
        mock_agent.process_turn.side_effect = RuntimeError("Test error")

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action",
                "playerId": "player-1",
            }
        )

        assert controller._consecutive_errors == 1
        assert controller._error_count == 1
        assert controller.enabled is True  # Not yet at threshold

    def test_auto_disable_after_max_consecutive_errors(self, mock_agent, mock_bridge):
        """Test that automation is disabled after max consecutive errors."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=3,
        )
        controller.enabled = True
        mock_agent.process_turn.side_effect = RuntimeError("Test error")

        # First error
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 1",
                "playerId": "player-1",
            }
        )
        assert controller.enabled is True
        assert controller._consecutive_errors == 1

        # Second error
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 2",
                "playerId": "player-1",
            }
        )
        assert controller.enabled is True
        assert controller._consecutive_errors == 2

        # Third error - should trigger auto-disable
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 3",
                "playerId": "player-1",
            }
        )
        assert controller.enabled is False
        assert controller._consecutive_errors == 3

    def test_auto_disable_posts_notification(self, mock_agent, mock_bridge):
        """Test that auto-disable posts a notification to chat."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=1,  # Disable after first error
        )
        controller.enabled = True
        mock_agent.process_turn.side_effect = RuntimeError("Test error")

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action",
                "playerId": "player-1",
            }
        )

        # Find the auto-disable notification call
        calls = mock_bridge.send_command.call_args_list
        notification_call = None
        for call in calls:
            args = call[0]
            if args[0] == "createChat":
                content = args[1].get("content", "")
                if "disabled" in content.lower() and "consecutive errors" in content.lower():
                    notification_call = call
                    break

        assert notification_call is not None, "Auto-disable notification not found in chat"

    def test_consecutive_errors_in_stats(self, mock_agent, mock_bridge):
        """Test that consecutive errors are included in stats."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            max_consecutive_errors=5,
        )
        controller._consecutive_errors = 3

        stats = controller.get_stats()

        assert stats["consecutive_errors"] == 3
        assert stats["max_consecutive_errors"] == 5

    def test_npc_turn_error_triggers_auto_disable(self, mock_agent, mock_bridge):
        """Test that NPC turn errors also trigger auto-disable."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            cooldown_seconds=0,
            max_consecutive_errors=2,
        )
        controller.enabled = True
        mock_agent.process_turn.side_effect = RuntimeError("Test error")
        # ACA not active
        mock_bridge.send_command.return_value = {"active": False}

        # First NPC turn error
        controller._handle_combat_turn(
            {
                "combatant": {"name": "Goblin", "isNPC": True, "actorId": "npc-1"},
            }
        )
        assert controller.enabled is True
        assert controller._consecutive_errors == 1

        # Second NPC turn error - should trigger auto-disable
        controller._handle_combat_turn(
            {
                "combatant": {"name": "Orc", "isNPC": True, "actorId": "npc-2"},
            }
        )
        assert controller.enabled is False
        assert controller._consecutive_errors == 2

    def test_mixed_success_resets_consecutive(self, mock_agent, mock_bridge):
        """Test that a success between errors resets the consecutive count."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=3,
        )
        controller.enabled = True

        # First call fails
        mock_agent.process_turn.side_effect = RuntimeError("Error 1")
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 1",
                "playerId": "player-1",
            }
        )
        assert controller._consecutive_errors == 1

        # Second call succeeds
        mock_agent.process_turn.side_effect = None
        mock_agent.process_turn.return_value = "Success!"
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 2",
                "playerId": "player-1",
            }
        )
        assert controller._consecutive_errors == 0

        # Third call fails
        mock_agent.process_turn.side_effect = RuntimeError("Error 2")
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 3",
                "playerId": "player-1",
            }
        )
        assert controller._consecutive_errors == 1

        # Still enabled because not at threshold
        assert controller.enabled is True
        assert controller._error_count == 2  # Total errors

    def test_llm_unavailable_counts_as_error(self, mock_agent, mock_bridge):
        """Test that LLMUnavailableError counts toward consecutive errors."""
        from gm_agent.models import LLMUnavailableError

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=2,
        )
        controller.enabled = True
        mock_agent.process_turn.side_effect = LLMUnavailableError("Connection refused")

        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action",
                "playerId": "player-1",
            }
        )

        assert controller._consecutive_errors == 1

        # Second error should disable
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Test action 2",
                "playerId": "player-1",
            }
        )

        assert controller._consecutive_errors == 2
        assert controller.enabled is False

    def test_default_max_consecutive_errors(self, mock_agent, mock_bridge):
        """Test that default max_consecutive_errors is 5."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
        )

        assert controller.max_consecutive_errors == 5

    def test_start_resets_consecutive_errors(self, mock_agent, mock_bridge):
        """Test that start() resets consecutive errors for a fresh start."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            max_consecutive_errors=3,
        )
        controller._consecutive_errors = 2  # Simulate prior errors

        controller.start()

        assert controller._consecutive_errors == 0
        assert controller.enabled is True

    def test_re_enable_after_auto_disable(self, mock_agent, mock_bridge):
        """Test full flow: auto-disable then re-enable with fresh error count."""
        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            max_consecutive_errors=2,
        )
        controller.start()
        mock_agent.process_turn.side_effect = RuntimeError("Error")

        # Trigger auto-disable
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 1",
                "playerId": "player-1",
            }
        )
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "Action 2",
                "playerId": "player-1",
            }
        )

        assert controller.enabled is False
        assert controller._consecutive_errors == 2

        # Re-enable - should reset consecutive errors
        controller.start()

        assert controller.enabled is True
        assert controller._consecutive_errors == 0
