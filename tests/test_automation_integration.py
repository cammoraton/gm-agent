"""Integration tests for the full automation pipeline.

These tests verify that components are wired together correctly:
- API SocketIO handlers → GameLoopController
- GameLoopController → GMAgent → FoundryBridge
- Event flow from Foundry through the full chain
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from gm_agent.game_loop import GameLoopController
from gm_agent.mcp.foundry_vtt import FoundryBridge

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_socketio():
    """Create a mock SocketIO instance."""
    return MagicMock()


@pytest.fixture
def mock_agent():
    """Create a mock GMAgent."""
    agent = MagicMock()
    agent.process_turn.return_value = "The GM responds to your action."
    # Mock campaign with empty preferences (use defaults)
    agent.campaign.preferences = {}
    return agent


@pytest.fixture
def mock_bridge(mock_socketio):
    """Create a mock FoundryBridge."""
    bridge = FoundryBridge(mock_socketio)
    bridge.set_connected("test-sid")
    bridge.send_command = MagicMock(return_value={"success": True})
    return bridge


# =============================================================================
# Full Event Chain Tests
# =============================================================================


class TestFullEventChain:
    """Tests verifying the complete event flow through all components."""

    def test_player_chat_full_chain(self, mock_agent, mock_bridge):
        """Test full chain: playerChat event → agent.process_turn → createChat."""
        # Set up controller
        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Simulate playerChat event
        controller._handle_player_chat(
            {
                "actorName": "Valeros",
                "content": "I search the room for treasure",
                "playerId": "player-123",
            }
        )

        # Verify agent was called with formatted prompt
        mock_agent.process_turn.assert_called_once()
        prompt = mock_agent.process_turn.call_args[0][0]
        assert "Valeros" in prompt
        assert "search the room" in prompt

        # Verify response was posted to Foundry
        mock_bridge.send_command.assert_called_with(
            "createChat",
            {"content": "The GM responds to your action.", "speaker": "GM Agent"},
        )

    def test_npc_turn_full_chain_gm_control(self, mock_agent, mock_bridge):
        """Test full chain: combatTurn event → agent decides → createChat."""
        # Mock ACA as inactive (GM Agent has full control)
        mock_bridge.send_command.side_effect = [
            {"active": False},  # getACAState
            {"success": True},  # createChat
        ]

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Simulate combatTurn event for NPC
        controller._handle_combat_turn(
            {
                "round": 2,
                "turn": 3,
                "combatant": {
                    "name": "Goblin Warrior",
                    "isNPC": True,
                    "actorId": "actor-456",
                },
            }
        )

        # Verify agent was called to decide NPC action
        mock_agent.process_turn.assert_called_once()
        prompt = mock_agent.process_turn.call_args[0][0]
        assert "Goblin Warrior" in prompt
        assert "turn" in prompt.lower()

    def test_npc_turn_full_chain_aca_control(self, mock_agent, mock_bridge):
        """Test full chain: combatTurn with ACA → narration only."""
        # Mock ACA as controlling this NPC
        mock_bridge.send_command.side_effect = [
            {  # getACAState
                "active": True,
                "designations": {"actor-456": "ai"},
            },
            {  # getACATurnState
                "actorName": "Goblin Warrior",
                "isProcessing": True,
                "actionsRemaining": 2,
                "actionHistory": ["Strike (hit)"],
            },
            {"success": True},  # createChat
        ]

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Simulate combatTurn event for NPC
        controller._handle_combat_turn(
            {
                "round": 2,
                "turn": 3,
                "combatant": {
                    "name": "Goblin Warrior",
                    "isNPC": True,
                    "actorId": "actor-456",
                },
            }
        )

        # Verify agent was called for narration (not tactical decision)
        mock_agent.process_turn.assert_called_once()
        prompt = mock_agent.process_turn.call_args[0][0]
        assert "AI Combat Assistant" in prompt
        assert "narrative" in prompt.lower()


class TestEventHandlerRegistration:
    """Tests verifying event handlers are registered correctly on the bridge."""

    def test_handlers_registered_on_start(self, mock_agent):
        """Test that event handlers are registered when controller starts."""
        # Use a fully mocked bridge to track on_event calls
        fully_mocked_bridge = MagicMock()
        fully_mocked_bridge.is_connected.return_value = True

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=fully_mocked_bridge,
        )

        # Before start, no handlers registered
        assert fully_mocked_bridge.on_event.call_count == 0

        controller.start()

        # After start, handlers should be registered
        assert fully_mocked_bridge.on_event.call_count == 2

        # Verify correct events
        event_names = [call[0][0] for call in fully_mocked_bridge.on_event.call_args_list]
        assert "playerChat" in event_names
        assert "combatTurn" in event_names

    def test_handlers_reregistered_on_bridge_change(self, mock_agent, mock_bridge):
        """Test that handlers are re-registered when bridge changes."""
        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
        )
        controller.start()

        # Create new bridge (simulating reconnect)
        new_bridge = MagicMock()
        new_bridge.is_connected.return_value = True
        new_bridge.send_command.return_value = {"success": True}

        controller.set_bridge(new_bridge)

        # Handlers should be registered on new bridge
        assert new_bridge.on_event.call_count == 2

    def test_event_triggers_handler(self, mock_agent, mock_socketio):
        """Test that bridge events actually trigger the handler functions."""
        # Use real bridge to test actual event dispatch
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"success": True})

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Simulate event from Foundry (as bridge would receive it)
        bridge.handle_event(
            {
                "eventType": "playerChat",
                "payload": {
                    "actorName": "Test Player",
                    "content": "Test action",
                },
            }
        )

        # Verify the handler was triggered
        mock_agent.process_turn.assert_called_once()


class TestMultiCampaignManagement:
    """Tests for managing multiple campaign game loops."""

    def test_multiple_controllers_independent(self, mock_agent, mock_bridge):
        """Test that multiple controllers for different campaigns work independently."""
        controller1 = GameLoopController(
            campaign_id="campaign-1",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller2 = GameLoopController(
            campaign_id="campaign-2",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )

        controller1.start()
        # controller2 not started

        assert controller1.enabled
        assert not controller2.enabled

        controller1.stop()
        controller2.start()

        assert not controller1.enabled
        assert controller2.enabled


class TestBridgeEventFlow:
    """Tests verifying the FoundryBridge event dispatch mechanism."""

    def test_bridge_dispatches_to_registered_handlers(self, mock_socketio):
        """Test that bridge correctly dispatches events to handlers."""
        bridge = FoundryBridge(mock_socketio)

        # Register a handler
        handler = MagicMock()
        bridge.on_event("testEvent", handler)

        # Simulate event from Foundry
        bridge.handle_event(
            {
                "eventType": "testEvent",
                "payload": {"key": "value"},
            }
        )

        # Handler should be called with payload
        handler.assert_called_once_with({"key": "value"})

    def test_bridge_multiple_handlers_same_event(self, mock_socketio):
        """Test that multiple handlers for same event all get called."""
        bridge = FoundryBridge(mock_socketio)

        handler1 = MagicMock()
        handler2 = MagicMock()
        bridge.on_event("testEvent", handler1)
        bridge.on_event("testEvent", handler2)

        bridge.handle_event(
            {
                "eventType": "testEvent",
                "payload": {"data": 123},
            }
        )

        handler1.assert_called_once_with({"data": 123})
        handler2.assert_called_once_with({"data": 123})

    def test_bridge_ignores_unregistered_events(self, mock_socketio):
        """Test that unregistered events don't cause errors."""
        bridge = FoundryBridge(mock_socketio)

        # No handler registered
        # Should not raise
        bridge.handle_event(
            {
                "eventType": "unknownEvent",
                "payload": {},
            }
        )


class TestReconnectionStateRestoration:
    """Tests for state restoration after reconnection."""

    def test_controller_survives_bridge_replacement(self, mock_agent, mock_bridge):
        """Test that controller continues working after bridge replacement."""
        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Process an event
        controller._handle_player_chat(
            {
                "actorName": "Player1",
                "content": "Action 1",
            }
        )
        assert mock_agent.process_turn.call_count == 1

        # Simulate disconnect/reconnect
        controller.stop()
        new_bridge = MagicMock()
        new_bridge.is_connected.return_value = True
        new_bridge.send_command.return_value = {"success": True}

        controller.set_bridge(new_bridge)
        controller.start()

        # Process another event - should use new bridge
        controller._handle_player_chat(
            {
                "actorName": "Player2",
                "content": "Action 2",
            }
        )

        assert mock_agent.process_turn.call_count == 2
        # Response should go to new bridge
        new_bridge.send_command.assert_called()

    def test_enabled_state_preserved_through_bridge_change(self, mock_agent, mock_bridge):
        """Test that enabled state is correctly handled through bridge changes."""
        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
        )
        controller.start()
        assert controller.enabled

        # Change bridge while enabled
        new_bridge = MagicMock()
        new_bridge.is_connected.return_value = True
        controller.set_bridge(new_bridge)

        # Should still be enabled
        assert controller.enabled
        # Handlers should be re-registered
        assert new_bridge.on_event.call_count == 2


class TestErrorIsolation:
    """Tests verifying errors in one component don't crash others."""

    def test_agent_error_doesnt_crash_controller(self, mock_agent, mock_bridge):
        """Test that agent errors are caught and don't crash the controller."""
        mock_agent.process_turn.side_effect = RuntimeError("LLM exploded")

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Should not raise
        controller._handle_player_chat(
            {
                "actorName": "Player",
                "content": "Test",
            }
        )

        # Controller should still be enabled
        assert controller.enabled

        # Error message should be posted
        mock_bridge.send_command.assert_called()
        content = mock_bridge.send_command.call_args[0][1]["content"]
        assert "error" in content.lower()

    def test_bridge_error_doesnt_crash_controller(self, mock_agent, mock_bridge):
        """Test that bridge errors are caught and don't crash the controller."""
        mock_bridge.send_command.side_effect = ConnectionError("Disconnected")

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        # Should not raise
        controller._handle_player_chat(
            {
                "actorName": "Player",
                "content": "Test",
            }
        )

        # Controller should still be enabled
        assert controller.enabled

    def test_aca_check_error_defaults_to_gm_control(self, mock_agent, mock_bridge):
        """Test that ACA check errors default to GM Agent control."""
        # ACA check fails
        mock_bridge.send_command.side_effect = [
            TimeoutError("ACA timeout"),  # getACAState
            {"success": True},  # createChat (for response)
        ]

        controller = GameLoopController(
            campaign_id="test-campaign",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
        )
        controller.start()

        controller._handle_combat_turn(
            {
                "combatant": {
                    "name": "Goblin",
                    "isNPC": True,
                    "actorId": "actor-123",
                },
            }
        )

        # Agent should be called (GM Agent took control due to ACA error)
        mock_agent.process_turn.assert_called_once()
