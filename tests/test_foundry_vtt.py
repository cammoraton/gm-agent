"""Tests for Foundry VTT MCP server and bridge."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from gm_agent.mcp.foundry_vtt import FoundryBridge, FoundryVTTServer

# =============================================================================
# FoundryBridge Tests
# =============================================================================


class TestFoundryBridge:
    """Tests for the WebSocket bridge to Foundry VTT."""

    def test_init(self):
        """Test bridge initialization."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)

        assert bridge._socketio == mock_socketio
        assert not bridge._connected
        assert bridge._sid is None
        assert bridge._pending_requests == {}

    def test_set_connected(self):
        """Test marking bridge as connected."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)

        bridge.set_connected("test-sid-123")

        assert bridge.is_connected()
        assert bridge._sid == "test-sid-123"

    def test_set_disconnected(self):
        """Test marking bridge as disconnected."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        bridge.set_disconnected()

        assert not bridge.is_connected()
        assert bridge._sid is None

    def test_send_command_not_connected(self):
        """Test sending command when not connected raises error."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)

        with pytest.raises(ConnectionError, match="not connected"):
            bridge.send_command("getScene")

    def test_send_command_timeout(self):
        """Test command timeout."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        with pytest.raises(TimeoutError):
            bridge.send_command("getScene", timeout=0.1)

    def test_send_command_success(self):
        """Test successful command/response flow."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        # Simulate async response in separate thread
        def send_response():
            time.sleep(0.1)
            # Find the request ID from the emit call
            calls = mock_socketio.emit.call_args_list
            if calls:
                request_data = calls[0][0][1]
                request_id = request_data["requestId"]
                bridge.handle_response(
                    {
                        "requestId": request_id,
                        "success": True,
                        "data": {"name": "Test Scene", "tokens": []},
                    }
                )

        thread = threading.Thread(target=send_response)
        thread.start()

        result = bridge.send_command("getScene", timeout=1.0)
        thread.join()

        assert result == {"name": "Test Scene", "tokens": []}
        mock_socketio.emit.assert_called_once()

    def test_handle_response_error(self):
        """Test handling error response."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        # Set up a pending request manually
        event = threading.Event()
        bridge._pending_requests["test-request"] = {
            "event": event,
            "response": None,
            "error": None,
        }

        bridge.handle_response(
            {
                "requestId": "test-request",
                "success": False,
                "error": "Token not found",
            }
        )

        assert event.is_set()
        assert bridge._pending_requests["test-request"]["error"] == "Token not found"

    def test_handle_event(self):
        """Test handling events from Foundry."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)

        # Register event handler
        handler = MagicMock()
        bridge.on_event("combatStart", handler)

        bridge.handle_event(
            {
                "eventType": "combatStart",
                "payload": {"round": 1, "combatantCount": 4},
            }
        )

        handler.assert_called_once_with({"round": 1, "combatantCount": 4})

    def test_cache_operations(self):
        """Test cache get/set operations."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)

        # Set cached value
        bridge.set_cached("scene", {"name": "Test Scene"})

        # Get cached value
        result = bridge.get_cached("scene")
        assert result == {"name": "Test Scene"}

        # Non-existent key
        assert bridge.get_cached("nonexistent") is None

    def test_cache_expiry(self):
        """Test cache expiry."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge._cache_ttl = 0.1  # Very short TTL for testing

        bridge.set_cached("scene", {"name": "Test"})
        assert bridge.get_cached("scene") is not None

        time.sleep(0.2)
        assert bridge.get_cached("scene") is None


# =============================================================================
# FoundryVTTServer Tests
# =============================================================================


class TestFoundryVTTServer:
    """Tests for the Foundry VTT MCP server."""

    def test_init_no_bridge(self):
        """Test server initialization without bridge."""
        server = FoundryVTTServer()
        assert not server.is_connected()

    def test_init_with_bridge(self):
        """Test server initialization with bridge."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        assert server.is_connected()

    def test_list_tools(self):
        """Test listing available tools."""
        server = FoundryVTTServer()
        tools = server.list_tools()

        # Should have 33 tools (12 base + 6 ACA + 8 combat/token/chat + 3 exploration + 2 rest + 2 time)
        assert len(tools) == 33

        tool_names = {t.name for t in tools}
        expected_tools = {
            "foundry_get_scene",
            "foundry_get_actors",
            "foundry_get_combat_state",
            "foundry_is_combat_active",
            "foundry_get_combat_summary",
            "foundry_update_token",
            "foundry_create_chat",
            "foundry_show_journal",
            "foundry_roll_check",
            "foundry_start_combat",
            "foundry_end_combat",
            "foundry_narrate",
            "foundry_get_aca_state",
            "foundry_get_aca_turn_state",
            "foundry_set_aca_notes",
            "foundry_trigger_aca_suggestion",
            "foundry_get_aca_game_state",
            "foundry_set_aca_designation",
            "foundry_apply_damage",
            "foundry_heal",
            "foundry_apply_condition",
            "foundry_remove_condition",
            "foundry_advance_turn",
            "foundry_spawn_token",
            "foundry_remove_token",
            "foundry_whisper",
            "foundry_set_exploration_activity",
            "foundry_get_exploration_state",
            "foundry_roll_secret_check",
            "foundry_take_rest",
            "foundry_refocus",
            "foundry_advance_time",
            "foundry_get_time",
        }
        assert tool_names == expected_tools

        # Verify all tools have category set
        for tool in tools:
            assert tool.category == "foundry", f"Tool {tool.name} missing category"

    def test_call_unknown_tool(self):
        """Test calling unknown tool returns error."""
        server = FoundryVTTServer()
        result = server.call_tool("unknown_tool", {})

        assert not result.success
        assert "Unknown tool" in result.error

    def test_get_scene_not_connected(self):
        """Test get_scene when not connected."""
        server = FoundryVTTServer()
        result = server.call_tool("foundry_get_scene", {})

        assert not result.success
        assert "not connected" in result.error

    def test_get_scene_success(self):
        """Test successful get_scene."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        # Mock the send_command method
        bridge.send_command = MagicMock(
            return_value={
                "name": "Dark Cave",
                "darkness": 0.8,
                "tokens": [
                    {"name": "Goblin", "x": 5, "y": 3, "hidden": False},
                    {"name": "Fighter", "x": 7, "y": 3, "hidden": False},
                ],
                "wallCount": 10,
                "lightCount": 2,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_scene", {})

        assert result.success
        assert "Dark Cave" in result.data
        assert "Goblin" in result.data
        bridge.send_command.assert_called_once_with("getScene")

    def test_get_actors_with_filter(self):
        """Test get_actors with name filter."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actors": [
                    {"name": "Goblin Boss", "level": 3, "hp": {"value": 20, "max": 45}},
                ],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_actors", {"names": ["Goblin Boss"]})

        assert result.success
        assert "Goblin Boss" in result.data
        bridge.send_command.assert_called_once_with("getActors", {"names": ["Goblin Boss"]})

    def test_get_combat_state(self):
        """Test get_combat_state."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "active": True,
                "round": 3,
                "turn": 1,
                "combatants": [
                    {
                        "name": "Fighter",
                        "initiative": 22,
                        "hp": {"value": 35, "max": 45},
                    },
                    {
                        "name": "Goblin",
                        "initiative": 15,
                        "hp": {"value": 10, "max": 20},
                    },
                ],
                "aiCombatAssistantActive": True,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_combat_state", {})

        assert result.success
        assert "Round 3" in result.data
        assert "Fighter" in result.data
        assert "AI Combat Assistant" in result.data

    def test_is_combat_active(self):
        """Test is_combat_active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"active": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_is_combat_active", {})

        assert result.success
        assert "active" in result.data.lower()

    def test_update_token_not_connected(self):
        """Test update_token when not connected."""
        server = FoundryVTTServer()
        result = server.call_tool("foundry_update_token", {"name": "Goblin", "x": 5})

        assert not result.success
        assert "not connected" in result.error

    def test_update_token_missing_name(self):
        """Test update_token without name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_update_token", {"x": 5})

        assert not result.success
        assert "name is required" in result.error.lower()

    def test_update_token_success(self):
        """Test successful token update."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"updated": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_update_token",
            {
                "name": "Goblin",
                "x": 5,
                "y": 3,
                "hidden": True,
            },
        )

        assert result.success
        assert "updated" in result.data.lower()

    def test_create_chat(self):
        """Test creating chat message."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"created": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_create_chat",
            {
                "content": "The goblin snarls!",
                "speaker": "Goblin",
            },
        )

        assert result.success
        assert "created" in result.data.lower()

    def test_create_chat_missing_content(self):
        """Test create_chat without content."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_create_chat", {})

        assert not result.success
        assert "content is required" in result.error.lower()

    def test_show_journal(self):
        """Test showing journal entry."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"found": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_show_journal", {"name": "Ancient Scroll"})

        assert result.success
        assert "Ancient Scroll" in result.data

    def test_show_journal_not_found(self):
        """Test showing non-existent journal entry."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"found": False})

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_show_journal", {"name": "Missing Entry"})

        assert not result.success
        assert "not found" in result.error.lower()

    def test_roll_check(self):
        """Test rolling a check."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "rolled": True,
                "total": 18,
                "degree": "success",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_roll_check",
            {
                "actor_name": "Fighter",
                "check_type": "perception",
                "dc": 15,
            },
        )

        assert result.success
        assert "Fighter" in result.data
        assert "18" in result.data
        assert "success" in result.data.lower()

    def test_roll_check_missing_actor(self):
        """Test roll_check without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_roll_check", {"check_type": "perception"})

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_start_combat(self):
        """Test starting combat."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "started": True,
                "combatantCount": 5,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_start_combat",
            {
                "combatants": ["Fighter", "Goblin 1", "Goblin 2"],
            },
        )

        assert result.success
        assert "5 combatants" in result.data

    def test_end_combat(self):
        """Test ending combat."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"ended": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_end_combat", {})

        assert result.success
        assert "ended" in result.data.lower()

    def test_end_combat_no_active(self):
        """Test ending combat when none active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "ended": False,
                "error": "No active combat",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_end_combat", {})

        assert not result.success
        assert "no active combat" in result.error.lower()

    def test_graceful_fallback_scene(self):
        """Test graceful fallback to cache for scene data."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)

        # Set up cached data before disconnecting
        bridge.set_connected("test-sid")
        bridge.set_cached(
            "scene",
            {
                "name": "Cached Scene",
                "tokens": [],
                "wallCount": 0,
                "lightCount": 0,
            },
        )

        server = FoundryVTTServer(bridge)

        # Simulate disconnect
        bridge.set_disconnected()

        # Should return cached data
        result = server.call_tool("foundry_get_scene", {})

        assert result.success
        assert "Cached Scene" in result.data
        assert "disconnected" in result.data.lower()

    def test_graceful_fallback_write_fails(self):
        """Test that write operations fail gracefully when disconnected."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        # Not connected

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_update_token", {"name": "Goblin", "x": 5})

        assert not result.success
        assert "not connected" in result.error.lower()

    def test_narrate_success(self):
        """Test successful narration."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"created": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_narrate",
            {
                "content": "The shadows lengthen as dusk approaches...",
            },
        )

        assert result.success
        assert "GM Agent" in result.data

    def test_narrate_with_speaker(self):
        """Test narration with custom speaker."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"created": True})

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_narrate",
            {
                "content": "You dare challenge me?",
                "speaker": "Dragon",
            },
        )

        assert result.success
        assert "Dragon" in result.data

        # Verify the speaker was passed correctly
        call_args = bridge.send_command.call_args[0][1]
        assert call_args["speaker"] == "Dragon"

    def test_narrate_missing_content(self):
        """Test narration without content."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_narrate", {})

        assert not result.success
        assert "content is required" in result.error.lower()

    def test_narrate_not_connected(self):
        """Test narration when not connected."""
        server = FoundryVTTServer()
        result = server.call_tool(
            "foundry_narrate",
            {
                "content": "A distant thunder rumbles...",
            },
        )

        assert not result.success
        assert "not connected" in result.error.lower()

    # =========================================================================
    # AI Combat Assistant Integration Tests
    # =========================================================================

    def test_get_aca_state_active(self):
        """Test getting ACA state when active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "active": True,
                "installed": True,
                "combatActive": True,
                "designations": {"Goblin": "ai", "Orc": "ai"},
                "processingActors": [{"name": "Goblin", "id": "abc123"}],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_aca_state", {})

        assert result.success
        assert "Active" in result.data
        assert "Goblin" in result.data
        assert "processing" in result.data.lower()

    def test_get_aca_state_inactive(self):
        """Test getting ACA state when not active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "active": False,
                "installed": True,
                "designations": {},
                "processingActors": [],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_aca_state", {})

        assert result.success
        assert "inactive" in result.data.lower()

    def test_get_aca_turn_state(self):
        """Test getting ACA turn state for an actor."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Goblin Boss",
                "isProcessing": True,
                "hasTurnState": True,
                "actionsRemaining": 2,
                "currentMAP": -5,
                "actionHistory": ["Strike (hit)", "Move"],
                "permanentNotes": "Focus on casters",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_aca_turn_state", {"actor_name": "Goblin Boss"})

        assert result.success
        assert "Goblin Boss" in result.data
        assert "2" in result.data  # Actions remaining
        assert "-5" in result.data  # MAP
        assert "Focus on casters" in result.data

    def test_get_aca_turn_state_no_turn(self):
        """Test getting ACA turn state when not actor's turn."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Goblin",
                "isProcessing": False,
                "hasTurnState": False,
                "permanentNotes": "",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_aca_turn_state", {"actor_name": "Goblin"})

        assert result.success
        assert "No active turn state" in result.data

    def test_get_aca_turn_state_missing_actor(self):
        """Test getting ACA turn state without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_aca_turn_state", {})

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_set_aca_notes_success(self):
        """Test setting ACA notes."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "success": True,
                "actorName": "Goblin Boss",
                "notes": "Protect the shaman at all costs",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_aca_notes",
            {
                "actor_name": "Goblin Boss",
                "notes": "Protect the shaman at all costs",
            },
        )

        assert result.success
        assert "Goblin Boss" in result.data

    def test_set_aca_notes_actor_not_found(self):
        """Test setting ACA notes for non-existent actor."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "Actor 'Unknown' not found",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_aca_notes",
            {
                "actor_name": "Unknown",
                "notes": "Some notes",
            },
        )

        assert not result.success
        assert "not found" in result.error.lower()

    def test_trigger_aca_suggestion_success(self):
        """Test triggering ACA suggestion."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "success": True,
                "combatantName": "Goblin Boss",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_trigger_aca_suggestion",
            {
                "actor_name": "Goblin Boss",
            },
        )

        assert result.success
        assert "Goblin Boss" in result.data

    def test_trigger_aca_suggestion_with_notes(self):
        """Test triggering ACA suggestion with manual notes."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "success": True,
                "combatantName": "Goblin Boss",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_trigger_aca_suggestion",
            {
                "actor_name": "Goblin Boss",
                "manual_notes": "Focus on the cleric",
            },
        )

        assert result.success
        # Verify notes were passed
        call_args = bridge.send_command.call_args[0][1]
        assert call_args["manualNotes"] == "Focus on the cleric"

    def test_trigger_aca_suggestion_api_unavailable(self):
        """Test triggering ACA suggestion when API not available."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "AI Combat Assistant API not available",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_trigger_aca_suggestion",
            {
                "actor_name": "Goblin",
            },
        )

        assert not result.success
        assert "not available" in result.error.lower()

    def test_get_aca_game_state_success(self):
        """Test getting ACA game state."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "success": True,
                "combatantName": "Goblin Boss",
                "actionsRemaining": 2,
                "gameState": {
                    "self": {
                        "name": "Goblin Boss",
                        "hp": {"value": 45, "max": 60},
                        "ac": 18,
                    },
                    "enemies": [
                        {
                            "name": "Fighter",
                            "hp": {"value": 30, "max": 50},
                            "distance": 15,
                        }
                    ],
                    "allies": [],
                },
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_get_aca_game_state",
            {
                "actor_name": "Goblin Boss",
            },
        )

        assert result.success
        assert "Goblin Boss" in result.data
        assert "Fighter" in result.data

    def test_get_aca_game_state_no_combat(self):
        """Test getting ACA game state when no combat active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "No active combat",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_get_aca_game_state",
            {
                "actor_name": "Goblin",
            },
        )

        assert not result.success
        assert "no active combat" in result.error.lower()

    def test_set_aca_designation_ai(self):
        """Test setting ACA designation to AI control."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "success": True,
                "actorName": "Goblin Boss",
                "designation": "ai",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_aca_designation",
            {
                "actor_name": "Goblin Boss",
                "designation": "ai",
            },
        )

        assert result.success
        assert "Goblin Boss" in result.data
        assert "ai" in result.data.lower()

    def test_set_aca_designation_manual(self):
        """Test setting ACA designation to manual control."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "success": True,
                "actorName": "Goblin Boss",
                "designation": "manual",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_aca_designation",
            {
                "actor_name": "Goblin Boss",
                "designation": "manual",
            },
        )

        assert result.success
        assert "manual" in result.data.lower()

    def test_set_aca_designation_missing_actor(self):
        """Test setting ACA designation without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_aca_designation",
            {
                "designation": "ai",
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_set_aca_designation_missing_designation(self):
        """Test setting ACA designation without designation value."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_aca_designation",
            {
                "actor_name": "Goblin Boss",
            },
        )

        assert not result.success
        assert "designation is required" in result.error.lower()

    # =========================================================================
    # Combat Resolution Tool Tests
    # =========================================================================

    def test_apply_damage_success(self):
        """Test applying damage to an actor."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actualDamage": 15,
                "resistanceApplied": False,
                "weaknessApplied": False,
                "newHP": {"value": 35, "max": 50},
                "defeated": False,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_damage",
            {
                "actor_name": "Goblin",
                "amount": 15,
                "damage_type": "slashing",
            },
        )

        assert result.success
        assert "15" in result.data
        assert "slashing" in result.data
        assert "35/50" in result.data

    def test_apply_damage_with_resistance(self):
        """Test applying damage with resistance."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actualDamage": 10,
                "resistanceApplied": True,
                "resistanceAmount": 5,
                "newHP": {"value": 40, "max": 50},
                "defeated": False,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_damage",
            {
                "actor_name": "Golem",
                "amount": 15,
                "damage_type": "fire",
            },
        )

        assert result.success
        assert "resisted" in result.data.lower()
        assert "5" in result.data

    def test_apply_damage_with_weakness(self):
        """Test applying damage with weakness."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actualDamage": 20,
                "weaknessApplied": True,
                "weaknessAmount": 5,
                "newHP": {"value": 30, "max": 50},
                "defeated": False,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_damage",
            {
                "actor_name": "Troll",
                "amount": 15,
                "damage_type": "fire",
            },
        )

        assert result.success
        assert "weakness" in result.data.lower()

    def test_apply_damage_defeats_actor(self):
        """Test applying lethal damage."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actualDamage": 50,
                "newHP": {"value": 0, "max": 50},
                "defeated": True,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_damage",
            {
                "actor_name": "Goblin",
                "amount": 50,
            },
        )

        assert result.success
        assert "DEFEATED" in result.data

    def test_apply_damage_missing_actor(self):
        """Test applying damage without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_damage",
            {
                "amount": 10,
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_apply_damage_missing_amount(self):
        """Test applying damage without amount."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_damage",
            {
                "actor_name": "Goblin",
            },
        )

        assert not result.success
        assert "amount is required" in result.error.lower()

    def test_heal_success(self):
        """Test healing an actor."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actualHealing": 10,
                "newHP": {"value": 40, "max": 50},
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_heal",
            {
                "actor_name": "Fighter",
                "amount": 10,
            },
        )

        assert result.success
        assert "10" in result.data
        assert "40/50" in result.data

    def test_heal_capped_at_max(self):
        """Test healing is capped at max HP."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actualHealing": 5,  # Only healed 5 because capped
                "newHP": {"value": 50, "max": 50},
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_heal",
            {
                "actor_name": "Fighter",
                "amount": 20,  # Requested 20
            },
        )

        assert result.success
        assert "5" in result.data  # Only healed 5
        assert "50/50" in result.data

    def test_heal_missing_actor(self):
        """Test healing without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_heal",
            {
                "amount": 10,
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_heal_missing_amount(self):
        """Test healing without amount."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_heal",
            {
                "actor_name": "Fighter",
            },
        )

        assert not result.success
        assert "amount is required" in result.error.lower()

    def test_apply_condition_success(self):
        """Test applying a condition."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Goblin",
                "condition": "frightened",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_condition",
            {
                "actor_name": "Goblin",
                "condition": "frightened",
            },
        )

        assert result.success
        assert "frightened" in result.data.lower()
        assert "Goblin" in result.data

    def test_apply_condition_with_value(self):
        """Test applying a valued condition."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Fighter",
                "condition": "frightened",
                "value": 2,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_condition",
            {
                "actor_name": "Fighter",
                "condition": "frightened",
                "value": 2,
            },
        )

        assert result.success
        assert "frightened 2" in result.data.lower()

    def test_apply_condition_missing_actor(self):
        """Test applying condition without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_condition",
            {
                "condition": "poisoned",
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_apply_condition_missing_condition(self):
        """Test applying condition without condition name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_apply_condition",
            {
                "actor_name": "Fighter",
            },
        )

        assert not result.success
        assert "condition name is required" in result.error.lower()

    def test_remove_condition_success(self):
        """Test removing a condition."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Fighter",
                "condition": "frightened",
                "removed": True,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_remove_condition",
            {
                "actor_name": "Fighter",
                "condition": "frightened",
            },
        )

        assert result.success
        assert "Removed" in result.data
        assert "frightened" in result.data.lower()

    def test_remove_condition_missing_actor(self):
        """Test removing condition without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_remove_condition",
            {
                "condition": "poisoned",
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_remove_condition_missing_condition(self):
        """Test removing condition without condition name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_remove_condition",
            {
                "actor_name": "Fighter",
            },
        )

        assert not result.success
        assert "condition name is required" in result.error.lower()

    def test_advance_turn_success(self):
        """Test advancing combat turn."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "nextCombatant": "Goblin",
                "round": 2,
                "turn": 3,
                "newRound": False,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_turn", {})

        assert result.success
        assert "Goblin" in result.data
        bridge.send_command.assert_called_with("advanceTurn")

    def test_advance_turn_new_round(self):
        """Test advancing to a new round."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "nextCombatant": "Fighter",
                "round": 3,
                "turn": 0,
                "newRound": True,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_turn", {})

        assert result.success
        assert "Fighter" in result.data
        assert "Round 3" in result.data

    def test_advance_turn_no_combat(self):
        """Test advancing turn when no combat active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "No active combat",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_turn", {})

        assert not result.success
        assert "no active combat" in result.error.lower()

    def test_spawn_token_success(self):
        """Test spawning a token."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "tokenId": "token-123",
                "tokenName": "Goblin",
                "x": 5,
                "y": 10,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_spawn_token",
            {
                "actor_name": "Goblin",
                "x": 5,
                "y": 10,
            },
        )

        assert result.success
        assert "Goblin" in result.data
        assert "(5, 10)" in result.data

    def test_spawn_token_hidden(self):
        """Test spawning a hidden token."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "tokenId": "token-123",
                "tokenName": "Goblin",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_spawn_token",
            {
                "actor_name": "Goblin",
                "x": 5,
                "y": 10,
                "hidden": True,
            },
        )

        assert result.success
        assert "[hidden]" in result.data

    def test_spawn_token_missing_actor(self):
        """Test spawning token without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_spawn_token",
            {
                "x": 5,
                "y": 10,
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_spawn_token_missing_position(self):
        """Test spawning token without position."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_spawn_token",
            {
                "actor_name": "Goblin",
            },
        )

        assert not result.success
        assert "position" in result.error.lower()

    def test_remove_token_success(self):
        """Test removing a token."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "removed": True,
                "tokenName": "Goblin",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_remove_token", {"name": "Goblin"})

        assert result.success
        assert "Goblin" in result.data

    def test_remove_token_not_found(self):
        """Test removing non-existent token."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "Token 'Unknown' not found",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_remove_token", {"name": "Unknown"})

        assert not result.success
        assert "not found" in result.error.lower()

    def test_remove_token_missing_name(self):
        """Test removing token without name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_remove_token", {})

        assert not result.success
        assert "name is required" in result.error.lower()

    def test_whisper_success(self):
        """Test sending a whisper message."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "sent": True,
                "playerName": "John",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_whisper",
            {
                "player_name": "John",
                "content": "You notice a secret door",
            },
        )

        assert result.success
        assert "John" in result.data

    def test_whisper_player_not_found(self):
        """Test whisper to non-existent player."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "Player 'Unknown' not found",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_whisper",
            {
                "player_name": "Unknown",
                "content": "Hello",
            },
        )

        assert not result.success
        assert "not found" in result.error.lower()

    def test_whisper_missing_player(self):
        """Test whisper without player name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_whisper",
            {
                "content": "Hello",
            },
        )

        assert not result.success
        assert "player name is required" in result.error.lower()

    def test_whisper_missing_content(self):
        """Test whisper without content."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_whisper",
            {
                "player_name": "John",
            },
        )

        assert not result.success
        assert "content is required" in result.error.lower()

    # =========================================================================
    # Exploration Mode Tool Tests
    # =========================================================================

    def test_set_exploration_activity_success(self):
        """Test setting exploration activity."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Valeros",
                "activity": "Scout",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_exploration_activity",
            {
                "actor_name": "Valeros",
                "activity": "Scout",
            },
        )

        assert result.success
        assert "Valeros" in result.data
        assert "Scout" in result.data

    def test_set_exploration_activity_missing_actor(self):
        """Test setting activity without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_exploration_activity",
            {
                "activity": "Scout",
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_set_exploration_activity_missing_activity(self):
        """Test setting activity without activity."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_set_exploration_activity",
            {
                "actor_name": "Valeros",
            },
        )

        assert not result.success
        assert "activity is required" in result.error.lower()

    def test_get_exploration_state_success(self):
        """Test getting exploration state."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "activities": {
                    "Valeros": "Scout",
                    "Ezren": "Detect Magic",
                    "Merisiel": "Avoid Notice",
                },
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_exploration_state", {})

        assert result.success
        assert "Valeros" in result.data
        assert "Scout" in result.data
        assert "Ezren" in result.data
        assert "Detect Magic" in result.data

    def test_get_exploration_state_empty(self):
        """Test getting exploration state when no activities set."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"activities": {}})

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_exploration_state", {})

        assert result.success
        assert "no exploration activities" in result.data.lower()

    def test_roll_secret_check_success(self):
        """Test rolling a secret check."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "rolled": True,
                "total": 25,
                "degree": "Success",
                "secret": True,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_roll_secret_check",
            {
                "actor_name": "Valeros",
                "check_type": "perception",
                "dc": 20,
            },
        )

        assert result.success
        assert "25" in result.data
        assert "perception" in result.data.lower()
        assert "DC 20" in result.data

    def test_roll_secret_check_with_degree(self):
        """Test rolling a secret check with degree of success."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "rolled": True,
                "total": 30,
                "degree": "Critical Success",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_roll_secret_check",
            {
                "actor_name": "Ezren",
                "check_type": "arcana",
            },
        )

        assert result.success
        assert "Critical Success" in result.data

    def test_roll_secret_check_missing_actor(self):
        """Test secret check without actor name."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_roll_secret_check",
            {
                "check_type": "perception",
            },
        )

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_roll_secret_check_missing_type(self):
        """Test secret check without check type."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool(
            "foundry_roll_secret_check",
            {
                "actor_name": "Valeros",
            },
        )

        assert not result.success
        assert "check type is required" in result.error.lower()

    # =========================================================================
    # Rest and Recovery Tool Tests
    # =========================================================================

    def test_take_rest_long_success(self):
        """Test taking a long rest."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "results": [
                    {
                        "actorName": "Valeros",
                        "hpHealed": 30,
                        "conditionsRemoved": ["fatigued"],
                    },
                    {"actorName": "Ezren", "hpHealed": 15, "conditionsRemoved": []},
                ],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_take_rest", {"rest_type": "long"})

        assert result.success
        assert "Long Rest" in result.data
        assert "Valeros" in result.data
        assert "30" in result.data
        assert "fatigued" in result.data.lower()

    def test_take_rest_short_success(self):
        """Test taking a short rest."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "results": [
                    {"actorName": "Valeros", "hpHealed": 5, "conditionsRemoved": []},
                ],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_take_rest", {"rest_type": "short"})

        assert result.success
        assert "Short Rest" in result.data

    def test_take_rest_missing_type(self):
        """Test rest without type."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_take_rest", {})

        assert not result.success
        assert "rest type is required" in result.error.lower()

    def test_take_rest_invalid_type(self):
        """Test rest with invalid type."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_take_rest", {"rest_type": "medium"})

        assert not result.success
        assert "short" in result.error.lower() or "long" in result.error.lower()

    def test_refocus_success(self):
        """Test refocusing."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actorName": "Ezren",
                "focusPoints": {"value": 2, "max": 3},
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_refocus", {"actor_name": "Ezren"})

        assert result.success
        assert "Ezren" in result.data
        assert "2/3" in result.data

    def test_refocus_missing_actor(self):
        """Test refocus without actor."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_refocus", {})

        assert not result.success
        assert "actor name is required" in result.error.lower()

    def test_refocus_no_focus_pool(self):
        """Test refocus when actor has no focus pool."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "error": "Valeros has no focus pool",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_refocus", {"actor_name": "Valeros"})

        assert not result.success
        assert "no focus pool" in result.error.lower()

    # =========================================================================
    # Time Tracking Tool Tests
    # =========================================================================

    def test_advance_time_hours(self):
        """Test advancing time by hours."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "newTime": "14:30",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_time", {"amount": 2, "unit": "hours"})

        assert result.success
        assert "2 hours" in result.data
        assert "14:30" in result.data

    def test_advance_time_days(self):
        """Test advancing time by days."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "newTime": "08:00",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_time", {"amount": 1, "unit": "days"})

        assert result.success
        assert "1 days" in result.data

    def test_advance_time_missing_amount(self):
        """Test advancing time without amount."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_time", {"unit": "hours"})

        assert not result.success
        assert "amount is required" in result.error.lower()

    def test_advance_time_missing_unit(self):
        """Test advancing time without unit."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_time", {"amount": 2})

        assert not result.success
        assert "unit is required" in result.error.lower()

    def test_advance_time_invalid_unit(self):
        """Test advancing time with invalid unit."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_advance_time", {"amount": 2, "unit": "weeks"})

        assert not result.success
        assert "rounds" in result.error.lower() or "minutes" in result.error.lower()

    def test_get_time_success(self):
        """Test getting current time."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "time": "14:30",
                "date": "15/3/4721",
                "dayOfWeek": "Moonday",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_time", {})

        assert result.success
        assert "14:30" in result.data
        assert "15/3/4721" in result.data
        assert "Moonday" in result.data

    def test_get_time_minimal(self):
        """Test getting time when only time is available."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "time": "09:15",
                "date": "",
                "dayOfWeek": "",
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_time", {})

        assert result.success
        assert "09:15" in result.data


# =============================================================================
# Format Helper Tests
# =============================================================================


class TestFormatHelpers:
    """Tests for output formatting helpers."""

    def test_format_scene(self):
        """Test scene formatting."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "name": "Dark Dungeon",
                "darkness": 0.5,
                "tokens": [
                    {"name": "Hero", "x": 5, "y": 5, "hidden": False, "elevation": 0},
                    {
                        "name": "Lurker",
                        "x": 10,
                        "y": 3,
                        "hidden": True,
                        "elevation": 10,
                    },
                ],
                "wallCount": 20,
                "lightCount": 3,
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_scene", {})

        assert "Dark Dungeon" in result.data
        assert "Hero" in result.data
        assert "Lurker" in result.data
        assert "[HIDDEN]" in result.data
        assert "elevation: 10ft" in result.data

    def test_format_combat_state_no_combat(self):
        """Test formatting when no combat active."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(return_value={"active": False})

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_combat_state", {})

        assert result.success
        assert "no combat" in result.data.lower()

    def test_format_combat_state_with_defeated(self):
        """Test formatting combat with defeated combatants."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "active": True,
                "round": 5,
                "turn": 0,
                "combatants": [
                    {
                        "name": "Fighter",
                        "initiative": 22,
                        "defeated": False,
                        "hp": {"value": 10, "max": 45},
                    },
                    {
                        "name": "Goblin",
                        "initiative": 15,
                        "defeated": True,
                        "hp": {"value": 0, "max": 20},
                    },
                ],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_combat_state", {})

        assert "[DEFEATED]" in result.data
        assert "Round 5" in result.data

    def test_format_actors_with_conditions(self):
        """Test formatting actors with conditions."""
        mock_socketio = MagicMock()
        bridge = FoundryBridge(mock_socketio)
        bridge.set_connected("test-sid")
        bridge.send_command = MagicMock(
            return_value={
                "actors": [
                    {
                        "name": "Poisoned Fighter",
                        "level": 5,
                        "hp": {"value": 25, "max": 50},
                        "conditions": ["Poisoned", "Frightened 1"],
                    },
                ],
            }
        )

        server = FoundryVTTServer(bridge)
        result = server.call_tool("foundry_get_actors", {})

        assert "Poisoned Fighter" in result.data
        assert "Level 5" in result.data
        assert "Poisoned" in result.data
        assert "Frightened 1" in result.data
