"""Foundry VTT MCP server with bidirectional WebSocket bridge."""

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from .base import MCPServer, ToolDef, ToolParameter, ToolResult
from .foundry_bridge_base import FoundryBridgeBase

logger = logging.getLogger(__name__)


class FoundryBridge(FoundryBridgeBase):
    """WebSocket bridge to Foundry VTT.

    Handles bidirectional communication between Python and Foundry VTT
    via Socket.IO. Python acts as the server, Foundry as the client.

    Implements FoundryBridgeBase interface for compatibility with
    GameLoopController and other components.
    """

    def __init__(self, socketio: Any):
        """Initialize the bridge.

        Args:
            socketio: Flask-SocketIO instance
        """
        self._socketio = socketio
        self._connected = False
        self._sid: str | None = None
        self._pending_requests: dict[str, dict] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

        # Cache for graceful fallback
        self._cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = 60.0  # Cache TTL in seconds

    def set_connected(self, sid: str) -> None:
        """Mark Foundry as connected."""
        with self._lock:
            self._connected = True
            self._sid = sid
        logger.info(f"Foundry VTT connected (sid: {sid})")

    def set_disconnected(self) -> None:
        """Mark Foundry as disconnected."""
        with self._lock:
            self._connected = False
            self._sid = None
            # Fail any pending requests
            pending_count = len(self._pending_requests)
            for request_id, request in self._pending_requests.items():
                request["error"] = "Foundry disconnected"
                request["event"].set()
        if pending_count > 0:
            logger.warning(f"Foundry disconnected with {pending_count} pending requests")
        else:
            logger.info("Foundry VTT disconnected")

    def is_connected(self) -> bool:
        """Check if Foundry is connected."""
        with self._lock:
            return self._connected

    def send_command(self, command: str, data: dict | None = None, timeout: float = 5.0) -> dict:
        """Send a command to Foundry and wait for response.

        Args:
            command: The command name (e.g., "getScene", "updateToken")
            data: Optional data payload
            timeout: Timeout in seconds

        Returns:
            Response data from Foundry

        Raises:
            ConnectionError: If Foundry is not connected
            TimeoutError: If response not received in time
        """
        if not self.is_connected():
            raise ConnectionError("Foundry VTT is not connected")

        request_id = str(uuid.uuid4())
        event = threading.Event()

        with self._lock:
            self._pending_requests[request_id] = {
                "event": event,
                "response": None,
                "error": None,
            }

        # Send command via Socket.IO
        self._socketio.emit(
            "foundry:command",
            {"requestId": request_id, "command": command, "data": data or {}},
            room=self._sid,
        )

        # Wait for response
        if not event.wait(timeout):
            with self._lock:
                del self._pending_requests[request_id]
            raise TimeoutError(f"Timeout waiting for Foundry response to {command}")

        with self._lock:
            request = self._pending_requests.pop(request_id)

        if request["error"]:
            raise RuntimeError(request["error"])

        return request["response"]

    def handle_response(self, data: dict) -> None:
        """Handle a response from Foundry.

        Args:
            data: Response data with requestId, success, data/error
        """
        request_id = data.get("requestId")
        if not request_id:
            return

        with self._lock:
            if request_id not in self._pending_requests:
                return

            request = self._pending_requests[request_id]
            if data.get("success"):
                request["response"] = data.get("data", {})
            else:
                request["error"] = data.get("error", "Unknown error")
            request["event"].set()

    def handle_event(self, data: dict) -> None:
        """Handle an event pushed from Foundry.

        Args:
            data: Event data with eventType and payload
        """
        event_type = data.get("eventType")
        payload = data.get("payload", {})

        if not event_type:
            return

        # Update cache if relevant
        self._update_cache_from_event(event_type, payload)

        # Notify handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                pass  # Don't let handler errors break the event loop

    def on_event(self, event_type: str, handler: Callable[[dict], None]) -> None:
        """Register a handler for Foundry events.

        Args:
            event_type: Event type (e.g., "combatStart", "updateActor")
            handler: Callback function receiving event payload dict
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_cached(self, key: str) -> Any | None:
        """Get cached data if still valid."""
        with self._lock:
            if key not in self._cache:
                return None
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp > self._cache_ttl:
                return None
            return self._cache[key]

    def set_cached(self, key: str, value: Any) -> None:
        """Set cached data."""
        with self._lock:
            self._cache[key] = value
            self._cache_timestamps[key] = time.time()

    def _update_cache_from_event(self, event_type: str, payload: dict) -> None:
        """Update cache based on Foundry events."""
        if event_type == "combatUpdate":
            self.set_cached("combat_state", payload)
        elif event_type == "sceneChange":
            self.set_cached("scene", payload)
        elif event_type == "combatEnd":
            # Clear combat cache
            with self._lock:
                self._cache.pop("combat_state", None)


class FoundryVTTServer(MCPServer):
    """MCP server for Foundry VTT integration.

    Provides tools for reading scene/combat state and controlling
    Foundry VTT from the GM Agent.
    """

    def __init__(self, bridge: FoundryBridgeBase | None = None):
        """Initialize the server.

        Args:
            bridge: Optional FoundryBridgeBase instance (FoundryBridge or
                FoundryPollingClient). If None, tools will return errors
                indicating Foundry is not connected.
        """
        self._bridge = bridge

    def set_bridge(self, bridge: FoundryBridgeBase) -> None:
        """Set the Foundry bridge."""
        self._bridge = bridge

    def is_connected(self) -> bool:
        """Check if Foundry is connected."""
        return self._bridge is not None and self._bridge.is_connected()

    def list_tools(self) -> list[ToolDef]:
        """List all available Foundry VTT tools."""
        return [
            # Read tools
            ToolDef(
                name="foundry_get_scene",
                description="Get current Foundry scene data including tokens, lighting, and walls",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_actors",
                description="Get actor data (HP, conditions, level) for tokens in the scene",
                parameters=[
                    ToolParameter(
                        name="names",
                        type="array",
                        description="Optional list of actor names to filter",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_combat_state",
                description="Get current combat tracker state including round, turn order, and combatant HP",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_is_combat_active",
                description="Check if combat is currently active in Foundry",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_combat_summary",
                description="Get a narrative summary of the combat so far",
                parameters=[],
                category="foundry",
            ),
            # Write tools
            ToolDef(
                name="foundry_update_token",
                description="Move, hide, or update a token on the scene",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Name of the token to update",
                    ),
                    ToolParameter(
                        name="x",
                        type="number",
                        description="New X position (grid units)",
                        required=False,
                    ),
                    ToolParameter(
                        name="y",
                        type="number",
                        description="New Y position (grid units)",
                        required=False,
                    ),
                    ToolParameter(
                        name="hidden",
                        type="boolean",
                        description="Whether token should be hidden",
                        required=False,
                    ),
                    ToolParameter(
                        name="elevation",
                        type="number",
                        description="Token elevation in feet",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_create_chat",
                description="Post a message to Foundry chat",
                parameters=[
                    ToolParameter(
                        name="content",
                        type="string",
                        description="Message content (supports HTML)",
                    ),
                    ToolParameter(
                        name="speaker",
                        type="string",
                        description="Speaker alias",
                        required=False,
                    ),
                    ToolParameter(
                        name="whisper",
                        type="boolean",
                        description="Whisper to GM only",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_show_journal",
                description="Display a journal entry to players",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Name of the journal entry",
                    ),
                    ToolParameter(
                        name="show_to_players",
                        type="boolean",
                        description="Show to all players (default: true)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_roll_check",
                description="Request a skill check or saving throw from an actor",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to roll",
                    ),
                    ToolParameter(
                        name="check_type",
                        type="string",
                        description="Type of check (perception, stealth, fortitude, reflex, will, etc.)",
                    ),
                    ToolParameter(
                        name="dc",
                        type="number",
                        description="Difficulty Class for the check",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_start_combat",
                description="Start combat with specified tokens",
                parameters=[
                    ToolParameter(
                        name="combatants",
                        type="array",
                        description="List of token names to add to combat",
                        required=False,
                    ),
                    ToolParameter(
                        name="roll_initiative",
                        type="boolean",
                        description="Automatically roll initiative (default: true)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_end_combat",
                description="End the current combat encounter",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_narrate",
                description="Post narrative description to Foundry chat as GM. Use for dramatic scene descriptions, NPC dialogue, or atmospheric text.",
                parameters=[
                    ToolParameter(
                        name="content",
                        type="string",
                        description="Narrative text to display",
                    ),
                    ToolParameter(
                        name="speaker",
                        type="string",
                        description="Speaker alias for NPC dialogue (optional, defaults to 'GM Agent')",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            # AI Combat Assistant integration tools
            ToolDef(
                name="foundry_get_aca_state",
                description="Get AI Combat Assistant module state: whether it's active, which NPCs are designated for AI control, and which actors are currently being processed.",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_aca_turn_state",
                description="Get AI Combat Assistant turn state for a specific NPC: actions remaining, current MAP penalty, action history, and tactical notes.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to get turn state for",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_set_aca_notes",
                description="Set permanent tactical notes for an NPC that AI Combat Assistant will use when deciding actions. Use this to give ACA hints about tactics.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to set notes for",
                    ),
                    ToolParameter(
                        name="notes",
                        type="string",
                        description="Tactical notes (e.g., 'Protect the boss', 'Focus fire on casters', 'Flee when below 25% HP')",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_trigger_aca_suggestion",
                description="Trigger AI Combat Assistant to generate an action suggestion for a combatant. ACA will analyze the combat and suggest the next action.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the combatant (optional, defaults to current combatant)",
                        required=False,
                    ),
                    ToolParameter(
                        name="manual_notes",
                        type="string",
                        description="One-time tactical notes for this suggestion only",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_aca_game_state",
                description="Get AI Combat Assistant's view of the current combat state, including detailed info about all combatants, their abilities, and tactical situation.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the combatant to get state for (optional, defaults to current)",
                        required=False,
                    ),
                    ToolParameter(
                        name="actions_remaining",
                        type="number",
                        description="Actions remaining this turn (default: 3)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_set_aca_designation",
                description="Set AI Combat Assistant designation for an actor: 'ai' (ACA controls), 'friendly', 'enemy', or 'none' (remove designation).",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to designate",
                    ),
                    ToolParameter(
                        name="designation",
                        type="string",
                        description="Designation: 'ai', 'friendly', 'enemy', or 'none'",
                    ),
                ],
                category="foundry",
            ),
            # Combat resolution tools
            ToolDef(
                name="foundry_apply_damage",
                description="Apply damage to an actor. Supports damage types for resistance/weakness calculation.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to damage",
                    ),
                    ToolParameter(
                        name="amount",
                        type="number",
                        description="Amount of damage to apply",
                    ),
                    ToolParameter(
                        name="damage_type",
                        type="string",
                        description="Damage type (slashing, piercing, bludgeoning, fire, cold, electricity, acid, poison, mental, etc.)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_heal",
                description="Heal an actor by restoring hit points.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to heal",
                    ),
                    ToolParameter(
                        name="amount",
                        type="number",
                        description="Amount of HP to restore",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_apply_condition",
                description="Apply a condition to an actor (frightened, poisoned, stunned, etc.).",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor",
                    ),
                    ToolParameter(
                        name="condition",
                        type="string",
                        description="Condition name (frightened, poisoned, stunned, slowed, etc.)",
                    ),
                    ToolParameter(
                        name="value",
                        type="number",
                        description="Value for valued conditions like Frightened 2 (optional)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_remove_condition",
                description="Remove a condition from an actor.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor",
                    ),
                    ToolParameter(
                        name="condition",
                        type="string",
                        description="Condition name to remove",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_advance_turn",
                description="Advance combat to the next combatant's turn.",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_spawn_token",
                description="Spawn a token on the current scene from an actor.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor to spawn a token for",
                    ),
                    ToolParameter(
                        name="x",
                        type="number",
                        description="X position in grid units",
                    ),
                    ToolParameter(
                        name="y",
                        type="number",
                        description="Y position in grid units",
                    ),
                    ToolParameter(
                        name="hidden",
                        type="boolean",
                        description="Whether to spawn hidden (default: false)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_remove_token",
                description="Remove a token from the current scene.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Name of the token to remove",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_whisper",
                description="Send a private whisper message to a specific player.",
                parameters=[
                    ToolParameter(
                        name="player_name",
                        type="string",
                        description="Name of the player to whisper to",
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="The message content",
                    ),
                ],
                category="foundry",
            ),
            # Exploration mode tools
            ToolDef(
                name="foundry_set_exploration_activity",
                description="Set exploration activity for an actor (Scout, Search, Avoid Notice, Defend, Detect Magic, etc.).",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor",
                    ),
                    ToolParameter(
                        name="activity",
                        type="string",
                        description="Exploration activity (Scout, Search, Avoid Notice, Defend, Detect Magic, Investigate, etc.)",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_exploration_state",
                description="Get current exploration activities for all party members.",
                parameters=[],
                category="foundry",
            ),
            ToolDef(
                name="foundry_roll_secret_check",
                description="Roll a secret check (GM rolls, player doesn't see result). Used for Recall Knowledge, Perception, etc.",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor",
                    ),
                    ToolParameter(
                        name="check_type",
                        type="string",
                        description="Type of check (perception, stealth, nature, etc.)",
                    ),
                    ToolParameter(
                        name="dc",
                        type="number",
                        description="DC for the check",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            # Rest and recovery tools
            ToolDef(
                name="foundry_take_rest",
                description="Have party take a rest. Short rest recovers some HP, long rest (8 hours) fully heals and removes many conditions.",
                parameters=[
                    ToolParameter(
                        name="rest_type",
                        type="string",
                        description="Type of rest: 'short' (10 min) or 'long' (8 hours)",
                    ),
                    ToolParameter(
                        name="actor_names",
                        type="array",
                        description="List of actor names to rest (optional, defaults to all party members)",
                        required=False,
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_refocus",
                description="Have an actor Refocus to recover 1 focus point (requires 10 minutes and appropriate activity).",
                parameters=[
                    ToolParameter(
                        name="actor_name",
                        type="string",
                        description="Name of the actor",
                    ),
                ],
                category="foundry",
            ),
            # Time tracking tools
            ToolDef(
                name="foundry_advance_time",
                description="Advance in-game time by a specified amount.",
                parameters=[
                    ToolParameter(
                        name="amount",
                        type="number",
                        description="Amount of time to advance",
                    ),
                    ToolParameter(
                        name="unit",
                        type="string",
                        description="Time unit: 'rounds', 'minutes', 'hours', 'days'",
                    ),
                ],
                category="foundry",
            ),
            ToolDef(
                name="foundry_get_time",
                description="Get current in-game time and date.",
                parameters=[],
                category="foundry",
            ),
        ]

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a Foundry VTT tool."""
        # Route to appropriate handler
        handlers = {
            "foundry_get_scene": self._get_scene,
            "foundry_get_actors": self._get_actors,
            "foundry_get_combat_state": self._get_combat_state,
            "foundry_is_combat_active": self._is_combat_active,
            "foundry_get_combat_summary": self._get_combat_summary,
            "foundry_update_token": self._update_token,
            "foundry_create_chat": self._create_chat,
            "foundry_show_journal": self._show_journal,
            "foundry_roll_check": self._roll_check,
            "foundry_start_combat": self._start_combat,
            "foundry_end_combat": self._end_combat,
            "foundry_narrate": self._narrate,
            "foundry_get_aca_state": self._get_aca_state,
            "foundry_get_aca_turn_state": self._get_aca_turn_state,
            "foundry_set_aca_notes": self._set_aca_notes,
            "foundry_trigger_aca_suggestion": self._trigger_aca_suggestion,
            "foundry_get_aca_game_state": self._get_aca_game_state,
            "foundry_set_aca_designation": self._set_aca_designation,
            "foundry_apply_damage": self._apply_damage,
            "foundry_heal": self._heal,
            "foundry_apply_condition": self._apply_condition,
            "foundry_remove_condition": self._remove_condition,
            "foundry_advance_turn": self._advance_turn,
            "foundry_spawn_token": self._spawn_token,
            "foundry_remove_token": self._remove_token,
            "foundry_whisper": self._whisper,
            "foundry_set_exploration_activity": self._set_exploration_activity,
            "foundry_get_exploration_state": self._get_exploration_state,
            "foundry_roll_secret_check": self._roll_secret_check,
            "foundry_take_rest": self._take_rest,
            "foundry_refocus": self._refocus,
            "foundry_advance_time": self._advance_time,
            "foundry_get_time": self._get_time,
        }

        handler = handlers.get(name)
        if not handler:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

        try:
            return handler(args)
        except ConnectionError as e:
            return self._handle_disconnected(name, args, str(e))
        except TimeoutError as e:
            return ToolResult(success=False, error=f"Timeout: {e}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def close(self) -> None:
        """Clean up resources."""
        self._bridge = None

    # =========================================================================
    # Read Tool Handlers
    # =========================================================================

    def _get_scene(self, args: dict) -> ToolResult:
        """Get current scene data from Foundry."""
        if not self._bridge:
            return ToolResult(success=False, error="Foundry VTT is not connected")

        data = self._bridge.send_command("getScene")
        self._bridge.set_cached("scene", data)

        return ToolResult(success=True, data=self._format_scene(data))

    def _get_actors(self, args: dict) -> ToolResult:
        """Get actor data from Foundry."""
        if not self._bridge:
            return ToolResult(success=False, error="Foundry VTT is not connected")

        names = args.get("names")
        data = self._bridge.send_command("getActors", {"names": names})

        return ToolResult(success=True, data=self._format_actors(data))

    def _get_combat_state(self, args: dict) -> ToolResult:
        """Get current combat state from Foundry."""
        if not self._bridge:
            return ToolResult(success=False, error="Foundry VTT is not connected")

        data = self._bridge.send_command("getCombatState")
        self._bridge.set_cached("combat_state", data)

        return ToolResult(success=True, data=self._format_combat_state(data))

    def _is_combat_active(self, args: dict) -> ToolResult:
        """Check if combat is active."""
        if not self._bridge:
            # Try cache for read-only operation
            cached = None
            return ToolResult(success=True, data="Combat status unknown (Foundry not connected)")

        data = self._bridge.send_command("isCombatActive")
        active = data.get("active", False)

        return ToolResult(
            success=True,
            data=f"Combat is {'active' if active else 'not active'}",
        )

    def _get_combat_summary(self, args: dict) -> ToolResult:
        """Get a narrative combat summary."""
        if not self._bridge:
            return ToolResult(success=False, error="Foundry VTT is not connected")

        data = self._bridge.send_command("getCombatSummary")
        return ToolResult(success=True, data=data.get("summary", "No combat summary available"))

    # =========================================================================
    # Write Tool Handlers
    # =========================================================================

    def _update_token(self, args: dict) -> ToolResult:
        """Update a token in Foundry."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot update token: Foundry VTT is not connected",
            )

        name = args.get("name")
        if not name:
            return ToolResult(success=False, error="Token name is required")

        updates = {}
        if "x" in args:
            updates["x"] = args["x"]
        if "y" in args:
            updates["y"] = args["y"]
        if "hidden" in args:
            updates["hidden"] = args["hidden"]
        if "elevation" in args:
            updates["elevation"] = args["elevation"]

        if not updates:
            return ToolResult(success=False, error="No updates specified")

        data = self._bridge.send_command("updateToken", {"name": name, "updates": updates})
        return ToolResult(
            success=True,
            data=f"Token '{name}' updated: {json.dumps(updates)}",
        )

    def _create_chat(self, args: dict) -> ToolResult:
        """Create a chat message in Foundry."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot create chat: Foundry VTT is not connected",
            )

        content = args.get("content")
        if not content:
            return ToolResult(success=False, error="Message content is required")

        message_data = {
            "content": content,
            "speaker": args.get("speaker"),
            "whisper": args.get("whisper", False),
        }

        self._bridge.send_command("createChat", message_data)
        return ToolResult(success=True, data="Chat message created")

    def _show_journal(self, args: dict) -> ToolResult:
        """Show a journal entry in Foundry."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot show journal: Foundry VTT is not connected",
            )

        name = args.get("name")
        if not name:
            return ToolResult(success=False, error="Journal entry name is required")

        data = self._bridge.send_command(
            "showJournal",
            {"name": name, "showToPlayers": args.get("show_to_players", True)},
        )

        if data.get("found"):
            return ToolResult(success=True, data=f"Showing journal entry: {name}")
        return ToolResult(success=False, error=f"Journal entry '{name}' not found")

    def _roll_check(self, args: dict) -> ToolResult:
        """Request a check roll from an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot roll check: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        check_type = args.get("check_type")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if not check_type:
            return ToolResult(success=False, error="Check type is required")

        data = self._bridge.send_command(
            "rollCheck",
            {
                "actorName": actor_name,
                "checkType": check_type,
                "dc": args.get("dc"),
            },
        )

        if data.get("rolled"):
            result_str = f"{actor_name} rolled {check_type}: {data.get('total', '?')}"
            if data.get("degree"):
                result_str += f" ({data['degree']})"
            return ToolResult(success=True, data=result_str)
        return ToolResult(success=False, error=data.get("error", "Roll failed"))

    def _start_combat(self, args: dict) -> ToolResult:
        """Start combat in Foundry."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot start combat: Foundry VTT is not connected",
            )

        combatants = args.get("combatants")
        roll_initiative = args.get("roll_initiative", True)

        data = self._bridge.send_command(
            "startCombat",
            {"combatants": combatants, "rollInitiative": roll_initiative},
        )

        if data.get("started"):
            count = data.get("combatantCount", 0)
            return ToolResult(
                success=True,
                data=f"Combat started with {count} combatants",
            )
        return ToolResult(success=False, error=data.get("error", "Failed to start combat"))

    def _end_combat(self, args: dict) -> ToolResult:
        """End combat in Foundry."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot end combat: Foundry VTT is not connected",
            )

        data = self._bridge.send_command("endCombat")

        if data.get("ended"):
            return ToolResult(success=True, data="Combat ended")
        return ToolResult(
            success=False,
            error=data.get("error", "No active combat to end"),
        )

    def _narrate(self, args: dict) -> ToolResult:
        """Post narrative text to Foundry chat."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot narrate: Foundry VTT is not connected",
            )

        content = args.get("content")
        if not content:
            return ToolResult(success=False, error="Narrative content is required")

        speaker = args.get("speaker", "GM Agent")

        message_data = {
            "content": content,
            "speaker": speaker,
            "whisper": False,  # Narration is always public
        }

        self._bridge.send_command("createChat", message_data)
        return ToolResult(success=True, data=f"Narrated as {speaker}")

    # =========================================================================
    # AI Combat Assistant Integration
    # =========================================================================

    def _get_aca_state(self, args: dict) -> ToolResult:
        """Get AI Combat Assistant state."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot get ACA state: Foundry VTT is not connected",
            )

        data = self._bridge.send_command("getACAState")
        return ToolResult(success=True, data=self._format_aca_state(data))

    def _get_aca_turn_state(self, args: dict) -> ToolResult:
        """Get AI Combat Assistant turn state for an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot get ACA turn state: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")

        data = self._bridge.send_command("getACATurnState", {"actorName": actor_name})

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(success=True, data=self._format_aca_turn_state(data))

    def _set_aca_notes(self, args: dict) -> ToolResult:
        """Set permanent notes for AI Combat Assistant."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot set ACA notes: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        notes = args.get("notes", "")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")

        data = self._bridge.send_command("setACANotes", {"actorName": actor_name, "notes": notes})

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Set ACA tactical notes for {data.get('actorName', actor_name)}",
        )

    def _trigger_aca_suggestion(self, args: dict) -> ToolResult:
        """Trigger ACA to generate an action suggestion."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot trigger ACA: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        manual_notes = args.get("manual_notes")

        data = self._bridge.send_command(
            "triggerACASuggestion",
            {"actorName": actor_name, "manualNotes": manual_notes},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Triggered ACA suggestion for {data.get('combatantName', 'current combatant')}. ACA is now generating tactical advice.",
        )

    def _get_aca_game_state(self, args: dict) -> ToolResult:
        """Get ACA's gathered game state."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot get ACA game state: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        actions_remaining = args.get("actions_remaining", 3)

        data = self._bridge.send_command(
            "getACAGameState",
            {"actorName": actor_name, "actionsRemaining": actions_remaining},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=self._format_aca_game_state(data),
        )

    def _set_aca_designation(self, args: dict) -> ToolResult:
        """Set ACA designation for an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot set ACA designation: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        designation = args.get("designation")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if not designation:
            return ToolResult(success=False, error="Designation is required")

        data = self._bridge.send_command(
            "setACADesignation",
            {"actorName": actor_name, "designation": designation},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Set ACA designation for {data.get('actorName', actor_name)}: {data.get('designation')}",
        )

    # =========================================================================
    # Combat Resolution Tools
    # =========================================================================

    def _apply_damage(self, args: dict) -> ToolResult:
        """Apply damage to an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot apply damage: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        amount = args.get("amount")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if amount is None:
            return ToolResult(success=False, error="Damage amount is required")

        damage_type = args.get("damage_type", "untyped")

        data = self._bridge.send_command(
            "applyDamage",
            {
                "actorName": actor_name,
                "amount": amount,
                "damageType": damage_type,
            },
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        # Format result
        actual = data.get("actualDamage", amount)
        new_hp = data.get("newHP", {})
        hp_str = f"{new_hp.get('value', '?')}/{new_hp.get('max', '?')}" if new_hp else "?"

        result_parts = [f"Applied {actual} {damage_type} damage to {actor_name}"]
        if data.get("resistanceApplied"):
            result_parts.append(f"(resisted {data.get('resistanceAmount', 0)})")
        if data.get("weaknessApplied"):
            result_parts.append(f"(+{data.get('weaknessAmount', 0)} weakness)")
        result_parts.append(f"- HP now: {hp_str}")

        if data.get("defeated"):
            result_parts.append("- DEFEATED!")

        return ToolResult(success=True, data=" ".join(result_parts))

    def _heal(self, args: dict) -> ToolResult:
        """Heal an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot heal: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        amount = args.get("amount")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if amount is None:
            return ToolResult(success=False, error="Heal amount is required")

        data = self._bridge.send_command(
            "heal",
            {"actorName": actor_name, "amount": amount},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        actual = data.get("actualHealing", amount)
        new_hp = data.get("newHP", {})
        hp_str = f"{new_hp.get('value', '?')}/{new_hp.get('max', '?')}" if new_hp else "?"

        return ToolResult(
            success=True,
            data=f"Healed {actor_name} for {actual} HP - now at {hp_str}",
        )

    def _apply_condition(self, args: dict) -> ToolResult:
        """Apply a condition to an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot apply condition: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        condition = args.get("condition")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if not condition:
            return ToolResult(success=False, error="Condition name is required")

        value = args.get("value")

        data = self._bridge.send_command(
            "applyCondition",
            {
                "actorName": actor_name,
                "condition": condition,
                "value": value,
            },
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        condition_str = condition
        if value:
            condition_str = f"{condition} {value}"

        return ToolResult(
            success=True,
            data=f"Applied {condition_str} to {data.get('actorName', actor_name)}",
        )

    def _remove_condition(self, args: dict) -> ToolResult:
        """Remove a condition from an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot remove condition: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        condition = args.get("condition")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if not condition:
            return ToolResult(success=False, error="Condition name is required")

        data = self._bridge.send_command(
            "removeCondition",
            {"actorName": actor_name, "condition": condition},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Removed {condition} from {data.get('actorName', actor_name)}",
        )

    def _advance_turn(self, args: dict) -> ToolResult:
        """Advance combat to the next combatant's turn."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot advance turn: Foundry VTT is not connected",
            )

        data = self._bridge.send_command("advanceTurn")

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        next_combatant = data.get("nextCombatant", "Unknown")
        new_round = data.get("newRound", False)

        result = f"Advanced to {next_combatant}'s turn"
        if new_round:
            result += f" (Round {data.get('round', '?')})"

        return ToolResult(success=True, data=result)

    def _spawn_token(self, args: dict) -> ToolResult:
        """Spawn a token on the current scene."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot spawn token: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        x = args.get("x")
        y = args.get("y")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if x is None or y is None:
            return ToolResult(success=False, error="Position (x, y) is required")

        hidden = args.get("hidden", False)

        data = self._bridge.send_command(
            "spawnToken",
            {
                "actorName": actor_name,
                "x": x,
                "y": y,
                "hidden": hidden,
            },
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Spawned {data.get('tokenName', actor_name)} at ({x}, {y})"
            + (" [hidden]" if hidden else ""),
        )

    def _remove_token(self, args: dict) -> ToolResult:
        """Remove a token from the current scene."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot remove token: Foundry VTT is not connected",
            )

        name = args.get("name")
        if not name:
            return ToolResult(success=False, error="Token name is required")

        data = self._bridge.send_command("removeToken", {"name": name})

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Removed token: {name}",
        )

    def _whisper(self, args: dict) -> ToolResult:
        """Send a private whisper message to a specific player."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot whisper: Foundry VTT is not connected",
            )

        player_name = args.get("player_name")
        content = args.get("content")

        if not player_name:
            return ToolResult(success=False, error="Player name is required")
        if not content:
            return ToolResult(success=False, error="Message content is required")

        data = self._bridge.send_command(
            "whisper",
            {"playerName": player_name, "content": content},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Whispered to {data.get('playerName', player_name)}",
        )

    # =========================================================================
    # Exploration Mode Tools
    # =========================================================================

    def _set_exploration_activity(self, args: dict) -> ToolResult:
        """Set exploration activity for an actor."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot set activity: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        activity = args.get("activity")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if not activity:
            return ToolResult(success=False, error="Activity is required")

        data = self._bridge.send_command(
            "setExplorationActivity",
            {"actorName": actor_name, "activity": activity},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        return ToolResult(
            success=True,
            data=f"Set {data.get('actorName', actor_name)}'s exploration activity to {activity}",
        )

    def _get_exploration_state(self, args: dict) -> ToolResult:
        """Get current exploration activities for all party members."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot get exploration state: Foundry VTT is not connected",
            )

        data = self._bridge.send_command("getExplorationState")

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        # Format the exploration state
        activities = data.get("activities", {})
        if not activities:
            return ToolResult(
                success=True,
                data="No exploration activities set for any party members.",
            )

        lines = ["## Current Exploration Activities"]
        for actor_name, activity in activities.items():
            lines.append(f"- **{actor_name}**: {activity}")

        return ToolResult(success=True, data="\n".join(lines))

    def _roll_secret_check(self, args: dict) -> ToolResult:
        """Roll a secret check (GM sees result, player doesn't)."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot roll: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        check_type = args.get("check_type")

        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")
        if not check_type:
            return ToolResult(success=False, error="Check type is required")

        dc = args.get("dc")

        data = self._bridge.send_command(
            "rollSecretCheck",
            {"actorName": actor_name, "checkType": check_type, "dc": dc},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        # Format result for GM
        total = data.get("total", "?")
        degree = data.get("degree")
        degree_str = f" ({degree})" if degree else ""

        result = f"Secret {check_type} check for {actor_name}: {total}{degree_str}"
        if dc:
            result += f" vs DC {dc}"

        return ToolResult(success=True, data=result)

    # =========================================================================
    # Rest and Recovery Tools
    # =========================================================================

    def _take_rest(self, args: dict) -> ToolResult:
        """Have party take a rest."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot rest: Foundry VTT is not connected",
            )

        rest_type = args.get("rest_type")
        if not rest_type:
            return ToolResult(success=False, error="Rest type is required")

        if rest_type not in ("short", "long"):
            return ToolResult(
                success=False,
                error="Rest type must be 'short' or 'long'",
            )

        actor_names = args.get("actor_names")

        data = self._bridge.send_command(
            "takeRest",
            {"restType": rest_type, "actorNames": actor_names},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        # Format result
        results = data.get("results", [])
        if not results:
            return ToolResult(
                success=True,
                data=f"Completed {rest_type} rest. No characters to heal.",
            )

        lines = [f"## {rest_type.title()} Rest Complete"]
        for result in results:
            name = result.get("actorName", "Unknown")
            healed = result.get("hpHealed", 0)
            conditions = result.get("conditionsRemoved", [])

            line = f"- **{name}**"
            if healed > 0:
                line += f": healed {healed} HP"
            if conditions:
                line += f", removed: {', '.join(conditions)}"
            if healed == 0 and not conditions:
                line += ": fully rested"
            lines.append(line)

        return ToolResult(success=True, data="\n".join(lines))

    def _refocus(self, args: dict) -> ToolResult:
        """Have an actor Refocus to recover a focus point."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot refocus: Foundry VTT is not connected",
            )

        actor_name = args.get("actor_name")
        if not actor_name:
            return ToolResult(success=False, error="Actor name is required")

        data = self._bridge.send_command(
            "refocus",
            {"actorName": actor_name},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        new_points = data.get("focusPoints", {})
        current = new_points.get("value", "?")
        max_points = new_points.get("max", "?")

        return ToolResult(
            success=True,
            data=f"{actor_name} refocused. Focus points: {current}/{max_points}",
        )

    # =========================================================================
    # Time Tracking Tools
    # =========================================================================

    def _advance_time(self, args: dict) -> ToolResult:
        """Advance in-game time."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot advance time: Foundry VTT is not connected",
            )

        amount = args.get("amount")
        unit = args.get("unit")

        if amount is None:
            return ToolResult(success=False, error="Amount is required")
        if not unit:
            return ToolResult(success=False, error="Unit is required")

        valid_units = ("rounds", "minutes", "hours", "days")
        if unit not in valid_units:
            return ToolResult(
                success=False,
                error=f"Unit must be one of: {', '.join(valid_units)}",
            )

        data = self._bridge.send_command(
            "advanceTime",
            {"amount": amount, "unit": unit},
        )

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        new_time = data.get("newTime", "Unknown")
        return ToolResult(
            success=True,
            data=f"Advanced time by {amount} {unit}. Current time: {new_time}",
        )

    def _get_time(self, args: dict) -> ToolResult:
        """Get current in-game time."""
        if not self._bridge or not self._bridge.is_connected():
            return ToolResult(
                success=False,
                error="Cannot get time: Foundry VTT is not connected",
            )

        data = self._bridge.send_command("getTime")

        if data.get("error"):
            return ToolResult(success=False, error=data["error"])

        time_str = data.get("time", "Unknown")
        date_str = data.get("date", "")
        day_of_week = data.get("dayOfWeek", "")

        result = f"Current time: {time_str}"
        if date_str:
            result += f"\nDate: {date_str}"
        if day_of_week:
            result += f" ({day_of_week})"

        return ToolResult(success=True, data=result)

    def _format_aca_game_state(self, data: dict) -> str:
        """Format ACA game state for LLM context."""
        game_state = data.get("gameState", {})
        combatant_name = data.get("combatantName", "Unknown")

        lines = [f"## ACA Game State for {combatant_name}"]

        # Self info
        self_info = game_state.get("self", {})
        if self_info:
            lines.append("\n### Self:")
            if self_info.get("name"):
                lines.append(f"- Name: {self_info['name']}")
            if self_info.get("hp"):
                hp = self_info["hp"]
                lines.append(f"- HP: {hp.get('value', '?')}/{hp.get('max', '?')}")
            if self_info.get("ac"):
                lines.append(f"- AC: {self_info['ac']}")
            if self_info.get("conditions"):
                lines.append(f"- Conditions: {', '.join(self_info['conditions'])}")

        # Allies
        allies = game_state.get("allies", [])
        if allies:
            lines.append("\n### Allies:")
            for ally in allies[:5]:  # Limit to 5
                name = ally.get("name", "Unknown")
                hp = ally.get("hp", {})
                hp_str = f" (HP: {hp.get('value', '?')}/{hp.get('max', '?')})" if hp else ""
                lines.append(f"- {name}{hp_str}")

        # Enemies
        enemies = game_state.get("enemies", [])
        if enemies:
            lines.append("\n### Enemies:")
            for enemy in enemies[:5]:  # Limit to 5
                name = enemy.get("name", "Unknown")
                hp = enemy.get("hp", {})
                hp_str = f" (HP: {hp.get('value', '?')}/{hp.get('max', '?')})" if hp else ""
                dist = enemy.get("distance")
                dist_str = f" [{dist}ft away]" if dist else ""
                lines.append(f"- {name}{hp_str}{dist_str}")

        # Available actions summary
        strikes = game_state.get("strikes", [])
        if strikes:
            lines.append(f"\n### Available Strikes: {len(strikes)}")
            for strike in strikes[:3]:
                name = strike.get("name", "Strike")
                bonus = strike.get("totalModifier", "?")
                lines.append(f"- {name} (+{bonus})")

        spells = game_state.get("spells", [])
        if spells:
            lines.append(f"\n### Available Spells: {len(spells)}")

        return "\n".join(lines)

    def _format_aca_state(self, data: dict) -> str:
        """Format ACA state for LLM context."""
        if not data.get("active"):
            installed = "installed but inactive" if data.get("installed") else "not installed"
            return f"AI Combat Assistant is {installed}"

        lines = ["## AI Combat Assistant Status"]
        lines.append("Status: Active")

        if data.get("combatActive"):
            lines.append("Combat: In progress")
        else:
            lines.append("Combat: Not active")

        designations = data.get("designations", {})
        if designations:
            lines.append("\n### NPC Designations:")
            for name, designation in designations.items():
                lines.append(f"- {name}: {designation}")
        else:
            lines.append("\nNo NPCs designated for AI control")

        processing = data.get("processingActors", [])
        if processing:
            lines.append("\n### Currently Processing:")
            for actor in processing:
                lines.append(f"- {actor['name']} (thinking...)")

        return "\n".join(lines)

    def _format_aca_turn_state(self, data: dict) -> str:
        """Format ACA turn state for LLM context."""
        lines = [f"## ACA Turn State: {data.get('actorName', 'Unknown')}"]

        if data.get("isProcessing"):
            lines.append("Status: Currently processing turn")
        else:
            lines.append("Status: Idle")

        if not data.get("hasTurnState"):
            lines.append("No active turn state (not this actor's turn)")
        else:
            lines.append(f"Actions Remaining: {data.get('actionsRemaining', '?')}")
            lines.append(f"Current MAP: {data.get('currentMAP', 0)}")

            history = data.get("actionHistory", [])
            if history:
                lines.append("\n### Actions Taken This Turn:")
                for action in history:
                    lines.append(f"- {action}")

        notes = data.get("permanentNotes", "")
        if notes:
            lines.append(f"\n### Tactical Notes:\n{notes}")

        return "\n".join(lines)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _handle_disconnected(self, tool_name: str, args: dict, error: str) -> ToolResult:
        """Handle tool call when Foundry is disconnected."""
        # For read tools, try to return cached data
        read_tools = {
            "foundry_get_scene": "scene",
            "foundry_get_combat_state": "combat_state",
            "foundry_is_combat_active": "combat_state",
        }

        if tool_name in read_tools and self._bridge:
            cache_key = read_tools[tool_name]
            cached = self._bridge.get_cached(cache_key)
            if cached:
                if tool_name == "foundry_get_scene":
                    return ToolResult(
                        success=True,
                        data=self._format_scene(cached) + "\n(Cached data - Foundry disconnected)",
                    )
                elif tool_name == "foundry_get_combat_state":
                    return ToolResult(
                        success=True,
                        data=self._format_combat_state(cached)
                        + "\n(Cached data - Foundry disconnected)",
                    )
                elif tool_name == "foundry_is_combat_active":
                    active = cached.get("active", False)
                    return ToolResult(
                        success=True,
                        data=f"Combat is {'active' if active else 'not active'} (cached)",
                    )

        return ToolResult(success=False, error=f"{error}. Cannot execute {tool_name}.")

    def _format_scene(self, data: dict) -> str:
        """Format scene data for LLM context."""
        lines = []
        lines.append(f"## Scene: {data.get('name', 'Unknown')}")

        if data.get("darkness"):
            lines.append(f"Darkness Level: {data['darkness']}")

        tokens = data.get("tokens", [])
        if tokens:
            lines.append("\n### Tokens:")
            for token in tokens:
                pos = f"({token.get('x', '?')}, {token.get('y', '?')})"
                hidden = " [HIDDEN]" if token.get("hidden") else ""
                elevation = (
                    f" (elevation: {token['elevation']}ft)" if token.get("elevation") else ""
                )
                lines.append(f"- {token.get('name', 'Unknown')} at {pos}{elevation}{hidden}")

        walls = data.get("wallCount", 0)
        lights = data.get("lightCount", 0)
        if walls or lights:
            lines.append(f"\nScene has {walls} walls and {lights} light sources")

        return "\n".join(lines)

    def _format_actors(self, data: dict) -> str:
        """Format actor data for LLM context."""
        actors = data.get("actors", [])
        if not actors:
            return "No actors found"

        lines = ["## Actors:"]
        for actor in actors:
            name = actor.get("name", "Unknown")
            level = actor.get("level")
            level_str = f" (Level {level})" if level else ""

            hp = actor.get("hp", {})
            hp_str = ""
            if hp:
                hp_str = f" - HP: {hp.get('value', '?')}/{hp.get('max', '?')}"

            conditions = actor.get("conditions", [])
            cond_str = f" [{', '.join(conditions)}]" if conditions else ""

            lines.append(f"- {name}{level_str}{hp_str}{cond_str}")

        return "\n".join(lines)

    def _format_combat_state(self, data: dict) -> str:
        """Format combat state for LLM context."""
        if not data.get("active"):
            return "No combat is currently active"

        lines = []
        lines.append(f"## Combat - Round {data.get('round', 1)}")

        ai_assistant = data.get("aiCombatAssistantActive", False)
        if ai_assistant:
            lines.append("(AI Combat Assistant is handling NPC tactics)")

        current_turn = data.get("turn", 0)
        combatants = data.get("combatants", [])

        if combatants:
            lines.append("\n### Turn Order:")
            for i, c in enumerate(combatants):
                marker = ">>>" if i == current_turn else "   "
                name = c.get("name", "Unknown")
                init = c.get("initiative", "?")

                hp = c.get("hp", {})
                hp_str = ""
                if hp:
                    hp_str = f" HP: {hp.get('value', '?')}/{hp.get('max', '?')}"

                conditions = c.get("conditions", [])
                cond_str = f" [{', '.join(conditions)}]" if conditions else ""

                defeated = " [DEFEATED]" if c.get("defeated") else ""

                lines.append(f"{marker} {init}: {name}{hp_str}{cond_str}{defeated}")

        return "\n".join(lines)
