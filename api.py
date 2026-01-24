"""Flask REST API for GM Agent."""

import atexit
import logging
import os
import signal
import sys
from datetime import timedelta
from functools import wraps

from flask import Flask, jsonify, request, Response, stream_with_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from flask_socketio import SocketIO, emit
from flasgger import Swagger, swag_from

from gm_agent.agent import GMAgent
from gm_agent.chat import ChatAgent
from gm_agent.config import (
    LLM_BACKEND,
    CHECKPOINT_STALE_THRESHOLD,
    FOUNDRY_MODE,
    FOUNDRY_POLL_URL,
    FOUNDRY_API_KEY,
    FOUNDRY_CAMPAIGN_ID,
    FOUNDRY_POLL_INTERVAL,
    FOUNDRY_LONG_POLL_TIMEOUT,
    FOUNDRY_VERIFY_SSL,
)
from gm_agent.game_loop import GameLoopController
from gm_agent.mcp.foundry_vtt import FoundryBridge, FoundryVTTServer
from gm_agent.mcp.foundry_bridge_base import FoundryBridgeBase
from gm_agent.mcp.foundry_polling import FoundryPollingClient, PollingConfig
from gm_agent.models.factory import get_backend, list_backends
from gm_agent.storage.campaign import campaign_store
from gm_agent.storage.session import session_store
from gm_agent.storage.characters import get_character_store
from gm_agent.storage.schemas import PartyMember

app = Flask(__name__)

# Enable CORS for browser requests (Foundry VTT integration)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# =============================================================================
# Configuration
# =============================================================================

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

# Auth can be disabled via environment variable
AUTH_ENABLED = os.environ.get("API_AUTH_ENABLED", "false").lower() == "true"

jwt = JWTManager(app)

# Swagger Configuration
app.config["SWAGGER"] = {
    "title": "GM Agent API",
    "description": "REST API for Pathfinder 2E GM Agent",
    "version": "0.1.0",
    "termsOfService": "",
    "specs_route": "/api/docs/",
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "GM Agent API",
        "description": "REST API for AI-powered Pathfinder 2E Game Master assistant",
        "version": "0.1.0",
    },
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "JWT Authorization header. Example: 'Bearer {token}'",
        }
    },
    "security": [{"Bearer": []}] if AUTH_ENABLED else [],
}

swagger = Swagger(app, template=swagger_template)

# Store active agents by campaign ID
_active_agents: dict[str, GMAgent] = {}

# =============================================================================
# Socket.IO Configuration (Foundry VTT Bridge)
# =============================================================================

# Initialize Socket.IO for bidirectional Foundry communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global Foundry bridge instance (can be FoundryBridge or FoundryPollingClient)
foundry_bridge: FoundryBridgeBase | None = None
foundry_server: FoundryVTTServer | None = None

# Polling client instance (only used in polling mode)
_polling_client: FoundryPollingClient | None = None

# Global game loop controllers by campaign ID
_game_loops: dict[str, GameLoopController] = {}


@socketio.on("connect")
def handle_connect():
    """Handle Foundry VTT client connection."""
    global foundry_bridge, foundry_server
    logger.info(f"Foundry VTT connected: {request.sid}")

    foundry_bridge = FoundryBridge(socketio)
    foundry_bridge.set_connected(request.sid)

    foundry_server = FoundryVTTServer(foundry_bridge)

    # Update existing agents with the new foundry server
    for agent in _active_agents.values():
        agent.set_foundry_server(foundry_server)

    # Update existing game loops with the new bridge
    # (Foundry will re-send automationToggle to restart them)
    for loop in _game_loops.values():
        loop.set_bridge(foundry_bridge)


@socketio.on("disconnect")
def handle_disconnect():
    """Handle Foundry VTT client disconnection."""
    global foundry_bridge, foundry_server
    logger.info(f"Foundry VTT disconnected: {request.sid}")

    if foundry_bridge:
        foundry_bridge.set_disconnected()

    # Remove foundry server from existing agents
    for agent in _active_agents.values():
        agent.set_foundry_server(None)

    # Stop all game loops (they can't function without Foundry)
    for loop in _game_loops.values():
        loop.stop()

    foundry_bridge = None
    foundry_server = None


@socketio.on("foundry:response")
def handle_foundry_response(data):
    """Handle response from Foundry VTT command (WebSocket mode only)."""
    if foundry_bridge and isinstance(foundry_bridge, FoundryBridge):
        foundry_bridge.handle_response(data)


@socketio.on("foundry:event")
def handle_foundry_event(data):
    """Handle event pushed from Foundry VTT (WebSocket mode only)."""
    if foundry_bridge and isinstance(foundry_bridge, FoundryBridge):
        foundry_bridge.handle_event(data)


@socketio.on("automationToggle")
def handle_automation_toggle(data):
    """Handle automation enable/disable from Foundry.

    Args:
        data: Dict with campaignId, enabled, and optional dry_run fields
    """
    campaign_id = data.get("campaignId")
    enabled = data.get("enabled", False)

    if not campaign_id:
        emit("automationError", {"error": "Campaign ID is required"})
        return

    # Check if campaign exists
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        emit("automationError", {"error": f"Campaign '{campaign_id}' not found"})
        return

    # Get dry_run from data or campaign preferences
    dry_run = data.get("dry_run", campaign.preferences.get("dry_run", False))

    if enabled:
        # Start automation
        if campaign_id not in _game_loops:
            try:
                agent = get_agent(campaign_id)
                _game_loops[campaign_id] = GameLoopController(
                    campaign_id, agent, foundry_bridge, dry_run=dry_run
                )
            except Exception as e:
                emit("automationError", {"error": str(e)})
                return
        else:
            # Update dry_run setting on existing controller
            _game_loops[campaign_id].dry_run = dry_run

        _game_loops[campaign_id].start()
        logger.info(
            f"Automation enabled for campaign: {campaign_id}"
            f"{' (dry run mode)' if dry_run else ''}"
        )
        emit("automationStatus", {"campaignId": campaign_id, "enabled": True, "dryRun": dry_run})
        _publish_loop_state()  # Checkpoint state change
    else:
        # Stop automation
        if campaign_id in _game_loops:
            _game_loops[campaign_id].stop()
            logger.info(f"Automation disabled for campaign: {campaign_id}")
        emit("automationStatus", {"campaignId": campaign_id, "enabled": False})
        _publish_loop_state()  # Checkpoint state change


@socketio.on("playerChat")
def handle_player_chat(data):
    """Forward player chat to game loop controller.

    This is called when the Foundry module forwards a player chat message.
    The game loop controller will decide whether to respond based on
    automation settings.

    Args:
        data: Dict with campaignId, actorName, content, playerId, timestamp
    """
    if not foundry_bridge:
        return

    # Forward to bridge event system which will trigger game loop handlers
    foundry_bridge.handle_event(
        {
            "eventType": "playerChat",
            "payload": data,
        }
    )


@socketio.on("combatTurn")
def handle_combat_turn_event(data):
    """Forward combat turn event to game loop controller.

    This is called when combat turn changes in Foundry.
    The game loop controller will handle NPC turns automatically.

    Args:
        data: Dict with round, turn, combatant info
    """
    if not foundry_bridge:
        return

    # Forward to bridge event system which will trigger game loop handlers
    foundry_bridge.handle_event(
        {
            "eventType": "combatTurn",
            "payload": data,
        }
    )


# =============================================================================
# Auth Helpers
# =============================================================================


def optional_jwt_required(fn):
    """Decorator that requires JWT only if AUTH_ENABLED is True."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if AUTH_ENABLED:
            return jwt_required()(fn)(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


def get_agent(campaign_id: str) -> GMAgent:
    """Get or create an agent for a campaign.

    If a Foundry VTT server is connected, it will be passed to the agent
    to enable Foundry tool usage (get_combat_state, roll_check, etc.).
    """
    if campaign_id not in _active_agents:
        _active_agents[campaign_id] = GMAgent(
            campaign_id,
            foundry_server=foundry_server,
        )
    return _active_agents[campaign_id]


def close_agent(campaign_id: str) -> None:
    """Close and remove an agent."""
    if campaign_id in _active_agents:
        _active_agents[campaign_id].close()
        del _active_agents[campaign_id]


# =============================================================================
# Auth Endpoints
# =============================================================================


@app.route("/api/auth/token", methods=["POST"])
def get_token():
    """
    Get JWT access token
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
              example: admin
            password:
              type: string
              example: password
    responses:
      200:
        description: Access token
        schema:
          type: object
          properties:
            access_token:
              type: string
            token_type:
              type: string
              example: bearer
            expires_in:
              type: integer
              example: 86400
      400:
        description: Missing credentials
      401:
        description: Invalid credentials
      403:
        description: Auth is disabled
    """
    if not AUTH_ENABLED:
        return jsonify({"error": "Authentication is disabled"}), 403

    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"error": "Username and password required"}), 400

    # Simple auth - in production, use proper user management
    api_username = os.environ.get("API_USERNAME", "admin")
    api_password = os.environ.get("API_PASSWORD", "changeme")

    if data["username"] == api_username and data["password"] == api_password:
        access_token = create_access_token(identity=data["username"])
        return jsonify(
            {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": 86400,
            }
        )

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """
    Check authentication status
    ---
    tags:
      - Authentication
    responses:
      200:
        description: Auth status
        schema:
          type: object
          properties:
            auth_enabled:
              type: boolean
            message:
              type: string
    """
    return jsonify(
        {
            "auth_enabled": AUTH_ENABLED,
            "message": (
                "Authentication is enabled" if AUTH_ENABLED else "Authentication is disabled"
            ),
        }
    )


# =============================================================================
# Campaign Endpoints
# =============================================================================


@app.route("/api/campaigns", methods=["GET"])
@optional_jwt_required
def list_campaigns():
    """
    List all campaigns
    ---
    tags:
      - Campaigns
    responses:
      200:
        description: List of campaigns
        schema:
          type: object
          properties:
            campaigns:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: string
                  name:
                    type: string
                  background:
                    type: string
                  party_size:
                    type: integer
                  created_at:
                    type: string
                    format: date-time
    """
    campaigns = campaign_store.list()
    return jsonify(
        {
            "campaigns": [
                {
                    "id": c.id,
                    "name": c.name,
                    "background": (
                        c.background[:200] + "..." if len(c.background) > 200 else c.background
                    ),
                    "party_size": len(c.party),
                    "created_at": c.created_at.isoformat(),
                }
                for c in campaigns
            ]
        }
    )


@app.route("/api/campaigns", methods=["POST"])
@optional_jwt_required
def create_campaign():
    """
    Create a new campaign
    ---
    tags:
      - Campaigns
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
          properties:
            name:
              type: string
              example: Abomination Vaults
            background:
              type: string
              example: A dungeon crawl adventure in Otari
            current_arc:
              type: string
            party:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  class_name:
                    type: string
                  level:
                    type: integer
            preferences:
              type: object
    responses:
      201:
        description: Campaign created
      400:
        description: Invalid request
    """
    data = request.get_json()

    if not data or "name" not in data:
        return jsonify({"error": "Campaign name is required"}), 400

    # Parse party members if provided
    party = []
    if "party" in data:
        for member_data in data["party"]:
            party.append(PartyMember(**member_data))

    try:
        campaign = campaign_store.create(
            name=data["name"],
            background=data.get("background", ""),
            current_arc=data.get("current_arc", ""),
            party=party,
            preferences=data.get("preferences", {}),
        )
        return (
            jsonify(
                {
                    "id": campaign.id,
                    "name": campaign.name,
                    "message": f"Campaign '{campaign.name}' created successfully",
                }
            ),
            201,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/campaigns/<campaign_id>", methods=["GET"])
@optional_jwt_required
def get_campaign(campaign_id: str):
    """
    Get a campaign by ID
    ---
    tags:
      - Campaigns
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Campaign details
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    return jsonify(
        {
            "id": campaign.id,
            "name": campaign.name,
            "background": campaign.background,
            "current_arc": campaign.current_arc,
            "party": [m.model_dump() for m in campaign.party],
            "preferences": campaign.preferences,
            "created_at": campaign.created_at.isoformat(),
            "updated_at": campaign.updated_at.isoformat(),
        }
    )


@app.route("/api/campaigns/<campaign_id>", methods=["PUT"])
@optional_jwt_required
def update_campaign(campaign_id: str):
    """
    Update a campaign
    ---
    tags:
      - Campaigns
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        schema:
          type: object
          properties:
            name:
              type: string
            background:
              type: string
            current_arc:
              type: string
            preferences:
              type: object
            party:
              type: array
              items:
                type: object
    responses:
      200:
        description: Campaign updated
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Update fields
    if "name" in data:
        campaign.name = data["name"]
    if "background" in data:
        campaign.background = data["background"]
    if "current_arc" in data:
        campaign.current_arc = data["current_arc"]
    if "preferences" in data:
        campaign.preferences.update(data["preferences"])
    if "party" in data:
        campaign.party = [PartyMember(**m) for m in data["party"]]

    campaign_store.update(campaign)

    return jsonify(
        {
            "id": campaign.id,
            "name": campaign.name,
            "message": "Campaign updated successfully",
        }
    )


@app.route("/api/campaigns/<campaign_id>", methods=["DELETE"])
@optional_jwt_required
def delete_campaign(campaign_id: str):
    """
    Delete a campaign
    ---
    tags:
      - Campaigns
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Campaign deleted
      404:
        description: Campaign not found
    """
    # Stop and remove any game loop for this campaign
    if campaign_id in _game_loops:
        _game_loops[campaign_id].stop()
        del _game_loops[campaign_id]

    # Close any active agent
    close_agent(campaign_id)

    if campaign_store.delete(campaign_id):
        return jsonify({"message": f"Campaign '{campaign_id}' deleted"})
    else:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404


# =============================================================================
# Session Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/sessions", methods=["GET"])
@optional_jwt_required
def list_sessions(campaign_id: str):
    """
    List all sessions for a campaign
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: List of sessions
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    sessions = session_store.list(campaign_id)
    current = session_store.get_current(campaign_id)

    result = {
        "sessions": [
            {
                "id": s.id,
                "turns": len(s.turns),
                "started_at": s.started_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "summary": (s.summary[:200] + "..." if len(s.summary) > 200 else s.summary),
            }
            for s in sessions
        ],
        "current_session": None,
    }

    if current:
        result["current_session"] = {
            "id": current.id,
            "turns": len(current.turns),
            "started_at": current.started_at.isoformat(),
        }

    return jsonify(result)


@app.route("/api/campaigns/<campaign_id>/sessions/start", methods=["POST"])
@optional_jwt_required
def start_session(campaign_id: str):
    """
    Start a new session
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      201:
        description: Session started
      400:
        description: Session already active
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    # Check if there's already an active session
    current = session_store.get_current(campaign_id)
    if current:
        return (
            jsonify(
                {
                    "error": "Session already active. End it first with /sessions/end",
                    "session_id": current.id,
                }
            ),
            400,
        )

    session = session_store.start(campaign_id)

    # Initialize agent
    get_agent(campaign_id)

    return (
        jsonify(
            {
                "session_id": session.id,
                "message": "Session started",
                "scene": {
                    "location": session.scene_state.location,
                    "time_of_day": session.scene_state.time_of_day,
                },
            }
        ),
        201,
    )


@app.route("/api/campaigns/<campaign_id>/sessions/turn", methods=["POST"])
@optional_jwt_required
def process_turn(campaign_id: str):
    """
    Process a player turn
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - input
          properties:
            input:
              type: string
              example: I examine the altar
    responses:
      200:
        description: GM response
        schema:
          type: object
          properties:
            response:
              type: string
            turn_number:
              type: integer
            scene:
              type: object
      400:
        description: No active session or missing input
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    current = session_store.get_current(campaign_id)
    if not current:
        return (
            jsonify({"error": "No active session. Start one first with /sessions/start"}),
            400,
        )

    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Player input is required"}), 400

    player_input = data["input"]

    try:
        agent = get_agent(campaign_id)
        response = agent.process_turn(player_input)

        # Get updated session state
        session = session_store.get_current(campaign_id)

        return jsonify(
            {
                "response": response,
                "turn_number": len(session.turns),
                "scene": {
                    "location": session.scene_state.location,
                    "time_of_day": session.scene_state.time_of_day,
                    "npcs_present": session.scene_state.npcs_present,
                    "conditions": session.scene_state.conditions,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/campaigns/<campaign_id>/sessions/turn/stream", methods=["POST"])
@optional_jwt_required
def process_turn_stream(campaign_id: str):
    """
    Process a player turn with streaming response (Server-Sent Events)
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - input
          properties:
            input:
              type: string
              example: I examine the altar
    produces:
      - text/event-stream
    responses:
      200:
        description: Streaming GM response (Server-Sent Events)
        headers:
          Content-Type:
            type: string
            default: text/event-stream
      400:
        description: No active session or missing input
      404:
        description: Campaign not found
    """
    import json

    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    current = session_store.get_current(campaign_id)
    if not current:
        return (
            jsonify({"error": "No active session. Start one first with /sessions/start"}),
            400,
        )

    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Player input is required"}), 400

    player_input = data["input"]

    def generate():
        """Generator for SSE stream."""
        try:
            agent = get_agent(campaign_id)

            # Stream response chunks
            for chunk in agent.process_turn_stream(player_input):
                # Format as SSE
                event_data = {
                    "delta": chunk.delta,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "args": tc.args} for tc in chunk.tool_calls
                    ]
                    if chunk.tool_calls
                    else [],
                    "finish_reason": chunk.finish_reason,
                    "usage": chunk.usage,
                }

                yield f"data: {json.dumps(event_data)}\n\n"

            # Send final metadata with updated session state
            session = session_store.get_current(campaign_id)
            final_event = {
                "type": "complete",
                "turn_number": len(session.turns),
                "scene": {
                    "location": session.scene_state.location,
                    "time_of_day": session.scene_state.time_of_day,
                    "npcs_present": session.scene_state.npcs_present,
                    "conditions": session.scene_state.conditions,
                },
            }
            yield f"data: {json.dumps(final_event)}\n\n"

        except Exception as e:
            error_event = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.route("/api/campaigns/<campaign_id>/sessions/end", methods=["POST"])
@optional_jwt_required
def end_session(campaign_id: str):
    """
    End the current session
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        schema:
          type: object
          properties:
            summary:
              type: string
            auto_generate:
              type: boolean
              default: false
    responses:
      200:
        description: Session ended
      400:
        description: No active session
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    current = session_store.get_current(campaign_id)
    if not current:
        return jsonify({"error": "No active session"}), 400

    data = request.get_json() or {}
    summary = data.get("summary", "")
    auto_generate = data.get("auto_generate", False)

    # Use agent to end session (may auto-generate summary)
    try:
        agent = get_agent(campaign_id)
        session = agent.end_session(summary=summary, auto_generate=auto_generate)
        close_agent(campaign_id)

        if session:
            return jsonify(
                {
                    "session_id": session.id,
                    "turns": len(session.turns),
                    "summary": session.summary,
                    "message": "Session ended",
                }
            )
        else:
            return jsonify({"error": "Failed to end session"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/campaigns/<campaign_id>/sessions/<session_id>/replay", methods=["POST"])
@optional_jwt_required
def replay_session_endpoint(campaign_id: str, session_id: str):
    """
    Replay a recorded session with optional speed control
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: session_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        schema:
          type: object
          properties:
            speed_multiplier:
              type: number
              default: 0.0
              description: Speed multiplier (1.0 = real-time, 10.0 = 10x, 0.0 = instant)
            backend:
              type: string
              description: Optional LLM backend name (ollama, openai, anthropic)
            model:
              type: string
              description: Optional model name to use
    produces:
      - text/event-stream
    responses:
      200:
        description: Streaming replay results (Server-Sent Events)
      404:
        description: Campaign or session not found
    """
    import json
    from gm_agent.replay import SessionReplayer
    from gm_agent.models.factory import get_backend

    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json() or {}
    speed_multiplier = data.get("speed_multiplier", 0.0)
    backend_name = data.get("backend")
    model_name = data.get("model")

    # Get LLM backend
    llm = None
    if backend_name:
        llm = get_backend(backend=backend_name, model=model_name)

    def generate():
        """Generator for SSE stream."""
        try:
            replayer = SessionReplayer(campaign_id)

            for result in replayer.replay(session_id, speed_multiplier=speed_multiplier, llm=llm):
                yield f"data: {json.dumps(result)}\n\n"

            # Send completion event
            completion = {"type": "complete", "message": "Replay finished"}
            yield f"data: {json.dumps(completion)}\n\n"

        except Exception as e:
            error_event = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/campaigns/<campaign_id>/sessions/<session_id>/compare", methods=["POST"])
@optional_jwt_required
def compare_models_endpoint(campaign_id: str, session_id: str):
    """
    Compare different models by replaying the same session
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: session_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - models
          properties:
            models:
              type: array
              description: List of model configurations to compare
              items:
                type: object
                properties:
                  name:
                    type: string
                  backend:
                    type: string
                  model:
                    type: string
              example:
                - name: "llama3"
                  backend: "ollama"
                  model: "llama3"
                - name: "gpt-4"
                  backend: "openai"
                  model: "gpt-4"
    responses:
      200:
        description: Model comparison results
        schema:
          type: object
          properties:
            results:
              type: object
              description: Results for each model
            _summary:
              type: object
              description: Comparison summary statistics
      400:
        description: Invalid request
      404:
        description: Campaign or session not found
    """
    from gm_agent.replay import SessionReplayer
    from gm_agent.models.factory import get_backend

    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "models" not in data:
        return jsonify({"error": "Request must include 'models' array"}), 400

    model_configs = data["models"]
    if not isinstance(model_configs, list) or len(model_configs) == 0:
        return jsonify({"error": "'models' must be a non-empty array"}), 400

    try:
        # Build list of (name, backend) tuples
        model_backends = []
        for config in model_configs:
            name = config.get("name", "unnamed")
            backend_name = config.get("backend", "ollama")
            model_name = config.get("model")

            backend = get_backend(backend=backend_name, model=model_name)
            model_backends.append((name, backend))

        # Run comparison
        replayer = SessionReplayer(campaign_id)
        results = replayer.compare_models(session_id, model_backends, verbose=True)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/campaigns/<campaign_id>/automation", methods=["POST"])
@optional_jwt_required
def toggle_automation(campaign_id: str):
    """
    Toggle automation mode for a campaign
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - enabled
          properties:
            enabled:
              type: boolean
              description: Whether to enable or disable automation
            dry_run:
              type: boolean
              description: If true, log responses instead of posting to Foundry (default false)
    responses:
      200:
        description: Automation status changed
        schema:
          type: object
          properties:
            campaign_id:
              type: string
            enabled:
              type: boolean
            dry_run:
              type: boolean
            message:
              type: string
      400:
        description: Missing required field or Foundry not connected
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "enabled" not in data:
        return jsonify({"error": "enabled field is required"}), 400

    enabled = data.get("enabled", False)
    dry_run = data.get("dry_run", False)

    if not foundry_bridge or not foundry_bridge.is_connected():
        return jsonify({"error": "Foundry VTT is not connected"}), 400

    if enabled:
        # Start automation
        if campaign_id not in _game_loops:
            try:
                agent = get_agent(campaign_id)
                _game_loops[campaign_id] = GameLoopController(
                    campaign_id, agent, foundry_bridge, dry_run=dry_run
                )
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            # Update dry_run setting on existing controller
            _game_loops[campaign_id].dry_run = dry_run

        _game_loops[campaign_id].start()
        _publish_loop_state()  # Checkpoint state change
        return jsonify(
            {
                "campaign_id": campaign_id,
                "enabled": True,
                "dry_run": dry_run,
                "message": f"Automation enabled{' (dry run mode)' if dry_run else ''}",
            }
        )
    else:
        # Stop automation
        if campaign_id in _game_loops:
            _game_loops[campaign_id].stop()
        _publish_loop_state()  # Checkpoint state change
        return jsonify(
            {
                "campaign_id": campaign_id,
                "enabled": False,
                "message": "Automation disabled",
            }
        )


@app.route("/api/campaigns/<campaign_id>/automation", methods=["GET"])
@optional_jwt_required
def get_automation_status(campaign_id: str):
    """
    Get automation status for a campaign
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Automation status with detailed stats
        schema:
          type: object
          properties:
            campaign_id:
              type: string
            enabled:
              type: boolean
            foundry_connected:
              type: boolean
            stats:
              type: object
              description: Detailed automation statistics (present when controller exists)
              properties:
                enabled:
                  type: boolean
                  description: Whether automation is currently enabled
                started_at:
                  type: number
                  description: Unix timestamp when automation was started
                uptime_seconds:
                  type: number
                  description: Seconds since automation was started
                response_count:
                  type: integer
                  description: Total responses sent
                player_chat_count:
                  type: integer
                  description: Player chat responses sent
                npc_turn_count:
                  type: integer
                  description: NPC turn responses sent
                batched_message_count:
                  type: integer
                  description: Total messages processed across all batches
                average_batch_size:
                  type: number
                  description: Average messages per batch (null if no batches)
                pending_batches:
                  type: integer
                  description: Number of player batches awaiting flush
                pending_messages:
                  type: integer
                  description: Total messages in pending batches
                oldest_batch_age_seconds:
                  type: number
                  description: Age of oldest pending batch (null if none)
                error_count:
                  type: integer
                  description: Total errors encountered
                consecutive_errors:
                  type: integer
                  description: Current consecutive error count
                max_consecutive_errors:
                  type: integer
                  description: Threshold for auto-disable
                total_processing_time_ms:
                  type: number
                  description: Total LLM processing time in milliseconds
                last_response_time:
                  type: number
                  description: Unix timestamp of last response
                seconds_since_last_response:
                  type: number
                  description: Seconds since last response (null if none)
                batch_window_seconds:
                  type: number
                  description: Configured batch window duration
                max_batch_size:
                  type: integer
                  description: Configured max messages per batch
                cooldown_seconds:
                  type: number
                  description: NPC turn cooldown duration
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    loop = _game_loops.get(campaign_id)
    enabled = loop is not None and loop.enabled

    result = {
        "campaign_id": campaign_id,
        "enabled": enabled,
        "foundry_connected": foundry_bridge is not None and foundry_bridge.is_connected(),
    }

    # Include stats if automation is active
    if loop:
        result["stats"] = loop.get_stats()

    return jsonify(result)


@app.route("/api/campaigns/<campaign_id>/automation/stats/reset", methods=["POST"])
@optional_jwt_required
def reset_automation_stats(campaign_id: str):
    """
    Reset automation statistics for a campaign
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Stats reset successfully
        schema:
          type: object
          properties:
            campaign_id:
              type: string
            message:
              type: string
      404:
        description: Campaign not found or automation not initialized
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    loop = _game_loops.get(campaign_id)
    if not loop:
        return jsonify({"error": "Automation not initialized for this campaign"}), 404

    loop.reset_stats()
    return jsonify(
        {
            "campaign_id": campaign_id,
            "message": "Automation stats reset successfully",
        }
    )


@app.route("/api/campaigns/<campaign_id>/automation/queue/stats", methods=["GET"])
@optional_jwt_required
def get_queue_stats(campaign_id: str):
    """
    Get event queue statistics for a campaign
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Queue statistics
        schema:
          type: object
          properties:
            campaign_id:
              type: string
            queue_enabled:
              type: boolean
            stats:
              type: object
      404:
        description: Campaign not found or automation not initialized
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    loop = _game_loops.get(campaign_id)
    if not loop:
        return jsonify({"error": "Automation not initialized for this campaign"}), 404

    queue_stats = loop.get_queue_stats()
    return jsonify(
        {
            "campaign_id": campaign_id,
            "queue_enabled": loop.enable_queue,
            "stats": queue_stats,
        }
    )


@app.route("/api/campaigns/<campaign_id>/automation/queue/clear", methods=["POST"])
@optional_jwt_required
def clear_queue(campaign_id: str):
    """
    Clear all events from the automation queue
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Queue cleared successfully
        schema:
          type: object
          properties:
            campaign_id:
              type: string
            events_removed:
              type: integer
            message:
              type: string
      404:
        description: Campaign not found or automation not initialized
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    loop = _game_loops.get(campaign_id)
    if not loop:
        return jsonify({"error": "Automation not initialized for this campaign"}), 404

    events_removed = loop.clear_queue()
    return jsonify(
        {
            "campaign_id": campaign_id,
            "events_removed": events_removed,
            "message": f"Cleared {events_removed} events from queue",
        }
    )


@app.route("/api/campaigns/<campaign_id>/automation/async", methods=["POST"])
@optional_jwt_required
def process_turn_async(campaign_id: str):
    """
    Process a turn asynchronously via Celery
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            type:
              type: string
              enum: [player_chat, npc_turn]
              description: Type of turn to process
            player_input:
              type: string
              description: Required for player_chat type
            npc_name:
              type: string
              description: Required for npc_turn type
            prompt:
              type: string
              description: Optional custom prompt for npc_turn
            metadata:
              type: object
              description: Optional turn metadata
    responses:
      202:
        description: Task accepted and queued
        schema:
          type: object
          properties:
            task_id:
              type: string
              description: Celery task ID for polling
            campaign_id:
              type: string
            status_url:
              type: string
              description: URL to poll for task status
      400:
        description: Invalid request
      404:
        description: Campaign not found
    """
    from gm_agent.tasks import process_player_chat_async, process_npc_turn_async

    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "type" not in data:
        return jsonify({"error": "Request must include 'type' field"}), 400

    turn_type = data["type"]

    if turn_type == "player_chat":
        if "player_input" not in data:
            return jsonify({"error": "player_input required for player_chat type"}), 400

        task = process_player_chat_async.apply_async(
            args=[campaign_id, data["player_input"]],
            kwargs={
                "session_id": data.get("session_id"),
                "metadata": data.get("metadata"),
            },
        )

    elif turn_type == "npc_turn":
        if "npc_name" not in data:
            return jsonify({"error": "npc_name required for npc_turn type"}), 400

        task = process_npc_turn_async.apply_async(
            args=[campaign_id, data["npc_name"]],
            kwargs={
                "prompt": data.get("prompt"),
                "metadata": data.get("metadata"),
            },
        )

    else:
        return jsonify({"error": f"Invalid type: {turn_type}"}), 400

    return (
        jsonify(
            {
                "task_id": task.id,
                "campaign_id": campaign_id,
                "status_url": f"/api/campaigns/{campaign_id}/automation/tasks/{task.id}",
                "message": "Task queued for async processing",
            }
        ),
        202,
    )


@app.route("/api/campaigns/<campaign_id>/automation/tasks/<task_id>", methods=["GET"])
@optional_jwt_required
def get_task_status(campaign_id: str, task_id: str):
    """
    Get status of an async automation task
    ---
    tags:
      - Automation
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: task_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Task status
        schema:
          type: object
          properties:
            task_id:
              type: string
            campaign_id:
              type: string
            state:
              type: string
              enum: [PENDING, PROCESSING, SUCCESS, FAILURE]
            result:
              type: object
              description: Task result (present if SUCCESS)
            error:
              type: string
              description: Error message (present if FAILURE)
            timestamp:
              type: number
              description: Unix timestamp of last update
      404:
        description: Task not found
    """
    from gm_agent.tasks import get_task_progress
    from celery.result import AsyncResult

    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    # Check Celery task state
    task_result = AsyncResult(task_id)

    # Get our custom progress info from Redis
    progress = get_task_progress(task_id)

    if progress:
        # We have custom progress tracking
        response = {
            "task_id": task_id,
            "campaign_id": campaign_id,
            "state": progress["state"],
            "timestamp": progress["timestamp"],
        }

        # Add result or error if present
        if progress["state"] == "SUCCESS" and "response" in progress:
            response["result"] = {
                "response": progress.get("response"),
                "turn_number": progress.get("turn_number"),
            }
        elif progress["state"] == "FAILURE" and "error" in progress:
            response["error"] = progress["error"]

        return jsonify(response)

    # Fall back to Celery state if no custom progress
    return jsonify(
        {
            "task_id": task_id,
            "campaign_id": campaign_id,
            "state": task_result.state,
            "info": str(task_result.info) if task_result.info else None,
        }
    )


@app.route("/api/campaigns/<campaign_id>/analytics/tools", methods=["GET"])
@optional_jwt_required
def get_tool_analytics(campaign_id: str):
    """
    Get tool usage analytics for a campaign
    ---
    tags:
      - Analytics
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: session_id
        in: query
        type: string
        required: false
        description: Optional session ID to filter analytics
      - name: limit
        in: query
        type: integer
        required: false
        description: Limit number of sessions to analyze (default 10, 0 for all)
    responses:
      200:
        description: Tool usage analytics
        schema:
          type: object
          properties:
            campaign_id:
              type: string
            session_count:
              type: integer
              description: Number of sessions analyzed
            turn_count:
              type: integer
              description: Total number of turns analyzed
            tool_usage:
              type: object
              description: Map of tool name to usage count
            tool_failures:
              type: object
              description: Map of tool name to failure count
            failure_rate:
              type: object
              description: Map of tool name to failure rate (0.0-1.0)
            most_used_tools:
              type: array
              description: Top 10 most used tools
              items:
                type: object
                properties:
                  name:
                    type: string
                  count:
                    type: integer
            most_failed_tools:
              type: array
              description: Top 10 most failed tools
              items:
                type: object
                properties:
                  name:
                    type: string
                  count:
                    type: integer
                  rate:
                    type: number
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    # Get query parameters
    session_id = request.args.get("session_id")
    limit = int(request.args.get("limit", 10))

    # Aggregate tool usage across sessions
    tool_usage: dict[str, int] = {}
    tool_failures: dict[str, int] = {}
    turn_count = 0

    if session_id:
        # Analyze specific session
        session = session_store.get(campaign_id, session_id)
        if not session:
            return jsonify({"error": f"Session '{session_id}' not found"}), 404
        sessions = [session]
    else:
        # Get recent sessions
        sessions = session_store.list(campaign_id)
        if limit > 0:
            sessions = sessions[-limit:]

    # Aggregate stats from sessions
    for session in sessions:
        for turn in session.turns:
            turn_count += 1
            if turn.metadata:
                # Aggregate tool usage
                for tool_name, count in turn.metadata.tool_usage.items():
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + count

                # Aggregate tool failures
                for tool_name in turn.metadata.tool_failures:
                    tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1

    # Calculate failure rates
    failure_rate = {}
    for tool_name in tool_failures:
        usage = tool_usage.get(tool_name, 0)
        if usage > 0:
            failure_rate[tool_name] = tool_failures[tool_name] / usage

    # Sort tools by usage and failures
    most_used_tools = [
        {"name": name, "count": count} for name, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    most_failed_tools = [
        {"name": name, "count": count, "rate": failure_rate.get(name, 0.0)}
        for name, count in sorted(tool_failures.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    return jsonify(
        {
            "campaign_id": campaign_id,
            "session_count": len(sessions),
            "turn_count": turn_count,
            "tool_usage": tool_usage,
            "tool_failures": tool_failures,
            "failure_rate": failure_rate,
            "most_used_tools": most_used_tools,
            "most_failed_tools": most_failed_tools,
        }
    )


@app.route("/api/campaigns/<campaign_id>/sessions/<session_id>", methods=["GET"])
@optional_jwt_required
def get_session(campaign_id: str, session_id: str):
    """
    Get a specific session
    ---
    tags:
      - Sessions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: session_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Session details
      404:
        description: Campaign or session not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    session = session_store.get(campaign_id, session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404

    return jsonify(
        {
            "id": session.id,
            "campaign_id": session.campaign_id,
            "turns": [
                {
                    "player_input": t.player_input,
                    "gm_response": t.gm_response,
                    "tool_calls": (
                        [tc.model_dump() for tc in t.tool_calls] if t.tool_calls else []
                    ),
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in session.turns
            ],
            "summary": session.summary,
            "scene_state": session.scene_state.model_dump(),
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        }
    )


# =============================================================================
# Character Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/characters", methods=["GET"])
@optional_jwt_required
def list_characters(campaign_id: str):
    """
    List all character profiles
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: type
        in: query
        type: string
        required: false
        description: Filter by character type (npc, monster, player)
    responses:
      200:
        description: List of characters
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    character_type = request.args.get("type")
    store = get_character_store(campaign_id)
    characters = store.list(character_type)

    return jsonify(
        {
            "characters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "character_type": c.character_type,
                    "personality": (
                        c.personality[:100] + "..." if len(c.personality) > 100 else c.personality
                    ),
                }
                for c in characters
            ]
        }
    )


@app.route("/api/campaigns/<campaign_id>/characters", methods=["POST"])
@optional_jwt_required
def create_character(campaign_id: str):
    """
    Create a new character profile
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
          properties:
            name:
              type: string
              example: Sheriff Hemlock
            character_type:
              type: string
              enum: [npc, monster, player]
              default: npc
            personality:
              type: string
            knowledge:
              type: array
              items:
                type: string
            goals:
              type: array
              items:
                type: string
            secrets:
              type: array
              items:
                type: string
            intelligence:
              type: string
              enum: [animal, low, average, high, genius]
            instincts:
              type: array
              items:
                type: string
            morale:
              type: string
    responses:
      201:
        description: Character created
      400:
        description: Invalid request or duplicate
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "Character name is required"}), 400

    store = get_character_store(campaign_id)

    # Check for duplicate
    existing = store.get_by_name(data["name"])
    if existing:
        return jsonify({"error": f"Character '{data['name']}' already exists"}), 400

    # Parse list fields from comma-separated strings
    def parse_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    character = store.create(
        name=data["name"],
        character_type=data.get("character_type", "npc"),
        personality=data.get("personality", ""),
        speech_patterns=data.get("speech_patterns", ""),
        knowledge=parse_list(data.get("knowledge", [])),
        goals=parse_list(data.get("goals", [])),
        secrets=parse_list(data.get("secrets", [])),
        intelligence=data.get("intelligence", "average"),
        instincts=parse_list(data.get("instincts", [])),
        morale=data.get("morale", ""),
        playstyle=data.get("playstyle", ""),
        risk_tolerance=data.get("risk_tolerance", "medium"),
        party_role=data.get("party_role", ""),
        quirks=parse_list(data.get("quirks", [])),
    )

    return (
        jsonify(
            {
                "id": character.id,
                "name": character.name,
                "character_type": character.character_type,
                "message": f"Character '{character.name}' created",
            }
        ),
        201,
    )


@app.route("/api/campaigns/<campaign_id>/characters/<name>", methods=["GET"])
@optional_jwt_required
def get_character(campaign_id: str, name: str):
    """
    Get a character profile by name
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Character details
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    store = get_character_store(campaign_id)
    character = store.get_by_name(name)

    if not character:
        return jsonify({"error": f"Character '{name}' not found"}), 404

    return jsonify(character.model_dump(mode="json"))


@app.route("/api/campaigns/<campaign_id>/characters/<name>", methods=["DELETE"])
@optional_jwt_required
def delete_character(campaign_id: str, name: str):
    """
    Delete a character profile
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Character deleted
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    store = get_character_store(campaign_id)
    character = store.get_by_name(name)

    if not character:
        return jsonify({"error": f"Character '{name}' not found"}), 404

    store.delete(character.id)
    return jsonify({"message": f"Character '{name}' deleted"})


@app.route("/api/campaigns/<campaign_id>/characters/build", methods=["POST"])
@optional_jwt_required
def build_character_profile(campaign_id: str):
    """
    Build a character profile automatically from RAG data
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
          properties:
            name:
              type: string
              description: Character name to build profile for
            force_rebuild:
              type: boolean
              description: Rebuild even if profile already exists (default false)
    responses:
      200:
        description: Character profile built successfully
        schema:
          type: object
          properties:
            message:
              type: string
            character_id:
              type: string
            character_name:
              type: string
            character_type:
              type: string
      400:
        description: Missing required fields or LLM unavailable
      404:
        description: Campaign not found or no RAG data found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "name field is required"}), 400

    name = data.get("name")
    force_rebuild = data.get("force_rebuild", False)

    # Get agent to access MCP client with npc-builder
    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    # Call build_npc_profile tool
    try:
        result = agent._mcp.call_tool(
            "build_npc_profile",
            {"name": name, "force_rebuild": force_rebuild}
        )

        if not result.success:
            if "No information found" in result.error:
                return jsonify({"error": result.error}), 404
            elif "LLM backend required" in result.error:
                return jsonify({"error": result.error}), 400
            else:
                return jsonify({"error": result.error}), 500

        # Parse the result to extract character ID
        # Result format: "Character profile created for 'Name' (ID: abc123)\n..."
        import re
        match = re.search(r"ID: ([a-f0-9]+)", result.data)
        character_id = match.group(1) if match else None

        # Get the created/updated character
        store = get_character_store(campaign_id)
        character = store.get_by_name(name)

        return jsonify(
            {
                "message": result.data.split("\n")[0],  # First line
                "character_id": character.id if character else character_id,
                "character_name": name,
                "character_type": character.character_type if character else "unknown",
            }
        )

    except Exception as e:
        logger.exception(f"Error building character profile: {e}")
        return jsonify({"error": f"Failed to build character profile: {e}"}), 500


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/relationships", methods=["GET"])
@optional_jwt_required
def get_character_relationships(campaign_id: str, character_name: str):
    """
    Get all relationships for a character
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: List of relationships
        schema:
          type: object
          properties:
            character_name:
              type: string
            relationships:
              type: array
              items:
                type: object
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    store = get_character_store(campaign_id)
    character = store.get_by_name(character_name)

    if not character:
        return jsonify({"error": f"Character '{character_name}' not found"}), 404

    # Convert relationships to dict format
    relationships = [
        {
            "target_character_id": r.target_character_id,
            "target_name": r.target_name,
            "relationship_type": r.relationship_type,
            "attitude": r.attitude,
            "trust_level": r.trust_level,
            "history": r.history,
            "notes": r.notes,
        }
        for r in character.relationships
    ]

    return jsonify(
        {
            "character_name": character.name,
            "character_id": character.id,
            "relationships": relationships,
        }
    )


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/relationships", methods=["POST"])
@optional_jwt_required
def add_character_relationship(campaign_id: str, character_name: str):
    """
    Add or update a relationship for a character
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - target_name
          properties:
            target_name:
              type: string
            relationship_type:
              type: string
            attitude:
              type: string
            trust_level:
              type: integer
            history:
              type: string
            notes:
              type: string
    responses:
      200:
        description: Relationship added/updated
      400:
        description: Missing required fields
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "target_name" not in data:
        return jsonify({"error": "target_name is required"}), 400

    # Use agent to access MCP for relationship management
    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    # Call add_relationship tool
    args = {
        "character_name": character_name,
        "target_name": data.get("target_name"),
        "relationship_type": data.get("relationship_type", "acquaintance"),
        "attitude": data.get("attitude", "neutral"),
        "trust_level": data.get("trust_level", 0),
        "history": data.get("history", ""),
        "notes": data.get("notes", ""),
    }

    result = agent._mcp.call_tool("add_relationship", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/relationships/<target_name>", methods=["DELETE"])
@optional_jwt_required
def delete_character_relationship(campaign_id: str, character_name: str, target_name: str):
    """
    Remove a relationship
    ---
    tags:
      - Characters
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
      - name: target_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Relationship removed
      404:
        description: Campaign, character, or relationship not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool(
        "remove_relationship",
        {"character_name": character_name, "target_name": target_name}
    )

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"message": result.data})


# =============================================================================
# Dialogue History Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/dialogue", methods=["GET"])
@optional_jwt_required
def search_campaign_dialogue(campaign_id: str):
    """
    Search dialogue history for a campaign
    ---
    tags:
      - Dialogue
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: query
        in: query
        type: string
        description: Search query (searches character names and dialogue content)
      - name: character_name
        in: query
        type: string
        description: Filter by character name
      - name: dialogue_type
        in: query
        type: string
        description: Filter by dialogue type (statement, promise, threat, lie, rumor, secret)
      - name: flagged_only
        in: query
        type: boolean
        description: Only return flagged dialogue
      - name: limit
        in: query
        type: integer
        description: Maximum number of results (default 20)
    responses:
      200:
        description: List of dialogue entries
        schema:
          type: object
          properties:
            dialogue:
              type: array
              items:
                type: object
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    # Get query parameters
    query = request.args.get("query", "")
    character_name = request.args.get("character_name")
    dialogue_type = request.args.get("dialogue_type")
    flagged_only = request.args.get("flagged_only", "false").lower() == "true"
    limit = int(request.args.get("limit", 20))

    # Call search_dialogue tool
    args = {
        "query": query,
        "character_name": character_name,
        "dialogue_type": dialogue_type,
        "flagged_only": flagged_only,
        "limit": limit,
    }

    result = agent._mcp.call_tool("search_dialogue", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"result": result.data})


@app.route("/api/campaigns/<campaign_id>/dialogue/<int:dialogue_id>/flag", methods=["PATCH"])
@optional_jwt_required
def flag_dialogue_entry(campaign_id: str, dialogue_id: int):
    """
    Flag or unflag a dialogue entry
    ---
    tags:
      - Dialogue
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: dialogue_id
        in: path
        type: integer
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            flagged:
              type: boolean
              description: True to flag, False to unflag
    responses:
      200:
        description: Dialogue flagged/unflagged
      400:
        description: Invalid request
      404:
        description: Campaign or dialogue not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    flagged = data.get("flagged", True) if data else True

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool(
        "flag_dialogue",
        {"dialogue_id": dialogue_id, "flagged": flagged}
    )

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"message": result.data})


# =============================================================================
# NPC Knowledge Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/knowledge", methods=["GET"])
@optional_jwt_required
def get_character_knowledge(campaign_id: str, character_name: str):
    """
    Get knowledge for a character
    ---
    tags:
      - Knowledge
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
      - name: knowledge_type
        in: query
        type: string
        description: Filter by knowledge type
      - name: min_importance
        in: query
        type: integer
        description: Minimum importance (1-10)
      - name: tags
        in: query
        type: string
        description: Comma-separated tags to filter by
      - name: limit
        in: query
        type: integer
        description: Maximum number of results (default 20)
    responses:
      200:
        description: Knowledge entries
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    # Get query parameters
    knowledge_type = request.args.get("knowledge_type")
    min_importance = request.args.get("min_importance", type=int)
    tags = request.args.get("tags")
    limit = int(request.args.get("limit", 20))

    # Call query_npc_knowledge tool
    args = {
        "character_name": character_name,
        "knowledge_type": knowledge_type,
        "min_importance": min_importance,
        "tags": tags,
        "limit": limit,
    }

    result = agent._mcp.call_tool("query_npc_knowledge", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"result": result.data})


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/knowledge", methods=["POST"])
@optional_jwt_required
def add_character_knowledge(campaign_id: str, character_name: str):
    """
    Add knowledge to a character
    ---
    tags:
      - Knowledge
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - content
          properties:
            content:
              type: string
              description: The knowledge content
            knowledge_type:
              type: string
              description: Type (fact, rumor, secret, etc.)
            sharing_condition:
              type: string
              description: Sharing condition (free, trust, persuasion_dc_X, never)
            source:
              type: string
              description: Source of the knowledge
            importance:
              type: integer
              description: Importance 1-10
            tags:
              type: string
              description: Comma-separated tags
    responses:
      200:
        description: Knowledge added
      400:
        description: Missing required fields
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "content" not in data:
        return jsonify({"error": "content is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    # Call add_npc_knowledge tool
    args = {
        "character_name": character_name,
        "content": data.get("content"),
        "knowledge_type": data.get("knowledge_type", "fact"),
        "sharing_condition": data.get("sharing_condition", "free"),
        "source": data.get("source", ""),
        "importance": data.get("importance", 5),
        "tags": data.get("tags", ""),
    }

    result = agent._mcp.call_tool("add_npc_knowledge", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/knowledge/shareable", methods=["GET"])
@optional_jwt_required
def get_shareable_knowledge(campaign_id: str, character_name: str):
    """
    Get knowledge that an NPC will share given conditions
    ---
    tags:
      - Knowledge
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
      - name: trust_level
        in: query
        type: integer
        description: Trust level (-5 to +5)
      - name: persuasion_dc_met
        in: query
        type: integer
        description: Highest persuasion DC met
      - name: under_duress
        in: query
        type: boolean
        description: Is character under duress?
      - name: knowledge_type
        in: query
        type: string
        description: Filter by knowledge type
      - name: limit
        in: query
        type: integer
        description: Maximum results (default 20)
    responses:
      200:
        description: Shareable knowledge
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    # Get query parameters
    trust_level = int(request.args.get("trust_level", 0))
    persuasion_dc_met = int(request.args.get("persuasion_dc_met", 0))
    under_duress = request.args.get("under_duress", "false").lower() == "true"
    knowledge_type = request.args.get("knowledge_type")
    limit = int(request.args.get("limit", 20))

    # Call what_will_npc_share tool
    args = {
        "character_name": character_name,
        "trust_level": trust_level,
        "persuasion_dc_met": persuasion_dc_met,
        "under_duress": under_duress,
        "knowledge_type": knowledge_type,
        "limit": limit,
    }

    result = agent._mcp.call_tool("what_will_npc_share", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"result": result.data})


# =============================================================================
# Faction Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/factions", methods=["GET"])
@optional_jwt_required
def list_factions(campaign_id: str):
    """
    List all factions in a campaign
    ---
    tags:
      - Factions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: List of factions
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("list_factions", {})

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"result": result.data})


@app.route("/api/campaigns/<campaign_id>/factions", methods=["POST"])
@optional_jwt_required
def create_faction(campaign_id: str):
    """
    Create a new faction
    ---
    tags:
      - Factions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
          properties:
            name:
              type: string
              description: Faction name
            description:
              type: string
              description: Faction description
            goals:
              type: string
              description: Comma-separated goals
            resources:
              type: string
              description: Comma-separated resources
    responses:
      200:
        description: Faction created
      400:
        description: Missing required fields
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "name is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "name": data.get("name"),
        "description": data.get("description", ""),
        "goals": data.get("goals", ""),
        "resources": data.get("resources", ""),
    }

    result = agent._mcp.call_tool("create_faction", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/factions/<faction_name>", methods=["GET"])
@optional_jwt_required
def get_faction(campaign_id: str, faction_name: str):
    """
    Get faction information
    ---
    tags:
      - Factions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: faction_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Faction information
      404:
        description: Campaign or faction not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("get_faction_info", {"faction_name": faction_name})

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"result": result.data})


@app.route("/api/campaigns/<campaign_id>/factions/<faction_name>/members", methods=["GET"])
@optional_jwt_required
def get_faction_members(campaign_id: str, faction_name: str):
    """
    Get faction members
    ---
    tags:
      - Factions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: faction_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Faction members
      404:
        description: Campaign or faction not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("get_faction_members", {"faction_name": faction_name})

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"result": result.data})


@app.route("/api/campaigns/<campaign_id>/factions/<faction_name>/members", methods=["POST"])
@optional_jwt_required
def add_faction_member(campaign_id: str, faction_name: str):
    """
    Add a member to a faction
    ---
    tags:
      - Factions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: faction_name
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - character_name
          properties:
            character_name:
              type: string
              description: Character name to add
    responses:
      200:
        description: Member added
      400:
        description: Missing required fields
      404:
        description: Campaign, faction, or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "character_name" not in data:
        return jsonify({"error": "character_name is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "character_name": data.get("character_name"),
        "faction_name": faction_name,
    }

    result = agent._mcp.call_tool("add_npc_to_faction", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/factions/<faction_name>/reputation", methods=["PATCH"])
@optional_jwt_required
def update_faction_reputation(campaign_id: str, faction_name: str):
    """
    Update faction reputation with the party
    ---
    tags:
      - Factions
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: faction_name
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - reputation
          properties:
            reputation:
              type: integer
              description: New reputation value (-100 to +100)
    responses:
      200:
        description: Reputation updated
      400:
        description: Missing required fields
      404:
        description: Campaign or faction not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "reputation" not in data:
        return jsonify({"error": "reputation is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "faction_name": faction_name,
        "reputation": data.get("reputation"),
    }

    result = agent._mcp.call_tool("update_faction_reputation", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


# =============================================================================
# Location Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/locations", methods=["GET"])
@optional_jwt_required
def list_locations(campaign_id: str):
    """
    List all locations in a campaign
    ---
    tags:
      - Locations
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: List of locations
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("list_locations", {})

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"data": result.data})


@app.route("/api/campaigns/<campaign_id>/locations", methods=["POST"])
@optional_jwt_required
def create_location(campaign_id: str):
    """
    Create a new location
    ---
    tags:
      - Locations
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
          properties:
            name:
              type: string
            description:
              type: string
            isolation_level:
              type: string
              enum: [connected, remote, isolated]
    responses:
      200:
        description: Location created successfully
      400:
        description: Invalid request
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "name is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "name": data.get("name"),
        "description": data.get("description", ""),
        "isolation_level": data.get("isolation_level", "connected"),
    }

    result = agent._mcp.call_tool("create_location", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/locations/<location_name>", methods=["GET"])
@optional_jwt_required
def get_location(campaign_id: str, location_name: str):
    """
    Get location information
    ---
    tags:
      - Locations
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: location_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Location information
      404:
        description: Campaign or location not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("get_location_info", {"location_name": location_name})

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"data": result.data})


@app.route("/api/campaigns/<campaign_id>/locations/<location_name>/knowledge", methods=["POST"])
@optional_jwt_required
def add_location_knowledge(campaign_id: str, location_name: str):
    """
    Add common knowledge to a location
    ---
    tags:
      - Locations
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: location_name
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - knowledge_id
          properties:
            knowledge_id:
              type: string
    responses:
      200:
        description: Knowledge added successfully
      400:
        description: Invalid request
      404:
        description: Campaign or location not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "knowledge_id" not in data:
        return jsonify({"error": "knowledge_id is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "location_name": location_name,
        "knowledge_id": data.get("knowledge_id"),
    }

    result = agent._mcp.call_tool("add_location_knowledge", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/locations/<location_name>/events", methods=["POST"])
@optional_jwt_required
def add_location_event(campaign_id: str, location_name: str):
    """
    Add a recent event to a location
    ---
    tags:
      - Locations
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: location_name
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - event
          properties:
            event:
              type: string
    responses:
      200:
        description: Event added successfully
      400:
        description: Invalid request
      404:
        description: Campaign or location not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "event" not in data:
        return jsonify({"error": "event is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "location_name": location_name,
        "event": data.get("event"),
    }

    result = agent._mcp.call_tool("add_location_event", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/locations/connections", methods=["POST"])
@optional_jwt_required
def connect_locations(campaign_id: str):
    """
    Connect two locations
    ---
    tags:
      - Locations
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - location1
            - location2
          properties:
            location1:
              type: string
            location2:
              type: string
    responses:
      200:
        description: Locations connected successfully
      400:
        description: Invalid request
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "location1" not in data or "location2" not in data:
        return jsonify({"error": "location1 and location2 are required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "location1": data.get("location1"),
        "location2": data.get("location2"),
    }

    result = agent._mcp.call_tool("connect_locations", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


# =============================================================================
# Rumor Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/rumors", methods=["GET"])
@optional_jwt_required
def list_all_rumors(campaign_id: str):
    """
    List all rumors in the campaign
    ---
    tags:
      - Rumors
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: List of rumors
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("list_all_rumors", {})

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"data": result.data})


@app.route("/api/campaigns/<campaign_id>/rumors", methods=["POST"])
@optional_jwt_required
def seed_rumor(campaign_id: str):
    """
    Seed a new rumor
    ---
    tags:
      - Rumors
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - content
          properties:
            content:
              type: string
            starting_locations:
              type: string
              description: Comma-separated location names
            spread_rate:
              type: string
              enum: [slow, medium, fast]
            source_type:
              type: string
              enum: [pc_seeded, event, npc_created]
    responses:
      200:
        description: Rumor seeded successfully
      400:
        description: Invalid request
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "content" not in data:
        return jsonify({"error": "content is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "content": data.get("content"),
        "starting_locations": data.get("starting_locations", ""),
        "spread_rate": data.get("spread_rate", "medium"),
        "source_type": data.get("source_type", "pc_seeded"),
    }

    result = agent._mcp.call_tool("seed_rumor", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/rumors/propagate", methods=["POST"])
@optional_jwt_required
def propagate_rumors(campaign_id: str):
    """
    Propagate all rumors based on time passed
    ---
    tags:
      - Rumors
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: false
        schema:
          type: object
          properties:
            days:
              type: integer
              description: Number of days to simulate (default 1)
    responses:
      200:
        description: Rumors propagated successfully
      400:
        description: Invalid request
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json() or {}
    days = data.get("days", 1)

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("propagate_rumors", {"days": days})

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/locations/<location_name>/rumors", methods=["GET"])
@optional_jwt_required
def get_location_rumors(campaign_id: str, location_name: str):
    """
    Get all rumors at a location
    ---
    tags:
      - Rumors
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: location_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Rumors at location
      404:
        description: Campaign or location not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("get_rumors_at_location", {"location_name": location_name})

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"data": result.data})


@app.route("/api/campaigns/<campaign_id>/characters/<character_name>/rumors", methods=["GET"])
@optional_jwt_required
def get_character_rumors_endpoint(campaign_id: str, character_name: str):
    """
    Get all rumors known by a character
    ---
    tags:
      - Rumors
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: character_name
        in: path
        type: string
        required: true
    responses:
      200:
        description: Rumors known by character
      404:
        description: Campaign or character not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("get_character_rumors", {"character_name": character_name})

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"data": result.data})


# =============================================================================
# Secret Endpoints
# =============================================================================


@app.route("/api/campaigns/<campaign_id>/secrets", methods=["GET"])
@optional_jwt_required
def list_secrets(campaign_id: str):
    """
    List all secrets in the campaign
    ---
    tags:
      - Secrets
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: revealed
        in: query
        type: string
        enum: [true, false, all]
        description: Filter by revelation status
    responses:
      200:
        description: List of secrets
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    revealed = request.args.get("revealed", "all")

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("list_secrets", {"revealed": revealed})

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"data": result.data})


@app.route("/api/campaigns/<campaign_id>/secrets", methods=["POST"])
@optional_jwt_required
def create_secret(campaign_id: str):
    """
    Create a new secret
    ---
    tags:
      - Secrets
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - content
          properties:
            content:
              type: string
            importance:
              type: string
              enum: [minor, major, critical]
            consequences:
              type: string
              description: Comma-separated consequences
    responses:
      200:
        description: Secret created successfully
      400:
        description: Invalid request
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json()
    if not data or "content" not in data:
        return jsonify({"error": "content is required"}), 400

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "content": data.get("content"),
        "importance": data.get("importance", "major"),
        "consequences": data.get("consequences", ""),
    }

    result = agent._mcp.call_tool("create_secret", args)

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/secrets/<secret_id>/reveal", methods=["POST"])
@optional_jwt_required
def reveal_secret(campaign_id: str, secret_id: str):
    """
    Reveal a secret to the party
    ---
    tags:
      - Secrets
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
      - name: secret_id
        in: path
        type: string
        required: true
      - in: body
        name: body
        required: false
        schema:
          type: object
          properties:
            revealer:
              type: string
            method:
              type: string
    responses:
      200:
        description: Secret revealed successfully
      400:
        description: Invalid request
      404:
        description: Campaign or secret not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    data = request.get_json() or {}

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    args = {
        "secret_id": secret_id,
        "revealer": data.get("revealer", ""),
        "method": data.get("method", ""),
    }

    result = agent._mcp.call_tool("reveal_secret", args)

    if not result.success:
        return jsonify({"error": result.error}), 404

    return jsonify({"message": result.data})


@app.route("/api/campaigns/<campaign_id>/secrets/revelations", methods=["GET"])
@optional_jwt_required
def get_revelation_history(campaign_id: str):
    """
    Get revelation history
    ---
    tags:
      - Secrets
    parameters:
      - name: campaign_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Revelation history
      404:
        description: Campaign not found
    """
    campaign = campaign_store.get(campaign_id)
    if not campaign:
        return jsonify({"error": f"Campaign '{campaign_id}' not found"}), 404

    try:
        agent = get_agent(campaign_id)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize agent: {e}"}), 500

    result = agent._mcp.call_tool("get_revelation_history", {})

    if not result.success:
        return jsonify({"error": result.error}), 400

    return jsonify({"data": result.data})


# =============================================================================
# Stateless Chat Endpoint (for Foundry VTT)
# =============================================================================


@app.route("/api/chat", methods=["POST"])
@optional_jwt_required
def chat():
    """
    Stateless chat with GM assistant
    ---
    tags:
      - Chat
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              example: What is flanking in Pathfinder 2e?
            context:
              type: object
              description: Optional context (combat state, token info, etc.)
              properties:
                combat:
                  type: object
                  description: Current combat state from Foundry
                tokens:
                  type: array
                  description: Selected token information
    responses:
      200:
        description: Assistant response
        schema:
          type: object
          properties:
            response:
              type: string
              example: "Flanking in Pathfinder 2e occurs when..."
      400:
        description: Message required
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Message required"}), 400

    message = data["message"]
    context = data.get("context", {})

    # If context provided, prepend it to the message
    if context:
        context_str = _format_context(context)
        if context_str:
            message = f"{context_str}\n\nQuestion: {message}"

    agent = ChatAgent()
    try:
        response = agent.chat(message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        agent.close()


def _format_context(context: dict) -> str:
    """Format Foundry context into a string for the agent."""
    parts = []

    # Combat context
    combat = context.get("combat")
    if combat:
        parts.append("Current Combat State:")
        if combat.get("round"):
            parts.append(f"  Round: {combat['round']}")
        if combat.get("combatants"):
            parts.append("  Combatants:")
            for c in combat["combatants"]:
                name = c.get("name", "Unknown")
                hp = c.get("hp", {})
                hp_str = f" (HP: {hp.get('value', '?')}/{hp.get('max', '?')})" if hp else ""
                conditions = c.get("conditions", [])
                cond_str = f" [{', '.join(conditions)}]" if conditions else ""
                parts.append(f"    - {name}{hp_str}{cond_str}")

    # Token context
    tokens = context.get("tokens")
    if tokens:
        parts.append("Selected Tokens:")
        for token in tokens:
            name = token.get("name", "Unknown")
            actor_type = token.get("type", "")
            level = token.get("level")
            level_str = f" (Level {level})" if level else ""
            parts.append(f"  - {name}{level_str} [{actor_type}]")

    return "\n".join(parts)


# =============================================================================
# MCP Endpoints (Model Context Protocol)
# =============================================================================


@app.route("/api/mcp/servers", methods=["GET"])
@optional_jwt_required
def list_mcp_servers():
    """
    List available MCP servers
    ---
    tags:
      - MCP
    responses:
      200:
        description: List of MCP servers with metadata
        schema:
          type: object
          properties:
            servers:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                    example: pf2e-rag
                  stateless:
                    type: boolean
                  celery_eligible:
                    type: boolean
                  requires_campaign:
                    type: boolean
                  requires_websocket:
                    type: boolean
    """
    from gm_agent.mcp.registry import SERVERS

    servers = []
    for name, info in SERVERS.items():
        servers.append(
            {
                "name": name,
                "stateless": info.stateless,
                "celery_eligible": info.celery_eligible,
                "requires_campaign": info.requires_campaign,
                "requires_llm": info.requires_llm,
                "requires_websocket": info.requires_websocket,
            }
        )

    return jsonify({"servers": servers})


@app.route("/api/mcp/tools", methods=["GET"])
@optional_jwt_required
def list_mcp_tools():
    """
    List all available MCP tools
    ---
    tags:
      - MCP
    parameters:
      - name: campaign_id
        in: query
        type: string
        required: false
        description: Campaign ID to include campaign-specific tools
    responses:
      200:
        description: List of tools with metadata
        schema:
          type: object
          properties:
            tools:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  description:
                    type: string
                  server:
                    type: string
                  parameters:
                    type: array
    """
    from gm_agent.mcp.client import MCPClient
    from gm_agent.mcp.registry import TOOL_TO_SERVER

    campaign_id = request.args.get("campaign_id")

    # Create client with or without campaign context
    context = {"campaign_id": campaign_id} if campaign_id else {}
    client = MCPClient(mode="local", context=context, foundry_server=foundry_server)

    try:
        tools = client.list_tools()
        tool_list = []
        for tool in tools:
            tool_data = {
                "name": tool.name,
                "description": tool.description,
                "server": TOOL_TO_SERVER.get(tool.name, "unknown"),
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                    }
                    for p in tool.parameters
                ],
            }
            if tool.category:
                tool_data["category"] = tool.category
            tool_list.append(tool_data)

        return jsonify({"tools": tool_list})
    finally:
        client.close()


@app.route("/api/mcp/call", methods=["POST"])
@optional_jwt_required
def call_mcp_tool():
    """
    Execute an MCP tool
    ---
    tags:
      - MCP
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - tool
          properties:
            tool:
              type: string
              description: Name of the tool to call
              example: lookup_creature
            args:
              type: object
              description: Tool arguments
              example: {"name": "goblin"}
            context:
              type: object
              description: Execution context (campaign_id, session_id)
              example: {"campaign_id": "my-campaign"}
            async:
              type: boolean
              description: If true, return task_id for async execution
              default: false
    responses:
      200:
        description: Tool execution result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
            error:
              type: string
      202:
        description: Task queued (async mode)
        schema:
          type: object
          properties:
            task_id:
              type: string
      400:
        description: Missing required field
    """
    from gm_agent.mcp.client import MCPClient
    from gm_agent.mcp.registry import get_server_for_tool, get_server_info

    data = request.get_json()
    if not data or "tool" not in data:
        return jsonify({"error": "Tool name is required"}), 400

    tool_name = data["tool"]
    args = data.get("args", {})
    context = data.get("context", {})
    async_mode = data.get("async", False)

    # Get server info for this tool
    server_name = get_server_for_tool(tool_name)
    if not server_name:
        return jsonify({"success": False, "error": f"Unknown tool: {tool_name}"}), 400

    server_info = get_server_info(server_name)

    # Foundry tools must execute in-process (require WebSocket)
    if server_info and server_info.requires_websocket:
        if not foundry_server:
            return (
                jsonify({"success": False, "error": "Foundry VTT is not connected"}),
                400,
            )

        # Execute directly with Foundry server
        result = foundry_server.call_tool(tool_name, args)
        return jsonify(
            {
                "success": result.success,
                "data": result.data,
                "error": result.error,
            }
        )

    # Check if async execution requested and tool is Celery-eligible
    if async_mode and server_info and server_info.celery_eligible:
        try:
            from gm_agent.mcp_tasks import execute_mcp_tool

            task = execute_mcp_tool.delay(server_name, tool_name, args, context)
            return jsonify({"task_id": task.id}), 202
        except ImportError:
            # Celery not available, fall back to sync
            pass

    # Execute synchronously
    client = MCPClient(mode="local", context=context, foundry_server=foundry_server)
    try:
        result = client.call_tool(tool_name, args)
        return jsonify(
            {
                "success": result.success,
                "data": result.data,
                "error": result.error,
            }
        )
    finally:
        client.close()


@app.route("/api/mcp/task/<task_id>", methods=["GET"])
@optional_jwt_required
def get_mcp_task_status(task_id: str):
    """
    Check status of an async MCP task
    ---
    tags:
      - MCP
    parameters:
      - name: task_id
        in: path
        type: string
        required: true
        description: Celery task ID
    responses:
      200:
        description: Task status
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [pending, started, success, failure]
            result:
              type: object
              description: Tool result (if completed)
            error:
              type: string
              description: Error message (if failed)
      404:
        description: Task not found
    """
    try:
        from gm_agent.mcp_tasks import celery_app

        task = celery_app.AsyncResult(task_id)

        response = {
            "status": task.status.lower(),
        }

        if task.ready():
            if task.successful():
                response["result"] = task.result
            else:
                response["error"] = str(task.result) if task.result else "Unknown error"

        return jsonify(response)
    except ImportError:
        return jsonify({"error": "Celery not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Health Check
# =============================================================================


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Service status
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            service:
              type: string
              example: gm-agent
            version:
              type: string
              example: 0.1.0
            auth_enabled:
              type: boolean
            foundry_connected:
              type: boolean
            automation:
              type: object
              properties:
                active_loops:
                  type: integer
                campaigns:
                  type: array
                  items:
                    type: string
    """
    # Get automation status
    active_loops = [campaign_id for campaign_id, loop in _game_loops.items() if loop.enabled]

    return jsonify(
        {
            "status": "ok",
            "service": "gm-agent",
            "version": "0.1.0",
            "auth_enabled": AUTH_ENABLED,
            "foundry_connected": foundry_bridge is not None and foundry_bridge.is_connected(),
            "automation": {
                "active_loops": len(active_loops),
                "campaigns": active_loops,
            },
        }
    )


@app.route("/api/health/foundry", methods=["GET"])
def health_foundry():
    """
    Check Foundry VTT bridge connection status
    ---
    tags:
      - System
    responses:
      200:
        description: Foundry bridge status
        schema:
          type: object
          properties:
            connected:
              type: boolean
            session_id:
              type: string
              description: Socket.IO session ID when connected
            automation:
              type: object
              properties:
                active_campaigns:
                  type: array
                  items:
                    type: string
      503:
        description: Foundry not connected
    """
    if foundry_bridge and foundry_bridge.is_connected():
        # Get active automation campaigns
        active_campaigns = [
            campaign_id for campaign_id, loop in _game_loops.items() if loop.enabled
        ]

        return jsonify(
            {
                "connected": True,
                "session_id": foundry_bridge._session_id if foundry_bridge else None,
                "automation": {
                    "active_campaigns": active_campaigns,
                },
            }
        )
    else:
        return (
            jsonify(
                {
                    "connected": False,
                    "session_id": None,
                    "automation": {
                        "active_campaigns": [],
                    },
                }
            ),
            503,
        )


@app.route("/api/health/llm", methods=["GET"])
def health_llm():
    """
    Check LLM backend availability
    ---
    tags:
      - System
    responses:
      200:
        description: LLM backend status
        schema:
          type: object
          properties:
            available:
              type: boolean
            backend:
              type: string
              example: ollama
            model:
              type: string
              example: gpt-oss:20b
            available_backends:
              type: array
              items:
                type: string
      503:
        description: LLM backend unavailable
    """
    # Create a backend instance to check availability
    try:
        backend = get_backend()
        available = backend.is_available()
        model = backend.get_model_name()
    except Exception as e:
        return (
            jsonify(
                {
                    "available": False,
                    "backend": LLM_BACKEND,
                    "model": None,
                    "error": str(e),
                    "available_backends": list_backends(),
                }
            ),
            503,
        )

    response_data = {
        "available": available,
        "backend": LLM_BACKEND,
        "model": model,
        "available_backends": list_backends(),
    }

    if available:
        return jsonify(response_data)
    else:
        return jsonify(response_data), 503


# =============================================================================
# Error Handlers
# =============================================================================


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


@jwt.unauthorized_loader
def unauthorized_callback(reason):
    return (
        jsonify({"error": "Missing or invalid authorization token", "reason": reason}),
        401,
    )


@jwt.invalid_token_loader
def invalid_token_callback(reason):
    return jsonify({"error": "Invalid token", "reason": reason}), 401


@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({"error": "Token has expired"}), 401


# =============================================================================
# State Checkpointing
# =============================================================================


def _publish_loop_state():
    """Publish current game loop state to Redis for checkpointing."""
    try:
        from gm_agent.tasks import publish_loop_state
        import time

        active_loops = {}
        for campaign_id, loop in _game_loops.items():
            stats = loop.get_stats()
            active_loops[campaign_id] = {
                "enabled": loop.enabled,
                "started_at": stats.get("started_at"),
            }

        publish_loop_state(active_loops)
    except Exception as e:
        logger.warning(f"Failed to publish loop state: {e}")


def _recover_checkpointed_state():
    """Recover game loop state from Redis checkpoint on startup.

    This is called during app initialization to restore any active
    game loops that were running before a restart.
    """
    try:
        from gm_agent.tasks import get_checkpointed_state
        import time

        state = get_checkpointed_state()
        if not state:
            logger.info("No checkpointed state to recover")
            return

        # Check if checkpoint is stale
        checkpoint_age = time.time() - state.get("timestamp", 0)
        if checkpoint_age > CHECKPOINT_STALE_THRESHOLD:
            logger.warning(
                f"Checkpointed state is stale ({checkpoint_age:.0f}s old), skipping recovery"
            )
            return

        loops = state.get("loops", {})
        if not loops:
            return

        logger.info(f"Recovering {len(loops)} game loop(s) from checkpoint")

        for campaign_id, loop_info in loops.items():
            if not loop_info.get("enabled"):
                continue

            # Verify campaign still exists
            campaign = campaign_store.get(campaign_id)
            if not campaign:
                logger.warning(f"Campaign '{campaign_id}' no longer exists, skipping")
                continue

            try:
                # Recreate the game loop (will start paused, waiting for Foundry)
                agent = get_agent(campaign_id)
                _game_loops[campaign_id] = GameLoopController(campaign_id, agent, foundry_bridge)
                logger.info(f"Recovered game loop for campaign: {campaign_id}")
                # Note: Loop will be started when Foundry reconnects and sends automationToggle
            except Exception as e:
                logger.error(f"Failed to recover loop for {campaign_id}: {e}")

    except ImportError:
        logger.debug("Redis/Celery not available, skipping state recovery")
    except Exception as e:
        logger.warning(f"Failed to recover checkpointed state: {e}")


def _clear_checkpoint():
    """Clear checkpointed state on clean shutdown."""
    try:
        from gm_agent.tasks import clear_checkpoint

        clear_checkpoint.delay()
    except ImportError:
        pass  # Redis/Celery not available
    except Exception as e:
        logger.warning(f"Failed to clear checkpoint: {e}")


# =============================================================================
# Graceful Shutdown
# =============================================================================


_shutdown_in_progress = False


def shutdown():
    """Clean up resources on shutdown."""
    global _shutdown_in_progress

    if _shutdown_in_progress:
        return

    _shutdown_in_progress = True
    logger.info("Initiating graceful shutdown...")

    # Clear checkpoint (clean shutdown = no recovery needed)
    _clear_checkpoint()

    # Stop all game loops
    for campaign_id, loop in list(_game_loops.items()):
        logger.info(f"Stopping automation for campaign: {campaign_id}")
        loop.stop()
    _game_loops.clear()

    # Close all active agents
    for campaign_id, agent in list(_active_agents.items()):
        logger.info(f"Closing agent for campaign: {campaign_id}")
        try:
            agent.close()
        except Exception as e:
            logger.warning(f"Error closing agent {campaign_id}: {e}")
    _active_agents.clear()

    # Disconnect Foundry bridge
    global foundry_bridge, foundry_server
    if foundry_bridge:
        logger.info("Disconnecting Foundry bridge")
        foundry_bridge.set_disconnected()
        foundry_bridge = None
        foundry_server = None

    logger.info("Shutdown complete")


def _signal_handler(signum, frame):
    """Handle shutdown signals."""
    signal_name = signal.Signals(signum).name
    logger.info(f"Received signal {signal_name}, shutting down...")
    shutdown()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# Register atexit handler as fallback
atexit.register(shutdown)


# =============================================================================
# Polling Mode Initialization
# =============================================================================


def init_polling_mode() -> bool:
    """Initialize polling mode if configured.

    Returns:
        True if polling mode was initialized, False otherwise.
    """
    global foundry_bridge, foundry_server, _polling_client

    if FOUNDRY_MODE != "polling":
        return False

    if not FOUNDRY_POLL_URL:
        logger.warning("FOUNDRY_MODE=polling but FOUNDRY_POLL_URL not set")
        return False

    if not FOUNDRY_API_KEY:
        logger.warning("FOUNDRY_MODE=polling but FOUNDRY_API_KEY not set")
        return False

    logger.info(f"Initializing polling mode for {FOUNDRY_POLL_URL}")

    config = PollingConfig(
        base_url=FOUNDRY_POLL_URL,
        api_key=FOUNDRY_API_KEY,
        campaign_id=FOUNDRY_CAMPAIGN_ID,
        poll_interval=FOUNDRY_POLL_INTERVAL,
        long_poll_timeout=FOUNDRY_LONG_POLL_TIMEOUT,
        verify_ssl=FOUNDRY_VERIFY_SSL,
    )

    _polling_client = FoundryPollingClient(config)
    foundry_bridge = _polling_client
    foundry_server = FoundryVTTServer(_polling_client)

    # Update existing agents with the new foundry server
    for agent in _active_agents.values():
        agent.set_foundry_server(foundry_server)

    # Start the polling client
    _polling_client.start()

    logger.info("Polling mode initialized successfully")
    return True


def shutdown_polling_mode() -> None:
    """Shutdown polling client if active."""
    global _polling_client, foundry_bridge, foundry_server

    if _polling_client:
        logger.info("Shutting down polling client")
        _polling_client.stop()
        _polling_client = None
        foundry_bridge = None
        foundry_server = None

        # Stop all game loops
        for loop in _game_loops.values():
            loop.stop()


# =============================================================================
# Run Server
# =============================================================================


def create_app():
    """Application factory for uWSGI/Gunicorn."""
    # Recover any checkpointed state from previous run
    _recover_checkpointed_state()

    # Initialize polling mode if configured
    if FOUNDRY_MODE == "polling":
        init_polling_mode()

    return app


def get_foundry_server() -> FoundryVTTServer | None:
    """Get the current Foundry VTT server instance."""
    return foundry_server


def get_foundry_bridge() -> FoundryBridgeBase | None:
    """Get the current Foundry bridge instance."""
    return foundry_bridge


if __name__ == "__main__":
    if FOUNDRY_MODE == "polling":
        # Polling mode - no Socket.IO needed, use standard Flask server
        logger.info("Starting in polling mode")
        init_polling_mode()
        try:
            app.run(debug=True, host="0.0.0.0", port=5000)
        finally:
            shutdown_polling_mode()
    else:
        # WebSocket mode - use socketio.run for Socket.IO support
        logger.info("Starting in WebSocket mode")
        socketio.run(app, debug=True, host="0.0.0.0", port=5000)
