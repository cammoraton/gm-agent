"""Tests for Flask REST API."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from api import app, _active_agents
from gm_agent.storage.campaign import CampaignStore
from gm_agent.storage.session import SessionStore
from gm_agent.storage.characters import CharacterStore


@pytest.fixture
def client(tmp_path: Path):
    """Create a test client with isolated storage."""
    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()

    # Create stores
    campaign_store = CampaignStore(base_dir=campaigns_dir)
    session_store = SessionStore(base_dir=campaigns_dir)

    # Patch global stores
    with (
        patch("api.campaign_store", campaign_store),
        patch("api.session_store", session_store),
        patch("gm_agent.agent.campaign_store", campaign_store),
        patch("gm_agent.agent.session_store", session_store),
    ):

        # Also patch get_character_store to use temp dir
        def mock_get_character_store(campaign_id):
            return CharacterStore(campaign_id, base_dir=campaigns_dir)

        with patch("api.get_character_store", mock_get_character_store):
            app.config["TESTING"] = True
            with app.test_client() as client:
                yield client, campaign_store, session_store, campaigns_dir

    # Clean up active agents
    _active_agents.clear()


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return ok status."""
        test_client, _, _, _ = client
        response = test_client.get("/api/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"
        assert data["service"] == "gm-agent"

    def test_health_foundry_disconnected(self, client):
        """Foundry health endpoint returns 503 when not connected."""
        test_client, _, _, _ = client
        with patch("api.foundry_bridge", None):
            response = test_client.get("/api/health/foundry")

        assert response.status_code == 503
        data = json.loads(response.data)
        assert data["connected"] is False
        assert data["session_id"] is None
        assert data["automation"]["active_campaigns"] == []

    def test_health_foundry_connected(self, client):
        """Foundry health endpoint returns 200 when connected."""
        test_client, _, _, _ = client
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = True
        mock_bridge._session_id = "test-session-123"

        with patch("api.foundry_bridge", mock_bridge), patch("api._game_loops", {}):
            response = test_client.get("/api/health/foundry")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["connected"] is True
        assert data["session_id"] == "test-session-123"

    def test_health_llm_available(self, client):
        """LLM health endpoint returns 200 when backend is available."""
        test_client, _, _, _ = client
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = True
        mock_backend.get_model_name.return_value = "gpt-oss:20b"

        with patch("api.get_backend", return_value=mock_backend):
            with patch("api.LLM_BACKEND", "ollama"):
                response = test_client.get("/api/health/llm")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["available"] is True
        assert data["backend"] == "ollama"
        assert data["model"] == "gpt-oss:20b"

    def test_health_llm_unavailable(self, client):
        """LLM health endpoint returns 503 when backend is unavailable."""
        test_client, _, _, _ = client
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = False
        mock_backend.get_model_name.return_value = "gpt-oss:20b"

        with patch("api.get_backend", return_value=mock_backend):
            with patch("api.LLM_BACKEND", "ollama"):
                response = test_client.get("/api/health/llm")

        assert response.status_code == 503
        data = json.loads(response.data)
        assert data["available"] is False


class TestCampaignEndpoints:
    """Tests for campaign CRUD endpoints."""

    def test_list_campaigns_empty(self, client):
        """Should return empty list when no campaigns."""
        test_client, _, _, _ = client
        response = test_client.get("/api/campaigns")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["campaigns"] == []

    def test_create_campaign(self, client):
        """Should create a new campaign."""
        test_client, _, _, _ = client
        response = test_client.post(
            "/api/campaigns",
            json={"name": "Test Campaign", "background": "A test adventure"},
        )

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data["id"] == "test-campaign"
        assert data["name"] == "Test Campaign"

    def test_create_campaign_with_party(self, client):
        """Should create campaign with party members."""
        test_client, _, _, _ = client
        response = test_client.post(
            "/api/campaigns",
            json={
                "name": "Party Campaign",
                "party": [
                    {"name": "Valeros", "class_name": "Fighter", "level": 3},
                    {"name": "Seoni", "class_name": "Sorcerer", "level": 3},
                ],
            },
        )

        assert response.status_code == 201

        # Verify party was saved
        get_response = test_client.get("/api/campaigns/party-campaign")
        data = json.loads(get_response.data)
        assert len(data["party"]) == 2

    def test_create_campaign_missing_name(self, client):
        """Should error when name is missing."""
        test_client, _, _, _ = client
        response = test_client.post("/api/campaigns", json={})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "name is required" in data["error"]

    def test_create_duplicate_campaign(self, client):
        """Should error on duplicate campaign name."""
        test_client, _, _, _ = client
        test_client.post("/api/campaigns", json={"name": "Duplicate"})
        response = test_client.post("/api/campaigns", json={"name": "Duplicate"})

        assert response.status_code == 400
        assert "already exists" in json.loads(response.data)["error"]

    def test_get_campaign(self, client):
        """Should return campaign details."""
        test_client, _, _, _ = client
        test_client.post(
            "/api/campaigns",
            json={"name": "Get Test", "background": "Test background"},
        )

        response = test_client.get("/api/campaigns/get-test")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["name"] == "Get Test"
        assert data["background"] == "Test background"

    def test_get_campaign_not_found(self, client):
        """Should return 404 for missing campaign."""
        test_client, _, _, _ = client
        response = test_client.get("/api/campaigns/nonexistent")

        assert response.status_code == 404

    def test_update_campaign(self, client):
        """Should update campaign fields."""
        test_client, _, _, _ = client
        test_client.post("/api/campaigns", json={"name": "Update Test"})

        response = test_client.put(
            "/api/campaigns/update-test",
            json={"background": "Updated background", "current_arc": "New arc"},
        )

        assert response.status_code == 200

        # Verify update
        get_response = test_client.get("/api/campaigns/update-test")
        data = json.loads(get_response.data)
        assert data["background"] == "Updated background"
        assert data["current_arc"] == "New arc"

    def test_delete_campaign(self, client):
        """Should delete campaign."""
        test_client, _, _, _ = client
        test_client.post("/api/campaigns", json={"name": "Delete Me"})

        response = test_client.delete("/api/campaigns/delete-me")

        assert response.status_code == 200

        # Verify deleted
        get_response = test_client.get("/api/campaigns/delete-me")
        assert get_response.status_code == 404

    def test_delete_campaign_not_found(self, client):
        """Should return 404 when deleting nonexistent campaign."""
        test_client, _, _, _ = client
        response = test_client.delete("/api/campaigns/nonexistent")

        assert response.status_code == 404

    def test_list_campaigns(self, client):
        """Should list all campaigns."""
        test_client, _, _, _ = client
        test_client.post("/api/campaigns", json={"name": "Campaign 1"})
        test_client.post("/api/campaigns", json={"name": "Campaign 2"})

        response = test_client.get("/api/campaigns")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["campaigns"]) == 2


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    @pytest.fixture
    def campaign_client(self, client):
        """Create a client with a pre-made campaign."""
        test_client, campaign_store, session_store, campaigns_dir = client
        test_client.post("/api/campaigns", json={"name": "Session Test"})
        return test_client, campaign_store, session_store, campaigns_dir

    def test_list_sessions_empty(self, campaign_client):
        """Should return empty list when no sessions."""
        test_client, _, _, _ = campaign_client
        response = test_client.get("/api/campaigns/session-test/sessions")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["sessions"] == []
        assert data["current_session"] is None

    def test_start_session(self, campaign_client):
        """Should start a new session."""
        test_client, _, _, _ = campaign_client

        # Mock the GMAgent to avoid LLM calls
        with patch("api.GMAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            response = test_client.post("/api/campaigns/session-test/sessions/start")

        assert response.status_code == 201
        data = json.loads(response.data)
        assert "session_id" in data
        assert data["message"] == "Session started"

    def test_start_session_campaign_not_found(self, client):
        """Should error when campaign doesn't exist."""
        test_client, _, _, _ = client
        response = test_client.post("/api/campaigns/nonexistent/sessions/start")

        assert response.status_code == 404

    def test_start_session_already_active(self, campaign_client):
        """Should error when session already active."""
        test_client, _, _, _ = campaign_client

        with patch("api.GMAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            test_client.post("/api/campaigns/session-test/sessions/start")
            response = test_client.post("/api/campaigns/session-test/sessions/start")

        assert response.status_code == 400
        assert "already active" in json.loads(response.data)["error"]

    def test_process_turn(self, campaign_client):
        """Should process a turn and return response."""
        test_client, _, _, _ = campaign_client

        with patch("api.GMAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.process_turn.return_value = "You see a dark corridor ahead."
            mock_agent_class.return_value = mock_agent

            test_client.post("/api/campaigns/session-test/sessions/start")

            response = test_client.post(
                "/api/campaigns/session-test/sessions/turn",
                json={"input": "I look around"},
            )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["response"] == "You see a dark corridor ahead."

    def test_process_turn_no_session(self, campaign_client):
        """Should error when no active session."""
        test_client, _, _, _ = campaign_client
        response = test_client.post(
            "/api/campaigns/session-test/sessions/turn",
            json={"input": "Hello"},
        )

        assert response.status_code == 400
        assert "No active session" in json.loads(response.data)["error"]

    def test_process_turn_missing_input(self, campaign_client):
        """Should error when input is missing."""
        test_client, _, _, _ = campaign_client

        with patch("api.GMAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            test_client.post("/api/campaigns/session-test/sessions/start")
            response = test_client.post(
                "/api/campaigns/session-test/sessions/turn",
                json={},
            )

        assert response.status_code == 400
        assert "input is required" in json.loads(response.data)["error"]

    def test_end_session(self, campaign_client):
        """Should end the current session."""
        test_client, _, _, _ = campaign_client

        with patch("api.GMAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_session = MagicMock()
            mock_session.id = "test-session"
            mock_session.turns = []
            mock_session.summary = "Test summary"
            mock_agent.end_session.return_value = mock_session
            mock_agent_class.return_value = mock_agent

            test_client.post("/api/campaigns/session-test/sessions/start")
            response = test_client.post(
                "/api/campaigns/session-test/sessions/end",
                json={"summary": "Test summary"},
            )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["message"] == "Session ended"

    def test_end_session_no_active(self, campaign_client):
        """Should error when no active session."""
        test_client, _, _, _ = campaign_client
        response = test_client.post("/api/campaigns/session-test/sessions/end")

        assert response.status_code == 400
        assert "No active session" in json.loads(response.data)["error"]


class TestCharacterEndpoints:
    """Tests for character profile endpoints."""

    @pytest.fixture
    def campaign_client(self, client):
        """Create a client with a pre-made campaign."""
        test_client, campaign_store, session_store, campaigns_dir = client
        test_client.post("/api/campaigns", json={"name": "Character Test"})
        return test_client, campaign_store, session_store, campaigns_dir

    def test_list_characters_empty(self, campaign_client):
        """Should return empty list when no characters."""
        test_client, _, _, _ = campaign_client
        response = test_client.get("/api/campaigns/character-test/characters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["characters"] == []

    def test_create_character(self, campaign_client):
        """Should create a new character."""
        test_client, _, _, _ = campaign_client
        response = test_client.post(
            "/api/campaigns/character-test/characters",
            json={
                "name": "Sheriff Hemlock",
                "character_type": "npc",
                "personality": "Stern but fair",
                "knowledge": "Local laws, Town history",
                "goals": "Keep the peace",
            },
        )

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data["name"] == "Sheriff Hemlock"
        assert data["character_type"] == "npc"

    def test_create_monster_character(self, campaign_client):
        """Should create a monster profile."""
        test_client, _, _, _ = campaign_client
        response = test_client.post(
            "/api/campaigns/character-test/characters",
            json={
                "name": "Goblin Scout",
                "character_type": "monster",
                "intelligence": "low",
                "instincts": ["cowardly", "alert allies"],
                "morale": "Flees when outnumbered",
            },
        )

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data["character_type"] == "monster"

    def test_create_character_missing_name(self, campaign_client):
        """Should error when name is missing."""
        test_client, _, _, _ = campaign_client
        response = test_client.post(
            "/api/campaigns/character-test/characters",
            json={"character_type": "npc"},
        )

        assert response.status_code == 400
        assert "name is required" in json.loads(response.data)["error"]

    def test_create_duplicate_character(self, campaign_client):
        """Should error on duplicate character name."""
        test_client, _, _, _ = campaign_client
        test_client.post(
            "/api/campaigns/character-test/characters",
            json={"name": "Duplicate"},
        )
        response = test_client.post(
            "/api/campaigns/character-test/characters",
            json={"name": "Duplicate"},
        )

        assert response.status_code == 400
        assert "already exists" in json.loads(response.data)["error"]

    def test_get_character(self, campaign_client):
        """Should return character details."""
        test_client, _, _, _ = campaign_client
        test_client.post(
            "/api/campaigns/character-test/characters",
            json={
                "name": "Innkeeper",
                "personality": "Friendly and talkative",
            },
        )

        response = test_client.get("/api/campaigns/character-test/characters/Innkeeper")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["name"] == "Innkeeper"
        assert data["personality"] == "Friendly and talkative"

    def test_get_character_not_found(self, campaign_client):
        """Should return 404 for missing character."""
        test_client, _, _, _ = campaign_client
        response = test_client.get("/api/campaigns/character-test/characters/Nobody")

        assert response.status_code == 404

    def test_delete_character(self, campaign_client):
        """Should delete a character."""
        test_client, _, _, _ = campaign_client
        test_client.post(
            "/api/campaigns/character-test/characters",
            json={"name": "Delete Me"},
        )

        response = test_client.delete("/api/campaigns/character-test/characters/Delete Me")

        assert response.status_code == 200

        # Verify deleted
        get_response = test_client.get("/api/campaigns/character-test/characters/Delete Me")
        assert get_response.status_code == 404

    def test_list_characters_by_type(self, campaign_client):
        """Should filter characters by type."""
        test_client, _, _, _ = campaign_client
        test_client.post(
            "/api/campaigns/character-test/characters",
            json={"name": "NPC 1", "character_type": "npc"},
        )
        test_client.post(
            "/api/campaigns/character-test/characters",
            json={"name": "Monster 1", "character_type": "monster"},
        )

        response = test_client.get("/api/campaigns/character-test/characters?type=npc")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["characters"]) == 1
        assert data["characters"][0]["name"] == "NPC 1"

    def test_character_campaign_not_found(self, client):
        """Should error when campaign doesn't exist."""
        test_client, _, _, _ = client
        response = test_client.get("/api/campaigns/nonexistent/characters")

        assert response.status_code == 404


class TestErrorHandlers:
    """Tests for error handlers."""

    def test_404_endpoint(self, client):
        """Should return JSON for unknown endpoints."""
        test_client, _, _, _ = client
        response = test_client.get("/api/unknown/endpoint")

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestGracefulShutdown:
    """Tests for graceful shutdown handling."""

    def test_shutdown_clears_game_loops(self, client):
        """Shutdown should stop and clear all game loops."""
        from api import _game_loops, shutdown, _shutdown_in_progress
        import api

        # Reset shutdown state
        api._shutdown_in_progress = False

        # Create mock game loops
        mock_loop1 = MagicMock()
        mock_loop2 = MagicMock()
        _game_loops["campaign-1"] = mock_loop1
        _game_loops["campaign-2"] = mock_loop2

        try:
            shutdown()

            mock_loop1.stop.assert_called_once()
            mock_loop2.stop.assert_called_once()
            assert len(_game_loops) == 0
        finally:
            api._shutdown_in_progress = False

    def test_shutdown_closes_agents(self, client):
        """Shutdown should close all active agents."""
        from api import _active_agents, shutdown
        import api

        # Reset shutdown state
        api._shutdown_in_progress = False

        # Create mock agents
        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        _active_agents["campaign-1"] = mock_agent1
        _active_agents["campaign-2"] = mock_agent2

        try:
            shutdown()

            mock_agent1.close.assert_called_once()
            mock_agent2.close.assert_called_once()
            assert len(_active_agents) == 0
        finally:
            api._shutdown_in_progress = False

    def test_shutdown_idempotent(self, client):
        """Multiple shutdown calls should only execute once."""
        from api import shutdown
        import api

        # Reset shutdown state
        api._shutdown_in_progress = False

        # Call shutdown twice
        shutdown()
        assert api._shutdown_in_progress is True

        # Second call should be a no-op
        shutdown()  # Should not raise
        assert api._shutdown_in_progress is True

        # Reset for other tests
        api._shutdown_in_progress = False


class TestAuthentication:
    """Tests for JWT authentication."""

    def test_auth_status_disabled(self, client):
        """Should report auth disabled by default."""
        test_client, _, _, _ = client
        response = test_client.get("/api/auth/status")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["auth_enabled"] is False

    def test_get_token_when_disabled(self, client):
        """Should return 403 when auth is disabled."""
        test_client, _, _, _ = client
        response = test_client.post(
            "/api/auth/token",
            json={"username": "admin", "password": "changeme"},
        )

        assert response.status_code == 403
        data = json.loads(response.data)
        assert "disabled" in data["error"]

    def test_endpoints_work_without_auth(self, client):
        """Endpoints should work without auth when disabled."""
        test_client, _, _, _ = client
        # Should work without token
        response = test_client.get("/api/campaigns")
        assert response.status_code == 200

    def test_health_check_always_available(self, client):
        """Health check should always be accessible."""
        test_client, _, _, _ = client
        response = test_client.get("/api/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"
        assert "auth_enabled" in data


class TestChatEndpoint:
    """Tests for stateless chat endpoint."""

    def test_chat_missing_message(self, client):
        """Should error when message is missing."""
        test_client, _, _, _ = client
        response = test_client.post("/api/chat", json={})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Message required" in data["error"]

    def test_chat_empty_body(self, client):
        """Should error when body is empty."""
        test_client, _, _, _ = client
        response = test_client.post("/api/chat", data="", content_type="application/json")

        assert response.status_code == 400

    def test_chat_success(self, client):
        """Should return response from ChatAgent."""
        test_client, _, _, _ = client

        with patch("api.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "Flanking gives you a +2 circumstance bonus..."
            mock_agent_class.return_value = mock_agent

            response = test_client.post("/api/chat", json={"message": "What is flanking?"})

            assert response.status_code == 200
            data = json.loads(response.data)
            assert "response" in data
            assert "Flanking" in data["response"]

            # Verify agent was called correctly
            mock_agent.chat.assert_called_once_with("What is flanking?")
            mock_agent.close.assert_called_once()

    def test_chat_with_context(self, client):
        """Should include context in message when provided."""
        test_client, _, _, _ = client

        with patch("api.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "The encounter looks challenging..."
            mock_agent_class.return_value = mock_agent

            response = test_client.post(
                "/api/chat",
                json={
                    "message": "How difficult is this fight?",
                    "context": {
                        "combat": {
                            "round": 2,
                            "combatants": [
                                {"name": "Goblin", "hp": {"value": 10, "max": 15}},
                                {"name": "Fighter", "hp": {"value": 30, "max": 40}},
                            ],
                        }
                    },
                },
            )

            assert response.status_code == 200

            # Verify context was prepended to message
            call_args = mock_agent.chat.call_args[0][0]
            assert "Combat State" in call_args
            assert "Goblin" in call_args
            assert "How difficult is this fight?" in call_args

    def test_chat_with_token_context(self, client):
        """Should include selected tokens in context."""
        test_client, _, _, _ = client

        with patch("api.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "The dragon is level 12..."
            mock_agent_class.return_value = mock_agent

            response = test_client.post(
                "/api/chat",
                json={
                    "message": "What level is this creature?",
                    "context": {
                        "tokens": [{"name": "Adult Red Dragon", "type": "npc", "level": 12}]
                    },
                },
            )

            assert response.status_code == 200
            call_args = mock_agent.chat.call_args[0][0]
            assert "Adult Red Dragon" in call_args
            assert "Level 12" in call_args

    def test_chat_agent_error(self, client):
        """Should return 500 when agent throws error."""
        test_client, _, _, _ = client

        with patch("api.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.side_effect = Exception("LLM connection failed")
            mock_agent_class.return_value = mock_agent

            response = test_client.post("/api/chat", json={"message": "Test query"})

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "LLM connection failed" in data["error"]

            # Verify close was still called
            mock_agent.close.assert_called_once()

    def test_chat_cors_headers(self, client):
        """Should include CORS headers in response."""
        test_client, _, _, _ = client

        with patch("api.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "Test response"
            mock_agent_class.return_value = mock_agent

            response = test_client.post(
                "/api/chat",
                json={"message": "Test"},
                headers={"Origin": "http://localhost:30000"},
            )

            assert response.status_code == 200
            # CORS headers should be present (flask-cors echoes origin or uses *)
            cors_header = response.headers.get("Access-Control-Allow-Origin")
            assert cors_header in ["*", "http://localhost:30000"]


class TestAuthenticationEnabled:
    """Tests for JWT authentication when enabled."""

    @pytest.fixture
    def auth_client(self, tmp_path: Path):
        """Create a test client with auth enabled."""
        import os
        from api import create_app, _active_agents
        from gm_agent.storage.campaign import CampaignStore
        from gm_agent.storage.session import SessionStore

        # Enable auth
        os.environ["API_AUTH_ENABLED"] = "true"
        os.environ["API_USERNAME"] = "testuser"
        os.environ["API_PASSWORD"] = "testpass"

        campaigns_dir = tmp_path / "campaigns"
        campaigns_dir.mkdir()

        campaign_store = CampaignStore(base_dir=campaigns_dir)
        session_store = SessionStore(base_dir=campaigns_dir)

        with (
            patch("api.campaign_store", campaign_store),
            patch("api.session_store", session_store),
            patch("api.AUTH_ENABLED", True),
        ):

            # Need to reimport to get new AUTH_ENABLED value
            import importlib
            import api as api_module

            importlib.reload(api_module)

            app = api_module.create_app()
            app.config["TESTING"] = True
            with app.test_client() as client:
                yield client

        # Clean up
        _active_agents.clear()
        os.environ.pop("API_AUTH_ENABLED", None)
        os.environ.pop("API_USERNAME", None)
        os.environ.pop("API_PASSWORD", None)

        # Reload again to reset state
        importlib.reload(api_module)

    def test_get_token_valid_credentials(self, auth_client):
        """Should return token with valid credentials."""
        response = auth_client.post(
            "/api/auth/token",
            json={"username": "testuser", "password": "testpass"},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_get_token_invalid_credentials(self, auth_client):
        """Should return 401 with invalid credentials."""
        response = auth_client.post(
            "/api/auth/token",
            json={"username": "wrong", "password": "wrong"},
        )

        assert response.status_code == 401

    def test_get_token_missing_credentials(self, auth_client):
        """Should return 400 with missing credentials."""
        response = auth_client.post("/api/auth/token", json={})
        assert response.status_code == 400

    def test_protected_endpoint_without_token(self, auth_client):
        """Should return 401 without token."""
        response = auth_client.get("/api/campaigns")
        assert response.status_code == 401

    def test_protected_endpoint_with_token(self, auth_client):
        """Should work with valid token."""
        # Get token
        token_response = auth_client.post(
            "/api/auth/token",
            json={"username": "testuser", "password": "testpass"},
        )
        token = json.loads(token_response.data)["access_token"]

        # Use token
        response = auth_client.get(
            "/api/campaigns",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
