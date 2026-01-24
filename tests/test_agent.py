"""Integration tests for GMAgent loop."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from gm_agent.agent import GMAgent
from gm_agent.models.base import LLMResponse, Message, ToolCall
from gm_agent.storage.schemas import Campaign, Session, SceneState, PartyMember
from gm_agent.storage.campaign import CampaignStore
from gm_agent.storage.session import SessionStore

from tests.conftest import MockLLMBackend, MockMCPServer, MockMCPClient


class TestGMAgentInitialization:
    """Tests for GMAgent.__init__."""

    def test_init_loads_campaign(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """GMAgent should load campaign on initialization."""
        # Create campaign
        campaign = campaign_store.create(name="Init Test")

        # Patch stores and MCP client
        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

        assert agent.campaign.id == campaign.id
        assert agent.campaign.name == "Init Test"

    def test_init_creates_session(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """GMAgent should create session if none exists."""
        campaign = campaign_store.create(name="Session Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

        assert agent.session is not None
        assert agent.session.campaign_id == campaign.id

    def test_init_reuses_existing_session(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """GMAgent should reuse existing active session."""
        campaign = campaign_store.create(name="Reuse Session Test")
        existing_session = session_store.start(campaign.id)

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

        assert agent.session.id == existing_session.id

    def test_init_missing_campaign_raises(
        self,
        campaign_store: CampaignStore,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """GMAgent should raise ValueError for missing campaign."""
        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)

        with pytest.raises(ValueError, match="not found"):
            GMAgent(campaign_id="nonexistent")

    def test_init_default_llm(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """GMAgent should use get_backend by default."""
        campaign = campaign_store.create(name="Default LLM Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            with patch("gm_agent.agent.get_backend") as mock_get_backend:
                mock_get_backend.return_value = MockLLMBackend()
                agent = GMAgent(campaign_id=campaign.id)
                mock_get_backend.assert_called_once()


class TestProcessTurn:
    """Tests for GMAgent.process_turn."""

    @pytest.fixture
    def agent_with_mocks(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Create GMAgent with mocked dependencies."""
        campaign = campaign_store.create(
            name="Process Turn Test",
            background="Test campaign background",
        )

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)
            yield agent
            agent.close()

    def test_process_turn_returns_response(self, agent_with_mocks: GMAgent):
        """process_turn should return LLM response text."""
        response = agent_with_mocks.process_turn("Hello, GM!")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_process_turn_records_turn(
        self,
        agent_with_mocks: GMAgent,
        session_store: SessionStore,
    ):
        """process_turn should record turn in session."""
        agent_with_mocks.process_turn("I attack the goblin!")

        session = session_store.get_current(agent_with_mocks.campaign.id)
        assert len(session.turns) == 1
        assert session.turns[0].player_input == "I attack the goblin!"

    def test_process_turn_with_tool_call(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_with_tool_call: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should execute tool calls."""
        campaign = campaign_store.create(name="Tool Call Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_with_tool_call)

            response = agent.process_turn("What is a goblin?")

            # LLM should have been called twice (once for tool call, once for response)
            assert len(mock_llm_with_tool_call.calls) == 2

            # Tool call is recorded in turn - verify via session
            session = session_store.get_current(campaign.id)
            assert len(session.turns[0].tool_calls) == 1
            assert session.turns[0].tool_calls[0].name == "lookup_creature"

            agent.close()

    def test_process_turn_records_tool_calls(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_with_tool_call: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should record tool calls in turn."""
        campaign = campaign_store.create(name="Record Tool Calls Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_with_tool_call)

            agent.process_turn("What is a goblin?")

            session = session_store.get_current(campaign.id)
            turn = session.turns[0]

            assert len(turn.tool_calls) == 1
            assert turn.tool_calls[0].name == "lookup_creature"
            assert turn.tool_calls[0].args == {"name": "goblin"}

            agent.close()

    def test_process_turn_multiple_tool_calls(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_multiple_tool_calls: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should handle multiple tool calls."""
        campaign = campaign_store.create(name="Multiple Tools Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_multiple_tool_calls)

            response = agent.process_turn("Goblin vs fireball?")

            # Both tools should have been recorded in turn
            session = session_store.get_current(campaign.id)
            assert len(session.turns[0].tool_calls) == 2
            tool_names = [tc.name for tc in session.turns[0].tool_calls]
            assert "lookup_creature" in tool_names
            assert "lookup_spell" in tool_names

            agent.close()

    def test_process_turn_updates_session(
        self,
        agent_with_mocks: GMAgent,
    ):
        """process_turn should refresh session state after recording."""
        initial_session = agent_with_mocks.session

        agent_with_mocks.process_turn("Test input")

        # Session should be updated with new turn
        assert len(agent_with_mocks.session.turns) == 1
        assert agent_with_mocks.session.turns[0].player_input == "Test input"

    def test_process_turn_max_iterations(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should limit tool call iterations."""
        # Create LLM that always wants to make tool calls
        infinite_tool_llm = MockLLMBackend(
            responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{i}",
                            name="lookup_creature",
                            args={"name": "goblin"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
                for i in range(10)  # More than max_iterations (5)
            ]
        )

        campaign = campaign_store.create(name="Max Iterations Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=infinite_tool_llm)

            # Should complete without infinite loop
            response = agent.process_turn("Infinite loop test")

            # Should have stopped at max_iterations (5)
            assert len(infinite_tool_llm.calls) == 5

            agent.close()


class TestEndSession:
    """Tests for GMAgent.end_session."""

    def test_end_session_archives(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """end_session should archive the current session."""
        campaign = campaign_store.create(name="End Session Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)
            session_id = agent.session.id

            agent.process_turn("Test turn")
            ended = agent.end_session(summary="Great session!")

            assert ended is not None
            assert ended.id == session_id
            assert ended.summary == "Great session!"
            assert ended.ended_at is not None

            # Session should be archived
            archived = session_store.list(campaign.id)
            assert len(archived) == 1

            agent.close()

    def test_end_session_returns_none_when_no_session(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """end_session should return None if already ended."""
        campaign = campaign_store.create(name="No Session Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

            agent.end_session()
            result = agent.end_session()

            assert result is None

            agent.close()


class TestUpdateScene:
    """Tests for GMAgent.update_scene."""

    def test_update_scene_persists(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """update_scene should persist scene changes."""
        campaign = campaign_store.create(name="Scene Update Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

            agent.update_scene(
                location="Dragon's Lair",
                npcs_present=["Ancient Red Dragon"],
                time_of_day="night",
                conditions=["dark", "hot"],
                notes="Very dangerous!",
            )

            # Check agent's session
            assert agent.session.scene_state.location == "Dragon's Lair"
            assert "Ancient Red Dragon" in agent.session.scene_state.npcs_present
            assert agent.session.scene_state.time_of_day == "night"
            assert "dark" in agent.session.scene_state.conditions

            # Check persisted session
            persisted = session_store.get_current(campaign.id)
            assert persisted.scene_state.location == "Dragon's Lair"

            agent.close()

    def test_update_scene_partial(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """update_scene should preserve unspecified fields."""
        campaign = campaign_store.create(name="Partial Scene Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

            # Set initial scene
            agent.update_scene(
                location="Tavern",
                time_of_day="evening",
            )

            # Update only location
            agent.update_scene(location="Town Square")

            # Time should be preserved
            assert agent.session.scene_state.location == "Town Square"
            assert agent.session.scene_state.time_of_day == "evening"

            agent.close()


class TestAgentClose:
    """Tests for GMAgent.close."""

    def test_close_cleans_up_rag(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """close should clean up MCP client."""
        campaign = campaign_store.create(name="Close Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        mock_client = MagicMock()
        with patch("gm_agent.agent.MCPClient", return_value=mock_client):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)
            agent.close()

            mock_client.close.assert_called_once()


class TestAgentContextBuilding:
    """Tests for how GMAgent builds context for LLM."""

    def test_context_includes_campaign_info(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        sample_party: list[PartyMember],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should pass campaign context to LLM."""
        campaign = campaign_store.create(
            name="Context Test Campaign",
            background="A dark and dangerous world",
            current_arc="The heroes seek the artifact",
            party=sample_party,
        )

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

            agent.process_turn("What do I see?")

            # Check what was sent to LLM
            messages, tools = mock_llm_backend.calls[0]

            # System message should have campaign info
            system_msg = next(m for m in messages if m.role == "system")
            assert "dark and dangerous" in system_msg.content
            assert "artifact" in system_msg.content
            assert "Valeros" in system_msg.content

            agent.close()

    def test_context_includes_previous_turns(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should include previous turns in context."""
        campaign = campaign_store.create(name="Previous Turns Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        # Create LLM with multiple responses
        multi_response_llm = MockLLMBackend(
            responses=[
                LLMResponse(text="Response 1", tool_calls=[], finish_reason="stop"),
                LLMResponse(text="Response 2", tool_calls=[], finish_reason="stop"),
                LLMResponse(text="Response 3", tool_calls=[], finish_reason="stop"),
            ]
        )

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=multi_response_llm)

            agent.process_turn("First action")
            agent.process_turn("Second action")
            agent.process_turn("Third action")

            # Third call should include previous turns
            messages, tools = multi_response_llm.calls[2]

            user_messages = [m for m in messages if m.role == "user"]
            assert len(user_messages) == 3
            assert user_messages[0].content == "First action"
            assert user_messages[1].content == "Second action"
            assert user_messages[2].content == "Third action"

            agent.close()

    def test_context_passes_tools_to_llm(
        self,
        campaign_store: CampaignStore,
        session_store: SessionStore,
        mock_llm_backend: MockLLMBackend,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """process_turn should pass tool definitions to LLM."""
        campaign = campaign_store.create(name="Tools Context Test")

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.MCPClient", MockMCPClient):
            agent = GMAgent(campaign_id=campaign.id, llm=mock_llm_backend)

            agent.process_turn("What tools do you have?")

            messages, tools = mock_llm_backend.calls[0]

            # Tools should be passed
            assert tools is not None
            # 7 RAG tools + 9 campaign state tools + 6 character runner tools = 22 total
            assert len(tools) == 22

            tool_names = [t.name for t in tools]
            # RAG tools
            assert "lookup_creature" in tool_names
            assert "lookup_spell" in tool_names
            # Campaign state tools
            assert "update_scene" in tool_names
            assert "log_event" in tool_names
            assert "search_history" in tool_names
            # Character runner tools
            assert "run_npc" in tool_names
            assert "run_monster" in tool_names
            assert "create_character" in tool_names

            agent.close()
