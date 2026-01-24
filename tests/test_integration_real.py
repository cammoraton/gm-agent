"""
Real integration tests against actual Ollama server.

These tests are SKIPPED by default and only run when:
1. The --run-integration flag is passed to pytest
2. The OLLAMA_URL environment variable is set and reachable

Run these tests with:
    pytest tests/test_integration_real.py -v --run-integration

Or to include them in a full test run:
    pytest tests/ -v --run-integration

These tests verify end-to-end functionality with a real LLM and help catch
issues that mocks might mask.
"""

import os
import pytest
from pathlib import Path

# Custom marker for integration tests - all tests in this file are integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_ollama_backend():
    """Create a real OllamaBackend connected to actual server."""
    from gm_agent.models.ollama import OllamaBackend
    from gm_agent.config import OLLAMA_URL, OLLAMA_MODEL

    try:
        backend = OllamaBackend()
        # Test connection
        models = backend.list_models()
        if not models:
            pytest.skip("No models available on Ollama server")
        return backend
    except Exception as e:
        pytest.skip(f"Could not connect to Ollama server: {e}")


@pytest.fixture(scope="module")
def real_rag_server():
    """Create a real PF2eRAGServer connected to actual database."""
    from gm_agent.mcp.pf2e_rag import PF2eRAGServer
    from gm_agent.config import RAG_DB_PATH

    if not RAG_DB_PATH.exists():
        pytest.skip(f"RAG database not found at {RAG_DB_PATH}")

    try:
        server = PF2eRAGServer()
        yield server
        server.close()
    except Exception as e:
        pytest.skip(f"Could not initialize RAG server: {e}")


@pytest.fixture
def real_campaign_and_session(tmp_path: Path):
    """Create real campaign and session for integration tests."""
    from gm_agent.storage.campaign import CampaignStore
    from gm_agent.storage.session import SessionStore
    from gm_agent.storage.schemas import PartyMember

    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()

    campaign_store = CampaignStore(base_dir=campaigns_dir)
    session_store = SessionStore(base_dir=campaigns_dir)

    campaign = campaign_store.create(
        name="Integration Test Campaign",
        background="A test campaign for real integration testing.",
        party=[PartyMember(name="TestHero", ancestry="Human", class_name="Fighter", level=1)],
    )

    session = session_store.start(campaign.id)

    return campaign_store, session_store, campaign, session


class TestRealOllamaBackend:
    """Tests that verify OllamaBackend works with real server."""

    def test_list_models(self, real_ollama_backend):
        """Should list available models from real server."""
        models = real_ollama_backend.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        print(f"Available models: {models}")

    def test_simple_chat(self, real_ollama_backend):
        """Should complete simple chat without tools."""
        from gm_agent.models.base import Message

        messages = [
            Message(role="user", content="Reply with exactly: INTEGRATION TEST OK"),
        ]

        response = real_ollama_backend.chat(messages)

        assert response.text is not None
        assert len(response.text) > 0
        assert response.finish_reason in ["stop", "length"]
        print(f"LLM response: {response.text}")

    def test_chat_with_tools(self, real_ollama_backend):
        """Should handle chat with tool definitions."""
        from gm_agent.models.base import Message
        from gm_agent.mcp.base import ToolDef, ToolParameter

        messages = [
            Message(
                role="system",
                content="You are a helpful assistant. Use the provided tools when asked about creatures.",
            ),
            Message(
                role="user",
                content="Use the lookup_creature tool to find info about a goblin.",
            ),
        ]

        tools = [
            ToolDef(
                name="lookup_creature",
                description="Look up creature stats",
                parameters=[ToolParameter(name="name", type="string", description="Creature name")],
            )
        ]

        response = real_ollama_backend.chat(messages, tools=tools)

        # Response might include tool calls or just text
        assert response is not None
        print(f"Response with tools: text='{response.text}', tool_calls={response.tool_calls}")

        # If tool calls were made, verify structure
        if response.tool_calls:
            for tc in response.tool_calls:
                assert tc.name is not None
                assert tc.args is not None
                print(f"Tool call: {tc.name}({tc.args})")


class TestRealRAGServer:
    """Tests that verify PF2eRAGServer works with real database."""

    def test_list_tools(self, real_rag_server):
        """Should list all 5 tools."""
        tools = real_rag_server.list_tools()

        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "lookup_creature" in tool_names
        assert "lookup_spell" in tool_names
        print(f"RAG tools: {tool_names}")

    def test_lookup_creature_goblin(self, real_rag_server):
        """Should find goblin in real database."""
        result = real_rag_server.call_tool("lookup_creature", {"name": "goblin"})

        assert result.success is True
        assert result.data is not None
        assert len(result.data) > 0
        # Goblins should be in any PF2e database
        print(f"Goblin lookup result: {result.data[:500]}...")

    def test_lookup_spell_fireball(self, real_rag_server):
        """Should find fireball spell in real database."""
        result = real_rag_server.call_tool("lookup_spell", {"name": "fireball"})

        assert result.success is True
        assert result.data is not None
        print(f"Fireball lookup result: {result.data[:500]}...")

    def test_search_rules(self, real_rag_server):
        """Should search rules successfully."""
        result = real_rag_server.call_tool(
            "search_rules",
            {"query": "flanking", "limit": 3},
        )

        assert result.success is True
        assert result.data is not None
        print(f"Flanking rules search: {result.data[:500]}...")

    def test_search_content_general(self, real_rag_server):
        """Should perform general content search."""
        result = real_rag_server.call_tool(
            "search_content",
            {"query": "healing", "limit": 5},
        )

        assert result.success is True
        assert result.data is not None
        print(f"Healing content search: {result.data[:500]}...")


class TestRealGMAgent:
    """Tests that verify GMAgent works end-to-end with real services."""

    def test_full_turn_without_tools(
        self,
        real_ollama_backend,
        real_campaign_and_session,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Should process a turn that doesn't require tools."""
        from gm_agent.agent import GMAgent
        from unittest.mock import patch

        campaign_store, session_store, campaign, session = real_campaign_and_session

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        # Use mock RAG to avoid database dependency for this test
        from tests.conftest import MockMCPServer

        with patch("gm_agent.agent.PF2eRAGServer") as mock_rag:
            mock_rag.return_value = MockMCPServer()
            agent = GMAgent(campaign_id=campaign.id, llm=real_ollama_backend)

            try:
                response = agent.process_turn("Hello, GM! Describe the scene.")

                assert response is not None
                assert len(response) > 0
                print(f"GM Response: {response}")

                # Verify turn was recorded
                current_session = session_store.get_current(campaign.id)
                assert len(current_session.turns) == 1
            finally:
                agent.close()

    def test_full_turn_with_tool_call(
        self,
        real_ollama_backend,
        real_rag_server,
        real_campaign_and_session,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Should process a turn that triggers tool usage."""
        from gm_agent.agent import GMAgent
        from unittest.mock import patch

        campaign_store, session_store, campaign, session = real_campaign_and_session

        monkeypatch.setattr("gm_agent.agent.campaign_store", campaign_store)
        monkeypatch.setattr("gm_agent.agent.session_store", session_store)

        with patch("gm_agent.agent.PF2eRAGServer") as mock_rag:
            mock_rag.return_value = real_rag_server
            agent = GMAgent(campaign_id=campaign.id, llm=real_ollama_backend, verbose=True)

            try:
                # Ask something that should trigger a tool call
                response = agent.process_turn(
                    "What are the stats for a goblin? Use your tools to look it up."
                )

                assert response is not None
                assert len(response) > 0
                print(f"GM Response with tool: {response}")

                # Check if tools were used
                current_session = session_store.get_current(campaign.id)
                turn = current_session.turns[0]
                if turn.tool_calls:
                    print(f"Tool calls made: {[tc.name for tc in turn.tool_calls]}")
            finally:
                agent.close()


class TestRealCLI:
    """Tests that verify CLI works with real services."""

    def test_test_connection_real(self):
        """Should test real Ollama connection."""
        from click.testing import CliRunner
        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["test-connection"])

        # This should at least attempt to connect
        # Exit code depends on whether server is actually available
        print(f"test-connection output: {result.output}")

    def test_search_real(self, real_rag_server):
        """Should perform real search via CLI."""
        from click.testing import CliRunner
        from cli import cli
        from unittest.mock import patch

        runner = CliRunner()

        with patch("cli.PF2eRAGServer") as mock_rag:
            mock_rag.return_value = real_rag_server

            result = runner.invoke(cli, ["search", "goblin", "--limit", "3"])

            assert result.exit_code == 0
            assert len(result.output) > 0
            print(f"Search CLI output: {result.output[:500]}...")


# NOTE: pytest hooks for --run-integration are in the root conftest.py
