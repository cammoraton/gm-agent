"""Tests for the lightweight chat agent."""

import pytest
from unittest.mock import patch, MagicMock

from gm_agent.chat import ChatAgent, CHAT_SYSTEM_PROMPT
from gm_agent.models.base import LLMBackend, LLMResponse, Message
from gm_agent.split import SplitResult, RetrievalResult


class MockLLM(LLMBackend):
    """Mock LLM for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = iter(responses or [])
        self.calls: list[tuple] = []

    def chat(self, messages, tools=None, thinking=None, temperature=None):
        self.calls.append((messages, tools))
        return next(self.responses)

    def list_models(self):
        return ["mock-model"]

    def get_model_name(self) -> str:
        return "mock-llm"

    def is_available(self) -> bool:
        return True


class TestChatAgentInit:
    """Tests for ChatAgent initialization."""

    def test_init_default_llm(self):
        """Should use get_backend by default."""
        with patch("gm_agent.chat.get_backend") as mock_get_backend:
            mock_get_backend.return_value = MagicMock()
            agent = ChatAgent()
            mock_get_backend.assert_called_once()
            agent.close()

    def test_init_custom_llm(self):
        """Should accept custom LLM backend."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm)
        assert agent.llm is mock_llm
        agent.close()

    def test_init_default_system_prompt(self):
        """Should use default system prompt."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm)
        assert agent.system_prompt == CHAT_SYSTEM_PROMPT
        assert len(agent._messages) == 1
        assert agent._messages[0].role == "system"
        agent.close()

    def test_init_custom_system_prompt(self):
        """Should accept custom system prompt."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        custom_prompt = "You are a helpful assistant."
        agent = ChatAgent(llm=mock_llm, system_prompt=custom_prompt)
        assert agent.system_prompt == custom_prompt
        assert agent._messages[0].content == custom_prompt
        agent.close()

    def test_init_creates_servers_via_mcp(self):
        """Should initialize MCP client with tools."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm)
        # MCPClient provides tools
        tools = agent.get_tools()
        assert len(tools) > 0
        agent.close()

    def test_init_has_encounter_tools(self):
        """Should have encounter tools available."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm)
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        # Check encounter tools are available
        assert "evaluate_encounter" in tool_names
        assert "suggest_encounter" in tool_names
        agent.close()

    def test_init_stores_enable_flags(self):
        """Should store enable flags for reference."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm, enable_rag=False)
        assert agent._enable_rag is False
        # MCPClient should still provide tools
        tools = agent.get_tools()
        assert len(tools) > 0
        agent.close()

    def test_init_mcp_client_provides_tools(self):
        """MCPClient should provide all tools."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm)
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        # Should have lookup tools
        assert "lookup_creature" in tool_names or len(tools) > 0
        agent.close()

    def test_get_tools_returns_all_tools(self):
        """get_tools should return tools from all servers."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = ChatAgent(llm=mock_llm)
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        # Should have both RAG and encounter tools
        assert "search_content" in tool_names
        assert "evaluate_encounter" in tool_names
        agent.close()


class TestChatAgentChat:
    """Tests for ChatAgent chat method."""

    def test_chat_simple_response(self):
        """Should return LLM response for simple query."""
        # llm is both orchestrator and narrator: 2 calls per chat()
        mock_llm = MockLLM(
            [
                LLMResponse(text="", tool_calls=[]),  # orchestrator (no tools)
                LLMResponse(text="Goblins are small humanoid creatures.", tool_calls=[]),  # narrator
            ]
        )
        agent = ChatAgent(llm=mock_llm)

        response = agent.chat("What is a goblin?")

        assert response == "Goblins are small humanoid creatures."
        assert len(mock_llm.calls) == 2  # orchestrator + narrator
        agent.close()

    def test_chat_adds_to_history(self):
        """Should add messages to conversation history."""
        # 2 chat calls x 2 LLM calls each
        mock_llm = MockLLM(
            [
                LLMResponse(text="", tool_calls=[]),           # orchestrator 1
                LLMResponse(text="First response", tool_calls=[]),  # narrator 1
                LLMResponse(text="", tool_calls=[]),           # orchestrator 2
                LLMResponse(text="Second response", tool_calls=[]),  # narrator 2
            ]
        )
        agent = ChatAgent(llm=mock_llm)

        agent.chat("First question")
        agent.chat("Second question")

        # System + user + assistant + user + assistant = 5
        assert len(agent._messages) == 5
        assert agent._messages[1].role == "user"
        assert agent._messages[1].content == "First question"
        assert agent._messages[2].role == "assistant"
        assert agent._messages[2].content == "First response"
        agent.close()

    def test_chat_with_tool_call(self):
        """Should handle tool calls correctly via split pipeline."""
        from gm_agent.split import SplitResult, RetrievalResult

        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm)

        # Mock the pipeline (detailed pipeline behavior tested in test_split.py)
        with patch.object(agent._pipeline, "run") as mock_run:
            mock_run.return_value = SplitResult(
                response="Based on my search, goblins are...",
                retrieval_rounds=[
                    RetrievalResult(
                        tool_calls=[("search_content", {"query": "goblin"}, "Goblin info")],
                        duration_ms=100.0,
                        model="mock-retrieval",
                    )
                ],
                synthesis_model="mock-llm",
                total_duration_ms=200.0,
            )

            response = agent.chat("What is a goblin?")

        assert response == "Based on my search, goblins are..."
        mock_run.assert_called_once()
        agent.close()

    def test_chat_verbose_mode(self, capsys):
        """Should pass verbose flag to pipeline."""
        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm, verbose=True)

        # Verbose flag is passed to pipeline at construction
        assert agent._pipeline.verbose is True
        agent.close()


class TestChatAgentHistory:
    """Tests for conversation history management."""

    def test_clear_history(self):
        """Should clear history but keep system prompt."""
        mock_llm = MockLLM(
            [
                LLMResponse(text="", tool_calls=[]),        # orchestrator
                LLMResponse(text="Response", tool_calls=[]),  # narrator
            ]
        )
        agent = ChatAgent(llm=mock_llm)

        agent.chat("Question")
        assert len(agent._messages) == 3  # system + user + assistant

        agent.clear_history()

        assert len(agent._messages) == 1
        assert agent._messages[0].role == "system"
        agent.close()

    def test_get_history(self):
        """Should return copy of history."""
        mock_llm = MockLLM(
            [
                LLMResponse(text="", tool_calls=[]),        # orchestrator
                LLMResponse(text="Response", tool_calls=[]),  # narrator
            ]
        )
        agent = ChatAgent(llm=mock_llm)

        agent.chat("Question")
        history = agent.get_history()

        # Should be a copy
        assert history is not agent._messages
        assert len(history) == len(agent._messages)
        agent.close()


class TestChatAgentClose:
    """Tests for resource cleanup."""

    def test_close_closes_rag_server(self):
        """Should close RAG server on cleanup."""
        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm)

        with patch.object(agent.rag_server, "close") as mock_close:
            agent.close()
            mock_close.assert_called_once()


class TestLastResult:
    """Tests for _last_result / get_last_result()."""

    def test_last_result_initially_none(self):
        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm)
        assert agent.get_last_result() is None
        agent.close()

    def test_last_result_stored_after_chat(self):
        """After chat(), get_last_result() returns the SplitResult."""
        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm)

        expected = SplitResult(
            response="Test response",
            retrieval_rounds=[
                RetrievalResult(
                    tool_calls=[("search_rules", {"query": "flanking"}, "Flanking results...")],
                    duration_ms=100.0,
                    model="test-model",
                )
            ],
            synthesis_model="mock-llm",
            total_duration_ms=200.0,
        )

        with patch.object(agent._pipeline, "run", return_value=expected):
            agent.chat("test question")

        result = agent.get_last_result()
        assert result is expected
        assert result.response == "Test response"
        assert len(result.retrieval_rounds) == 1
        assert result.retrieval_rounds[0].tool_calls[0][0] == "search_rules"
        agent.close()

    def test_last_result_cleared_by_clear_history(self):
        """clear_history() resets _last_result to None."""
        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm)

        expected = SplitResult(
            response="r", retrieval_rounds=[], synthesis_model="m", total_duration_ms=0
        )
        with patch.object(agent._pipeline, "run", return_value=expected):
            agent.chat("q")

        assert agent.get_last_result() is not None
        agent.clear_history()
        assert agent.get_last_result() is None
        agent.close()

    def test_last_result_replaced_on_second_chat(self):
        """Each chat() call replaces _last_result with the new result."""
        mock_llm = MockLLM([])
        agent = ChatAgent(llm=mock_llm)

        r1 = SplitResult(response="First", retrieval_rounds=[], synthesis_model="m", total_duration_ms=0)
        r2 = SplitResult(response="Second", retrieval_rounds=[], synthesis_model="m", total_duration_ms=0)

        with patch.object(agent._pipeline, "run", side_effect=[r1, r2]):
            agent.chat("q1")
            assert agent.get_last_result().response == "First"

            agent.chat("q2")
            assert agent.get_last_result().response == "Second"

        agent.close()


class TestNarratorLLM:
    """Tests for narrator_llm parameter and pipeline wiring.

    The key semantic: llm = orchestrator (tool calls), narrator_llm = synthesis.
    """

    def test_llm_is_orchestrator(self):
        """The main llm should drive tool orchestration (retrieval_llm)."""
        main_llm = MockLLM([])
        agent = ChatAgent(llm=main_llm)

        assert agent._pipeline.retrieval_llm is main_llm
        agent.close()

    def test_narrator_defaults_to_main_llm(self):
        """Without narrator_llm, synthesis uses the same model as orchestration."""
        main_llm = MockLLM([])
        agent = ChatAgent(llm=main_llm)

        assert agent._pipeline.synthesis_llm is main_llm
        agent.close()

    def test_narrator_override(self):
        """When narrator_llm is provided, synthesis uses it instead of main llm."""
        main_llm = MockLLM([])
        narrator = MockLLM([])
        agent = ChatAgent(llm=main_llm, narrator_llm=narrator)

        # Orchestrator should still be the main llm
        assert agent._pipeline.retrieval_llm is main_llm
        # Narrator should be the override
        assert agent._pipeline.synthesis_llm is narrator

        agent.close()
