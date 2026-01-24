"""Tests for the lightweight chat agent."""

import pytest
from unittest.mock import patch, MagicMock

from gm_agent.chat import ChatAgent, CHAT_SYSTEM_PROMPT
from gm_agent.models.base import LLMBackend, LLMResponse, Message


class MockLLM(LLMBackend):
    """Mock LLM for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = iter(responses or [])
        self.calls: list[tuple] = []

    def chat(self, messages, tools=None):
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
        mock_llm = MockLLM(
            [LLMResponse(text="Goblins are small humanoid creatures.", tool_calls=[])]
        )
        agent = ChatAgent(llm=mock_llm)

        response = agent.chat("What is a goblin?")

        assert response == "Goblins are small humanoid creatures."
        assert len(mock_llm.calls) == 1
        agent.close()

    def test_chat_adds_to_history(self):
        """Should add messages to conversation history."""
        mock_llm = MockLLM(
            [
                LLMResponse(text="First response", tool_calls=[]),
                LLMResponse(text="Second response", tool_calls=[]),
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
        """Should handle tool calls correctly."""
        from gm_agent.models.base import ToolCall

        mock_llm = MockLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="1", name="search_content", args={"query": "goblin"})],
                ),
                LLMResponse(text="Based on my search, goblins are...", tool_calls=[]),
            ]
        )

        agent = ChatAgent(llm=mock_llm)

        # Mock the RAG server call
        with patch.object(agent.rag_server, "call_tool") as mock_call:
            from gm_agent.mcp.base import ToolResult

            mock_call.return_value = ToolResult(
                success=True, data={"results": [{"name": "Goblin", "type": "creature"}]}
            )

            response = agent.chat("What is a goblin?")

        assert response == "Based on my search, goblins are..."
        mock_call.assert_called_once()
        agent.close()

    def test_chat_verbose_mode(self, capsys):
        """Should print tool calls in verbose mode."""
        from gm_agent.models.base import ToolCall

        mock_llm = MockLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="1", name="search_content", args={"query": "test"})],
                ),
                LLMResponse(text="Result", tool_calls=[]),
            ]
        )

        agent = ChatAgent(llm=mock_llm, verbose=True)

        with patch.object(agent.rag_server, "call_tool") as mock_call:
            from gm_agent.mcp.base import ToolResult

            mock_call.return_value = ToolResult(success=True, data={"results": []})

            agent.chat("Test query")

        captured = capsys.readouterr()
        assert "Tool calls:" in captured.out
        assert "search_content" in captured.out
        agent.close()


class TestChatAgentHistory:
    """Tests for conversation history management."""

    def test_clear_history(self):
        """Should clear history but keep system prompt."""
        mock_llm = MockLLM(
            [
                LLMResponse(text="Response", tool_calls=[]),
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
                LLMResponse(text="Response", tool_calls=[]),
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
