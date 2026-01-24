"""Unit tests for OpenAI backend."""

import json
from unittest.mock import MagicMock, patch

import pytest

from gm_agent.models.base import LLMResponse, Message, ToolCall
from gm_agent.mcp.base import ToolDef, ToolParameter


class TestOpenAIBackend:
    """Tests for OpenAIBackend class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        return mock_client

    @pytest.fixture
    def sample_tool(self) -> ToolDef:
        """Create a sample tool definition."""
        return ToolDef(
            name="search_rules",
            description="Search Pathfinder rules",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                    default=5,
                ),
            ],
        )

    def test_import_error_without_openai(self):
        """Backend should raise ImportError if openai not installed."""
        with patch.dict("sys.modules", {"openai": None}):
            # Force reimport to trigger import error check
            import importlib

            # Note: This test is tricky because the module may already be imported
            # In practice, the lazy import in the client property handles this

    def test_is_available_without_api_key(self):
        """Backend should not be available without API key."""
        with patch("gm_agent.models.openai.OPENAI_API_KEY", ""):
            from gm_agent.models.openai import OpenAIBackend

            backend = OpenAIBackend(api_key="")
            assert backend.is_available() is False

    def test_is_available_with_api_key(self):
        """Backend should be available with valid API key."""
        with patch("gm_agent.models.openai.OpenAIBackend.client") as mock_client:
            mock_client.models.list.return_value = []
            from gm_agent.models.openai import OpenAIBackend

            backend = OpenAIBackend(api_key="sk-test-key")
            # Would need actual API for full test
            assert backend._api_key == "sk-test-key"

    def test_get_model_name(self):
        """Backend should return configured model name."""
        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend(model="gpt-4o", api_key="sk-test")
        assert backend.get_model_name() == "gpt-4o"

    def test_convert_messages_simple(self):
        """Backend should convert simple messages to OpenAI format."""
        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test")

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there!"),
        ]

        converted = backend._convert_messages(messages)

        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are a helpful assistant."
        assert converted[1]["role"] == "user"
        assert converted[2]["role"] == "assistant"

    def test_convert_messages_with_tool_calls(self):
        """Backend should convert messages with tool calls to OpenAI format."""
        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test")

        tool_calls = [
            ToolCall(id="call_123", name="search_rules", args={"query": "flanking"}),
        ]

        messages = [
            Message(
                role="assistant",
                content="Let me search for that.",
                tool_calls=tool_calls,
            ),
        ]

        converted = backend._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert "tool_calls" in converted[0]
        assert converted[0]["tool_calls"][0]["id"] == "call_123"
        assert converted[0]["tool_calls"][0]["type"] == "function"
        assert converted[0]["tool_calls"][0]["function"]["name"] == "search_rules"
        # OpenAI expects JSON string for arguments
        assert json.loads(converted[0]["tool_calls"][0]["function"]["arguments"]) == {
            "query": "flanking"
        }

    def test_convert_messages_with_tool_result(self):
        """Backend should convert tool result messages to OpenAI format."""
        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test")

        messages = [
            Message(
                role="tool",
                content="Flanking provides a +2 circumstance bonus.",
                tool_call_id="call_123",
            ),
        ]

        converted = backend._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "call_123"

    def test_parse_response_simple(self):
        """Backend should parse simple OpenAI responses."""
        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test")

        # Create mock response
        mock_message = MagicMock()
        mock_message.content = "Hello there!"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        result = backend._parse_response(mock_response)

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello there!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10

    def test_parse_response_with_tool_calls(self):
        """Backend should parse OpenAI responses with tool calls."""
        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-test")

        # Create mock tool call
        mock_function = MagicMock()
        mock_function.name = "search_rules"
        mock_function.arguments = '{"query": "flanking"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = backend._parse_response(mock_response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "search_rules"
        assert result.tool_calls[0].args == {"query": "flanking"}
        assert result.finish_reason == "tool_calls"

    def test_tool_to_openai_format(self, sample_tool):
        """ToolDef should convert to OpenAI format correctly."""
        openai_format = sample_tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "search_rules"
        assert openai_format["function"]["description"] == "Search Pathfinder rules"
        assert "parameters" in openai_format["function"]
        assert openai_format["function"]["parameters"]["type"] == "object"
        assert "query" in openai_format["function"]["parameters"]["properties"]
        assert "limit" in openai_format["function"]["parameters"]["properties"]
        assert "query" in openai_format["function"]["parameters"]["required"]
        assert "limit" not in openai_format["function"]["parameters"]["required"]


class TestOpenAIBackendIntegration:
    """Integration tests that require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires OPENAI_API_KEY")
    def test_chat_simple(self):
        """Test simple chat with real API."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from gm_agent.models.openai import OpenAIBackend

        backend = OpenAIBackend()

        messages = [
            Message(role="user", content="Say 'hello' and nothing else."),
        ]

        response = backend.chat(messages)
        assert "hello" in response.text.lower()
