"""Unit tests for Anthropic backend."""

from unittest.mock import MagicMock, patch

import pytest

from gm_agent.models.base import LLMResponse, Message, ToolCall
from gm_agent.mcp.base import ToolDef, ToolParameter


class TestAnthropicBackend:
    """Tests for AnthropicBackend class."""

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

    def test_is_available_without_api_key(self):
        """Backend should not be available without API key."""
        with patch("gm_agent.models.anthropic.ANTHROPIC_API_KEY", ""):
            from gm_agent.models.anthropic import AnthropicBackend

            backend = AnthropicBackend(api_key="")
            assert backend.is_available() is False

    def test_is_available_with_api_key(self):
        """Backend should be available with valid API key."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test-key")
        # Anthropic doesn't have a simple list endpoint, so we just check key exists
        assert backend.is_available() is True

    def test_get_model_name(self):
        """Backend should return configured model name."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(model="claude-3-opus-20240229", api_key="sk-ant-test")
        assert backend.get_model_name() == "claude-3-opus-20240229"

    def test_convert_message_simple_user(self):
        """Backend should convert simple user messages."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        msg = Message(role="user", content="Hello!")
        converted = backend._convert_message(msg)

        assert converted["role"] == "user"
        assert converted["content"] == "Hello!"

    def test_convert_message_simple_assistant(self):
        """Backend should convert simple assistant messages."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        msg = Message(role="assistant", content="Hi there!")
        converted = backend._convert_message(msg)

        assert converted["role"] == "assistant"
        assert converted["content"] == "Hi there!"

    def test_convert_message_tool_result(self):
        """Backend should convert tool result to Anthropic format."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        msg = Message(
            role="tool",
            content="Flanking provides a +2 circumstance bonus.",
            tool_call_id="toolu_123",
        )
        converted = backend._convert_message(msg)

        # Anthropic uses "user" role with tool_result content block
        assert converted["role"] == "user"
        assert isinstance(converted["content"], list)
        assert converted["content"][0]["type"] == "tool_result"
        assert converted["content"][0]["tool_use_id"] == "toolu_123"
        assert "Flanking" in converted["content"][0]["content"]

    def test_convert_message_with_tool_calls(self):
        """Backend should convert assistant messages with tool calls."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        tool_calls = [
            ToolCall(id="toolu_123", name="search_rules", args={"query": "flanking"}),
        ]

        msg = Message(
            role="assistant",
            content="Let me search for that.",
            tool_calls=tool_calls,
        )
        converted = backend._convert_message(msg)

        assert converted["role"] == "assistant"
        assert isinstance(converted["content"], list)
        # Should have text block + tool_use block
        assert len(converted["content"]) == 2
        assert converted["content"][0]["type"] == "text"
        assert converted["content"][0]["text"] == "Let me search for that."
        assert converted["content"][1]["type"] == "tool_use"
        assert converted["content"][1]["id"] == "toolu_123"
        assert converted["content"][1]["name"] == "search_rules"
        # Anthropic expects dict, not JSON string
        assert converted["content"][1]["input"] == {"query": "flanking"}

    def test_convert_message_with_tool_calls_no_text(self):
        """Backend should handle tool calls without accompanying text."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        tool_calls = [
            ToolCall(id="toolu_123", name="search_rules", args={"query": "flanking"}),
        ]

        msg = Message(
            role="assistant",
            content="",
            tool_calls=tool_calls,
        )
        converted = backend._convert_message(msg)

        assert converted["role"] == "assistant"
        assert isinstance(converted["content"], list)
        # Should only have tool_use block (no empty text block)
        assert len(converted["content"]) == 1
        assert converted["content"][0]["type"] == "tool_use"

    def test_parse_response_simple_text(self):
        """Backend should parse simple text responses."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        # Create mock response
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello there!"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = mock_usage

        result = backend._parse_response(mock_response)

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello there!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10

    def test_parse_response_with_tool_use(self):
        """Backend should parse responses with tool use blocks."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        # Create mock tool use block
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "search_rules"
        mock_tool_block.input = {"query": "flanking"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = None

        result = backend._parse_response(mock_response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].name == "search_rules"
        assert result.tool_calls[0].args == {"query": "flanking"}
        assert result.finish_reason == "tool_calls"

    def test_parse_response_mixed_content(self):
        """Backend should handle mixed text and tool_use content."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Let me look that up."

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "search_rules"
        mock_tool_block.input = {"query": "flanking"}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = None

        result = backend._parse_response(mock_response)

        assert result.text == "Let me look that up."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_rules"

    def test_tool_to_anthropic_format(self, sample_tool):
        """ToolDef should convert to Anthropic format correctly."""
        anthropic_format = sample_tool.to_anthropic_format()

        assert anthropic_format["name"] == "search_rules"
        assert anthropic_format["description"] == "Search Pathfinder rules"
        # Anthropic uses input_schema instead of parameters
        assert "input_schema" in anthropic_format
        assert anthropic_format["input_schema"]["type"] == "object"
        assert "query" in anthropic_format["input_schema"]["properties"]
        assert "limit" in anthropic_format["input_schema"]["properties"]
        assert "query" in anthropic_format["input_schema"]["required"]
        assert "limit" not in anthropic_format["input_schema"]["required"]

    def test_system_message_extraction(self):
        """Backend should extract system message from messages list."""
        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="sk-ant-test")

        messages = [
            Message(role="system", content="You are a helpful GM."),
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there!"),
        ]

        # The chat method extracts system message, so we test the conversion logic
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append(backend._convert_message(msg))

        assert system_msg == "You are a helpful GM."
        assert len(anthropic_messages) == 2
        assert anthropic_messages[0]["role"] == "user"
        assert anthropic_messages[1]["role"] == "assistant"


class TestAnthropicBackendIntegration:
    """Integration tests that require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires ANTHROPIC_API_KEY")
    def test_chat_simple(self):
        """Test simple chat with real API."""
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from gm_agent.models.anthropic import AnthropicBackend

        backend = AnthropicBackend()

        messages = [
            Message(role="user", content="Say 'hello' and nothing else."),
        ]

        response = backend.chat(messages)
        assert "hello" in response.text.lower()
