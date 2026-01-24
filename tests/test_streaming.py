"""Tests for streaming LLM responses (Phase 4.3)."""

import pytest
from unittest.mock import Mock, patch

from gm_agent.models.base import StreamChunk, Message, ToolCall
from gm_agent.models.anthropic import AnthropicBackend
from gm_agent.models.openai import OpenAIBackend
from gm_agent.models.ollama import OllamaBackend
from gm_agent.mcp.base import ToolDef, ToolParameter


class TestStreamChunk:
    """Tests for StreamChunk model."""

    def test_stream_chunk_text_only(self):
        """Test creating a text-only stream chunk."""
        chunk = StreamChunk(delta="Hello, world!")
        assert chunk.delta == "Hello, world!"
        assert chunk.tool_calls == []
        assert chunk.finish_reason is None
        assert chunk.usage == {}

    def test_stream_chunk_with_tool_calls(self):
        """Test stream chunk with tool calls."""
        tool_call = ToolCall(id="call_1", name="search", args={"query": "test"})
        chunk = StreamChunk(delta="", tool_calls=[tool_call], finish_reason="tool_calls")
        assert chunk.delta == ""
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].name == "search"
        assert chunk.finish_reason == "tool_calls"

    def test_stream_chunk_final(self):
        """Test final chunk with usage."""
        chunk = StreamChunk(
            delta="",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        assert chunk.finish_reason == "stop"
        assert chunk.usage["total_tokens"] == 30


class TestAnthropicStreaming:
    """Tests for Anthropic backend streaming."""

    def test_chat_stream_text_only(self):
        """Test streaming a text-only response."""
        backend = AnthropicBackend(api_key="test-key")

        # Mock streaming response
        mock_stream = Mock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=None)
        mock_stream.response.stop_reason = "end_turn"
        mock_stream.response.usage = Mock(input_tokens=10, output_tokens=20)

        # Simulate stream events
        mock_events = [
            Mock(type="content_block_start", content_block=Mock(type="text")),
            Mock(type="content_block_delta", delta=Mock(type="text_delta", text="Hello, ")),
            Mock(type="content_block_delta", delta=Mock(type="text_delta", text="world!")),
            Mock(type="content_block_stop"),
            Mock(type="message_stop"),
        ]
        mock_stream.__iter__ = Mock(return_value=iter(mock_events))

        mock_client = Mock()
        mock_client.messages.stream.return_value = mock_stream
        backend._client = mock_client

        messages = [Message(role="user", content="Hello")]
        chunks = list(backend.chat_stream(messages))

        # Should yield text deltas
        assert len(chunks) > 0
        assert any(c.delta == "Hello, " for c in chunks)
        assert any(c.delta == "world!" for c in chunks)

        # Final chunk should have finish_reason
        assert chunks[-1].finish_reason == "stop"

    def test_chat_stream_with_tool_calls(self):
        """Test streaming with tool calls."""
        backend = AnthropicBackend(api_key="test-key")

        mock_stream = Mock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=None)
        mock_stream.response.stop_reason = "tool_use"
        mock_stream.response.usage = Mock(input_tokens=10, output_tokens=20)

        # Simulate tool use stream
        content_block = Mock()
        content_block.type = "tool_use"
        content_block.id = "call_1"
        content_block.name = "search"

        delta1 = Mock()
        delta1.type = "input_json_delta"
        delta1.partial_json = '{"query": '

        delta2 = Mock()
        delta2.type = "input_json_delta"
        delta2.partial_json = '"test"}'

        mock_events = [
            Mock(type="content_block_start", content_block=content_block),
            Mock(type="content_block_delta", delta=delta1),
            Mock(type="content_block_delta", delta=delta2),
            Mock(type="content_block_stop"),
            Mock(type="message_stop"),
        ]
        mock_stream.__iter__ = Mock(return_value=iter(mock_events))

        mock_client = Mock()
        mock_client.messages.stream.return_value = mock_stream
        backend._client = mock_client

        messages = [Message(role="user", content="Search for test")]
        tools = [
            ToolDef(
                name="search",
                description="Search tool",
                parameters=[
                    ToolParameter(
                        name="query", type="string", description="Search query", required=True
                    )
                ],
            )
        ]

        chunks = list(backend.chat_stream(messages, tools=tools))

        # Final chunk should have tool calls
        assert chunks[-1].finish_reason == "tool_calls"
        assert len(chunks[-1].tool_calls) == 1
        assert chunks[-1].tool_calls[0].name == "search"


class TestOpenAIStreaming:
    """Tests for OpenAI backend streaming."""

    def test_chat_stream_text_only(self):
        """Test streaming a text-only response."""
        backend = OpenAIBackend(api_key="test-key")

        # Mock streaming chunks
        mock_chunks = [
            Mock(
                choices=[
                    Mock(
                        delta=Mock(content="Hello, ", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            Mock(
                choices=[
                    Mock(
                        delta=Mock(content="world!", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            Mock(
                choices=[
                    Mock(
                        delta=Mock(content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            ),
        ]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        backend._client = mock_client

        messages = [Message(role="user", content="Hello")]
        chunks = list(backend.chat_stream(messages))

        # Should yield text deltas
        assert len(chunks) > 0
        assert chunks[0].delta == "Hello, "
        assert chunks[1].delta == "world!"

        # Final chunk should have finish_reason
        assert chunks[-1].finish_reason == "stop"


class TestOllamaStreaming:
    """Tests for Ollama backend streaming."""

    def test_chat_stream_text_only(self):
        """Test streaming a text-only response."""
        backend = OllamaBackend(model="llama2")

        # Mock streaming chunks
        mock_chunks = [
            {"message": {"content": "Hello, "}, "done": False},
            {"message": {"content": "world!"}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 20,
            },
        ]

        mock_client = Mock()
        mock_client.chat.return_value = iter(mock_chunks)
        backend.client = mock_client

        messages = [Message(role="user", content="Hello")]
        chunks = list(backend.chat_stream(messages))

        # Should yield text deltas
        assert len(chunks) > 0
        assert chunks[0].delta == "Hello, "
        assert chunks[1].delta == "world!"

        # Final chunk should have finish_reason
        assert chunks[-1].finish_reason == "stop"


class TestAgentStreaming:
    """Tests for agent-level streaming."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create test agent."""
        from gm_agent.agent import GMAgent
        from gm_agent.storage.campaign import campaign_store
        import uuid

        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            unique_name = f"test-campaign-{uuid.uuid4().hex[:8]}"
            campaign = campaign_store.create(unique_name, "Test Campaign")

            # Mock LLM backend with streaming support
            mock_llm = Mock()
            mock_llm.get_model_name.return_value = "test-model"

            agent = GMAgent(
                campaign_id=campaign.id,
                llm=mock_llm,
                enable_rag=False,
                enable_campaign_state=False,
                enable_character_runner=False,
            )
            yield agent

    def test_process_turn_stream_text_only(self, agent):
        """Test streaming a text-only turn."""
        # Mock streaming response
        agent.llm.chat_stream.return_value = iter(
            [
                StreamChunk(delta="Hello, "),
                StreamChunk(delta="world!"),
                StreamChunk(delta="", finish_reason="stop", usage={"total_tokens": 30}),
            ]
        )

        chunks = list(agent.process_turn_stream("Hello"))

        # Should yield text chunks (final chunk is buffered, not re-yielded)
        assert len(chunks) >= 2
        assert chunks[0].delta == "Hello, "
        assert chunks[1].delta == "world!"

    def test_process_turn_stream_with_tools(self, agent):
        """Test streaming with tool calls."""
        # Mock streaming with tool call
        tool_call = ToolCall(id="call_1", name="test_tool", args={})

        agent.llm.chat_stream.return_value = iter(
            [
                StreamChunk(delta="Thinking..."),
                StreamChunk(delta="", tool_calls=[tool_call], finish_reason="tool_calls"),
            ]
        )

        # Mock tool execution
        agent._mcp.call_tool = Mock(
            return_value=Mock(
                success=True,
                data={"result": "success"},
                to_string=Mock(return_value="success"),
            )
        )

        # Second stream after tool execution
        agent.llm.chat_stream.side_effect = [
            iter(
                [
                    StreamChunk(delta="Thinking..."),
                    StreamChunk(delta="", tool_calls=[tool_call], finish_reason="tool_calls"),
                ]
            ),
            iter(
                [
                    StreamChunk(delta="Done!"),
                    StreamChunk(delta="", finish_reason="stop"),
                ]
            ),
        ]

        chunks = list(agent.process_turn_stream("Test input"))

        # Should yield text, then continue after tool execution
        assert any(c.delta == "Thinking..." for c in chunks)
        assert any(c.delta == "Done!" for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
