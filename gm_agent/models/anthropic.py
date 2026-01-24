"""Anthropic LLM backend implementation."""

import logging
import time
from typing import Any, Iterator

from ..config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from ..mcp.base import ToolDef
from .base import (
    LLMBackend,
    LLMResponse,
    LLMUnavailableError,
    Message,
    StreamChunk,
    ToolCall,
)

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """Anthropic backend with tool use support."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model or ANTHROPIC_MODEL
        self._api_key = api_key or ANTHROPIC_API_KEY
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    @property
    def client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic backend. "
                    "Install it with: pip install anthropic"
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if the Anthropic backend is available and configured."""
        if not self._api_key:
            return False
        # We don't have a simple "list models" endpoint like OpenAI,
        # so just check if the key is configured
        return True

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> LLMResponse:
        """Send messages and get a response.

        Includes retry logic for transient connection failures.

        Raises:
            LLMUnavailableError: If the API cannot be reached after retries.
        """
        try:
            from anthropic import APIConnectionError, APIStatusError, RateLimitError
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic backend. "
                "Install it with: pip install anthropic"
            )

        # Extract system message (Anthropic handles it separately)
        system_msg = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append(self._convert_message(msg))

        # Convert tools to Anthropic format
        anthropic_tools = [t.to_anthropic_format() for t in tools] if tools else None

        # Make the API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": anthropic_messages,
                }
                if system_msg:
                    kwargs["system"] = system_msg
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                response = self.client.messages.create(**kwargs)
                return self._parse_response(response)

            except (APIConnectionError, RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Anthropic API connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API failed after {self.max_retries} attempts: {e}")

            except APIStatusError as e:
                # Authentication or other API errors shouldn't be retried
                logger.error(f"Anthropic API error: {e}")
                raise LLMUnavailableError(f"Anthropic API error: {e}")

        raise LLMUnavailableError(
            f"Anthropic unavailable after {self.max_retries} attempts. " f"Last error: {last_error}"
        )

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a Message to Anthropic format."""
        # Tool results are special content blocks with role "user"
        if msg.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                ],
            }

        # Assistant messages with tool calls have mixed content
        if msg.tool_calls:
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args,  # Already a dict, not JSON string
                    }
                )
            return {"role": "assistant", "content": content}

        # Regular messages
        return {"role": msg.role, "content": msg.content}

    def _parse_response(self, response) -> LLMResponse:
        """Parse Anthropic response into LLMResponse."""
        content_text = ""
        tool_calls = []

        # Anthropic responses can have mixed content blocks
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=block.input,  # Already a dict
                    )
                )

        # Determine finish reason
        finish_reason = "stop"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"

        # Extract usage info
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return LLMResponse(
            text=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Iterator[StreamChunk]:
        """Send messages and get a streaming response.

        Yields:
            StreamChunk objects with incremental content

        Raises:
            LLMUnavailableError: If the API cannot be reached after retries.
        """
        try:
            from anthropic import APIConnectionError, APIStatusError, RateLimitError
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic backend. "
                "Install it with: pip install anthropic"
            )

        # Extract system message
        system_msg = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append(self._convert_message(msg))

        # Convert tools to Anthropic format
        anthropic_tools = [t.to_anthropic_format() for t in tools] if tools else None

        # Make the streaming API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": anthropic_messages,
                }
                if system_msg:
                    kwargs["system"] = system_msg
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                # Use streaming API
                with self.client.messages.stream(**kwargs) as stream:
                    accumulated_text = ""
                    tool_calls = []
                    current_tool_use = None

                    for event in stream:
                        # Handle different event types
                        if event.type == "content_block_start":
                            if event.content_block.type == "tool_use":
                                current_tool_use = {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input": "",
                                }

                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                # Yield text delta
                                yield StreamChunk(delta=event.delta.text)
                                accumulated_text += event.delta.text

                            elif event.delta.type == "input_json_delta" and current_tool_use:
                                # Accumulate tool input JSON
                                current_tool_use["input"] += event.delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_tool_use:
                                # Parse complete tool input
                                import json
                                tool_calls.append(
                                    ToolCall(
                                        id=current_tool_use["id"],
                                        name=current_tool_use["name"],
                                        args=json.loads(current_tool_use["input"]),
                                    )
                                )
                                current_tool_use = None

                        elif event.type == "message_stop":
                            # Final chunk with metadata
                            finish_reason = "stop"
                            if stream.response.stop_reason == "tool_use":
                                finish_reason = "tool_calls"
                            elif stream.response.stop_reason == "max_tokens":
                                finish_reason = "length"

                            usage = {}
                            if stream.response.usage:
                                usage = {
                                    "prompt_tokens": stream.response.usage.input_tokens,
                                    "completion_tokens": stream.response.usage.output_tokens,
                                    "total_tokens": stream.response.usage.input_tokens
                                    + stream.response.usage.output_tokens,
                                }

                            yield StreamChunk(
                                delta="",
                                tool_calls=tool_calls,
                                finish_reason=finish_reason,
                                usage=usage,
                            )

                # Successfully streamed, return
                return

            except (APIConnectionError, RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Anthropic API connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API failed after {self.max_retries} attempts: {e}")

            except APIStatusError as e:
                # Authentication or other API errors shouldn't be retried
                logger.error(f"Anthropic API error: {e}")
                raise LLMUnavailableError(f"Anthropic API error: {e}")

        raise LLMUnavailableError(
            f"Anthropic unavailable after {self.max_retries} attempts. " f"Last error: {last_error}"
        )

    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model
