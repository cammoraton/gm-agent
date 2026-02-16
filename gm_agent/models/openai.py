"""OpenAI LLM backend implementation.

Also works with OpenRouter by setting a custom base_url.
"""

import json
import logging
import time
from typing import Any, Iterator

from ..config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
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


class OpenAIBackend(LLMBackend):
    """OpenAI backend with tool calling support.

    Also works with OpenRouter by setting base_url to their API endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model or OPENAI_MODEL
        self._api_key = api_key or OPENAI_API_KEY
        self._base_url = base_url or OPENAI_BASE_URL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    @property
    def client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI backend. "
                    "Install it with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=0,  # We handle retries ourselves
            )
        return self._client

    def is_available(self) -> bool:
        """Check if the OpenAI backend is available and configured."""
        if not self._api_key:
            return False
        try:
            # Try a simple API call to verify connectivity
            self.client.models.list()
            return True
        except Exception as e:
            logger.debug(f"OpenAI availability check failed: {e}")
            return False

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        thinking: dict | None = None,
    ) -> LLMResponse:
        """Send messages and get a response.

        Args:
            messages: Conversation messages.
            tools: Optional tool definitions.
            thinking: Optional reasoning config. For OpenRouter with Anthropic
                      models, pass {"type": "enabled", "budget_tokens": N}.
                      Mapped to provider-specific params as needed.

        Includes retry logic for transient connection failures.

        Raises:
            LLMUnavailableError: If the API cannot be reached after retries.
        """
        try:
            from openai import APIConnectionError, APIStatusError, RateLimitError
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI backend. "
                "Install it with: pip install openai"
            )

        # Convert messages and tools to OpenAI format
        openai_messages = self._convert_messages(messages)
        openai_tools = [t.to_openai_format() for t in tools] if tools else None

        # Make the API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": openai_messages,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools
                if thinking:
                    # OpenRouter passes provider-specific params via extra_body
                    kwargs["extra_body"] = {"thinking": thinking}

                response = self.client.chat.completions.create(**kwargs)
                return self._parse_response(response)

            except (APIConnectionError, RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"OpenAI API connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"OpenAI API failed after {self.max_retries} attempts: {e}")

            except APIStatusError as e:
                # Authentication or other API errors shouldn't be retried
                logger.error(f"OpenAI API error: {e}")
                raise LLMUnavailableError(f"OpenAI API error: {e}")

        raise LLMUnavailableError(
            f"OpenAI unavailable after {self.max_retries} attempts. " f"Last error: {last_error}"
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI format."""
        openai_messages = []

        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Handle tool results
            if msg.role == "tool" and msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            # Handle assistant messages with tool calls
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args),  # OpenAI expects JSON string
                        },
                    }
                    for tc in msg.tool_calls
                ]

            openai_messages.append(openai_msg)

        return openai_messages

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""

        # Extract reasoning/thinking content if present
        # OpenRouter returns this for models that support reasoning
        # (Anthropic models, DeepSeek R1, o1/o3, etc.)
        thinking_text = None
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            thinking_text = reasoning

        # Parse tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                # Arguments are a JSON string that needs parsing
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=args,
                    )
                )

        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif choice.finish_reason == "length":
            finish_reason = "length"

        # Extract usage info
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            thinking=thinking_text,
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
            from openai import APIConnectionError, APIStatusError, RateLimitError
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI backend. "
                "Install it with: pip install openai"
            )

        # Convert messages and tools to OpenAI format
        openai_messages = self._convert_messages(messages)
        openai_tools = [t.to_openai_format() for t in tools] if tools else None

        # Make the streaming API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": openai_messages,
                    "stream": True,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools

                stream = self.client.chat.completions.create(**kwargs)

                accumulated_tool_calls: dict[int, dict] = {}
                finish_reason = None
                usage = {}

                for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Handle text content
                    if delta.content:
                        yield StreamChunk(delta=delta.content)

                    # Handle tool calls (can be streamed in fragments)
                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc_chunk.id or "",
                                    "name": "",
                                    "arguments": "",
                                }

                            if tc_chunk.id:
                                accumulated_tool_calls[idx]["id"] = tc_chunk.id
                            if tc_chunk.function and tc_chunk.function.name:
                                accumulated_tool_calls[idx]["name"] = tc_chunk.function.name
                            if tc_chunk.function and tc_chunk.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += (
                                    tc_chunk.function.arguments
                                )

                    # Handle finish
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                    # Extract usage if present (some providers include this)
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }

                # Parse accumulated tool calls
                tool_calls = []
                for tc_data in accumulated_tool_calls.values():
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}

                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            args=args,
                        )
                    )

                # Map OpenAI finish reasons to our format
                mapped_finish_reason = "stop"
                if tool_calls:
                    mapped_finish_reason = "tool_calls"
                elif finish_reason == "length":
                    mapped_finish_reason = "length"

                # Yield final chunk with metadata
                yield StreamChunk(
                    delta="",
                    tool_calls=tool_calls,
                    finish_reason=mapped_finish_reason,
                    usage=usage,
                )

                # Successfully streamed, return
                return

            except (APIConnectionError, RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"OpenAI API connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"OpenAI API failed after {self.max_retries} attempts: {e}")

            except APIStatusError as e:
                # Authentication or other API errors shouldn't be retried
                logger.error(f"OpenAI API error: {e}")
                raise LLMUnavailableError(f"OpenAI API error: {e}")

        raise LLMUnavailableError(
            f"OpenAI unavailable after {self.max_retries} attempts. " f"Last error: {last_error}"
        )

    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model
