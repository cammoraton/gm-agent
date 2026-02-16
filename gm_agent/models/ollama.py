"""Ollama LLM backend implementation."""

import json
import logging
import time
from typing import Any, Iterator

import httpx
import ollama

from ..config import OLLAMA_URL, OLLAMA_MODEL
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


class OllamaBackend(LLMBackend):
    """Ollama backend with tool calling support."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_URL
        self.client = ollama.Client(host=self.host)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._available = True
        self._last_error_time: float = 0.0

    def is_available(self) -> bool:
        """Check if the Ollama server is available."""
        try:
            self.client.list()
            self._available = True
            return True
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError):
            self._available = False
            return False
        except Exception:
            # Other errors might be transient
            return self._available

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        thinking: dict | None = None,
    ) -> LLMResponse:
        """Send messages and get a response.

        Includes retry logic for transient connection failures.

        Raises:
            LLMUnavailableError: If the LLM cannot be reached after retries.
        """
        # Convert messages to Ollama format
        ollama_messages = self._convert_messages(messages)

        # Convert tools to Ollama format
        ollama_tools = None
        if tools:
            ollama_tools = [tool.to_ollama_format() for tool in tools]

        # Make the API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    tools=ollama_tools,
                )
                self._available = True
                return self._parse_response(response)

            except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as e:
                last_error = e
                self._available = False
                self._last_error_time = time.time()

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Ollama connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Ollama connection failed after {self.max_retries} attempts: {e}")

            except ollama.ResponseError as e:
                # Model or API errors shouldn't be retried
                logger.error(f"Ollama API error: {e}")
                raise

        # All retries exhausted
        raise LLMUnavailableError(
            f"LLM unavailable after {self.max_retries} attempts. " f"Last error: {last_error}"
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to Ollama format."""
        ollama_messages = []

        for msg in messages:
            ollama_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Handle tool results
            if msg.role == "tool" and msg.tool_call_id:
                # Ollama expects tool results in a specific format
                ollama_msg["role"] = "tool"

            # Handle assistant messages with tool calls
            if msg.tool_calls:
                ollama_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.args,  # Ollama expects dict, not JSON string
                        },
                    }
                    for tc in msg.tool_calls
                ]

            ollama_messages.append(ollama_msg)

        return ollama_messages

    def _parse_response(self, response: dict) -> LLMResponse:
        """Parse Ollama response into LLMResponse."""
        message = response.get("message", {})
        content = message.get("content", "")

        # Parse tool calls if present
        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", {})

                # Arguments might be a string that needs parsing
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        name=func.get("name", ""),
                        args=args,
                    )
                )

        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.get("done_reason") == "length":
            finish_reason = "length"

        return LLMResponse(
            text=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
            },
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
            LLMUnavailableError: If the LLM cannot be reached after retries.
        """
        # Convert messages to Ollama format
        ollama_messages = self._convert_messages(messages)

        # Convert tools to Ollama format
        ollama_tools = None
        if tools:
            ollama_tools = [tool.to_ollama_format() for tool in tools]

        # Make the streaming API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                stream = self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    tools=ollama_tools,
                    stream=True,
                )

                accumulated_tool_calls = []
                finish_reason = "stop"
                usage = {}

                for chunk in stream:
                    message = chunk.get("message", {})

                    # Handle text content
                    content = message.get("content", "")
                    if content:
                        yield StreamChunk(delta=content)

                    # Handle tool calls
                    if "tool_calls" in message:
                        for tc in message["tool_calls"]:
                            func = tc.get("function", {})
                            args = func.get("arguments", {})

                            # Arguments might be a string that needs parsing
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}

                            accumulated_tool_calls.append(
                                ToolCall(
                                    id=tc.get("id", f"call_{len(accumulated_tool_calls)}"),
                                    name=func.get("name", ""),
                                    args=args,
                                )
                            )

                    # Check if done
                    if chunk.get("done"):
                        # Determine finish reason
                        if accumulated_tool_calls:
                            finish_reason = "tool_calls"
                        elif chunk.get("done_reason") == "length":
                            finish_reason = "length"

                        # Extract usage
                        usage = {
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": chunk.get("eval_count", 0),
                            "total_tokens": chunk.get("prompt_eval_count", 0)
                            + chunk.get("eval_count", 0),
                        }

                # Yield final chunk with metadata
                yield StreamChunk(
                    delta="",
                    tool_calls=accumulated_tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                )

                self._available = True
                return

            except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as e:
                last_error = e
                self._available = False
                self._last_error_time = time.time()

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Ollama connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Ollama connection failed after {self.max_retries} attempts: {e}")

            except ollama.ResponseError as e:
                # Model or API errors shouldn't be retried
                logger.error(f"Ollama API error: {e}")
                raise

        raise LLMUnavailableError(
            f"LLM unavailable after {self.max_retries} attempts. " f"Last error: {last_error}"
        )

    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model

    def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        response = self.client.list()
        # Handle both dict and object response formats
        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        else:
            return []

        result = []
        for model in models:
            if hasattr(model, "model"):
                result.append(model.model)
            elif isinstance(model, dict):
                result.append(model.get("name", model.get("model", "")))
            else:
                result.append(str(model))
        return result
