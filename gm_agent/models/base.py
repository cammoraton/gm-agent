"""Abstract LLM backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from pydantic import BaseModel

from ..mcp.base import ToolDef


class LLMUnavailableError(Exception):
    """Raised when the LLM backend is unavailable."""

    pass


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    args: dict[str, Any]


class LLMResponse(BaseModel):
    """Response from an LLM backend."""

    text: str
    tool_calls: list[ToolCall] = []
    finish_reason: str = "stop"
    usage: dict[str, int] = {}
    thinking: str | None = None


class StreamChunk(BaseModel):
    """A chunk of streaming response."""

    delta: str  # Incremental text
    tool_calls: list[ToolCall] = []  # Tool calls (sent when detected)
    finish_reason: str | None = None  # Set on final chunk
    usage: dict[str, int] = {}  # Set on final chunk


class Message(BaseModel):
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        thinking: dict | None = None,
    ) -> LLMResponse:
        """Send messages and get a response, optionally with tool definitions.

        Args:
            messages: Conversation messages.
            tools: Optional tool definitions for tool use.
            thinking: Optional extended thinking config. When supported by
                      the backend, enables reasoning capture. Example:
                      {"type": "enabled", "budget_tokens": 4096}.
                      Backends that don't support thinking ignore this.
        """
        pass

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Iterator[StreamChunk]:
        """Send messages and get a streaming response.

        Default implementation falls back to non-streaming.
        Backends should override this for true streaming support.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions

        Yields:
            StreamChunk objects with incremental content

        Note:
            Tool calls may appear mid-stream. Consumer should buffer
            and handle tool execution before continuing.
        """
        # Default: non-streaming fallback
        response = self.chat(messages, tools)
        yield StreamChunk(
            delta=response.text,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and configured."""
        pass
