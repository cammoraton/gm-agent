"""MCP server base classes."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """A parameter for a tool."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class ToolDef(BaseModel):
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: list[ToolParameter]
    category: str | None = None

    def _param_to_json_schema(self, param: ToolParameter) -> dict:
        """Convert a ToolParameter to JSON schema format."""
        schema = {
            "type": param.type,
            "description": param.description,
        }
        if param.default is not None:
            schema["default"] = param.default
        return schema

    def _build_parameters_schema(self) -> dict:
        """Build the JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {p.name: self._param_to_json_schema(p) for p in self.parameters},
            "required": [p.name for p in self.parameters if p.required],
        }

    def to_ollama_format(self) -> dict:
        """Convert to Ollama tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_parameters_schema(),
            },
        }

    def to_openai_format(self) -> dict:
        """Convert to OpenAI tool format.

        OpenAI uses the same format as Ollama for function tools.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_parameters_schema(),
            },
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format.

        Anthropic uses a different structure with input_schema
        instead of parameters nested under function.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._build_parameters_schema(),
        }


class ToolResult(BaseModel):
    """Result of a tool call."""

    success: bool
    data: Any = None
    error: str | None = None

    def to_string(self) -> str:
        """Convert result to string for LLM context."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, list):
            return "\n\n".join(str(item) for item in self.data)
        return str(self.data)


class MCPServer(ABC):
    """Abstract base class for MCP servers."""

    @abstractmethod
    def list_tools(self) -> list[ToolDef]:
        """List all available tools."""
        pass

    @abstractmethod
    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments."""
        pass

    def get_tool(self, name: str) -> ToolDef | None:
        """Get a tool definition by name."""
        for tool in self.list_tools():
            if tool.name == name:
                return tool
        return None
