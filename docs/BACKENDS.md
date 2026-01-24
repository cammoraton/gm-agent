# Multi-Backend LLM Support

The GM Agent supports multiple LLM backends, allowing you to choose the best option for your use case.

## Available Backends

| Backend | Description | Default Model |
|---------|-------------|---------------|
| `ollama` | Local Ollama server | `gpt-oss:20b` |
| `openai` | OpenAI API | `gpt-4o-mini` |
| `anthropic` | Anthropic API | `claude-haiku` |
| `openrouter` | OpenRouter (multi-provider) | `openai/gpt-4o-mini` |

## Configuration

### Backend Selection

Set the `LLM_BACKEND` environment variable to choose which backend to use:

```bash
export LLM_BACKEND=ollama      # Local Ollama (default)
export LLM_BACKEND=openai      # OpenAI API
export LLM_BACKEND=anthropic   # Anthropic API
export LLM_BACKEND=openrouter  # OpenRouter (multi-provider)
```

Or use the `--backend` flag with CLI commands:

```bash
gm chat --backend openai
gm test-connection --backend anthropic
```

### Ollama (Local)

```bash
# Optional: Override defaults
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:20b
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini           # Optional
export OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL=claude-haiku       # Optional
```

### OpenRouter

OpenRouter provides access to multiple model providers through a single API. It uses the OpenAI SDK with a different base URL.

```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=openai/gpt-4o-mini  # Optional
```

## Usage

### CLI

```bash
# Use default backend (from LLM_BACKEND env)
gm chat

# Specify backend explicitly
gm chat --backend openai
gm chat --backend anthropic

# Test connection
gm test-connection --backend openai
```

### Python API

```python
from gm_agent.models.factory import get_backend, list_backends

# Get default backend
backend = get_backend()

# Get specific backend
backend = get_backend("openai")

# List available backends
backends = list_backends()  # ["ollama", "openai", "anthropic", "openrouter"]

# Check availability
if backend.is_available():
    response = backend.chat(messages, tools=tools)
```

### With Agents

```python
from gm_agent.agent import GMAgent
from gm_agent.chat import ChatAgent
from gm_agent.models.factory import get_backend

# Create agent with specific backend
llm = get_backend("anthropic")
agent = ChatAgent(llm=llm)

# Or use environment variable (LLM_BACKEND)
agent = ChatAgent()  # Uses default from env
```

## Adding New Backends

To add a new LLM backend:

### 1. Create Backend Class

Create a new file in `gm_agent/models/` (e.g., `my_backend.py`):

```python
"""My custom LLM backend implementation."""

import logging
from typing import Any

from ..mcp.base import ToolDef
from .base import LLMBackend, LLMResponse, LLMUnavailableError, Message, ToolCall

logger = logging.getLogger(__name__)


class MyBackend(LLMBackend):
    """Custom backend implementation."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.model = model or "default-model"
        self._api_key = api_key or os.getenv("MY_BACKEND_API_KEY", "")
        self.max_retries = max_retries
        self._client = None

    def is_available(self) -> bool:
        """Check if backend is available and configured."""
        return bool(self._api_key)

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> LLMResponse:
        """Send messages and get a response."""
        # Convert messages to your API format
        api_messages = self._convert_messages(messages)

        # Convert tools using ToolDef methods
        api_tools = None
        if tools:
            # Use to_openai_format() or to_anthropic_format()
            # Or implement your own format
            api_tools = [self._convert_tool(t) for t in tools]

        # Make API call with retry logic
        # ...

        # Parse response into LLMResponse
        return self._parse_response(response)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to API format."""
        # Handle different message roles:
        # - "system": System prompt
        # - "user": User input
        # - "assistant": Assistant response (may include tool_calls)
        # - "tool": Tool result (has tool_call_id)
        pass

    def _parse_response(self, response) -> LLMResponse:
        """Parse API response into LLMResponse."""
        return LLMResponse(
            text="response text",
            tool_calls=[],  # List of ToolCall objects
            finish_reason="stop",  # "stop", "tool_calls", or "length"
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model
```

### 2. Register in Factory

Update `gm_agent/models/factory.py`:

```python
def _get_backend_class(name: str) -> Type[LLMBackend]:
    if name == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend
    elif name == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend
    elif name == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend
    elif name == "my_backend":  # Add your backend
        from .my_backend import MyBackend
        return MyBackend
    else:
        raise ValueError(f"Unknown backend: {name}")


def list_backends() -> list[str]:
    return ["ollama", "openai", "anthropic", "openrouter", "my_backend"]
```

### 3. Add Configuration

Update `gm_agent/config.py`:

```python
# My Backend settings
MY_BACKEND_API_KEY = os.getenv("MY_BACKEND_API_KEY", "")
MY_BACKEND_MODEL = os.getenv("MY_BACKEND_MODEL", "default-model")
```

### 4. Update Exports

Update `gm_agent/models/__init__.py`:

```python
from .my_backend import MyBackend

__all__ = [
    # ... existing exports ...
    "MyBackend",
]
```

### 5. Add Tests

Create `tests/test_my_backend.py` with:
- Unit tests for message conversion
- Unit tests for response parsing
- Unit tests for tool format conversion
- Integration tests (skipped by default)

## Tool Format Conversion

The `ToolDef` class provides conversion methods for different APIs:

```python
tool = ToolDef(
    name="search_rules",
    description="Search Pathfinder rules",
    parameters=[
        ToolParameter(name="query", type="string", description="Search query", required=True),
    ],
)

# OpenAI/OpenRouter format
openai_tool = tool.to_openai_format()
# {
#     "type": "function",
#     "function": {
#         "name": "search_rules",
#         "description": "Search Pathfinder rules",
#         "parameters": {"type": "object", "properties": {...}, "required": [...]}
#     }
# }

# Anthropic format
anthropic_tool = tool.to_anthropic_format()
# {
#     "name": "search_rules",
#     "description": "Search Pathfinder rules",
#     "input_schema": {"type": "object", "properties": {...}, "required": [...]}
# }

# Ollama format (same as OpenAI)
ollama_tool = tool.to_ollama_format()
```

## Error Handling

All backends should:
1. Raise `LLMUnavailableError` when the service is unreachable
2. Implement retry logic with exponential backoff for transient failures
3. Provide clear error messages for authentication failures
4. Handle rate limiting gracefully

```python
from gm_agent.models.base import LLMUnavailableError

try:
    response = backend.chat(messages)
except LLMUnavailableError as e:
    print(f"LLM unavailable: {e}")
```
