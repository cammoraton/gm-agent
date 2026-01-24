# Claude Code Project Guidelines

## Package Management
- Always use `uv` for Python package management
- Never use `pip install` directly
- Use `uv run` to execute commands within the virtual environment
- Use `uv sync` to install dependencies from pyproject.toml

## Project Structure
- `cli.py` - Command-line interface entry point
- `api.py` - REST API (Flask)
- `gm_agent/` - Core library package
  - `agent.py` - Full GMAgent with campaign state
  - `chat.py` - Lightweight ChatAgent without campaign state
  - `mcp/` - MCP tool servers (RAG, dice, notes, encounters, etc.)
  - `rag/` - Pathfinder content search (FTS5 + semantic search)
  - `storage/` - Persistence layer (campaigns, sessions, characters)
  - `models/` - LLM backend abstraction

## Testing
- All tests in `tests/` directory
- Run tests: `uv run pytest tests/`
- 729 tests with comprehensive coverage (88%)
- Use fixtures from `tests/conftest.py`

## Linting
- Format code: `uv run black .`
- Run linter: `uv run pylint gm_agent`
- Target pylint score: 9.0+
- Configuration in pyproject.toml

## Full Automation Mode
- `gm_agent/game_loop.py` - GameLoopController for event-driven automation
- Responds to `playerChat` and `combatTurn` events from Foundry VTT
- Per-player message batching with configurable window (default 2s)
- NPC turn cooldown rate limiting (separate from player batching)
- Error threshold auto-disable (default: 5 consecutive errors)
- Stats tracking: response counts, processing time, batch sizes, errors
- Coordinates with AI Combat Assistant for NPC turns
- `reset_stats()` method for monitoring resets

## Code Patterns
- Use Pydantic models for data validation (`storage/schemas.py`)
- MCP servers extend `MCPServer` base class
- Tools use `ToolDef` with `ToolParameter` objects (not JSON schema dicts)
- Return `ToolResult(success=True/False, data=..., error=...)`

## LLM Backends
- Multi-backend support: Ollama, OpenAI, Anthropic, OpenRouter
- Backend selection via `LLM_BACKEND` env var (default: ollama)
- CLI `--backend` flag: `gm chat --backend openai`
- Factory pattern: `from gm_agent.models import get_backend`
- See `BACKENDS.md` for detailed configuration

## Key Dependencies
- Flask for REST API
- Pydantic for data models
- Multi-backend LLM support (Ollama, OpenAI, Anthropic)
- SQLite FTS5 for full-text Pathfinder content search
