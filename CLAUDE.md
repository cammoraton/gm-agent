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
  - `config.py` - Configuration and environment variables
  - `context.py` - Context assembly for LLM prompts
  - `propagation.py` - Cross-system knowledge propagation (PropagationBus)
  - `mcp/` - MCP tool servers (campaign state, grounding, knowledge, etc.)
    - `base.py` - MCPServer ABC, ToolDef, ToolParameter, ToolResult, SystemToolPlugin protocol
    - `campaign_state.py` - 36 core narrative tools + plugin injection
    - `grounding.py` - Fiction-to-mechanics grounding with LLM reranking (6 tools)
    - `npc_knowledge.py` - NPC + party knowledge management (7 tools)
    - `character_runner.py` - NPC/monster behavior (6 tools)
  - `rag/` - Pathfinder content search (FTS5 + semantic search)
  - `storage/` - Persistence layer (campaigns, sessions, characters, knowledge, factions, locations, secrets, fiction tree, dialogue, history)
  - `models/` - LLM backend abstraction (Ollama, OpenAI, Anthropic, OpenRouter)
  - `prep/` - Campaign knowledge pipeline (LLM-synthesized seeding, session crunch, fiction extraction)
  - `systems/` - Multi-system game support
    - `__init__.py` - GameSystem ABC, @register_system, SYSTEM_REGISTRY
    - `pf2e/` - Pathfinder 2e (RAG, encounters, creatures, subsystems, campaign tools plugin)
    - `microscope/` - Microscope timeline generation (24 tools, seeds, oracles, mini-games)
    - `ex_novo/` - Ex Novo settlement generation (13 tools)
    - `delve/` - Delve underground kingdom generation (13 tools)
    - `ex_umbra/` - Ex Umbra dungeon generation (13 tools)
    - `shared/` - Shared components (virtual player engine, personality profiles, card decks)

## Testing
- All tests in `tests/` directory (54 test files)
- Run tests: `uv run pytest tests/`
- 1755 tests passing (14 skipped)
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
- Game-system-specific tools use `SystemToolPlugin` protocol for plugin injection into CampaignStateServer
- Game systems implement `GameSystem` ABC and register via `@register_system`
- Generation game servers (Microscope, Ex Novo, Delve, Ex Umbra) support virtual players with personality profiles and decision memory
- Prep pipeline uses `_call_llm()` + `_parse_json_array()` pattern for LLM synthesis
- Knowledge dedup via `has_similar_knowledge()` before all `add_knowledge()` calls

## LLM Backends
- Multi-backend support: Ollama, OpenAI, Anthropic, OpenRouter
- Backend selection via `LLM_BACKEND` env var (default: ollama)
- CLI `--backend` flag: `gm chat --backend openai`
- Factory pattern: `from gm_agent.models import get_backend`
- Thinking trace support: `LLMResponse.thinking` captures reasoning chains
- See `BACKENDS.md` for detailed configuration

## Key Dependencies
- Flask for REST API
- Pydantic for data models
- Multi-backend LLM support (Ollama, OpenAI, Anthropic)
- SQLite FTS5 for full-text Pathfinder content search
