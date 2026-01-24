# GM Agent

An AI-powered Game Master for Pathfinder 2nd Edition (Remaster). GM Agent can run full tabletop RPG sessions autonomously, handling narrative, combat, NPCs, and rules adjudication. Supports multiple LLM backends (Ollama, OpenAI, Anthropic, OpenRouter) with Retrieval-Augmented Generation (RAG) for accurate rules and lore.

## Why GM Agent?

- **Playtest Your Scenarios**: Test adventures and encounters without using valuable player time
- **Address the GM Shortage**: GMs can finally experience the game as players
- **Solo Play**: Run through modules or homebrew content on your own schedule
- **GM Prep Aid**: Use chat mode for rules lookups, encounter building, and session planning
- **VTT Automation**: Connect to Foundry VTT for fully automated game sessions

## Features

### Full GM Capabilities
- **Autonomous Game Sessions**: Run complete RPG sessions with narrative, dialogue, and combat
- **NPC & Monster Behavior**: Consistent character portrayals with personality profiles
- **Combat Management**: Tactical decision-making for NPCs with Foundry VTT integration
- **Dynamic Narration**: Contextual storytelling that adapts to player actions

### Campaign & Session Management
- **Campaign Tracking**: Manage campaigns, party members, story arcs, and history
- **Session Storage**: All turns stored as structured JSON with complete metadata (player input, GM response, tool calls, timing, model used)
- **Session Continuity**: Automatic rolling summaries maintain context across sessions
- **Scene State**: Track locations, NPCs present, time of day, and conditions
- **Event Logging**: Full-text search over campaign history
- **Training Data Collection**: Sessions are ready for fine-tuning datasets and pattern mining

### Pathfinder 2e Integration
- **RAG-Powered Rules**: Look up creatures, spells, items, feats, and rules from official content
- **Encounter Evaluation**: Calculate XP budgets, threat levels, and get tactical advice
- **Dice Rolling**: Full PF2e mechanics including fortune/misfortune and degree of success
- **Golarion Lore**: Search world lore, locations, deities, and organizations

### Deployment Options
- **CLI Mode**: Interactive terminal sessions for local play
- **REST API**: Full-featured API with optional JWT authentication
- **Chat Mode**: Lightweight assistant for rules lookups and GM prep
- **Foundry VTT Integration**: Bidirectional communication via Socket.IO
- **Full Automation Mode**: Agent handles player chat and NPC combat turns automatically
- **Multi-Backend LLM Support**: Ollama, OpenAI, Anthropic, and OpenRouter
- **Distributed Architecture**: Docker Compose deployment with Celery workers

## Advanced Features

### NPC System
- **Character Profiles**: Build NPC profiles from RAG data with personality, speech patterns, and goals
- **Relationships**: Track bidirectional relationships between NPCs and PCs with trust levels and attitudes
- **Dialogue History**: SQLite FTS5-indexed conversation logs with searchable past statements
- **Memory & Knowledge**: Redis-backed NPC knowledge with conditional sharing (trust, persuasion DC, duress)
- **Factions**: Organization membership with shared knowledge, goals, and party reputation

### World Simulation
- **Location-based Awareness**: NPCs automatically know location-specific common knowledge
- **Rumor Mill**: Information spreads between locations with accuracy degradation and propagation tracking
- **Secret & Revelation Tracking**: Track plot-critical secrets with automatic consequence triggering
- **Dynamic Information Flow**: Time-based propagation through connected locations and chatty NPCs

### Automation Enhancements
- **Configurable Prompts**: Customize per-campaign prompts for player chat and NPC turns with template variables
- **Dry Run Mode**: Test automation behavior without posting to Foundry
- **Tool Usage Analytics**: Track tool calls per turn with success/failure rates and performance metrics
- **Message Queuing**: Redis-backed event queue with priority ordering and depth limits
- **Async Processing**: Celery-based async turn processing with progress notifications
- **Response Streaming**: Server-Sent Events (SSE) for real-time LLM response streaming
- **Per-Player Batching**: Configurable batch windows (default 2s) to group rapid player actions

### Session Replay & Testing
- **Session Replay**: Replay recorded sessions at any speed (instant, real-time, 10x) for debugging and testing
- **Model Comparison**: Side-by-side comparison of different LLM models with performance metrics
- **Performance Metrics**: Track response time, token usage, and error rates per model
- **Training Data**: All sessions stored as structured JSON for fine-tuning and pattern mining
- **CLI & API Access**: Both `gm replay` command and REST endpoints available

**Note:** Current replay is useful for debugging, performance comparison, and basic testing. Rigorous model efficacy evaluation (with deterministic outputs, evaluation metrics, and ground truth comparison) is a future enhancement.

## Architecture

```
LOCAL MODE (standalone CLI)
┌─────────────┐
│     CLI     │──────► Direct MCP execution (in-process servers)
└─────────────┘

REMOTE MODE (Docker Compose)
┌─────────────┐      ┌─────────────────────────┐      ┌─────────────────┐
│     CLI     │ HTTP │          API            │Celery│   MCP Workers   │
│  (client)   │─────►│  (MCP Server facade)    │─────►│  (tool executors)│
└─────────────┘      │  - routes tool calls    │      │  - pf2e-rag     │
                     │  - Foundry in-process   │      │  - dice         │
                     └─────────────────────────┘      │  - campaign     │
                                                      │  - notes (Redis)│
                                                      └─────────────────┘
```

### Project Structure

```
gm-agent/
├── cli.py                    # Command-line interface
├── api.py                    # Flask REST API with MCP endpoints
├── wsgi.py                   # WSGI entry point for production
├── docker-compose.yml        # Docker Compose deployment
├── data/                     # Data directory (gitignored)
│   └── pathfinder_search.db  # Search database (built externally)
├── gm_agent/
│   ├── agent.py              # Core GMAgent class (full campaign mode)
│   ├── chat.py               # Lightweight ChatAgent (no campaign state)
│   ├── config.py             # Configuration and environment variables
│   ├── context.py            # Context assembly for LLM prompts
│   ├── summarizer.py         # Rolling session summaries
│   ├── replay.py             # Session replay and model comparison
│   ├── game_loop.py          # Full automation mode controller
│   ├── event_queue.py        # Redis-backed event queuing
│   ├── rumor_mill.py         # Rumor propagation engine
│   ├── celery_app.py         # Celery application configuration
│   ├── tasks.py              # Celery tasks (chat, summaries, async turns)
│   ├── mcp_tasks.py          # Celery tasks for MCP tool execution
│   ├── models/
│   │   ├── base.py           # LLM backend interface
│   │   ├── factory.py        # Backend factory (get_backend)
│   │   ├── ollama.py         # Ollama integration
│   │   ├── openai_backend.py # OpenAI integration
│   │   ├── anthropic_backend.py # Anthropic integration
│   │   └── openrouter.py     # OpenRouter integration
│   ├── rag/                   # Pathfinder search (RAG)
│   │   ├── __init__.py       # Module exports
│   │   └── search.py         # PathfinderSearch (FTS5 + semantic)
│   ├── mcp/                   # MCP (Model Context Protocol) servers
│   │   ├── base.py           # Base classes for tools
│   │   ├── registry.py       # Server registry and tool routing
│   │   ├── client.py         # Unified MCP client (local/remote modes)
│   │   ├── pf2e_rag.py       # RAG tools wrapper
│   │   ├── encounter.py      # Encounter evaluation tools
│   │   ├── dice.py           # Dice rolling tools
│   │   ├── notes.py          # GM notes tools
│   │   ├── redis_notes.py    # Redis-backed notes (distributed)
│   │   ├── campaign_state.py # Scene/event management
│   │   ├── character_runner.py # NPC/monster behavior
│   │   └── foundry_vtt.py    # Foundry VTT integration (33 tools)
│   └── storage/
│       ├── schemas.py        # Pydantic data models (Campaign, Session, Character, Relationship, Faction, Location, Rumor, Secret, Knowledge)
│       ├── campaign.py       # Campaign persistence
│       ├── session.py        # Session persistence
│       ├── characters.py     # Character profiles
│       ├── dialogue.py       # NPC dialogue history (SQLite FTS5)
│       ├── knowledge.py      # NPC memory & knowledge (Redis)
│       ├── factions.py       # Faction & organization tracking
│       ├── locations.py      # Location-based awareness
│       ├── rumors.py         # Rumor storage and tracking
│       ├── secrets.py        # Secret & revelation tracking
│       └── history.py        # Campaign event history (SQLite FTS5)
└── tests/                    # Test suite (740+ tests, 88% coverage)
```

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- One of: [Ollama](https://ollama.ai/), OpenAI API key, Anthropic API key, or OpenRouter API key
- Pathfinder search database (`data/pathfinder_search.db`) - see [RAG Database](#rag-database) below

### Setup

```bash
# Clone and enter directory
cd gm-agent

# Install dependencies
uv sync

# Install with test dependencies
uv sync --all-extras

# Copy environment file
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Create a `.env` file with the following settings:

```bash
# =============================================================================
# LLM Backend Selection
# =============================================================================
# Available backends: ollama, openai, anthropic, openrouter
LLM_BACKEND=ollama

# =============================================================================
# Ollama Settings (if using ollama backend)
# =============================================================================
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:latest

# =============================================================================
# OpenAI Settings (if using openai backend)
# =============================================================================
# OPENAI_API_KEY=sk-your-api-key-here
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1

# =============================================================================
# Anthropic Settings (if using anthropic backend)
# =============================================================================
# ANTHROPIC_API_KEY=sk-ant-your-api-key-here
# ANTHROPIC_MODEL=claude-haiku

# =============================================================================
# OpenRouter Settings (if using openrouter backend)
# =============================================================================
# OPENROUTER_API_KEY=sk-or-your-api-key-here
# OPENROUTER_MODEL=openai/gpt-4o-mini

# =============================================================================
# Path Settings
# =============================================================================
CAMPAIGNS_DIR=./data/campaigns
RAG_DB_PATH=./data/pathfinder_search.db

# =============================================================================
# Context Settings
# =============================================================================
MAX_RECENT_TURNS=15
MAX_CONTEXT_TOKENS=8000
TURNS_BETWEEN_SUMMARIES=10

# =============================================================================
# MCP Mode Settings
# =============================================================================
# "local" = Direct in-process server execution (default)
# "remote" = CLI -> API -> Celery -> Workers (for Docker deployment)
MCP_MODE=local
MCP_API_URL=http://localhost:5000

# =============================================================================
# API Authentication (optional)
# =============================================================================
API_AUTH_ENABLED=false
JWT_SECRET_KEY=your-secret-key
API_USERNAME=admin
API_PASSWORD=changeme
```

## CLI Commands

### Campaign Management

```bash
# Create a campaign
gm campaign create "Abomination Vaults" -b "A dungeon crawl in Otari"

# List campaigns
gm campaign list

# Show campaign details
gm campaign show abomination-vaults

# Update campaign
gm campaign update abomination-vaults --arc "Level 1: Gauntlight Ruins"

# Delete campaign
gm campaign delete abomination-vaults
```

### Session Management

```bash
# Start interactive session (REPL)
gm session start abomination-vaults

# With verbose mode (shows tool calls)
gm session start abomination-vaults -v

# Specify LLM backend
gm session start abomination-vaults --backend openai

# List sessions
gm session list abomination-vaults

# Show session details
gm session show abomination-vaults session-2024-01-15
```

### Session Commands (during REPL)

```
/end     - End and save the session
/scene   - Update scene state (location, time, NPCs, conditions)
/status  - Show current scene
/help    - Show available commands
```

### Chat Mode

Lightweight assistant for rules lookups and GM prep without campaign state.

```bash
# Start chat
gm chat

# With verbose mode
gm chat -v

# With specific backend
gm chat --backend anthropic
```

Chat commands:
```
/clear  - Clear conversation history
/tools  - List available tools
/quit   - Exit chat
/help   - Show commands
```

### Utilities

```bash
# Search Pathfinder content directly
gm search "fireball"
gm search "goblin" -t creature
gm search "longsword" -l 3

# Test LLM connection
gm test-connection

# Replay a recorded session
gm replay my-campaign session-123                    # Instant replay
gm replay my-campaign session-123 --speed 1.0        # Real-time speed
gm replay my-campaign session-123 --speed 10.0       # 10x faster
gm replay my-campaign session-123 --backend openai --model gpt-4

# Compare different models on same session
gm compare my-campaign session-123 --models "ollama:llama3,openai:gpt-4,anthropic:claude-3-sonnet"

# Start REST API server
gm server
gm server --host 0.0.0.0 --port 8080
gm server --auth  # Enable JWT authentication
```

## Docker Deployment

### Quick Start

```bash
# From the infrastructure directory
cd infrastructure

# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Run CLI commands in Docker
docker compose --profile cli run --rm cli campaign list
docker compose --profile cli run --rm cli chat
```

### Services

| Service | Description | Port |
|---------|-------------|------|
| nginx | Reverse proxy with WebSocket support | 80 |
| api | Flask + Socket.IO API server | 5000 |
| celery-worker | Async task processing (4 workers) | - |
| redis | Task broker & result backend | 6379 |
| cli | Admin commands (profile-activated) | - |

### Docker Environment

The CLI service in Docker automatically uses remote mode (`MCP_MODE=remote`), routing tool calls through the API to Celery workers for distributed execution.

## REST API

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/auth/status` | Check auth status |
| POST | `/api/auth/token` | Get JWT token |
| GET | `/api/campaigns` | List campaigns |
| POST | `/api/campaigns` | Create campaign |
| GET | `/api/campaigns/{id}` | Get campaign |
| PUT | `/api/campaigns/{id}` | Update campaign |
| DELETE | `/api/campaigns/{id}` | Delete campaign |
| GET | `/api/campaigns/{id}/sessions` | List sessions |
| POST | `/api/campaigns/{id}/sessions/start` | Start session |
| POST | `/api/campaigns/{id}/sessions/turn` | Process turn |
| POST | `/api/campaigns/{id}/sessions/end` | End session |
| GET | `/api/campaigns/{id}/sessions/{sid}` | Get session |
| GET | `/api/campaigns/{id}/characters` | List characters |
| POST | `/api/campaigns/{id}/characters` | Create character |
| GET | `/api/campaigns/{id}/characters/{name}` | Get character |
| DELETE | `/api/campaigns/{id}/characters/{name}` | Delete character |
| POST | `/api/chat` | Stateless chat |

### MCP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/mcp/servers` | List available MCP servers |
| GET | `/api/mcp/tools` | List all tools with metadata |
| POST | `/api/mcp/call` | Execute tool (sync or async) |
| GET | `/api/mcp/task/{id}` | Check async task status |

#### Execute Tool Example

```bash
# Synchronous execution
curl -X POST http://localhost:5000/api/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "roll_dice", "args": {"expression": "1d20+5"}}'

# Async execution (returns task_id)
curl -X POST http://localhost:5000/api/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "lookup_creature", "args": {"name": "goblin"}, "async": true}'

# Check task status
curl http://localhost:5000/api/mcp/task/{task_id}
```

### Swagger Documentation

When the server is running, visit `/api/docs/` for interactive API documentation.

### Production Deployment

```bash
# Gunicorn (single worker for WebSocket)
gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:5000 wsgi:app

# With Docker Compose (recommended)
cd infrastructure && docker compose up -d
```

## MCP Tools

### Server Registry

| Server | Stateless | Celery Eligible | Notes |
|--------|-----------|-----------------|-------|
| pf2e-rag | Yes | Yes | Direct instantiation |
| dice | Yes | Yes | Direct instantiation |
| encounter | Yes | Yes | Direct instantiation |
| notes | No | Yes | Redis-backed in remote mode |
| campaign-state | No | Yes | Requires campaign_id |
| character-runner | No | Yes | Requires campaign_id + LLM |
| foundry-vtt | No | **No** | Must stay in API (WebSocket) |

### RAG Tools (7 tools)

| Tool | Description |
|------|-------------|
| `lookup_creature` | Look up creature stats by name |
| `lookup_spell` | Look up spell details by name |
| `lookup_item` | Look up item/equipment by name |
| `lookup_location` | Look up location/city lore by name |
| `search_rules` | Search rules, conditions, mechanics |
| `search_lore` | Search world lore, history, nations |
| `search_content` | General search with type filters |

### Encounter Tools (4 tools)

| Tool | Description |
|------|-------------|
| `evaluate_encounter` | Calculate XP, threat level, warnings |
| `suggest_encounter` | Get creature composition suggestions |
| `calculate_creature_xp` | XP value of single creature |
| `get_encounter_advice` | GM advice on encounter design |

### Dice Tools (6 tools)

| Tool | Description |
|------|-------------|
| `roll_dice` | Roll dice (e.g., "2d6+4") |
| `roll_multiple` | Roll multiple expressions |
| `roll_fortune` | Roll twice, take higher |
| `roll_misfortune` | Roll twice, take lower |
| `check_result` | Determine degree of success vs DC |
| `roll_check` | Combined roll + check in one step |

### Notes Tools (6 tools)

| Tool | Description |
|------|-------------|
| `add_note` | Add note with tags/importance |
| `list_notes` | List notes (filter by tag/importance) |
| `search_notes` | Search notes by content/tags |
| `delete_note` | Delete a note |
| `update_note` | Update note content/tags |
| `get_important_notes` | Get high/critical importance notes |

### Campaign State Tools (5 tools)

| Tool | Description |
|------|-------------|
| `update_scene` | Update current scene state |
| `log_event` | Record significant events |
| `search_history` | Search session history |
| `get_recent_events` | Get recent events |
| `get_scene` | Get current scene state |

### Character Runner Tools (3 tools)

| Tool | Description |
|------|-------------|
| `embody_character` | Generate in-character dialogue |
| `get_character_action` | Determine NPC/monster action |
| `list_characters` | List available character profiles |

## Foundry VTT Integration

The GM Agent integrates with Foundry VTT through a companion module (`pf2e-gm-agent`) that enables bidirectional communication.

### Communication Modes

GM Agent supports two communication modes:

| Mode | Description | Best For |
|------|-------------|----------|
| **WebSocket** (default) | Foundry connects TO gm-agent | Local network, same machine |
| **Polling** | gm-agent polls Foundry | Cloud hosting, firewalls, NAT |

**WebSocket Mode (Default):**
```
Foundry VTT ────WebSocket────► gm-agent
(client)                      (server)
```

**Polling Mode:**
```
Foundry VTT ◄────HTTP/REST────  gm-agent
(server)                       (client)
```

For detailed deployment configurations, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

### Setup

1. **Start the GM Agent API server:**
   ```bash
   gm server
   # or with Docker
   cd infrastructure && docker compose up -d
   ```

2. **Install the Foundry module** from the `pf2e-gm-agent` directory

3. **Configure the module** in Foundry VTT:
   - Enable "GM Agent Bridge"
   - Set "WebSocket URL" to your GM Agent server (default: `http://localhost:5000`)
   - Set "Campaign ID" to match your campaign
   - Optionally enable "Full Automation Mode"

4. **For remote/cloud Foundry** (polling mode):
   - In Foundry: Enable "Enable Polling API" and set an API key
   - In gm-agent `.env`:
     ```bash
     FOUNDRY_MODE=polling
     FOUNDRY_POLL_URL=https://your-foundry.example.com
     FOUNDRY_API_KEY=your-api-key
     ```

### Full Automation Mode

When automation is enabled, the GM Agent automatically:

- **Responds to player chat messages** - Players describe actions, the agent narrates consequences
- **Handles NPC combat turns** - Decides actions and narrates them dramatically
- **Coordinates with AI Combat Assistant** - If ACA is controlling an NPC, the agent provides narration only

#### Enabling Automation

**Via Foundry Settings:**
1. Open Module Settings -> GM Agent
2. Enable "Full Automation Mode"
3. Set a Campaign ID

**Via API:**
```bash
curl -X POST http://localhost:5000/api/campaigns/{campaign_id}/automation \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

#### Automation Stats & Monitoring

Get automation status and stats via API:

```bash
curl http://localhost:5000/api/campaigns/{campaign_id}/automation
```

Response includes:
```json
{
  "enabled": true,
  "campaign_id": "my-campaign",
  "stats": {
    "enabled": true,
    "started_at": 1706200000.0,
    "uptime_seconds": 3600.0,
    "response_count": 42,
    "player_chat_count": 35,
    "npc_turn_count": 7,
    "batched_message_count": 58,
    "average_batch_size": 1.66,
    "pending_batches": 0,
    "pending_messages": 0,
    "oldest_batch_age_seconds": null,
    "error_count": 2,
    "consecutive_errors": 0,
    "max_consecutive_errors": 5,
    "total_processing_time_ms": 45230.5,
    "last_response_time": 1706203500.0,
    "seconds_since_last_response": 100.0,
    "batch_window_seconds": 2.0,
    "max_batch_size": 5,
    "cooldown_seconds": 2.0
  }
}
```

Stats can be reset via POST to `/api/campaigns/{campaign_id}/automation/stats/reset`.

### Foundry VTT Tools (33 tools)

All Foundry tools are prefixed with `foundry_` and have `category="foundry"`.

#### Scene & Combat Tools

| Tool | Description |
|------|-------------|
| `foundry_get_scene` | Get current scene data (tokens, lighting, walls) |
| `foundry_get_actors` | Get actor data (HP, conditions, level) |
| `foundry_get_combat_state` | Get combat tracker state |
| `foundry_is_combat_active` | Check if combat is active |
| `foundry_get_combat_summary` | Get narrative combat summary |
| `foundry_update_token` | Move, hide, or update a token |
| `foundry_start_combat` | Start combat with specified tokens |
| `foundry_end_combat` | End current combat |
| `foundry_advance_turn` | Advance to next combatant |

#### Combat Resolution Tools

| Tool | Description |
|------|-------------|
| `foundry_apply_damage` | Apply damage (supports damage types) |
| `foundry_heal` | Restore hit points |
| `foundry_apply_condition` | Apply condition (frightened, poisoned, etc.) |
| `foundry_remove_condition` | Remove a condition |

#### Token Management Tools

| Tool | Description |
|------|-------------|
| `foundry_spawn_token` | Spawn token at position |
| `foundry_remove_token` | Remove token from scene |

#### Communication Tools

| Tool | Description |
|------|-------------|
| `foundry_create_chat` | Post message to chat |
| `foundry_narrate` | Post narrative description as GM |
| `foundry_whisper` | Send private message to player |
| `foundry_show_journal` | Display journal entry |
| `foundry_roll_check` | Request skill/save roll |

#### AI Combat Assistant Tools

| Tool | Description |
|------|-------------|
| `foundry_get_aca_state` | Get ACA module state |
| `foundry_get_aca_turn_state` | Get ACA turn state for actor |
| `foundry_set_aca_notes` | Set tactical notes for NPC |
| `foundry_trigger_aca_suggestion` | Trigger ACA action suggestion |
| `foundry_get_aca_game_state` | Get ACA's view of combat |
| `foundry_set_aca_designation` | Set actor's ACA designation |

#### Exploration Mode Tools

| Tool | Description |
|------|-------------|
| `foundry_set_exploration_activity` | Set exploration activity |
| `foundry_get_exploration_state` | Get party exploration state |
| `foundry_roll_secret_check` | Roll secret GM check |

#### Rest & Recovery Tools

| Tool | Description |
|------|-------------|
| `foundry_take_rest` | Short or long rest |
| `foundry_refocus` | Recover focus point |

#### Time Tracking Tools

| Tool | Description |
|------|-------------|
| `foundry_advance_time` | Advance in-game time |
| `foundry_get_time` | Get current in-game time |

## LLM Backends

### Supported Backends

| Backend | Description | Model Examples |
|---------|-------------|----------------|
| `ollama` | Local LLM via Ollama | qwen3:latest, llama3.2, mistral |
| `openai` | OpenAI API | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| `anthropic` | Anthropic API | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| `openrouter` | OpenRouter API | Any model on OpenRouter |

### Backend Selection

```bash
# Via environment variable
export LLM_BACKEND=openai

# Via CLI flag
gm chat --backend anthropic
gm session start my-campaign --backend openrouter
```

### Backend-Specific Configuration

```bash
# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:latest

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, for compatible APIs

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-haiku-20240307

# OpenRouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-3-haiku
```

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=gm_agent --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_dice.py -v

# Run specific test
uv run pytest tests/test_dice.py::TestRollDice::test_roll_with_modifier -v

# Run integration tests (requires real Ollama)
uv run pytest --run-integration
```

### Test Coverage

The test suite includes 740+ tests with 88% coverage:

- **Schemas** (`test_schemas.py`): Pydantic model validation
- **Context** (`test_context.py`): Context assembly for LLM
- **Storage** (`test_storage.py`): Campaign/session persistence
- **MCP** (`test_mcp.py`): MCP server tools
- **Agent** (`test_agent.py`): GMAgent integration
- **API** (`test_api.py`): REST API endpoints
- **CLI** (`test_cli.py`): Command-line interface
- **Chat** (`test_chat.py`): ChatAgent
- **Dice** (`test_dice.py`): Dice rolling mechanics
- **Notes** (`test_notes.py`): Notes management
- **Encounter** (`test_encounter.py`): Encounter evaluation
- **Campaign State** (`test_campaign_state.py`): Scene/event tools
- **Character Runner** (`test_character_runner.py`): NPC behaviors
- **Summarizer** (`test_summarizer.py`): Rolling summaries
- **Foundry VTT** (`test_foundry_vtt.py`): Bridge and 33 Foundry tools
- **Game Loop** (`test_game_loop.py`): Full automation mode
- **History** (`test_history.py`): History search index
- **Backend Tests**: Ollama, OpenAI, Anthropic backends

## RAG Database

GM Agent uses a SQLite database with FTS5 full-text search for looking up Pathfinder 2e content (creatures, spells, items, rules, lore).

### ⚠️ Copyright Notice: Database Cannot Be Distributed

**The RAG database (`pathfinder_search.db`) is NOT included in this repository and cannot be legally distributed.**

**Why Not:**
The database contains copyrighted Paizo content that has been:
- Extracted from PDFs (adventure text, creature stats, spell descriptions, etc.)
- Chunked into searchable segments
- Indexed with embeddings for semantic search

This is **NOT transformative use** - it's just repackaging copyrighted material in a different format. Distributing it would be a clear copyright violation.

**What's Inside:**
- Text chunks from copyrighted Paizo books (Core Rulebook, Bestiaries, Adventure Paths, etc.)
- Vector embeddings of that copyrighted text
- Metadata and structured data extracted from copyrighted sources

**Exceptions:**
- **SRD Content**: System Reference Document content is open under the OGL and CAN be shared
- **Summaries**: AI-generated summaries might be transformative fair use, but we're being conservative and not distributing these either

### Architecture

The RAG system is split into two components:

1. **Extraction** (private, not included) - Processes PDFs and source content to extract text, clean watermarks, and structure data. This lives outside the repository due to copyright.

2. **Search/Indexing** (included in `gm_agent/rag/`) - The `PathfinderSearch` class provides FTS5 full-text search and optional semantic search via sentence-transformers. This code is fully open source.

### Database Schema

The search database structure:

- `content` table with FTS5 index for full-text search
- Columns: `type`, `name`, `level`, `traits`, `source`, `content`, `url`
- Content types: `creature`, `spell`, `item`, `feat`, `class`, `ancestry`, `rule`, `location`, `deity`, `npc`, `lore`
- Optional: `embeddings` table for semantic search vectors

### Building Your Own Database

**For Personal Use**: You can build your own database from legally-obtained Paizo PDFs.

**Options:**

1. **Build from PDFs**:
   - Purchase Paizo PDFs legally
   - Use extraction tools (not included in this repo)
   - Build your own personal database
   - Use for your own games

2. **SRD Only**:
   - Build a database with only OGL/SRD content
   - Significantly reduced capability but legally distributable
   - Good for basic rules lookups

3. **Archives of Nethys**:
   - Public website can be used as a reference
   - Respect their terms of service and rate limits
   - Consider donating if you find it valuable

**Cannot Do:**
- ❌ Download/share a pre-built database with copyrighted content
- ❌ Distribute your database publicly
- ❌ Share databases in "community" forums/Discord

**Can Do:**
- ✅ Build your own from your legally-purchased PDFs
- ✅ Use it for personal games
- ✅ Share the extraction/building code (just not the data)

Place the database at `data/pathfinder_search.db` or set `RAG_DB_PATH` in your `.env`.

## Development

### Code Style

The project uses:
- **black** for code formatting (line length 100)
- **pylint** for linting (score target: 9.0+)
- Python type hints throughout
- Pydantic models for data validation

```bash
# Format code
uv run black .

# Run linter
uv run pylint gm_agent

# Run both
uv run black . && uv run pylint gm_agent
```

### Adding New Tools

1. Create a new MCP server in `gm_agent/mcp/`:

```python
from .base import MCPServer, ToolDef, ToolParameter, ToolResult

class MyServer(MCPServer):
    def __init__(self):
        self._tools = [
            ToolDef(
                name="my_tool",
                description="What the tool does",
                parameters=[
                    ToolParameter(
                        name="arg1",
                        type="string",
                        description="Argument description",
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict) -> ToolResult:
        if name == "my_tool":
            return self._my_tool(args)
        return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _my_tool(self, args: dict) -> ToolResult:
        # Implementation
        return ToolResult(success=True, data={"result": "value"})

    def close(self) -> None:
        pass
```

2. Register the server in `gm_agent/mcp/registry.py`:

```python
SERVERS["my-server"] = ServerInfo(
    stateless=True,
    celery_eligible=True,
)
```

3. Update `_build_tool_mapping()` in registry.py to include your server

4. Add tests in `tests/test_my_server.py`

### Project Structure Guidelines

- Use Pydantic models for all data structures
- MCP servers should be stateless where possible
- Storage is file-based JSON (campaigns directory)
- Tests should use fixtures from `conftest.py`
- Use MCPClient for tool execution (supports local/remote modes)

## License

MIT License

## Future Vision: Specialized GM Models

### The Self-Improving Loop

GM Agent is designed to generate its own training data for creating specialized, efficient GM models:

**Phase 1: Data Collection** (Current State)
- Run with powerful cloud models (GPT-4, Claude Opus, o1)
- Capture complete session data: inputs, responses, tool calls, timing
- **Critical**: Capture thinking/reasoning traces from thinking models
- Build corpus of high-quality GM decision-making with explicit reasoning

**Phase 2: Specialized Fine-Tuning**
- Fine-tune smaller base models (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) on:
  - Session transcripts (input → response pairs)
  - Thinking traces (input → reasoning → response) ← **Key differentiator**
  - Tool call patterns (when to search rules, roll dice, run NPCs)
  - Pathfinder-specific knowledge baked into weights
- Result: **GM-specialized model** that's small but deeply competent at this specific task

**Phase 3: Efficient Local Deployment**
- Run quantized fine-tuned model (4-bit or 2-bit quantization)
- 8GB VRAM becomes viable for the LLM alone
- 12-16GB can run LLM + image generation (SDXL, Flux)
- 24GB can do LLM + image + video generation simultaneously

**Phase 4: Multimodal Gaming Experience**
```
Consumer GPU (24GB VRAM) Running:
├── Fine-tuned GM Agent (4-bit 8B model)    ~4-6GB
├── Image Generation (SDXL/Flux quantized)  ~8GB
├── Text-to-Video (Stable Video)            ~10GB
└── Foundry VTT + other services
```

### Why Thinking Traces Matter

Most fine-tuning datasets only have input→output pairs. Thinking traces capture the **reasoning process**:

```json
{
  "player_input": "I search the room for hidden doors",
  "thinking": "Player is exploring. Need to: 1) Check room description for clues,
              2) Determine appropriate skill (Perception for Seek action),
              3) Set DC based on room complexity, 4) Roll check,
              5) Describe results based on degree of success.",
  "tool_calls": [
    {"name": "search_rules", "args": {"query": "Seek action Perception"}},
    {"name": "roll_dice", "args": {"dice": "1d20+5", "dc": 20}}
  ],
  "gm_response": "You methodically examine the walls..."
}
```

That middle part—the thinking—is extremely rare training data. It teaches the model **how to think like a GM**, not just mimic responses.

### The Efficiency Gain

- **Before**: Run GPT-4 (expensive, cloud-dependent, ~100B+ parameters)
- **After**: Run fine-tuned 8B model at 4-bit (cheap, local, ~4GB VRAM)
- **Quality**: Potentially better at GM tasks due to specialization
- **Cost**: $0/month vs hundreds in API fees
- **Latency**: ~100ms locally vs ~2s cloud round-trip
- **Privacy**: All game data stays local

This opens up multimodal gaming on consumer hardware that would be impossible with cloud models.

### Important Note on Distribution and Legal Considerations

**Personal Use**: Fine-tuning a model on your own game sessions for personal use is legally straightforward and works great.

**Public Distribution**: This project **does not distribute fine-tuned models** due to multiple overlapping legal and ethical concerns:

**1. Paizo Content (Copyright)**
Training data includes copyrighted material that gets encoded into model weights:
- Read-aloud text from published adventures
- Examples of play from rulebooks
- Adventure-specific content (NPCs, locations, plot details)
- Game mechanics and rules text

**2. LLM Service Terms of Service**
Thinking traces and outputs from commercial LLM APIs (OpenAI, Anthropic, etc.) may be subject to their Terms of Service:
- Usage rights vs. redistribution rights unclear
- Model training on API outputs may be restricted
- Varies by provider and service tier

**3. AI-Generated Content (Unsettled Law)**
Legal status of AI-generated content (including thinking traces) is currently murky:
- Can AI outputs be copyrighted? (Currently: generally no, but evolving)
- Who owns derivative works from AI outputs?
- Is model training "transformative use"? (Courts haven't settled this)

**4. Paizo's Stated Position**
Paizo has publicly expressed concerns about AI use in their ecosystem. Out of respect for the content creators whose work makes this possible, we err on the side of caution.

**Our Interpretation**
We take a **maximalist/conservative interpretation** of copyright law for public distribution:
- Assume model training is NOT transformative fair use
- Respect content creator positions even where law is unsettled
- Better to be overly cautious than disrespectful

*Note: We don't necessarily agree this is how the law will settle, but it's the ethical position we've chosen.*

**What This Project Shares**
- ✅ The base GM Agent code (Apache 2.0 license)
- ✅ RAG search/indexing code (`gm_agent/rag/`)
- ✅ Fine-tuning pipeline and scripts
- ✅ Documentation on the approach
- ✅ Evaluation frameworks
- ❌ RAG database (`pathfinder_search.db`) - contains copyrighted Paizo content
- ❌ Fine-tuned model weights - contains copyrighted content in weights
- ❌ Training datasets - contains copyrighted and/or TOS-restricted content

**For Users**
Each user can:
1. Run their own sessions to collect data (with appropriate API access)
2. Fine-tune their own model with their own data
3. Use it locally for personal games

Think of it like a personalized AI trained on your own book collection - powerful for personal use, but distribution is legally/ethically complex.

## Acknowledgments

- [Pathfinder 2e](https://paizo.com/pathfinder) by Paizo Inc.
- [Ollama](https://ollama.ai/) for local LLM inference
- [Archives of Nethys](https://2e.aonprd.com/) for Pathfinder content
