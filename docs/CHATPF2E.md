# chatPF2E — Sessionless Web UI

chatPF2E is a lightweight browser chat interface for the `ChatAgent`. It requires no login, stores nothing on the server, and works out of the box with any LLM backend.

```
┌─────────────────────────────────────────┐
│  chatPF2E                   [New Chat ↺]│
├─────────────────────────────────────────┤
│                                         │
│   [user message]                        │
│             [GM Assistant response]     │
│   [user message]                        │
│             [GM Assistant response]     │
│                                         │
├─────────────────────────────────────────┤
│  [Ask a Pathfinder 2e question…  Send ] │
└─────────────────────────────────────────┘
```

## Session Model

Each browser tab gets a UUID generated at page load (never stored in `localStorage`). The server keeps an in-memory `ChatAgent` per UUID with a 30-minute idle TTL.

```
Tab opens  →  JS generates UUID  →  sent as X-Session-ID header
                                     ↓
Server: first request  →  create ChatAgent, cache it
Server: subsequent   →  reuse agent (conversation history preserved)
Tab refresh  →  new UUID  →  old agent expires via TTL (30 min)
```

- **Multi-turn context**: conversations persist within a tab's lifetime.
- **True incognito**: refresh = new session, nothing written to disk.
- **New Chat button**: calls `POST /api/reset`, generates a new UUID, reloads the page.

## Running

### Standalone (development)

```bash
# Default backend (reads CHAT_BACKEND / LLM_BACKEND env vars)
uv run python chat_ui.py --port 5001

# Explicit ollama backend
CHAT_BACKEND=ollama:gpt-oss:20b uv run python chat_ui.py --port 5001

# Or via the CLI
uv run python cli.py serve --port 5001
```

Open **http://localhost:5001** in your browser.

### Via Docker Compose

chatPF2E runs as the `chat-ui` service alongside the rest of the stack. It shares the RAG database (read-only) and the same LLM backend configuration as the API.

```bash
# Start the full stack (chat-ui included by default)
docker compose up

# Chat UI is available at http://localhost/chat/
# (proxied through nginx — no extra port needed)
```

For live code reloading during development the override file mounts `chat_ui.py` and `gm_agent/` directly:

```bash
docker compose up          # uses docker-compose.override.yml automatically
# chat-ui also exposed directly at http://localhost:5001
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI page |
| `POST` | `/api/chat` | Send a message; returns `{"response": "...", "session_id": "..."}` |
| `POST` | `/api/reset` | Clear conversation for this session |
| `GET` | `/api/health` | Health check; returns `{"ok": true, "active_sessions": N}` |

All requests should include the `X-Session-ID` header with the tab's UUID. The server creates a new session automatically if the header is missing.

### Example

```bash
# Single question
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{"message": "What is flanking in PF2e?"}'

# Follow-up (same session — agent has conversation context)
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{"message": "How does that interact with the flat-footed condition?"}'

# Reset
curl -X POST http://localhost:5001/api/reset \
  -H "X-Session-ID: my-session"
```

## Frontend

Single-page app — no build tools, no npm. CDN-only dependencies:

| Library | Purpose |
|---------|---------|
| [marked.js](https://marked.js.org) v12 | Markdown rendering (agent responses are markdown-heavy) |
| [highlight.js](https://highlightjs.org) v11 | Syntax highlighting for code blocks and stat block tables |
| [Inter](https://fonts.google.com/specimen/Inter) (Google Fonts) | UI typography |

Static files are served by Flask from `gm_agent/static/chatpf2e/`:

```
gm_agent/static/chatpf2e/
├── index.html   # Page shell, CDN script tags
├── app.js       # Session UUID, send/receive loop, markdown rendering
└── style.css    # Dark theme with amber/gold accents
```

## Configuration

chatPF2E uses the same backend selection as the rest of gm-agent:

| Env var | Default | Description |
|---------|---------|-------------|
| `CHAT_BACKEND` | `openrouter:openai/gpt-oss-20b` | Orchestrator LLM (tool calls) |
| `NARRATOR_BACKEND` | _(same as CHAT_BACKEND)_ | Synthesis LLM (final answer) |
| `LLM_BACKEND` | `ollama` | Fallback if CHAT_BACKEND is unset |
| `RAG_DB_PATH` | `./data/pathfinder_search.db` | Pathfinder content database |

See [BACKENDS.md](BACKENDS.md) for full backend configuration options.

## nginx Routing

When running behind nginx (Docker Compose), all traffic to `/chat/` is proxied to `chat-ui:5001/` with the prefix stripped:

```
GET  http://host/chat/          →  chat-ui:5001/         (index.html)
GET  http://host/chat/app.js    →  chat-ui:5001/app.js
POST http://host/chat/api/chat  →  chat-ui:5001/api/chat
POST http://host/chat/api/reset →  chat-ui:5001/api/reset
```

The JS uses relative URLs (`api/chat`, `api/reset`) so it works correctly whether accessed directly on port 5001 or through the nginx prefix — no base-URL configuration needed.

## Public Access via ngrok

To share the UI over the internet without a public server, use the `ngrok` profile. See [DEPLOYMENT.md — ngrok](DEPLOYMENT.md#ngrok-optional-public-tunnel) for setup.
