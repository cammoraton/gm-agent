"""Configuration settings for GM Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
CAMPAIGNS_DIR = Path(os.getenv("CAMPAIGNS_DIR", BASE_DIR / "data" / "campaigns"))
RAG_DB_PATH = Path(os.getenv("RAG_DB_PATH", BASE_DIR / "data" / "pathfinder_search.db"))

# Backend selection
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://172.30.16.72:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku")

# OpenRouter settings (uses OpenAI SDK)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Temperature settings for LLM calls.
# Creative calls (roleplay, NPC dialogue, player emulation) use higher temps
# for more varied, expressive output. Mechanical calls (reranking, extraction,
# summarization) use lower temps for consistency and accuracy.
TEMPERATURE_CREATIVE = float(os.getenv("LLM_TEMPERATURE_CREATIVE", "0.9"))
TEMPERATURE_MECHANICAL = float(os.getenv("LLM_TEMPERATURE_MECHANICAL", "0.3"))

# Pipeline model settings — two-phase retrieval/synthesis architecture.
# Tool selection (Phase 1) and response generation (Phase 2) use separate
# LLM calls with different prompts and temperatures.  The retrieval phase
# can optionally use a different (smaller/cheaper) model.
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND")  # None = use LLM_BACKEND
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL")  # None = use backend default
RETRIEVAL_TEMPERATURE = float(os.getenv("RETRIEVAL_TEMPERATURE", "0.0"))
SYNTHESIS_TEMPERATURE = float(os.getenv("SYNTHESIS_TEMPERATURE", "0.1"))

# Redis/Celery settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# MCP mode settings
# "local" = direct MCP execution (standalone, in-process servers)
# "remote" = CLI -> API (MCP facade) -> Celery -> MCP Workers
MCP_MODE = os.getenv("MCP_MODE", "local")
MCP_API_URL = os.getenv("MCP_API_URL", "http://localhost:5000")

# State checkpointing settings
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "30"))  # seconds
CHECKPOINT_STALE_THRESHOLD = int(os.getenv("CHECKPOINT_STALE_THRESHOLD", "300"))  # 5 minutes

# Tool execution settings
PARALLEL_TOOL_CALLS = os.getenv("PARALLEL_TOOL_CALLS", "false").lower() == "true"

# Context settings
MAX_RECENT_TURNS = int(os.getenv("MAX_RECENT_TURNS", "15"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))

# Rolling summary settings
TURNS_BETWEEN_SUMMARIES = int(os.getenv("TURNS_BETWEEN_SUMMARIES", "10"))

# Default campaign preferences
DEFAULT_PREFERENCES = {
    "rag_aggressiveness": "moderate",  # minimal, moderate, aggressive
    "uncertainty_mode": "gm",  # gm (make rulings) or introspective (admit gaps)
    "dry_run": False,  # If true, log responses instead of posting to Foundry
    # Automation prompt templates (optional, support {actor_name}, {content}, {context} variables):
    # "player_chat_prompt": "{actor_name}: {content}"  # Player chat message prompt
    # "npc_combat_turn_prompt": "It's {actor_name}'s turn..."  # NPC combat turn prompt
    # "npc_narration_prompt": "{actor_name} is taking their turn..."  # NPC narration prompt
}

# GM System prompt — canonical location: gm_agent.systems.pf2e.prompts
# Re-exported here for backward compatibility.
from gm_agent.systems.pf2e.prompts import GM_SYSTEM_PROMPT  # noqa: F401, E402

# RAG aggressiveness prompts - appended to system prompt based on setting
RAG_PROMPTS = {
    "minimal": """
Tool Usage: Use tools only when players explicitly ask about rules, creatures, spells, or items. Don't proactively look things up.
""",
    "moderate": """
Tool Usage: Use tools when:
- Players ask about rules, creatures, spells, or items
- You need to verify a rule before making a ruling
- An NPC or location mentioned might have official stats/details
""",
    "aggressive": """
Tool Usage: Proactively use tools to ensure accuracy:
- Always look up creature stats before describing encounters
- Verify rules before adjudicating actions
- Search for relevant content when introducing NPCs, locations, or items
- Pre-fetch information that might be relevant to the current scene
""",
}

# Uncertainty mode prompts
UNCERTAINTY_PROMPTS = {
    "gm": """
When uncertain about rules: Make a reasonable ruling and continue play. You can note "I'm ruling it this way for now" but keep the game moving.
""",
    "introspective": """
When uncertain about rules: Acknowledge the uncertainty. Say "I'm not sure about the exact rule for this - let me look it up" and use the search tools. If you still can't find it, suggest the player check the rulebook or make a temporary ruling.
""",
}

# =============================================================================
# Foundry Communication Mode
# =============================================================================

# Communication mode: "websocket" (default) or "polling"
# - websocket: Foundry connects TO gm-agent via Socket.IO (local/same network)
# - polling: gm-agent polls Foundry via REST API (remote/firewall scenarios)
FOUNDRY_MODE = os.getenv("FOUNDRY_MODE", "websocket")

# Polling mode settings (only used when FOUNDRY_MODE=polling)
FOUNDRY_POLL_URL = os.getenv("FOUNDRY_POLL_URL", "")
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY", "")
FOUNDRY_CAMPAIGN_ID = os.getenv("FOUNDRY_CAMPAIGN_ID", "")
FOUNDRY_POLL_INTERVAL = float(os.getenv("FOUNDRY_POLL_INTERVAL", "2.0"))
FOUNDRY_LONG_POLL_TIMEOUT = float(os.getenv("FOUNDRY_LONG_POLL_TIMEOUT", "25.0"))
FOUNDRY_VERIFY_SSL = os.getenv("FOUNDRY_VERIFY_SSL", "true").lower() == "true"
