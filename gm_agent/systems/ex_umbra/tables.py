"""Ex Umbra tables for dungeon generation.

Tremor table, threat/reward token scales, and dungeon card prompts.

Tables are loaded from data/games/ex_umbra/tables.json. Populate that file
from Ex Umbra (Worlds Without Master) to use the full game content.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parents[3] / "data" / "games" / "ex_umbra" / "tables.json"


def _load_tables() -> dict:
    if not _DATA_FILE.exists():
        logger.warning("Ex Umbra tables not found at %s", _DATA_FILE)
        return {}
    try:
        with open(_DATA_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load ex_umbra tables: %s", exc)
        return {}


def _int_keys(d: dict) -> dict:
    """Convert string keys to ints (JSON doesn't support int keys)."""
    return {int(k): v for k, v in d.items()}


_DATA = _load_tables()

# Tremor Table (D20)
TREMOR_TABLE: dict[int, dict] = _int_keys(_DATA.get("tremor_table", {}))

# Token scales
THREAT_SCALE: dict[int, str] = _int_keys(_DATA.get("threat_scale", {}))
REWARD_SCALE: dict[int, str] = _int_keys(_DATA.get("reward_scale", {}))

# Difficulty scaling
DIFFICULTY_SETTINGS: dict[int, dict] = _int_keys(_DATA.get("difficulty_settings", {}))

# Dungeon generation prompts
ADJECTIVE_PAIRS: list[list[str]] = _DATA.get("adjective_pairs", [])
ARCHITECTURE_TYPES: list[str] = _DATA.get("architecture_types", [])
