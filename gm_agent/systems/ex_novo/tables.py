"""Ex Novo random tables for settlement generation.

Tables are loaded from data/games/ex_novo/tables.json. Populate that file
from Ex Novo (Worlds Without Master) to use the full game content.
"""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parents[3] / "data" / "games" / "ex_novo" / "tables.json"


def _load_tables() -> dict:
    if not _DATA_FILE.exists():
        logger.warning("Ex Novo tables not found at %s", _DATA_FILE)
        return {}
    try:
        with open(_DATA_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load ex_novo tables: %s", exc)
        return {}


def _int_keys(d: dict) -> dict:
    """Convert string keys to ints (JSON doesn't support int keys)."""
    return {int(k): v for k, v in d.items()}


_DATA = _load_tables()

# Scale Tables (D6)
AGE_TABLE: dict[int, str] = _int_keys(_DATA.get("age_table", {}))
SIZE_TABLE: dict[int, str] = _int_keys(_DATA.get("size_table", {}))
INFLUENCE_TABLE: dict[int, str] = _int_keys(_DATA.get("influence_table", {}))

# Terrain Tables (2D6)
GEOGRAPHY_TABLE: dict[int, str] = _int_keys(_DATA.get("geography_table", {}))
FEATURES_TABLE: dict[int, str] = _int_keys(_DATA.get("features_table", {}))

# Purpose Tables (2D6)
LOCATION_TABLE: dict[int, str] = _int_keys(_DATA.get("location_table", {}))
DECISION_TABLE: dict[int, str] = _int_keys(_DATA.get("decision_table", {}))

# Power Tables (2D6)
PARADIGM_TABLE: dict[int, str] = _int_keys(_DATA.get("paradigm_table", {}))
RELATIONSHIP_TABLE: dict[int, str] = _int_keys(_DATA.get("relationship_table", {}))

# Event Table (D666)
EVENT_TABLE: dict[int, dict] = _int_keys(_DATA.get("event_table", {}))


def roll_d6() -> int:
    """Roll a single d6."""
    return random.randint(1, 6)


def roll_2d6() -> int:
    """Roll 2d6 and sum."""
    return random.randint(1, 6) + random.randint(1, 6)


def roll_d666() -> int:
    """Roll d666 (three d6 digits)."""
    return random.randint(1, 6) * 100 + random.randint(1, 6) * 10 + random.randint(1, 6)
