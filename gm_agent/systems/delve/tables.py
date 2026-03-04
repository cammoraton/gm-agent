"""Delve tables for dungeon/hold building.

Card suit determines discovery type, rank determines magnitude.

Tables are loaded from data/games/delve/tables.json. Populate that file
from Delve (Worlds Without Master) to use the full game content.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parents[3] / "data" / "games" / "delve" / "tables.json"


def _load_tables() -> dict:
    if not _DATA_FILE.exists():
        logger.warning("Delve tables not found at %s", _DATA_FILE)
        return {}
    try:
        with open(_DATA_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load delve tables: %s", exc)
        return {}


def _int_keys(d: dict) -> dict:
    """Convert string keys to ints where applicable."""
    result = {}
    for k, v in d.items():
        try:
            result[int(k)] = v
        except ValueError:
            result[k] = v
    return result


_DATA = _load_tables()

# Discovery tables by suit (card rank keys: A, 2-10, J, Q, K)
HEARTS_DISCOVERY: dict[str, dict] = _DATA.get("hearts_discovery", {})
DIAMONDS_DISCOVERY: dict[str, dict] = _DATA.get("diamonds_discovery", {})
CLUBS_DISCOVERY: dict[str, dict] = _DATA.get("clubs_discovery", {})
SPADES_DISCOVERY: dict[str, dict] = _DATA.get("spades_discovery", {})

# Room types and costs
ROOM_TYPES: dict[str, dict] = _DATA.get("room_types", {})

# Unit types
UNIT_TYPES: dict[str, dict] = _DATA.get("unit_types", {})

# Combat: depth scaling for enemy strength (int keys)
DEPTH_SCALING: dict[int, float] = _int_keys(_DATA.get("depth_scaling", {}))
