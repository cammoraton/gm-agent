"""Themed oracle tables for Microscope inspiration.

Each oracle has 20-30 entries. Roll randomly for creative prompts.

Oracles are loaded from data/games/microscope/oracles.json. Populate that file
from the Microscope RPG (Lame Mage Productions) to use the full game content.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parents[3] / "data" / "games" / "microscope" / "oracles.json"


@dataclass
class Oracle:
    """A themed oracle table."""

    name: str
    category: str  # "theme", "character", "conflict", "setting"
    entries: list[str] = field(default_factory=list)

    def consult(self) -> str:
        """Get a random entry from this oracle."""
        if not self.entries:
            return "(empty oracle)"
        return random.choice(self.entries)


def _load_oracles() -> dict[str, Oracle]:
    if not _DATA_FILE.exists():
        logger.warning("Microscope oracles not found at %s", _DATA_FILE)
        return {}
    try:
        with open(_DATA_FILE) as f:
            entries = json.load(f)
        return {
            e["key"]: Oracle(
                name=e["name"],
                category=e["category"],
                entries=e.get("entries", []),
            )
            for e in entries
        }
    except Exception as exc:
        logger.error("Failed to load microscope oracles: %s", exc)
        return {}


ORACLES: dict[str, Oracle] = _load_oracles()
