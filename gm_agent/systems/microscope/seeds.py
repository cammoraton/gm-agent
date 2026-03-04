"""Pre-written Microscope story seeds.

Each seed provides a complete starting point: Big Picture, bookend periods,
initial palette, and suggested first focus.

Seeds are loaded from data/games/microscope/seeds.json. Populate that file
from the Microscope RPG (Lame Mage Productions) to use the full game content.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parents[3] / "data" / "games" / "microscope" / "seeds.json"


@dataclass
class MicroscopeSeed:
    """A pre-written story seed for Microscope."""

    name: str
    big_picture: str
    start_period: str
    start_tone: str
    end_period: str
    end_tone: str
    palette_yes: list[str] = field(default_factory=list)
    palette_no: list[str] = field(default_factory=list)
    suggested_focus: str = ""


def _load_seeds() -> dict[str, MicroscopeSeed]:
    if not _DATA_FILE.exists():
        logger.warning("Microscope seeds not found at %s", _DATA_FILE)
        return {}
    try:
        with open(_DATA_FILE) as f:
            entries = json.load(f)
        return {
            e["key"]: MicroscopeSeed(
                name=e["name"],
                big_picture=e["big_picture"],
                start_period=e["start_period"],
                start_tone=e["start_tone"],
                end_period=e["end_period"],
                end_tone=e["end_tone"],
                palette_yes=e.get("palette_yes", []),
                palette_no=e.get("palette_no", []),
                suggested_focus=e.get("suggested_focus", ""),
            )
            for e in entries
        }
    except Exception as exc:
        logger.error("Failed to load microscope seeds: %s", exc)
        return {}


SEEDS: dict[str, MicroscopeSeed] = _load_seeds()
