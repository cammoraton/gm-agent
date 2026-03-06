"""Creature modifier MCP server for mechanical stat adjustments.

Provides deterministic arithmetic for Elite/Weak adjustments, template
application, structured stat extraction, and creature/hazard scaffolding
from GM Core creation tables.
"""

import json
import math
from typing import Any

from gm_agent.config import RAG_DB_PATH
from gm_agent.rag import PathfinderSearch
from gm_agent.mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult


# ---------------------------------------------------------------------------
# LLM prompt for custom creature ability generation
# ---------------------------------------------------------------------------
CREATURE_DESIGN_SYSTEM_PROMPT = """You are a Pathfinder 2e game designer creating custom creature abilities.

Output ONLY valid JSON in this exact format (no extra text):
{
  "name": "Creature Name",
  "level": <integer -1 to 25>,
  "abilities": [
    {
      "name": "Ability Name",
      "action_cost": "[one-action]",
      "traits": ["trait1", "trait2"],
      "description": "Full mechanical description with damage expressions and DCs."
    }
  ]
}

Rules:
- action_cost values: "[one-action]", "[two-actions]", "[three-actions]", "[free-action]", "[reaction]", or "" (passive)
- Scale all damage expressions to match the provided level_stats.damage_avg
- Scale all save DCs to match level_stats.spell_dc
- Use PF2e terminology: Fortitude/Reflex/Will saves, basic saves, Attack Rolls
- Include 3-6 abilities that fit the creature concept
- Output ONLY the JSON object, nothing else"""


# ---------------------------------------------------------------------------
# GM Core Elite/Weak HP Adjustment Table
# Fallback for deployments without v7 search.db; loaded from DB at runtime.
# ---------------------------------------------------------------------------
ELITE_HP_ADJUSTMENT: dict[int, int] = {
    -1: 10, 0: 10, 1: 10, 2: 10, 3: 15, 4: 15,
    5: 20, 6: 20, 7: 25, 8: 25, 9: 30, 10: 30,
    11: 35, 12: 35, 13: 40, 14: 40, 15: 45, 16: 45,
    17: 50, 18: 55, 19: 60, 20: 65, 21: 70, 22: 75,
    23: 80, 24: 85, 25: 90,
}

# Fallback for levels outside the table
_DEFAULT_ELITE_HP = 10

# ---------------------------------------------------------------------------
# GM Core Table 2-1 - Creature Stats by Level (moderate baseline)
# Fallback for deployments without v7 search.db; loaded from DB at runtime.
# ---------------------------------------------------------------------------
CREATURE_STATS_BY_LEVEL: dict[int, dict[str, Any]] = {
    -1: {"hp": 9, "ac": 15, "fort": 5, "ref": 5, "will": 5, "perception": 5,
         "attack": 7, "damage_avg": 4, "spell_dc": 14, "skill_high": 6, "skill_mod": 4},
    0: {"hp": 15, "ac": 16, "fort": 6, "ref": 6, "will": 6, "perception": 6,
        "attack": 8, "damage_avg": 5, "spell_dc": 15, "skill_high": 7, "skill_mod": 5},
    1: {"hp": 20, "ac": 17, "fort": 7, "ref": 7, "will": 7, "perception": 7,
        "attack": 9, "damage_avg": 6, "spell_dc": 16, "skill_high": 8, "skill_mod": 5},
    2: {"hp": 30, "ac": 18, "fort": 8, "ref": 8, "will": 8, "perception": 8,
        "attack": 11, "damage_avg": 8, "spell_dc": 18, "skill_high": 10, "skill_mod": 7},
    3: {"hp": 40, "ac": 19, "fort": 9, "ref": 9, "will": 9, "perception": 9,
        "attack": 12, "damage_avg": 10, "spell_dc": 19, "skill_high": 11, "skill_mod": 8},
    4: {"hp": 55, "ac": 21, "fort": 11, "ref": 11, "will": 10, "perception": 10,
        "attack": 14, "damage_avg": 12, "spell_dc": 21, "skill_high": 13, "skill_mod": 9},
    5: {"hp": 70, "ac": 22, "fort": 12, "ref": 12, "will": 11, "perception": 11,
        "attack": 15, "damage_avg": 14, "spell_dc": 22, "skill_high": 14, "skill_mod": 10},
    6: {"hp": 85, "ac": 24, "fort": 13, "ref": 13, "will": 12, "perception": 13,
        "attack": 17, "damage_avg": 16, "spell_dc": 24, "skill_high": 16, "skill_mod": 12},
    7: {"hp": 100, "ac": 25, "fort": 14, "ref": 14, "will": 13, "perception": 14,
        "attack": 18, "damage_avg": 18, "spell_dc": 25, "skill_high": 17, "skill_mod": 13},
    8: {"hp": 120, "ac": 27, "fort": 16, "ref": 15, "will": 14, "perception": 15,
        "attack": 20, "damage_avg": 20, "spell_dc": 27, "skill_high": 19, "skill_mod": 14},
    9: {"hp": 135, "ac": 28, "fort": 17, "ref": 16, "will": 15, "perception": 16,
        "attack": 21, "damage_avg": 22, "spell_dc": 28, "skill_high": 20, "skill_mod": 15},
    10: {"hp": 155, "ac": 30, "fort": 18, "ref": 17, "will": 16, "perception": 18,
         "attack": 23, "damage_avg": 24, "spell_dc": 30, "skill_high": 22, "skill_mod": 17},
    11: {"hp": 170, "ac": 31, "fort": 19, "ref": 18, "will": 17, "perception": 19,
         "attack": 24, "damage_avg": 26, "spell_dc": 31, "skill_high": 23, "skill_mod": 18},
    12: {"hp": 190, "ac": 33, "fort": 21, "ref": 19, "will": 18, "perception": 20,
         "attack": 26, "damage_avg": 29, "spell_dc": 33, "skill_high": 25, "skill_mod": 19},
    13: {"hp": 210, "ac": 34, "fort": 22, "ref": 20, "will": 19, "perception": 21,
         "attack": 27, "damage_avg": 31, "spell_dc": 34, "skill_high": 26, "skill_mod": 20},
    14: {"hp": 230, "ac": 36, "fort": 23, "ref": 21, "will": 20, "perception": 23,
         "attack": 29, "damage_avg": 33, "spell_dc": 36, "skill_high": 28, "skill_mod": 22},
    15: {"hp": 250, "ac": 37, "fort": 24, "ref": 22, "will": 21, "perception": 24,
         "attack": 30, "damage_avg": 35, "spell_dc": 37, "skill_high": 29, "skill_mod": 23},
    16: {"hp": 275, "ac": 39, "fort": 26, "ref": 24, "will": 23, "perception": 25,
         "attack": 32, "damage_avg": 37, "spell_dc": 39, "skill_high": 31, "skill_mod": 24},
    17: {"hp": 300, "ac": 40, "fort": 27, "ref": 25, "will": 24, "perception": 26,
         "attack": 33, "damage_avg": 39, "spell_dc": 40, "skill_high": 32, "skill_mod": 25},
    18: {"hp": 330, "ac": 42, "fort": 28, "ref": 26, "will": 25, "perception": 28,
         "attack": 35, "damage_avg": 42, "spell_dc": 42, "skill_high": 34, "skill_mod": 27},
    19: {"hp": 360, "ac": 43, "fort": 29, "ref": 27, "will": 26, "perception": 29,
         "attack": 36, "damage_avg": 44, "spell_dc": 43, "skill_high": 35, "skill_mod": 28},
    20: {"hp": 390, "ac": 45, "fort": 31, "ref": 29, "will": 28, "perception": 30,
         "attack": 38, "damage_avg": 46, "spell_dc": 45, "skill_high": 37, "skill_mod": 29},
    21: {"hp": 425, "ac": 46, "fort": 32, "ref": 30, "will": 29, "perception": 31,
         "attack": 39, "damage_avg": 48, "spell_dc": 46, "skill_high": 38, "skill_mod": 30},
    22: {"hp": 460, "ac": 48, "fort": 33, "ref": 31, "will": 30, "perception": 33,
         "attack": 41, "damage_avg": 50, "spell_dc": 48, "skill_high": 40, "skill_mod": 32},
    23: {"hp": 500, "ac": 49, "fort": 34, "ref": 32, "will": 31, "perception": 34,
         "attack": 42, "damage_avg": 52, "spell_dc": 49, "skill_high": 41, "skill_mod": 33},
    24: {"hp": 540, "ac": 51, "fort": 36, "ref": 34, "will": 33, "perception": 35,
         "attack": 44, "damage_avg": 55, "spell_dc": 51, "skill_high": 43, "skill_mod": 34},
    25: {"hp": 580, "ac": 52, "fort": 37, "ref": 35, "will": 34, "perception": 36,
         "attack": 45, "damage_avg": 57, "spell_dc": 52, "skill_high": 44, "skill_mod": 35},
}

# Role adjustments (applied on top of baseline)
ROLE_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    "brute": {
        "hp_mult": 1.2,
        "ac": -1,
        "attack": -1,
        "damage_mult": 1.1,
        "fort": 2,
        "notes": "High HP and Fort, lower AC and attack. Hits hard when it connects.",
    },
    "sniper": {
        "hp_mult": 0.8,
        "attack": 2,
        "damage_mult": 0.9,
        "ref": 1,
        "notes": "High accuracy, lower HP. Prefers ranged attacks.",
    },
    "soldier": {
        "ac": 1,
        "fort": 1,
        "notes": "Balanced combatant with slightly higher AC and Fort.",
    },
    "skirmisher": {
        "ref": 2,
        "speed_bonus": 10,
        "notes": "High Reflex and extra speed. Moves around the battlefield.",
    },
    "spellcaster": {
        "hp_mult": 0.8,
        "spell_dc": 2,
        "attack": -1,
        "will": 2,
        "notes": "Lower HP, higher spell DC and Will. Primary threat is spells.",
    },
}

# ---------------------------------------------------------------------------
# GM Core Hazard Creation Table (moderate baseline)
# Fallback for deployments without v7 search.db; loaded from DB at runtime.
# ---------------------------------------------------------------------------
HAZARD_STATS_BY_LEVEL: dict[int, dict[str, Any]] = {
    -1: {"ac": 15, "fort": 5, "ref": 2, "hp": 8, "hardness": 3,
         "stealth_dc": 14, "disable_dc": 14, "attack": 7, "damage": "1d6+2"},
    0: {"ac": 16, "fort": 6, "ref": 3, "hp": 12, "hardness": 4,
        "stealth_dc": 15, "disable_dc": 15, "attack": 8, "damage": "1d6+3"},
    1: {"ac": 17, "fort": 7, "ref": 4, "hp": 16, "hardness": 5,
        "stealth_dc": 16, "disable_dc": 16, "attack": 9, "damage": "1d8+4"},
    2: {"ac": 18, "fort": 8, "ref": 5, "hp": 24, "hardness": 6,
        "stealth_dc": 18, "disable_dc": 18, "attack": 11, "damage": "2d6+4"},
    3: {"ac": 19, "fort": 9, "ref": 6, "hp": 32, "hardness": 7,
        "stealth_dc": 19, "disable_dc": 19, "attack": 12, "damage": "2d6+6"},
    4: {"ac": 21, "fort": 11, "ref": 7, "hp": 40, "hardness": 8,
        "stealth_dc": 21, "disable_dc": 21, "attack": 14, "damage": "2d8+6"},
    5: {"ac": 22, "fort": 12, "ref": 8, "hp": 50, "hardness": 9,
        "stealth_dc": 22, "disable_dc": 22, "attack": 15, "damage": "2d8+8"},
    6: {"ac": 24, "fort": 13, "ref": 9, "hp": 60, "hardness": 10,
        "stealth_dc": 24, "disable_dc": 24, "attack": 17, "damage": "2d10+8"},
    7: {"ac": 25, "fort": 14, "ref": 10, "hp": 70, "hardness": 11,
        "stealth_dc": 25, "disable_dc": 25, "attack": 18, "damage": "2d10+10"},
    8: {"ac": 27, "fort": 16, "ref": 11, "hp": 80, "hardness": 12,
        "stealth_dc": 27, "disable_dc": 27, "attack": 20, "damage": "3d8+10"},
    9: {"ac": 28, "fort": 17, "ref": 12, "hp": 90, "hardness": 13,
        "stealth_dc": 28, "disable_dc": 28, "attack": 21, "damage": "3d8+12"},
    10: {"ac": 30, "fort": 18, "ref": 13, "hp": 100, "hardness": 14,
         "stealth_dc": 30, "disable_dc": 30, "attack": 23, "damage": "3d10+12"},
    11: {"ac": 31, "fort": 19, "ref": 14, "hp": 115, "hardness": 15,
         "stealth_dc": 31, "disable_dc": 31, "attack": 24, "damage": "3d10+14"},
    12: {"ac": 33, "fort": 21, "ref": 15, "hp": 130, "hardness": 16,
         "stealth_dc": 33, "disable_dc": 33, "attack": 26, "damage": "3d12+14"},
    13: {"ac": 34, "fort": 22, "ref": 16, "hp": 145, "hardness": 17,
         "stealth_dc": 34, "disable_dc": 34, "attack": 27, "damage": "3d12+16"},
    14: {"ac": 36, "fort": 23, "ref": 17, "hp": 160, "hardness": 18,
         "stealth_dc": 36, "disable_dc": 36, "attack": 29, "damage": "4d10+16"},
    15: {"ac": 37, "fort": 24, "ref": 18, "hp": 175, "hardness": 19,
         "stealth_dc": 37, "disable_dc": 37, "attack": 30, "damage": "4d10+18"},
    16: {"ac": 39, "fort": 26, "ref": 19, "hp": 195, "hardness": 20,
         "stealth_dc": 39, "disable_dc": 39, "attack": 32, "damage": "4d10+20"},
    17: {"ac": 40, "fort": 27, "ref": 20, "hp": 210, "hardness": 21,
         "stealth_dc": 40, "disable_dc": 40, "attack": 33, "damage": "4d12+18"},
    18: {"ac": 42, "fort": 28, "ref": 21, "hp": 230, "hardness": 22,
         "stealth_dc": 42, "disable_dc": 42, "attack": 35, "damage": "4d12+20"},
    19: {"ac": 43, "fort": 29, "ref": 22, "hp": 250, "hardness": 23,
         "stealth_dc": 43, "disable_dc": 43, "attack": 36, "damage": "5d10+20"},
    20: {"ac": 45, "fort": 31, "ref": 23, "hp": 270, "hardness": 24,
         "stealth_dc": 45, "disable_dc": 45, "attack": 38, "damage": "5d10+22"},
    21: {"ac": 46, "fort": 32, "ref": 24, "hp": 290, "hardness": 25,
         "stealth_dc": 46, "disable_dc": 46, "attack": 39, "damage": "5d12+22"},
    22: {"ac": 48, "fort": 33, "ref": 25, "hp": 310, "hardness": 26,
         "stealth_dc": 48, "disable_dc": 48, "attack": 41, "damage": "5d12+24"},
    23: {"ac": 49, "fort": 34, "ref": 26, "hp": 335, "hardness": 27,
         "stealth_dc": 49, "disable_dc": 49, "attack": 42, "damage": "6d10+24"},
    24: {"ac": 51, "fort": 36, "ref": 27, "hp": 360, "hardness": 28,
         "stealth_dc": 51, "disable_dc": 51, "attack": 44, "damage": "6d10+26"},
    25: {"ac": 52, "fort": 37, "ref": 28, "hp": 380, "hardness": 29,
         "stealth_dc": 52, "disable_dc": 52, "attack": 45, "damage": "6d12+26"},
}


# ---------------------------------------------------------------------------
# Helper functions for stat modifications
# ---------------------------------------------------------------------------

def _get_elite_hp_adjustment(level: int, table: dict[int, int] | None = None) -> int:
    """Get HP adjustment for Elite template at given level."""
    t = table if table is not None else ELITE_HP_ADJUSTMENT
    return t.get(level, _DEFAULT_ELITE_HP)


def _adjust_damage_per_die(damage_str: str, adjustment: int) -> str:
    """Adjust damage by +N per damage die in a damage string.

    E.g., "2d6+3 slashing" with adjustment=2 -> "2d6+7 slashing"
    (2 dice * +2 = +4, plus original +3 = +7)
    """
    import re
    # Match patterns like "1d6", "2d8+3", "2d6+3 slashing"
    match = re.match(r"(\d+)d(\d+)([+-]\d+)?(.*)", damage_str.strip())
    if not match:
        return damage_str

    num_dice = int(match.group(1))
    die_size = int(match.group(2))
    flat_bonus = int(match.group(3)) if match.group(3) else 0
    remainder = match.group(4)

    total_adjustment = num_dice * adjustment
    new_bonus = flat_bonus + total_adjustment

    if new_bonus > 0:
        bonus_str = f"+{new_bonus}"
    elif new_bonus < 0:
        bonus_str = str(new_bonus)
    else:
        bonus_str = ""

    return f"{num_dice}d{die_size}{bonus_str}{remainder}"


def _parse_metadata(result: dict) -> dict[str, Any]:
    """Parse creature metadata from a search result."""
    metadata = result.get("metadata")
    if metadata is None:
        return {}
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            return {}
    return dict(metadata) if metadata else {}


def _apply_elite_adjustments(
    stats: dict[str, Any], level: int, elite_hp_table: dict[int, int] | None = None
) -> dict[str, Any]:
    """Apply Elite adjustments to a stat block."""
    result = dict(stats)
    hp_adj = _get_elite_hp_adjustment(level, elite_hp_table)

    # +1 level
    result["level"] = level + 1

    # +HP
    result["hp"] = stats.get("hp", 0) + hp_adj

    # +2 to AC, saves, Perception, skills, attack bonuses, DCs
    result["ac"] = stats.get("ac", 0) + 2

    if "saves" in stats:
        result["saves"] = {k: v + 2 for k, v in stats["saves"].items()}

    if "perception" in stats:
        result["perception"] = stats["perception"] + 2

    if "skills" in stats:
        result["skills"] = {k: v + 2 for k, v in stats["skills"].items()}

    if "attacks" in stats:
        new_attacks = []
        for atk in stats["attacks"]:
            new_atk = dict(atk)
            new_atk["bonus"] = atk.get("bonus", 0) + 2
            if "damage" in atk:
                new_atk["damage"] = _adjust_damage_per_die(atk["damage"], 2)
            new_attacks.append(new_atk)
        result["attacks"] = new_attacks

    result["_adjustment"] = "elite"
    result["_hp_adjustment"] = f"+{hp_adj}"
    return result


def _apply_weak_adjustments(
    stats: dict[str, Any], level: int, elite_hp_table: dict[int, int] | None = None
) -> dict[str, Any]:
    """Apply Weak adjustments to a stat block."""
    result = dict(stats)
    hp_adj = _get_elite_hp_adjustment(level, elite_hp_table)

    # -1 level
    result["level"] = level - 1

    # -HP
    result["hp"] = max(1, stats.get("hp", 0) - hp_adj)

    # -2 to AC, saves, Perception, skills, attack bonuses, DCs
    result["ac"] = stats.get("ac", 0) - 2

    if "saves" in stats:
        result["saves"] = {k: v - 2 for k, v in stats["saves"].items()}

    if "perception" in stats:
        result["perception"] = stats["perception"] - 2

    if "skills" in stats:
        result["skills"] = {k: v - 2 for k, v in stats["skills"].items()}

    if "attacks" in stats:
        new_attacks = []
        for atk in stats["attacks"]:
            new_atk = dict(atk)
            new_atk["bonus"] = atk.get("bonus", 0) - 2
            if "damage" in atk:
                new_atk["damage"] = _adjust_damage_per_die(atk["damage"], -2)
            new_attacks.append(new_atk)
        result["attacks"] = new_attacks

    result["_adjustment"] = "weak"
    result["_hp_adjustment"] = f"-{hp_adj}"
    return result


def _format_stat_block(name: str, stats: dict[str, Any], adjustment: str = "") -> str:
    """Format a stat block as human-readable text."""
    adj_label = f" ({adjustment.title()})" if adjustment else ""
    lines = [f"**{name}{adj_label}** — Level {stats.get('level', '?')}"]

    if stats.get("hp") is not None:
        hp_note = f" ({stats['_hp_adjustment']} HP)" if "_hp_adjustment" in stats else ""
        lines.append(f"**HP** {stats['hp']}{hp_note}")

    if stats.get("ac") is not None:
        lines.append(f"**AC** {stats['ac']}")

    if "saves" in stats:
        saves = stats["saves"]
        save_parts = [f"**{k}** +{v}" if v >= 0 else f"**{k}** {v}" for k, v in saves.items()]
        lines.append(f"Saves: {', '.join(save_parts)}")

    if stats.get("perception") is not None:
        p = stats["perception"]
        lines.append(f"**Perception** +{p}" if p >= 0 else f"**Perception** {p}")

    if "speed" in stats:
        speed_parts = [f"{k} {v} ft." for k, v in stats["speed"].items()]
        lines.append(f"**Speed** {', '.join(speed_parts)}")

    if "skills" in stats:
        skill_parts = [f"{k} +{v}" if v >= 0 else f"{k} {v}" for k, v in stats["skills"].items()]
        lines.append(f"**Skills** {', '.join(skill_parts)}")

    if "attacks" in stats:
        for atk in stats["attacks"]:
            bonus = atk.get("bonus", 0)
            bonus_str = f"+{bonus}" if bonus >= 0 else str(bonus)
            damage = atk.get("damage", "")
            lines.append(f"**{atk.get('name', 'Strike')}** {bonus_str} ({damage})")

    return "\n".join(lines)


# ===========================================================================
# CreatureModifierServer
# ===========================================================================

# ---------------------------------------------------------------------------
# Troop/Swarm Tables
# ---------------------------------------------------------------------------
TROOP_SIZE_TABLE: dict[str, dict[str, Any]] = {
    "large": {"area": "10 ft. square", "hp_mult": 1.5, "ac_adj": -1},
    "huge": {"area": "15 ft. square", "hp_mult": 2.0, "ac_adj": -2},
    "gargantuan": {"area": "20 ft. square", "hp_mult": 3.0, "ac_adj": -3},
}

TROOP_THRESHOLDS = {
    "full": {"threshold_fraction": 1.0, "attack_penalty": 0, "label": "Full Strength"},
    "diminished": {"threshold_fraction": 2 / 3, "attack_penalty": -2, "label": "Diminished (2/3 HP)"},
    "broken": {"threshold_fraction": 1 / 3, "attack_penalty": -4, "label": "Broken (1/3 HP)"},
}

TROOP_DEFENSES = (
    "**Troop Defenses** The troop is immune to effects that target a specific number of "
    "creatures (such as *chromatic orb*), though it can be affected by effects that target "
    "an area. A troop takes double damage from fireball and similar area effects. "
    "The troop treats its space as difficult terrain for non-troop creatures."
)

SWARM_SIZE_TABLE: dict[str, dict[str, Any]] = {
    "large": {"area": "10 ft. square", "hp_mult": 1.0},
    "huge": {"area": "15 ft. square", "hp_mult": 1.5},
}

SWARM_DEFENSES = (
    "**Swarm Defenses** The swarm is immune to effects that target a specific number of "
    "creatures. It has weakness to area damage and resistance to physical damage "
    "(except splash). A swarm can occupy the same space as other creatures."
)


class CreatureModifierServer(MCPServer):
    """MCP server for deterministic creature stat modifications.

    Wraps PathfinderSearch to look up creatures, then applies Elite/Weak
    adjustments, templates, or scaffolds new stat blocks from GM Core tables.
    """

    # Class-level defaults — overridden by __init__ when DB v7 is available.
    # Ensures subclasses that bypass __init__ (e.g., in tests) still have valid tables.
    _creature_stats = CREATURE_STATS_BY_LEVEL
    _hazard_stats = HAZARD_STATS_BY_LEVEL
    _elite_hp = ELITE_HP_ADJUSTMENT
    _role_adjustments = ROLE_ADJUSTMENTS
    _troop_size_table = TROOP_SIZE_TABLE
    _swarm_size_table = SWARM_SIZE_TABLE

    def __init__(self, db_path: str | None = None, backend: Any = None):
        self.search = PathfinderSearch(db_path=str(db_path or RAG_DB_PATH))
        self.backend = backend  # Optional LLM backend; required for design_creature with concept/abilities
        # Load creation tables from DB (v7+); fall back to module-level constants
        self._creature_stats = self.search.get_creation_table("creature_stats") or CREATURE_STATS_BY_LEVEL
        self._hazard_stats = self.search.get_creation_table("hazard_stats") or HAZARD_STATS_BY_LEVEL
        self._elite_hp = self.search.get_creation_table("elite_hp") or ELITE_HP_ADJUSTMENT
        self._role_adjustments = self.search.get_creation_table("role_adjustments") or ROLE_ADJUSTMENTS
        self._troop_size_table = TROOP_SIZE_TABLE  # original creative content, not from publisher
        self._swarm_size_table = SWARM_SIZE_TABLE  # original creative content, not from publisher
        self._tools = self._build_tools()

    def _build_tools(self) -> list[ToolDef]:
        return [
            # --- Phase 1 tools ---
            ToolDef(
                name="apply_elite_weak",
                description=(
                    "Apply Elite (+2 stats, +HP, +1 level) or Weak (-2 stats, -HP, -1 level) "
                    "adjustment to a creature. Looks up the creature and returns the fully "
                    "modified stat block with all arithmetic pre-computed."
                ),
                parameters=[
                    ToolParameter(
                        name="creature_name",
                        type="string",
                        description="The creature name to look up (e.g., 'goblin warrior')",
                    ),
                    ToolParameter(
                        name="adjustment",
                        type="string",
                        description="'elite' or 'weak'",
                    ),
                ],
            ),
            ToolDef(
                name="apply_template",
                description=(
                    "Apply a creature template to a creature. Looks up both the creature's stat block "
                    "and the template rules from the database, then returns them side by side so you can "
                    "apply the modifications. Works with any template in the database (ghost, skeletal, "
                    "zombie, vampire, graveknight, werecreature, lich, and many more)."
                ),
                parameters=[
                    ToolParameter(
                        name="creature_name",
                        type="string",
                        description="The base creature name (e.g., 'goblin warrior')",
                    ),
                    ToolParameter(
                        name="template_name",
                        type="string",
                        description="Template to apply (e.g., 'ghost', 'skeletal', 'vampire', 'graveknight')",
                    ),
                ],
            ),
            ToolDef(
                name="get_creature_stats",
                description=(
                    "Return a creature's stat block as structured JSON for calculation use. "
                    "Includes level, hp, ac, saves, attacks, skills, perception, speed. "
                    "Distinct from lookup(type='creature') which returns narrative text."
                ),
                parameters=[
                    ToolParameter(
                        name="creature_name",
                        type="string",
                        description="The creature name to look up",
                    ),
                ],
            ),
            # --- Phase 5 tools ---
            ToolDef(
                name="scaffold_creature",
                description=(
                    "Generate a baseline stat block from GM Core creation tables. "
                    "type='creature' (default): standard creature scaffold by role. "
                    "type='troop': group of creatures acting as one, with HP thresholds and area attacks. "
                    "type='swarm': mass of Tiny creatures with auto-damage and swarm defenses."
                ),
                parameters=[
                    ToolParameter(
                        name="level",
                        type="integer",
                        description="Creature level (-1 to 25)",
                    ),
                    ToolParameter(
                        name="type",
                        type="string",
                        description="Creature variant: 'creature', 'troop', or 'swarm'",
                        required=False,
                        default="creature",
                    ),
                    ToolParameter(
                        name="role",
                        type="string",
                        description="Creature role (creature type only): 'brute', 'sniper', 'skirmisher', 'soldier', 'spellcaster'",
                        required=False,
                        default="soldier",
                    ),
                    ToolParameter(
                        name="size",
                        type="string",
                        description="Size for troop ('large','huge','gargantuan') or swarm ('large','huge')",
                        required=False,
                        default="large",
                    ),
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Optional name for the scaffold",
                        required=False,
                        default="",
                    ),
                ],
            ),
            ToolDef(
                name="scaffold_hazard",
                description=(
                    "Generate a baseline hazard stat block from GM Core hazard creation tables. "
                    "type='trap' (default): physical trap with AC, HP, attack, disable DC. "
                    "type='haunt': spiritual hazard immune to physical damage, disabled with Religion/Occultism; "
                    "returns stat block plus JSON config for start_subsystem(type='haunt')."
                ),
                parameters=[
                    ToolParameter(
                        name="level",
                        type="integer",
                        description="Hazard level (-1 to 25)",
                    ),
                    ToolParameter(
                        name="type",
                        type="string",
                        description="Hazard variant: 'trap' or 'haunt'",
                        required=False,
                        default="trap",
                    ),
                    ToolParameter(
                        name="complexity",
                        type="string",
                        description="'simple' or 'complex'",
                        required=False,
                        default="simple",
                    ),
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Optional hazard name",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="trigger", type="string",
                        description="(haunt) Trigger condition",
                        required=False, default="When a living creature enters the area",
                    ),
                    ToolParameter(
                        name="disable_skills", type="string",
                        description="(haunt) Comma-separated disable skills (default: 'Religion,Occultism')",
                        required=False, default="",
                    ),
                    ToolParameter(
                        name="successes_needed", type="integer",
                        description="(haunt) Successes needed per condition (default: 1 simple, 2 complex)",
                        required=False,
                    ),
                    ToolParameter(
                        name="anchor", type="string",
                        description="(haunt) Object or location the haunt is anchored to",
                        required=False, default="",
                    ),
                    ToolParameter(
                        name="reset_interval", type="string",
                        description="(haunt) Time until haunt resets (default: '1 day')",
                        required=False, default="1 day",
                    ),
                ],
            ),
            ToolDef(
                name="design_creature",
                description=(
                    "Design a custom PF2e creature. Without concept/abilities: returns a pure stat scaffold. "
                    "With concept or abilities: uses an LLM backend to generate custom abilities scaled to level. "
                    "Requires backend to be configured for the LLM-assisted path."
                ),
                parameters=[
                    ToolParameter(
                        name="level", type="integer",
                        description="Creature level (-1 to 25). Required for pure scaffold path.",
                        required=False,
                    ),
                    ToolParameter(
                        name="role", type="string",
                        description="Creature role: brute, sniper, skirmisher, soldier, spellcaster",
                        required=False, default="soldier",
                    ),
                    ToolParameter(
                        name="concept", type="string",
                        description="Freeform concept description (e.g., 'A forge-bound fire elemental that mourns its imprisonment')",
                        required=False, default="",
                    ),
                    ToolParameter(
                        name="name", type="string",
                        description="Creature name",
                        required=False, default="",
                    ),
                    ToolParameter(
                        name="abilities", type="string",
                        description="Ability descriptions: JSON array or newline-separated list",
                        required=False, default="",
                    ),
                    ToolParameter(
                        name="save_to_campaign", type="boolean",
                        description="Persist to CharacterStore as a monster (requires campaign context)",
                        required=False, default=False,
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "apply_elite_weak":
                return self._apply_elite_weak(args["creature_name"], args["adjustment"])
            elif name == "apply_template":
                return self._apply_template(args["creature_name"], args["template_name"])
            elif name == "get_creature_stats":
                return self._get_creature_stats(args["creature_name"])
            elif name == "scaffold_creature":
                creature_type = args.get("type", "creature").lower()
                if creature_type == "troop":
                    return self._scaffold_troop(
                        args["level"],
                        args.get("size", "large"),
                        args.get("name", "") or "Custom Troop",
                    )
                elif creature_type == "swarm":
                    return self._scaffold_swarm(
                        args["level"],
                        args.get("size", "large"),
                        args.get("name", "") or "Custom Swarm",
                    )
                else:
                    return self._scaffold_creature(
                        args["level"],
                        args.get("role", "soldier"),
                        args.get("name", "") or "Custom Creature",
                    )
            elif name == "scaffold_hazard":
                hazard_type = args.get("type", "trap").lower()
                if hazard_type == "haunt":
                    return self._scaffold_haunt(
                        args["level"],
                        args.get("name", "") or "Custom Haunt",
                        args.get("trigger", "When a living creature enters the area"),
                        args.get("complexity", "simple"),
                        args.get("disable_skills", ""),
                        args.get("successes_needed"),
                        args.get("anchor", ""),
                        args.get("reset_interval", "1 day"),
                    )
                else:
                    return self._scaffold_hazard(
                        args["level"],
                        args.get("complexity", "simple"),
                        args.get("name", "") or "Custom Hazard",
                    )
            elif name == "design_creature":
                return self._design_creature(
                    args.get("level"),
                    args.get("role", "soldier"),
                    args.get("concept", ""),
                    args.get("name", ""),
                    args.get("abilities", ""),
                    args.get("save_to_campaign", False),
                )
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    # -------------------------------------------------------------------
    # Internal: creature lookup helper
    # -------------------------------------------------------------------

    def _find_creature(self, name: str) -> tuple[dict | None, dict[str, Any]]:
        """Look up a creature and return (result_dict, parsed_metadata).

        Returns (None, {}) if not found.
        """
        results = self.search.search(name, doc_type="creature", top_k=1)
        if not results:
            results = self.search.search(name, category="creature", top_k=3)
        if not results:
            return None, {}

        result = results[0]
        metadata = _parse_metadata(result)
        return result, metadata

    # -------------------------------------------------------------------
    # Tool: apply_elite_weak
    # -------------------------------------------------------------------

    def _apply_elite_weak(self, creature_name: str, adjustment: str) -> ToolResult:
        adjustment = adjustment.lower().strip()
        if adjustment not in ("elite", "weak"):
            return ToolResult(success=False, error=f"Invalid adjustment '{adjustment}'. Must be 'elite' or 'weak'.")

        result, metadata = self._find_creature(creature_name)
        if result is None:
            return ToolResult(success=False, error=f"Creature '{creature_name}' not found.")

        name = result.get("name", creature_name)
        level = metadata.get("level", 0)

        # Build stat dict from metadata
        stats = {
            "level": level,
            "hp": metadata.get("hp"),
            "ac": metadata.get("ac"),
            "saves": metadata.get("saves", {}),
            "perception": metadata.get("perception"),
            "speed": metadata.get("speed", {}),
            "skills": metadata.get("skills", {}),
            "attacks": metadata.get("attacks", []),
        }

        if adjustment == "elite":
            modified = _apply_elite_adjustments(stats, level, self._elite_hp)
        else:
            modified = _apply_weak_adjustments(stats, level, self._elite_hp)

        output = _format_stat_block(name, modified, adjustment)
        return ToolResult(success=True, data=output)

    # -------------------------------------------------------------------
    # Tool: apply_template
    # -------------------------------------------------------------------

    def _apply_template(self, creature_name: str, template_name: str) -> ToolResult:
        result, metadata = self._find_creature(creature_name)
        if result is None:
            return ToolResult(success=False, error=f"Creature '{creature_name}' not found.")

        name = result.get("name", creature_name)
        level = metadata.get("level", 0)

        stats = {
            "level": level,
            "hp": metadata.get("hp"),
            "ac": metadata.get("ac"),
            "saves": metadata.get("saves", {}),
            "perception": metadata.get("perception"),
            "speed": metadata.get("speed", {}),
            "skills": metadata.get("skills", {}),
            "attacks": metadata.get("attacks", []),
        }

        # Search for the template rules in the RAG database
        template_results = self.search.search(
            template_name,
            include_types=["creature_template"],
            top_k=3,
        )

        lines = [
            _format_stat_block(name, stats),
            "\n---\n",
        ]

        if template_results:
            lines.insert(0, f"**Applying '{template_name}' template to {name}.** "
                         "Creature stats and template rules shown side by side.\n")
            for tr in template_results:
                page_info = f", p.{tr['page']}" if tr.get("page") else ""
                lines.append(f"**{tr['name']}** — {tr.get('book', '')}{page_info}")
                content = tr.get("content", "")
                if len(content) > 2000:
                    content = content[:2000] + "..."
                lines.append(content)
                lines.append("")
        else:
            lines.insert(0, "")
            lines.append(f"No creature template matching '{template_name}' found in the database.")

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Tool: get_creature_stats
    # -------------------------------------------------------------------

    def _get_creature_stats(self, creature_name: str) -> ToolResult:
        result, metadata = self._find_creature(creature_name)
        if result is None:
            return ToolResult(success=False, error=f"Creature '{creature_name}' not found.")

        name = result.get("name", creature_name)
        stats = {
            "name": name,
            "book": result.get("book", ""),
            "page": result.get("page"),
            "level": metadata.get("level"),
            "hp": metadata.get("hp"),
            "ac": metadata.get("ac"),
            "perception": metadata.get("perception"),
            "saves": metadata.get("saves", {}),
            "speed": metadata.get("speed", {}),
            "attacks": metadata.get("attacks", []),
            "skills": metadata.get("skills", {}),
        }

        # Include any additional metadata fields
        for key in ("abilities", "immunities", "resistances", "weaknesses", "traits", "spells"):
            if key in metadata:
                stats[key] = metadata[key]

        return ToolResult(success=True, data=json.dumps(stats, indent=2))

    # -------------------------------------------------------------------
    # Tool: scaffold_creature (Phase 5)
    # -------------------------------------------------------------------

    def _scaffold_creature(self, level: int, role: str, name: str) -> ToolResult:
        level = max(-1, min(25, level))
        role = role.lower().strip()

        baseline = self._creature_stats.get(level)
        if baseline is None:
            return ToolResult(success=False, error=f"No baseline stats for level {level}.")

        stats = dict(baseline)
        stats["level"] = level

        # Apply role adjustments
        role_adj = self._role_adjustments.get(role, {})
        role_notes = role_adj.get("notes", "")

        # Multiplicative adjustments
        if "hp_mult" in role_adj:
            stats["hp"] = int(stats["hp"] * role_adj["hp_mult"])
        if "damage_mult" in role_adj:
            stats["damage_avg"] = int(stats["damage_avg"] * role_adj["damage_mult"])

        # Additive adjustments
        for key in ("ac", "fort", "ref", "will", "perception", "attack", "spell_dc", "skill_high", "skill_mod"):
            if key in role_adj:
                stats[key] = stats.get(key, 0) + role_adj[key]

        speed_bonus = role_adj.get("speed_bonus", 0)

        # Format output
        role_label = f" ({role.title()})" if role != "soldier" else ""
        lines = [
            f"**{name}{role_label}** — Level {level} Scaffold",
            f"**HP** {stats['hp']}",
            f"**AC** {stats['ac']}",
            f"**Fort** +{stats['fort']}, **Ref** +{stats['ref']}, **Will** +{stats['will']}",
            f"**Perception** +{stats['perception']}",
            f"**Speed** {25 + speed_bonus} ft.",
            f"**Attack** +{stats['attack']}, **Average Damage** {stats['damage_avg']}",
            f"**Spell DC** {stats['spell_dc']}",
            f"**Skills** High +{stats['skill_high']}, Moderate +{stats['skill_mod']}",
        ]

        if role_notes:
            lines.append(f"\n*{role_notes}*")

        lines.append(
            "\n*These are moderate baseline values from GM Core Table 2-1, "
            "adjusted for role. Customize freely.*"
        )

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Tool: scaffold_hazard (Phase 5)
    # -------------------------------------------------------------------

    def _scaffold_hazard(self, level: int, complexity: str, name: str) -> ToolResult:
        level = max(-1, min(25, level))
        complexity = complexity.lower().strip()

        baseline = self._hazard_stats.get(level)
        if baseline is None:
            return ToolResult(success=False, error=f"No baseline stats for hazard level {level}.")

        stats = dict(baseline)

        # Complex hazards get more HP and a routine
        if complexity == "complex":
            stats["hp"] = int(stats["hp"] * 1.5)
            stats["hardness"] = int(stats["hardness"] * 1.2)

        complexity_label = complexity.title()
        lines = [
            f"**{name}** — Level {level} {complexity_label} Hazard Scaffold",
            f"**Stealth DC** {stats['stealth_dc']} (to detect)",
            f"**Disable DC** {stats['disable_dc']}",
            f"**AC** {stats['ac']}; **Fort** +{stats['fort']}, **Ref** +{stats['ref']}",
            f"**HP** {stats['hp']}; **Hardness** {stats['hardness']}",
            f"**Attack** +{stats['attack']}; **Damage** {stats['damage']}",
        ]

        if complexity == "complex":
            lines.append(
                "\n**Routine** (2 actions per round)\n"
                "- Action 1: *[Define action]*\n"
                "- Action 2: *[Define action]*"
            )
        else:
            lines.append("\n**Reaction** *[Define trigger and effect]*")

        lines.append(
            "\n*Baseline values from GM Core hazard creation table. Customize freely.*"
        )

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Tool: scaffold_troop
    # -------------------------------------------------------------------

    def _scaffold_troop(self, level: int, size: str, name: str) -> ToolResult:
        level = max(-1, min(25, level))
        size = size.lower().strip()

        size_data = self._troop_size_table.get(size)
        if size_data is None:
            return ToolResult(
                success=False,
                error=f"Invalid troop size '{size}'. Must be: large, huge, gargantuan",
            )

        baseline = self._creature_stats.get(level)
        if baseline is None:
            return ToolResult(success=False, error=f"No baseline stats for level {level}.")

        hp = int(baseline["hp"] * size_data["hp_mult"])
        ac = baseline["ac"] + size_data["ac_adj"]
        diminished_hp = int(hp * 2 / 3)
        broken_hp = int(hp / 3)

        lines = [
            f"**{name}** — Level {level} Troop ({size.title()}, {size_data['area']})",
            f"**HP** {hp} (Diminished at {diminished_hp}, Broken at {broken_hp})",
            f"**AC** {ac}",
            f"**Fort** +{baseline['fort']}, **Ref** +{baseline['ref']}, **Will** +{baseline['will']}",
            f"**Perception** +{baseline['perception']}",
            f"**Speed** 25 ft.",
            "",
            f"**Troop Attack** (area, {size_data['area']}): {baseline['damage_avg']} damage; "
            f"diminished {max(1, baseline['damage_avg'] * 2 // 3)}, broken {max(1, baseline['damage_avg'] // 3)}",
            "",
            TROOP_DEFENSES,
            "",
            "**Form Up** [1 action] The troop reassembles if it's been disrupted.",
        ]

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Tool: scaffold_swarm
    # -------------------------------------------------------------------

    def _scaffold_swarm(self, level: int, size: str, name: str) -> ToolResult:
        level = max(-1, min(25, level))
        size = size.lower().strip()

        size_data = self._swarm_size_table.get(size)
        if size_data is None:
            return ToolResult(
                success=False,
                error=f"Invalid swarm size '{size}'. Must be: large, huge",
            )

        baseline = self._creature_stats.get(level)
        if baseline is None:
            return ToolResult(success=False, error=f"No baseline stats for level {level}.")

        hp = int(baseline["hp"] * size_data["hp_mult"])
        phys_resist = max(2, level + 2)
        area_weakness = max(3, level + 3)
        auto_damage = max(1, baseline["damage_avg"] // 2)

        lines = [
            f"**{name}** — Level {level} Swarm ({size.title()}, {size_data['area']})",
            f"**HP** {hp}",
            f"**AC** {baseline['ac'] - 2}",
            f"**Fort** +{baseline['fort']}, **Ref** +{baseline['ref'] + 2}, **Will** +{baseline['will'] - 2}",
            f"**Perception** +{baseline['perception']}",
            f"**Speed** 25 ft.",
            "",
            f"**Resistance** physical {phys_resist} (except splash)",
            f"**Weakness** area damage {area_weakness}",
            "",
            SWARM_DEFENSES,
            "",
            f"**Swarming Bites** (automatic) Each enemy in the swarm's space takes "
            f"{auto_damage} piercing damage at the start of their turn.",
        ]

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Tool: scaffold_haunt
    # -------------------------------------------------------------------

    def _scaffold_haunt(
        self,
        level: int,
        name: str,
        trigger: str,
        complexity: str,
        disable_skills: str,
        successes_needed: int | None,
        anchor: str,
        reset_interval: str,
    ) -> ToolResult:
        level = max(-1, min(25, level))
        complexity = complexity.lower().strip()

        baseline = self._hazard_stats.get(level)
        if baseline is None:
            return ToolResult(success=False, error=f"No baseline stats for haunt level {level}.")

        stats = dict(baseline)

        # Parse disable skills
        skills = [s.strip() for s in disable_skills.split(",") if s.strip()] if disable_skills else [
            "Religion", "Occultism"
        ]

        # Default successes_needed based on complexity
        if successes_needed is None:
            successes_needed = 2 if complexity == "complex" else 1

        # Build disable conditions
        disable_conditions = [
            {"skill": skill, "dc": stats["disable_dc"], "successes_needed": successes_needed}
            for skill in skills
        ]

        # Build config dict ready for start_subsystem
        config = {
            "hp": stats["hp"],
            "hardness": 0,
            "routine_actions": [],
            "disable_conditions": disable_conditions,
            "trigger_condition": trigger,
            "anchor": anchor,
            "reset_interval": reset_interval,
            "spiritual_immunity": True,
        }

        complexity_label = complexity.title()
        lines = [
            f"**{name}** — Level {level} {complexity_label} Haunt",
            f"**Stealth DC** {stats['stealth_dc']} (Occultism/Religion to detect)",
            f"**Haunt NPC** [reaction]",
            f"  **Trigger** {trigger}",
            "",
            f"**HP** {stats['hp']}",
            "**Immune** to physical damage (spiritual immunity)",
            "",
            "**Disable Conditions:**",
        ]

        for cond in disable_conditions:
            note = f" ({successes_needed} successes needed)" if successes_needed > 1 else ""
            lines.append(f"  - {cond['skill']} DC {cond['dc']}{note}")

        if anchor:
            lines.append(f"\n**Anchor:** {anchor}")
        lines.append(f"**Reset Interval:** {reset_interval}")
        lines.append("\n**Effect:** [Define the haunt's effect here]")
        lines.append(
            "\n*Pass the config below to start_subsystem(type='haunt', ...) to begin tracking.*"
        )
        lines.append(f"\n**Config:**\n```json\n{json.dumps(config, indent=2)}\n```")

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Tool: design_creature
    # -------------------------------------------------------------------

    def _design_creature(
        self,
        level: int | None,
        role: str,
        concept: str,
        name: str,
        abilities: str,
        save_to_campaign: bool,
    ) -> ToolResult:
        needs_llm = bool(concept or abilities)

        # Pure scaffold path
        if not needs_llm:
            if level is None:
                return ToolResult(
                    success=False,
                    error="'level' is required when no concept or abilities are provided.",
                )
            return self._scaffold_creature(level, role or "soldier", name or "Custom Creature")

        # LLM-assisted path
        if self.backend is None:
            return ToolResult(
                success=False,
                error=(
                    "design_creature with concept/abilities requires an LLM backend. "
                    "Initialize CreatureModifierServer with a backend parameter."
                ),
            )

        effective_level = level
        level_stats: dict[str, Any] = {}
        if effective_level is not None:
            effective_level = max(-1, min(25, effective_level))
            level_stats = dict(self._creature_stats.get(effective_level, {}))
            # Apply role adjustments
            role_adj = self._role_adjustments.get((role or "soldier").lower(), {})
            if "hp_mult" in role_adj:
                level_stats["hp"] = int(level_stats["hp"] * role_adj["hp_mult"])
            for key in ("ac", "fort", "ref", "will", "perception", "attack", "spell_dc"):
                if key in role_adj:
                    level_stats[key] = level_stats.get(key, 0) + role_adj[key]

        # Build prompt
        user_parts: list[str] = []
        if name:
            user_parts.append(f"Creature name: {name}")
        if effective_level is not None:
            user_parts.append(f"Level: {effective_level} (role: {role or 'soldier'})")
        else:
            user_parts.append("Infer an appropriate level from the concept and include 'level' in your JSON output.")
        if concept:
            user_parts.append(f"Concept: {concept}")
        if abilities:
            user_parts.append(f"Abilities to include:\n{abilities}")
        if level_stats:
            user_parts.append(f"Level stats (scale damage/DCs to match):\n{json.dumps(level_stats, indent=2)}")

        user_prompt = "\n\n".join(user_parts)

        try:
            from gm_agent.models.base import Message  # pylint: disable=import-outside-toplevel
            messages = [
                Message(role="system", content=CREATURE_DESIGN_SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ]
            response = self.backend.chat(messages, tools=[], temperature=0.3)

            # Extract JSON from possible markdown fences
            text = response.text.strip()
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            data = json.loads(text)

        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"LLM returned invalid JSON: {e}")
        except Exception as e:  # pylint: disable=broad-except
            return ToolResult(success=False, error=f"LLM call failed: {e}")

        # Determine final level
        result_level = data.get("level", effective_level)
        if result_level is None:
            result_level = 1
        result_level = max(-1, min(25, result_level))

        # Build scaffold header
        creature_name = data.get("name") or name or "Custom Creature"
        scaffold_result = self._scaffold_creature(result_level, role or "soldier", creature_name)
        scaffold_text = scaffold_result.data if scaffold_result.success else ""

        # Append LLM-generated abilities
        creature_abilities = data.get("abilities", [])
        lines = [scaffold_text, "\n**Custom Abilities:**"]
        for ab in creature_abilities:
            action_cost = ab.get("action_cost", "")
            traits = ab.get("traits", [])
            trait_str = f" [{', '.join(traits)}]" if traits else ""
            lines.append(f"\n**{ab.get('name', 'Ability')}** {action_cost}{trait_str}")
            lines.append(ab.get("description", ""))

        if save_to_campaign:
            lines.append(
                "\n*Note: save_to_campaign requires a campaign-scoped server instance "
                "with campaign_id set.*"
            )

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Close the search connection."""
        self.search.close()
