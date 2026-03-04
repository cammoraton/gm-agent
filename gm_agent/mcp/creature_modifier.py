"""Creature modifier MCP server — re-export stub.

Canonical location: gm_agent.systems.pf2e.servers.creature_modifier
"""

from gm_agent.systems.pf2e.servers.creature_modifier import (  # noqa: F401
    CreatureModifierServer,
    ELITE_HP_ADJUSTMENT,
    CREATURE_STATS_BY_LEVEL,
    HAZARD_STATS_BY_LEVEL,
    ROLE_ADJUSTMENTS,
    TROOP_SIZE_TABLE,
    SWARM_SIZE_TABLE,
    _adjust_damage_per_die,
    _apply_elite_adjustments,
    _apply_weak_adjustments,
    _get_elite_hp_adjustment,
)
