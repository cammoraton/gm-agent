"""Tests for CreatureModifierServer — Phase 1 + Phase 5."""

import json
import pytest
from unittest.mock import patch, MagicMock

from gm_agent.mcp.creature_modifier import (
    CreatureModifierServer,
    ELITE_HP_ADJUSTMENT,
    CREATURE_STATS_BY_LEVEL,
    HAZARD_STATS_BY_LEVEL,
    ROLE_ADJUSTMENTS,
    _adjust_damage_per_die,
    _apply_elite_adjustments,
    _apply_weak_adjustments,
    _get_elite_hp_adjustment,
)


# ---------------------------------------------------------------------------
# Unit tests for adjustment math (no mocks needed)
# ---------------------------------------------------------------------------

class TestDamageAdjustment:
    """Test damage-per-die adjustment arithmetic."""

    def test_simple_damage(self):
        assert _adjust_damage_per_die("1d6+3 slashing", 2) == "1d6+5 slashing"

    def test_multi_die_damage(self):
        # 2 dice * +2 = +4, plus existing +3 = +7
        assert _adjust_damage_per_die("2d6+3 slashing", 2) == "2d6+7 slashing"

    def test_negative_adjustment(self):
        # 2 dice * -2 = -4, plus existing +5 = +1
        assert _adjust_damage_per_die("2d6+5 slashing", -2) == "2d6+1 slashing"

    def test_adjustment_to_zero_bonus(self):
        # 1 die * -3 = -3, plus existing +3 = 0
        assert _adjust_damage_per_die("1d6+3 slashing", -3) == "1d6 slashing"

    def test_adjustment_to_negative_bonus(self):
        # 1 die * -2 = -2, plus existing +1 = -1
        assert _adjust_damage_per_die("1d6+1 slashing", -2) == "1d6-1 slashing"

    def test_no_bonus_damage(self):
        # 2 dice * +2 = +4
        assert _adjust_damage_per_die("2d8 fire", 2) == "2d8+4 fire"

    def test_invalid_damage_string(self):
        # Non-standard format returned unchanged
        assert _adjust_damage_per_die("special damage", 2) == "special damage"


class TestEliteHPAdjustment:
    """Test HP adjustment table lookups."""

    def test_level_minus_one(self):
        assert _get_elite_hp_adjustment(-1) == 10

    def test_level_five(self):
        assert _get_elite_hp_adjustment(5) == 20

    def test_level_ten(self):
        assert _get_elite_hp_adjustment(10) == 30

    def test_level_twenty(self):
        assert _get_elite_hp_adjustment(20) == 65

    def test_level_twenty_five(self):
        assert _get_elite_hp_adjustment(25) == 90

    def test_hp_table_coverage(self):
        """All levels -1 through 25 should have entries."""
        for level in range(-1, 26):
            assert level in ELITE_HP_ADJUSTMENT, f"Missing HP adjustment for level {level}"


class TestEliteWeakAdjustments:
    """Test Elite/Weak stat modifications."""

    SAMPLE_STATS = {
        "level": 5,
        "hp": 70,
        "ac": 22,
        "saves": {"Fort": 12, "Ref": 12, "Will": 11},
        "perception": 11,
        "speed": {"land": 25},
        "skills": {"Acrobatics": 14, "Stealth": 14},
        "attacks": [
            {"name": "Dogslicer", "bonus": 15, "damage": "2d6+5 slashing"},
        ],
    }

    def test_elite_level_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["level"] == 6

    def test_elite_hp_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["hp"] == 70 + ELITE_HP_ADJUSTMENT[5]

    def test_elite_ac_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["ac"] == 24

    def test_elite_saves_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["saves"]["Fort"] == 14
        assert result["saves"]["Ref"] == 14
        assert result["saves"]["Will"] == 13

    def test_elite_perception_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["perception"] == 13

    def test_elite_skills_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["skills"]["Acrobatics"] == 16

    def test_elite_attack_bonus_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        assert result["attacks"][0]["bonus"] == 17

    def test_elite_damage_increase(self):
        result = _apply_elite_adjustments(self.SAMPLE_STATS, 5)
        # 2d6+5 with +2 per die = 2d6+9
        assert result["attacks"][0]["damage"] == "2d6+9 slashing"

    def test_weak_level_decrease(self):
        result = _apply_weak_adjustments(self.SAMPLE_STATS, 5)
        assert result["level"] == 4

    def test_weak_hp_decrease(self):
        result = _apply_weak_adjustments(self.SAMPLE_STATS, 5)
        assert result["hp"] == 70 - ELITE_HP_ADJUSTMENT[5]

    def test_weak_hp_minimum_one(self):
        """Weak HP should never go below 1."""
        tiny_stats = {**self.SAMPLE_STATS, "hp": 5}
        result = _apply_weak_adjustments(tiny_stats, 5)
        assert result["hp"] >= 1

    def test_weak_ac_decrease(self):
        result = _apply_weak_adjustments(self.SAMPLE_STATS, 5)
        assert result["ac"] == 20

    def test_weak_saves_decrease(self):
        result = _apply_weak_adjustments(self.SAMPLE_STATS, 5)
        assert result["saves"]["Fort"] == 10

    def test_weak_attack_bonus_decrease(self):
        result = _apply_weak_adjustments(self.SAMPLE_STATS, 5)
        assert result["attacks"][0]["bonus"] == 13

    def test_weak_damage_decrease(self):
        result = _apply_weak_adjustments(self.SAMPLE_STATS, 5)
        # 2d6+5 with -2 per die = 2d6+1
        assert result["attacks"][0]["damage"] == "2d6+1 slashing"


# ---------------------------------------------------------------------------
# Server tests using mock_pathfinder_search
# ---------------------------------------------------------------------------

class TestCreatureModifierServer:
    """Tests for CreatureModifierServer MCP tools."""

    @pytest.fixture
    def mock_search(self):
        """Create a mock PathfinderSearch."""
        mock = MagicMock()

        def search_side_effect(query, **kwargs):
            if "goblin warrior" in query.lower() or "goblin" in query.lower():
                return [
                    {
                        "name": "Goblin Warrior",
                        "type": "creature",
                        "book": "Monster Core",
                        "page": 178,
                        "content": "A goblin warrior fights with cunning...",
                        "metadata": json.dumps({
                            "level": 1,
                            "hp": 15,
                            "ac": 16,
                            "perception": 5,
                            "saves": {"Fort": 6, "Ref": 9, "Will": 3},
                            "speed": {"land": 25},
                            "attacks": [
                                {"name": "Dogslicer", "bonus": 7, "damage": "1d6+1 slashing"},
                            ],
                            "skills": {"Acrobatics": 7, "Stealth": 7},
                        }),
                    }
                ]
            if "ghost" in query.lower() and kwargs.get("include_types") == ["creature_template"]:
                return [
                    {
                        "name": "Creating a Ghost",
                        "type": "creature_template",
                        "book": "Monster Core",
                        "page": 155,
                        "content": "Ghost template rules...",
                    }
                ]
            return []

        mock.search.side_effect = search_side_effect
        return mock

    @pytest.fixture
    def server(self, mock_search):
        """Create CreatureModifierServer with mocked search."""
        with patch("gm_agent.mcp.creature_modifier.PathfinderSearch", return_value=mock_search):
            s = CreatureModifierServer()
            yield s

    def test_list_tools(self, server):
        tools = server.list_tools()
        tool_names = [t.name for t in tools]
        assert "apply_elite_weak" in tool_names
        assert "apply_template" in tool_names
        assert "get_creature_stats" in tool_names
        assert "scaffold_creature" in tool_names
        assert "scaffold_hazard" in tool_names
        assert "scaffold_troop" in tool_names
        assert "scaffold_swarm" in tool_names
        assert len(tools) == 7

    # --- apply_elite_weak tests ---

    def test_apply_elite(self, server):
        result = server.call_tool("apply_elite_weak", {
            "creature_name": "goblin warrior",
            "adjustment": "elite",
        })
        assert result.success
        assert "Elite" in result.data
        assert "Level 2" in result.data  # 1+1
        assert "25" in result.data  # HP: 15+10
        assert "18" in result.data  # AC: 16+2

    def test_apply_weak(self, server):
        result = server.call_tool("apply_elite_weak", {
            "creature_name": "goblin warrior",
            "adjustment": "weak",
        })
        assert result.success
        assert "Weak" in result.data
        assert "Level 0" in result.data  # 1-1

    def test_apply_invalid_adjustment(self, server):
        result = server.call_tool("apply_elite_weak", {
            "creature_name": "goblin warrior",
            "adjustment": "mega",
        })
        assert not result.success
        assert "Invalid adjustment" in result.error

    def test_apply_creature_not_found(self, server):
        result = server.call_tool("apply_elite_weak", {
            "creature_name": "nonexistent creature",
            "adjustment": "elite",
        })
        assert not result.success
        assert "not found" in result.error

    # --- apply_template tests ---

    def test_apply_ghost_template(self, server):
        result = server.call_tool("apply_template", {
            "creature_name": "goblin warrior",
            "template_name": "ghost",
        })
        assert result.success
        # Should show creature stats and template rules from RAG
        assert "Goblin Warrior" in result.data
        assert "Creating a Ghost" in result.data
        assert "Ghost template rules" in result.data

    def test_apply_template_no_match(self, server):
        result = server.call_tool("apply_template", {
            "creature_name": "goblin warrior",
            "template_name": "unicorn sparkle",
        })
        assert result.success
        # Should show creature stats but note no template found
        assert "Goblin Warrior" in result.data
        assert "No creature template matching" in result.data

    def test_apply_template_creature_not_found(self, server):
        result = server.call_tool("apply_template", {
            "creature_name": "nonexistent",
            "template_name": "ghost",
        })
        assert not result.success
        assert "not found" in result.error

    # --- get_creature_stats tests ---

    def test_get_creature_stats(self, server):
        result = server.call_tool("get_creature_stats", {
            "creature_name": "goblin warrior",
        })
        assert result.success
        stats = json.loads(result.data)
        assert stats["name"] == "Goblin Warrior"
        assert stats["level"] == 1
        assert stats["hp"] == 15
        assert stats["ac"] == 16
        assert stats["saves"]["Fort"] == 6
        assert len(stats["attacks"]) == 1
        assert stats["attacks"][0]["name"] == "Dogslicer"

    def test_get_creature_stats_not_found(self, server):
        result = server.call_tool("get_creature_stats", {
            "creature_name": "nonexistent",
        })
        assert not result.success

    # --- scaffold_creature tests (Phase 5) ---

    def test_scaffold_creature_level_5(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 5,
        })
        assert result.success
        assert "Level 5" in result.data
        assert "HP" in result.data
        assert "AC" in result.data

    def test_scaffold_creature_with_role(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 5,
            "role": "brute",
            "name": "Test Brute",
        })
        assert result.success
        assert "Brute" in result.data
        assert "Test Brute" in result.data
        # Brute should have higher HP than baseline
        baseline_hp = CREATURE_STATS_BY_LEVEL[5]["hp"]
        brute_hp = int(baseline_hp * ROLE_ADJUSTMENTS["brute"]["hp_mult"])
        assert str(brute_hp) in result.data

    def test_scaffold_creature_all_roles(self, server):
        """All roles should produce valid output."""
        for role in ("brute", "sniper", "skirmisher", "soldier", "spellcaster"):
            result = server.call_tool("scaffold_creature", {
                "level": 3,
                "role": role,
            })
            assert result.success, f"Failed for role {role}"
            assert "Level 3" in result.data

    def test_scaffold_creature_various_levels(self, server):
        """Test scaffold at boundary levels."""
        for level in (-1, 0, 1, 10, 20, 25):
            result = server.call_tool("scaffold_creature", {
                "level": level,
            })
            assert result.success, f"Failed for level {level}"
            assert f"Level {level}" in result.data

    def test_scaffold_creature_level_clamping(self, server):
        """Levels outside -1..25 should be clamped."""
        result = server.call_tool("scaffold_creature", {"level": 30})
        assert result.success
        assert "Level 25" in result.data

        result = server.call_tool("scaffold_creature", {"level": -5})
        assert result.success
        assert "Level -1" in result.data

    # --- scaffold_hazard tests (Phase 5) ---

    def test_scaffold_hazard_simple(self, server):
        result = server.call_tool("scaffold_hazard", {
            "level": 5,
            "complexity": "simple",
        })
        assert result.success
        assert "Simple Hazard" in result.data
        assert "Reaction" in result.data

    def test_scaffold_hazard_complex(self, server):
        result = server.call_tool("scaffold_hazard", {
            "level": 5,
            "complexity": "complex",
        })
        assert result.success
        assert "Complex Hazard" in result.data
        assert "Routine" in result.data
        # Complex should have higher HP than simple
        baseline_hp = HAZARD_STATS_BY_LEVEL[5]["hp"]
        complex_hp = int(baseline_hp * 1.5)
        assert str(complex_hp) in result.data

    def test_scaffold_hazard_named(self, server):
        result = server.call_tool("scaffold_hazard", {
            "level": 3,
            "name": "Poison Dart Trap",
        })
        assert result.success
        assert "Poison Dart Trap" in result.data

    def test_scaffold_hazard_various_levels(self, server):
        """Test hazard scaffold at various levels."""
        for level in (-1, 0, 5, 10, 20, 25):
            result = server.call_tool("scaffold_hazard", {"level": level})
            assert result.success, f"Failed for level {level}"

    # --- scaffold_troop tests ---

    def test_scaffold_troop_large(self, server):
        result = server.call_tool("scaffold_troop", {
            "level": 5,
            "size": "large",
            "name": "Goblin Troop",
        })
        assert result.success
        assert "Goblin Troop" in result.data
        assert "Level 5" in result.data
        assert "Large" in result.data
        assert "Troop Defenses" in result.data
        assert "Diminished" in result.data
        assert "Broken" in result.data

    def test_scaffold_troop_huge(self, server):
        result = server.call_tool("scaffold_troop", {
            "level": 10,
            "size": "huge",
            "name": "Skeleton Horde",
        })
        assert result.success
        assert "Skeleton Horde" in result.data
        assert "Huge" in result.data
        assert "15 ft." in result.data

    def test_scaffold_troop_gargantuan(self, server):
        result = server.call_tool("scaffold_troop", {
            "level": 15,
            "size": "gargantuan",
            "name": "Army of the Dead",
        })
        assert result.success
        assert "Gargantuan" in result.data
        assert "20 ft." in result.data

    def test_scaffold_troop_invalid_size(self, server):
        result = server.call_tool("scaffold_troop", {
            "level": 5,
            "size": "tiny",
            "name": "Bad Troop",
        })
        assert not result.success
        assert "Invalid troop size" in result.error

    # --- scaffold_swarm tests ---

    def test_scaffold_swarm_large(self, server):
        result = server.call_tool("scaffold_swarm", {
            "level": 3,
            "size": "large",
            "name": "Rat Swarm",
        })
        assert result.success
        assert "Rat Swarm" in result.data
        assert "Level 3" in result.data
        assert "Resistance" in result.data
        assert "Weakness" in result.data
        assert "Swarm Defenses" in result.data
        assert "Swarming Bites" in result.data

    def test_scaffold_swarm_huge(self, server):
        result = server.call_tool("scaffold_swarm", {
            "level": 7,
            "size": "huge",
            "name": "Centipede Swarm",
        })
        assert result.success
        assert "Centipede Swarm" in result.data
        assert "Huge" in result.data

    def test_scaffold_swarm_invalid_size(self, server):
        result = server.call_tool("scaffold_swarm", {
            "level": 3,
            "size": "gargantuan",
            "name": "Bad Swarm",
        })
        assert not result.success
        assert "Invalid swarm size" in result.error

    def test_scaffold_troop_level_clamping(self, server):
        """Levels outside -1..25 should be clamped."""
        result = server.call_tool("scaffold_troop", {
            "level": 30, "size": "large", "name": "Test",
        })
        assert result.success
        assert "Level 25" in result.data

    # --- Unknown tool test ---

    def test_unknown_tool(self, server):
        result = server.call_tool("unknown_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error


class TestStatTableCompleteness:
    """Verify stat tables cover all expected levels."""

    def test_creature_stats_all_levels(self):
        for level in range(-1, 26):
            assert level in CREATURE_STATS_BY_LEVEL, f"Missing creature stats for level {level}"
            stats = CREATURE_STATS_BY_LEVEL[level]
            assert "hp" in stats
            assert "ac" in stats
            assert "attack" in stats

    def test_hazard_stats_all_levels(self):
        for level in range(-1, 26):
            assert level in HAZARD_STATS_BY_LEVEL, f"Missing hazard stats for level {level}"
            stats = HAZARD_STATS_BY_LEVEL[level]
            assert "hp" in stats
            assert "ac" in stats
            assert "stealth_dc" in stats
            assert "disable_dc" in stats

    def test_role_adjustments_all_roles(self):
        expected_roles = {"brute", "sniper", "soldier", "skirmisher", "spellcaster"}
        assert set(ROLE_ADJUSTMENTS.keys()) == expected_roles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
