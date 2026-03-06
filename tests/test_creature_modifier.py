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
        mock.get_creation_table.return_value = {}
        return mock

    @pytest.fixture
    def server(self, mock_search):
        """Create CreatureModifierServer with mocked search."""
        with patch("gm_agent.systems.pf2e.servers.creature_modifier.PathfinderSearch", return_value=mock_search):
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
        assert "scaffold_troop" not in tool_names
        assert "scaffold_swarm" not in tool_names
        assert "scaffold_haunt" not in tool_names
        assert "design_creature" in tool_names
        assert len(tools) == 6

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

    # --- scaffold_creature(type='troop') tests ---

    def test_scaffold_troop_large(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 5,
            "type": "troop",
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
        result = server.call_tool("scaffold_creature", {
            "level": 10,
            "type": "troop",
            "size": "huge",
            "name": "Skeleton Horde",
        })
        assert result.success
        assert "Skeleton Horde" in result.data
        assert "Huge" in result.data
        assert "15 ft." in result.data

    def test_scaffold_troop_gargantuan(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 15,
            "type": "troop",
            "size": "gargantuan",
            "name": "Army of the Dead",
        })
        assert result.success
        assert "Gargantuan" in result.data
        assert "20 ft." in result.data

    def test_scaffold_troop_invalid_size(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 5,
            "type": "troop",
            "size": "tiny",
            "name": "Bad Troop",
        })
        assert not result.success
        assert "Invalid troop size" in result.error

    # --- scaffold_creature(type='swarm') tests ---

    def test_scaffold_swarm_large(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 3,
            "type": "swarm",
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
        result = server.call_tool("scaffold_creature", {
            "level": 7,
            "type": "swarm",
            "size": "huge",
            "name": "Centipede Swarm",
        })
        assert result.success
        assert "Centipede Swarm" in result.data
        assert "Huge" in result.data

    def test_scaffold_swarm_invalid_size(self, server):
        result = server.call_tool("scaffold_creature", {
            "level": 3,
            "type": "swarm",
            "size": "gargantuan",
            "name": "Bad Swarm",
        })
        assert not result.success
        assert "Invalid swarm size" in result.error

    def test_scaffold_troop_level_clamping(self, server):
        """Levels outside -1..25 should be clamped."""
        result = server.call_tool("scaffold_creature", {
            "level": 30, "type": "troop", "size": "large", "name": "Test",
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


class TestScaffoldHaunt:
    """Tests for scaffold_hazard(type='haunt') tool."""

    @pytest.fixture
    def server(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._creature_stats = CREATURE_STATS_BY_LEVEL
                self._hazard_stats = HAZARD_STATS_BY_LEVEL
                self._elite_hp = ELITE_HP_ADJUSTMENT
                self._role_adjustments = ROLE_ADJUSTMENTS
                from gm_agent.systems.pf2e.servers.creature_modifier import TROOP_SIZE_TABLE, SWARM_SIZE_TABLE
                self._troop_size_table = TROOP_SIZE_TABLE
                self._swarm_size_table = SWARM_SIZE_TABLE
                self._tools = self._build_tools()

        return _Server()

    def test_scaffold_haunt_returns_stat_block(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer, HAZARD_STATS_BY_LEVEL

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._creature_stats = CREATURE_STATS_BY_LEVEL
                self._hazard_stats = HAZARD_STATS_BY_LEVEL
                self._elite_hp = ELITE_HP_ADJUSTMENT
                self._role_adjustments = ROLE_ADJUSTMENTS
                from gm_agent.systems.pf2e.servers.creature_modifier import TROOP_SIZE_TABLE, SWARM_SIZE_TABLE
                self._troop_size_table = TROOP_SIZE_TABLE
                self._swarm_size_table = SWARM_SIZE_TABLE
                self._tools = self._build_tools()

        server = _Server()
        result = server.call_tool("scaffold_hazard", {
            "level": 4,
            "type": "haunt",
            "name": "Weeping Widow",
            "trigger": "When a living creature enters room 7",
            "disable_skills": "Religion,Occultism",
        })
        assert result.success
        assert "Weeping Widow" in result.data
        assert "Level 4" in result.data
        assert "Religion" in result.data
        assert "Occultism" in result.data
        assert "spiritual_immunity" in result.data or "spiritual immunity" in result.data.lower()

    def test_scaffold_haunt_returns_config_json(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._creature_stats = CREATURE_STATS_BY_LEVEL
                self._hazard_stats = HAZARD_STATS_BY_LEVEL
                self._elite_hp = ELITE_HP_ADJUSTMENT
                self._role_adjustments = ROLE_ADJUSTMENTS
                from gm_agent.systems.pf2e.servers.creature_modifier import TROOP_SIZE_TABLE, SWARM_SIZE_TABLE
                self._troop_size_table = TROOP_SIZE_TABLE
                self._swarm_size_table = SWARM_SIZE_TABLE
                self._tools = self._build_tools()

        server = _Server()
        result = server.call_tool("scaffold_hazard", {
            "level": 3,
            "type": "haunt",
            "name": "Blood Oath",
            "trigger": "When entering the chapel",
        })
        assert result.success
        # Config JSON should be embedded
        assert '"disable_conditions"' in result.data
        assert '"spiritual_immunity"' in result.data
        assert '"trigger_condition"' in result.data

    def test_scaffold_haunt_correct_dcs(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import (
            CreatureModifierServer, HAZARD_STATS_BY_LEVEL
        )

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._creature_stats = CREATURE_STATS_BY_LEVEL
                self._hazard_stats = HAZARD_STATS_BY_LEVEL
                self._elite_hp = ELITE_HP_ADJUSTMENT
                self._role_adjustments = ROLE_ADJUSTMENTS
                from gm_agent.systems.pf2e.servers.creature_modifier import TROOP_SIZE_TABLE, SWARM_SIZE_TABLE
                self._troop_size_table = TROOP_SIZE_TABLE
                self._swarm_size_table = SWARM_SIZE_TABLE
                self._tools = self._build_tools()

        server = _Server()
        level = 5
        result = server.call_tool("scaffold_hazard", {"level": level, "type": "haunt", "name": "Test"})
        assert result.success
        expected_dc = str(HAZARD_STATS_BY_LEVEL[level]["disable_dc"])
        assert expected_dc in result.data

    def test_scaffold_haunt_complex_uses_2_successes(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._creature_stats = CREATURE_STATS_BY_LEVEL
                self._hazard_stats = HAZARD_STATS_BY_LEVEL
                self._elite_hp = ELITE_HP_ADJUSTMENT
                self._role_adjustments = ROLE_ADJUSTMENTS
                from gm_agent.systems.pf2e.servers.creature_modifier import TROOP_SIZE_TABLE, SWARM_SIZE_TABLE
                self._troop_size_table = TROOP_SIZE_TABLE
                self._swarm_size_table = SWARM_SIZE_TABLE
                self._tools = self._build_tools()

        server = _Server()
        result = server.call_tool("scaffold_hazard", {
            "level": 3,
            "type": "haunt",
            "name": "Complex Haunt",
            "complexity": "complex",
        })
        assert result.success
        assert "2 successes needed" in result.data

    def test_scaffold_haunt_various_levels(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._creature_stats = CREATURE_STATS_BY_LEVEL
                self._hazard_stats = HAZARD_STATS_BY_LEVEL
                self._elite_hp = ELITE_HP_ADJUSTMENT
                self._role_adjustments = ROLE_ADJUSTMENTS
                from gm_agent.systems.pf2e.servers.creature_modifier import TROOP_SIZE_TABLE, SWARM_SIZE_TABLE
                self._troop_size_table = TROOP_SIZE_TABLE
                self._swarm_size_table = SWARM_SIZE_TABLE
                self._tools = self._build_tools()

        server = _Server()
        for level in (-1, 0, 5, 10, 20, 25):
            result = server.call_tool("scaffold_hazard", {"level": level, "type": "haunt"})
            assert result.success, f"Failed for level {level}"


class TestDesignCreature:
    """Tests for design_creature tool."""

    @pytest.fixture
    def server_no_backend(self):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer

        class _Server(CreatureModifierServer):
            def __init__(self):
                self.search = MagicMock()
                self.backend = None
                self._tools = self._build_tools()

        return _Server()

    @pytest.fixture
    def mock_backend(self):
        from gm_agent.models.base import LLMResponse
        backend = MagicMock()
        backend.chat.return_value = LLMResponse(
            text=json.dumps({
                "name": "Forge Wraith",
                "level": 5,
                "abilities": [
                    {
                        "name": "Searing Touch",
                        "action_cost": "[one-action]",
                        "traits": ["fire", "magical"],
                        "description": "The forge wraith touches a creature, dealing 2d6 fire damage (DC 22 basic Reflex).",
                    },
                    {
                        "name": "Imprisoned Lament",
                        "action_cost": "[reaction]",
                        "traits": ["auditory", "emotion", "mental"],
                        "description": "The forge wraith wails when struck, demoralizing all creatures within 30 feet (DC 22 Will).",
                    },
                ],
            }),
        )
        return backend

    @pytest.fixture
    def server_with_backend(self, mock_backend):
        from gm_agent.systems.pf2e.servers.creature_modifier import CreatureModifierServer

        class _Server(CreatureModifierServer):
            def __init__(self, backend):
                self.search = MagicMock()
                self.backend = backend
                self._tools = self._build_tools()

        return _Server(mock_backend)

    def test_design_creature_scaffold_only(self, server_no_backend):
        """No concept/abilities → pure scaffold, no LLM required."""
        result = server_no_backend.call_tool("design_creature", {
            "level": 3,
            "role": "brute",
            "name": "Cave Troll",
        })
        assert result.success
        assert "Cave Troll" in result.data
        assert "Level 3" in result.data

    def test_design_creature_no_level_no_concept_fails(self, server_no_backend):
        """No level and no concept → error."""
        result = server_no_backend.call_tool("design_creature", {
            "name": "Mystery Creature",
        })
        assert not result.success
        assert "level" in result.error.lower()

    def test_design_creature_no_backend_error(self, server_no_backend):
        """concept provided but no backend → clean error."""
        result = server_no_backend.call_tool("design_creature", {
            "level": 5,
            "concept": "A forge-bound fire elemental that mourns its imprisonment",
        })
        assert not result.success
        assert "backend" in result.error.lower()

    def test_design_creature_with_backend(self, server_with_backend):
        """concept + backend → calls LLM, returns scaffold + abilities."""
        result = server_with_backend.call_tool("design_creature", {
            "level": 5,
            "concept": "A forge-bound fire elemental that mourns its imprisonment",
            "name": "Forge Wraith",
        })
        assert result.success
        assert "Forge Wraith" in result.data
        assert "Searing Touch" in result.data

    def test_design_creature_backend_called_once(self, server_with_backend, mock_backend):
        """LLM is called exactly once."""
        server_with_backend.call_tool("design_creature", {
            "level": 5,
            "concept": "A test creature",
        })
        assert mock_backend.chat.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
