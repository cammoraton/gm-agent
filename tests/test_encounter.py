"""Tests for the encounter evaluation MCP server."""

import pytest

from gm_agent.mcp.encounter import (
    EncounterServer,
    creature_xp,
    hazard_xp,
    get_threat_level,
    evaluate_encounter,
    suggest_encounter,
)


class TestCreatureXP:
    """Tests for creature XP calculation."""

    def test_same_level_creature(self):
        """Creature at party level should be 40 XP."""
        assert creature_xp(5, 5) == 40

    def test_higher_level_creature(self):
        """Higher level creatures should give more XP."""
        assert creature_xp(6, 5) == 60  # +1
        assert creature_xp(7, 5) == 80  # +2
        assert creature_xp(8, 5) == 120  # +3
        assert creature_xp(9, 5) == 160  # +4

    def test_lower_level_creature(self):
        """Lower level creatures should give less XP."""
        assert creature_xp(4, 5) == 30  # -1
        assert creature_xp(3, 5) == 20  # -2
        assert creature_xp(2, 5) == 15  # -3
        assert creature_xp(1, 5) == 10  # -4

    def test_trivial_creature(self):
        """Very low level creatures should be worth 5 XP."""
        assert creature_xp(0, 5) == 5  # -5
        assert creature_xp(-1, 5) == 5  # -6

    def test_extreme_creature(self):
        """Very high level creatures should scale up."""
        assert creature_xp(10, 5) == 200 + 40  # +5
        assert creature_xp(11, 5) == 200 + 80  # +6


class TestHazardXP:
    """Tests for hazard XP calculation."""

    def test_simple_hazard_same_level(self):
        """Simple hazard at party level should be 8 XP."""
        assert hazard_xp(5, 5, is_complex=False) == 8

    def test_complex_hazard_same_level(self):
        """Complex hazard should use creature XP table."""
        assert hazard_xp(5, 5, is_complex=True) == 40

    def test_simple_hazard_higher_level(self):
        """Higher level simple hazards give more XP."""
        assert hazard_xp(6, 5, is_complex=False) == 12
        assert hazard_xp(7, 5, is_complex=False) == 16


class TestThreatLevel:
    """Tests for threat level classification."""

    def test_trivial_threat(self):
        """XP < 40 should be trivial."""
        assert get_threat_level(30) == "trivial"
        assert get_threat_level(39) == "trivial"

    def test_low_threat(self):
        """40 <= XP < 60 should be low."""
        assert get_threat_level(40) == "low"
        assert get_threat_level(59) == "low"

    def test_moderate_threat(self):
        """60 <= XP < 80 should be moderate."""
        assert get_threat_level(60) == "moderate"
        assert get_threat_level(79) == "moderate"

    def test_severe_threat(self):
        """80 <= XP < 120 should be severe."""
        assert get_threat_level(80) == "severe"
        assert get_threat_level(119) == "severe"

    def test_extreme_threat(self):
        """120 <= XP < 160 should be extreme."""
        assert get_threat_level(120) == "extreme"
        assert get_threat_level(159) == "extreme"

    def test_impossible_threat(self):
        """XP >= 160 should be impossible."""
        assert get_threat_level(160) == "impossible"
        assert get_threat_level(200) == "impossible"

    def test_party_size_adjustment(self):
        """Larger parties should have adjusted thresholds."""
        # 5 players: moderate is 80-100 instead of 60-80
        assert get_threat_level(80, party_size=5) == "moderate"
        assert get_threat_level(60, party_size=5) == "low"


class TestEvaluateEncounter:
    """Tests for encounter evaluation."""

    def test_single_moderate_creature(self):
        """Single party-level creature should be moderate."""
        result = evaluate_encounter(creatures=[("Goblin", 3)], party_level=3, party_size=4)
        assert result["total_xp"] == 40
        assert result["threat_level"] == "low"

    def test_multiple_creatures(self):
        """Multiple creatures should sum XP."""
        result = evaluate_encounter(
            creatures=[("Goblin", 3), ("Goblin", 3)], party_level=3, party_size=4
        )
        assert result["total_xp"] == 80
        assert result["threat_level"] == "severe"

    def test_with_hazards(self):
        """Hazards should add to XP budget."""
        result = evaluate_encounter(
            creatures=[("Skeleton", 3)],
            party_level=3,
            party_size=4,
            hazards=[("Pit Trap", 3, False)],
        )
        assert result["creature_xp"] == 40
        assert result["hazard_xp"] == 8
        assert result["total_xp"] == 48

    def test_extreme_creature_warning(self):
        """Should warn about level +4 creatures."""
        result = evaluate_encounter(creatures=[("Dragon", 7)], party_level=3, party_size=4)
        assert len(result["warnings"]) > 0
        assert any("extreme" in w.lower() for w in result["warnings"])

    def test_includes_creature_details(self):
        """Should include per-creature breakdown."""
        result = evaluate_encounter(
            creatures=[("Goblin", 3), ("Wolf", 2)], party_level=3, party_size=4
        )
        assert len(result["creatures"]) == 2
        assert result["creatures"][0]["name"] == "Goblin"
        assert result["creatures"][0]["xp"] == 40
        assert result["creatures"][1]["name"] == "Wolf"
        assert result["creatures"][1]["xp"] == 30


class TestSuggestEncounter:
    """Tests for encounter suggestions."""

    def test_moderate_suggestions(self):
        """Should suggest patterns for moderate encounters."""
        result = suggest_encounter(target_threat="moderate", party_level=5, party_size=4)
        assert result["target_threat"] == "moderate"
        assert result["xp_budget"] == 80
        assert len(result["suggestions"]) > 0

    def test_includes_multiple_patterns(self):
        """Should include different encounter patterns."""
        result = suggest_encounter(target_threat="severe", party_level=5, party_size=4)
        patterns = [s["pattern"] for s in result["suggestions"]]
        # Should have at least 2 different patterns
        assert len(patterns) >= 2

    def test_party_size_adjustment(self):
        """Larger parties should have higher XP budget."""
        result_4 = suggest_encounter("moderate", party_level=5, party_size=4)
        result_5 = suggest_encounter("moderate", party_level=5, party_size=5)
        assert result_5["xp_budget"] > result_4["xp_budget"]

    def test_includes_advice(self):
        """Should include general advice."""
        result = suggest_encounter("moderate", party_level=5)
        assert "general_advice" in result
        assert len(result["general_advice"]) > 0


class TestEncounterServer:
    """Tests for the EncounterServer MCP server."""

    @pytest.fixture
    def server(self):
        server = EncounterServer()
        yield server
        server.close()

    def test_list_tools(self, server):
        """Should list all encounter tools."""
        tools = server.list_tools()
        tool_names = [t.name for t in tools]

        assert "evaluate_encounter" in tool_names
        assert "suggest_encounter" in tool_names
        assert "calculate_creature_xp" in tool_names
        assert "get_encounter_advice" in tool_names

    def test_evaluate_encounter_tool(self, server):
        """Should evaluate encounter via tool call."""
        import json

        result = server.call_tool(
            "evaluate_encounter",
            {
                "creatures": json.dumps([{"name": "Goblin", "level": 3}]),
                "party_level": 3,
                "party_size": 4,
            },
        )

        assert result.success
        assert result.data["total_xp"] == 40
        assert result.data["threat_level"] == "low"

    def test_evaluate_encounter_tool_with_list(self, server):
        """Should accept creatures as a list directly."""
        result = server.call_tool(
            "evaluate_encounter",
            {
                "creatures": [{"name": "Goblin", "level": 3}],
                "party_level": 3,
                "party_size": 4,
            },
        )

        assert result.success
        assert result.data["total_xp"] == 40

    def test_suggest_encounter_tool(self, server):
        """Should suggest encounters via tool call."""
        result = server.call_tool(
            "suggest_encounter", {"target_threat": "moderate", "party_level": 5}
        )

        assert result.success
        assert result.data["xp_budget"] == 80
        assert len(result.data["suggestions"]) > 0

    def test_calculate_creature_xp_tool(self, server):
        """Should calculate single creature XP."""
        result = server.call_tool("calculate_creature_xp", {"creature_level": 6, "party_level": 5})

        assert result.success
        assert result.data["xp"] == 60
        assert result.data["relative_level"] == 1

    def test_get_encounter_advice_tool(self, server):
        """Should return encounter advice."""
        result = server.call_tool("get_encounter_advice", {"topic": "threat_levels"})

        assert result.success
        assert "advice" in result.data
        assert "Trivial" in result.data["advice"]

    def test_get_all_advice(self, server):
        """Should return all advice topics."""
        result = server.call_tool("get_encounter_advice", {"topic": "all"})

        assert result.success
        assert "Threat Levels" in result.data["advice"]
        assert "Variety" in result.data["advice"]
        assert "Morale" in result.data["advice"]

    def test_unknown_tool(self, server):
        """Should return error for unknown tool."""
        result = server.call_tool("unknown_tool", {})

        assert not result.success
        assert "Unknown tool" in result.error
