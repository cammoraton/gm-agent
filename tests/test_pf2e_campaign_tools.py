"""Tests for PF2eCampaignToolPlugin — run_skill_check."""

import pytest
from unittest.mock import MagicMock, patch

from gm_agent.systems.pf2e.campaign_tools import PF2eCampaignToolPlugin


# ---------------------------------------------------------------------------
# Fixture: minimal server stub (campaign tools only use server for location BFS)
# ---------------------------------------------------------------------------

class _MockServer:
    """Minimal server stub for campaign tool tests."""
    class _Locations:
        def get_by_name(self, name):
            return None
        def get(self, loc_id):
            return None
    locations = _Locations()


@pytest.fixture
def plugin(tmp_path):
    return PF2eCampaignToolPlugin("test-campaign", base_dir=tmp_path)


@pytest.fixture
def server():
    return _MockServer()


# ---------------------------------------------------------------------------
# run_skill_check tests
# ---------------------------------------------------------------------------

class TestRunSkillCheck:
    """Tests for the run_skill_check tool."""

    def test_tool_in_tool_defs(self, plugin):
        names = [t.name for t in plugin.tool_defs()]
        assert "run_skill_check" in names

    def test_manual_roll_success(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Thievery",
            "dc": 20,
            "roll_total": 22,
        }, server)
        assert result.success
        assert result.data["skill"] == "Thievery"
        assert result.data["dc"] == 20
        assert result.data["degree"] == "success"
        assert result.data["total"] == 22

    def test_manual_roll_critical_success(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Religion",
            "dc": 20,
            "roll_total": 30,
        }, server)
        assert result.success
        assert result.data["degree"] == "critical_success"

    def test_manual_roll_failure(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Athletics",
            "dc": 20,
            "roll_total": 15,
        }, server)
        assert result.success
        assert result.data["degree"] == "failure"

    def test_manual_roll_critical_failure(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Stealth",
            "dc": 20,
            "roll_total": 5,
        }, server)
        assert result.success
        assert result.data["degree"] == "critical_failure"

    def test_auto_roll_with_modifier(self, plugin, server):
        """Providing modifier should auto-roll d20."""
        result = plugin.call_tool("run_skill_check", {
            "skill": "Perception",
            "dc": 15,
            "modifier": 10,
        }, server)
        assert result.success
        assert result.data["auto_rolled"] is True
        # Total should be between 11 (1+10) and 30 (20+10)
        assert 11 <= result.data["total"] <= 30

    def test_missing_both_params_fails(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Arcana",
            "dc": 18,
        }, server)
        assert not result.success
        assert "roll_total" in result.error or "modifier" in result.error

    def test_context_included_in_output(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Thievery",
            "dc": 22,
            "roll_total": 24,
            "context": "Disabling dart trap in room A3",
        }, server)
        assert result.success
        assert result.data["context"] == "Disabling dart trap in room A3"

    def test_nat_20_upgrades_degree(self, plugin, server):
        """A natural 20 that results in success should upgrade to critical success."""
        # DC 25, modifier 0 → total 20 = failure normally. nat 20 upgrades failure→success.
        # But we can't guarantee dice; use mocking instead.
        with patch("gm_agent.mcp.dice.roll_dice") as mock_roll:
            from gm_agent.mcp.dice import DiceResult
            mock_roll.return_value = DiceResult(
                expression="1d20", rolls=[20], modifier=0, total=20, dice_type=20, num_dice=1
            )
            result = plugin.call_tool("run_skill_check", {
                "skill": "Diplomacy",
                "dc": 25,
                "modifier": 0,
            }, server)
        assert result.success
        # Roll: d20(20) + 0 = 20 vs DC 25 → normally failure, but nat 20 upgrades to success
        assert result.data["degree"] == "success"

    def test_nat_1_downgrades_degree(self, plugin, server):
        """A natural 1 that results in success should downgrade to failure."""
        with patch("gm_agent.mcp.dice.roll_dice") as mock_roll:
            from gm_agent.mcp.dice import DiceResult
            mock_roll.return_value = DiceResult(
                expression="1d20", rolls=[1], modifier=0, total=1, dice_type=20, num_dice=1
            )
            result = plugin.call_tool("run_skill_check", {
                "skill": "Diplomacy",
                "dc": 1,
                "modifier": 0,
            }, server)
        assert result.success
        # Roll: d20(1) + 0 = 1 vs DC 1 → normally success, but nat 1 downgrades to failure
        assert result.data["degree"] == "failure"

    def test_fortune_field_accepted(self, plugin, server):
        """fortune=True should not error."""
        result = plugin.call_tool("run_skill_check", {
            "skill": "Deception",
            "dc": 18,
            "modifier": 8,
            "fortune": True,
        }, server)
        assert result.success

    def test_roll_breakdown_in_result(self, plugin, server):
        result = plugin.call_tool("run_skill_check", {
            "skill": "Crafting",
            "dc": 20,
            "roll_total": 18,
        }, server)
        assert result.success
        assert result.data["roll_breakdown"]
        assert result.data["formatted"]
