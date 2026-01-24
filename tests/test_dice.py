"""Tests for the dice rolling MCP server."""

import pytest

from gm_agent.mcp.dice import (
    DiceServer,
    DiceResult,
    parse_dice_expression,
    roll_dice,
    roll_fortune,
    roll_misfortune,
    roll_multiple,
    check_result,
)


class TestParseDiceExpression:
    """Tests for dice expression parsing."""

    def test_simple_d20(self):
        """Should parse 'd20'."""
        num, dtype, mod = parse_dice_expression("d20")
        assert num == 1
        assert dtype == 20
        assert mod == 0

    def test_with_count(self):
        """Should parse '2d6'."""
        num, dtype, mod = parse_dice_expression("2d6")
        assert num == 2
        assert dtype == 6
        assert mod == 0

    def test_with_positive_modifier(self):
        """Should parse '1d20+5'."""
        num, dtype, mod = parse_dice_expression("1d20+5")
        assert num == 1
        assert dtype == 20
        assert mod == 5

    def test_with_negative_modifier(self):
        """Should parse '2d6-2'."""
        num, dtype, mod = parse_dice_expression("2d6-2")
        assert num == 2
        assert dtype == 6
        assert mod == -2

    def test_case_insensitive(self):
        """Should handle uppercase."""
        num, dtype, mod = parse_dice_expression("2D6+4")
        assert num == 2
        assert dtype == 6
        assert mod == 4

    def test_invalid_expression(self):
        """Should raise error for invalid expression."""
        with pytest.raises(ValueError):
            parse_dice_expression("invalid")

    def test_too_many_dice(self):
        """Should raise error for >100 dice."""
        with pytest.raises(ValueError):
            parse_dice_expression("101d6")


class TestRollDice:
    """Tests for dice rolling."""

    def test_roll_returns_result(self):
        """Should return DiceResult."""
        result = roll_dice("1d20")
        assert isinstance(result, DiceResult)
        assert 1 <= result.total <= 20

    def test_roll_multiple_dice(self):
        """Should roll correct number of dice."""
        result = roll_dice("3d6")
        assert len(result.rolls) == 3
        assert all(1 <= r <= 6 for r in result.rolls)
        assert result.total == sum(result.rolls)

    def test_roll_with_modifier(self):
        """Should add modifier to total."""
        result = roll_dice("1d20+5")
        assert result.modifier == 5
        assert result.total == result.rolls[0] + 5

    def test_roll_preserves_expression(self):
        """Should preserve original expression."""
        result = roll_dice("2d6+4")
        assert result.expression == "2d6+4"


class TestRollFortune:
    """Tests for fortune rolls (roll twice, take higher)."""

    def test_fortune_takes_higher(self):
        """Should return the higher of two rolls."""
        result = roll_fortune("1d20")
        assert result["type"] == "fortune"
        assert result["result"] == result["chosen"]["total"]
        assert result["chosen"]["total"] >= result["discarded"]["total"]

    def test_fortune_has_both_rolls(self):
        """Should include both roll results."""
        result = roll_fortune("1d20")
        assert "roll_1" in result
        assert "roll_2" in result


class TestRollMisfortune:
    """Tests for misfortune rolls (roll twice, take lower)."""

    def test_misfortune_takes_lower(self):
        """Should return the lower of two rolls."""
        result = roll_misfortune("1d20")
        assert result["type"] == "misfortune"
        assert result["result"] == result["chosen"]["total"]
        assert result["chosen"]["total"] <= result["discarded"]["total"]


class TestRollMultiple:
    """Tests for rolling multiple expressions."""

    def test_multiple_expressions(self):
        """Should roll all expressions."""
        results = roll_multiple(["1d20", "2d6", "1d8"])
        assert len(results) == 3
        assert all("total" in r for r in results)

    def test_handles_invalid_expression(self):
        """Should include error for invalid expressions."""
        results = roll_multiple(["1d20", "invalid", "1d6"])
        assert len(results) == 3
        assert "error" in results[1]


class TestCheckResult:
    """Tests for degree of success checking."""

    def test_critical_success(self):
        """Should detect critical success (beat by 10+)."""
        result = check_result(25, 15)
        assert result["degree"] == "critical_success"

    def test_success(self):
        """Should detect success (meet or beat DC)."""
        result = check_result(15, 15)
        assert result["degree"] == "success"

        result = check_result(17, 15)
        assert result["degree"] == "success"

    def test_failure(self):
        """Should detect failure (below DC)."""
        result = check_result(14, 15)
        assert result["degree"] == "failure"

    def test_critical_failure(self):
        """Should detect critical failure (below by 10+)."""
        result = check_result(5, 15)
        assert result["degree"] == "critical_failure"

    def test_includes_difference(self):
        """Should include difference from DC."""
        result = check_result(20, 15)
        assert result["difference"] == 5


class TestDiceServer:
    """Tests for the DiceServer MCP server."""

    @pytest.fixture
    def server(self):
        server = DiceServer()
        yield server
        server.close()

    def test_list_tools(self, server):
        """Should list all dice tools."""
        tools = server.list_tools()
        tool_names = [t.name for t in tools]

        assert "roll_dice" in tool_names
        assert "roll_multiple" in tool_names
        assert "roll_fortune" in tool_names
        assert "roll_misfortune" in tool_names
        assert "check_result" in tool_names
        assert "roll_check" in tool_names

    def test_roll_dice_tool(self, server):
        """Should roll dice via tool call."""
        result = server.call_tool("roll_dice", {"expression": "2d6+4"})

        assert result.success
        assert "total" in result.data
        assert len(result.data["rolls"]) == 2

    def test_roll_multiple_tool(self, server):
        """Should roll multiple expressions."""
        result = server.call_tool("roll_multiple", {"expressions": "1d20+5, 2d6+3"})

        assert result.success
        assert len(result.data["rolls"]) == 2
        assert "grand_total" in result.data

    def test_roll_fortune_tool(self, server):
        """Should roll with fortune."""
        result = server.call_tool("roll_fortune", {"expression": "1d20"})

        assert result.success
        assert result.data["type"] == "fortune"

    def test_roll_misfortune_tool(self, server):
        """Should roll with misfortune."""
        result = server.call_tool("roll_misfortune", {"expression": "1d20"})

        assert result.success
        assert result.data["type"] == "misfortune"

    def test_check_result_tool(self, server):
        """Should check result against DC."""
        result = server.call_tool("check_result", {"roll_total": 25, "dc": 15})

        assert result.success
        assert result.data["degree"] == "critical_success"

    def test_roll_check_tool(self, server):
        """Should roll and check in one step."""
        result = server.call_tool("roll_check", {"modifier": 10, "dc": 15})

        assert result.success
        assert "d20_result" in result.data
        assert "total" in result.data
        assert "degree" in result.data

    def test_roll_check_with_fortune(self, server):
        """Should support fortune on roll_check."""
        result = server.call_tool("roll_check", {"modifier": 10, "dc": 15, "fortune": True})

        assert result.success
        assert result.data["roll_type"] == "fortune"

    def test_roll_check_with_misfortune(self, server):
        """Should support misfortune on roll_check."""
        result = server.call_tool("roll_check", {"modifier": 10, "dc": 15, "misfortune": True})

        assert result.success
        assert result.data["roll_type"] == "misfortune"

    def test_invalid_expression(self, server):
        """Should return error for invalid expression."""
        result = server.call_tool("roll_dice", {"expression": "invalid"})

        assert not result.success
        assert "Invalid" in result.error

    def test_unknown_tool(self, server):
        """Should return error for unknown tool."""
        result = server.call_tool("unknown_tool", {})

        assert not result.success
        assert "Unknown tool" in result.error
