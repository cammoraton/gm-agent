"""Dice rolling MCP server.

Provides tools for rolling dice in Pathfinder 2e format:
- Standard dice rolls (1d20, 2d6+4, etc.)
- Multiple dice expressions
- Fortune/misfortune (roll twice, take best/worst)
"""

import random
import re
from dataclasses import dataclass

from .base import MCPServer, ToolDef, ToolParameter, ToolResult


@dataclass
class DiceResult:
    """Result of a dice roll."""

    expression: str
    rolls: list[int]
    modifier: int
    total: int
    dice_type: int
    num_dice: int

    def __str__(self) -> str:
        if self.modifier > 0:
            return f"{self.expression}: [{', '.join(map(str, self.rolls))}] + {self.modifier} = {self.total}"
        elif self.modifier < 0:
            return f"{self.expression}: [{', '.join(map(str, self.rolls))}] - {abs(self.modifier)} = {self.total}"
        else:
            return f"{self.expression}: [{', '.join(map(str, self.rolls))}] = {self.total}"


def parse_dice_expression(expr: str) -> tuple[int, int, int]:
    """Parse a dice expression like '2d6+4' into (num_dice, dice_type, modifier).

    Args:
        expr: Dice expression (e.g., '2d6+4', '1d20', 'd8-1')

    Returns:
        Tuple of (num_dice, dice_type, modifier)
    """
    expr = expr.lower().strip()

    # Match patterns like: 2d6+4, d20, 1d8-2, 3d6
    match = re.match(r"^(\d*)d(\d+)([+-]\d+)?$", expr)
    if not match:
        raise ValueError(f"Invalid dice expression: {expr}")

    num_dice = int(match.group(1)) if match.group(1) else 1
    dice_type = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0

    if num_dice < 1 or num_dice > 100:
        raise ValueError(f"Number of dice must be 1-100, got {num_dice}")
    if dice_type < 1 or dice_type > 100:
        raise ValueError(f"Dice type must be 1-100, got d{dice_type}")

    return num_dice, dice_type, modifier


def roll_dice(expr: str) -> DiceResult:
    """Roll dice according to an expression.

    Args:
        expr: Dice expression (e.g., '2d6+4', '1d20')

    Returns:
        DiceResult with the roll details
    """
    num_dice, dice_type, modifier = parse_dice_expression(expr)

    rolls = [random.randint(1, dice_type) for _ in range(num_dice)]
    total = sum(rolls) + modifier

    return DiceResult(
        expression=expr,
        rolls=rolls,
        modifier=modifier,
        total=total,
        dice_type=dice_type,
        num_dice=num_dice,
    )


def roll_fortune(expr: str) -> dict:
    """Roll with fortune (roll twice, take higher).

    Used for hero points, certain abilities.
    """
    result1 = roll_dice(expr)
    result2 = roll_dice(expr)

    best = result1 if result1.total >= result2.total else result2
    worst = result2 if result1.total >= result2.total else result1

    return {
        "type": "fortune",
        "expression": expr,
        "roll_1": {"rolls": result1.rolls, "total": result1.total},
        "roll_2": {"rolls": result2.rolls, "total": result2.total},
        "chosen": {"rolls": best.rolls, "total": best.total},
        "discarded": {"rolls": worst.rolls, "total": worst.total},
        "result": best.total,
    }


def roll_misfortune(expr: str) -> dict:
    """Roll with misfortune (roll twice, take lower).

    Used for frightened, sickened, etc.
    """
    result1 = roll_dice(expr)
    result2 = roll_dice(expr)

    worst = result1 if result1.total <= result2.total else result2
    best = result2 if result1.total <= result2.total else result1

    return {
        "type": "misfortune",
        "expression": expr,
        "roll_1": {"rolls": result1.rolls, "total": result1.total},
        "roll_2": {"rolls": result2.rolls, "total": result2.total},
        "chosen": {"rolls": worst.rolls, "total": worst.total},
        "discarded": {"rolls": best.rolls, "total": best.total},
        "result": worst.total,
    }


def roll_multiple(expressions: list[str]) -> list[dict]:
    """Roll multiple dice expressions.

    Args:
        expressions: List of dice expressions

    Returns:
        List of roll results
    """
    results = []
    for expr in expressions:
        try:
            result = roll_dice(expr)
            results.append(
                {
                    "expression": expr,
                    "rolls": result.rolls,
                    "modifier": result.modifier,
                    "total": result.total,
                }
            )
        except ValueError as e:
            results.append(
                {
                    "expression": expr,
                    "error": str(e),
                }
            )
    return results


def check_result(roll: int, dc: int) -> dict:
    """Determine the degree of success for a check.

    PF2e degrees of success:
    - Critical Success: Beat DC by 10+ OR natural 20 that succeeds
    - Success: Meet or beat DC
    - Failure: Below DC
    - Critical Failure: Below DC by 10+ OR natural 1 that fails
    """
    difference = roll - dc

    if difference >= 10:
        degree = "critical_success"
    elif difference >= 0:
        degree = "success"
    elif difference >= -9:
        degree = "failure"
    else:
        degree = "critical_failure"

    return {
        "roll": roll,
        "dc": dc,
        "difference": difference,
        "degree": degree,
        "description": degree.replace("_", " ").title(),
    }


class DiceServer(MCPServer):
    """MCP server for dice rolling tools."""

    def __init__(self):
        self._tools = [
            ToolDef(
                name="roll_dice",
                description="Roll dice using standard notation (e.g., '2d6+4', '1d20', '3d8-2'). "
                "Returns individual rolls and total.",
                parameters=[
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="Dice expression like '1d20', '2d6+4', '3d8-2'",
                    ),
                ],
            ),
            ToolDef(
                name="roll_multiple",
                description="Roll multiple dice expressions at once. Useful for damage rolls "
                "or rolling for multiple creatures.",
                parameters=[
                    ToolParameter(
                        name="expressions",
                        type="string",
                        description="Comma-separated dice expressions (e.g., '1d20+5, 2d6+3, 1d8')",
                    ),
                ],
            ),
            ToolDef(
                name="roll_fortune",
                description="Roll with fortune/hero point - roll twice and take the higher result. "
                "Used when spending hero points or with certain abilities.",
                parameters=[
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="Dice expression to roll twice",
                    ),
                ],
            ),
            ToolDef(
                name="roll_misfortune",
                description="Roll with misfortune - roll twice and take the lower result. "
                "Used with frightened, enfeebled, or other penalties.",
                parameters=[
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="Dice expression to roll twice",
                    ),
                ],
            ),
            ToolDef(
                name="check_result",
                description="Determine the degree of success for a d20 check against a DC. "
                "Returns critical success, success, failure, or critical failure.",
                parameters=[
                    ToolParameter(
                        name="roll_total",
                        type="integer",
                        description="The total of the d20 roll including modifiers",
                    ),
                    ToolParameter(
                        name="dc",
                        type="integer",
                        description="The Difficulty Class to check against",
                    ),
                ],
            ),
            ToolDef(
                name="roll_check",
                description="Roll a d20 check against a DC and determine degree of success. "
                "Combines rolling and checking in one step.",
                parameters=[
                    ToolParameter(
                        name="modifier",
                        type="integer",
                        description="The modifier to add to the d20 roll",
                    ),
                    ToolParameter(
                        name="dc",
                        type="integer",
                        description="The Difficulty Class to check against",
                    ),
                    ToolParameter(
                        name="fortune",
                        type="boolean",
                        description="Roll with fortune (take higher of two rolls)",
                        required=False,
                        default=False,
                    ),
                    ToolParameter(
                        name="misfortune",
                        type="boolean",
                        description="Roll with misfortune (take lower of two rolls)",
                        required=False,
                        default=False,
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict) -> ToolResult:
        if name == "roll_dice":
            return self._roll_dice(args)
        elif name == "roll_multiple":
            return self._roll_multiple(args)
        elif name == "roll_fortune":
            return self._roll_fortune(args)
        elif name == "roll_misfortune":
            return self._roll_misfortune(args)
        elif name == "check_result":
            return self._check_result(args)
        elif name == "roll_check":
            return self._roll_check(args)
        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _roll_dice(self, args: dict) -> ToolResult:
        try:
            result = roll_dice(args["expression"])
            return ToolResult(
                success=True,
                data={
                    "expression": result.expression,
                    "rolls": result.rolls,
                    "modifier": result.modifier,
                    "total": result.total,
                    "formatted": str(result),
                },
            )
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

    def _roll_multiple(self, args: dict) -> ToolResult:
        try:
            expressions = [e.strip() for e in args["expressions"].split(",")]
            results = roll_multiple(expressions)
            total = sum(r.get("total", 0) for r in results if "total" in r)
            return ToolResult(
                success=True,
                data={
                    "rolls": results,
                    "grand_total": total,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _roll_fortune(self, args: dict) -> ToolResult:
        try:
            result = roll_fortune(args["expression"])
            return ToolResult(success=True, data=result)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

    def _roll_misfortune(self, args: dict) -> ToolResult:
        try:
            result = roll_misfortune(args["expression"])
            return ToolResult(success=True, data=result)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

    def _check_result(self, args: dict) -> ToolResult:
        try:
            result = check_result(args["roll_total"], args["dc"])
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _roll_check(self, args: dict) -> ToolResult:
        try:
            modifier = args["modifier"]
            dc = args["dc"]
            fortune = args.get("fortune", False)
            misfortune = args.get("misfortune", False)

            # Roll the d20
            if fortune and not misfortune:
                roll_result = roll_fortune("1d20")
                d20 = roll_result["result"]
                natural_roll = roll_result["chosen"]["rolls"][0]
                roll_type = "fortune"
            elif misfortune and not fortune:
                roll_result = roll_misfortune("1d20")
                d20 = roll_result["result"]
                natural_roll = roll_result["chosen"]["rolls"][0]
                roll_type = "misfortune"
            else:
                result = roll_dice("1d20")
                d20 = result.total
                natural_roll = result.rolls[0]
                roll_result = {"rolls": result.rolls, "total": d20}
                roll_type = "normal"

            total = d20 + modifier
            check = check_result(total, dc)

            # Handle natural 20/1 adjustments
            if natural_roll == 20 and check["degree"] in ("success", "failure"):
                check["degree"] = "critical_success" if check["degree"] == "success" else "success"
                check["natural_20"] = True
            elif natural_roll == 1 and check["degree"] in ("success", "failure"):
                check["degree"] = "failure" if check["degree"] == "success" else "critical_failure"
                check["natural_1"] = True

            check["description"] = check["degree"].replace("_", " ").title()

            return ToolResult(
                success=True,
                data={
                    "roll_type": roll_type,
                    "d20_result": d20,
                    "natural_roll": natural_roll,
                    "modifier": modifier,
                    "total": total,
                    "dc": dc,
                    "degree": check["degree"],
                    "description": check["description"],
                    "roll_details": roll_result,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def close(self) -> None:
        """Clean up resources. No-op for stateless server."""
        pass
