"""Encounter evaluation MCP server.

Provides tools for evaluating and building Pathfinder 2e encounters:
- Calculate XP budgets and threat levels
- Evaluate encounter balance
- Get encounter building advice
"""

from .base import MCPServer, ToolDef, ToolParameter, ToolResult

# XP values by creature level relative to party level
CREATURE_XP_TABLE = {
    -4: 10,
    -3: 15,
    -2: 20,
    -1: 30,
    0: 40,
    1: 60,
    2: 80,
    3: 120,
    4: 160,
}

# Simple hazard XP (complex hazards use creature XP)
SIMPLE_HAZARD_XP_TABLE = {-4: 2, -3: 3, -2: 4, -1: 6, 0: 8, 1: 12, 2: 16, 3: 24, 4: 32}

# XP budget thresholds for 4-player party
THREAT_THRESHOLDS = {
    "trivial": (0, 40),
    "low": (40, 60),
    "moderate": (60, 80),
    "severe": (80, 120),
    "extreme": (120, 160),
    "impossible": (160, float("inf")),
}

# XP adjustment per additional player beyond 4
XP_PER_EXTRA_PLAYER = {
    "trivial": 10,
    "low": 15,
    "moderate": 20,
    "severe": 30,
    "extreme": 40,
}

# GM Core advice snippets
THREAT_LEVEL_ADVICE = {
    "trivial": "Warm-up encounters, palate cleansers, or showing off PC power. Uses minimal resources.",
    "low": "Uses some resources but rarely dangerous. Good for exploration or minor obstacles.",
    "moderate": "Serious challenge requiring good tactics. The baseline difficulty for most encounters.",
    "severe": "Boss fights where PC death is possible. Use carefully and telegraph the danger.",
    "extreme": "Even match with possible TPK. Reserve for campaign climaxes. Always provide escape options.",
    "impossible": "Beyond extreme threat. Consider splitting into multiple encounters or adding allies.",
}

VARIETY_ADVICE = """For good encounter variety:
- Include trivial encounters as 'palate cleansers' between harder fights
- Not all encounters should be moderate - mix in low and severe
- Use severe encounters beyond just final bosses for drama
- Aim for ~8-12 encounters per character level, ~1000 XP total"""

MORALE_ADVICE = """Consider creature morale:
- Most creatures flee when obviously losing
- Morale breaks can shift combat to social encounters
- Mindless creatures or fanatics fight to the death
- Surrender and negotiation create memorable moments"""


def creature_xp(creature_level: int, party_level: int) -> int:
    """Calculate XP value of a creature relative to party level.

    Args:
        creature_level: The creature's level
        party_level: The party's average level

    Returns:
        XP value of the creature
    """
    diff = creature_level - party_level
    if diff < -4:
        return 5  # Trivial
    elif diff > 4:
        return 200 + (diff - 4) * 40  # Very dangerous
    return CREATURE_XP_TABLE.get(diff, 40)


def hazard_xp(hazard_level: int, party_level: int, is_complex: bool = False) -> int:
    """Calculate XP value of a hazard relative to party level.

    Args:
        hazard_level: The hazard's level
        party_level: The party's average level
        is_complex: Whether the hazard is complex (acts like a creature)

    Returns:
        XP value of the hazard
    """
    if is_complex:
        return creature_xp(hazard_level, party_level)

    diff = hazard_level - party_level
    if diff < -4:
        return 1
    elif diff > 4:
        return 40 + (diff - 4) * 8
    return SIMPLE_HAZARD_XP_TABLE.get(diff, 8)


def get_threat_level(xp_budget: int, party_size: int = 4) -> str:
    """Classify encounter threat based on XP budget.

    Args:
        xp_budget: Total XP of all threats
        party_size: Number of players (adjusts thresholds)

    Returns:
        Threat level string
    """
    # Adjust for party size
    adjustment = party_size - 4

    for threat, (low, high) in THREAT_THRESHOLDS.items():
        adj_low = low + (adjustment * XP_PER_EXTRA_PLAYER.get(threat, 20))
        adj_high = high + (adjustment * XP_PER_EXTRA_PLAYER.get(threat, 20))
        if adj_low <= xp_budget < adj_high:
            return threat

    return "impossible"


def evaluate_encounter(
    creatures: list[tuple[str, int]],  # List of (name, level)
    party_level: int,
    party_size: int = 4,
    hazards: list[tuple[str, int, bool]] | None = None,  # (name, level, is_complex)
) -> dict:
    """Evaluate an encounter's difficulty.

    Args:
        creatures: List of (creature_name, creature_level) tuples
        party_level: The party's average level
        party_size: Number of players
        hazards: Optional list of (hazard_name, level, is_complex) tuples

    Returns:
        Dictionary with evaluation results
    """
    hazards = hazards or []

    # Calculate creature XP
    creature_details = []
    total_creature_xp = 0
    for name, level in creatures:
        xp = creature_xp(level, party_level)
        creature_details.append(
            {
                "name": name,
                "level": level,
                "xp": xp,
                "relative_level": level - party_level,
            }
        )
        total_creature_xp += xp

    # Calculate hazard XP
    hazard_details = []
    total_hazard_xp = 0
    for name, level, is_complex in hazards:
        xp = hazard_xp(level, party_level, is_complex)
        hazard_details.append({"name": name, "level": level, "xp": xp, "is_complex": is_complex})
        total_hazard_xp += xp

    total_xp = total_creature_xp + total_hazard_xp
    threat = get_threat_level(total_xp, party_size)

    # Generate warnings
    warnings = []

    # Check for level +4 creatures (extreme single threat)
    extreme_creatures = [c for c in creature_details if c["relative_level"] >= 4]
    if extreme_creatures:
        warnings.append(
            f"Contains {len(extreme_creatures)} creature(s) at party level +4 or higher. "
            "These are extreme threats that could easily kill a PC."
        )

    # Check for too many enemies (action economy)
    if len(creatures) >= party_size * 2:
        warnings.append(
            f"Large number of enemies ({len(creatures)}) may overwhelm party through action economy. "
            "Consider giving some creatures reduced actions or having them act in groups."
        )

    # Check for trivial creatures that might be filler
    trivial_creatures = [c for c in creature_details if c["relative_level"] <= -4]
    if trivial_creatures and len(trivial_creatures) == len(creatures):
        warnings.append(
            "All creatures are trivial threats. Consider if this encounter is necessary "
            "or could be resolved narratively."
        )

    return {
        "total_xp": total_xp,
        "threat_level": threat,
        "advice": THREAT_LEVEL_ADVICE.get(threat, ""),
        "creature_xp": total_creature_xp,
        "hazard_xp": total_hazard_xp,
        "creatures": creature_details,
        "hazards": hazard_details,
        "party_level": party_level,
        "party_size": party_size,
        "warnings": warnings,
    }


def suggest_encounter(
    target_threat: str, party_level: int, party_size: int = 4, theme: str | None = None
) -> dict:
    """Suggest creature compositions for a target threat level.

    Args:
        target_threat: Desired threat level (trivial, low, moderate, severe, extreme)
        party_level: The party's average level
        party_size: Number of players
        theme: Optional theme hint (e.g., "undead", "beasts")

    Returns:
        Dictionary with suggested compositions
    """
    # Get XP budget for target threat
    adjustment = party_size - 4
    budget = {
        "trivial": 40 + adjustment * 10,
        "low": 60 + adjustment * 15,
        "moderate": 80 + adjustment * 20,
        "severe": 120 + adjustment * 30,
        "extreme": 160 + adjustment * 40,
    }.get(target_threat.lower(), 80)

    suggestions = []

    # Single boss pattern
    for level_diff in range(-1, 5):
        single_xp = creature_xp(party_level + level_diff, party_level)
        if abs(single_xp - budget) < 20 or single_xp <= budget:
            suggestions.append(
                {
                    "pattern": "Single Boss",
                    "composition": f"1x Level {party_level + level_diff} creature",
                    "xp": single_xp,
                    "notes": "Good for dramatic solo fights. Add hazards or terrain for complexity.",
                }
            )
            break

    # Boss + minions pattern
    boss_level = party_level + 2
    boss_xp = creature_xp(boss_level, party_level)
    minion_xp = creature_xp(party_level - 2, party_level)
    if boss_xp + minion_xp * 2 <= budget + 10:
        num_minions = max(1, (budget - boss_xp) // minion_xp)
        suggestions.append(
            {
                "pattern": "Boss + Minions",
                "composition": f"1x Level {boss_level} + {num_minions}x Level {party_level - 2}",
                "xp": boss_xp + minion_xp * num_minions,
                "notes": "Classic boss fight pattern. Minions can be defeated quickly while boss poses main threat.",
            }
        )

    # Balanced group pattern
    equal_xp = creature_xp(party_level, party_level)
    num_equal = budget // equal_xp
    if num_equal >= 2:
        suggestions.append(
            {
                "pattern": "Balanced Group",
                "composition": f"{num_equal}x Level {party_level} creatures",
                "xp": equal_xp * num_equal,
                "notes": "Evenly matched opponents. Tactical combat with no obvious priority target.",
            }
        )

    # Swarm pattern
    weak_xp = creature_xp(party_level - 3, party_level)
    num_weak = budget // weak_xp if weak_xp > 0 else 0
    if num_weak >= 4:
        suggestions.append(
            {
                "pattern": "Swarm",
                "composition": f"{num_weak}x Level {party_level - 3} creatures",
                "xp": weak_xp * num_weak,
                "notes": "Many weak enemies. Tests area damage and action economy. Can feel overwhelming.",
            }
        )

    # Elite pair pattern
    elite_xp = creature_xp(party_level + 1, party_level)
    if elite_xp * 2 <= budget + 10:
        suggestions.append(
            {
                "pattern": "Elite Pair",
                "composition": f"2x Level {party_level + 1} creatures",
                "xp": elite_xp * 2,
                "notes": "Two dangerous opponents. Good for fights where PCs must split focus.",
            }
        )

    return {
        "target_threat": target_threat,
        "xp_budget": budget,
        "party_level": party_level,
        "party_size": party_size,
        "suggestions": suggestions,
        "theme": theme,
        "general_advice": VARIETY_ADVICE,
    }


class EncounterServer(MCPServer):
    """MCP server for encounter evaluation tools."""

    def __init__(self):
        self._tools = [
            ToolDef(
                name="evaluate_encounter",
                description="Evaluate an encounter's difficulty given creatures, hazards, and party info. "
                "Returns XP budget, threat level, and tactical advice. "
                'Pass creatures as JSON string: \'[{"name": "Goblin", "level": 1}]\'',
                parameters=[
                    ToolParameter(
                        name="creatures",
                        type="string",
                        description="JSON array of creatures, each with 'name' and 'level' fields. "
                        'Example: \'[{"name": "Goblin", "level": 1}, {"name": "Wolf", "level": 2}]\'',
                    ),
                    ToolParameter(
                        name="party_level",
                        type="integer",
                        description="Average level of the party",
                    ),
                    ToolParameter(
                        name="party_size",
                        type="integer",
                        description="Number of players (default: 4)",
                        required=False,
                        default=4,
                    ),
                    ToolParameter(
                        name="hazards",
                        type="string",
                        description="Optional JSON array of hazards with 'name', 'level', and optional 'is_complex' fields",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="suggest_encounter",
                description="Suggest creature compositions for a target threat level. "
                "Returns multiple patterns (boss+minions, swarm, balanced, etc.) with XP totals.",
                parameters=[
                    ToolParameter(
                        name="target_threat",
                        type="string",
                        description="Desired threat level: trivial, low, moderate, severe, or extreme",
                    ),
                    ToolParameter(
                        name="party_level",
                        type="integer",
                        description="Average level of the party",
                    ),
                    ToolParameter(
                        name="party_size",
                        type="integer",
                        description="Number of players (default: 4)",
                        required=False,
                        default=4,
                    ),
                    ToolParameter(
                        name="theme",
                        type="string",
                        description="Optional theme hint (e.g., 'undead', 'beasts', 'demons')",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="calculate_creature_xp",
                description="Calculate the XP value of a single creature relative to party level.",
                parameters=[
                    ToolParameter(
                        name="creature_level",
                        type="integer",
                        description="The creature's level",
                    ),
                    ToolParameter(
                        name="party_level",
                        type="integer",
                        description="Average level of the party",
                    ),
                ],
            ),
            ToolDef(
                name="get_encounter_advice",
                description="Get GM Core advice on encounter design topics like threat levels, variety, morale, etc.",
                parameters=[
                    ToolParameter(
                        name="topic",
                        type="string",
                        description="Topic to get advice on: threat_levels, variety, morale, or all",
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict) -> ToolResult:
        if name == "evaluate_encounter":
            return self._evaluate_encounter(args)
        elif name == "suggest_encounter":
            return self._suggest_encounter(args)
        elif name == "calculate_creature_xp":
            return self._calculate_creature_xp(args)
        elif name == "get_encounter_advice":
            return self._get_encounter_advice(args)
        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _evaluate_encounter(self, args: dict) -> ToolResult:
        import json

        try:
            # Parse creatures from JSON string or list
            creatures_arg = args.get("creatures", "[]")
            if isinstance(creatures_arg, str):
                creatures_data = json.loads(creatures_arg)
            else:
                creatures_data = creatures_arg

            creatures = [(c["name"], c["level"]) for c in creatures_data]
            party_level = args["party_level"]
            party_size = args.get("party_size", 4)

            # Parse hazards from JSON string or list
            hazards = None
            hazards_arg = args.get("hazards")
            if hazards_arg:
                if isinstance(hazards_arg, str):
                    hazards_data = json.loads(hazards_arg)
                else:
                    hazards_data = hazards_arg
                hazards = [
                    (h["name"], h["level"], h.get("is_complex", False)) for h in hazards_data
                ]

            result = evaluate_encounter(creatures, party_level, party_size, hazards)
            return ToolResult(success=True, data=result)
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {e}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _suggest_encounter(self, args: dict) -> ToolResult:
        try:
            target_threat = args["target_threat"]
            party_level = args["party_level"]
            party_size = args.get("party_size", 4)
            theme = args.get("theme")

            result = suggest_encounter(target_threat, party_level, party_size, theme)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _calculate_creature_xp(self, args: dict) -> ToolResult:
        try:
            c_level = args["creature_level"]
            p_level = args["party_level"]
            xp = creature_xp(c_level, p_level)

            return ToolResult(
                success=True,
                data={
                    "creature_level": c_level,
                    "party_level": p_level,
                    "relative_level": c_level - p_level,
                    "xp": xp,
                    "threat_contribution": (
                        "trivial"
                        if xp <= 10
                        else (
                            "low"
                            if xp <= 30
                            else ("moderate" if xp <= 60 else "severe" if xp <= 120 else "extreme")
                        )
                    ),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_encounter_advice(self, args: dict) -> ToolResult:
        topic = args.get("topic", "all")

        if topic == "threat_levels":
            advice = "# Threat Levels\n\n"
            for level, desc in THREAT_LEVEL_ADVICE.items():
                advice += f"**{level.title()}**: {desc}\n\n"
            return ToolResult(success=True, data={"advice": advice})

        elif topic == "variety":
            return ToolResult(success=True, data={"advice": VARIETY_ADVICE})

        elif topic == "morale":
            return ToolResult(success=True, data={"advice": MORALE_ADVICE})

        elif topic == "all":
            advice = "# Encounter Design Advice\n\n"
            advice += "## Threat Levels\n\n"
            for level, desc in THREAT_LEVEL_ADVICE.items():
                advice += f"**{level.title()}**: {desc}\n\n"
            advice += "## Variety\n" + VARIETY_ADVICE + "\n\n"
            advice += "## Morale\n" + MORALE_ADVICE
            return ToolResult(success=True, data={"advice": advice})

        return ToolResult(success=False, error=f"Unknown topic: {topic}")

    def close(self) -> None:
        """Clean up resources. No-op for stateless server."""
        pass
