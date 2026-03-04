"""PF2e-specific campaign tools plugin.

Provides travel time calculation, hazard detection, AP progress tracking,
and treasure management as a SystemToolPlugin for CampaignStateServer.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ...mcp.base import ToolDef, ToolParameter, ToolResult
from .storage.annotations import AnnotationStore, VALID_ISSUE_TYPES, VALID_SEVERITIES
from .storage.ap_progress import APProgressStore
from .storage.treasure import TreasureStore, TREASURE_BY_LEVEL

# PF2e terrain multipliers for travel speed
TERRAIN_MULTIPLIERS = {
    "easy": 1.5,
    "normal": 1.0,
    "difficult": 0.5,
    "greater_difficult": 0.25,
}


class PF2eCampaignToolPlugin:
    """PF2e-specific tools for CampaignStateServer.

    Provides:
    - calculate_travel_time: Overland travel using PF2e rules
    - check_hazard_detection: Advisory hazard perception checks
    - ap_progress: Adventure Path progression tracking
    - treasure: Party treasure management
    """

    def __init__(self, campaign_id: str, base_dir: Any = None):
        self.campaign_id = campaign_id
        self._base_dir = base_dir
        self._ap_progress_store: APProgressStore | None = None
        self._treasure_store: TreasureStore | None = None
        self._annotation_store: AnnotationStore | None = None

    @property
    def ap_progress(self) -> APProgressStore:
        if self._ap_progress_store is None:
            kwargs: dict[str, Any] = {"campaign_id": self.campaign_id}
            if self._base_dir is not None:
                kwargs["base_dir"] = self._base_dir
            self._ap_progress_store = APProgressStore(**kwargs)
        return self._ap_progress_store

    @property
    def treasure(self) -> TreasureStore:
        if self._treasure_store is None:
            kwargs: dict[str, Any] = {"campaign_id": self.campaign_id}
            if self._base_dir is not None:
                kwargs["base_dir"] = self._base_dir
            self._treasure_store = TreasureStore(**kwargs)
        return self._treasure_store

    @property
    def annotations(self) -> AnnotationStore:
        if self._annotation_store is None:
            kwargs: dict[str, Any] = {"campaign_id": self.campaign_id}
            if self._base_dir is not None:
                kwargs["base_dir"] = self._base_dir
            self._annotation_store = AnnotationStore(**kwargs)
        return self._annotation_store

    def tool_defs(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="run_skill_check",
                description=(
                    "Roll a skill check against a DC and determine degree of success (PF2e rules). "
                    "Provide modifier for auto-roll (d20 + modifier) or roll_total for a GM-provided total. "
                    "Returns skill, dc, degree (critical_success/success/failure/critical_failure), "
                    "total, and roll breakdown."
                ),
                parameters=[
                    ToolParameter(name="skill", type="string", description="Skill name for display (e.g., 'Thievery', 'Religion')"),
                    ToolParameter(name="dc", type="integer", description="Target Difficulty Class"),
                    ToolParameter(
                        name="modifier", type="integer",
                        description="Character's total skill modifier — d20 will be auto-rolled",
                        required=False,
                    ),
                    ToolParameter(
                        name="roll_total", type="integer",
                        description="GM-provided total (skips dice roll; use when rolling physical dice)",
                        required=False,
                    ),
                    ToolParameter(
                        name="fortune", type="boolean",
                        description="Roll twice, take higher (hero point, etc.)",
                        required=False, default=False,
                    ),
                    ToolParameter(
                        name="misfortune", type="boolean",
                        description="Roll twice, take lower (frightened, etc.)",
                        required=False, default=False,
                    ),
                    ToolParameter(
                        name="context", type="string",
                        description="Description for the log (e.g., 'Thievery to disable dart trap')",
                        required=False, default="",
                    ),
                ],
            ),
            ToolDef(
                name="calculate_travel_time",
                description="Calculate overland travel time between locations using PF2e travel rules. "
                "Accounts for speed, terrain, mounting, and forced march.",
                parameters=[
                    ToolParameter(name="from_location", type="string", description="Starting location name"),
                    ToolParameter(name="to_location", type="string", description="Destination location name"),
                    ToolParameter(
                        name="speed_ft", type="integer",
                        description="Base land speed in feet (default: 25)", required=False, default=25,
                    ),
                    ToolParameter(
                        name="terrain", type="string",
                        description="Terrain type: easy, normal, difficult, greater_difficult (default: normal)",
                        required=False, default="normal",
                    ),
                    ToolParameter(
                        name="mounted", type="boolean",
                        description="Whether the party is mounted (default: false)", required=False, default=False,
                    ),
                    ToolParameter(
                        name="forced_march", type="boolean",
                        description="Whether to force-march (+50%% distance, requires Fort saves)",
                        required=False, default=False,
                    ),
                ],
            ),
            ToolDef(
                name="check_hazard_detection",
                description="Advisory tool: check which PCs would detect a hazard based on Stealth DC and "
                "exploration activities. No dice rolls — reports who auto-detects and who needs a check.",
                parameters=[
                    ToolParameter(name="stealth_dc", type="integer", description="The hazard's Stealth DC"),
                    ToolParameter(
                        name="party_perception", type="string",
                        description="JSON map of PC name to Perception modifier, e.g. '{\"Valeros\": 12, \"Ezren\": 8}'",
                        required=False,
                    ),
                    ToolParameter(
                        name="scouting", type="boolean",
                        description="Is someone Scouting? (+1 initiative, not direct detection)",
                        required=False, default=False,
                    ),
                    ToolParameter(
                        name="searching", type="boolean",
                        description="Is someone Searching? (auto-detect if Perception >= DC)",
                        required=False, default=False,
                    ),
                ],
            ),
            ToolDef(
                name="ap_progress",
                description="Track Adventure Path progression. Actions: "
                "'complete_encounter' (mark encounter done, award XP), "
                "'explore_area' (mark area explored), "
                "'milestone' (record a milestone), "
                "'get_progress' (view progress for a book), "
                "'list_incomplete' (show upcoming entries).",
                parameters=[
                    ToolParameter(
                        name="action", type="string",
                        description="Action: complete_encounter, explore_area, milestone, get_progress, list_incomplete",
                    ),
                    ToolParameter(
                        name="name", type="string",
                        description="Entry name (e.g., 'Goblin Ambush', 'Room B12')",
                        required=False,
                    ),
                    ToolParameter(
                        name="xp", type="integer",
                        description="XP to award (for complete_encounter/milestone)",
                        required=False, default=0,
                    ),
                    ToolParameter(
                        name="book", type="string",
                        description="Book name for filtering (e.g., 'Abomination Vaults Book 1')",
                        required=False,
                    ),
                    ToolParameter(
                        name="description", type="string",
                        description="Optional notes about the entry",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="treasure",
                description="Track party treasure and loot. Actions: "
                "'log' (add item to party loot), "
                "'distribute' (give item to a character), "
                "'sell' (sell item at half price), "
                "'wealth' (show party wealth summary), "
                "'by_level' (compare wealth to expected for party level).",
                parameters=[
                    ToolParameter(
                        name="action", type="string",
                        description="Action: log, distribute, sell, wealth, by_level",
                    ),
                    ToolParameter(
                        name="item_name", type="string",
                        description="Item name (for log action)",
                        required=False,
                    ),
                    ToolParameter(
                        name="value_gp", type="number",
                        description="Item value in gold pieces (for log action)",
                        required=False, default=0,
                    ),
                    ToolParameter(
                        name="item_level", type="integer",
                        description="Item level (for log action)",
                        required=False, default=0,
                    ),
                    ToolParameter(
                        name="character", type="string",
                        description="Character name (for distribute action)",
                        required=False,
                    ),
                    ToolParameter(
                        name="item_id", type="integer",
                        description="Item ID (for distribute/sell actions)",
                        required=False,
                    ),
                    ToolParameter(
                        name="source", type="string",
                        description="Where the item came from (for log action)",
                        required=False,
                    ),
                    ToolParameter(
                        name="party_level", type="integer",
                        description="Party level (for by_level action)",
                        required=False,
                    ),
                    ToolParameter(
                        name="party_size", type="integer",
                        description="Number of PCs (for by_level action)",
                        required=False, default=4,
                    ),
                ],
            ),
            ToolDef(
                name="data_quality",
                description="Report data quality issues in search results. Actions: "
                "'flag' (report a problem with an entity), "
                "'list' (show flagged issues), "
                "'count' (total flagged issues).",
                parameters=[
                    ToolParameter(
                        name="action", type="string",
                        description="Action: flag, list, count",
                    ),
                    ToolParameter(
                        name="entity_name", type="string",
                        description="Entity name (required for flag)",
                        required=False,
                    ),
                    ToolParameter(
                        name="entity_type", type="string",
                        description="Entity type (e.g., creature, spell)",
                        required=False,
                    ),
                    ToolParameter(
                        name="entity_id", type="string",
                        description="Entity ID from search results",
                        required=False,
                    ),
                    ToolParameter(
                        name="book", type="string",
                        description="Source book name",
                        required=False,
                    ),
                    ToolParameter(
                        name="page", type="integer",
                        description="Page number",
                        required=False,
                    ),
                    ToolParameter(
                        name="issue", type="string",
                        description="Issue type: truncated, mistyped, missing, wrong_info, search_miss",
                        required=False, default="wrong_info",
                    ),
                    ToolParameter(
                        name="severity", type="string",
                        description="Severity: low, medium, high, critical",
                        required=False, default="medium",
                    ),
                    ToolParameter(
                        name="description", type="string",
                        description="Description of the issue",
                        required=False,
                    ),
                    ToolParameter(
                        name="search_query", type="string",
                        description="The search query that surfaced the issue",
                        required=False,
                    ),
                ],
            ),
        ]

    def call_tool(self, name: str, args: dict[str, Any], server: Any) -> ToolResult:
        if name == "run_skill_check":
            return self._run_skill_check(args)
        elif name == "calculate_travel_time":
            return self._calculate_travel_time(args, server)
        elif name == "check_hazard_detection":
            return self._check_hazard_detection(args)
        elif name == "ap_progress":
            return self._ap_progress(args)
        elif name == "treasure":
            return self._treasure(args)
        elif name == "data_quality":
            return self._data_quality(args)
        raise KeyError(name)

    def close(self) -> None:
        if self._ap_progress_store:
            self._ap_progress_store.close()
            self._ap_progress_store = None
        if self._treasure_store:
            self._treasure_store.close()
            self._treasure_store = None
        if self._annotation_store:
            self._annotation_store.close()
            self._annotation_store = None

    # ---------------------------------------------------------------
    # Skill check
    # ---------------------------------------------------------------

    def _run_skill_check(self, args: dict[str, Any]) -> ToolResult:
        from gm_agent.mcp.dice import check_result, roll_dice, roll_fortune, roll_misfortune  # pylint: disable=import-outside-toplevel

        skill = args.get("skill", "Unknown")
        dc = int(args.get("dc", 20))
        fortune = args.get("fortune", False)
        misfortune = args.get("misfortune", False)
        context = args.get("context", "")
        roll_total = args.get("roll_total")
        modifier = args.get("modifier")

        if roll_total is None and modifier is None:
            return ToolResult(
                success=False,
                error="Provide either 'roll_total' (manual total) or 'modifier' (for auto-roll).",
            )

        if roll_total is not None:
            total = int(roll_total)
            result = check_result(total, dc)
            degree = result["degree"]
            roll_breakdown = f"Manual: {total}"
            auto_rolled = False
        else:
            modifier = int(modifier)
            if fortune and not misfortune:
                r = roll_fortune("1d20")
                natural_roll = r["chosen"]["rolls"][0]
                d20 = r["result"]
                r1, r2 = r["roll_1"]["rolls"][0], r["roll_2"]["rolls"][0]
                roll_breakdown = f"d20({r1}, {r2} → {natural_roll}) + {modifier}"
            elif misfortune and not fortune:
                r = roll_misfortune("1d20")
                natural_roll = r["chosen"]["rolls"][0]
                d20 = r["result"]
                r1, r2 = r["roll_1"]["rolls"][0], r["roll_2"]["rolls"][0]
                roll_breakdown = f"d20({r1}, {r2} → {natural_roll}) + {modifier}"
            else:
                d20_result = roll_dice("1d20")
                natural_roll = d20_result.rolls[0]
                d20 = d20_result.total
                roll_breakdown = f"d20({natural_roll}) + {modifier}"

            total = d20 + modifier
            result = check_result(total, dc)
            degree = result["degree"]

            # Nat 20/1 upgrades
            if natural_roll == 20 and degree in ("success", "failure"):
                degree = "critical_success" if degree == "success" else "success"
            elif natural_roll == 1 and degree in ("success", "failure"):
                degree = "failure" if degree == "success" else "critical_failure"

            roll_breakdown = f"{roll_breakdown} = {total}"
            auto_rolled = True

        description = degree.replace("_", " ").title()

        lines = [
            f"**{skill} Check** vs DC {dc}",
            f"**Roll:** {roll_breakdown}",
            f"**Result:** {description}",
        ]
        if context:
            lines.append(f"*{context}*")

        return ToolResult(success=True, data={
            "skill": skill,
            "dc": dc,
            "degree": degree,
            "total": total,
            "description": description,
            "context": context,
            "auto_rolled": auto_rolled,
            "roll_breakdown": roll_breakdown,
            "formatted": "\n".join(lines),
        })

    # ---------------------------------------------------------------
    # Travel time
    # ---------------------------------------------------------------

    def _calculate_travel_time(self, args: dict[str, Any], server: Any) -> ToolResult:
        from_name = args.get("from_location", "")
        to_name = args.get("to_location", "")
        speed_ft = int(args.get("speed_ft", 25))
        terrain = args.get("terrain", "normal")
        mounted = args.get("mounted", False)
        forced_march = args.get("forced_march", False)

        if not from_name or not to_name:
            return ToolResult(success=False, error="Both from_location and to_location are required.")

        # Resolve locations and count hops
        from_loc = server.locations.get_by_name(from_name)
        to_loc = server.locations.get_by_name(to_name)

        hops = 1
        if from_loc and to_loc:
            if to_loc.id in from_loc.connected_locations:
                hops = 1
            else:
                hops = self._bfs_hops(from_loc.id, to_loc.id, server.locations) or 1

        base_speed = 40 if mounted else speed_ft
        mph = base_speed / 25 * 3
        miles_per_day = mph * 8

        terrain_mult = TERRAIN_MULTIPLIERS.get(terrain, 1.0)
        miles_per_day *= terrain_mult

        if forced_march:
            miles_per_day *= 1.5

        distance_miles = hops * 24
        travel_days = distance_miles / miles_per_day if miles_per_day > 0 else float("inf")

        lines = [
            f"**Travel: {from_name} -> {to_name}**",
            f"**Distance:** ~{distance_miles} miles ({hops} hop{'s' if hops != 1 else ''})",
            f"**Speed:** {base_speed} ft. ({mph:.1f} mph)",
            f"**Terrain:** {terrain} (x{terrain_mult})",
            f"**Daily Travel:** ~{miles_per_day:.0f} miles/day",
            f"**Estimated Time:** ~{travel_days:.1f} days",
        ]

        if forced_march:
            lines.append("**Forced March:** +50% distance; Fort saves required (DC 10 + 1/hour after 8h)")
        if mounted:
            lines.append("**Mounted:** Using mount speed (40 ft.)")

        return ToolResult(success=True, data="\n".join(lines))

    @staticmethod
    def _bfs_hops(start_id: str, end_id: str, locations: Any) -> int | None:
        visited = {start_id}
        queue = deque([(start_id, 0)])
        while queue:
            current, depth = queue.popleft()
            loc = locations.get(current)
            if not loc:
                continue
            for neighbor_id in loc.connected_locations:
                if neighbor_id == end_id:
                    return depth + 1
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
        return None

    # ---------------------------------------------------------------
    # Hazard detection
    # ---------------------------------------------------------------

    def _check_hazard_detection(self, args: dict[str, Any]) -> ToolResult:
        import json

        stealth_dc = int(args.get("stealth_dc", 20))
        scouting = args.get("scouting", False)
        searching = args.get("searching", False)

        party_perception_str = args.get("party_perception", "{}")
        try:
            if isinstance(party_perception_str, str):
                party_perception = json.loads(party_perception_str)
            else:
                party_perception = party_perception_str
        except (json.JSONDecodeError, TypeError):
            party_perception = {}

        lines = [f"**Hazard Detection Check** (Stealth DC {stealth_dc})"]
        if not party_perception:
            lines.append("No party perception data provided — provide party_perception for specific analysis.")
            lines.append(f"A character needs Perception >= {stealth_dc} to detect (if Searching).")
            if searching:
                lines.append("Someone is Searching: auto-detect if Perception meets DC.")
            return ToolResult(success=True, data="\n".join(lines))

        auto_detect = []
        needs_check = []
        cannot_detect = []

        for pc_name, perception in party_perception.items():
            perception = int(perception)
            if searching and perception >= stealth_dc:
                auto_detect.append(f"{pc_name} (Perception +{perception})")
            elif perception >= stealth_dc - 5:
                needs_check.append(f"{pc_name} (Perception +{perception}, needs roll)")
            else:
                cannot_detect.append(f"{pc_name} (Perception +{perception})")

        if auto_detect:
            lines.append(f"\n**Auto-Detect (Searching):** {', '.join(auto_detect)}")
        if needs_check:
            lines.append(f"**Needs Check:** {', '.join(needs_check)}")
        if cannot_detect:
            lines.append(f"**Unlikely to Detect:** {', '.join(cannot_detect)}")

        if scouting:
            lines.append("\n*Scout active: party gets +1 circumstance bonus to initiative if combat starts.*")

        return ToolResult(success=True, data="\n".join(lines))

    # ---------------------------------------------------------------
    # AP progress
    # ---------------------------------------------------------------

    def _ap_progress(self, args: dict[str, Any]) -> ToolResult:
        action = args.get("action", "")
        name = args.get("name", "")
        book = args.get("book", "")
        xp = int(args.get("xp", 0))
        description = args.get("description", "")

        if action == "complete_encounter":
            if not name:
                return ToolResult(success=False, error="'name' is required for complete_encounter")
            self.ap_progress.mark_complete(
                name=name, entry_type="encounter", book=book,
                xp_awarded=xp, notes=description,
            )
            return ToolResult(
                success=True,
                data=f"Encounter completed: **{name}**"
                + (f" (+{xp} XP)" if xp else "")
                + (f"\nTotal XP awarded: {self.ap_progress.total_xp()}" if xp else ""),
            )
        elif action == "explore_area":
            if not name:
                return ToolResult(success=False, error="'name' is required for explore_area")
            self.ap_progress.mark_complete(
                name=name, entry_type="area", book=book, notes=description,
            )
            return ToolResult(success=True, data=f"Area explored: **{name}**")
        elif action == "milestone":
            if not name:
                return ToolResult(success=False, error="'name' is required for milestone")
            self.ap_progress.mark_complete(
                name=name, entry_type="milestone", book=book,
                xp_awarded=xp, notes=description,
            )
            return ToolResult(
                success=True,
                data=f"Milestone reached: **{name}**"
                + (f" (+{xp} XP)" if xp else ""),
            )
        elif action == "get_progress":
            entries = self.ap_progress.get_progress(book=book)
            if not entries:
                return ToolResult(success=True, data="No progress entries recorded yet.")
            lines = [f"**AP Progress** ({len(entries)} entries)"]
            if book:
                summary = self.ap_progress.get_book_progress(book)
                lines.append(f"**Book:** {book} — {summary['total_xp']} XP total")
                for etype, info in summary["types"].items():
                    lines.append(f"  {etype}: {info['count']} ({info['xp']} XP)")
            else:
                for e in entries:
                    check = "x" if e["completed"] else " "
                    lines.append(f"  [{check}] {e['entry_type']}: {e['name']}"
                                + (f" (+{e['xp_awarded']} XP)" if e["xp_awarded"] else ""))
            lines.append(f"\n**Total XP:** {self.ap_progress.total_xp()}")
            return ToolResult(success=True, data="\n".join(lines))
        elif action == "list_incomplete":
            entries = self.ap_progress.list_incomplete(book=book)
            if not entries:
                return ToolResult(success=True, data="No incomplete entries.")
            lines = [f"**Incomplete Entries** ({len(entries)})"]
            for e in entries:
                lines.append(f"  [ ] {e['entry_type']}: {e['name']}")
            return ToolResult(success=True, data="\n".join(lines))
        else:
            return ToolResult(
                success=False,
                error=f"Unknown ap_progress action '{action}'. "
                "Valid: complete_encounter, explore_area, milestone, get_progress, list_incomplete",
            )

    # ---------------------------------------------------------------
    # Treasure
    # ---------------------------------------------------------------

    def _treasure(self, args: dict[str, Any]) -> ToolResult:
        import json

        action = args.get("action", "")

        if action == "log":
            item_name = args.get("item_name", "")
            if not item_name:
                return ToolResult(success=False, error="'item_name' is required for log")
            value_gp = float(args.get("value_gp", 0))
            item_level = int(args.get("item_level", 0))
            source = args.get("source", "")
            entry = self.treasure.add_item(
                item_name=item_name, value_gp=value_gp,
                item_level=item_level, source=source,
            )
            return ToolResult(
                success=True,
                data=f"Logged: **{item_name}** (Level {item_level}, {value_gp} gp)"
                + (f" from {source}" if source else "")
                + f" [ID: {entry.id}]",
            )
        elif action == "distribute":
            item_id = args.get("item_id")
            character = args.get("character", "")
            if not item_id or not character:
                return ToolResult(success=False, error="Both 'item_id' and 'character' are required for distribute")
            success = self.treasure.distribute_item(int(item_id), character)
            if success:
                return ToolResult(success=True, data=f"Item #{item_id} distributed to {character}.")
            return ToolResult(success=False, error=f"Item #{item_id} not found.")
        elif action == "sell":
            item_id = args.get("item_id")
            if not item_id:
                return ToolResult(success=False, error="'item_id' is required for sell")
            sale_value = self.treasure.sell_item(int(item_id))
            if sale_value > 0:
                return ToolResult(success=True, data=f"Item #{item_id} sold for {sale_value:.1f} gp.")
            return ToolResult(success=False, error=f"Item #{item_id} not found.")
        elif action == "wealth":
            wealth = self.treasure.get_party_wealth()
            lines = [
                "**Party Wealth Summary**",
                f"**Total Items:** {wealth['total_items']}",
                f"**Total Value:** {wealth['total_value_gp']:.1f} gp",
                f"**Sold Income:** {wealth['sold_income_gp']:.1f} gp",
                f"**Effective Wealth:** {wealth['effective_wealth_gp']:.1f} gp",
            ]
            if wealth["holders"]:
                lines.append("\n**By Holder:**")
                for holder, info in wealth["holders"].items():
                    lines.append(f"  {holder}: {info['count']} items ({info['value_gp']:.1f} gp)")
            return ToolResult(success=True, data="\n".join(lines))
        elif action == "by_level":
            party_level = int(args.get("party_level", 1))
            party_size = int(args.get("party_size", 4))
            comparison = self.treasure.get_wealth_by_level(party_level, party_size)
            diff = comparison["difference_gp"]
            status = "on track" if abs(diff) < comparison["expected_wealth_gp"] * 0.1 else (
                "ahead" if diff > 0 else "behind"
            )
            lines = [
                f"**Wealth vs. Expected (Level {party_level}, {party_size} PCs)**",
                f"**Current:** {comparison['current_wealth_gp']:.1f} gp",
                f"**Expected:** {comparison['expected_wealth_gp']:.1f} gp",
                f"**Difference:** {diff:+.1f} gp ({comparison['percentage']}%)",
                f"**Status:** {status}",
            ]
            return ToolResult(success=True, data="\n".join(lines))
        else:
            return ToolResult(
                success=False,
                error=f"Unknown treasure action '{action}'. "
                "Valid: log, distribute, sell, wealth, by_level",
            )

    # ---------------------------------------------------------------
    # Data quality annotations
    # ---------------------------------------------------------------

    def _data_quality(self, args: dict[str, Any]) -> ToolResult:
        action = args.get("action", "")

        if action == "flag":
            entity_name = args.get("entity_name", "")
            if not entity_name:
                return ToolResult(success=False, error="'entity_name' is required for flag")

            issue = args.get("issue", "wrong_info")
            if issue not in VALID_ISSUE_TYPES:
                return ToolResult(
                    success=False,
                    error=f"Invalid issue type '{issue}'. "
                    f"Valid: {', '.join(sorted(VALID_ISSUE_TYPES))}",
                )

            severity = args.get("severity", "medium")
            if severity not in VALID_SEVERITIES:
                return ToolResult(
                    success=False,
                    error=f"Invalid severity '{severity}'. "
                    f"Valid: {', '.join(sorted(VALID_SEVERITIES))}",
                )

            entry = self.annotations.add_annotation(
                entity_name=entity_name,
                issue_type=issue,
                entity_type=args.get("entity_type", ""),
                entity_id=args.get("entity_id", ""),
                book=args.get("book", ""),
                page=args.get("page"),
                severity=severity,
                description=args.get("description", ""),
                search_query=args.get("search_query", ""),
            )
            return ToolResult(
                success=True,
                data=f"Flagged: **{entity_name}** ({issue}, {entry.severity})"
                + (f" — {entry.description}" if entry.description else ""),
            )

        elif action == "list":
            items = self.annotations.list_annotations()
            if not items:
                return ToolResult(success=True, data="No data quality issues flagged.")
            lines = [f"**Data Quality Issues** ({len(items)} total)"]
            for item in items[:20]:
                sev = item["severity"]
                lines.append(
                    f"  [{sev}] {item['entity_name']} ({item['issue_type']})"
                    + (f": {item['description']}" if item["description"] else "")
                )
            if len(items) > 20:
                lines.append(f"  ... and {len(items) - 20} more")
            return ToolResult(success=True, data="\n".join(lines))

        elif action == "count":
            total = self.annotations.count()
            return ToolResult(success=True, data=f"**Data quality issues:** {total}")

        else:
            return ToolResult(
                success=False,
                error=f"Unknown data_quality action '{action}'. Valid: flag, list, count",
            )
