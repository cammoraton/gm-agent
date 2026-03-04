"""Subsystem MCP server for stateful encounter subsystems.

Provides tools for running Victory Point subsystems, Chases, Influence,
Research, Infiltration, and complex Hazards with state tracking.
"""

import json
from datetime import datetime
from typing import Any

from gm_agent.config import CAMPAIGNS_DIR
from gm_agent.mcp.dice import check_result, roll_dice
from gm_agent.storage.subsystems import SubsystemStore, SubsystemInstance
from gm_agent.mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult


class SubsystemServer(MCPServer):
    """MCP server for encounter subsystem state machines.

    Campaign-scoped, stateful server managing VP tracking, chases,
    complex hazards, influence encounters, and infiltration subsystems.
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self._store: SubsystemStore | None = None
        self._tools = self._build_tools()

    @property
    def store(self) -> SubsystemStore:
        """Lazy-load subsystem store."""
        if self._store is None:
            self._store = SubsystemStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._store

    def _build_tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="start_subsystem",
                description=(
                    "Initialize a new encounter subsystem. Supports VP (victory points), "
                    "influence, research, chase, infiltration, hazard, and haunt types. "
                    "Config varies by type: VP/influence/research need targets+thresholds, "
                    "chase needs participants+length+obstacles, hazard needs HP/routine/disable conditions, "
                    "infiltration needs detection_threshold+edge_benefits, "
                    "haunt needs trigger_condition/disable_conditions/spiritual_immunity."
                ),
                parameters=[
                    ToolParameter(
                        name="type",
                        type="string",
                        description="Subsystem type: 'vp', 'influence', 'research', 'chase', 'infiltration', 'hazard', 'haunt'",
                    ),
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Human-readable name (e.g., 'Chase Through the Market', 'Influence Lord Gyr')",
                    ),
                    ToolParameter(
                        name="config",
                        type="string",
                        description=(
                            "JSON config string. Examples:\n"
                            "VP: {\"targets\": {\"Lord Gyr\": {\"minor\": 3, \"major\": 6}}}\n"
                            "Influence: {\"npcs\": {\"Lord Gyr\": {\"approach_dcs\": {\"Diplomacy\": 20, \"Intimidation\": 24}, \"weaknesses\": [\"flattery\"], \"resistances\": [\"threats\"], \"min_vp\": 0, \"max_vp\": 5}}}\n"
                            "Chase: {\"participants\": [\"Party\", \"Thief\"], \"chase_length\": 10, \"obstacles\": [{\"name\": \"Crowded Market\", \"skill\": \"Athletics\", \"dc\": 18, \"success_advances\": 1, \"cf_retreats\": 1}]}\n"
                            "Research: {\"sources\": [{\"name\": \"Ancient Tome\", \"skill\": \"Arcana\", \"dc\": 20, \"per_success\": 2, \"max_contribution\": 6, \"discovery_note\": \"The tome reveals...\"}], \"vp_target\": 12}\n"
                            "Hazard: {\"hp\": 50, \"hardness\": 10, \"routine_actions\": [\"Fires darts\"], \"disable_conditions\": [{\"skill\": \"Thievery\", \"dc\": 22}]}\n"
                            "Haunt: {\"hp\": 30, \"trigger_condition\": \"...\", \"disable_conditions\": [{\"skill\": \"Religion\", \"dc\": 20, \"successes_needed\": 2}], \"spiritual_immunity\": true, \"anchor\": \"...\", \"reset_interval\": \"1 day\"}\n"
                            "Infiltration: {\"detection_threshold\": 10, \"edge_benefits\": [{\"cost\": 1, \"description\": \"Bypass one guard\"}, {\"cost\": 2, \"description\": \"Stolen uniform\"}], \"targets\": {\"Main Objective\": {\"minor\": 4, \"major\": 8}}}"
                        ),
                        required=False,
                        default="{}",
                    ),
                ],
            ),
            ToolDef(
                name="subsystem_action",
                description=(
                    "Take an action in a running subsystem. Actions: "
                    "'add_vp' (target, amount), 'remove_vp' (target, amount), "
                    "'advance_round', 'move' (participant, distance), "
                    "'damage_hazard' (amount), 'disable_hazard', "
                    "'hazard_routine' (advance the hazard's routine), "
                    "'hazard_attack' (attack_bonus, target_ac, optional roll_total; hazard/haunt), "
                    "'hazard_save' (save_dc, save_type, save_modifier, optional roll_total; hazard/haunt), "
                    "'add_awareness' (amount, for infiltration), "
                    "'earn_edge' (amount, action; infiltration), "
                    "'spend_edge' (benefit_index OR benefit, amount; infiltration), "
                    "'attempt_disable' (condition_index, skill, modifier OR roll_total; hazard/haunt), "
                    "'attempt_influence' (npc, approach, modifier OR roll_total, optional situation; influence), "
                    "'attempt_obstacle' (participant, obstacle_index, skill, modifier OR roll_total; chase), "
                    "'research_check' (source_index, skill, modifier OR roll_total; research), "
                    "'trigger_haunt' (haunt only), "
                    "'reset_haunt' (haunt only). "
                    "Returns updated state and any threshold notifications."
                ),
                parameters=[
                    ToolParameter(
                        name="subsystem_id",
                        type="string",
                        description="The subsystem instance ID",
                    ),
                    ToolParameter(
                        name="action",
                        type="string",
                        description=(
                            "Action type: 'add_vp', 'remove_vp', 'advance_round', 'move', "
                            "'damage_hazard', 'disable_hazard', 'hazard_routine', 'hazard_attack', 'hazard_save', "
                            "'add_awareness', 'earn_edge', 'spend_edge', "
                            "'attempt_disable', 'attempt_influence', 'attempt_obstacle', 'research_check', "
                            "'trigger_haunt', 'reset_haunt'"
                        ),
                    ),
                    ToolParameter(
                        name="args",
                        type="string",
                        description=(
                            "JSON args for the action. "
                            "attempt_disable: {\"condition_index\": 0, \"skill\": \"Religion\", \"modifier\": 8} "
                            "or {\"condition_index\": 0, \"skill\": \"Religion\", \"roll_total\": 24}. "
                            "attempt_influence: {\"npc\": \"Lord Gyr\", \"approach\": \"Diplomacy\", \"modifier\": 8, \"situation\": \"using flattery\"}. "
                            "attempt_obstacle: {\"participant\": \"Party\", \"obstacle_index\": 0, \"skill\": \"Athletics\", \"modifier\": 6}. "
                            "research_check: {\"source_index\": 0, \"skill\": \"Arcana\", \"modifier\": 10}. "
                            "hazard_attack: {\"attack_bonus\": 12, \"target_ac\": 18}. "
                            "hazard_save: {\"save_dc\": 20, \"save_type\": \"Reflex\", \"save_modifier\": 7}."
                        ),
                        required=False,
                        default="{}",
                    ),
                ],
            ),
            ToolDef(
                name="get_subsystem_state",
                description="Get the current state of a subsystem: VP totals, round, positions, HP, awareness, etc.",
                parameters=[
                    ToolParameter(
                        name="subsystem_id",
                        type="string",
                        description="The subsystem instance ID",
                    ),
                    ToolParameter(
                        name="include_log",
                        type="boolean",
                        description="Include the action history log",
                        required=False,
                        default=False,
                    ),
                ],
            ),
            ToolDef(
                name="end_subsystem",
                description="Mark a subsystem as completed, failed, or abandoned. Returns a summary.",
                parameters=[
                    ToolParameter(
                        name="subsystem_id",
                        type="string",
                        description="The subsystem instance ID",
                    ),
                    ToolParameter(
                        name="outcome",
                        type="string",
                        description="Outcome: 'completed', 'failed', 'abandoned'",
                    ),
                ],
            ),
            ToolDef(
                name="list_subsystems",
                description="List all active subsystems in the current campaign.",
                parameters=[],
            ),
            ToolDef(
                name="attempt_disable",
                description=(
                    "Attempt to fulfill one disable condition on a hazard or haunt via skill check. "
                    "Provide modifier for auto-roll (d20 + modifier) or roll_total for a GM-provided total. "
                    "Critical success instantly completes the condition; success advances progress; "
                    "critical failure on a haunt warns that it triggers. "
                    "When all conditions meet their required successes the hazard/haunt is disabled."
                ),
                parameters=[
                    ToolParameter(name="subsystem_id", type="string", description="The subsystem instance ID"),
                    ToolParameter(
                        name="condition_index", type="integer",
                        description="Index of the disable condition to attempt (0-based)",
                    ),
                    ToolParameter(
                        name="skill", type="string",
                        description="Skill used (for display/log, e.g. 'Religion', 'Thievery')",
                        required=False, default="",
                    ),
                    ToolParameter(
                        name="modifier", type="integer",
                        description="Character's total skill modifier — d20 will be auto-rolled",
                        required=False,
                    ),
                    ToolParameter(
                        name="roll_total", type="integer",
                        description="GM-provided roll total (skips dice roll)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="trigger_haunt",
                description="Trigger the haunt's effect. Marks the haunt as triggered and returns the trigger description.",
                parameters=[
                    ToolParameter(name="subsystem_id", type="string", description="The haunt subsystem instance ID"),
                ],
            ),
            ToolDef(
                name="reset_haunt",
                description=(
                    "Reset a haunt after its reset interval has passed. "
                    "Clears triggered state, zeroes disable progress, restores HP, and sets status back to active."
                ),
                parameters=[
                    ToolParameter(name="subsystem_id", type="string", description="The haunt subsystem instance ID"),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "start_subsystem":
                config_str = args.get("config", "{}")
                config = json.loads(config_str) if isinstance(config_str, str) else config_str
                return self._start_subsystem(args["type"], args["name"], config)
            elif name == "subsystem_action":
                action_args_str = args.get("args", "{}")
                action_args = json.loads(action_args_str) if isinstance(action_args_str, str) else action_args_str
                return self._subsystem_action(args["subsystem_id"], args["action"], action_args)
            elif name == "get_subsystem_state":
                return self._get_subsystem_state(
                    args["subsystem_id"], args.get("include_log", False)
                )
            elif name == "end_subsystem":
                return self._end_subsystem(args["subsystem_id"], args["outcome"])
            elif name == "list_subsystems":
                return self._list_subsystems()
            elif name == "attempt_disable":
                # Convenience wrapper: routes through subsystem_action
                return self._subsystem_action(
                    args["subsystem_id"], "attempt_disable", {
                        "condition_index": args.get("condition_index", 0),
                        "skill": args.get("skill", ""),
                        **({k: args[k] for k in ("modifier", "roll_total") if k in args}),
                    },
                )
            elif name == "trigger_haunt":
                return self._subsystem_action(args["subsystem_id"], "trigger_haunt", {})
            elif name == "reset_haunt":
                return self._subsystem_action(args["subsystem_id"], "reset_haunt", {})
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON in args: {e}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    # -------------------------------------------------------------------
    # Tool: start_subsystem
    # -------------------------------------------------------------------

    def _start_subsystem(
        self, subsystem_type: str, name: str, config: dict[str, Any],
    ) -> ToolResult:
        valid_types = ("vp", "influence", "research", "chase", "infiltration", "hazard", "exploration", "haunt")
        if subsystem_type not in valid_types:
            return ToolResult(
                success=False,
                error=f"Invalid subsystem type '{subsystem_type}'. Must be one of: {', '.join(valid_types)}",
            )

        instance = self.store.create(subsystem_type, name, config)
        state_text = self._format_state(instance, include_log=False)

        return ToolResult(
            success=True,
            data=f"**Subsystem Started:** {name} (ID: {instance.id})\n\n{state_text}",
        )

    # -------------------------------------------------------------------
    # Tool: subsystem_action
    # -------------------------------------------------------------------

    def _subsystem_action(
        self, subsystem_id: str, action: str, action_args: dict[str, Any],
    ) -> ToolResult:
        instance = self.store.get(subsystem_id)
        if instance is None:
            return ToolResult(success=False, error=f"Subsystem '{subsystem_id}' not found.")

        if instance.status != "active":
            return ToolResult(
                success=False,
                error=f"Subsystem '{instance.name}' is {instance.status}, not active.",
            )

        notifications: list[str] = []

        # --- State validation guards ---
        hazard_or_haunt_actions = (
            "damage_hazard", "disable_hazard", "hazard_routine", "attempt_disable",
            "hazard_attack", "hazard_save",
        )
        if action in hazard_or_haunt_actions:
            if instance.subsystem_type not in ("hazard", "haunt"):
                return ToolResult(
                    success=False,
                    error=f"Action '{action}' is only valid for hazard/haunt subsystems, "
                    f"not {instance.subsystem_type}.",
                )
            if instance.destroyed:
                return ToolResult(success=False, error="Hazard is already destroyed.")
            if instance.disabled:
                return ToolResult(success=False, error="Hazard is already disabled.")

        haunt_only_actions = ("trigger_haunt", "reset_haunt")
        if action in haunt_only_actions:
            if instance.subsystem_type != "haunt":
                return ToolResult(
                    success=False,
                    error=f"Action '{action}' is only valid for haunt subsystems, "
                    f"not {instance.subsystem_type}.",
                )

        if action == "move" and instance.subsystem_type != "chase":
            return ToolResult(
                success=False,
                error=f"Action 'move' is only valid for chase subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action == "attempt_obstacle" and instance.subsystem_type != "chase":
            return ToolResult(
                success=False,
                error=f"Action 'attempt_obstacle' is only valid for chase subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action == "attempt_influence" and instance.subsystem_type != "influence":
            return ToolResult(
                success=False,
                error=f"Action 'attempt_influence' is only valid for influence subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action == "research_check" and instance.subsystem_type != "research":
            return ToolResult(
                success=False,
                error=f"Action 'research_check' is only valid for research subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action in ("earn_edge", "spend_edge") and instance.subsystem_type != "infiltration":
            return ToolResult(
                success=False,
                error=f"Action '{action}' is only valid for infiltration subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action == "add_awareness" and instance.subsystem_type != "infiltration":
            return ToolResult(
                success=False,
                error=f"Action 'add_awareness' is only valid for infiltration subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action == "set_activity" and instance.subsystem_type != "exploration":
            return ToolResult(
                success=False,
                error=f"Action 'set_activity' is only valid for exploration subsystems, "
                f"not {instance.subsystem_type}.",
            )
        if action == "set_marching_order" and instance.subsystem_type != "exploration":
            return ToolResult(
                success=False,
                error=f"Action 'set_marching_order' is only valid for exploration subsystems, "
                f"not {instance.subsystem_type}.",
            )

        if action == "add_vp":
            notifications = self._action_add_vp(instance, action_args)
        elif action == "remove_vp":
            notifications = self._action_remove_vp(instance, action_args)
        elif action == "advance_round":
            notifications = self._action_advance_round(instance, action_args)
        elif action == "move":
            notifications = self._action_move(instance, action_args)
        elif action == "damage_hazard":
            notifications = self._action_damage_hazard(instance, action_args)
        elif action == "disable_hazard":
            notifications = self._action_disable_hazard(instance, action_args)
        elif action == "hazard_routine":
            notifications = self._action_hazard_routine(instance, action_args)
        elif action == "add_awareness":
            notifications = self._action_add_awareness(instance, action_args)
        elif action == "set_activity":
            notifications = self._action_set_activity(instance, action_args)
        elif action == "set_marching_order":
            notifications = self._action_set_marching_order(instance, action_args)
        elif action == "attempt_disable":
            notifications = self._action_attempt_disable(instance, action_args)
        elif action == "hazard_attack":
            notifications = self._action_hazard_attack(instance, action_args)
        elif action == "hazard_save":
            notifications = self._action_hazard_save(instance, action_args)
        elif action == "attempt_influence":
            notifications = self._action_attempt_influence(instance, action_args)
        elif action == "attempt_obstacle":
            notifications = self._action_attempt_obstacle(instance, action_args)
        elif action == "research_check":
            notifications = self._action_research_check(instance, action_args)
        elif action == "earn_edge":
            notifications = self._action_earn_edge(instance, action_args)
        elif action == "spend_edge":
            notifications = self._action_spend_edge(instance, action_args)
        elif action == "trigger_haunt":
            notifications = self._action_trigger_haunt(instance, action_args)
        elif action == "reset_haunt":
            notifications = self._action_reset_haunt(instance, action_args)
        else:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown action '{action}'. Valid: add_vp, remove_vp, advance_round, "
                    "move, damage_hazard, disable_hazard, hazard_routine, hazard_attack, hazard_save, "
                    "add_awareness, earn_edge, spend_edge, "
                    "attempt_disable, attempt_influence, attempt_obstacle, research_check, "
                    "trigger_haunt, reset_haunt"
                ),
            )

        # Log the action
        instance.action_log.append({
            "action": action,
            "args": action_args,
            "round": instance.round_number,
            "timestamp": datetime.now().isoformat(),
        })

        self.store.save(instance)

        # Build response
        lines = []
        if notifications:
            for note in notifications:
                lines.append(f"**!** {note}")
            lines.append("")

        lines.append(self._format_state(instance, include_log=False))
        return ToolResult(success=True, data="\n".join(lines))

    def _action_add_vp(self, instance: SubsystemInstance, args: dict) -> list[str]:
        target = args.get("target", "")
        amount = int(args.get("amount", 1))
        notifications = []

        if target not in instance.victory_points:
            # Auto-create target if not present
            instance.victory_points[target] = 0
            if target not in instance.thresholds:
                instance.thresholds[target] = {"minor": 3, "major": 6}

        old_vp = instance.victory_points[target]
        instance.victory_points[target] = old_vp + amount
        new_vp = instance.victory_points[target]

        # Check thresholds
        thresholds = instance.thresholds.get(target, {})
        minor = thresholds.get("minor", 999)
        major = thresholds.get("major", 999)

        if old_vp < minor <= new_vp:
            notifications.append(f"Minor threshold reached for {target}! ({new_vp} VP)")
        if old_vp < major <= new_vp:
            notifications.append(f"Major threshold reached for {target}! ({new_vp} VP)")
            # Auto-complete if all targets hit major
            if self._all_targets_at_major(instance):
                instance.status = "completed"
                notifications.append("All targets at major threshold - subsystem completed!")

        return notifications

    def _action_remove_vp(self, instance: SubsystemInstance, args: dict) -> list[str]:
        target = args.get("target", "")
        amount = int(args.get("amount", 1))

        if target in instance.victory_points:
            instance.victory_points[target] = max(0, instance.victory_points[target] - amount)

        return []

    def _action_advance_round(self, instance: SubsystemInstance, args: dict) -> list[str]:
        instance.round_number += 1
        notifications = [f"Round {instance.round_number}"]

        # Auto-execute hazard routine if applicable
        if (
            instance.subsystem_type == "hazard"
            and instance.routine_actions
            and not instance.destroyed
            and not instance.disabled
        ):
            routine_count = int(args.get("routine_actions_count", 1))
            for _ in range(routine_count):
                if not instance.routine_actions:
                    break
                action_text = instance.routine_actions[instance.routine_index]
                notifications.append(f"Routine action: {action_text}")
                instance.routine_index = (instance.routine_index + 1) % len(instance.routine_actions)

        return notifications

    def _action_move(self, instance: SubsystemInstance, args: dict) -> list[str]:
        participant = args.get("participant", "")
        distance = int(args.get("distance", 1))
        notifications = []

        if participant not in instance.positions:
            instance.positions[participant] = 0

        instance.positions[participant] += distance

        # Check for chase endpoint
        if instance.chase_length > 0:
            pos = instance.positions[participant]
            if pos >= instance.chase_length:
                instance.positions[participant] = instance.chase_length
                notifications.append(f"{participant} reached the end! (position {instance.chase_length})")
            elif pos <= 0:
                instance.positions[participant] = 0
                notifications.append(f"{participant} at start position.")

            # Report gap between participants
            if len(instance.positions) >= 2:
                sorted_pos = sorted(instance.positions.items(), key=lambda x: x[1], reverse=True)
                leader = sorted_pos[0]
                trailer = sorted_pos[-1]
                gap = leader[1] - trailer[1]
                notifications.append(f"Gap: {leader[0]} ({leader[1]}) leads {trailer[0]} ({trailer[1]}) by {gap}")

        return notifications

    def _action_damage_hazard(self, instance: SubsystemInstance, args: dict) -> list[str]:
        raw_damage = int(args.get("amount", 0))
        notifications = []

        if instance.config.get("spiritual_immunity", False):
            return [
                "Haunts are immune to physical damage. "
                "Use attempt_disable (Religion/Occultism) or magical attacks that affect spirits."
            ]

        if instance.hp is None:
            return ["Hazard has no HP to track."]

        effective = max(0, raw_damage - instance.hardness)
        instance.hp = max(0, instance.hp - effective)

        notifications.append(
            f"Dealt {raw_damage} damage (hardness {instance.hardness} reduces to {effective}). "
            f"HP: {instance.hp}/{instance.max_hp}"
        )

        if instance.hp <= 0:
            instance.destroyed = True
            instance.status = "completed"
            notifications.append("Hazard destroyed!")

        return notifications

    def _action_disable_hazard(self, instance: SubsystemInstance, args: dict) -> list[str]:
        instance.disabled = True
        instance.status = "completed"
        return ["Hazard disabled!"]

    def _action_hazard_routine(self, instance: SubsystemInstance, args: dict) -> list[str]:
        notifications = []

        if not instance.routine_actions:
            return ["No routine actions defined for this hazard."]

        action_text = instance.routine_actions[instance.routine_index]
        notifications.append(f"Routine action: {action_text}")

        # Advance index, cycling back to start
        instance.routine_index = (instance.routine_index + 1) % len(instance.routine_actions)
        return notifications

    def _action_add_awareness(self, instance: SubsystemInstance, args: dict) -> list[str]:
        amount = int(args.get("amount", 1))
        notifications = []

        instance.awareness_points += amount
        notifications.append(f"Awareness: {instance.awareness_points}/{instance.detection_threshold}")

        if instance.detection_threshold > 0 and instance.awareness_points >= instance.detection_threshold:
            instance.status = "failed"
            notifications.append("Detection threshold reached - infiltration failed!")

        return notifications

    def _action_set_activity(self, instance: SubsystemInstance, args: dict) -> list[str]:
        character = args.get("character", "")
        activity = args.get("activity", "")

        if not character or not activity:
            return ["Both 'character' and 'activity' args are required."]

        activities = instance.config.get("activities", {})
        activities[character] = activity
        instance.config["activities"] = activities

        return [f"{character} is now {activity}."]

    def _action_set_marching_order(self, instance: SubsystemInstance, args: dict) -> list[str]:
        order = args.get("order", [])
        if isinstance(order, str):
            order = [o.strip() for o in order.split(",") if o.strip()]

        if not order:
            return ["Provide 'order' as a list or comma-separated string."]

        instance.config["marching_order"] = order
        return [f"Marching order set: {' > '.join(order)}"]

    def _action_attempt_disable(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Attempt to fulfill one disable condition via skill check."""
        condition_index = args.get("condition_index")
        if condition_index is None:
            return ["'condition_index' is required."]
        condition_index = int(condition_index)

        if not instance.disable_conditions:
            return ["No disable conditions defined for this hazard/haunt."]

        if condition_index < 0 or condition_index >= len(instance.disable_conditions):
            return [
                f"Invalid condition_index {condition_index}. "
                f"Valid range: 0-{len(instance.disable_conditions) - 1}"
            ]

        condition = instance.disable_conditions[condition_index]
        dc = condition.get("dc", 20)
        successes_needed = condition.get("successes_needed", 1)
        skill = args.get("skill") or condition.get("skill", "Unknown")
        notifications: list[str] = []

        if "roll_total" in args:
            total = int(args["roll_total"])
            natural_roll = None
            notifications.append(f"{skill} check (total {total}) vs DC {dc}")
        elif "modifier" in args:
            modifier = int(args["modifier"])
            d20_result = roll_dice("1d20")
            natural_roll = d20_result.rolls[0]
            total = natural_roll + modifier
            notifications.append(f"{skill} check: d20({natural_roll}) + {modifier} = {total} vs DC {dc}")
        else:
            return ["Provide either 'roll_total' (manual) or 'modifier' (auto-roll)."]

        result = check_result(total, dc)
        degree = result["degree"]

        # Apply nat 20/1 adjustments
        if natural_roll is not None:
            if natural_roll == 20 and degree in ("success", "failure"):
                degree = "critical_success" if degree == "success" else "success"
                notifications.append("Natural 20 — degree upgraded!")
            elif natural_roll == 1 and degree in ("success", "failure"):
                degree = "failure" if degree == "success" else "critical_failure"
                notifications.append("Natural 1 — degree downgraded!")

        notifications.append(f"Result: {degree.replace('_', ' ').title()}")

        current = instance.disable_conditions_progress.get(str(condition_index), 0)

        if degree == "critical_success":
            instance.disable_conditions_progress[str(condition_index)] = successes_needed
            notifications.append(f"Condition [{condition_index}] instantly completed!")
        elif degree == "success":
            instance.disable_conditions_progress[str(condition_index)] = current + 1
            notifications.append(f"Progress: {current + 1}/{successes_needed}")
        elif degree == "failure":
            notifications.append(f"No progress. ({current}/{successes_needed})")
        elif degree == "critical_failure":
            notifications.append(f"Critical Failure! No progress. ({current}/{successes_needed})")
            if instance.subsystem_type == "haunt":
                notifications.append("The haunt triggers its effect!")

        # Check if all conditions are fully progressed
        all_done = all(
            instance.disable_conditions_progress.get(str(i), 0)
            >= instance.disable_conditions[i].get("successes_needed", 1)
            for i in range(len(instance.disable_conditions))
        )
        if all_done:
            instance.disabled = True
            instance.status = "completed"
            notifications.append("All disable conditions met — hazard/haunt disabled!")

        return notifications

    def _action_trigger_haunt(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Fire the haunt's configured effect."""
        instance.config["triggered"] = True
        trigger_desc = instance.config.get("trigger_condition", "The haunt activates.")
        return [f"Haunt triggered: {trigger_desc}"]

    def _action_reset_haunt(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Reset haunt state after its reset interval."""
        reset_interval = instance.config.get("reset_interval", "1 day")
        if reset_interval == "never":
            return ["This haunt never resets (reset_interval: 'never')."]

        instance.config.pop("triggered", None)
        instance.disable_conditions_progress = {
            str(i): 0 for i in range(len(instance.disable_conditions))
        }
        if instance.max_hp is not None:
            instance.hp = instance.max_hp
        instance.destroyed = False
        instance.disabled = False
        instance.status = "active"

        return [f"Haunt reset. Will trigger again after {reset_interval}."]

    def _action_attempt_influence(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Attempt to influence an NPC via skill check."""
        npc = args.get("npc", "")
        approach = args.get("approach", "")
        situation = args.get("situation", "")
        notifications: list[str] = []

        if not npc:
            return ["'npc' is required."]

        npcs_cfg = instance.config.get("npcs", {})
        npc_cfg = npcs_cfg.get(npc, {})

        # Get DC from approach_dcs; default 20
        approach_dcs = npc_cfg.get("approach_dcs", {})
        dc = approach_dcs.get(approach, 20)

        # Apply weaknesses (−2) and resistances (+2)
        if situation:
            for weakness in npc_cfg.get("weaknesses", []):
                if weakness.lower() in situation.lower():
                    dc -= 2
                    notifications.append(f"Weakness '{weakness}' matched — DC reduced by 2 (now {dc})")
                    break
            for resistance in npc_cfg.get("resistances", []):
                if resistance.lower() in situation.lower():
                    dc += 2
                    notifications.append(f"Resistance '{resistance}' matched — DC increased by 2 (now {dc})")
                    break

        # Roll or use total
        if "roll_total" in args:
            total = int(args["roll_total"])
            natural_roll = None
        elif "modifier" in args:
            modifier = int(args["modifier"])
            d20_result = roll_dice("1d20")
            natural_roll = d20_result.rolls[0]
            total = natural_roll + modifier
            notifications.append(f"{approach} check: d20({natural_roll}) + {modifier} = {total} vs DC {dc}")
        else:
            return ["Provide either 'roll_total' or 'modifier'."]

        if "roll_total" in args and natural_roll is None:
            notifications.append(f"{approach} check (total {total}) vs DC {dc}")

        result = check_result(total, dc)
        degree = result["degree"]

        # Nat 20/1 adjustments
        if natural_roll is not None:
            if natural_roll == 20 and degree in ("success", "failure"):
                degree = "critical_success" if degree == "success" else "success"
                notifications.append("Natural 20 — degree upgraded!")
            elif natural_roll == 1 and degree in ("success", "failure"):
                degree = "failure" if degree == "success" else "critical_failure"
                notifications.append("Natural 1 — degree downgraded!")

        # VP delta
        vp_delta_map = {"critical_success": 2, "success": 1, "failure": 0, "critical_failure": -1}
        vp_delta = vp_delta_map.get(degree, 0)

        min_vp = npc_cfg.get("min_vp", 0)
        max_vp = npc_cfg.get("max_vp", 5)

        if npc not in instance.victory_points:
            instance.victory_points[npc] = min_vp
            instance.thresholds[npc] = {"minor": min_vp + 2, "major": max_vp}

        old_vp = instance.victory_points[npc]
        new_vp = max(min_vp, min(max_vp, old_vp + vp_delta))
        instance.victory_points[npc] = new_vp

        notifications.append(
            f"{npc}: {approach} check {total} vs DC {dc} → {degree.replace('_', ' ')} "
            f"(+{vp_delta} VP, now {new_vp}/{max_vp})"
        )

        # Check all NPCs at max_vp
        if npcs_cfg and all(
            instance.victory_points.get(n, 0) >= n_cfg.get("max_vp", 5)
            for n, n_cfg in npcs_cfg.items()
        ):
            instance.status = "completed"
            notifications.append("All NPCs at maximum VP — influence subsystem completed!")

        return notifications

    def _action_attempt_obstacle(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Attempt to pass a chase obstacle via skill check."""
        participant = args.get("participant", "")
        obstacle_index = args.get("obstacle_index")
        skill = args.get("skill", "")
        notifications: list[str] = []

        if obstacle_index is None:
            return ["'obstacle_index' is required."]
        obstacle_index = int(obstacle_index)

        # Prefer config obstacles, fall back to instance.obstacles
        obstacles = instance.config.get("obstacles") or instance.obstacles
        if not obstacles:
            return ["No obstacles defined for this chase."]

        if obstacle_index < 0 or obstacle_index >= len(obstacles):
            return [f"Invalid obstacle_index {obstacle_index}. Valid range: 0–{len(obstacles) - 1}"]

        obstacle = obstacles[obstacle_index]
        obstacle_name = obstacle.get("name", f"Obstacle {obstacle_index}")

        # Determine DC: alt skill / alt DC
        alt_skill = obstacle.get("alternative_skill", "")
        if alt_skill and skill.lower() == alt_skill.lower():
            dc = obstacle.get("alternative_dc", obstacle.get("dc", 20))
        else:
            dc = obstacle.get("dc", 20)

        # Roll or use total
        if "roll_total" in args:
            total = int(args["roll_total"])
            natural_roll = None
            notifications.append(f"{skill} check (total {total}) vs DC {dc} [{obstacle_name}]")
        elif "modifier" in args:
            modifier = int(args["modifier"])
            d20_result = roll_dice("1d20")
            natural_roll = d20_result.rolls[0]
            total = natural_roll + modifier
            notifications.append(
                f"{skill} check: d20({natural_roll}) + {modifier} = {total} vs DC {dc} [{obstacle_name}]"
            )
        else:
            return ["Provide either 'roll_total' or 'modifier'."]

        result = check_result(total, dc)
        degree = result["degree"]

        # Nat 20/1 adjustments
        if natural_roll is not None:
            if natural_roll == 20 and degree in ("success", "failure"):
                degree = "critical_success" if degree == "success" else "success"
                notifications.append("Natural 20 — degree upgraded!")
            elif natural_roll == 1 and degree in ("success", "failure"):
                degree = "failure" if degree == "success" else "critical_failure"
                notifications.append("Natural 1 — degree downgraded!")

        success_advances = obstacle.get("success_advances", 1)
        cf_retreats = obstacle.get("cf_retreats", 1)
        pos_delta_map = {
            "critical_success": success_advances * 2,
            "success": success_advances,
            "failure": 0,
            "critical_failure": -cf_retreats,
        }
        pos_delta = pos_delta_map.get(degree, 0)

        notifications.append(f"Result: {degree.replace('_', ' ').title()}")

        # Update position
        if participant not in instance.positions:
            instance.positions[participant] = 0
        old_pos = instance.positions[participant]
        chase_max = instance.chase_length if instance.chase_length > 0 else 9999
        new_pos = max(0, min(chase_max, old_pos + pos_delta))
        instance.positions[participant] = new_pos

        notifications.append(f"{participant} [{obstacle_name}]: {old_pos} → {new_pos}")

        # Chase end condition
        if instance.chase_length > 0 and new_pos >= instance.chase_length:
            instance.status = "completed"
            notifications.append(f"{participant} completed the chase! (position {instance.chase_length})")

        # Gap between participants
        if len(instance.positions) >= 2:
            sorted_pos = sorted(instance.positions.items(), key=lambda x: x[1], reverse=True)
            leader = sorted_pos[0]
            trailer = sorted_pos[-1]
            gap = leader[1] - trailer[1]
            notifications.append(
                f"Gap: {leader[0]} ({leader[1]}) leads {trailer[0]} ({trailer[1]}) by {gap}"
            )

        return notifications

    def _action_research_check(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Consult a research source via skill check."""
        source_index = args.get("source_index")
        skill = args.get("skill", "")
        notifications: list[str] = []

        if source_index is None:
            return ["'source_index' is required."]
        source_index = int(source_index)

        sources = instance.config.get("sources", [])
        if not sources:
            return ["No sources defined for this research subsystem."]

        if source_index < 0 or source_index >= len(sources):
            return [f"Invalid source_index {source_index}. Valid range: 0–{len(sources) - 1}"]

        source = sources[source_index]
        source_name = source.get("name", f"Source {source_index}")
        dc = source.get("dc", 20)
        per_success = source.get("per_success", 1)
        max_contribution = source.get("max_contribution", 9999)
        discovery_note = source.get("discovery_note", "")

        # Check source cap
        source_contributions = instance.config.setdefault("source_contributions", {})
        current_contribution = source_contributions.get(str(source_index), 0)

        if current_contribution >= max_contribution:
            return [f"Source '{source_name}' has reached its max contribution ({max_contribution} VP)."]

        # Roll or use total
        if "roll_total" in args:
            total = int(args["roll_total"])
            natural_roll = None
            notifications.append(f"{skill} check (total {total}) vs DC {dc} [{source_name}]")
        elif "modifier" in args:
            modifier = int(args["modifier"])
            d20_result = roll_dice("1d20")
            natural_roll = d20_result.rolls[0]
            total = natural_roll + modifier
            notifications.append(
                f"{skill} check: d20({natural_roll}) + {modifier} = {total} vs DC {dc} [{source_name}]"
            )
        else:
            return ["Provide either 'roll_total' or 'modifier'."]

        result = check_result(total, dc)
        degree = result["degree"]

        # Nat 20/1 adjustments
        if natural_roll is not None:
            if natural_roll == 20 and degree in ("success", "failure"):
                degree = "critical_success" if degree == "success" else "success"
                notifications.append("Natural 20 — degree upgraded!")
            elif natural_roll == 1 and degree in ("success", "failure"):
                degree = "failure" if degree == "success" else "critical_failure"
                notifications.append("Natural 1 — degree downgraded!")

        vp_gain_map = {"critical_success": per_success * 2, "success": per_success, "failure": 0, "critical_failure": 0}
        vp_gain = vp_gain_map.get(degree, 0)

        notifications.append(f"Result: {degree.replace('_', ' ').title()}")

        if vp_gain > 0:
            # Clamp to remaining cap
            available = max_contribution - current_contribution
            vp_gain = min(vp_gain, available)
            source_contributions[str(source_index)] = current_contribution + vp_gain

            # Add to single VP pool — use first target or instance name
            if instance.victory_points:
                target = next(iter(instance.victory_points))
            else:
                target = instance.name
                instance.victory_points[target] = 0
                if target not in instance.thresholds:
                    instance.thresholds[target] = {"minor": 5, "major": 10}

            instance.victory_points[target] += vp_gain
            new_vp = instance.victory_points[target]
            notifications.append(f"+{vp_gain} VP from {source_name} ({skill}). Total: {new_vp}")

            if degree == "critical_success" and discovery_note:
                notifications.append(f"Discovery: {discovery_note}")

            # Check vp_target
            vp_target = instance.config.get("vp_target", 0)
            if vp_target > 0 and new_vp >= vp_target:
                instance.status = "completed"
                notifications.append(f"Research complete! ({new_vp}/{vp_target} VP)")

        return notifications

    def _action_earn_edge(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Earn infiltration edge points."""
        amount = int(args.get("amount", 1))
        action_desc = args.get("action", "")

        instance.edge_points += amount
        if action_desc:
            return [f"Earned {amount} edge (via {action_desc}). Edge points: {instance.edge_points}"]
        return [f"Edge points: {instance.edge_points}"]

    def _action_spend_edge(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Spend infiltration edge points on a benefit."""
        benefit_index = args.get("benefit_index")
        benefit_desc = args.get("benefit", "")
        amount = int(args.get("amount", 1))

        if benefit_index is not None:
            benefit_index = int(benefit_index)
            edge_benefits = instance.config.get("edge_benefits", [])
            if benefit_index < 0 or benefit_index >= len(edge_benefits):
                return [f"Invalid benefit_index {benefit_index}. Valid range: 0–{len(edge_benefits) - 1}"]
            benefit = edge_benefits[benefit_index]
            amount = benefit.get("cost", amount)
            benefit_desc = benefit.get("description", f"benefit {benefit_index}")

        if instance.edge_points < amount:
            return [
                f"Insufficient edge points. Have {instance.edge_points}, need {amount}."
            ]

        instance.edge_points = max(0, instance.edge_points - amount)
        return [
            f"Spent {amount} edge on '{benefit_desc}'. Edge points remaining: {instance.edge_points}"
        ]

    def _action_hazard_attack(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Resolve a hazard's attack roll against a target."""
        attack_bonus = int(args.get("attack_bonus", 0))
        target_ac = int(args.get("target_ac", 15))
        notifications: list[str] = []

        # Roll or use total
        if "roll_total" in args:
            total = int(args["roll_total"])
            natural_roll = None
            notifications.append(f"Attack: total {total} vs AC {target_ac}")
        else:
            d20_result = roll_dice("1d20")
            natural_roll = d20_result.rolls[0]
            total = natural_roll + attack_bonus
            notifications.append(f"Attack: d20({natural_roll}) + {attack_bonus} = {total} vs AC {target_ac}")

        # Degree based on AC thresholds
        if total >= target_ac + 10:
            degree = "critical_hit"
        elif total >= target_ac:
            degree = "hit"
        elif total <= target_ac - 10:
            degree = "critical_miss"
        else:
            degree = "miss"

        # Nat 20/1 adjustments
        if natural_roll is not None:
            if natural_roll == 20:
                if degree in ("hit", "miss"):
                    degree = "critical_hit" if degree == "hit" else "hit"
                    notifications.append("Natural 20 — degree upgraded!")
            elif natural_roll == 1:
                if degree in ("hit", "miss"):
                    degree = "miss" if degree == "hit" else "critical_miss"
                    notifications.append("Natural 1 — degree downgraded!")

        notifications.append(f"Result: {degree.replace('_', ' ').title()}")

        if degree in ("hit", "critical_hit"):
            notifications.append("Hit! Apply damage from the hazard's stat block.")

        return notifications

    def _action_hazard_save(self, instance: SubsystemInstance, args: dict) -> list[str]:
        """Resolve a saving throw against a hazard or haunt."""
        save_dc = int(args.get("save_dc", 20))
        save_type = args.get("save_type", "Reflex")
        save_modifier = int(args.get("save_modifier", 0))
        notifications: list[str] = []

        # Roll or use total
        if "roll_total" in args:
            total = int(args["roll_total"])
            natural_roll = None
            save_desc = f"total {total}"
        else:
            d20_result = roll_dice("1d20")
            natural_roll = d20_result.rolls[0]
            total = natural_roll + save_modifier
            save_desc = f"d20({natural_roll}) + {save_modifier} = {total}"

        result = check_result(total, save_dc)
        degree = result["degree"]

        # Nat 20/1 adjustments
        if natural_roll is not None:
            if natural_roll == 20 and degree in ("success", "failure"):
                degree = "critical_success" if degree == "success" else "success"
                notifications.append("Natural 20 — degree upgraded!")
            elif natural_roll == 1 and degree in ("success", "failure"):
                degree = "failure" if degree == "success" else "critical_failure"
                notifications.append("Natural 1 — degree downgraded!")

        notifications.append(
            f"{save_type} save: {save_desc} vs DC {save_dc} → {degree.replace('_', ' ').title()}"
        )
        return notifications

    def _all_targets_at_major(self, instance: SubsystemInstance) -> bool:
        """Check if all VP targets have reached their major threshold."""
        for target, vp in instance.victory_points.items():
            thresholds = instance.thresholds.get(target, {})
            major = thresholds.get("major", 999)
            if vp < major:
                return False
        return True

    # -------------------------------------------------------------------
    # Tool: get_subsystem_state
    # -------------------------------------------------------------------

    def _get_subsystem_state(self, subsystem_id: str, include_log: bool) -> ToolResult:
        instance = self.store.get(subsystem_id)
        if instance is None:
            return ToolResult(success=False, error=f"Subsystem '{subsystem_id}' not found.")

        return ToolResult(
            success=True,
            data=self._format_state(instance, include_log=include_log),
        )

    # -------------------------------------------------------------------
    # Tool: end_subsystem
    # -------------------------------------------------------------------

    def _end_subsystem(self, subsystem_id: str, outcome: str) -> ToolResult:
        instance = self.store.get(subsystem_id)
        if instance is None:
            return ToolResult(success=False, error=f"Subsystem '{subsystem_id}' not found.")

        valid_outcomes = ("completed", "failed", "abandoned")
        if outcome not in valid_outcomes:
            return ToolResult(
                success=False,
                error=f"Invalid outcome '{outcome}'. Must be: {', '.join(valid_outcomes)}",
            )

        instance.status = outcome
        instance.action_log.append({
            "action": "end",
            "outcome": outcome,
            "round": instance.round_number,
            "timestamp": datetime.now().isoformat(),
        })
        self.store.save(instance)

        summary = self._format_summary(instance)
        return ToolResult(success=True, data=summary)

    # -------------------------------------------------------------------
    # Tool: list_subsystems
    # -------------------------------------------------------------------

    def _list_subsystems(self) -> ToolResult:
        active = self.store.list_active()
        if not active:
            return ToolResult(success=True, data="No active subsystems.")

        lines = [f"**Active Subsystems** ({len(active)})\n"]
        for inst in active:
            lines.append(
                f"- **{inst.name}** (ID: {inst.id}) — {inst.subsystem_type}, round {inst.round_number}"
            )
        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------

    def _format_state(self, instance: SubsystemInstance, include_log: bool = False) -> str:
        """Format subsystem state as human-readable text."""
        st = instance.subsystem_type
        lines = [
            f"**{instance.name}** ({st}) — Round {instance.round_number} [{instance.status}]",
        ]

        # VP tracking
        if instance.victory_points:
            lines.append("\n**Victory Points:**")
            for target, vp in instance.victory_points.items():
                thresholds = instance.thresholds.get(target, {})
                minor = thresholds.get("minor", "?")
                major = thresholds.get("major", "?")
                lines.append(f"  {target}: {vp} VP (minor: {minor}, major: {major})")

        # Chase positions
        if instance.positions:
            lines.append("\n**Positions:**")
            for participant, pos in sorted(instance.positions.items(), key=lambda x: -x[1]):
                lines.append(f"  {participant}: {pos}" + (f"/{instance.chase_length}" if instance.chase_length else ""))

        # Hazard/Haunt HP
        if instance.hp is not None:
            hp_status = ""
            if instance.destroyed:
                hp_status = " [DESTROYED]"
            elif instance.disabled:
                hp_status = " [DISABLED]"
            lines.append(f"\n**HP** {instance.hp}/{instance.max_hp} (Hardness {instance.hardness}){hp_status}")
            if instance.routine_actions:
                lines.append(f"**Routine** ({len(instance.routine_actions)} actions, next: #{instance.routine_index + 1})")

        # Haunt-specific fields
        if st == "haunt":
            trigger = instance.config.get("trigger_condition", "Unknown")
            anchor = instance.config.get("anchor", "")
            reset_int = instance.config.get("reset_interval", "1 day")
            spiritual = instance.config.get("spiritual_immunity", True)
            triggered = instance.config.get("triggered", False)

            lines.append(f"\n**Trigger:** {trigger}")
            if anchor:
                lines.append(f"**Anchor:** {anchor}")
            lines.append(f"**Reset Interval:** {reset_int}")
            if spiritual:
                lines.append("**Spiritual Immunity:** Yes (physical damage ineffective)")
            if triggered:
                lines.append("**Currently Triggered**")

        # Disable conditions (hazard and haunt)
        if instance.disable_conditions and st in ("hazard", "haunt"):
            lines.append("\n**Disable Conditions:**")
            for i, cond in enumerate(instance.disable_conditions):
                progress = instance.disable_conditions_progress.get(str(i), 0)
                needed = cond.get("successes_needed", 1)
                skill = cond.get("skill", "Unknown")
                dc = cond.get("dc", "?")
                if needed > 1:
                    lines.append(f"  [{i}] {skill} DC {dc}: {progress}/{needed} successes")
                else:
                    lines.append(f"  [{i}] {skill} DC {dc}")

        # Infiltration awareness + edge
        if st == "infiltration":
            pct = 0
            if instance.detection_threshold > 0:
                pct = int(100 * instance.awareness_points / instance.detection_threshold)
            lines.append(f"\n**Awareness** {instance.awareness_points}/{instance.detection_threshold} ({pct}%)")
            lines.append(f"**Edge Points** {instance.edge_points}")

        # Chase obstacle progress
        if st == "chase" and instance.obstacles:
            total_obs = len(instance.obstacles)
            lines.append(f"\n**Obstacles:** {total_obs} total")
            for i, obs in enumerate(instance.obstacles[:5]):  # Show first 5
                obs_name = obs.get("name", f"Obstacle {i}")
                obs_dc = obs.get("dc", "?")
                lines.append(f"  [{i}] {obs_name} (DC {obs_dc})")

        # Research source contributions
        if st == "research":
            source_contributions = instance.config.get("source_contributions", {})
            sources = instance.config.get("sources", [])
            vp_target = instance.config.get("vp_target", 0)
            if vp_target:
                lines.append(f"\n**VP Target:** {vp_target}")
            if sources and source_contributions:
                lines.append("**Source Contributions:**")
                for i, src in enumerate(sources):
                    contrib = source_contributions.get(str(i), 0)
                    max_c = src.get("max_contribution", "∞")
                    lines.append(f"  [{i}] {src.get('name', f'Source {i}')}: {contrib}/{max_c}")

        # Exploration activities
        if st == "exploration":
            activities = instance.config.get("activities", {})
            marching_order = instance.config.get("marching_order", [])
            if activities:
                lines.append("\n**Activities:**")
                for char, act in activities.items():
                    lines.append(f"  {char}: {act}")
            if marching_order:
                lines.append(f"\n**Marching Order:** {' > '.join(marching_order)}")

        # Action log
        if include_log and instance.action_log:
            lines.append(f"\n**Action Log** ({len(instance.action_log)} entries)")
            for entry in instance.action_log[-20:]:  # Last 20
                lines.append(f"  R{entry.get('round', '?')}: {entry.get('action', '?')} {entry.get('args', '')}")

        return "\n".join(lines)

    def _format_summary(self, instance: SubsystemInstance) -> str:
        """Format a completion summary."""
        lines = [
            f"**Subsystem Ended:** {instance.name}",
            f"**Outcome:** {instance.status}",
            f"**Rounds:** {instance.round_number}",
            f"**Actions Taken:** {len(instance.action_log)}",
        ]

        if instance.victory_points:
            lines.append("\n**Final VP:**")
            for target, vp in instance.victory_points.items():
                lines.append(f"  {target}: {vp}")

        if instance.positions:
            lines.append("\n**Final Positions:**")
            for p, pos in sorted(instance.positions.items(), key=lambda x: -x[1]):
                lines.append(f"  {p}: {pos}")

        if instance.hp is not None:
            lines.append(f"\n**Final HP:** {instance.hp}/{instance.max_hp}")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Close resources."""
        if self._store:
            self._store.close()
            self._store = None
