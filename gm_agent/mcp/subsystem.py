"""Subsystem MCP server for stateful encounter subsystems.

Provides tools for running Victory Point subsystems, Chases, Influence,
Research, Infiltration, and complex Hazards with state tracking.
"""

import json
from datetime import datetime
from typing import Any

from ..config import CAMPAIGNS_DIR
from ..storage.subsystems import SubsystemStore, SubsystemInstance
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


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
                    "influence, research, chase, infiltration, and hazard types. "
                    "Config varies by type: VP/influence/research need targets+thresholds, "
                    "chase needs participants+length, hazard needs HP/routine/disable conditions, "
                    "infiltration needs detection_threshold."
                ),
                parameters=[
                    ToolParameter(
                        name="type",
                        type="string",
                        description="Subsystem type: 'vp', 'influence', 'research', 'chase', 'infiltration', 'hazard'",
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
                            "Chase: {\"participants\": [\"Party\", \"Thief\"], \"chase_length\": 10}\n"
                            "Hazard: {\"hp\": 50, \"hardness\": 10, \"routine_actions\": [\"Fires darts\", \"Resets\"], \"disable_conditions\": [{\"skill\": \"Thievery\", \"dc\": 22}]}\n"
                            "Infiltration: {\"detection_threshold\": 10, \"targets\": {\"Main Objective\": {\"minor\": 4, \"major\": 8}}}"
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
                    "'add_awareness' (amount, for infiltration). "
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
                        description="Action type: 'add_vp', 'remove_vp', 'advance_round', 'move', 'damage_hazard', 'disable_hazard', 'hazard_routine', 'add_awareness'",
                    ),
                    ToolParameter(
                        name="args",
                        type="string",
                        description="JSON args for the action (e.g., {\"target\": \"Lord Gyr\", \"amount\": 1})",
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
        valid_types = ("vp", "influence", "research", "chase", "infiltration", "hazard", "exploration")
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
        hazard_only_actions = ("damage_hazard", "disable_hazard", "hazard_routine")
        if action in hazard_only_actions:
            if instance.subsystem_type != "hazard":
                return ToolResult(
                    success=False,
                    error=f"Action '{action}' is only valid for hazard subsystems, "
                    f"not {instance.subsystem_type}.",
                )
            if instance.destroyed:
                return ToolResult(success=False, error="Hazard is already destroyed.")
            if instance.disabled:
                return ToolResult(success=False, error="Hazard is already disabled.")

        if action == "move" and instance.subsystem_type != "chase":
            return ToolResult(
                success=False,
                error=f"Action 'move' is only valid for chase subsystems, "
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
        else:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Valid: add_vp, remove_vp, advance_round, move, damage_hazard, disable_hazard, hazard_routine, add_awareness",
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

        # Hazard HP
        if instance.hp is not None:
            status = ""
            if instance.destroyed:
                status = " [DESTROYED]"
            elif instance.disabled:
                status = " [DISABLED]"
            lines.append(f"\n**HP** {instance.hp}/{instance.max_hp} (Hardness {instance.hardness}){status}")
            if instance.routine_actions:
                lines.append(f"**Routine** ({len(instance.routine_actions)} actions, next: #{instance.routine_index + 1})")

        # Infiltration awareness
        if st == "infiltration":
            pct = 0
            if instance.detection_threshold > 0:
                pct = int(100 * instance.awareness_points / instance.detection_threshold)
            lines.append(f"\n**Awareness** {instance.awareness_points}/{instance.detection_threshold} ({pct}%)")

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
