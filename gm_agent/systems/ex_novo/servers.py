"""Ex Novo settlement MCP server.

Provides tools for running an Ex Novo settlement-building game,
managing settlement state and writing to the Fiction Tree.
"""

import json
from typing import Any

from ...config import CAMPAIGNS_DIR
from ...storage.fiction_tree import FictionTreeStore, FictionNode
from ...mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult
from .tables import (
    AGE_TABLE, SIZE_TABLE, INFLUENCE_TABLE,
    GEOGRAPHY_TABLE, FEATURES_TABLE,
    LOCATION_TABLE, DECISION_TABLE,
    PARADIGM_TABLE, RELATIONSHIP_TABLE,
    EVENT_TABLE,
    roll_d6, roll_2d6, roll_d666,
)


class SettlementServer(MCPServer):
    """MCP server for Ex Novo settlement building.

    Campaign-scoped, stateful server managing settlement generation
    and writing to the Fiction Tree.
    """

    def __init__(self, campaign_id: str, llm: Any = None):
        self.campaign_id = campaign_id
        self.llm = llm
        self._fiction_store: FictionTreeStore | None = None
        self._state: dict[str, Any] | None = None
        self._state_path = CAMPAIGNS_DIR / campaign_id / "ex_novo_state.json"
        self._tools = self._build_tools()

    @property
    def fiction_store(self) -> FictionTreeStore:
        if self._fiction_store is None:
            self._fiction_store = FictionTreeStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._fiction_store

    @property
    def state(self) -> dict[str, Any] | None:
        if self._state is None:
            self._state = self._load_state()
        return self._state

    def _load_state(self) -> dict[str, Any] | None:
        if self._state_path.exists():
            with open(self._state_path) as f:
                return json.load(f)
        return None

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state = state
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _build_tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="ex_novo_setup",
                description=(
                    "Initialize an Ex Novo settlement: roll scale, terrain, purpose, "
                    "power tables. Create founding factions and initial districts."
                ),
                parameters=[
                    ToolParameter(name="name", type="string", description="Settlement name."),
                    ToolParameter(name="num_factions", type="integer",
                                  description="Number of founding factions (1-4).",
                                  required=False, default=2),
                ],
            ),
            ToolDef(
                name="add_faction",
                description="Create a new faction in the settlement.",
                parameters=[
                    ToolParameter(name="name", type="string", description="Faction name."),
                    ToolParameter(name="description", type="string",
                                  description="Faction description.", required=False, default=""),
                    ToolParameter(name="power_tokens", type="integer",
                                  description="Initial power tokens.", required=False, default=1),
                ],
            ),
            ToolDef(
                name="add_district",
                description="Create a new district in the settlement.",
                parameters=[
                    ToolParameter(name="name", type="string", description="District name."),
                    ToolParameter(name="citizen_tokens", type="integer",
                                  description="Initial citizen tokens.", required=False, default=1),
                ],
            ),
            ToolDef(
                name="add_landmark",
                description="Add a named landmark to a district.",
                parameters=[
                    ToolParameter(name="name", type="string", description="Landmark name."),
                    ToolParameter(name="district", type="string",
                                  description="District to place the landmark in.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="roll_event",
                description="Roll D666 on the events table. Returns the narrative prompt and mechanical effect.",
                parameters=[],
            ),
            ToolDef(
                name="resolve_event",
                description=(
                    "Apply an event outcome — modify tokens, add/remove factions, "
                    "landmarks, resources, or problems."
                ),
                parameters=[
                    ToolParameter(name="event_code", type="integer",
                                  description="The D666 event code."),
                    ToolParameter(name="narrative", type="string",
                                  description="How this event played out narratively."),
                    ToolParameter(name="effect_target", type="string",
                                  description="Name of the affected entity (faction, district, etc.).",
                                  required=False, default=""),
                ],
            ),
            ToolDef(
                name="advance_phase",
                description="Move from founding to development to topping out.",
                parameters=[],
            ),
            ToolDef(
                name="get_settlement_state",
                description="Get full settlement overview: factions, districts, resources, problems.",
                parameters=[],
            ),
            ToolDef(
                name="browse_settlement_history",
                description="View the settlement's fiction tree history.",
                parameters=[],
            ),
            ToolDef(
                name="end_settlement",
                description="Finalize the settlement.",
                parameters=[],
            ),
            # --- Virtual Players ---
            ToolDef(
                name="add_virtual_player",
                description="Register an AI-driven virtual player for Ex Novo with a personality archetype or custom traits.",
                parameters=[
                    ToolParameter(name="name", type="string",
                                  description="Virtual player name."),
                    ToolParameter(name="archetype", type="string",
                                  description="Personality archetype name (e.g. 'the_sage', 'the_trickster'). "
                                  "Leave empty for custom traits.",
                                  required=False, default=""),
                    ToolParameter(name="traits", type="string",
                                  description="JSON object of custom trait weights (e.g. '{\"openness\": 0.9}').",
                                  required=False, default="{}"),
                ],
            ),
            ToolDef(
                name="virtual_player_turn",
                description="Generate and execute a virtual player's turn in Ex Novo using AI.",
                parameters=[
                    ToolParameter(name="player_name", type="string",
                                  description="Name of the virtual player to act."),
                    ToolParameter(name="decision_type", type="string",
                                  description="Type of decision: 'roll_event', 'add_faction', 'add_district', etc.",
                                  required=False, default="create"),
                ],
            ),
            ToolDef(
                name="remove_virtual_player",
                description="Remove a virtual player from the Ex Novo session.",
                parameters=[
                    ToolParameter(name="player_name", type="string",
                                  description="Name of the virtual player to remove."),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "ex_novo_setup":
                return self._setup(args)
            elif name == "add_faction":
                return self._add_faction(args)
            elif name == "add_district":
                return self._add_district(args)
            elif name == "add_landmark":
                return self._add_landmark(args)
            elif name == "roll_event":
                return self._roll_event()
            elif name == "resolve_event":
                return self._resolve_event(args)
            elif name == "advance_phase":
                return self._advance_phase()
            elif name == "get_settlement_state":
                return self._get_state()
            elif name == "browse_settlement_history":
                return self._browse_history()
            elif name == "end_settlement":
                return self._end()
            # Virtual Players
            elif name == "add_virtual_player":
                return self._add_virtual_player(args)
            elif name == "virtual_player_turn":
                return self._virtual_player_turn(args)
            elif name == "remove_virtual_player":
                return self._remove_virtual_player(args)
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _setup(self, args: dict[str, Any]) -> ToolResult:
        if self.state and self.state.get("phase") != "ended":
            return ToolResult(success=False, error="A settlement is already active.")

        # Roll scale tables
        age_roll = roll_d6()
        size_roll = roll_d6()
        influence_roll = roll_d6()

        # Roll terrain
        geography_roll = roll_2d6()
        features_roll = roll_2d6()

        # Roll purpose
        location_roll = roll_2d6()
        decision_roll = roll_2d6()

        # Roll power
        paradigm_roll = roll_2d6()
        relationship_roll = roll_2d6()

        # Create root fiction node
        root = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            title=args["name"],
            zoom_level="era",
            source_system="ex_novo",
            summary=f"Settlement: {args['name']}",
        ))

        def _t(table: dict, roll: int) -> str:
            """Look up a roll, falling back gracefully if table is incomplete."""
            if roll in table:
                return table[roll]
            # Find nearest key for partial tables
            keys = sorted(table.keys())
            return table[keys[-1]] if keys else f"Roll {roll}"

        # Create founding event node
        self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=root.id,
            title="Founding",
            zoom_level="period",
            tone="light",
            source_system="ex_novo",
            content=(
                f"Age: {_t(AGE_TABLE, age_roll)}\n"
                f"Size: {_t(SIZE_TABLE, size_roll)}\n"
                f"Influence: {_t(INFLUENCE_TABLE, influence_roll)}\n"
                f"Geography: {_t(GEOGRAPHY_TABLE, geography_roll)}\n"
                f"Features: {_t(FEATURES_TABLE, features_roll)}\n"
                f"Location Purpose: {_t(LOCATION_TABLE, location_roll)}\n"
                f"Decision: {_t(DECISION_TABLE, decision_roll)}\n"
                f"Paradigm: {_t(PARADIGM_TABLE, paradigm_roll)}\n"
                f"Relationship: {_t(RELATIONSHIP_TABLE, relationship_roll)}"
            ),
            sort_order=0,
        ))

        state = {
            "name": args["name"],
            "phase": "founding",
            "scale": {
                "age": _t(AGE_TABLE, age_roll),
                "size": _t(SIZE_TABLE, size_roll),
                "influence": _t(INFLUENCE_TABLE, influence_roll),
            },
            "terrain": {
                "geography": _t(GEOGRAPHY_TABLE, geography_roll),
                "features": [_t(FEATURES_TABLE, features_roll)],
            },
            "purpose": {
                "location": _t(LOCATION_TABLE, location_roll),
                "decision": _t(DECISION_TABLE, decision_roll),
            },
            "power": {
                "paradigm": _t(PARADIGM_TABLE, paradigm_roll),
                "relationship": _t(RELATIONSHIP_TABLE, relationship_roll),
            },
            "factions": [],
            "districts": [],
            "resources": [],
            "problems": [],
            "landmarks": [],
            "growth_pool": 0,
            "turn_number": 0,
            "history_root_id": root.id,
        }
        self._save_state(state)

        lines = [
            f"**{args['name']} — Settlement Created**\n",
            f"**Scale:** {state['scale']['age']} | {state['scale']['size']} | {state['scale']['influence']}",
            f"**Terrain:** {state['terrain']['geography']} + {state['terrain']['features'][0]}",
            f"**Purpose:** {state['purpose']['location']} — {state['purpose']['decision']}",
            f"**Power:** {state['power']['paradigm']} — {state['power']['relationship']}",
            f"\nNext: Add factions and districts, then begin rolling events.",
        ]

        return ToolResult(success=True, data="\n".join(lines))

    def _add_faction(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        faction = {
            "name": args["name"],
            "description": args.get("description", ""),
            "power_tokens": int(args.get("power_tokens", 1)),
        }
        self.state["factions"].append(faction)
        self._save_state(self.state)
        return ToolResult(success=True, data=f"**Faction Added:** {args['name']} ({faction['power_tokens']} power)")

    def _add_district(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        district = {
            "name": args["name"],
            "citizen_tokens": int(args.get("citizen_tokens", 1)),
            "landmarks": [],
        }
        self.state["districts"].append(district)
        self._save_state(self.state)
        return ToolResult(success=True, data=f"**District Added:** {args['name']} ({district['citizen_tokens']} citizens)")

    def _add_landmark(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        landmark_name = args["name"]
        district_name = args.get("district", "")

        self.state["landmarks"].append(landmark_name)

        if district_name:
            for d in self.state["districts"]:
                if d["name"] == district_name:
                    d["landmarks"].append(landmark_name)
                    break

        # Write to fiction tree
        root_id = self.state.get("history_root_id")
        if root_id:
            self.fiction_store.add_node(FictionNode(
                campaign_id=self.campaign_id,
                parent_id=root_id,
                title=landmark_name,
                zoom_level="detail",
                source_system="ex_novo",
                summary=f"Landmark in {district_name}" if district_name else "Landmark",
            ))

        self._save_state(self.state)
        loc = f" in {district_name}" if district_name else ""
        return ToolResult(success=True, data=f"**Landmark Added:** {landmark_name}{loc}")

    def _roll_event(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        code = roll_d666()
        event = EVENT_TABLE.get(code)

        self.state["turn_number"] = self.state.get("turn_number", 0) + 1
        self._save_state(self.state)

        if event is None:
            return ToolResult(
                success=True,
                data=f"**Event Roll: {code}** (Turn {self.state['turn_number']}) — No event for this code. Roll again.",
            )

        text = event["text"]
        effect = event["effect"]

        return ToolResult(
            success=True,
            data=(
                f"**Event Roll: {code}** (Turn {self.state['turn_number']})\n"
                f"**Prompt:** {text}\n"
                f"**Effect:** {effect}\n\n"
                f"Narrate what happens, then use `resolve_event` to apply the effect."
            ),
        )

    def _resolve_event(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        code = int(args["event_code"])
        narrative = args["narrative"]
        target = args.get("effect_target", "")

        event = EVENT_TABLE.get(code)
        if event is None:
            return ToolResult(success=False, error=f"No event for code {code}.")

        effect = event["effect"]
        result_text = f"**Resolved:** {narrative}\n"

        # Apply mechanical effect
        if effect == "add_citizen":
            if target and any(d["name"] == target for d in self.state["districts"]):
                for d in self.state["districts"]:
                    if d["name"] == target:
                        d["citizen_tokens"] += 1
                        result_text += f"+1 citizen in {target}"
                        break
            else:
                self.state["growth_pool"] += 1
                result_text += "+1 to growth pool"
        elif effect == "remove_citizen":
            if target and any(d["name"] == target for d in self.state["districts"]):
                for d in self.state["districts"]:
                    if d["name"] == target:
                        d["citizen_tokens"] = max(0, d["citizen_tokens"] - 1)
                        result_text += f"-1 citizen in {target}"
                        break
            else:
                self.state["growth_pool"] = max(0, self.state.get("growth_pool", 0) - 1)
                result_text += "-1 from growth pool"
        elif effect == "add_power":
            if target and any(f["name"] == target for f in self.state["factions"]):
                for f in self.state["factions"]:
                    if f["name"] == target:
                        f["power_tokens"] += 1
                        result_text += f"+1 power to {target}"
                        break
            else:
                result_text += "Power gained (assign to a faction)"
        elif effect == "remove_power":
            if target and any(f["name"] == target for f in self.state["factions"]):
                for f in self.state["factions"]:
                    if f["name"] == target:
                        f["power_tokens"] = max(0, f["power_tokens"] - 1)
                        result_text += f"-1 power from {target}"
                        break
            else:
                result_text += "Power lost (choose a faction)"
        elif effect == "add_resource":
            self.state["resources"].append(target or narrative[:40])
            result_text += f"Resource added: {target or narrative[:40]}"
        elif effect == "remove_resource":
            if self.state["resources"]:
                removed = target if target in self.state["resources"] else self.state["resources"][-1]
                if removed in self.state["resources"]:
                    self.state["resources"].remove(removed)
                    result_text += f"Resource lost: {removed}"
            else:
                result_text += "No resources to remove"
        elif effect == "add_problem":
            self.state["problems"].append(target or narrative[:40])
            result_text += f"Problem added: {target or narrative[:40]}"
        elif effect == "remove_problem":
            if self.state["problems"]:
                removed = target if target in self.state["problems"] else self.state["problems"][-1]
                if removed in self.state["problems"]:
                    self.state["problems"].remove(removed)
                    result_text += f"Problem resolved: {removed}"
            else:
                result_text += "No problems to resolve"
        elif effect == "add_landmark":
            self.state["landmarks"].append(target or narrative[:40])
            result_text += f"Landmark added: {target or narrative[:40]}"
        elif effect == "remove_landmark":
            if self.state["landmarks"]:
                removed = target if target in self.state["landmarks"] else self.state["landmarks"][-1]
                if removed in self.state["landmarks"]:
                    self.state["landmarks"].remove(removed)
                    result_text += f"Landmark destroyed: {removed}"
            else:
                result_text += "No landmarks to remove"
        elif effect == "add_faction":
            if target:
                self.state["factions"].append({"name": target, "description": narrative[:80], "power_tokens": 1})
                result_text += f"New faction: {target}"
            else:
                result_text += "New faction emerged (name it)"
        elif effect == "remove_faction":
            if target:
                self.state["factions"] = [f for f in self.state["factions"] if f["name"] != target]
                result_text += f"Faction dissolved: {target}"
            else:
                result_text += "A faction dissolves (choose which)"
        elif effect == "catastrophe":
            result_text += "CATASTROPHE — choose the worst possible outcome!"
        else:
            result_text += f"Effect: {effect}"

        # Write event to fiction tree
        root_id = self.state.get("history_root_id")
        if root_id:
            self.fiction_store.add_node(FictionNode(
                campaign_id=self.campaign_id,
                parent_id=root_id,
                title=event["text"][:60],
                zoom_level="event",
                tone="dark" if "remove" in effect or effect == "catastrophe" else "light",
                source_system="ex_novo",
                summary=narrative[:200],
                content=f"{narrative}\n\nEffect: {result_text}",
            ))

        self._save_state(self.state)
        return ToolResult(success=True, data=result_text)

    def _advance_phase(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        phase_order = ["founding", "development", "topping_out"]
        current = self.state["phase"]
        if current not in phase_order:
            return ToolResult(success=False, error=f"Cannot advance from phase: {current}")

        idx = phase_order.index(current)
        if idx >= len(phase_order) - 1:
            return ToolResult(success=True, data="Already at final phase (topping out).")

        new_phase = phase_order[idx + 1]
        self.state["phase"] = new_phase
        self._save_state(self.state)

        return ToolResult(success=True, data=f"**Phase Advanced:** {current} → {new_phase}")

    def _get_state(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active settlement.")

        s = self.state
        lines = [
            f"**{s['name']}** — {s['phase']}  (Turn {s.get('turn_number', 0)})\n",
            f"**Scale:** {s['scale']['age']} | {s['scale']['size']} | {s['scale']['influence']}",
            f"**Terrain:** {s['terrain']['geography']}",
        ]

        if s["factions"]:
            lines.append("\n**Factions:**")
            for f in s["factions"]:
                lines.append(f"  - {f['name']} ({f['power_tokens']} power)")

        if s["districts"]:
            lines.append("\n**Districts:**")
            for d in s["districts"]:
                lm = f" — Landmarks: {', '.join(d['landmarks'])}" if d.get("landmarks") else ""
                lines.append(f"  - {d['name']} ({d['citizen_tokens']} citizens){lm}")

        if s["resources"]:
            lines.append(f"\n**Resources:** {', '.join(s['resources'])}")
        if s["problems"]:
            lines.append(f"**Problems:** {', '.join(s['problems'])}")
        if s["landmarks"]:
            lines.append(f"**Landmarks:** {', '.join(s['landmarks'])}")

        lines.append(f"\n**Growth Pool:** {s.get('growth_pool', 0)}")

        return ToolResult(success=True, data="\n".join(lines))

    def _browse_history(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        root_id = self.state.get("history_root_id")
        if not root_id:
            return ToolResult(success=True, data="No history recorded.")

        summary = self.fiction_store.get_tree_summary(root_id)
        return ToolResult(success=True, data=summary or "No history entries.")

    def _end(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active settlement to end.")

        self.state["phase"] = "ended"
        self._save_state(self.state)

        s = self.state
        lines = [
            f"**Settlement Finalized:** {s['name']}\n",
            f"**Turns:** {s.get('turn_number', 0)}",
            f"**Factions:** {len(s['factions'])}",
            f"**Districts:** {len(s['districts'])}",
            f"**Landmarks:** {len(s['landmarks'])}",
            f"**Resources:** {len(s['resources'])}",
            f"**Problems:** {len(s['problems'])}",
        ]
        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Virtual Players
    # -------------------------------------------------------------------

    def _add_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        name = args.get("name", "")
        if not name:
            return ToolResult(success=False, error="Player name is required.")

        archetype = args.get("archetype", "")
        traits_str = args.get("traits", "{}")
        if isinstance(traits_str, str):
            try:
                custom_traits = json.loads(traits_str)
            except json.JSONDecodeError:
                custom_traits = {}
        else:
            custom_traits = traits_str

        # Build personality profile
        personality_profile: dict[str, Any] = {}
        if archetype:
            try:
                from ..shared.personality import PersonalityProfile
                pp = PersonalityProfile.from_archetype(archetype)
                personality_profile = pp.to_dict()
            except (ImportError, KeyError) as e:
                return ToolResult(success=False, error=f"Unknown archetype: '{archetype}'")
        elif custom_traits:
            personality_profile = {"name": name, "traits": custom_traits, "archetype": ""}

        vp = {
            "name": name,
            "personality_profile": personality_profile,
            "creative_preferences": {},
            "is_active": True,
            "history": [],
        }

        if "virtual_players" not in self.state:
            self.state["virtual_players"] = []

        # Check for duplicate
        for existing in self.state["virtual_players"]:
            if existing["name"] == name:
                return ToolResult(success=False, error=f"Virtual player '{name}' already exists.")

        self.state["virtual_players"].append(vp)
        self._save_state(self.state)

        desc = f"archetype: {archetype}" if archetype else f"{len(custom_traits)} custom traits"
        return ToolResult(
            success=True,
            data=f"**Virtual Player Added:** {name} ({desc})",
        )

    def _virtual_player_turn(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")
        if not self.llm:
            return ToolResult(success=False, error="No LLM backend available for virtual players.")

        player_name = args.get("player_name", "")
        decision_type = args.get("decision_type", "create")

        vps = self.state.get("virtual_players", [])
        vp = next((v for v in vps if v["name"] == player_name and v.get("is_active", True)), None)
        if vp is None:
            return ToolResult(success=False, error=f"Virtual player '{player_name}' not found or inactive.")

        from ..shared.virtual_player_engine import VirtualPlayerEngine

        engine = VirtualPlayerEngine(self.llm)

        # Build game state summary for the LLM
        game_state = {
            "settlement_name": self.state.get("name", ""),
            "phase": self.state.get("phase", ""),
            "turn_number": self.state.get("turn_number", 0),
            "scale": self.state.get("scale", {}),
            "terrain": self.state.get("terrain", {}),
            "purpose": self.state.get("purpose", {}),
            "power": self.state.get("power", {}),
            "factions": self.state.get("factions", []),
            "districts": self.state.get("districts", []),
            "resources": self.state.get("resources", []),
            "problems": self.state.get("problems", []),
            "landmarks": self.state.get("landmarks", []),
            "growth_pool": self.state.get("growth_pool", 0),
        }

        # Tell the VP engine what actions are available
        available = [
            "add_district", "add_faction", "add_landmark",
            "roll_event", "resolve_event",
        ]
        if decision_type == "create":
            decision_type = "add_district"  # sensible default

        decision = engine.generate_decision(
            player_name=player_name,
            personality_profile=vp.get("personality_profile"),
            creative_preferences=vp.get("creative_preferences"),
            game_state=game_state,
            decision_type=decision_type,
            available_options=available,
            decision_history=vp.get("history", []),
        )

        # Track decision in VP history
        vp.setdefault("history", []).append(decision)
        self._save_state(self.state)

        # Auto-execute the decision
        actual_type = decision.get("decision_type", decision_type)
        content = decision.get("content", {})
        reasoning = decision.get("reasoning", "")

        exec_result = self._execute_vp_decision(actual_type, content)

        header = f"**{player_name}** ({actual_type})"
        if reasoning:
            header += f"\n*Reasoning: {reasoning}*"

        if exec_result.success:
            return ToolResult(
                success=True,
                data=f"{header}\n\n{exec_result.data}",
            )
        else:
            # Decision generated but execution failed — return both
            return ToolResult(
                success=True,
                data=(
                    f"{header}\n\n"
                    f"Decision generated but could not auto-execute: {exec_result.error}\n"
                    f"Content: {json.dumps(content, indent=2)}"
                ),
            )

    def _execute_vp_decision(
        self, decision_type: str, content: dict[str, Any],
    ) -> ToolResult:
        """Execute a virtual player's decision by calling the appropriate internal method."""
        if decision_type == "add_district":
            return self._add_district({
                "name": content.get("name", content.get("title", "Untitled District")),
                "citizen_tokens": content.get("citizen_tokens", 1),
            })
        elif decision_type == "add_faction":
            return self._add_faction({
                "name": content.get("name", content.get("title", "Untitled Faction")),
                "description": content.get("description", ""),
                "power_tokens": content.get("power_tokens", 1),
            })
        elif decision_type == "add_landmark":
            district = content.get("district", "")
            if not district:
                # Try to place in first available district
                districts = self.state.get("districts", []) if self.state else []
                if districts:
                    district = districts[0]["name"]
            return self._add_landmark({
                "name": content.get("name", content.get("title", "Untitled Landmark")),
                "district": district,
            })
        elif decision_type == "roll_event":
            return self._roll_event()
        elif decision_type == "resolve_event":
            event_code = content.get("event_code", 0)
            if not event_code:
                return ToolResult(success=False, error="No event_code specified for resolve_event.")
            return self._resolve_event({
                "event_code": event_code,
                "narrative": content.get("narrative", ""),
                "effect_target": content.get("effect_target", ""),
            })
        else:
            return ToolResult(
                success=False,
                error=f"Unknown decision type '{decision_type}' — could not auto-execute.",
            )

    def _remove_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active settlement.")

        player_name = args.get("player_name", "")
        vps = self.state.get("virtual_players", [])

        for i, vp in enumerate(vps):
            if vp["name"] == player_name:
                vps.pop(i)
                self._save_state(self.state)
                return ToolResult(success=True, data=f"**Virtual Player Removed:** {player_name}")

        return ToolResult(success=False, error=f"Virtual player '{player_name}' not found.")

    def close(self) -> None:
        if self._fiction_store:
            self._fiction_store.close()
            self._fiction_store = None
