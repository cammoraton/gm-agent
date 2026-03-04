"""Delve hold-building MCP server.

Provides tools for running a Delve underground hold-building game.
"""

import json
import math
from typing import Any

from ...config import CAMPAIGNS_DIR
from ...storage.fiction_tree import FictionTreeStore, FictionNode
from ...mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult
from ..shared.cards import CardDeck
from .tables import (
    HEARTS_DISCOVERY, DIAMONDS_DISCOVERY, CLUBS_DISCOVERY, SPADES_DISCOVERY,
    ROOM_TYPES, UNIT_TYPES, DEPTH_SCALING,
)


class DelveServer(MCPServer):
    """MCP server for Delve hold-building game."""

    def __init__(self, campaign_id: str, llm: Any = None):
        self.campaign_id = campaign_id
        self.llm = llm
        self._fiction_store: FictionTreeStore | None = None
        self._state: dict[str, Any] | None = None
        self._state_path = CAMPAIGNS_DIR / campaign_id / "delve_state.json"
        self._deck: CardDeck | None = None
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

    @property
    def deck(self) -> CardDeck:
        if self._deck is None:
            self._deck = CardDeck()
            # Advance deck to match state
            if self.state:
                cards_drawn = 52 - self.state.get("deck_remaining", 52)
                for _ in range(cards_drawn):
                    self._deck.draw()
        return self._deck

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
                name="delve_setup",
                description="Initialize a new hold: name, starting resources, shuffle deck.",
                parameters=[
                    ToolParameter(name="hold_name", type="string", description="Name of the hold."),
                    ToolParameter(name="resources", type="integer",
                                  description="Starting resources.", required=False, default=20),
                    ToolParameter(name="trade_goods", type="integer",
                                  description="Starting trade goods.", required=False, default=10),
                ],
            ),
            ToolDef(
                name="explore",
                description="Draw a card — suit determines discovery type. Returns what was found.",
                parameters=[],
            ),
            ToolDef(
                name="resolve_combat",
                description="Calculate combat: party STR vs enemy STR. Returns outcome.",
                parameters=[
                    ToolParameter(name="enemy_str", type="integer", description="Enemy strength."),
                    ToolParameter(name="party_units", type="string",
                                  description="JSON array of unit names to send to combat.",
                                  required=False, default="[]"),
                ],
            ),
            ToolDef(
                name="trade",
                description="Spend trade goods at market rates.",
                parameters=[
                    ToolParameter(name="trade_goods_spent", type="integer",
                                  description="Trade goods to spend."),
                    ToolParameter(name="resources_gained", type="integer",
                                  description="Resources to gain."),
                ],
            ),
            ToolDef(
                name="build_room",
                description="Build a room (pay resources cost).",
                parameters=[
                    ToolParameter(name="room_type", type="string",
                                  description=f"Room type: {', '.join(ROOM_TYPES.keys())}"),
                    ToolParameter(name="name", type="string",
                                  description="Custom name for the room.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="recruit_unit",
                description="Recruit a unit (pay resources cost, assign to room).",
                parameters=[
                    ToolParameter(name="unit_type", type="string",
                                  description=f"Unit type: {', '.join(UNIT_TYPES.keys())}"),
                    ToolParameter(name="name", type="string",
                                  description="Unit name.", required=False, default=""),
                    ToolParameter(name="room", type="string",
                                  description="Room to assign to.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="build_trap",
                description="Build a trap or barricade at current depth.",
                parameters=[
                    ToolParameter(name="trap_type", type="string",
                                  description="Type of trap (e.g., 'pit', 'arrow', 'barricade')."),
                    ToolParameter(name="level", type="integer",
                                  description="Trap effectiveness level (1-5).", required=False, default=1),
                ],
            ),
            ToolDef(
                name="get_hold_state",
                description="Full hold overview: rooms, units, resources, depth.",
                parameters=[],
            ),
            ToolDef(
                name="descend",
                description="Go deeper — increases depth, scales enemy difficulty.",
                parameters=[],
            ),
            ToolDef(
                name="end_delve",
                description="End the game — summarize the hold.",
                parameters=[],
            ),
            # --- Virtual Players ---
            ToolDef(
                name="add_virtual_player",
                description="Register an AI-driven virtual player for Delve with a personality archetype or custom traits.",
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
                description="Generate and execute a virtual player's turn in Delve using AI.",
                parameters=[
                    ToolParameter(name="player_name", type="string",
                                  description="Name of the virtual player to act."),
                    ToolParameter(name="decision_type", type="string",
                                  description="Type of decision: 'explore', 'build_room', 'recruit_unit', etc.",
                                  required=False, default="create"),
                ],
            ),
            ToolDef(
                name="remove_virtual_player",
                description="Remove a virtual player from the Delve session.",
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
            if name == "delve_setup":
                return self._setup(args)
            elif name == "explore":
                return self._explore()
            elif name == "resolve_combat":
                return self._resolve_combat(args)
            elif name == "trade":
                return self._trade(args)
            elif name == "build_room":
                return self._build_room(args)
            elif name == "recruit_unit":
                return self._recruit_unit(args)
            elif name == "build_trap":
                return self._build_trap(args)
            elif name == "get_hold_state":
                return self._get_state()
            elif name == "descend":
                return self._descend()
            elif name == "end_delve":
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
            return ToolResult(success=False, error="A Delve session is already active.")

        self._deck = CardDeck()

        root = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            title=args["hold_name"],
            zoom_level="era",
            source_system="delve",
            summary=f"Underground hold: {args['hold_name']}",
        ))

        state = {
            "hold_name": args["hold_name"],
            "phase": "active",
            "depth": 1,
            "resources": int(args.get("resources", 20)),
            "trade_goods": int(args.get("trade_goods", 10)),
            "resource_cap": 50,
            "trade_goods_cap": 50,
            "rooms": [],
            "units": [],
            "traps": [],
            "deck_remaining": 52,
            "turn_number": 0,
            "combat_log": [],
            "discoveries": [],
            "history_root_id": root.id,
        }
        self._save_state(state)

        return ToolResult(
            success=True,
            data=(
                f"**{args['hold_name']} — Hold Founded**\n\n"
                f"**Resources:** {state['resources']}/{state['resource_cap']}\n"
                f"**Trade Goods:** {state['trade_goods']}/{state['trade_goods_cap']}\n"
                f"**Depth:** {state['depth']}\n"
                f"**Deck:** {state['deck_remaining']} cards\n\n"
                f"Begin exploring with the `explore` tool."
            ),
        )

    def _explore(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        card = self.deck.draw()
        if card is None:
            return ToolResult(success=True, data="**Deck exhausted!** The hold has explored all it can at this depth.")

        self.state["deck_remaining"] = self.deck.remaining()
        self.state["turn_number"] += 1

        suit = card.suit
        rank = card.rank

        if suit == "hearts":
            discovery = HEARTS_DISCOVERY.get(rank, HEARTS_DISCOVERY["A"])
            gained = discovery["resources"]
            self.state["resources"] = min(self.state["resource_cap"], self.state["resources"] + gained)
            result_text = f"**{card}** — {discovery['text']}\n+{gained} Resources (now {self.state['resources']})"
        elif suit == "diamonds":
            discovery = DIAMONDS_DISCOVERY.get(rank, DIAMONDS_DISCOVERY["A"])
            gained = discovery["trade_goods"]
            self.state["trade_goods"] = min(self.state["trade_goods_cap"], self.state["trade_goods"] + gained)
            result_text = f"**{card}** — {discovery['text']}\n+{gained} Trade Goods (now {self.state['trade_goods']})"
        elif suit == "clubs":
            discovery = CLUBS_DISCOVERY.get(rank, CLUBS_DISCOVERY["A"])
            result_text = (
                f"**{card}** — {discovery['text']}\n"
                f"Type: {discovery['type']} | Build cost: {discovery['room_cost']} resources"
            )
        else:  # spades
            discovery = SPADES_DISCOVERY.get(rank, SPADES_DISCOVERY["A"])
            if discovery.get("threat"):
                enemy_str = discovery.get("str", 1)
                depth_mult = DEPTH_SCALING.get(self.state["depth"], 1.0)
                scaled_str = math.ceil(enemy_str * depth_mult)
                result_text = (
                    f"**{card}** — {discovery['text']}\n"
                    f"**THREAT!** Enemy STR: {scaled_str} (base {enemy_str} x depth {depth_mult})\n"
                    f"Use `resolve_combat` to fight, or retreat."
                )
            else:
                result_text = f"**{card}** — {discovery['text']}\nNo threat. Describe what you find."

        self.state["discoveries"].append({
            "card": str(card),
            "turn": self.state["turn_number"],
            "depth": self.state["depth"],
        })

        # Write to fiction tree
        self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=self.state["history_root_id"],
            title=f"Turn {self.state['turn_number']}: {str(card)}",
            zoom_level="event",
            tone="dark" if suit == "spades" and discovery.get("threat") else "light",
            source_system="delve",
            summary=result_text[:200],
        ))

        self._save_state(self.state)
        return ToolResult(success=True, data=result_text)

    def _resolve_combat(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        enemy_str = int(args["enemy_str"])
        party_units = args.get("party_units", "[]")
        if isinstance(party_units, str):
            party_units = json.loads(party_units)

        # Calculate party STR
        party_str = 0
        for unit_name in party_units:
            for u in self.state["units"]:
                if u["name"] == unit_name:
                    party_str += u["str"]
                    break

        # If no specific units sent, use all
        if not party_units:
            party_str = sum(u["str"] for u in self.state["units"])

        victory = party_str > enemy_str
        lines = [
            f"**Combat:** Party STR {party_str} vs Enemy STR {enemy_str}",
        ]

        if victory:
            lines.append("**Victory!** The threat is defeated.")
        elif party_str == enemy_str:
            lines.append("**Stalemate!** Both sides withdraw (tie goes to defender).")
        else:
            # Lose a unit
            lines.append("**Defeat!** The hold's defenders are pushed back.")
            if self.state["units"]:
                lost = self.state["units"].pop()
                lines.append(f"**Lost unit:** {lost['name']} ({lost['type']})")

        self.state["combat_log"].append({
            "turn": self.state["turn_number"],
            "party_str": party_str,
            "enemy_str": enemy_str,
            "result": "victory" if victory else "defeat",
        })
        self._save_state(self.state)

        return ToolResult(success=True, data="\n".join(lines))

    def _trade(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        spent = int(args["trade_goods_spent"])
        gained = int(args["resources_gained"])

        if spent > self.state["trade_goods"]:
            return ToolResult(success=False, error=f"Not enough trade goods ({self.state['trade_goods']} available).")

        self.state["trade_goods"] -= spent
        self.state["resources"] = min(self.state["resource_cap"], self.state["resources"] + gained)
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Trade:** -{spent} trade goods, +{gained} resources\nResources: {self.state['resources']} | Trade Goods: {self.state['trade_goods']}",
        )

    def _build_room(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        room_type = args["room_type"]
        if room_type not in ROOM_TYPES:
            return ToolResult(success=False, error=f"Unknown room type '{room_type}'. Valid: {', '.join(ROOM_TYPES.keys())}")

        room_info = ROOM_TYPES[room_type]
        cost = room_info["cost"]

        if self.state["resources"] < cost:
            return ToolResult(success=False, error=f"Not enough resources ({self.state['resources']}/{cost}).")

        self.state["resources"] -= cost
        room = {
            "name": args.get("name") or f"{room_type.title()} {len(self.state['rooms']) + 1}",
            "type": room_type,
            "depth": self.state["depth"],
            "units_housed": [],
        }
        self.state["rooms"].append(room)

        if room_type == "treasury":
            self.state["resource_cap"] += 25
            self.state["trade_goods_cap"] += 25

        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Room Built:** {room['name']} ({room_type}) at depth {self.state['depth']}\n"
                f"Cost: {cost} resources | Remaining: {self.state['resources']}"
            ),
        )

    def _recruit_unit(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        unit_type = args["unit_type"]
        if unit_type not in UNIT_TYPES:
            return ToolResult(success=False, error=f"Unknown unit type '{unit_type}'. Valid: {', '.join(UNIT_TYPES.keys())}")

        unit_info = UNIT_TYPES[unit_type]
        cost = unit_info["cost"]

        if self.state["resources"] < cost:
            return ToolResult(success=False, error=f"Not enough resources ({self.state['resources']}/{cost}).")

        self.state["resources"] -= cost
        unit = {
            "name": args.get("name") or f"{unit_type.title()} {len(self.state['units']) + 1}",
            "type": unit_type,
            "str": unit_info["str"],
            "room": args.get("room", ""),
        }
        self.state["units"].append(unit)
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Unit Recruited:** {unit['name']} ({unit_type}, STR {unit['str']})\n"
                f"Cost: {cost} resources | Remaining: {self.state['resources']}"
            ),
        )

    def _build_trap(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        trap_cost = int(args.get("level", 1)) * 2
        if self.state["resources"] < trap_cost:
            return ToolResult(success=False, error=f"Not enough resources ({self.state['resources']}/{trap_cost}).")

        self.state["resources"] -= trap_cost
        trap = {
            "type": args["trap_type"],
            "level": int(args.get("level", 1)),
            "depth": self.state["depth"],
        }
        self.state["traps"].append(trap)
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Trap Built:** {args['trap_type']} (level {trap['level']}) at depth {self.state['depth']} | Cost: {trap_cost}",
        )

    def _get_state(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active Delve session.")

        s = self.state
        lines = [
            f"**{s['hold_name']}** — Depth {s['depth']}  (Turn {s['turn_number']})\n",
            f"**Resources:** {s['resources']}/{s['resource_cap']}",
            f"**Trade Goods:** {s['trade_goods']}/{s['trade_goods_cap']}",
            f"**Deck:** {s['deck_remaining']} cards remaining",
        ]

        total_str = sum(u["str"] for u in s["units"])
        lines.append(f"**Total Party STR:** {total_str}")

        if s["rooms"]:
            lines.append(f"\n**Rooms ({len(s['rooms'])}):**")
            for r in s["rooms"]:
                lines.append(f"  - {r['name']} ({r['type']}, depth {r['depth']})")

        if s["units"]:
            lines.append(f"\n**Units ({len(s['units'])}):**")
            for u in s["units"]:
                lines.append(f"  - {u['name']} ({u['type']}, STR {u['str']})")

        if s["traps"]:
            lines.append(f"\n**Traps ({len(s['traps'])}):**")
            for t in s["traps"]:
                lines.append(f"  - {t['type']} (level {t['level']}, depth {t['depth']})")

        return ToolResult(success=True, data="\n".join(lines))

    def _descend(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

        self.state["depth"] += 1
        depth = self.state["depth"]

        # Reshuffle deck for new depth
        self._deck = CardDeck()
        self.state["deck_remaining"] = 52

        self._save_state(self.state)

        scale = DEPTH_SCALING.get(depth, depth * 0.5)
        return ToolResult(
            success=True,
            data=(
                f"**Descended to Depth {depth}**\n"
                f"Enemy scaling: x{scale}\n"
                f"Deck reshuffled: 52 cards\n"
                f"Richer discoveries await — but so do greater dangers."
            ),
        )

    def _end(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active Delve session to end.")

        self.state["phase"] = "ended"
        self._save_state(self.state)

        s = self.state
        wins = sum(1 for c in s["combat_log"] if c["result"] == "victory")
        losses = sum(1 for c in s["combat_log"] if c["result"] != "victory")

        lines = [
            f"**{s['hold_name']} — Delve Ended**\n",
            f"**Depth Reached:** {s['depth']}",
            f"**Turns:** {s['turn_number']}",
            f"**Rooms Built:** {len(s['rooms'])}",
            f"**Units:** {len(s['units'])}",
            f"**Combats:** {wins} victories, {losses} defeats",
            f"**Final Resources:** {s['resources']}",
            f"**Final Trade Goods:** {s['trade_goods']}",
        ]
        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Virtual Players
    # -------------------------------------------------------------------

    def _add_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

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
            return ToolResult(success=False, error="No active Delve session.")
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
            "hold_name": self.state.get("hold_name", ""),
            "phase": self.state.get("phase", ""),
            "depth": self.state.get("depth", 1),
            "turn_number": self.state.get("turn_number", 0),
            "resources": self.state.get("resources", 0),
            "resource_cap": self.state.get("resource_cap", 50),
            "trade_goods": self.state.get("trade_goods", 0),
            "trade_goods_cap": self.state.get("trade_goods_cap", 50),
            "rooms": self.state.get("rooms", []),
            "units": self.state.get("units", []),
            "traps": self.state.get("traps", []),
            "deck_remaining": self.state.get("deck_remaining", 0),
            "total_str": sum(u["str"] for u in self.state.get("units", [])),
        }

        # Tell the VP engine what actions are available
        available = [
            "explore", "build_room", "recruit_unit", "build_trap",
            "trade", "descend",
        ]
        if decision_type == "create":
            decision_type = "explore"  # sensible default

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
            # Decision generated but execution failed -- return both
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
        if decision_type == "explore":
            return self._explore()
        elif decision_type == "build_room":
            room_type = content.get("room_type", content.get("type", "quarters"))
            return self._build_room({
                "room_type": room_type,
                "name": content.get("name", ""),
            })
        elif decision_type == "recruit_unit":
            unit_type = content.get("unit_type", content.get("type", "soldier"))
            return self._recruit_unit({
                "unit_type": unit_type,
                "name": content.get("name", ""),
                "room": content.get("room", ""),
            })
        elif decision_type == "build_trap":
            return self._build_trap({
                "trap_type": content.get("trap_type", content.get("type", "pit")),
                "level": content.get("level", 1),
            })
        elif decision_type == "trade":
            return self._trade({
                "trade_goods_spent": content.get("trade_goods_spent", 0),
                "resources_gained": content.get("resources_gained", 0),
            })
        elif decision_type == "descend":
            return self._descend()
        else:
            return ToolResult(
                success=False,
                error=f"Unknown decision type '{decision_type}' — could not auto-execute.",
            )

    def _remove_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Delve session.")

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
