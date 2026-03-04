"""Ex Umbra dungeon-building MCP server.

Provides tools for running an Ex Umbra collaborative dungeon generation game.
"""

import json
import random
from typing import Any

from ...config import CAMPAIGNS_DIR
from ...storage.fiction_tree import FictionTreeStore, FictionNode
from ...mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult
from .tables import (
    TREMOR_TABLE, DIFFICULTY_SETTINGS,
    ADJECTIVE_PAIRS, ARCHITECTURE_TYPES,
)


class DungeonServer(MCPServer):
    """MCP server for Ex Umbra dungeon building."""

    def __init__(self, campaign_id: str, llm: Any = None):
        self.campaign_id = campaign_id
        self.llm = llm
        self._fiction_store: FictionTreeStore | None = None
        self._state: dict[str, Any] | None = None
        self._state_path = CAMPAIGNS_DIR / campaign_id / "ex_umbra_state.json"
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
                name="ex_umbra_setup",
                description=(
                    "Initialize a dungeon: set size (2-12), difficulty (1-4), "
                    "and aspects (1-3 sources of peril). Calculates pools and turn clock."
                ),
                parameters=[
                    ToolParameter(name="dungeon_name", type="string", description="Name of the dungeon."),
                    ToolParameter(name="size", type="integer", description="Dungeon size (2-12)."),
                    ToolParameter(name="difficulty", type="integer", description="Difficulty level (1-4)."),
                    ToolParameter(name="aspects", type="string",
                                  description="JSON array of aspect names (1-3 peril sources)."),
                ],
            ),
            ToolDef(
                name="set_foundation",
                description="Planning phase: establish guide lines and initial architecture.",
                parameters=[
                    ToolParameter(name="guide_lines", type="string",
                                  description="JSON array of abstract guide line descriptions."),
                    ToolParameter(name="initial_rooms", type="string",
                                  description="JSON array of {name, type, connected_to: []} room dicts."),
                ],
            ),
            ToolDef(
                name="play_dungeon_card",
                description="Play a dungeon card on an aspect — add architecture, features, threats, or rewards.",
                parameters=[
                    ToolParameter(name="aspect", type="string", description="Which aspect this card relates to."),
                    ToolParameter(name="room_name", type="string", description="Name for the new room/area."),
                    ToolParameter(name="room_type", type="string",
                                  description=f"Architecture type: {', '.join(ARCHITECTURE_TYPES[:8])}, etc."),
                    ToolParameter(name="description", type="string",
                                  description="Description of what's in this area.", required=False, default=""),
                    ToolParameter(name="connected_to", type="string",
                                  description="JSON array of room names this connects to.",
                                  required=False, default="[]"),
                ],
            ),
            ToolDef(
                name="spend_tokens",
                description="Spend threat/reward tokens from the dungeon pool onto architecture.",
                parameters=[
                    ToolParameter(name="pool", type="string",
                                  description="'dungeon' or 'heart'."),
                    ToolParameter(name="token_type", type="string",
                                  description="'threat' or 'reward'."),
                    ToolParameter(name="amount", type="integer", description="Number of tokens to spend."),
                    ToolParameter(name="target", type="string",
                                  description="Name of the room/feature to apply tokens to."),
                    ToolParameter(name="description", type="string",
                                  description="What the tokens represent.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="exploration_turn",
                description="Execute one exploration turn: describe expansion of the dungeon.",
                parameters=[
                    ToolParameter(name="description", type="string",
                                  description="What happens during this exploration turn."),
                ],
            ),
            ToolDef(
                name="tremor_turn",
                description="Roll D20 on tremor table — a collaborative modification to the dungeon.",
                parameters=[],
            ),
            ToolDef(
                name="heart_turn",
                description="Create the heart/central chamber — spend all heart pool tokens.",
                parameters=[
                    ToolParameter(name="name", type="string", description="Name of the heart chamber."),
                    ToolParameter(name="description", type="string",
                                  description="Description of the heart chamber."),
                ],
            ),
            ToolDef(
                name="get_dungeon_state",
                description="Full dungeon overview: architecture, threats, rewards, pools, turns remaining.",
                parameters=[],
            ),
            ToolDef(
                name="cleanup",
                description="Cleanup phase: finalize connections, resolve loose ends.",
                parameters=[
                    ToolParameter(name="notes", type="string",
                                  description="Final notes and connections.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="end_dungeon",
                description="Finalize the dungeon, write complete structure to fiction tree.",
                parameters=[],
            ),
            # --- Virtual Players ---
            ToolDef(
                name="add_virtual_player",
                description="Register an AI-driven virtual player for Ex Umbra with a personality archetype or custom traits.",
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
                description="Generate and execute a virtual player's turn in Ex Umbra using AI.",
                parameters=[
                    ToolParameter(name="player_name", type="string",
                                  description="Name of the virtual player to act."),
                    ToolParameter(name="decision_type", type="string",
                                  description="Type of decision: 'play_dungeon_card', 'spend_tokens', 'exploration_turn', etc.",
                                  required=False, default="create"),
                ],
            ),
            ToolDef(
                name="remove_virtual_player",
                description="Remove a virtual player from the Ex Umbra session.",
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
            if name == "ex_umbra_setup":
                return self._setup(args)
            elif name == "set_foundation":
                return self._set_foundation(args)
            elif name == "play_dungeon_card":
                return self._play_card(args)
            elif name == "spend_tokens":
                return self._spend_tokens(args)
            elif name == "exploration_turn":
                return self._exploration_turn(args)
            elif name == "tremor_turn":
                return self._tremor_turn()
            elif name == "heart_turn":
                return self._heart_turn(args)
            elif name == "get_dungeon_state":
                return self._get_state()
            elif name == "cleanup":
                return self._cleanup(args)
            elif name == "end_dungeon":
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
            return ToolResult(success=False, error="A dungeon is already active.")

        size = int(args["size"])
        difficulty = int(args["difficulty"])
        aspects = args["aspects"]
        if isinstance(aspects, str):
            aspects = json.loads(aspects)

        if not 2 <= size <= 12:
            return ToolResult(success=False, error="Size must be 2-12.")
        if not 1 <= difficulty <= 4:
            return ToolResult(success=False, error="Difficulty must be 1-4.")
        if not 1 <= len(aspects) <= 3:
            return ToolResult(success=False, error="Need 1-3 aspects.")

        diff_settings = DIFFICULTY_SETTINGS[difficulty]
        threat_pool = int(size * 2 * diff_settings["threat_mult"])
        reward_pool = int(size * 2 * diff_settings["reward_mult"])
        turn_clock = size * 2
        tremors = max(1, size // 6) * diff_settings.get("tremors_per_6", 1)

        root = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            title=args["dungeon_name"],
            zoom_level="era",
            source_system="ex_umbra",
            summary=f"Dungeon: {args['dungeon_name']} (size {size}, {diff_settings['name']})",
        ))

        state = {
            "dungeon_name": args["dungeon_name"],
            "phase": "planning",
            "size": size,
            "difficulty": difficulty,
            "difficulty_name": diff_settings["name"],
            "aspects": aspects,
            "guide_lines": [],
            "architecture": [],
            "features": [],
            "threats": [],
            "rewards": [],
            "dungeon_pool": {"threat": threat_pool, "reward": reward_pool},
            "heart_pool": {"threat": 0, "reward": 0},
            "turn_clock": turn_clock,
            "turns_remaining": turn_clock,
            "tremors_remaining": tremors,
            "history_root_id": root.id,
        }
        self._save_state(state)

        # Generate adjective prompts
        adj_pairs = random.sample(ADJECTIVE_PAIRS, min(size, len(ADJECTIVE_PAIRS)))

        return ToolResult(
            success=True,
            data=(
                f"**{args['dungeon_name']} — Dungeon Initialized**\n\n"
                f"**Size:** {size} | **Difficulty:** {diff_settings['name']}\n"
                f"**Aspects:** {', '.join(aspects)}\n"
                f"**Dungeon Pool:** {threat_pool} threat, {reward_pool} reward tokens\n"
                f"**Turn Clock:** {turn_clock} exploration turns\n"
                f"**Tremors:** {tremors}\n\n"
                f"**Adjective Prompts:** {', '.join(f'{a}/{b}' for a, b in adj_pairs)}\n\n"
                f"Next: Use `set_foundation` to establish guide lines and initial architecture."
            ),
        )

    def _set_foundation(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        guide_lines = args.get("guide_lines", "[]")
        if isinstance(guide_lines, str):
            guide_lines = json.loads(guide_lines)

        initial_rooms = args.get("initial_rooms", "[]")
        if isinstance(initial_rooms, str):
            initial_rooms = json.loads(initial_rooms)

        self.state["guide_lines"] = guide_lines
        for room in initial_rooms:
            self.state["architecture"].append({
                "name": room.get("name", "Unknown"),
                "type": room.get("type", "chamber"),
                "connected_to": room.get("connected_to", []),
            })

        self.state["phase"] = "foundation"
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Foundation Set**\n"
                f"**Guide Lines:** {len(guide_lines)}\n"
                f"**Initial Rooms:** {len(initial_rooms)}\n\n"
                f"Next: Play dungeon cards with `play_dungeon_card`."
            ),
        )

    def _play_card(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        room = {
            "name": args["room_name"],
            "type": args["room_type"],
            "connected_to": json.loads(args.get("connected_to", "[]")) if isinstance(args.get("connected_to"), str) else args.get("connected_to", []),
        }
        self.state["architecture"].append(room)

        # Move some tokens to heart pool (1 of each per card played)
        if self.state["dungeon_pool"]["threat"] > 0:
            self.state["dungeon_pool"]["threat"] -= 1
            self.state["heart_pool"]["threat"] += 1
        if self.state["dungeon_pool"]["reward"] > 0:
            self.state["dungeon_pool"]["reward"] -= 1
            self.state["heart_pool"]["reward"] += 1

        # Write to fiction tree
        self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=self.state["history_root_id"],
            title=args["room_name"],
            zoom_level="event",
            source_system="ex_umbra",
            summary=args.get("description", ""),
            content=f"Type: {args['room_type']}\nAspect: {args['aspect']}\n{args.get('description', '')}",
        ))

        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Card Played:** {args['room_name']} ({args['room_type']}) on aspect '{args['aspect']}'\n"
                f"**Dungeon Pool:** {self.state['dungeon_pool']['threat']}T / {self.state['dungeon_pool']['reward']}R\n"
                f"**Heart Pool:** {self.state['heart_pool']['threat']}T / {self.state['heart_pool']['reward']}R"
            ),
        )

    def _spend_tokens(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        pool_name = args["pool"]
        token_type = args["token_type"]
        amount = int(args["amount"])
        target = args["target"]

        if pool_name not in ("dungeon", "heart"):
            return ToolResult(success=False, error="Pool must be 'dungeon' or 'heart'.")
        if token_type not in ("threat", "reward"):
            return ToolResult(success=False, error="Token type must be 'threat' or 'reward'.")

        pool = self.state[f"{pool_name}_pool"]
        if pool[token_type] < amount:
            return ToolResult(
                success=False,
                error=f"Not enough {token_type} tokens in {pool_name} pool ({pool[token_type]} available).",
            )

        pool[token_type] -= amount

        entry = {
            "name": target,
            "location": target,
            "tokens": amount,
            "description": args.get("description", ""),
        }

        if token_type == "threat":
            self.state["threats"].append(entry)
        else:
            self.state["rewards"].append(entry)

        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**{amount} {token_type} token(s)** spent on '{target}'\n"
                f"{args.get('description', '')}\n"
                f"**{pool_name.title()} Pool:** {pool['threat']}T / {pool['reward']}R"
            ),
        )

    def _exploration_turn(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        if self.state["turns_remaining"] <= 0:
            return ToolResult(success=False, error="No exploration turns remaining.")

        self.state["turns_remaining"] -= 1
        if self.state["phase"] in ("foundation", "planning"):
            self.state["phase"] = "discovery"

        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Exploration Turn** ({self.state['turns_remaining']} remaining)\n"
                f"{args['description']}"
            ),
        )

    def _tremor_turn(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        if self.state["tremors_remaining"] <= 0:
            return ToolResult(success=False, error="No tremors remaining.")

        roll = random.randint(1, 20)
        tremor = TREMOR_TABLE.get(roll, TREMOR_TABLE[1])

        self.state["tremors_remaining"] -= 1
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Tremor! (d20 = {roll})**\n"
                f"{tremor['text']}\n"
                f"Effect: {tremor['effect']}\n"
                f"Tremors remaining: {self.state['tremors_remaining']}"
            ),
        )

    def _heart_turn(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        heart = self.state["heart_pool"]
        total_tokens = heart["threat"] + heart["reward"]

        # Create the heart chamber
        room = {
            "name": args["name"],
            "type": "heart",
            "connected_to": [],
        }
        self.state["architecture"].append(room)

        if heart["threat"] > 0:
            self.state["threats"].append({
                "name": f"Heart Guardian",
                "location": args["name"],
                "tokens": heart["threat"],
            })

        if heart["reward"] > 0:
            self.state["rewards"].append({
                "name": f"Heart Treasure",
                "location": args["name"],
                "tokens": heart["reward"],
            })

        # Clear heart pool
        self.state["heart_pool"] = {"threat": 0, "reward": 0}
        self.state["phase"] = "cleanup"

        # Write to fiction tree
        self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=self.state["history_root_id"],
            title=args["name"],
            zoom_level="event",
            tone="dark",
            source_system="ex_umbra",
            summary=f"Heart chamber: {args['description'][:100]}",
            content=f"{args['description']}\n\nTokens spent: {total_tokens} ({heart['threat']}T, {heart['reward']}R)",
        ))

        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Heart Chamber Created:** {args['name']}\n"
                f"{args['description']}\n"
                f"**Tokens Spent:** {heart['threat']} threat, {heart['reward']} reward\n\n"
                f"The dungeon's heart beats. Proceed to `cleanup`."
            ),
        )

    def _get_state(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active dungeon.")

        s = self.state
        lines = [
            f"**{s['dungeon_name']}** — {s['phase']}",
            f"**Size:** {s['size']} | **Difficulty:** {s['difficulty_name']}",
            f"**Aspects:** {', '.join(s['aspects'])}",
            f"**Dungeon Pool:** {s['dungeon_pool']['threat']}T / {s['dungeon_pool']['reward']}R",
            f"**Heart Pool:** {s['heart_pool']['threat']}T / {s['heart_pool']['reward']}R",
            f"**Turns Remaining:** {s['turns_remaining']}/{s['turn_clock']}",
            f"**Tremors Remaining:** {s['tremors_remaining']}",
        ]

        if s["architecture"]:
            lines.append(f"\n**Architecture ({len(s['architecture'])}):**")
            for r in s["architecture"]:
                conns = f" → {', '.join(r['connected_to'])}" if r.get("connected_to") else ""
                lines.append(f"  - {r['name']} ({r['type']}){conns}")

        if s["threats"]:
            lines.append(f"\n**Threats ({len(s['threats'])}):**")
            for t in s["threats"]:
                lines.append(f"  - {t['name']} at {t['location']} ({t['tokens']} tokens)")

        if s["rewards"]:
            lines.append(f"\n**Rewards ({len(s['rewards'])}):**")
            for r in s["rewards"]:
                lines.append(f"  - {r['name']} at {r['location']} ({r['tokens']} tokens)")

        return ToolResult(success=True, data="\n".join(lines))

    def _cleanup(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

        self.state["phase"] = "cleanup"
        notes = args.get("notes", "")
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Cleanup Phase**\n{notes}\n\nUse `end_dungeon` to finalize.",
        )

    def _end(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active dungeon to end.")

        self.state["phase"] = "ended"
        self._save_state(self.state)

        s = self.state
        lines = [
            f"**{s['dungeon_name']} — Dungeon Finalized**\n",
            f"**Size:** {s['size']} | **Difficulty:** {s['difficulty_name']}",
            f"**Rooms:** {len(s['architecture'])}",
            f"**Threats:** {len(s['threats'])}",
            f"**Rewards:** {len(s['rewards'])}",
            f"**Unspent Dungeon Pool:** {s['dungeon_pool']['threat']}T / {s['dungeon_pool']['reward']}R",
        ]
        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Virtual Players
    # -------------------------------------------------------------------

    def _add_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

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
            return ToolResult(success=False, error="No active dungeon.")
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
            "dungeon_name": self.state.get("dungeon_name", ""),
            "phase": self.state.get("phase", ""),
            "size": self.state.get("size", 0),
            "difficulty_name": self.state.get("difficulty_name", ""),
            "aspects": self.state.get("aspects", []),
            "architecture": self.state.get("architecture", []),
            "threats": self.state.get("threats", []),
            "rewards": self.state.get("rewards", []),
            "dungeon_pool": self.state.get("dungeon_pool", {}),
            "heart_pool": self.state.get("heart_pool", {}),
            "turns_remaining": self.state.get("turns_remaining", 0),
            "turn_clock": self.state.get("turn_clock", 0),
            "tremors_remaining": self.state.get("tremors_remaining", 0),
        }

        # Tell the VP engine what actions are available
        available = [
            "play_dungeon_card", "spend_tokens", "exploration_turn",
            "tremor_turn", "heart_turn", "set_foundation",
        ]
        if decision_type == "create":
            decision_type = "play_dungeon_card"  # sensible default

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
        if decision_type == "play_dungeon_card":
            aspect = content.get("aspect", "")
            if not aspect:
                aspects = self.state.get("aspects", []) if self.state else []
                aspect = aspects[0] if aspects else "unknown"
            return self._play_card({
                "aspect": aspect,
                "room_name": content.get("room_name", content.get("name", "Unnamed Room")),
                "room_type": content.get("room_type", content.get("type", "chamber")),
                "description": content.get("description", ""),
                "connected_to": json.dumps(content.get("connected_to", [])),
            })
        elif decision_type == "spend_tokens":
            return self._spend_tokens({
                "pool": content.get("pool", "dungeon"),
                "token_type": content.get("token_type", "threat"),
                "amount": content.get("amount", 1),
                "target": content.get("target", content.get("name", "")),
                "description": content.get("description", ""),
            })
        elif decision_type == "exploration_turn":
            return self._exploration_turn({
                "description": content.get("description", "The dungeon expands."),
            })
        elif decision_type == "tremor_turn":
            return self._tremor_turn()
        elif decision_type == "heart_turn":
            return self._heart_turn({
                "name": content.get("name", content.get("title", "The Heart")),
                "description": content.get("description", "The dungeon's heart."),
            })
        elif decision_type == "set_foundation":
            return self._set_foundation({
                "guide_lines": json.dumps(content.get("guide_lines", [])),
                "initial_rooms": json.dumps(content.get("initial_rooms", [])),
            })
        else:
            return ToolResult(
                success=False,
                error=f"Unknown decision type '{decision_type}' — could not auto-execute.",
            )

    def _remove_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active dungeon.")

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
