"""Microscope session MCP server.

Provides tools for running a Microscope worldbuilding game,
managing session state and writing to the Fiction Tree.
"""

import json
import uuid
from datetime import datetime
from typing import Any

from ...config import CAMPAIGNS_DIR
from ...storage.fiction_tree import FictionTreeStore, FictionNode
from ...mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult


class MicroscopeSessionServer(MCPServer):
    """MCP server for Microscope worldbuilding sessions.

    Campaign-scoped, stateful server managing the Microscope procedure
    and writing narrative content to the Fiction Tree.
    """

    def __init__(self, campaign_id: str, llm: Any = None):
        self.campaign_id = campaign_id
        self.llm = llm
        self._fiction_store: FictionTreeStore | None = None
        self._state: dict[str, Any] | None = None
        self._state_path = CAMPAIGNS_DIR / campaign_id / "microscope_state.json"
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
                name="microscope_setup",
                description=(
                    "Initialize a Microscope session: set the Big Picture statement, "
                    "create the two bookend Periods (start and end of history), "
                    "and optionally initialize the Palette."
                ),
                parameters=[
                    ToolParameter(name="big_picture", type="string",
                                  description="The Big Picture — one-sentence premise of the history."),
                    ToolParameter(name="start_period", type="string",
                                  description="Title of the starting Period (beginning of history)."),
                    ToolParameter(name="start_tone", type="string",
                                  description="Tone of starting period: 'light' or 'dark'."),
                    ToolParameter(name="end_period", type="string",
                                  description="Title of the ending Period (end of history)."),
                    ToolParameter(name="end_tone", type="string",
                                  description="Tone of ending period: 'light' or 'dark'."),
                    ToolParameter(name="players", type="string",
                                  description="JSON array of player names.", required=False, default='["Player 1"]'),
                ],
            ),
            ToolDef(
                name="set_palette",
                description="Add an entry to the Palette: a Yes (allowed) or No (banned) element.",
                parameters=[
                    ToolParameter(name="entry", type="string", description="The palette entry text."),
                    ToolParameter(name="allowed", type="boolean",
                                  description="True for Yes (allowed), False for No (banned)."),
                ],
            ),
            ToolDef(
                name="get_palette",
                description="View the current Palette — Yes (allowed) and No (banned) elements.",
                parameters=[],
            ),
            ToolDef(
                name="set_focus",
                description="Set the current Focus for this round of play.",
                parameters=[
                    ToolParameter(name="focus", type="string", description="The Focus topic for this round."),
                ],
            ),
            ToolDef(
                name="create_period",
                description=(
                    "Create a new Period in the history. Periods are broad eras. "
                    "Specify where to place it relative to existing periods."
                ),
                parameters=[
                    ToolParameter(name="title", type="string", description="Period title."),
                    ToolParameter(name="tone", type="string", description="'light' or 'dark'."),
                    ToolParameter(name="description", type="string",
                                  description="Brief description of the period.", required=False, default=""),
                    ToolParameter(name="after_period_id", type="string",
                                  description="Place after this period ID. Omit to append at end.",
                                  required=False),
                ],
            ),
            ToolDef(
                name="create_event",
                description="Create a new Event within a Period. Events are specific happenings.",
                parameters=[
                    ToolParameter(name="period_id", type="string", description="Parent period node ID."),
                    ToolParameter(name="title", type="string", description="Event title."),
                    ToolParameter(name="tone", type="string", description="'light' or 'dark'."),
                    ToolParameter(name="description", type="string",
                                  description="Description of what happened.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="create_scene",
                description=(
                    "Start a Scene under an Event. Scenes have a question to answer, "
                    "a setting, and characters."
                ),
                parameters=[
                    ToolParameter(name="event_id", type="string", description="Parent event node ID."),
                    ToolParameter(name="question", type="string",
                                  description="The question this scene will answer."),
                    ToolParameter(name="setting", type="string",
                                  description="Where and when the scene takes place.", required=False, default=""),
                    ToolParameter(name="characters", type="string",
                                  description="JSON array of character names in the scene.",
                                  required=False, default="[]"),
                ],
            ),
            ToolDef(
                name="resolve_scene",
                description="Answer the scene's question and record what happened.",
                parameters=[
                    ToolParameter(name="scene_id", type="string", description="The scene node ID."),
                    ToolParameter(name="answer", type="string", description="The answer to the scene's question."),
                    ToolParameter(name="description", type="string",
                                  description="What happened in the scene.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="add_legacy",
                description="Create or update a Legacy — a recurring element tracked across history.",
                parameters=[
                    ToolParameter(name="name", type="string", description="Legacy name."),
                    ToolParameter(name="legacy_type", type="string",
                                  description="'character', 'place', or 'institution'."),
                    ToolParameter(name="owner", type="string",
                                  description="Player who champions this Legacy.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="browse_history",
                description="View the Microscope timeline — all periods in order, optionally expanding events and scenes.",
                parameters=[
                    ToolParameter(name="expand", type="boolean",
                                  description="If true, show events and scenes under each period.",
                                  required=False, default=False),
                ],
            ),
            ToolDef(
                name="get_microscope_state",
                description="Get current Microscope session state: phase, focus, palette, legacies, turn info.",
                parameters=[],
            ),
            ToolDef(
                name="end_microscope",
                description="End the Microscope session.",
                parameters=[],
            ),
            # --- Seeds & Oracles ---
            ToolDef(
                name="list_seeds",
                description="List available pre-written story seeds for Microscope.",
                parameters=[],
            ),
            ToolDef(
                name="use_seed",
                description="Start a Microscope session from a pre-written story seed.",
                parameters=[
                    ToolParameter(name="seed_name", type="string",
                                  description="Name of the seed (from list_seeds)."),
                    ToolParameter(name="players", type="string",
                                  description="JSON array of player names.",
                                  required=False, default='["Player 1"]'),
                ],
            ),
            ToolDef(
                name="consult_oracle",
                description="Get random inspiration from a themed oracle table.",
                parameters=[
                    ToolParameter(name="oracle_name", type="string",
                                  description="Oracle table name (use empty string to list all)."),
                ],
            ),
            # --- Mini-games ---
            ToolDef(
                name="start_chronicle",
                description="Begin a Chronicle mini-game: explore one Legacy across time by creating connected events.",
                parameters=[
                    ToolParameter(name="legacy_name", type="string",
                                  description="The Legacy to explore."),
                ],
            ),
            ToolDef(
                name="chronicle_entry",
                description="Add an entry to the active Chronicle — an event involving the Legacy.",
                parameters=[
                    ToolParameter(name="period_id", type="string",
                                  description="Period to place this entry in."),
                    ToolParameter(name="title", type="string", description="Event title."),
                    ToolParameter(name="tone", type="string", description="'light' or 'dark'."),
                    ToolParameter(name="description", type="string",
                                  description="What happens with the Legacy.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="start_echo",
                description="Begin an Echo mini-game: replay an existing Scene from a different perspective to explore an alternative answer.",
                parameters=[
                    ToolParameter(name="scene_id", type="string",
                                  description="The original scene node ID to echo."),
                    ToolParameter(name="new_perspective", type="string",
                                  description="Whose perspective or what angle to explore."),
                ],
            ),
            ToolDef(
                name="resolve_echo",
                description="Resolve the active Echo with an alternative answer.",
                parameters=[
                    ToolParameter(name="answer", type="string",
                                  description="The alternative answer from this perspective."),
                    ToolParameter(name="description", type="string",
                                  description="What happened differently.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="start_union",
                description="Begin a Union mini-game: explore the relationship between two characters across time.",
                parameters=[
                    ToolParameter(name="character_a", type="string",
                                  description="First character name."),
                    ToolParameter(name="character_b", type="string",
                                  description="Second character name."),
                ],
            ),
            ToolDef(
                name="union_moment",
                description="Create a moment in the active Union — a scene between the two characters.",
                parameters=[
                    ToolParameter(name="period_id", type="string",
                                  description="Period to place this moment in."),
                    ToolParameter(name="title", type="string", description="Moment title."),
                    ToolParameter(name="tone", type="string", description="'light' or 'dark'."),
                    ToolParameter(name="description", type="string",
                                  description="What happens between the characters.", required=False, default=""),
                ],
            ),
            ToolDef(
                name="end_minigame",
                description="End the current mini-game (Chronicle, Echo, or Union) and return to normal play.",
                parameters=[],
            ),
            # --- Virtual Players ---
            ToolDef(
                name="add_virtual_player",
                description="Register an AI-driven virtual player with a personality archetype or custom traits.",
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
                description="Generate and execute a virtual player's turn using AI.",
                parameters=[
                    ToolParameter(name="player_name", type="string",
                                  description="Name of the virtual player to act."),
                    ToolParameter(name="decision_type", type="string",
                                  description="Type of decision: 'create_period', 'create_event', 'set_focus', etc.",
                                  required=False, default="create"),
                ],
            ),
            ToolDef(
                name="remove_virtual_player",
                description="Remove a virtual player from the session.",
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
            if name == "microscope_setup":
                return self._setup(args)
            elif name == "set_palette":
                return self._set_palette(args)
            elif name == "get_palette":
                return self._get_palette()
            elif name == "set_focus":
                return self._set_focus(args)
            elif name == "create_period":
                return self._create_period(args)
            elif name == "create_event":
                return self._create_event(args)
            elif name == "create_scene":
                return self._create_scene(args)
            elif name == "resolve_scene":
                return self._resolve_scene(args)
            elif name == "add_legacy":
                return self._add_legacy(args)
            elif name == "browse_history":
                return self._browse_history(args)
            elif name == "get_microscope_state":
                return self._get_state()
            elif name == "end_microscope":
                return self._end()
            # Seeds & Oracles
            elif name == "list_seeds":
                return self._list_seeds()
            elif name == "use_seed":
                return self._use_seed(args)
            elif name == "consult_oracle":
                return self._consult_oracle(args)
            # Mini-games
            elif name == "start_chronicle":
                return self._start_chronicle(args)
            elif name == "chronicle_entry":
                return self._chronicle_entry(args)
            elif name == "start_echo":
                return self._start_echo(args)
            elif name == "resolve_echo":
                return self._resolve_echo(args)
            elif name == "start_union":
                return self._start_union(args)
            elif name == "union_moment":
                return self._union_moment(args)
            elif name == "end_minigame":
                return self._end_minigame()
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

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------

    def _setup(self, args: dict[str, Any]) -> ToolResult:
        if self.state and self.state.get("phase") != "ended":
            return ToolResult(success=False, error="A Microscope session is already active. End it first.")

        players = args.get("players", '["Player 1"]')
        if isinstance(players, str):
            players = json.loads(players)

        # Create root node in fiction tree
        root = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            title=args["big_picture"],
            zoom_level="era",
            source_system="microscope",
            summary=f"Microscope history: {args['big_picture']}",
        ))

        # Create bookend periods
        start = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=root.id,
            title=args["start_period"],
            zoom_level="period",
            tone=args["start_tone"],
            source_system="microscope",
            sort_order=0,
        ))

        end = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=root.id,
            title=args["end_period"],
            zoom_level="period",
            tone=args["end_tone"],
            source_system="microscope",
            sort_order=1000,
        ))

        state = {
            "big_picture": args["big_picture"],
            "palette_yes": [],
            "palette_no": [],
            "current_focus": None,
            "current_lens": None,
            "legacies": [],
            "turn_order": players,
            "current_turn": 0,
            "phase": "setup",
            "history_root_id": root.id,
        }
        self._save_state(state)

        return ToolResult(
            success=True,
            data=(
                f"**Microscope Session Started**\n\n"
                f"**Big Picture:** {args['big_picture']}\n"
                f"**Start:** {args['start_period']} [{args['start_tone']}]\n"
                f"**End:** {args['end_period']} [{args['end_tone']}]\n"
                f"**Players:** {', '.join(players)}\n\n"
                f"Root ID: {root.id}\n"
                f"Next: Set up the Palette, then begin the First Pass."
            ),
        )

    # -------------------------------------------------------------------
    # Palette
    # -------------------------------------------------------------------

    def _set_palette(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        entry = args["entry"]
        allowed = args.get("allowed", True)
        if isinstance(allowed, str):
            allowed = allowed.lower() in ("true", "1", "yes")

        if allowed:
            self.state["palette_yes"].append(entry)
        else:
            self.state["palette_no"].append(entry)
        self._save_state(self.state)

        label = "YES (allowed)" if allowed else "NO (banned)"
        return ToolResult(
            success=True,
            data=f"**Palette {label}:** {entry}",
        )

    def _get_palette(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        yes_list = self.state.get("palette_yes", [])
        no_list = self.state.get("palette_no", [])

        if not yes_list and not no_list:
            return ToolResult(success=True, data="**Palette** is empty — no elements added yet.")

        lines = ["**Palette**"]
        if yes_list:
            lines.append("\n**YES (allowed):**")
            for entry in yes_list:
                lines.append(f"  - {entry}")
        if no_list:
            lines.append("\n**NO (banned):**")
            for entry in no_list:
                lines.append(f"  - {entry}")

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # Focus
    # -------------------------------------------------------------------

    def _set_focus(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        self.state["current_focus"] = args["focus"]
        if self.state["phase"] == "setup":
            self.state["phase"] = "play"
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Focus set:** {args['focus']}",
        )

    # -------------------------------------------------------------------
    # Create Period
    # -------------------------------------------------------------------

    def _create_period(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        root_id = self.state["history_root_id"]
        tone = args["tone"]
        if tone not in ("light", "dark"):
            return ToolResult(success=False, error="Tone must be 'light' or 'dark'.")

        # Determine sort order
        siblings = self.fiction_store.get_children(root_id)
        after_id = args.get("after_period_id")
        if after_id:
            after_order = 0
            for s in siblings:
                if s.id == after_id:
                    after_order = s.sort_order
                    break
            sort_order = after_order + 1
            # Shift later siblings
            for s in siblings:
                if s.sort_order >= sort_order and s.id != after_id:
                    self.fiction_store.update_node(s.id, sort_order=s.sort_order + 2)
        else:
            # Place before the last period (the end bookend)
            if siblings:
                sort_order = siblings[-1].sort_order
                self.fiction_store.update_node(siblings[-1].id, sort_order=sort_order + 2)
            else:
                sort_order = 0

        node = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=root_id,
            title=args["title"],
            zoom_level="period",
            tone=tone,
            summary=args.get("description", ""),
            content=args.get("description", ""),
            source_system="microscope",
            sort_order=sort_order,
            palette_yes=self.state.get("palette_yes", []),
            palette_no=self.state.get("palette_no", []),
        ))

        if self.state["phase"] == "setup":
            self.state["phase"] = "first_pass"
        self._advance_turn()
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Period Created:** {args['title']} [{tone}] — ID: {node.id}",
        )

    # -------------------------------------------------------------------
    # Create Event
    # -------------------------------------------------------------------

    def _create_event(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        tone = args["tone"]
        if tone not in ("light", "dark"):
            return ToolResult(success=False, error="Tone must be 'light' or 'dark'.")

        period = self.fiction_store.get_node(args["period_id"])
        if period is None:
            return ToolResult(success=False, error=f"Period '{args['period_id']}' not found.")

        node = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=args["period_id"],
            title=args["title"],
            zoom_level="event",
            tone=tone,
            summary=args.get("description", ""),
            content=args.get("description", ""),
            source_system="microscope",
        ))

        self._advance_turn()
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Event Created:** {args['title']} [{tone}] under {period.title} — ID: {node.id}",
        )

    # -------------------------------------------------------------------
    # Create / Resolve Scene
    # -------------------------------------------------------------------

    def _create_scene(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        event = self.fiction_store.get_node(args["event_id"])
        if event is None:
            return ToolResult(success=False, error=f"Event '{args['event_id']}' not found.")

        characters = args.get("characters", "[]")
        if isinstance(characters, str):
            characters = json.loads(characters)

        content_parts = [f"**Question:** {args['question']}"]
        if args.get("setting"):
            content_parts.append(f"**Setting:** {args['setting']}")
        if characters:
            content_parts.append(f"**Characters:** {', '.join(characters)}")

        node = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=args["event_id"],
            title=f"Scene: {args['question'][:60]}",
            zoom_level="scene",
            summary=args["question"],
            content="\n".join(content_parts),
            source_system="microscope",
            tags=characters,
        ))

        return ToolResult(
            success=True,
            data=(
                f"**Scene Started** under {event.title}\n"
                f"**Question:** {args['question']}\n"
                f"**ID:** {node.id}\n"
                f"Play out the scene to answer the question."
            ),
        )

    def _resolve_scene(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        scene = self.fiction_store.get_node(args["scene_id"])
        if scene is None:
            return ToolResult(success=False, error=f"Scene '{args['scene_id']}' not found.")

        new_content = scene.content or ""
        new_content += f"\n\n**Answer:** {args['answer']}"
        if args.get("description"):
            new_content += f"\n{args['description']}"

        self.fiction_store.update_node(
            args["scene_id"],
            content=new_content,
            tone="light" if "light" in args.get("answer", "").lower() else scene.tone,
        )

        self._advance_turn()
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Scene Resolved:** {scene.title}\n**Answer:** {args['answer']}",
        )

    # -------------------------------------------------------------------
    # Legacy
    # -------------------------------------------------------------------

    def _add_legacy(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        legacy = {
            "name": args["name"],
            "type": args["legacy_type"],
            "owner": args.get("owner", ""),
        }

        # Update existing or add new
        for i, existing in enumerate(self.state["legacies"]):
            if existing["name"] == args["name"]:
                self.state["legacies"][i] = legacy
                self._save_state(self.state)
                return ToolResult(success=True, data=f"**Legacy Updated:** {args['name']} ({args['legacy_type']})")

        self.state["legacies"].append(legacy)
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Legacy Created:** {args['name']} ({args['legacy_type']})",
        )

    # -------------------------------------------------------------------
    # Browse History
    # -------------------------------------------------------------------

    def _browse_history(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        root_id = self.state["history_root_id"]
        expand = args.get("expand", False)
        if isinstance(expand, str):
            expand = expand.lower() in ("true", "1", "yes")

        periods = self.fiction_store.get_children(root_id)
        if not periods:
            return ToolResult(success=True, data="No periods in the timeline yet.")

        lines = [f"**{self.state['big_picture']}**\n"]

        for i, period in enumerate(periods):
            tone_marker = f"[{period.tone}]" if period.tone else ""
            lines.append(f"{i+1}. **{period.title}** {tone_marker} — ID: {period.id}")
            if period.summary:
                lines.append(f"   {period.summary[:120]}")

            if expand:
                events = self.fiction_store.get_children(period.id)
                for event in events:
                    etone = f"[{event.tone}]" if event.tone else ""
                    lines.append(f"   - {event.title} {etone} — ID: {event.id}")
                    scenes = self.fiction_store.get_children(event.id)
                    for scene in scenes:
                        lines.append(f"     - {scene.title} — ID: {scene.id}")

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------

    def _get_state(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active Microscope session.")

        lines = [
            f"**Microscope Session**",
            f"**Phase:** {self.state['phase']}",
            f"**Big Picture:** {self.state['big_picture']}",
        ]

        if self.state["current_focus"]:
            lines.append(f"**Focus:** {self.state['current_focus']}")

        if self.state["palette_yes"]:
            lines.append(f"**Palette YES:** {', '.join(self.state['palette_yes'])}")
        if self.state["palette_no"]:
            lines.append(f"**Palette NO:** {', '.join(self.state['palette_no'])}")

        if self.state["legacies"]:
            lines.append("\n**Legacies:**")
            for leg in self.state["legacies"]:
                owner = f" (championed by {leg['owner']})" if leg.get("owner") else ""
                lines.append(f"  - {leg['name']} ({leg['type']}){owner}")

        players = self.state.get("turn_order", [])
        current = self.state.get("current_turn", 0)
        if players:
            current_player = players[current % len(players)]
            lines.append(f"\n**Current Turn:** {current_player} (player {current % len(players) + 1}/{len(players)})")

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # End
    # -------------------------------------------------------------------

    def _end(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=True, data="No active Microscope session to end.")

        self.state["phase"] = "ended"
        self._save_state(self.state)

        root_id = self.state["history_root_id"]
        summary = self.fiction_store.get_tree_summary(root_id)

        return ToolResult(
            success=True,
            data=f"**Microscope Session Ended**\n\n**Timeline Summary:**\n{summary}",
        )

    # -------------------------------------------------------------------
    # Seeds
    # -------------------------------------------------------------------

    def _list_seeds(self) -> ToolResult:
        from .seeds import SEEDS
        lines = ["**Available Story Seeds:**\n"]
        for key, seed in SEEDS.items():
            lines.append(f"- **{key}**: {seed.big_picture}")
        return ToolResult(success=True, data="\n".join(lines))

    def _use_seed(self, args: dict[str, Any]) -> ToolResult:
        from .seeds import SEEDS
        seed_name = args.get("seed_name", "")
        if seed_name not in SEEDS:
            return ToolResult(success=False, error=f"Unknown seed: '{seed_name}'. Use list_seeds to see options.")

        seed = SEEDS[seed_name]
        # Delegate to _setup
        result = self._setup({
            "big_picture": seed.big_picture,
            "start_period": seed.start_period,
            "start_tone": seed.start_tone,
            "end_period": seed.end_period,
            "end_tone": seed.end_tone,
            "players": args.get("players", '["Player 1"]'),
        })
        if not result.success:
            return result

        # Apply seed palette
        for entry in seed.palette_yes:
            self.state["palette_yes"].append(entry)
        for entry in seed.palette_no:
            self.state["palette_no"].append(entry)
        if seed.suggested_focus:
            self.state["current_focus"] = seed.suggested_focus
        self._save_state(self.state)

        extra = ""
        if seed.suggested_focus:
            extra = f"\n**Suggested Focus:** {seed.suggested_focus}"
        return ToolResult(
            success=True,
            data=f"{result.data}\n\n**Seed:** {seed.name}{extra}",
        )

    # -------------------------------------------------------------------
    # Oracles
    # -------------------------------------------------------------------

    def _consult_oracle(self, args: dict[str, Any]) -> ToolResult:
        from .oracles import ORACLES
        oracle_name = args.get("oracle_name", "")
        if not oracle_name:
            lines = ["**Available Oracle Tables:**\n"]
            for key, oracle in ORACLES.items():
                lines.append(f"- **{key}** ({oracle.category}): {oracle.name}")
            return ToolResult(success=True, data="\n".join(lines))

        if oracle_name not in ORACLES:
            return ToolResult(success=False, error=f"Unknown oracle: '{oracle_name}'. Pass empty string to list all.")

        oracle = ORACLES[oracle_name]
        entry = oracle.consult()
        return ToolResult(
            success=True,
            data=f"**{oracle.name}:** {entry}",
        )

    # -------------------------------------------------------------------
    # Mini-games: Chronicle
    # -------------------------------------------------------------------

    def _start_chronicle(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")
        if self.state.get("active_minigame"):
            return ToolResult(success=False, error=f"A {self.state['active_minigame']} mini-game is already active. End it first.")

        legacy_name = args.get("legacy_name", "")
        if not legacy_name:
            return ToolResult(success=False, error="Legacy name is required.")

        self.state["active_minigame"] = "chronicle"
        self.state["minigame_state"] = {
            "legacy_name": legacy_name,
            "entries": [],
        }
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Chronicle Started:** Exploring the Legacy of *{legacy_name}* across time.",
        )

    def _chronicle_entry(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")
        if self.state.get("active_minigame") != "chronicle":
            return ToolResult(success=False, error="No active Chronicle. Start one with start_chronicle.")

        tone = args.get("tone", "light")
        if tone not in ("light", "dark"):
            return ToolResult(success=False, error="Tone must be 'light' or 'dark'.")

        period = self.fiction_store.get_node(args["period_id"])
        if period is None:
            return ToolResult(success=False, error=f"Period '{args['period_id']}' not found.")

        legacy_name = self.state["minigame_state"]["legacy_name"]

        node = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=args["period_id"],
            title=args["title"],
            zoom_level="event",
            tone=tone,
            summary=args.get("description", ""),
            content=args.get("description", ""),
            source_system="microscope",
            tags=[f"chronicle:{legacy_name}"],
        ))

        self.state["minigame_state"]["entries"].append(node.id)
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Chronicle Entry:** {args['title']} [{tone}] — Legacy: {legacy_name} — ID: {node.id}",
        )

    # -------------------------------------------------------------------
    # Mini-games: Echo
    # -------------------------------------------------------------------

    def _start_echo(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")
        if self.state.get("active_minigame"):
            return ToolResult(success=False, error=f"A {self.state['active_minigame']} mini-game is already active. End it first.")

        scene_id = args.get("scene_id", "")
        scene = self.fiction_store.get_node(scene_id)
        if scene is None:
            return ToolResult(success=False, error=f"Scene '{scene_id}' not found.")

        perspective = args.get("new_perspective", "")

        self.state["active_minigame"] = "echo"
        self.state["minigame_state"] = {
            "original_scene_id": scene_id,
            "original_title": scene.title,
            "perspective": perspective,
        }
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=(
                f"**Echo Started:** Replaying *{scene.title}*\n"
                f"**New Perspective:** {perspective}\n"
                f"**Original Question:** {scene.summary}"
            ),
        )

    def _resolve_echo(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")
        if self.state.get("active_minigame") != "echo":
            return ToolResult(success=False, error="No active Echo. Start one with start_echo.")

        mg = self.state["minigame_state"]
        original_scene = self.fiction_store.get_node(mg["original_scene_id"])
        parent_id = original_scene.parent_id if original_scene else None

        if not parent_id:
            return ToolResult(success=False, error="Cannot find parent event for the original scene.")

        answer = args.get("answer", "")
        description = args.get("description", "")

        node = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=parent_id,
            title=f"Echo: {mg['original_title']}",
            zoom_level="scene",
            summary=f"Echo from {mg['perspective']}: {answer}",
            content=f"**Perspective:** {mg['perspective']}\n**Answer:** {answer}\n{description}",
            source_system="microscope",
            tags=[f"echo:{mg['original_scene_id']}"],
        ))

        # End the mini-game
        self.state["active_minigame"] = None
        self.state["minigame_state"] = {}
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Echo Resolved:** {answer}\n**ID:** {node.id}",
        )

    # -------------------------------------------------------------------
    # Mini-games: Union
    # -------------------------------------------------------------------

    def _start_union(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")
        if self.state.get("active_minigame"):
            return ToolResult(success=False, error=f"A {self.state['active_minigame']} mini-game is already active. End it first.")

        char_a = args.get("character_a", "")
        char_b = args.get("character_b", "")
        if not char_a or not char_b:
            return ToolResult(success=False, error="Both character_a and character_b are required.")

        self.state["active_minigame"] = "union"
        self.state["minigame_state"] = {
            "character_a": char_a,
            "character_b": char_b,
            "moments": [],
        }
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Union Started:** Exploring the relationship between *{char_a}* and *{char_b}*.",
        )

    def _union_moment(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")
        if self.state.get("active_minigame") != "union":
            return ToolResult(success=False, error="No active Union. Start one with start_union.")

        tone = args.get("tone", "light")
        if tone not in ("light", "dark"):
            return ToolResult(success=False, error="Tone must be 'light' or 'dark'.")

        period = self.fiction_store.get_node(args["period_id"])
        if period is None:
            return ToolResult(success=False, error=f"Period '{args['period_id']}' not found.")

        mg = self.state["minigame_state"]
        char_a = mg["character_a"]
        char_b = mg["character_b"]

        node = self.fiction_store.add_node(FictionNode(
            campaign_id=self.campaign_id,
            parent_id=args["period_id"],
            title=args["title"],
            zoom_level="event",
            tone=tone,
            summary=args.get("description", ""),
            content=args.get("description", ""),
            source_system="microscope",
            tags=[f"union:{char_a}:{char_b}"],
        ))

        mg["moments"].append(node.id)
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**Union Moment:** {args['title']} [{tone}] — {char_a} & {char_b} — ID: {node.id}",
        )

    # -------------------------------------------------------------------
    # End Mini-game
    # -------------------------------------------------------------------

    def _end_minigame(self) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        active = self.state.get("active_minigame")
        if not active:
            return ToolResult(success=True, data="No active mini-game to end.")

        self.state["active_minigame"] = None
        self.state["minigame_state"] = {}
        self._save_state(self.state)

        return ToolResult(
            success=True,
            data=f"**{active.capitalize()} Ended.** Returning to normal Microscope play.",
        )

    # -------------------------------------------------------------------
    # Virtual Players
    # -------------------------------------------------------------------

    def _add_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

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
            return ToolResult(success=False, error="No active Microscope session.")
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
            "big_picture": self.state.get("big_picture", ""),
            "current_focus": self.state.get("current_focus", ""),
            "palette_yes": self.state.get("palette_yes", []),
            "palette_no": self.state.get("palette_no", []),
            "phase": self.state.get("phase", ""),
            "legacies": self.state.get("legacies", []),
        }

        # Add period/event tree so the VP knows what exists
        root_id = self.state.get("history_root_id")
        if root_id:
            periods = self.fiction_store.get_children(root_id)
            period_list = []
            for p in periods:
                period_info: dict[str, Any] = {
                    "id": p.id, "title": p.title, "tone": p.tone,
                }
                events = self.fiction_store.get_children(p.id)
                if events:
                    period_info["events"] = [
                        {"id": e.id, "title": e.title, "tone": e.tone}
                        for e in events
                    ]
                period_list.append(period_info)
            game_state["periods"] = period_list

        # Tell the VP engine what actions are available
        available = ["create_period", "create_event", "add_legacy", "set_focus"]
        if decision_type == "create":
            decision_type = "create_period"  # sensible default

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
        if decision_type == "create_period":
            return self._create_period({
                "title": content.get("title", content.get("name", "Untitled Period")),
                "tone": content.get("tone", "light"),
                "description": content.get("description", ""),
                "after_period_id": content.get("after_period_id"),
            })
        elif decision_type == "create_event":
            # VP must specify which period — pick first match or use provided ID
            period_id = content.get("period_id", "")
            if not period_id:
                period_id = self._find_period_by_title(content.get("period", ""))
            if not period_id:
                return ToolResult(success=False, error="No period_id specified for event.")
            return self._create_event({
                "period_id": period_id,
                "title": content.get("title", content.get("name", "Untitled Event")),
                "tone": content.get("tone", "light"),
                "description": content.get("description", ""),
            })
        elif decision_type == "set_focus":
            return self._set_focus({
                "focus": content.get("focus", content.get("title", "")),
            })
        elif decision_type == "add_legacy":
            return self._add_legacy({
                "name": content.get("name", ""),
                "legacy_type": content.get("legacy_type", content.get("type", "character")),
                "owner": content.get("owner", ""),
            })
        elif decision_type == "set_palette":
            return self._set_palette({
                "element": content.get("element", ""),
                "allowed": content.get("allowed", True),
            })
        else:
            return ToolResult(
                success=False,
                error=f"Unknown decision type '{decision_type}' — could not auto-execute.",
            )

    def _find_period_by_title(self, title: str) -> str:
        """Find a period ID by fuzzy title match."""
        if not title:
            return ""
        root_id = self.state.get("history_root_id", "")
        if not root_id:
            return ""
        periods = self.fiction_store.get_children(root_id)
        title_lower = title.lower()
        for p in periods:
            if p.title.lower() == title_lower:
                return p.id
        # Substring match
        for p in periods:
            if title_lower in p.title.lower() or p.title.lower() in title_lower:
                return p.id
        return ""

    def _remove_virtual_player(self, args: dict[str, Any]) -> ToolResult:
        if not self.state:
            return ToolResult(success=False, error="No active Microscope session.")

        player_name = args.get("player_name", "")
        vps = self.state.get("virtual_players", [])

        for i, vp in enumerate(vps):
            if vp["name"] == player_name:
                vps.pop(i)
                self._save_state(self.state)
                return ToolResult(success=True, data=f"**Virtual Player Removed:** {player_name}")

        return ToolResult(success=False, error=f"Virtual player '{player_name}' not found.")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _advance_turn(self) -> None:
        if self.state:
            self.state["current_turn"] = self.state.get("current_turn", 0) + 1

    def close(self) -> None:
        if self._fiction_store:
            self._fiction_store.close()
            self._fiction_store = None
