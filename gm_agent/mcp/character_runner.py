"""Character Runner MCP server for NPC/monster/player embodiment."""

from typing import Any

from ..config import CAMPAIGNS_DIR
from ..models.base import LLMBackend, Message
from ..storage.characters import CharacterStore
from ..storage.dialogue import DialogueStore
from ..storage.knowledge import KnowledgeStore
from ..storage.factions import FactionStore
from ..storage.locations import LocationStore
from ..storage.session import session_store
from ..storage.schemas import CharacterProfile
from .base import MCPServer, ToolDef, ToolParameter, ToolResult

# System prompts for different character types
NPC_SYSTEM_PROMPT = """You are embodying an NPC in a Pathfinder 2E tabletop RPG.

Your character:
- Name: {name}
- Personality: {personality}
- Speech patterns: {speech_patterns}

What you know:
{knowledge}

Your goals:
{goals}

{secrets_section}

Stay completely in character. Respond as this NPC would, based on their personality, knowledge, and goals. Use their speech patterns naturally. Do not break character or mention game mechanics unless the NPC would.

If asked about something you don't know, respond as the character would - they may deflect, lie, admit ignorance, or redirect the conversation based on their personality."""

MONSTER_SYSTEM_PROMPT = """You are making decisions for a creature in a Pathfinder 2E encounter.

Creature: {name}
Intelligence: {intelligence}
Personality/Nature: {personality}

Instincts:
{instincts}

Goals:
{goals}

Morale: {morale}

Based on the creature's intelligence and instincts, determine what action it would take. Consider:
- Animal intelligence: Act on instinct, no complex tactics
- Low intelligence: Simple tactics, easily tricked
- Average intelligence: Basic tactics, can adapt
- High intelligence: Complex tactics, may negotiate or deceive
- Genius: Sophisticated strategy, anticipates player actions

Respond with what the creature does and why, considering its nature and the situation."""

PLAYER_SYSTEM_PROMPT = """You are emulating a player character in a Pathfinder 2E game for playtesting purposes.

Character: {name}
Party Role: {party_role}
Playstyle: {playstyle}
Risk Tolerance: {risk_tolerance}
Decision Making: {decision_making}

Quirks:
{quirks}

Act as this player would during the game. Make decisions that fit their playstyle and personality. Consider:
- What would this player find interesting or fun?
- How cautious or bold would they be?
- What's their typical approach to problems?

Respond with what the player character does, in the style of a player at the table."""


class CharacterRunnerServer(MCPServer):
    """MCP server for character embodiment.

    Provides tools for running NPCs, monsters, and player characters
    with consistent personality and behavior based on their profiles.
    """

    def __init__(self, campaign_id: str, llm: LLMBackend | None = None):
        self.campaign_id = campaign_id
        self.llm = llm
        self._character_store = CharacterStore(campaign_id, base_dir=CAMPAIGNS_DIR)
        self._dialogue_store: DialogueStore | None = None
        self._knowledge_store: KnowledgeStore | None = None
        self._faction_store: FactionStore | None = None
        self._location_store: LocationStore | None = None
        self._tools = self._build_tools()

    @property
    def dialogue(self) -> DialogueStore:
        """Lazy-load dialogue store."""
        if self._dialogue_store is None:
            self._dialogue_store = DialogueStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._dialogue_store

    @property
    def knowledge(self) -> KnowledgeStore:
        """Lazy-load knowledge store."""
        if self._knowledge_store is None:
            self._knowledge_store = KnowledgeStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._knowledge_store

    @property
    def factions(self) -> FactionStore:
        """Lazy-load faction store."""
        if self._faction_store is None:
            self._faction_store = FactionStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._faction_store

    @property
    def locations(self) -> LocationStore:
        """Lazy-load location store."""
        if self._location_store is None:
            self._location_store = LocationStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._location_store

    def _build_tools(self) -> list[ToolDef]:
        """Build the tool definitions."""
        return [
            ToolDef(
                name="run_npc",
                description="Embody an NPC for social interaction. Returns the NPC's response in character based on their personality, knowledge, and goals.",
                parameters=[
                    ToolParameter(
                        name="npc_name",
                        type="string",
                        description="Name of the NPC to embody (must exist in character profiles)",
                    ),
                    ToolParameter(
                        name="player_input",
                        type="string",
                        description="What the player said or did that the NPC is responding to",
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Current scene context (location, situation, other NPCs present)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="run_monster",
                description="Get tactical decisions for a creature based on its intelligence, instincts, and goals.",
                parameters=[
                    ToolParameter(
                        name="monster_name",
                        type="string",
                        description="Name of the monster profile to use",
                    ),
                    ToolParameter(
                        name="situation",
                        type="string",
                        description="Current tactical situation (positions, threats, party status)",
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Additional context (environment, objectives)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="emulate_player",
                description="Act as a player character for playtesting. Makes decisions based on the player's typical playstyle.",
                parameters=[
                    ToolParameter(
                        name="player_name",
                        type="string",
                        description="Name of the player character profile to emulate",
                    ),
                    ToolParameter(
                        name="scenario",
                        type="string",
                        description="The current situation requiring a player decision",
                    ),
                    ToolParameter(
                        name="options",
                        type="string",
                        description="Available options or actions (optional, for specific choices)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="create_character",
                description="Create a new character profile for later use with run_npc, run_monster, or emulate_player.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Character name",
                    ),
                    ToolParameter(
                        name="character_type",
                        type="string",
                        description="Type: 'npc', 'monster', or 'player'",
                    ),
                    ToolParameter(
                        name="personality",
                        type="string",
                        description="Personality description",
                        required=False,
                    ),
                    ToolParameter(
                        name="knowledge",
                        type="string",
                        description="Comma-separated list of things the character knows",
                        required=False,
                    ),
                    ToolParameter(
                        name="goals",
                        type="string",
                        description="Comma-separated list of character goals",
                        required=False,
                    ),
                    ToolParameter(
                        name="secrets",
                        type="string",
                        description="Comma-separated list of character secrets (NPCs)",
                        required=False,
                    ),
                    ToolParameter(
                        name="speech_patterns",
                        type="string",
                        description="How the character speaks (NPCs)",
                        required=False,
                    ),
                    ToolParameter(
                        name="intelligence",
                        type="string",
                        description="Intelligence level for monsters: animal, low, average, high, genius",
                        required=False,
                    ),
                    ToolParameter(
                        name="instincts",
                        type="string",
                        description="Comma-separated instincts for monsters",
                        required=False,
                    ),
                    ToolParameter(
                        name="morale",
                        type="string",
                        description="Morale behavior for monsters",
                        required=False,
                    ),
                    ToolParameter(
                        name="playstyle",
                        type="string",
                        description="Playstyle description for player emulation",
                        required=False,
                    ),
                    ToolParameter(
                        name="risk_tolerance",
                        type="string",
                        description="Risk tolerance: low, medium, high",
                        required=False,
                    ),
                    ToolParameter(
                        name="party_role",
                        type="string",
                        description="Party role for player emulation",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="get_character",
                description="Get a character profile by name.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Character name to look up",
                    ),
                ],
            ),
            ToolDef(
                name="list_characters",
                description="List all character profiles, optionally filtered by type.",
                parameters=[
                    ToolParameter(
                        name="character_type",
                        type="string",
                        description="Filter by type: 'npc', 'monster', or 'player'",
                        required=False,
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        """List all available tools."""
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments."""
        try:
            if name == "run_npc":
                return self._run_npc(args)
            elif name == "run_monster":
                return self._run_monster(args)
            elif name == "emulate_player":
                return self._emulate_player(args)
            elif name == "create_character":
                return self._create_character(args)
            elif name == "get_character":
                return self._get_character(args)
            elif name == "list_characters":
                return self._list_characters(args)
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _run_npc(self, args: dict[str, Any]) -> ToolResult:
        """Embody an NPC for social interaction."""
        if not self.llm:
            return ToolResult(success=False, error="No LLM backend configured")

        npc_name = args.get("npc_name", "")
        player_input = args.get("player_input", "")
        context = args.get("context", "")

        if not npc_name:
            return ToolResult(success=False, error="NPC name is required")
        if not player_input:
            return ToolResult(success=False, error="Player input is required")

        # Get the NPC profile
        profile = self._character_store.get_by_name(npc_name)
        if not profile:
            return ToolResult(
                success=False,
                error=f"NPC '{npc_name}' not found. Create them first with create_character.",
            )

        if profile.character_type != "npc":
            return ToolResult(
                success=False,
                error=f"'{npc_name}' is a {profile.character_type}, not an NPC",
            )

        # Build the system prompt
        system_prompt = self._build_npc_prompt(profile)

        # Build the user message
        user_message = player_input
        if context:
            user_message = f"[Scene: {context}]\n\n{player_input}"

        # Call the LLM
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]

        response = self.llm.chat(messages, tools=None)

        # Log the dialogue
        session = session_store.get_current(self.campaign_id)
        session_id = session.id if session else "no-session"
        turn_number = len(session.turns) if session else None

        self.dialogue.log_dialogue(
            character_id=profile.id,
            character_name=profile.name,
            session_id=session_id,
            content=response.text,
            dialogue_type="statement",
            turn_number=turn_number,
        )

        return ToolResult(
            success=True,
            data=f"[{profile.name}]: {response.text}",
        )

    def _run_monster(self, args: dict[str, Any]) -> ToolResult:
        """Get tactical decisions for a creature."""
        if not self.llm:
            return ToolResult(success=False, error="No LLM backend configured")

        monster_name = args.get("monster_name", "")
        situation = args.get("situation", "")
        context = args.get("context", "")

        if not monster_name:
            return ToolResult(success=False, error="Monster name is required")
        if not situation:
            return ToolResult(success=False, error="Situation description is required")

        # Get the monster profile
        profile = self._character_store.get_by_name(monster_name)
        if not profile:
            return ToolResult(
                success=False,
                error=f"Monster '{monster_name}' not found. Create them first with create_character.",
            )

        if profile.character_type != "monster":
            return ToolResult(
                success=False,
                error=f"'{monster_name}' is a {profile.character_type}, not a monster",
            )

        # Build the system prompt
        system_prompt = self._build_monster_prompt(profile)

        # Build the user message
        user_message = f"Current situation: {situation}"
        if context:
            user_message += f"\n\nAdditional context: {context}"

        # Call the LLM
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]

        response = self.llm.chat(messages, tools=None)

        return ToolResult(
            success=True,
            data=f"[{profile.name} decides]: {response.text}",
        )

    def _emulate_player(self, args: dict[str, Any]) -> ToolResult:
        """Emulate a player for playtesting."""
        if not self.llm:
            return ToolResult(success=False, error="No LLM backend configured")

        player_name = args.get("player_name", "")
        scenario = args.get("scenario", "")
        options = args.get("options", "")

        if not player_name:
            return ToolResult(success=False, error="Player name is required")
        if not scenario:
            return ToolResult(success=False, error="Scenario description is required")

        # Get the player profile
        profile = self._character_store.get_by_name(player_name)
        if not profile:
            return ToolResult(
                success=False,
                error=f"Player '{player_name}' not found. Create them first with create_character.",
            )

        if profile.character_type != "player":
            return ToolResult(
                success=False,
                error=f"'{player_name}' is a {profile.character_type}, not a player",
            )

        # Build the system prompt
        system_prompt = self._build_player_prompt(profile)

        # Build the user message
        user_message = f"Scenario: {scenario}"
        if options:
            user_message += f"\n\nAvailable options: {options}"

        # Call the LLM
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]

        response = self.llm.chat(messages, tools=None)

        return ToolResult(
            success=True,
            data=f"[{profile.name} (player)]: {response.text}",
        )

    def _create_character(self, args: dict[str, Any]) -> ToolResult:
        """Create a new character profile."""
        name = args.get("name", "")
        character_type = args.get("character_type", "npc")

        if not name:
            return ToolResult(success=False, error="Character name is required")

        if character_type not in ("npc", "monster", "player"):
            return ToolResult(
                success=False,
                error="Character type must be 'npc', 'monster', or 'player'",
            )

        # Check if character already exists
        existing = self._character_store.get_by_name(name)
        if existing:
            return ToolResult(
                success=False,
                error=f"Character '{name}' already exists",
            )

        # Parse list fields
        kwargs = {
            "personality": args.get("personality", ""),
            "speech_patterns": args.get("speech_patterns", ""),
            "knowledge": self._parse_list(args.get("knowledge", "")),
            "goals": self._parse_list(args.get("goals", "")),
            "secrets": self._parse_list(args.get("secrets", "")),
            "intelligence": args.get("intelligence", "average"),
            "instincts": self._parse_list(args.get("instincts", "")),
            "morale": args.get("morale", ""),
            "playstyle": args.get("playstyle", ""),
            "risk_tolerance": args.get("risk_tolerance", "medium"),
            "decision_making": args.get("decision_making", ""),
            "party_role": args.get("party_role", ""),
            "quirks": self._parse_list(args.get("quirks", "")),
        }

        profile = self._character_store.create(
            name=name,
            character_type=character_type,
            **kwargs,
        )

        return ToolResult(
            success=True,
            data=f"Created {character_type} profile: {profile.name} (id: {profile.id})",
        )

    def _get_character(self, args: dict[str, Any]) -> ToolResult:
        """Get a character profile by name."""
        name = args.get("name", "")
        if not name:
            return ToolResult(success=False, error="Character name is required")

        profile = self._character_store.get_by_name(name)
        if not profile:
            return ToolResult(
                success=False,
                error=f"Character '{name}' not found",
            )

        return ToolResult(
            success=True,
            data=self._format_profile(profile),
        )

    def _list_characters(self, args: dict[str, Any]) -> ToolResult:
        """List all character profiles."""
        character_type = args.get("character_type")

        profiles = self._character_store.list(character_type)

        if not profiles:
            if character_type:
                return ToolResult(
                    success=True,
                    data=f"No {character_type} characters found.",
                )
            return ToolResult(success=True, data="No characters found.")

        lines = []
        for profile in profiles:
            lines.append(f"- {profile.name} ({profile.character_type})")

        return ToolResult(
            success=True,
            data="\n".join(lines),
        )

    def _build_npc_prompt(self, profile: CharacterProfile) -> str:
        """Build system prompt for NPC embodiment."""
        # Get knowledge from KnowledgeStore (high importance knowledge)
        knowledge_entries = self.knowledge.query_knowledge(
            character_id=profile.id,
            min_importance=3,  # Only include reasonably important knowledge
            limit=50,
        )

        # Combine KnowledgeStore entries with profile knowledge
        knowledge_items = []
        if knowledge_entries:
            for k in knowledge_entries:
                # Add type prefix for non-facts
                prefix = f"[{k.knowledge_type.upper()}] " if k.knowledge_type != "fact" else ""
                knowledge_items.append(f"{prefix}{k.content}")

        # Add profile knowledge (for backward compatibility)
        if profile.knowledge:
            knowledge_items.extend(profile.knowledge)

        # Add faction knowledge (if NPC belongs to factions)
        goal_items = list(profile.goals) if profile.goals else []
        if profile.faction_ids:
            for faction_id in profile.faction_ids:
                faction = self.factions.get(faction_id)
                if faction:
                    # Add faction's shared knowledge IDs to character knowledge
                    for knowledge_id in faction.shared_knowledge:
                        try:
                            k_id = int(knowledge_id)
                            k_entry = self.knowledge.get_by_id(k_id)
                            if k_entry:
                                prefix = f"[FACTION: {k_entry.knowledge_type.upper()}] " if k_entry.knowledge_type != "fact" else f"[FACTION] "
                                knowledge_items.append(f"{prefix}{k_entry.content}")
                        except (ValueError, AttributeError):
                            pass

                    # Add faction goals to character goals
                    for faction_goal in faction.goals:
                        if faction_goal not in goal_items:
                            goal_items.append(f"[FACTION: {faction.name}] {faction_goal}")

        # Add location knowledge (if scene has a location)
        session = session_store.get_current(self.campaign_id)
        if session and session.scene_state.location_id:
            location = self.locations.get(session.scene_state.location_id)
            if location:
                # Add location's common knowledge
                for knowledge_id in location.common_knowledge:
                    try:
                        k_id = int(knowledge_id)
                        k_entry = self.knowledge.get_by_id(k_id)
                        if k_entry:
                            prefix = f"[LOCATION: {k_entry.knowledge_type.upper()}] " if k_entry.knowledge_type != "fact" else f"[LOCATION] "
                            knowledge_items.append(f"{prefix}{k_entry.content}")
                    except (ValueError, AttributeError):
                        pass

                # Add recent events at this location
                if location.recent_events:
                    for event in location.recent_events:
                        knowledge_items.append(f"[LOCATION: RECENT EVENT] {event}")

        knowledge = (
            "\n".join(f"- {k}" for k in knowledge_items)
            if knowledge_items
            else "- General knowledge only"
        )
        goals = (
            "\n".join(f"- {g}" for g in goal_items) if goal_items else "- No specific goals"
        )

        secrets_section = ""
        if profile.secrets:
            secrets = "\n".join(f"- {s}" for s in profile.secrets)
            secrets_section = (
                f"\nSecrets you keep (do not reveal unless dramatically appropriate):\n{secrets}"
            )

        return NPC_SYSTEM_PROMPT.format(
            name=profile.name,
            personality=profile.personality or "A typical NPC",
            speech_patterns=profile.speech_patterns or "Normal speech",
            knowledge=knowledge,
            goals=goals,
            secrets_section=secrets_section,
        )

    def _build_monster_prompt(self, profile: CharacterProfile) -> str:
        """Build system prompt for monster decisions."""
        instincts = (
            "\n".join(f"- {i}" for i in profile.instincts)
            if profile.instincts
            else "- Basic survival instincts"
        )
        goals = (
            "\n".join(f"- {g}" for g in profile.goals)
            if profile.goals
            else "- Survive and protect territory"
        )

        return MONSTER_SYSTEM_PROMPT.format(
            name=profile.name,
            intelligence=profile.intelligence,
            personality=profile.personality or "Typical for its kind",
            instincts=instincts,
            goals=goals,
            morale=profile.morale or "Fights until threatened, then may flee",
        )

    def _build_player_prompt(self, profile: CharacterProfile) -> str:
        """Build system prompt for player emulation."""
        quirks = (
            "\n".join(f"- {q}" for q in profile.quirks)
            if profile.quirks
            else "- No particular quirks"
        )

        return PLAYER_SYSTEM_PROMPT.format(
            name=profile.name,
            party_role=profile.party_role or "General adventurer",
            playstyle=profile.playstyle or "Balanced approach",
            risk_tolerance=profile.risk_tolerance,
            decision_making=profile.decision_making or "Makes decisions as they come",
            quirks=quirks,
        )

    def _format_profile(self, profile: CharacterProfile) -> str:
        """Format a character profile for display."""
        lines = [
            f"Name: {profile.name}",
            f"Type: {profile.character_type}",
        ]

        if profile.personality:
            lines.append(f"Personality: {profile.personality}")

        if profile.character_type == "npc":
            if profile.speech_patterns:
                lines.append(f"Speech: {profile.speech_patterns}")
            if profile.knowledge:
                lines.append(f"Knows: {', '.join(profile.knowledge)}")
            if profile.goals:
                lines.append(f"Goals: {', '.join(profile.goals)}")
            if profile.secrets:
                lines.append(f"Secrets: {', '.join(profile.secrets)}")

        elif profile.character_type == "monster":
            lines.append(f"Intelligence: {profile.intelligence}")
            if profile.instincts:
                lines.append(f"Instincts: {', '.join(profile.instincts)}")
            if profile.morale:
                lines.append(f"Morale: {profile.morale}")

        elif profile.character_type == "player":
            if profile.party_role:
                lines.append(f"Role: {profile.party_role}")
            if profile.playstyle:
                lines.append(f"Playstyle: {profile.playstyle}")
            lines.append(f"Risk tolerance: {profile.risk_tolerance}")
            if profile.quirks:
                lines.append(f"Quirks: {', '.join(profile.quirks)}")

        return "\n".join(lines)

    def _parse_list(self, value: str) -> list[str]:
        """Parse a comma-separated string into a list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def close(self) -> None:
        """Close resources."""
        pass
