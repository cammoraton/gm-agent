"""Campaign State MCP server for narrative layer management."""

from typing import Any

from ..config import CAMPAIGNS_DIR
from ..storage.campaign import campaign_store
from ..storage.session import session_store
from ..storage.history import HistoryIndex
from ..storage.characters import CharacterStore
from ..storage.dialogue import DialogueStore
from ..storage.factions import FactionStore
from ..storage.locations import LocationStore
from ..storage.secrets import SecretStore
from ..storage.schemas import SceneState, Relationship
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class CampaignStateServer(MCPServer):
    """MCP server for campaign state management.

    Provides tools for:
    - Scene state updates (location, NPCs, time, conditions)
    - Event logging with importance levels
    - History search using FTS5
    - Session summary management
    - Campaign preferences
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self._history: HistoryIndex | None = None
        self._character_store: CharacterStore | None = None
        self._dialogue_store: DialogueStore | None = None
        self._faction_store: FactionStore | None = None
        self._location_store: LocationStore | None = None
        self._secret_store: SecretStore | None = None
        self._tools = self._build_tools()

    @property
    def history(self) -> HistoryIndex:
        """Lazy-load history index."""
        if self._history is None:
            self._history = HistoryIndex(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._history

    @property
    def characters(self) -> CharacterStore:
        """Lazy-load character store."""
        if self._character_store is None:
            self._character_store = CharacterStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._character_store

    @property
    def dialogue(self) -> DialogueStore:
        """Lazy-load dialogue store."""
        if self._dialogue_store is None:
            self._dialogue_store = DialogueStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._dialogue_store

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

    @property
    def secrets(self) -> SecretStore:
        """Lazy-load secret store."""
        if self._secret_store is None:
            self._secret_store = SecretStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._secret_store

    def _build_tools(self) -> list[ToolDef]:
        """Build the tool definitions."""
        return [
            ToolDef(
                name="update_scene",
                description="Update the current scene state. Use this when the party moves to a new location, NPCs enter/leave, time passes, or conditions change.",
                parameters=[
                    ToolParameter(
                        name="location",
                        type="string",
                        description="The current location name (e.g., 'Sandpoint Market', 'Dungeon Level 2 Room 4')",
                        required=False,
                    ),
                    ToolParameter(
                        name="npcs_present",
                        type="string",
                        description="Comma-separated list of NPCs currently present in the scene",
                        required=False,
                    ),
                    ToolParameter(
                        name="time_of_day",
                        type="string",
                        description="Current time (e.g., 'morning', 'afternoon', 'evening', 'night', 'Day 3 - Morning')",
                        required=False,
                    ),
                    ToolParameter(
                        name="conditions",
                        type="string",
                        description="Comma-separated environmental conditions (e.g., 'dark, raining, crowded')",
                        required=False,
                    ),
                    ToolParameter(
                        name="notes",
                        type="string",
                        description="Additional scene notes for context",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="advance_time",
                description="Advance time in the game world. Use for rests, travel, or significant time passing.",
                parameters=[
                    ToolParameter(
                        name="amount",
                        type="string",
                        description="How much time passes (e.g., '1 hour', '8 hours', '1 day', 'overnight')",
                    ),
                    ToolParameter(
                        name="activity",
                        type="string",
                        description="What the party does during this time (e.g., 'resting', 'traveling', 'crafting')",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="log_event",
                description="Record an important narrative event for future reference. Use for discoveries, decisions, NPC interactions, or plot developments.",
                parameters=[
                    ToolParameter(
                        name="event",
                        type="string",
                        description="Description of what happened",
                    ),
                    ToolParameter(
                        name="importance",
                        type="string",
                        description="How significant: 'session' (minor), 'arc' (affects current storyline), 'campaign' (major turning point)",
                        required=False,
                        default="session",
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="Comma-separated tags for categorization (e.g., 'combat, goblin, victory' or 'npc, ally, quest')",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="search_history",
                description="Search past session events and logged history. Use to recall what happened previously.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search terms to find in past events",
                    ),
                    ToolParameter(
                        name="importance",
                        type="string",
                        description="Filter by importance level: 'session', 'arc', or 'campaign'",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of results to return",
                        required=False,
                        default=10,
                    ),
                ],
            ),
            ToolDef(
                name="get_scene",
                description="Get the current scene state including location, NPCs present, time, and conditions.",
                parameters=[],
            ),
            ToolDef(
                name="get_session_summary",
                description="Get the current session's rolling summary of what has happened.",
                parameters=[],
            ),
            ToolDef(
                name="update_session_summary",
                description="Update the session's rolling summary. Call this periodically to maintain context.",
                parameters=[
                    ToolParameter(
                        name="summary",
                        type="string",
                        description="The updated summary of the session so far",
                    ),
                ],
            ),
            ToolDef(
                name="get_preferences",
                description="Get the campaign's GM preferences including RAG aggressiveness and uncertainty mode.",
                parameters=[],
            ),
            ToolDef(
                name="update_preferences",
                description="Update campaign GM preferences.",
                parameters=[
                    ToolParameter(
                        name="rag_aggressiveness",
                        type="string",
                        description="How aggressively to use RAG: 'minimal', 'moderate', or 'aggressive'",
                        required=False,
                    ),
                    ToolParameter(
                        name="uncertainty_mode",
                        type="string",
                        description="How to handle uncertainty: 'gm' (make rulings) or 'introspective' (admit gaps)",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="add_relationship",
                description="Add or update a relationship between two characters. Use this to track how NPCs view each other or how they relate to the party.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character whose perspective this is (e.g., 'Voz Lirayne')",
                    ),
                    ToolParameter(
                        name="target_name",
                        type="string",
                        description="Name of the character they have a relationship with",
                    ),
                    ToolParameter(
                        name="relationship_type",
                        type="string",
                        description="Type of relationship: 'ally', 'enemy', 'family', 'employer', 'employee', 'rival', 'friend', 'acquaintance'",
                        required=False,
                        default="acquaintance",
                    ),
                    ToolParameter(
                        name="attitude",
                        type="string",
                        description="Their attitude: 'friendly', 'unfriendly', 'hostile', 'helpful', 'indifferent', 'neutral'",
                        required=False,
                        default="neutral",
                    ),
                    ToolParameter(
                        name="trust_level",
                        type="integer",
                        description="Trust level from -5 (complete distrust) to +5 (complete trust)",
                        required=False,
                        default=0,
                    ),
                    ToolParameter(
                        name="history",
                        type="string",
                        description="Shared history or how they met",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="notes",
                        type="string",
                        description="Additional relationship notes",
                        required=False,
                        default="",
                    ),
                ],
            ),
            ToolDef(
                name="get_relationships",
                description="Get all relationships for a character. Shows how they view other characters.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character to get relationships for",
                    ),
                ],
            ),
            ToolDef(
                name="query_relationships",
                description="Query relationships by type or attitude. Useful for finding allies, enemies, or specific relationship dynamics.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character to query relationships for",
                    ),
                    ToolParameter(
                        name="relationship_type",
                        type="string",
                        description="Filter by relationship type",
                        required=False,
                    ),
                    ToolParameter(
                        name="attitude",
                        type="string",
                        description="Filter by attitude",
                        required=False,
                    ),
                    ToolParameter(
                        name="min_trust",
                        type="integer",
                        description="Minimum trust level",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="remove_relationship",
                description="Remove a relationship between two characters.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character whose perspective this is",
                    ),
                    ToolParameter(
                        name="target_name",
                        type="string",
                        description="Name of the target character",
                    ),
                ],
            ),
            ToolDef(
                name="search_dialogue",
                description="Search NPC dialogue history for past conversations. Useful for checking consistency and recalling what an NPC has said before.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search query to find dialogue (searches character names and content)",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Filter by character name",
                        required=False,
                    ),
                    ToolParameter(
                        name="dialogue_type",
                        type="string",
                        description="Filter by dialogue type: 'statement', 'promise', 'threat', 'lie', 'rumor', 'secret'",
                        required=False,
                    ),
                    ToolParameter(
                        name="flagged_only",
                        type="boolean",
                        description="Only return flagged (important) dialogue",
                        required=False,
                        default=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of results to return",
                        required=False,
                        default=20,
                    ),
                ],
            ),
            ToolDef(
                name="flag_dialogue",
                description="Flag or unflag a dialogue entry as important.",
                parameters=[
                    ToolParameter(
                        name="dialogue_id",
                        type="integer",
                        description="ID of the dialogue entry to flag",
                    ),
                    ToolParameter(
                        name="flagged",
                        type="boolean",
                        description="True to flag, False to unflag",
                        required=False,
                        default=True,
                    ),
                ],
            ),
            ToolDef(
                name="create_faction",
                description="Create a new faction or organization.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Faction name",
                    ),
                    ToolParameter(
                        name="description",
                        type="string",
                        description="Faction description",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="goals",
                        type="string",
                        description="Comma-separated faction goals",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="resources",
                        type="string",
                        description="Comma-separated faction resources",
                        required=False,
                        default="",
                    ),
                ],
            ),
            ToolDef(
                name="get_faction_info",
                description="Get information about a faction.",
                parameters=[
                    ToolParameter(
                        name="faction_name",
                        type="string",
                        description="Name of the faction",
                    ),
                ],
            ),
            ToolDef(
                name="list_factions",
                description="List all factions in the campaign.",
                parameters=[],
            ),
            ToolDef(
                name="add_npc_to_faction",
                description="Add an NPC to a faction.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the character to add",
                    ),
                    ToolParameter(
                        name="faction_name",
                        type="string",
                        description="Name of the faction",
                    ),
                ],
            ),
            ToolDef(
                name="get_faction_members",
                description="Get all members of a faction.",
                parameters=[
                    ToolParameter(
                        name="faction_name",
                        type="string",
                        description="Name of the faction",
                    ),
                ],
            ),
            ToolDef(
                name="update_faction_reputation",
                description="Update faction reputation with the party.",
                parameters=[
                    ToolParameter(
                        name="faction_name",
                        type="string",
                        description="Name of the faction",
                    ),
                    ToolParameter(
                        name="reputation",
                        type="integer",
                        description="New reputation value (-100 to +100)",
                    ),
                ],
            ),
            # Location tools
            ToolDef(
                name="create_location",
                description="Create a new location in the campaign world.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Name of the location",
                    ),
                    ToolParameter(
                        name="description",
                        type="string",
                        description="Description of the location",
                        required=False,
                    ),
                    ToolParameter(
                        name="isolation_level",
                        type="string",
                        description="Isolation level: connected, remote, or isolated",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="get_location_info",
                description="Get detailed information about a location.",
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="Name of the location",
                    ),
                ],
            ),
            ToolDef(
                name="list_locations",
                description="List all locations in the campaign.",
                parameters=[],
            ),
            ToolDef(
                name="add_location_knowledge",
                description="Add common knowledge to a location that all NPCs there know.",
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="Name of the location",
                    ),
                    ToolParameter(
                        name="knowledge_id",
                        type="string",
                        description="ID of the knowledge entry to add",
                    ),
                ],
            ),
            ToolDef(
                name="add_location_event",
                description="Add a recent event to a location.",
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="Name of the location",
                    ),
                    ToolParameter(
                        name="event",
                        type="string",
                        description="Description of the event",
                    ),
                ],
            ),
            ToolDef(
                name="set_scene_location",
                description="Set the current scene's location.",
                parameters=[
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="Name of the location",
                    ),
                ],
            ),
            ToolDef(
                name="connect_locations",
                description="Create a connection between two locations.",
                parameters=[
                    ToolParameter(
                        name="location1",
                        type="string",
                        description="Name of the first location",
                    ),
                    ToolParameter(
                        name="location2",
                        type="string",
                        description="Name of the second location",
                    ),
                ],
            ),
            # Secret tools
            ToolDef(
                name="create_secret",
                description="Create a new secret to track.",
                parameters=[
                    ToolParameter(
                        name="content",
                        type="string",
                        description="The secret content",
                    ),
                    ToolParameter(
                        name="importance",
                        type="string",
                        description="Importance level: minor, major, or critical",
                        required=False,
                    ),
                    ToolParameter(
                        name="consequences",
                        type="string",
                        description="Comma-separated consequences when revealed",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="reveal_secret",
                description="Mark a secret as revealed to the party.",
                parameters=[
                    ToolParameter(
                        name="secret_id",
                        type="string",
                        description="The secret ID",
                    ),
                    ToolParameter(
                        name="revealer",
                        type="string",
                        description="Who revealed it",
                        required=False,
                    ),
                    ToolParameter(
                        name="method",
                        type="string",
                        description="How it was revealed",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="list_secrets",
                description="List all secrets in the campaign.",
                parameters=[
                    ToolParameter(
                        name="revealed",
                        type="string",
                        description="Filter by revelation status: true, false, or all",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="get_revelation_history",
                description="Get all revealed secrets with revelation details.",
                parameters=[],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        """List all available tools."""
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments."""
        try:
            if name == "update_scene":
                return self._update_scene(args)
            elif name == "advance_time":
                return self._advance_time(args)
            elif name == "log_event":
                return self._log_event(args)
            elif name == "search_history":
                return self._search_history(args)
            elif name == "get_scene":
                return self._get_scene()
            elif name == "get_session_summary":
                return self._get_session_summary()
            elif name == "update_session_summary":
                return self._update_session_summary(args)
            elif name == "get_preferences":
                return self._get_preferences()
            elif name == "update_preferences":
                return self._update_preferences(args)
            elif name == "add_relationship":
                return self._add_relationship(args)
            elif name == "get_relationships":
                return self._get_relationships(args)
            elif name == "query_relationships":
                return self._query_relationships(args)
            elif name == "remove_relationship":
                return self._remove_relationship(args)
            elif name == "search_dialogue":
                return self._search_dialogue(args)
            elif name == "flag_dialogue":
                return self._flag_dialogue(args)
            elif name == "create_faction":
                return self._create_faction(args)
            elif name == "get_faction_info":
                return self._get_faction_info(args)
            elif name == "list_factions":
                return self._list_factions()
            elif name == "add_npc_to_faction":
                return self._add_npc_to_faction(args)
            elif name == "get_faction_members":
                return self._get_faction_members(args)
            elif name == "update_faction_reputation":
                return self._update_faction_reputation(args)
            elif name == "create_location":
                return self._create_location(args)
            elif name == "get_location_info":
                return self._get_location_info(args)
            elif name == "list_locations":
                return self._list_locations()
            elif name == "add_location_knowledge":
                return self._add_location_knowledge(args)
            elif name == "add_location_event":
                return self._add_location_event(args)
            elif name == "set_scene_location":
                return self._set_scene_location(args)
            elif name == "connect_locations":
                return self._connect_locations(args)
            elif name == "create_secret":
                return self._create_secret(args)
            elif name == "reveal_secret":
                return self._reveal_secret(args)
            elif name == "list_secrets":
                return self._list_secrets(args)
            elif name == "get_revelation_history":
                return self._get_revelation_history()
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _update_scene(self, args: dict[str, Any]) -> ToolResult:
        """Update the current scene state."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        current = session.scene_state

        # Parse comma-separated lists
        npcs = args.get("npcs_present")
        if npcs and isinstance(npcs, str):
            npcs = [n.strip() for n in npcs.split(",") if n.strip()]
        else:
            npcs = current.npcs_present

        conditions = args.get("conditions")
        if conditions and isinstance(conditions, str):
            conditions = [c.strip() for c in conditions.split(",") if c.strip()]
        else:
            conditions = current.conditions

        new_scene = SceneState(
            location=args.get("location", current.location),
            npcs_present=npcs,
            time_of_day=args.get("time_of_day", current.time_of_day),
            conditions=conditions,
            notes=args.get("notes", current.notes),
        )

        session_store.update_scene(self.campaign_id, new_scene)

        return ToolResult(
            success=True,
            data=f"Scene updated: {new_scene.location} ({new_scene.time_of_day})",
        )

    def _advance_time(self, args: dict[str, Any]) -> ToolResult:
        """Advance time in the game world."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        amount = args.get("amount", "")
        activity = args.get("activity", "")

        # Update time_of_day in scene state
        current = session.scene_state
        new_time = self._calculate_new_time(current.time_of_day, amount)

        new_scene = SceneState(
            location=current.location,
            npcs_present=current.npcs_present,
            time_of_day=new_time,
            conditions=current.conditions,
            notes=current.notes,
        )
        session_store.update_scene(self.campaign_id, new_scene)

        # Log the time advancement as an event
        event_text = f"Time advanced: {amount}"
        if activity:
            event_text += f" ({activity})"

        self.history.log_event(
            session_id=session.id,
            event=event_text,
            importance="session",
            tags=["time", activity.lower()] if activity else ["time"],
            turn_number=len(session.turns),
        )

        return ToolResult(
            success=True,
            data=f"Time advanced by {amount}. New time: {new_time}",
        )

    def _calculate_new_time(self, current: str, amount: str) -> str:
        """Calculate new time after advancement.

        This is a simplified time tracker - it handles common cases.
        """
        amount_lower = amount.lower()

        # Time periods in order
        times = ["morning", "afternoon", "evening", "night"]

        # Try to find current period
        current_lower = current.lower()
        current_idx = -1
        for i, t in enumerate(times):
            if t in current_lower:
                current_idx = i
                break

        if current_idx == -1:
            # Can't parse current time, just append
            return f"{current} + {amount}"

        # Calculate advancement
        if "overnight" in amount_lower or "8 hour" in amount_lower:
            # Full rest - advance to morning
            return "morning (next day)"
        elif "hour" in amount_lower:
            # Try to parse hours
            try:
                hours = int("".join(c for c in amount if c.isdigit()) or "1")
                periods = hours // 4  # ~4 hours per period
                new_idx = (current_idx + max(1, periods)) % 4
                day_advance = (current_idx + max(1, periods)) // 4
                result = times[new_idx]
                if day_advance > 0:
                    result += f" (day +{day_advance})"
                return result
            except ValueError:
                pass
        elif "day" in amount_lower:
            try:
                days = int("".join(c for c in amount if c.isdigit()) or "1")
                return f"{times[current_idx]} (+{days} days)"
            except ValueError:
                pass

        return f"{current} + {amount}"

    def _log_event(self, args: dict[str, Any]) -> ToolResult:
        """Log an important event."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        event = args.get("event", "")
        if not event:
            return ToolResult(success=False, error="Event text is required")

        importance = args.get("importance", "session")
        if importance not in ("session", "arc", "campaign"):
            importance = "session"

        tags_str = args.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        logged = self.history.log_event(
            session_id=session.id,
            event=event,
            importance=importance,
            tags=tags,
            turn_number=len(session.turns),
        )

        return ToolResult(
            success=True,
            data=f"Event logged (importance: {importance}): {event[:100]}...",
        )

    def _search_history(self, args: dict[str, Any]) -> ToolResult:
        """Search past events."""
        query = args.get("query", "")
        importance = args.get("importance")
        limit = args.get("limit", 10)

        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = 10

        results = self.history.search(
            query=query,
            importance=importance,
            limit=limit,
        )

        if not results:
            return ToolResult(
                success=True,
                data=f"No events found matching '{query}'",
            )

        formatted = []
        for event in results:
            importance_marker = {
                "campaign": "[CAMPAIGN]",
                "arc": "[ARC]",
                "session": "",
            }.get(event.importance, "")

            tags_str = f" #{' #'.join(event.tags)}" if event.tags else ""
            formatted.append(f"{importance_marker} {event.event}{tags_str}")

        return ToolResult(
            success=True,
            data="\n".join(formatted),
        )

    def _get_scene(self) -> ToolResult:
        """Get current scene state."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        scene = session.scene_state
        lines = [
            f"Location: {scene.location}",
            f"Time: {scene.time_of_day}",
        ]

        if scene.npcs_present:
            lines.append(f"NPCs Present: {', '.join(scene.npcs_present)}")

        if scene.conditions:
            lines.append(f"Conditions: {', '.join(scene.conditions)}")

        if scene.notes:
            lines.append(f"Notes: {scene.notes}")

        return ToolResult(success=True, data="\n".join(lines))

    def _get_session_summary(self) -> ToolResult:
        """Get current session summary."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        if session.summary:
            return ToolResult(success=True, data=session.summary)
        else:
            return ToolResult(
                success=True,
                data="No summary yet. The session has just started.",
            )

    def _update_session_summary(self, args: dict[str, Any]) -> ToolResult:
        """Update the session summary."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        summary = args.get("summary", "")
        if not summary:
            return ToolResult(success=False, error="Summary text is required")

        session.summary = summary
        session_store._save_current(session)

        return ToolResult(
            success=True,
            data="Session summary updated.",
        )

    def _get_preferences(self) -> ToolResult:
        """Get campaign preferences."""
        campaign = campaign_store.get(self.campaign_id)
        if not campaign:
            return ToolResult(success=False, error="Campaign not found")

        prefs = campaign.preferences
        lines = [
            f"RAG Aggressiveness: {prefs.get('rag_aggressiveness', 'moderate')}",
            f"Uncertainty Mode: {prefs.get('uncertainty_mode', 'gm')}",
        ]

        # Include any other preferences
        for key, value in prefs.items():
            if key not in ("rag_aggressiveness", "uncertainty_mode"):
                lines.append(f"{key}: {value}")

        return ToolResult(success=True, data="\n".join(lines))

    def _update_preferences(self, args: dict[str, Any]) -> ToolResult:
        """Update campaign preferences."""
        campaign = campaign_store.get(self.campaign_id)
        if not campaign:
            return ToolResult(success=False, error="Campaign not found")

        updated = []

        if "rag_aggressiveness" in args:
            value = args["rag_aggressiveness"]
            if value in ("minimal", "moderate", "aggressive"):
                campaign.preferences["rag_aggressiveness"] = value
                updated.append(f"rag_aggressiveness={value}")

        if "uncertainty_mode" in args:
            value = args["uncertainty_mode"]
            if value in ("gm", "introspective"):
                campaign.preferences["uncertainty_mode"] = value
                updated.append(f"uncertainty_mode={value}")

        if updated:
            campaign_store.update(campaign)
            return ToolResult(
                success=True,
                data=f"Preferences updated: {', '.join(updated)}",
            )
        else:
            return ToolResult(
                success=True,
                data="No valid preferences to update.",
            )

    def _add_relationship(self, args: dict[str, Any]) -> ToolResult:
        """Add or update a relationship between characters."""
        character_name = args.get("character_name")
        target_name = args.get("target_name")

        if not character_name or not target_name:
            return ToolResult(
                success=False,
                error="Both character_name and target_name are required"
            )

        # Get the character
        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Get the target character to get their ID
        target = self.characters.get_by_name(target_name)
        if not target:
            return ToolResult(
                success=False,
                error=f"Target character '{target_name}' not found"
            )

        # Create or update relationship
        relationship = Relationship(
            target_character_id=target.id,
            target_name=target.name,
            relationship_type=args.get("relationship_type", "acquaintance"),
            attitude=args.get("attitude", "neutral"),
            trust_level=args.get("trust_level", 0),
            history=args.get("history", ""),
            notes=args.get("notes", "")
        )

        # Check if relationship already exists
        existing_idx = None
        for idx, rel in enumerate(character.relationships):
            if rel.target_character_id == target.id:
                existing_idx = idx
                break

        if existing_idx is not None:
            character.relationships[existing_idx] = relationship
            action = "updated"
        else:
            character.relationships.append(relationship)
            action = "added"

        # Save the character
        self.characters.update(character)

        return ToolResult(
            success=True,
            data=f"Relationship {action}: {character.name} -> {target.name} "
                 f"({relationship.relationship_type}, {relationship.attitude}, "
                 f"trust: {relationship.trust_level})"
        )

    def _get_relationships(self, args: dict[str, Any]) -> ToolResult:
        """Get all relationships for a character."""
        character_name = args.get("character_name")

        if not character_name:
            return ToolResult(success=False, error="character_name is required")

        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        if not character.relationships:
            return ToolResult(
                success=True,
                data=f"{character.name} has no relationships recorded."
            )

        # Format relationships
        lines = [f"{character.name}'s relationships:"]
        for rel in character.relationships:
            trust_str = f"trust: {rel.trust_level:+d}"
            lines.append(
                f"  - {rel.target_name}: {rel.relationship_type}, {rel.attitude}, {trust_str}"
            )
            if rel.history:
                lines.append(f"    History: {rel.history}")
            if rel.notes:
                lines.append(f"    Notes: {rel.notes}")

        return ToolResult(success=True, data="\n".join(lines))

    def _query_relationships(self, args: dict[str, Any]) -> ToolResult:
        """Query relationships by type, attitude, or trust level."""
        character_name = args.get("character_name")

        if not character_name:
            return ToolResult(success=False, error="character_name is required")

        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Filter relationships
        filtered = character.relationships
        filter_type = args.get("relationship_type")
        filter_attitude = args.get("attitude")
        min_trust = args.get("min_trust")

        if filter_type:
            filtered = [r for r in filtered if r.relationship_type == filter_type]

        if filter_attitude:
            filtered = [r for r in filtered if r.attitude == filter_attitude]

        if min_trust is not None:
            filtered = [r for r in filtered if r.trust_level >= min_trust]

        if not filtered:
            filters = []
            if filter_type:
                filters.append(f"type={filter_type}")
            if filter_attitude:
                filters.append(f"attitude={filter_attitude}")
            if min_trust is not None:
                filters.append(f"min_trust={min_trust}")
            filter_str = ", ".join(filters) if filters else "no filters"

            return ToolResult(
                success=True,
                data=f"No relationships found for {character.name} with {filter_str}"
            )

        # Format results
        lines = [f"Matching relationships for {character.name}:"]
        for rel in filtered:
            trust_str = f"trust: {rel.trust_level:+d}"
            lines.append(
                f"  - {rel.target_name}: {rel.relationship_type}, {rel.attitude}, {trust_str}"
            )

        return ToolResult(success=True, data="\n".join(lines))

    def _remove_relationship(self, args: dict[str, Any]) -> ToolResult:
        """Remove a relationship between characters."""
        character_name = args.get("character_name")
        target_name = args.get("target_name")

        if not character_name or not target_name:
            return ToolResult(
                success=False,
                error="Both character_name and target_name are required"
            )

        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Find and remove the relationship
        target = self.characters.get_by_name(target_name)
        if not target:
            return ToolResult(
                success=False,
                error=f"Target character '{target_name}' not found"
            )

        removed = False
        for idx, rel in enumerate(character.relationships):
            if rel.target_character_id == target.id:
                character.relationships.pop(idx)
                removed = True
                break

        if not removed:
            return ToolResult(
                success=False,
                error=f"No relationship found between {character.name} and {target.name}"
            )

        # Save the character
        self.characters.update(character)

        return ToolResult(
            success=True,
            data=f"Relationship removed: {character.name} -> {target.name}"
        )

    def _search_dialogue(self, args: dict[str, Any]) -> ToolResult:
        """Search NPC dialogue history."""
        query = args.get("query", "")
        character_name = args.get("character_name")
        dialogue_type = args.get("dialogue_type")
        flagged_only = args.get("flagged_only", False)
        limit = args.get("limit", 20)

        # Search dialogue
        results = self.dialogue.search(
            query=query,
            character_name=character_name,
            dialogue_type=dialogue_type,
            flagged_only=flagged_only,
            limit=limit,
        )

        if not results:
            filters = []
            if character_name:
                filters.append(f"character={character_name}")
            if dialogue_type:
                filters.append(f"type={dialogue_type}")
            if flagged_only:
                filters.append("flagged only")
            filter_str = ", ".join(filters) if filters else "no filters"

            return ToolResult(
                success=True,
                data=f"No dialogue found for query '{query}' with {filter_str}"
            )

        # Format results
        lines = [f"Found {len(results)} dialogue entries:"]
        for entry in results:
            flag_marker = "⭐ " if entry.flagged else ""
            lines.append(
                f"\n{flag_marker}[{entry.id}] {entry.character_name} ({entry.timestamp.strftime('%Y-%m-%d %H:%M')})"
            )
            lines.append(f"  Type: {entry.dialogue_type}")
            lines.append(f"  \"{entry.content}\"")

        return ToolResult(success=True, data="\n".join(lines))

    def _flag_dialogue(self, args: dict[str, Any]) -> ToolResult:
        """Flag or unflag a dialogue entry."""
        dialogue_id = args.get("dialogue_id")
        flagged = args.get("flagged", True)

        if dialogue_id is None:
            return ToolResult(success=False, error="dialogue_id is required")

        success = self.dialogue.flag_dialogue(dialogue_id, flagged)

        if not success:
            return ToolResult(
                success=False,
                error=f"Dialogue entry {dialogue_id} not found"
            )

        action = "flagged" if flagged else "unflagged"
        return ToolResult(
            success=True,
            data=f"Dialogue entry {dialogue_id} {action}"
        )

    def _create_faction(self, args: dict[str, Any]) -> ToolResult:
        """Create a new faction."""
        name = args.get("name")

        if not name:
            return ToolResult(success=False, error="name is required")

        # Check if faction already exists
        existing = self.factions.get_by_name(name)
        if existing:
            return ToolResult(
                success=False,
                error=f"Faction '{name}' already exists"
            )

        # Parse goals and resources
        goals_str = args.get("goals", "")
        goals = [g.strip() for g in goals_str.split(",") if g.strip()] if goals_str else []

        resources_str = args.get("resources", "")
        resources = [r.strip() for r in resources_str.split(",") if r.strip()] if resources_str else []

        # Create faction
        faction = self.factions.create(
            name=name,
            description=args.get("description", ""),
            goals=goals,
            resources=resources,
        )

        return ToolResult(
            success=True,
            data=f"Faction created: {faction.name}"
        )

    def _get_faction_info(self, args: dict[str, Any]) -> ToolResult:
        """Get information about a faction."""
        faction_name = args.get("faction_name")

        if not faction_name:
            return ToolResult(success=False, error="faction_name is required")

        faction = self.factions.get_by_name(faction_name)
        if not faction:
            return ToolResult(
                success=False,
                error=f"Faction '{faction_name}' not found"
            )

        # Format faction information
        lines = [f"Faction: {faction.name}"]
        if faction.description:
            lines.append(f"Description: {faction.description}")

        if faction.goals:
            lines.append(f"\nGoals:")
            for goal in faction.goals:
                lines.append(f"  - {goal}")

        if faction.resources:
            lines.append(f"\nResources:")
            for resource in faction.resources:
                lines.append(f"  - {resource}")

        lines.append(f"\nMembers: {len(faction.member_character_ids)}")
        lines.append(f"Reputation with party: {faction.reputation_with_party}")

        if faction.inter_faction_attitudes:
            lines.append(f"\nRelationships with other factions:")
            for faction_id, attitude in faction.inter_faction_attitudes.items():
                lines.append(f"  - {faction_id}: {attitude}")

        return ToolResult(success=True, data="\n".join(lines))

    def _list_factions(self) -> ToolResult:
        """List all factions."""
        factions = self.factions.list_all()

        if not factions:
            return ToolResult(
                success=True,
                data="No factions in this campaign."
            )

        lines = [f"Factions ({len(factions)}):"]
        for faction in factions:
            rep_str = f" (rep: {faction.reputation_with_party:+d})"
            member_str = f" [{len(faction.member_character_ids)} members]"
            lines.append(f"  - {faction.name}{rep_str}{member_str}")

        return ToolResult(success=True, data="\n".join(lines))

    def _add_npc_to_faction(self, args: dict[str, Any]) -> ToolResult:
        """Add an NPC to a faction."""
        character_name = args.get("character_name")
        faction_name = args.get("faction_name")

        if not character_name or not faction_name:
            return ToolResult(
                success=False,
                error="Both character_name and faction_name are required"
            )

        # Get character
        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(
                success=False,
                error=f"Character '{character_name}' not found"
            )

        # Get faction
        faction = self.factions.get_by_name(faction_name)
        if not faction:
            return ToolResult(
                success=False,
                error=f"Faction '{faction_name}' not found"
            )

        # Add to faction
        self.factions.add_member(faction.id, character.id)

        # Update character's faction list
        if faction.id not in character.faction_ids:
            character.faction_ids.append(faction.id)
            self.characters.update(character)

        return ToolResult(
            success=True,
            data=f"{character.name} added to faction: {faction.name}"
        )

    def _get_faction_members(self, args: dict[str, Any]) -> ToolResult:
        """Get all members of a faction."""
        faction_name = args.get("faction_name")

        if not faction_name:
            return ToolResult(success=False, error="faction_name is required")

        faction = self.factions.get_by_name(faction_name)
        if not faction:
            return ToolResult(
                success=False,
                error=f"Faction '{faction_name}' not found"
            )

        if not faction.member_character_ids:
            return ToolResult(
                success=True,
                data=f"{faction.name} has no members."
            )

        # Get character names
        lines = [f"{faction.name} members ({len(faction.member_character_ids)}):"]
        for char_id in faction.member_character_ids:
            # Try to get character name
            character = self.characters.get(char_id)
            if character:
                leader_marker = " (Leader)" if char_id == faction.leader_character_id else ""
                lines.append(f"  - {character.name}{leader_marker}")
            else:
                lines.append(f"  - [Character ID: {char_id}]")

        return ToolResult(success=True, data="\n".join(lines))

    def _update_faction_reputation(self, args: dict[str, Any]) -> ToolResult:
        """Update faction reputation with the party."""
        faction_name = args.get("faction_name")
        reputation = args.get("reputation")

        if not faction_name:
            return ToolResult(success=False, error="faction_name is required")

        if reputation is None:
            return ToolResult(success=False, error="reputation is required")

        faction = self.factions.get_by_name(faction_name)
        if not faction:
            return ToolResult(
                success=False,
                error=f"Faction '{faction_name}' not found"
            )

        # Update reputation
        self.factions.update_reputation(faction.id, reputation)

        return ToolResult(
            success=True,
            data=f"{faction.name} reputation updated to {reputation}"
        )

    def _create_location(self, args: dict[str, Any]) -> ToolResult:
        """Create a new location."""
        name = args.get("name")
        description = args.get("description", "")
        isolation_level = args.get("isolation_level", "connected")

        if not name:
            return ToolResult(success=False, error="name is required")

        # Check if location already exists
        existing = self.locations.get_by_name(name)
        if existing:
            return ToolResult(
                success=False,
                error=f"Location '{name}' already exists"
            )

        # Create location
        location = self.locations.create(
            name=name,
            description=description,
            isolation_level=isolation_level
        )

        return ToolResult(
            success=True,
            data=f"Created location '{location.name}' (ID: {location.id})"
        )

    def _get_location_info(self, args: dict[str, Any]) -> ToolResult:
        """Get location information."""
        location_name = args.get("location_name")

        if not location_name:
            return ToolResult(success=False, error="location_name is required")

        location = self.locations.get_by_name(location_name)
        if not location:
            return ToolResult(
                success=False,
                error=f"Location '{location_name}' not found"
            )

        # Build info string
        lines = [
            f"Name: {location.name}",
            f"ID: {location.id}",
        ]

        if location.description:
            lines.append(f"Description: {location.description}")

        lines.append(f"Isolation: {location.isolation_level}")

        if location.common_knowledge:
            lines.append(f"Common Knowledge: {len(location.common_knowledge)} entries")

        if location.recent_events:
            lines.append("Recent Events:")
            for event in location.recent_events:
                lines.append(f"  - {event}")

        if location.connected_locations:
            lines.append(f"Connected to: {', '.join(location.connected_locations)}")

        return ToolResult(success=True, data="\n".join(lines))

    def _list_locations(self) -> ToolResult:
        """List all locations."""
        locations = self.locations.list_all()

        if not locations:
            return ToolResult(success=True, data="No locations found.")

        lines = ["Locations:"]
        for loc in locations:
            isolation = f" ({loc.isolation_level})" if loc.isolation_level != "connected" else ""
            lines.append(f"  - {loc.name}{isolation}")

        return ToolResult(success=True, data="\n".join(lines))

    def _add_location_knowledge(self, args: dict[str, Any]) -> ToolResult:
        """Add common knowledge to a location."""
        location_name = args.get("location_name")
        knowledge_id = args.get("knowledge_id")

        if not location_name:
            return ToolResult(success=False, error="location_name is required")

        if not knowledge_id:
            return ToolResult(success=False, error="knowledge_id is required")

        location = self.locations.get_by_name(location_name)
        if not location:
            return ToolResult(
                success=False,
                error=f"Location '{location_name}' not found"
            )

        # Add knowledge to location
        success = self.locations.add_common_knowledge(location.id, knowledge_id)

        if success:
            return ToolResult(
                success=True,
                data=f"Added knowledge {knowledge_id} to {location.name}"
            )
        else:
            return ToolResult(success=False, error="Failed to add knowledge")

    def _add_location_event(self, args: dict[str, Any]) -> ToolResult:
        """Add a recent event to a location."""
        location_name = args.get("location_name")
        event = args.get("event")

        if not location_name:
            return ToolResult(success=False, error="location_name is required")

        if not event:
            return ToolResult(success=False, error="event is required")

        location = self.locations.get_by_name(location_name)
        if not location:
            return ToolResult(
                success=False,
                error=f"Location '{location_name}' not found"
            )

        # Add event to location
        success = self.locations.add_event(location.id, event)

        if success:
            return ToolResult(
                success=True,
                data=f"Added event to {location.name}: {event}"
            )
        else:
            return ToolResult(success=False, error="Failed to add event")

    def _set_scene_location(self, args: dict[str, Any]) -> ToolResult:
        """Set the current scene's location."""
        location_name = args.get("location_name")

        if not location_name:
            return ToolResult(success=False, error="location_name is required")

        location = self.locations.get_by_name(location_name)
        if not location:
            return ToolResult(
                success=False,
                error=f"Location '{location_name}' not found"
            )

        # Get current session
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(
                success=False,
                error="No active session"
            )

        # Update scene state
        scene = session.scene_state
        scene.location = location.name
        scene.location_id = location.id

        session_store.update_scene(self.campaign_id, scene)

        return ToolResult(
            success=True,
            data=f"Scene location set to: {location.name}"
        )

    def _connect_locations(self, args: dict[str, Any]) -> ToolResult:
        """Connect two locations."""
        location1_name = args.get("location1")
        location2_name = args.get("location2")

        if not location1_name:
            return ToolResult(success=False, error="location1 is required")

        if not location2_name:
            return ToolResult(success=False, error="location2 is required")

        location1 = self.locations.get_by_name(location1_name)
        location2 = self.locations.get_by_name(location2_name)

        if not location1:
            return ToolResult(
                success=False,
                error=f"Location '{location1_name}' not found"
            )

        if not location2:
            return ToolResult(
                success=False,
                error=f"Location '{location2_name}' not found"
            )

        # Connect the locations (bidirectional)
        success = self.locations.connect_locations(location1.id, location2.id)

        if success:
            return ToolResult(
                success=True,
                data=f"Connected {location1.name} and {location2.name}"
            )
        else:
            return ToolResult(success=False, error="Failed to connect locations")

    def _create_secret(self, args: dict[str, Any]) -> ToolResult:
        """Create a new secret."""
        content = args.get("content")
        if not content:
            return ToolResult(success=False, error="content is required")

        importance = args.get("importance", "major")
        consequences_str = args.get("consequences", "")

        # Parse consequences
        consequences = []
        if consequences_str:
            consequences = [c.strip() for c in consequences_str.split(",")]

        secret = self.secrets.create(
            content=content,
            importance=importance,
            consequences=consequences,
        )

        return ToolResult(
            success=True,
            data=f"Created secret (ID: {secret.id}): {content[:50]}..."
        )

    def _reveal_secret(self, args: dict[str, Any]) -> ToolResult:
        """Reveal a secret to the party."""
        secret_id = args.get("secret_id")
        if not secret_id:
            return ToolResult(success=False, error="secret_id is required")

        revealer = args.get("revealer")
        method = args.get("method")

        # Get current session info
        session = session_store.get_current(self.campaign_id)
        session_id = session.id if session else None
        turn_number = len(session.turns) if session else None

        success = self.secrets.reveal_to_party(
            secret_id=secret_id,
            session_id=session_id,
            turn_number=turn_number,
            revealer=revealer,
            method=method,
        )

        if not success:
            return ToolResult(
                success=False,
                error=f"Secret '{secret_id}' not found"
            )

        # Get the secret to check consequences
        secret = self.secrets.get(secret_id)
        consequence_text = ""
        if secret and secret.consequences:
            untriggered = self.secrets.get_untriggered_consequences(secret_id)
            if untriggered:
                consequence_text = f"\n\nConsequences to trigger:\n" + "\n".join(f"  - {c}" for c in untriggered)

        return ToolResult(
            success=True,
            data=f"Revealed secret: {secret.content[:50]}...{consequence_text}"
        )

    def _list_secrets(self, args: dict[str, Any]) -> ToolResult:
        """List all secrets."""
        revealed_filter = args.get("revealed", "all")

        # Parse filter
        revealed = None
        if revealed_filter == "true":
            revealed = True
        elif revealed_filter == "false":
            revealed = False

        secrets = self.secrets.list_all(revealed=revealed)

        if not secrets:
            return ToolResult(success=True, data="No secrets found.")

        lines = ["Secrets:"]
        for secret in secrets:
            status = "REVEALED" if secret.revealed_to_party else "HIDDEN"
            lines.append(f"  [{status}] ({secret.importance.upper()}) {secret.content[:60]}...")
            if secret.revealed_to_party and secret.revelation_event:
                revealer = secret.revelation_event.get("revealer", "unknown")
                lines.append(f"    Revealed by: {revealer}")

        return ToolResult(success=True, data="\n".join(lines))

    def _get_revelation_history(self) -> ToolResult:
        """Get revelation history."""
        secrets = self.secrets.list_all(revealed=True)

        if not secrets:
            return ToolResult(success=True, data="No secrets have been revealed yet.")

        lines = ["Revelation History:"]
        for secret in sorted(secrets, key=lambda s: s.revelation_event.get("timestamp", "") if s.revelation_event else ""):
            if secret.revelation_event:
                timestamp = secret.revelation_event.get("timestamp", "unknown")
                revealer = secret.revelation_event.get("revealer", "unknown")
                method = secret.revelation_event.get("method", "unknown")

                lines.append(f"\n{secret.content}")
                lines.append(f"  When: {timestamp}")
                lines.append(f"  Revealed by: {revealer}")
                lines.append(f"  Method: {method}")

                if secret.consequences:
                    lines.append(f"  Consequences:")
                    for consequence in secret.consequences:
                        triggered = "✓" if consequence in secret.triggered_consequences else "○"
                        lines.append(f"    {triggered} {consequence}")

        return ToolResult(success=True, data="\n".join(lines))

    def close(self) -> None:
        """Close resources."""
        if self._history:
            self._history.close()
            self._history = None
        if self._dialogue_store:
            self._dialogue_store.close()
            self._dialogue_store = None
