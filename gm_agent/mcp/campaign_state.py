"""Campaign State MCP server for narrative layer management."""

from typing import Any

from ..config import CAMPAIGNS_DIR
from ..propagation import PropagationBus
from ..storage.ap_progress import APProgressStore
from ..storage.campaign import campaign_store
from ..storage.session import session_store
from ..storage.treasure import TreasureStore, TREASURE_BY_LEVEL
from ..storage.history import HistoryIndex
from ..storage.characters import CharacterStore
from ..storage.dialogue import DialogueStore
from ..storage.factions import FactionStore
from ..storage.knowledge import KnowledgeStore
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
        self._knowledge_store: KnowledgeStore | None = None
        self._location_store: LocationStore | None = None
        self._secret_store: SecretStore | None = None
        self._propagation_bus: PropagationBus | None = None
        self._ap_progress_store: APProgressStore | None = None
        self._treasure_store: TreasureStore | None = None
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

    @property
    def knowledge(self) -> KnowledgeStore:
        """Lazy-load knowledge store."""
        if self._knowledge_store is None:
            self._knowledge_store = KnowledgeStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._knowledge_store

    @property
    def ap_progress(self) -> APProgressStore:
        """Lazy-load AP progress store."""
        if self._ap_progress_store is None:
            self._ap_progress_store = APProgressStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._ap_progress_store

    @property
    def treasure(self) -> TreasureStore:
        """Lazy-load treasure store."""
        if self._treasure_store is None:
            self._treasure_store = TreasureStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._treasure_store

    @property
    def propagation(self) -> PropagationBus:
        """Lazy-load propagation bus."""
        if self._propagation_bus is None:
            self._propagation_bus = PropagationBus(
                knowledge=self.knowledge,
                factions=self.factions,
                locations=self.locations,
                characters=self.characters,
            )
        return self._propagation_bus

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
                name="log_dialogue",
                description="Record notable NPC dialogue for consistency tracking. Use this when an NPC says something significant — promises, threats, lies, secrets, or important rumor. This keeps a searchable record so you can maintain NPC consistency across sessions.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the NPC speaking",
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="What the NPC said or the gist of their dialogue",
                    ),
                    ToolParameter(
                        name="dialogue_type",
                        type="string",
                        description="Type of dialogue: 'statement', 'promise', 'threat', 'lie', 'rumor', 'secret'",
                        required=False,
                        default="statement",
                    ),
                    ToolParameter(
                        name="flagged",
                        type="boolean",
                        description="Flag this dialogue as important for future reference (e.g., promises to keep, lies to remember)",
                        required=False,
                        default=False,
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
            # Propagation tools
            ToolDef(
                name="propagate_location_to_npc",
                description="Give an NPC the common knowledge associated with a location. "
                "Use when an NPC is known to inhabit or have visited a location.",
                parameters=[
                    ToolParameter(
                        name="character_name",
                        type="string",
                        description="Name of the NPC",
                    ),
                    ToolParameter(
                        name="location_name",
                        type="string",
                        description="Name of the location whose common knowledge to propagate",
                    ),
                ],
            ),
            # Session recap tool
            ToolDef(
                name="get_session_recap",
                description="Get a structured 'Previously on...' recap of the last (or specified) session. "
                "Includes key events, flagged dialogue, and current arc.",
                parameters=[
                    ToolParameter(
                        name="session_id",
                        type="string",
                        description="Specific session ID to recap (default: most recent ended session)",
                        required=False,
                    ),
                ],
            ),
            # Travel time
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
                        description="Whether to force-march (+50%% distance, requires Fort saves)", required=False, default=False,
                    ),
                ],
            ),
            # Hazard detection
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
                        description="Is someone Scouting? (+1 initiative, not direct detection)", required=False, default=False,
                    ),
                    ToolParameter(
                        name="searching", type="boolean",
                        description="Is someone Searching? (auto-detect if Perception >= DC)", required=False, default=False,
                    ),
                ],
            ),
            # --- Phase 5: AP Progression ---
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
            # --- Phase 5: Treasure Tracking ---
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
            elif name == "log_dialogue":
                return self._log_dialogue(args)
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
            elif name == "propagate_location_to_npc":
                return self._propagate_location_to_npc(args)
            elif name == "get_session_recap":
                return self._get_session_recap(args)
            elif name == "calculate_travel_time":
                return self._calculate_travel_time(args)
            elif name == "check_hazard_detection":
                return self._check_hazard_detection(args)
            elif name == "ap_progress":
                return self._ap_progress(args)
            elif name == "treasure":
                return self._treasure(args)
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

    def _log_dialogue(self, args: dict[str, Any]) -> ToolResult:
        """Log notable NPC dialogue for consistency tracking."""
        session = session_store.get_current(self.campaign_id)
        if not session:
            return ToolResult(success=False, error="No active session")

        character_name = args.get("character_name", "")
        if not character_name:
            return ToolResult(success=False, error="character_name is required")

        content = args.get("content", "")
        if not content:
            return ToolResult(success=False, error="Dialogue content is required")

        dialogue_type = args.get("dialogue_type", "statement")
        valid_types = ("statement", "promise", "threat", "lie", "rumor", "secret")
        if dialogue_type not in valid_types:
            dialogue_type = "statement"

        flagged = args.get("flagged", False)
        if isinstance(flagged, str):
            flagged = flagged.lower() in ("true", "1", "yes")

        # Slugify character name for ID
        import re
        slug = character_name.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        character_id = re.sub(r"-+", "-", slug).strip("-")

        entry = self.dialogue.log_dialogue(
            character_id=character_id,
            character_name=character_name,
            session_id=session.id,
            content=content,
            dialogue_type=dialogue_type,
            flagged=bool(flagged),
            turn_number=len(session.turns),
        )

        flag_marker = " [FLAGGED]" if entry.flagged else ""
        return ToolResult(
            success=True,
            data=f"Dialogue logged ({dialogue_type}{flag_marker}): {character_name}: \"{content[:80]}\"",
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

        # Propagate faction's shared knowledge to new member
        inherited = self.propagation.on_npc_joins_faction(
            character.id, character.name, faction.id,
        )
        inherit_text = f" (inherited {inherited} knowledge entries)" if inherited else ""

        return ToolResult(
            success=True,
            data=f"{character.name} added to faction: {faction.name}{inherit_text}"
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

        # Get the secret to check consequences and propagate knowledge
        secret = self.secrets.get(secret_id)
        consequence_text = ""
        propagated = 0
        if secret:
            if secret.consequences:
                untriggered = self.secrets.get_untriggered_consequences(secret_id)
                if untriggered:
                    consequence_text = f"\n\nConsequences to trigger:\n" + "\n".join(f"  - {c}" for c in untriggered)

            # Auto-propagate to knowledge stores
            propagated = self.propagation.on_secret_revealed(secret, revealer=revealer)

        propagation_text = f"\n\nKnowledge propagated to {propagated} target(s)." if propagated else ""

        return ToolResult(
            success=True,
            data=f"Revealed secret: {secret.content[:50]}...{consequence_text}{propagation_text}"
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

    # PF2e travel table constants
    TERRAIN_MULTIPLIERS = {
        "easy": 1.5,
        "normal": 1.0,
        "difficult": 0.5,
        "greater_difficult": 0.25,
    }

    def _calculate_travel_time(self, args: dict[str, Any]) -> ToolResult:
        """Calculate overland travel time."""
        from_name = args.get("from_location", "")
        to_name = args.get("to_location", "")
        speed_ft = int(args.get("speed_ft", 25))
        terrain = args.get("terrain", "normal")
        mounted = args.get("mounted", False)
        forced_march = args.get("forced_march", False)

        if not from_name or not to_name:
            return ToolResult(success=False, error="Both from_location and to_location are required.")

        # Resolve locations and count hops
        from_loc = self.locations.get_by_name(from_name)
        to_loc = self.locations.get_by_name(to_name)

        hops = 1  # Default: 1 hop if locations not connected or not found
        if from_loc and to_loc:
            if to_loc.id in from_loc.connected_locations:
                hops = 1
            else:
                # BFS to find shortest path
                hops = self._bfs_hops(from_loc.id, to_loc.id) or 1

        # PF2e: speed 25ft -> ~3 mph -> ~24 miles/day
        base_speed = 40 if mounted else speed_ft
        mph = base_speed / 25 * 3  # 25ft = 3 mph
        miles_per_day = mph * 8  # 8 hours of travel

        # Terrain
        terrain_mult = self.TERRAIN_MULTIPLIERS.get(terrain, 1.0)
        miles_per_day *= terrain_mult

        # Forced march
        if forced_march:
            miles_per_day *= 1.5

        # Estimate distance (~24 miles per hop as baseline)
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

    def _bfs_hops(self, start_id: str, end_id: str) -> int | None:
        """BFS over location connections to find shortest hop count."""
        from collections import deque
        visited = {start_id}
        queue = deque([(start_id, 0)])

        while queue:
            current, depth = queue.popleft()
            loc = self.locations.get(current)
            if not loc:
                continue
            for neighbor_id in loc.connected_locations:
                if neighbor_id == end_id:
                    return depth + 1
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return None

    def _check_hazard_detection(self, args: dict[str, Any]) -> ToolResult:
        """Advisory hazard detection check."""
        import json as json_mod
        stealth_dc = int(args.get("stealth_dc", 20))
        scouting = args.get("scouting", False)
        searching = args.get("searching", False)

        party_perception_str = args.get("party_perception", "{}")
        try:
            if isinstance(party_perception_str, str):
                party_perception = json_mod.loads(party_perception_str)
            else:
                party_perception = party_perception_str
        except (json_mod.JSONDecodeError, TypeError):
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

    def _get_session_recap(self, args: dict[str, Any]) -> ToolResult:
        """Get a structured recap of a previous session."""
        session_id = args.get("session_id")

        if session_id:
            session = session_store.get(self.campaign_id, session_id)
        else:
            session = session_store.get_previous_session(self.campaign_id)

        if not session:
            return ToolResult(
                success=True,
                data="No previous sessions found for recap.",
            )

        lines = [f"**Previously On... (Session {session.id})**\n"]

        # Summary
        if session.summary:
            lines.append(f"**Summary:** {session.summary}\n")

        # Campaign arc
        campaign = campaign_store.get(self.campaign_id)
        if campaign and campaign.current_arc:
            lines.append(f"**Current Arc:** {campaign.current_arc}\n")

        # Key events (sorted by importance)
        importance_order = {"campaign": 0, "arc": 1, "session": 2}
        try:
            events = self.history.search("", session_id=session.id, limit=20)
            if events:
                events.sort(key=lambda e: importance_order.get(e.get("importance", "session"), 2))
                lines.append("**Key Events:**")
                for event in events:
                    imp = event.get("importance", "session")
                    prefix = {"campaign": "!!!", "arc": "!!", "session": "!"}.get(imp, "!")
                    lines.append(f"  {prefix} {event.get('event', event.get('text', ''))}")
                lines.append("")
        except Exception:
            pass

        # Flagged dialogue
        try:
            flagged = self.dialogue.search(session_id=session.id, flagged_only=True, limit=10)
            if flagged:
                lines.append("**Flagged Dialogue:**")
                for d in flagged:
                    lines.append(f"  [{d.get('character_name', 'Unknown')}]: \"{d.get('content', '')}\"")
                lines.append("")
        except Exception:
            pass

        return ToolResult(success=True, data="\n".join(lines))

    def _propagate_location_to_npc(self, args: dict[str, Any]) -> ToolResult:
        """Propagate location common knowledge to an NPC."""
        character_name = args.get("character_name")
        location_name = args.get("location_name")

        if not character_name or not location_name:
            return ToolResult(
                success=False,
                error="Both character_name and location_name are required",
            )

        character = self.characters.get_by_name(character_name)
        if not character:
            return ToolResult(success=False, error=f"Character '{character_name}' not found")

        location = self.locations.get_by_name(location_name)
        if not location:
            return ToolResult(success=False, error=f"Location '{location_name}' not found")

        count = self.propagation.propagate_location_knowledge(
            character.id, character.name, location.id,
        )

        if count == 0:
            return ToolResult(
                success=True,
                data=f"{character.name} already knows everything about {location.name}.",
            )

        return ToolResult(
            success=True,
            data=f"{character.name} learned {count} knowledge entries from {location.name}.",
        )

    def _ap_progress(self, args: dict[str, Any]) -> ToolResult:
        """Compound tool for AP progression tracking."""
        action = args.get("action", "")
        name = args.get("name", "")
        book = args.get("book", "")
        xp = int(args.get("xp", 0))
        description = args.get("description", "")

        if action == "complete_encounter":
            if not name:
                return ToolResult(success=False, error="'name' is required for complete_encounter")
            entry = self.ap_progress.mark_complete(
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

    def _treasure(self, args: dict[str, Any]) -> ToolResult:
        """Compound tool for treasure tracking."""
        import json as json_mod

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

    def close(self) -> None:
        """Close resources."""
        if self._history:
            self._history.close()
            self._history = None
        if self._dialogue_store:
            self._dialogue_store.close()
            self._dialogue_store = None
        if self._knowledge_store:
            self._knowledge_store.close()
            self._knowledge_store = None
        if self._ap_progress_store:
            self._ap_progress_store.close()
            self._ap_progress_store = None
        if self._treasure_store:
            self._treasure_store.close()
            self._treasure_store = None
