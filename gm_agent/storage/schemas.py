"""Data schemas for campaigns and sessions."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SceneState(BaseModel):
    """Current scene state."""

    location: str = "Unknown"
    location_id: str | None = None  # Reference to Location ID for detailed location data
    npcs_present: list[str] = Field(default_factory=list)
    time_of_day: str = "day"
    conditions: list[str] = Field(default_factory=list)
    notes: str = ""


class ToolCallRecord(BaseModel):
    """Record of a tool call made during a turn."""

    name: str
    args: dict[str, Any]
    result: str


class TurnMetadata(BaseModel):
    """Metadata for a turn, useful for analytics and fine-tuning.

    This flexible structure captures contextual information about how
    the turn was generated, timing data, and source information.
    """

    source: str = "manual"  # "manual", "automation", "api"
    event_type: str | None = None  # "playerChat", "combatTurn", None for manual
    player_id: str | None = None  # Foundry user ID if from automation
    actor_name: str | None = None  # Character name if from automation
    processing_time_ms: float | None = None  # Time to generate response
    model: str | None = None  # LLM model used
    tool_count: int = 0  # Number of tool calls made
    tool_usage: dict[str, int] = Field(default_factory=dict)  # {"search_rules": 2, "roll_dice": 1}
    tool_failures: list[str] = Field(default_factory=list)  # ["search_rules", "get_combat_state"]
    error: str | None = None  # Error message if response was an error fallback


class Turn(BaseModel):
    """A single turn in a session."""

    player_input: str
    gm_response: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    scene_state: SceneState | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: TurnMetadata = Field(default_factory=TurnMetadata)


class PartyMember(BaseModel):
    """A player character in the party."""

    name: str
    ancestry: str = ""
    class_name: str = ""
    level: int = 1
    player_name: str = ""
    notes: str = ""


class Campaign(BaseModel):
    """A campaign with its background and current state."""

    id: str
    name: str
    background: str = ""
    current_arc: str = ""
    books: list[str] = Field(default_factory=list)
    party: list[PartyMember] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    """A game session with turns and summary."""

    id: str
    campaign_id: str
    turns: list[Turn] = Field(default_factory=list)
    summary: str = ""
    scene_state: SceneState = Field(default_factory=SceneState)
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None


class Relationship(BaseModel):
    """Relationship between two characters.

    Tracks how one character views/relates to another, including
    relationship type, attitude, trust, and shared history.
    """

    target_character_id: str
    target_name: str
    relationship_type: str = "acquaintance"  # "ally", "enemy", "family", "employer", "employee", "rival", "friend", "acquaintance"
    attitude: str = "neutral"  # "friendly", "unfriendly", "hostile", "helpful", "indifferent", "neutral"
    trust_level: int = 0  # -5 (complete distrust) to +5 (complete trust)
    history: str = ""  # Shared history or how they met
    notes: str = ""  # Additional relationship notes


class CharacterProfile(BaseModel):
    """Profile for NPC, monster, or player character embodiment.

    Used by the character-runner to embody characters with consistent
    personality, knowledge, and behavior.
    """

    id: str
    campaign_id: str
    name: str
    character_type: str = "npc"  # "npc", "monster", or "player"

    # Core personality (all types)
    personality: str = ""
    speech_patterns: str = ""

    # Knowledge and motivations (NPCs and monsters)
    knowledge: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)

    # Rumor propagation (NPCs only)
    chattiness: float = 0.5  # 0.0 to 1.0 - how likely to spread rumors
    is_hub: bool = False  # True for innkeepers, merchants, town criers - accelerates rumor spread

    # Monster-specific
    intelligence: str = "average"  # "animal", "low", "average", "high", "genius"
    instincts: list[str] = Field(default_factory=list)
    morale: str = ""  # e.g., "fights to death", "flees at half HP"

    # Player emulation specific
    playstyle: str = ""
    risk_tolerance: str = "medium"  # "low", "medium", "high"
    decision_making: str = ""
    party_role: str = ""
    quirks: list[str] = Field(default_factory=list)

    # Relationships
    relationships: list[Relationship] = Field(default_factory=list)

    # Faction membership
    faction_ids: list[str] = Field(default_factory=list)

    # Metadata
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Faction(BaseModel):
    """Faction or organization in the campaign.

    Factions have shared goals, resources, knowledge, and members.
    They track reputation with the party and relationships with other factions.
    """

    id: str
    campaign_id: str
    name: str
    description: str = ""

    # Faction identity
    goals: list[str] = Field(default_factory=list)  # Faction objectives
    resources: list[str] = Field(default_factory=list)  # What the faction controls/owns
    shared_knowledge: list[str] = Field(default_factory=list)  # Knowledge IDs all members know

    # Membership
    member_character_ids: list[str] = Field(default_factory=list)  # Characters in this faction
    leader_character_id: str | None = None  # Optional faction leader

    # Relationships
    reputation_with_party: int = 0  # -100 (hostile) to +100 (allied)
    inter_faction_attitudes: dict[str, str] = Field(default_factory=dict)  # faction_id -> attitude

    # Metadata
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Location(BaseModel):
    """Location or area in the campaign world.

    Locations have common knowledge that all NPCs in that location would know.
    They can be connected to other locations and have varying isolation levels.
    """

    id: str
    campaign_id: str
    name: str
    description: str = ""

    # Knowledge and events
    common_knowledge: list[str] = Field(default_factory=list)  # Knowledge IDs everyone here knows
    recent_events: list[str] = Field(default_factory=list)  # Recent happenings at this location

    # Connectivity
    isolation_level: str = "connected"  # "connected", "remote", "isolated"
    connected_locations: list[str] = Field(default_factory=list)  # Location IDs connected to this one

    # Metadata
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Rumor(BaseModel):
    """Rumor or piece of information that spreads through the campaign world.

    Rumors have accuracy that degrades over time, spread at different rates,
    and propagate through connected locations based on NPC chattiness.
    """

    id: str
    campaign_id: str
    content: str
    source_type: str = "pc_seeded"  # "pc_seeded", "event", "npc_created"

    # Accuracy and distortion
    accuracy: float = 1.0  # 0.0 to 1.0 (1.0 = completely accurate)
    distortion_rate: float = 0.05  # How much accuracy decreases per hop

    # Propagation
    spread_rate: str = "medium"  # "slow", "medium", "fast"
    current_locations: list[str] = Field(default_factory=list)  # Location IDs where rumor is known
    known_by_characters: list[str] = Field(default_factory=list)  # Character IDs who know this rumor

    # History
    created_at: datetime = Field(default_factory=datetime.now)
    spread_history: list[dict[str, Any]] = Field(default_factory=list)  # [{timestamp, from_location, to_location, accuracy}]

    # Metadata
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class Secret(BaseModel):
    """Important secret or plot-critical information in the campaign.

    Secrets track who knows them, when they're revealed, and consequences
    that trigger upon revelation.
    """

    id: str
    campaign_id: str
    content: str
    importance: str = "major"  # "minor", "major", "critical"

    # Who knows this secret
    known_by_character_ids: list[str] = Field(default_factory=list)  # Characters who know
    known_by_faction_ids: list[str] = Field(default_factory=list)  # Factions whose members know

    # Revelation tracking
    revealed_to_party: bool = False
    revelation_event: dict[str, Any] | None = None  # {timestamp, session_id, turn_number, revealer, method}

    # Consequences
    consequences: list[str] = Field(default_factory=list)  # Descriptions of what happens when revealed
    triggered_consequences: list[str] = Field(default_factory=list)  # Which consequences have fired

    # Metadata
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
