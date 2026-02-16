# Multi-Game-System Architecture RFC

## Context

The GM Agent currently supports Pathfinder 2e exclusively. The conversation driving this RFC explored supporting additional game systems in two categories:

1. **Full RPGs** — played through the agent as GM: Blades in the Dark, Ironsworn/Starforged, Heart/Spire, *Without Number, Legacy
2. **Procedural generation games** — output fiction/setting that feeds into RPGs: Microscope, Ex Novo, Ex Umbra, Delve, How to Host a Dungeon, The Quiet Year

The key insight: procedural generation games are just game systems whose output is *setting fiction* instead of *play experiences*. The same architecture (corpus + tools + state tracking) supports both. Cross-system composition (run Ex Novo → feed into PF2e encounters) emerges naturally.

The Microscope zoom-in/zoom-out paradigm — a fractal hierarchy where any node can be expanded to arbitrary detail — is a useful shared representation underneath all of these.

This is an architectural sketch, not an implementation plan. It documents what the target looks like so we can plan incremental work toward it.

---

## A. Game System Plugin Architecture

### What a game system provides

Each game system is a Python package under `gm_agent/systems/` that registers:

```
gm_agent/systems/
    __init__.py          # GameSystem ABC, registry
    pf2e/
        __init__.py      # PF2eSystem(GameSystem) — registers everything
        servers.py       # PF2eRAGServer, EncounterServer, CreatureModifierServer
        prompts.py       # GM_SYSTEM_PROMPT, tool usage instructions
        stores.py        # APProgressStore, TreasureStore (PF2e-specific)
        tables.py        # CREATURE_STATS_BY_LEVEL, TREASURE_BY_LEVEL, etc.
    blades/
        __init__.py      # BladesSystem(GameSystem)
        servers.py       # ClockServer, ScoreServer, FactionTurnServer
        prompts.py       # Blades GM prompt
        tables.py        # Position/effect matrix, faction tier table
    microscope/
        __init__.py      # MicroscopeSystem(GameSystem)
        servers.py       # TimelineServer (period/event/scene CRUD + focus)
        prompts.py       # Facilitator prompt
    ironsworn/
        __init__.py      # IronswornSystem(GameSystem)
        servers.py       # OracleServer, MoveServer, ProgressTrackServer
        prompts.py       # Oracle/GM prompt
        tables.py        # Oracle tables, move reference
    ...
```

### The GameSystem ABC

```python
class GameSystem(ABC):
    name: str                           # "pf2e", "blades", "microscope"
    display_name: str                   # "Pathfinder 2e (Remaster)"
    category: str                       # "rpg" | "generation"

    @abstractmethod
    def servers(self, context: dict) -> list[MCPServer]:
        """Return MCP servers for this system, given campaign context."""

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt fragment for this game system."""

    def stores(self, campaign_id: str) -> dict[str, Any]:
        """Return system-specific stores. Default: none."""
        return {}

    def prep_pipeline(self) -> PrepPipeline | None:
        """Return a prep pipeline if this system supports knowledge seeding."""
        return None
```

### Registration and discovery

```python
# gm_agent/systems/__init__.py
SYSTEM_REGISTRY: dict[str, type[GameSystem]] = {}

def register_system(cls: type[GameSystem]):
    SYSTEM_REGISTRY[cls.name] = cls
    return cls

def get_system(name: str) -> GameSystem:
    return SYSTEM_REGISTRY[name]()
```

Each system's `__init__.py` calls `@register_system` on its class. Auto-discovery via package scanning or explicit imports in `systems/__init__.py`.

### Campaign model gains a `game_systems` field

```python
class Campaign(BaseModel):
    id: str
    name: str
    game_systems: list[str] = ["pf2e"]   # NEW — enables multi-system campaigns
    primary_system: str = "pf2e"          # NEW — drives prompt and default tools
    # ... existing fields unchanged
```

A Legacy campaign might be `game_systems=["legacy", "microscope"]`. A PF2e campaign using Ex Novo for session 0 worldbuilding would be `game_systems=["pf2e", "ex_novo"]`.

### MCPClient adapts to campaign systems

```python
# In _init_local_servers():
for system_name in campaign.game_systems:
    system = get_system(system_name)
    for server in system.servers(context):
        self._local_servers[f"{system_name}:{server.name}"] = server
```

Tool namespace becomes `system:tool_name` internally but the LLM sees flat names (collision handled by prefix only when ambiguous). The existing PF2e servers migrate into the `pf2e` system package — no behavioral change, just reorganization.

### Context building composes system prompts

```python
def build_system_prompt(campaign: Campaign) -> str:
    parts = []
    primary = get_system(campaign.primary_system)
    parts.append(primary.system_prompt())

    for name in campaign.game_systems:
        if name != campaign.primary_system:
            system = get_system(name)
            parts.append(f"\n## {system.display_name} Tools\n{system.tool_guidance()}")

    return "\n\n".join(parts)
```

---

## B. The Fiction Tree (Microscope-Inspired Shared Structure)

### The core idea

A **fiction tree** is a hierarchical structure where each node represents a piece of established fiction at a specific zoom level. Any node can be expanded into children with more detail. The tree is the shared canvas that all systems read from and write to.

This is not a replacement for existing stores — it's a **narrative index** that links to them. A fiction tree node saying "The Dwarven Citadel" might link to a LocationStore entry, a set of KnowledgeStore entries, and a batch of encounter locations in APProgressStore.

### Data model

```python
class FictionNode(BaseModel):
    id: str                                # uuid
    campaign_id: str
    parent_id: str | None                  # null = root
    zoom_level: str                        # "era" | "period" | "event" | "scene" | "detail"
    tone: str | None                       # "light" | "dark" | "ambiguous" (Microscope concept)
    title: str                             # "The Fall of Kolvar Citadel"
    summary: str                           # 1-3 sentences
    content: str                           # Full description (can be long)
    tags: list[str]                        # ["dwarven", "military", "loss"]
    source_system: str                     # "microscope" | "ex_novo" | "pf2e" | "manual"

    # Cross-references to other stores
    linked_locations: list[str]            # LocationStore IDs
    linked_characters: list[str]           # CharacterStore IDs
    linked_knowledge: list[str]            # KnowledgeStore IDs

    # Microscope-specific
    palette_yes: list[str]                 # "yes" list items (things allowed)
    palette_no: list[str]                  # "no" list items (things banned)

    # Ordering
    sort_order: int                        # Position among siblings
    created_at: datetime
    updated_at: datetime
```

### Zoom levels map to different tools

| Zoom Level | Microscope Term | RPG Equivalent | What the agent does |
|---|---|---|---|
| `era` | Period | Campaign arc / Age | Theme tracking, arc coherence |
| `period` | Event | Chapter / Quest | Plot synthesis, faction moves |
| `event` | Scene | Session / Encounter | Narration, scene-setting |
| `scene` | (drill-down) | Encounter room / Moment | Mechanical grounding (stats, DCs, maps) |
| `detail` | — | Item / NPC trait / Rule | Specific lookup, statblock |

### FictionTreeStore

```python
class FictionTreeStore:
    """Hierarchical fiction storage. SQLite-backed for queryable traversal."""

    def __init__(self, campaign_id: str, base_dir: Path | None = None): ...

    # CRUD
    def add_node(self, parent_id: str | None, zoom_level: str, title: str, ...) -> FictionNode
    def get_node(self, node_id: str) -> FictionNode | None
    def update_node(self, node_id: str, **fields) -> FictionNode
    def delete_node(self, node_id: str, cascade: bool = False) -> None

    # Navigation (the zoom-in/zoom-out operations)
    def get_children(self, node_id: str) -> list[FictionNode]
    def get_ancestors(self, node_id: str) -> list[FictionNode]   # path to root
    def get_roots(self) -> list[FictionNode]                     # top-level eras/periods
    def get_siblings(self, node_id: str) -> list[FictionNode]

    # Search
    def search(self, query: str, zoom_level: str | None = None) -> list[FictionNode]
    def by_tag(self, tag: str) -> list[FictionNode]
    def by_source_system(self, system: str) -> list[FictionNode]

    # Bulk operations for cross-system composition
    def export_subtree(self, node_id: str) -> dict       # JSON-serializable tree
    def import_subtree(self, parent_id: str, data: dict, source_system: str) -> list[str]
```

### How generation games write to the tree

**Microscope**: `TimelineServer` tools (`create_period`, `create_event`, `create_scene`, `set_focus`, `set_palette`) write directly to the fiction tree. The facilitator prompt guides the human+agent through the Microscope procedure. Each Microscope action creates one or more `FictionNode` entries.

**Ex Novo**: `SettlementServer` tools (`add_faction`, `add_landmark`, `add_problem`, `advance_era`) write nodes at the `event` and `scene` zoom levels under a parent `period` node representing the settlement. A completed Ex Novo session produces a subtree: settlement (period) → founding/growth/current (events) → specific landmarks, factions, problems (scenes).

**How to Host a Dungeon**: `DungeonHistoryServer` tools write geological era → primordial civilization → age of monsters → present day as nested nodes. Each civilization layer adds location nodes that become the keyed dungeon rooms.

**Delve**: Similar to How to Host a Dungeon but focused on underground kingdom building. Writes discovery/expansion/collapse nodes.

### How RPGs read from the tree

The fiction tree is **read-only from the RPG's perspective** during play (the RPG adds to it via session events, but doesn't modify generation-game nodes). The RPG agent sees the tree through:

1. **Context injection** — `build_context()` includes a summary of relevant fiction tree nodes at the current zoom level (e.g., if the party is in the Dwarven Citadel, include that node's summary and its immediate children)
2. **Tool access** — `browse_fiction_tree` tool lets the agent navigate the tree during play ("what do we know about the history of this place?")
3. **Knowledge seeding** — a prep step converts fiction tree nodes into KnowledgeStore entries at appropriate importance levels

---

## C. Cross-System Composition

### The pattern: generate → link → play

```
1. GENERATE:  Run Microscope → timeline tree
2. GENERATE:  Run Ex Novo for a key settlement → settlement subtree
3. GENERATE:  Run Ex Umbra for a ruin → dungeon subtree
4. LINK:      Attach subtrees to timeline nodes
5. GROUND:    PF2e prep converts tree nodes → encounters, NPCs, knowledge
6. PLAY:      Run PF2e campaign with grounded content
```

### Export/import between systems

The fiction tree is the interchange format. Each generation game writes to it in a structured way, and the consuming RPG reads from it. No separate export/import format needed — the tree *is* the shared state.

For external tools (if someone builds a settlement in Ex Novo on paper and wants to import it), a simple JSON format mapping to `FictionNode` fields suffices.

### Mechanical grounding step

The gap between "fiction tree says there's a Dwarven Citadel overrun by duergar" and "PF2e encounter with 3x Duergar Taskmaster (L7) in room B3" is bridged by a **grounding step** — either:

- **Automated**: A prep pipeline step that takes fiction tree nodes tagged as "encounter-worthy" and uses the PF2e encounter/creature tools to scaffold mechanically sound encounters
- **Agent-assisted**: The agent proposes encounters based on tree context, human approves/modifies
- **Manual**: Human keys encounters using the creature scaffolding tools, links them back to tree nodes

The grounding step is system-specific — PF2e grounds differently than Blades in the Dark (which would ground to clock setups and faction positions rather than statblocks).

---

## D. System Prompt Architecture

### Per-system prompt composition

Each `GameSystem` provides a `system_prompt()` that defines the agent's role, accuracy rules, and tool usage guidance for that system. The primary system's prompt is the base; secondary systems append tool guidance sections.

```
[Primary system prompt - full role definition]
  e.g., "You are a Game Master for Pathfinder 2e (Remaster)..."
  or    "You are a facilitator for a Microscope session..."

[Campaign context - from Campaign model, system-agnostic]
  Background, current arc, party info

[Secondary system tool guidance - brief]
  e.g., "You also have access to Microscope timeline tools for reviewing
         the campaign's established history..."

[Session context - from Session model, system-agnostic]
  Recent turns, scene state, NPC hints
```

### RPG vs generation game prompt differences

| Aspect | RPG Prompt | Generation Game Prompt |
|---|---|---|
| Agent role | GM — runs the game, controls NPCs | Facilitator — guides procedure, proposes options |
| Accuracy rules | "Never invent mechanics — look them up" | "Follow the game's procedure — propose, don't decide" |
| Initiative | Agent drives narration between player inputs | Agent presents choices, human decides at each step |
| State awareness | Scene state, combat, conditions | Fiction tree position, palette constraints, focus |

### The generation game prompt pattern

```
You are facilitating a {game_name} session. Your role is to:
- Guide the group through the {game_name} procedure step by step
- Propose options when it's time to create new {elements}
- Respect the palette (what's allowed and banned)
- Track the current focus/theme
- Record decisions to the fiction tree

You do NOT:
- Make creative decisions unilaterally — always offer 2-3 options or ask
- Skip procedural steps
- Contradict established fiction

Current state: {procedure_state}
Palette: {yes_list} / {no_list}
Focus: {current_focus}
```

---

## E. What Each Target System Looks Like

### Microscope

**Servers**: `TimelineServer` (campaign-scoped, ~8 tools)
- `create_period`, `create_event`, `create_scene` — write to fiction tree
- `set_focus`, `set_palette_yes`, `set_palette_no` — session-level state
- `browse_timeline`, `get_current_focus` — navigation

**State**: Fiction tree nodes + session-level focus/palette (stored on Session or as a SubsystemInstance)

**Prompt**: Facilitator role. Guides Microscope's specific turn structure (each player picks a type of action: add period, add event, dictate/play scene).

**Corpus**: The Microscope rulebook itself (small — ~80 pages). Could be embedded in the prompt rather than RAG'd.

### Ex Novo / Ex Umbra

**Servers**: `SettlementServer` / `RuinServer` (campaign-scoped, ~6-8 tools each)
- `add_faction`, `add_landmark`, `add_resource`, `add_problem`, `advance_era`
- `roll_event`, `resolve_event` (card/dice-driven procedure)
- `get_settlement_state`, `browse_map`

**State**: Fiction tree subtree + settlement-specific state (factions, resources, problems as structured data on config dict or dedicated model)

**Cross-system value**: Settlement nodes become PF2e locations. Factions become FactionStore entries. Landmarks become encounter sites. Problems become quest hooks.

### How to Host a Dungeon

**Servers**: `DungeonHistoryServer` (campaign-scoped, ~6 tools)
- `add_geological_feature`, `establish_civilization`, `civilization_event`, `advance_age`, `add_monster_group`, `finalize_present_day`

**State**: Layered dungeon history in the fiction tree (each age is a period, each civilization event is an event, rooms are scenes)

**Cross-system value**: The richest source of "dungeon with history." Each room has narrative archaeology — why it exists, who built it, what happened there. This is exactly what Ex Umbra also provides but through a different procedure.

### Ironsworn / Starforged

**Servers**: `OracleServer` (stateless, ~4 tools), `MoveServer` (campaign-scoped, ~6 tools), `ProgressTrackServer` (campaign-scoped, ~4 tools)
- Oracle: `ask_oracle`, `roll_on_table`, `get_oracle_tables`
- Moves: `make_move`, `pay_the_price`, `get_move_reference`
- Progress: `create_track`, `mark_progress`, `make_progress_roll`, `list_tracks`

**State**: Character momentum/health/spirit/supply as session state. Progress tracks as SubsystemStore instances. Bonds/vows as KnowledgeStore entries.

**Prompt**: The agent is the oracle — interprets move results in fiction, asks "what do you do?" after each outcome, manages the conversation between mechanical triggers and narrative consequences.

### Blades in the Dark

**Servers**: `ClockServer` (campaign-scoped, ~5 tools), `ScoreServer` (campaign-scoped, ~4 tools), `CrewServer` (campaign-scoped, ~5 tools), `FactionTurnServer` (campaign-scoped, ~3 tools)
- Clocks: `create_clock`, `tick_clock`, `get_clocks`, `clear_clock`, `list_clocks`
- Score: `start_score`, `set_position_effect`, `resolve_action`, `end_score`
- Crew: `get_crew_sheet`, `spend_coin`, `downtime_action`, `advance_crew`
- Factions: `faction_turn`, `get_faction_status`, `adjust_faction_tier`

**State**: Clocks are SubsystemStore instances. Crew sheet is a CharacterStore entry for the crew-as-entity. Faction game maps to existing FactionStore with tier/hold as fields.

**Prompt**: Agent tracks position/effect, proposes consequences, manages the fiction-mechanics loop. Heavy interpretive load — "you're in a desperate position with limited effect, what does that look like when you're trying to sneak past the Spirit Wardens?"

### *Without Number (Stars/Worlds/Cities)

**Servers**: `FactionTurnServer` (campaign-scoped, ~4 tools), `SandboxGenServer` (stateless, ~6 tools)
- Faction: `run_faction_turn`, `faction_action`, `get_faction_sheet`, `adjust_hp_assets`
- Sandbox: `generate_adventure_site`, `generate_npc`, `roll_tag`, `generate_encounter`, `get_table`, `generate_sector` (Starforged) / `generate_hex` (Worlds)

**State**: Faction sheets with HP, assets, goals. Tags on locations/factions. Hex map as fiction tree nodes.

**Cross-system value**: The tag system and sandbox generators are the most table-driven of any system here. Almost pure "roll on table, interpret result." The faction turn system is also highly mechanical and could share infrastructure with Blades' faction game.

---

## F. Migration Path

### Phase 0: Groundwork (no behavioral change)

1. **Add `game_systems` and `primary_system` to Campaign model** with defaults `["pf2e"]` / `"pf2e"` — backward compatible
2. **Define `GameSystem` ABC** in `gm_agent/systems/__init__.py`
3. **Move PF2e-specific code into `gm_agent/systems/pf2e/`** — reexport from original locations for backward compat
4. **Create `PF2eSystem(GameSystem)`** that returns existing servers/prompt/stores
5. **Parameterize `GM_SYSTEM_PROMPT`** — load from `GameSystem.system_prompt()` instead of hardcoded constant

All existing tests pass unchanged. The PF2e path is the only path.

### Phase 1: Fiction Tree + Microscope

1. **Implement `FictionTreeStore`** (SQLite-backed, under `gm_agent/storage/`)
2. **Implement `MicroscopeSystem`** with `TimelineServer`
3. **Add `browse_fiction_tree` tool** to a shared server (available to all systems)
4. **Test**: Run a Microscope session through the agent, verify fiction tree populates correctly

This is the first non-PF2e system and validates the plugin architecture. Microscope is ideal first because it's simple (no dice, no stats, just narrative structure) and the fiction tree it produces is the foundation for everything else.

### Phase 2: Ex Novo + Cross-System Composition

1. **Implement `ExNovoSystem`** with `SettlementServer`
2. **Implement grounding step**: fiction tree nodes → PF2e locations + NPCs + encounters
3. **Test**: Run Ex Novo → verify settlement feeds into PF2e prep pipeline

This validates cross-system composition. The "generation game output → RPG input" pipeline is the key architectural proof point.

### Phase 3: Table-Driven Systems

1. **Implement a shared `OracleTableServer`** pattern — loads tables from JSON/YAML per system
2. **Implement one of**: How to Host a Dungeon, Delve, or Ex Umbra (all are table + procedure driven)
3. **Generalize the SubsystemStore** if needed for system-specific state shapes

### Phase 4: Full RPG Systems

1. **Implement Ironsworn or Blades** — whichever has the most community interest
2. **Validate that KnowledgeStore, FactionStore, etc. work unchanged** for a non-PF2e RPG
3. **Test the full loop**: generation → grounding → play

### Phase 5: Advanced Composition

1. **Legacy support** — multi-generational play with fiction tree age transitions
2. **Cross-RPG composition** — e.g., use Blades' faction game to simulate political intrigue in a PF2e campaign
3. **The Quiet Year** — seasonal/card-driven procedure as a session 0 tool for any RPG

---

## G. Key Architectural Decisions

### 1. Fiction tree is a new store, not a replacement

The fiction tree complements existing stores (Location, Character, Knowledge). It's a narrative index — "here's the big picture" — with cross-references to detailed entries in specialized stores. A fiction tree node titled "The Siege of Kolvar" links to LocationStore entries for the citadel, CharacterStore entries for the besiegers, and KnowledgeStore entries for what the party knows about it.

### 2. Generation games write structured state, RPGs read it

The fiction tree is write-primary for generation games and read-primary for RPGs. During play, the RPG adds new nodes (session events, discoveries) but doesn't modify generation-game nodes. This preserves the "established fiction" contract from Microscope.

### 3. Tool namespace stays flat with system prefix on collision only

Most tools have unique enough names (`create_period`, `scaffold_creature`) that no prefix is needed. If two systems define `roll_on_table`, the client prefixes: `ironsworn:roll_on_table`, `swn:roll_on_table`. The agent sees clear names without constant namespace noise.

### 4. Each system owns its own RAG (if it has one)

PF2e has `pathfinder_search.db`. Ironsworn would have `ironsworn_search.db` (much smaller — just the rulebook). Microscope has no RAG (rules fit in the prompt). Each system's `servers()` method returns a RAG server pointing at its own DB, or none.

### 5. SubsystemStore generalizes via the config dict pattern

Rather than creating a new state store for every game system's mini-games, we extend the existing `SubsystemStore` pattern. Blades clocks, Ironsworn progress tracks, and PF2e victory point encounters all fit the same shape: named instance + numeric state + action log + status lifecycle. System-specific fields go in `config`.

### 6. Prep pipelines are per-system and composable

Each system can define a prep pipeline. PF2e has party/NPC/subsystem/world knowledge seeding. A Microscope-first campaign might have a prep step that converts timeline nodes into knowledge entries. A Blades campaign might seed faction knowledge from the crew's starting situation. The pipeline runner checks `campaign.game_systems` and runs each system's prep in order.

---

## Files That Would Change (Phase 0)

| File | Change |
|---|---|
| `gm_agent/systems/__init__.py` | NEW — GameSystem ABC, registry |
| `gm_agent/systems/pf2e/__init__.py` | NEW — PF2eSystem wrapping existing servers |
| `gm_agent/storage/schemas.py` | Add `game_systems`, `primary_system` to Campaign |
| `gm_agent/config.py` | Extract GM_SYSTEM_PROMPT to PF2eSystem.system_prompt() |
| `gm_agent/context.py` | Load system prompt from GameSystem instead of constant |
| `gm_agent/mcp/registry.py` | Support system-provided server discovery |
| `gm_agent/mcp/client.py` | Instantiate servers from campaign.game_systems |

## Files That Would Be New (Phase 1)

| File | Purpose |
|---|---|
| `gm_agent/storage/fiction_tree.py` | FictionTreeStore |
| `gm_agent/systems/microscope/__init__.py` | MicroscopeSystem |
| `gm_agent/systems/microscope/servers.py` | TimelineServer |
| `gm_agent/systems/microscope/prompts.py` | Facilitator prompt |
| `tests/test_fiction_tree.py` | Fiction tree store tests |
| `tests/test_microscope.py` | Microscope system tests |
