# Future Improvements

## Completed

### Multi-System Architecture (RFC Phases 0-2) -- DONE
- [x] `GameSystem` ABC with `@register_system` decorator
- [x] `game_systems` / `primary_system` fields on Campaign model
- [x] PF2e code moved to `gm_agent/systems/pf2e/`
- [x] `FictionTreeStore` (SQLite-backed hierarchical fiction storage)
- [x] Microscope system (24 tools, seeds, oracles, mini-games, virtual players)
- [x] Ex Novo settlement generation system (13 tools)
- [x] Delve underground kingdom system (13 tools)
- [x] Ex Umbra dungeon generation system (13 tools)
- [x] Cross-system fiction-to-knowledge extraction (`fiction_extraction.py`)
- [x] GroundingServer (6 tools) with LLM reranking for fiction-to-mechanics grounding

### Plugin Architecture (Phase 3 WS1) -- DONE
- [x] `SystemToolPlugin` protocol on `base.py`
- [x] `PF2eCampaignToolPlugin` extracts 4 PF2e-specific tools (travel time, hazard detection, AP progress, treasure)
- [x] CampaignStateServer accepts `system_plugins` param, injects tool defs, delegates call_tool

### LLM-Enhanced Grounding (Phase 3 WS2) -- DONE
- [x] `_llm_rerank()` on GroundingServer — LLM selects best matches with reasoning
- [x] Graceful fallback to raw BM25 results on LLM error
- [x] Reasoning display in output

### Fiction Knowledge Extraction (Phase 3 WS3) -- DONE
- [x] `extract_from_microscope()` — root/palette/periods/events/scenes to knowledge
- [x] `extract_from_settlement()` — factions/districts/resources/problems/landmarks
- [x] `extract_from_dungeon()` — rooms/details for Delve and Ex Umbra
- [x] `extract_fiction_knowledge` tool on CampaignStateServer
- [x] Idempotent via `has_similar_knowledge()` dedup

### Virtual Player Memory (Phase 3 WS4) -- DONE
- [x] `decision_history` param on `generate_decision()`
- [x] History prompt injection with truncation (max 5 recent)
- [x] All 4 generation game servers track and pass VP history

### Personality System -- DONE
- [x] PersonalityProfile (50 traits, 20 archetypes)
- [x] Virtual player personality integration in CharacterRunner
- [x] Archetype-based decision style injection

### Campaign Prep & Crunch -- DONE
- [x] PrepPipeline: party/NPC/subsystem/world knowledge seeding (LLM-synthesized)
- [x] CrunchPipeline: post-session event extraction, dialogue, knowledge updates, arc updates
- [x] Prep log (JSONL training data)
- [x] CLI: `campaign prep`, `campaign crunch`, `campaign generate`

### Cross-System Propagation -- DONE
- [x] PropagationBus mediator (secrets, factions, locations)
- [x] `on_secret_revealed()`, `on_faction_knowledge_added()`, `on_npc_joins_faction()`
- [x] Location-based knowledge pull model

### NPC & World Systems -- DONE
- [x] NPC knowledge with conditional sharing (trust, persuasion DC, duress)
- [x] Party knowledge (`__party__` virtual character)
- [x] Faction system (membership, shared knowledge, reputation)
- [x] Location system (connected graph, knowledge, events)
- [x] Secret & revelation tracking
- [x] Dialogue history (SQLite FTS5)
- [x] Session recap tool

### Creature & Encounter Tools -- DONE
- [x] CreatureModifierServer (elite/weak, templates, scaffold creature/hazard/troop/swarm)
- [x] SubsystemServer (VP, influence, research, chase, infiltration, hazard, exploration)
- [x] EncounterServer with random encounter generation
- [x] AP progress tracking, treasure management

---

## In Progress / Future

### Async Processing Optimization (Phase 4.2 Follow-up)

**Current Architecture:**
Phase 4.2 implements Redis locks for campaign serialization. Works correctly but has room for optimization.

**Improved Architecture: Campaign-Specific Workers (Defense in Depth)**

**Primary mechanism**: Campaign-specific routing
- Route all tasks for campaign-X to same worker queue
- Single-threaded worker = automatic serialization
- No lock contention in normal operation

**Safety mechanism**: Redis locks (guard rails)
- Catch misrouting bugs
- Protect during worker restarts/failover
- **Lock acquisition should succeed immediately** (uncontended)
- If locks ever block in production → bug signal (routing issue)

**Implementation Path:**
1. Add campaign-based routing to celery_app.py
2. Update worker startup docs
3. Add lock contention monitoring/alerting
4. Extract LLM calls to separate task pool

**Priority:** Medium (optimization, current approach is correct)

### State Storage Migration (Post-Async Processing)

Current file-based storage works with campaign-level task locking but may need migration for scaling.

**Recommendation**: Stick with files + locks for now, migrate to PostgreSQL when:
- Multiple simultaneous games per campaign needed
- Sub-second response times required
- Deployment scales beyond single-instance

**Priority:** Low (defer until scaling needed)

### Fine-Tuning Pipeline

**Data Collection (Infrastructure In Place):**
- Complete session storage with structured JSON
- Full metadata: player input, GM response, tool calls, timing, model used
- LLM thinking trace capture (`LLMResponse.thinking` field, all backends)
- Prep system JSONL training logs (`campaigns/{id}/prep_log.jsonl`)
- Session replay for quality verification

**Still Needed:**
- [ ] Store thinking traces in `TurnMetadata` during sessions (field exists on LLMResponse but not persisted to session JSON yet)
- [ ] Build dataset export tool (sessions → JSONL training format)
- [ ] Run 50+ production sessions with thinking-enabled models

**Training Pipeline (Not Started):**
- [ ] Quality filtering pipeline (error turns, short responses, stratification)
- [ ] QLoRA fine-tuning infrastructure (Llama/Qwen 8B base)
- [ ] Evaluation framework (tool accuracy, rules adherence, narrative quality)
- [ ] 4-bit GPTQ quantization for local deployment

**Multimodal Integration (Future):**
- [ ] Image generation pipeline (scene illustrations via Flux)
- [ ] Video generation for dramatic moments (experimental)

**Priority:** Long-term

### Equipment Sub-Typing in pf2e-extraction

Weapons, armor, shields, runes, and consumables are all stored as flat `equipment`/`item` types in
search.db. This prevents gm-agent from dynamically loading weapon/rune name lists for search query
decomposition (these remain hardcoded in `_decompose_complex_query` in `search.py`).

**Needed in pf2e-extraction:**
- [ ] Add fine-grained `type` values: `weapon`, `armor`, `shield`, `rune`, `consumable`, `worn_item`, etc.
- [ ] Or add a `subtype` column to `content` for equipment classification

Once available, gm-agent can load weapon/rune names from the DB at init (like conditions and classes).

**Priority:** Low (hardcoded lists work, just won't auto-update with new content)

### New Game Systems (RFC Phases 3-5)

Phase 0-2 of the Multi-System RFC are complete. Future systems to implement:

- [ ] **Ironsworn/Starforged** — Oracle tables, moves, progress tracks
- [ ] **Blades in the Dark** — Clocks, scores, crew sheet, faction turns
- [ ] **How to Host a Dungeon** — Layered geological history → dungeon rooms
- [ ] **The Quiet Year** — Seasonal/card-driven procedure for session 0 worldbuilding
- [ ] ***Without Number** — Faction turns, sandbox generators, hex/sector maps

**Priority:** Medium (foundation is solid, add based on community interest)

### Encounter Execution

Two mutually exclusive modes per campaign. Pick one.

#### Mode 1: Foundry VTT Puppeting (BUILT)

The `foundryvtt-pf2e-gm-agent` module already provides full combat automation (30+ commands, bidirectional bridge).

**What's missing for full agent-driven combat:**
- [ ] Agent combat loop — backend receives `combatTurn` for NPC but doesn't yet decide and execute a full turn
- [ ] Action sequencing — agent needs to chain commands within a turn (Stride → Strike → Strike with MAP tracking)
- [ ] Tactical decision-making — use creature enrichment metadata (tactics, morale, behavior)

#### Mode 2: Theater of the Mind (NOT BUILT)

Agent-native encounter execution — no Foundry dependency, no grid.

**Missing pieces:**
- [ ] Narrative positioning layer (lightweight zones instead of grid squares)
- [ ] Turn orchestration (agent drives initiative loop)
- [ ] Encounter lifecycle (start/end hooks with auto-stat/treasure/XP)
- [ ] Condition automation (persistent damage, frightened reduction, dying/recovery)
- [ ] Action economy tracking (3 actions + reaction, MAP)

**Priority:** Low (only needed if Foundry puppeting doesn't cover the use case)

### Agent Integrations

- [ ] Check Pathfinder Wiki agent implementation
- [ ] Check Archives of Nethys agent implementation
- [ ] Check Paizo Forums agent implementation

### Analytics & Tuning

**Response Time Analytics:**
- Dashboard for latency percentiles
- Alert on degraded performance
- Correlation with tool usage

**Tool Usage Analytics:**
Current: TurnMetadata includes `tool_count` and `tool_usage` dict.

Future:
- Per-tool success/failure rates
- Identify unused or underutilized tools
