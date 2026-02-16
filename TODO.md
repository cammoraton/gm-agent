# Future Improvements

## Async Processing Optimization (Phase 4.2 Follow-up)

### Current Architecture
Phase 4.2 implements Redis locks for campaign serialization. Works correctly but has room for optimization.

### Improved Architecture: Campaign-Specific Workers (Defense in Depth)

**Primary mechanism**: Campaign-specific routing
- Route all tasks for campaign-X to same worker queue
- Single-threaded worker = automatic serialization
- No lock contention in normal operation

**Safety mechanism**: Redis locks (guard rails)
- Catch misrouting bugs
- Protect during worker restarts/failover
- Detect race conditions in development
- **Lock acquisition should succeed immediately** (uncontended)
- If locks ever block in production → bug signal (routing issue)

```python
# Task routing (consistent hashing by campaign)
task_routes = {
    'process_player_chat_async': {
        'queue': lambda task_id, args: f"automation:{hash(args[0]) % 4}"
    }
}

# Worker pools
celery worker -Q automation:0  # Handles campaigns hash to 0
celery worker -Q automation:1  # Handles campaigns hash to 1
celery worker -Q automation:2  # etc.
celery worker -Q automation:3

# Task implementation (locks for safety, not primary mechanism)
@task(queue determined by routing above)
def process_player_chat_async(campaign_id, input):
    lock = acquire_campaign_lock(campaign_id)
    if not lock.acquire(blocking=True, timeout=1):
        # Should NEVER happen in normal operation
        logger.error(f"Lock contention for {campaign_id} - routing bug!")
        raise Retry()

    try:
        # Single-threaded per campaign, can delegate I/O
        context = build_context(campaign_id)
        llm_result = call_llm_task.delay(context, input).get()
        save_turn(campaign_id, llm_result)
    finally:
        lock.release()
```

**Worker architecture:**
- `automation:{0-3}`: Campaign coordination (single-threaded per campaign)
- `llm`: Stateless LLM calls (I/O-bound pool)
- `mcp`: Stateless tool execution

**Implementation Path:**
1. Add campaign-based routing to celery_app.py
2. Update worker startup docs
3. Add lock contention monitoring/alerting
4. Extract LLM calls to separate task pool

**Priority:** Medium (optimization, current approach is correct)

## State Storage Migration (Post-Async Processing)

Current file-based storage works with campaign-level task locking but may need migration for scaling.

**Recommendation**: Stick with files + locks for now, migrate to PostgreSQL when:
- Multiple simultaneous games per campaign needed
- Sub-second response times required
- Deployment scales beyond single-instance

**Priority:** Low (defer until scaling needed)

## Fine-Tuning Pipeline

### Vision: Self-Improving GM Agent

Self-sustaining improvement cycle: GM Agent generates training data → fine-tune specialized local models → run on consumer hardware with multimodal capabilities.

### Data Collection (Infrastructure In Place)

**What We Have:**
- Complete session storage with structured JSON
- Full metadata: player input, GM response, tool calls, timing, model used
- LLM thinking trace capture (`LLMResponse.thinking` field, all backends)
- Prep system JSONL training logs (`campaigns/{id}/prep_log.jsonl`)
- Session replay for quality verification

**Still Needed:**
- [ ] Store thinking traces in `TurnMetadata` during sessions (field exists on LLMResponse but not persisted to session JSON yet)
- [ ] Build dataset export tool (sessions → JSONL training format)
- [ ] Run 50+ production sessions with thinking-enabled models

### Training Pipeline (Not Started)
- [ ] Quality filtering pipeline (error turns, short responses, stratification)
- [ ] QLoRA fine-tuning infrastructure (Llama/Qwen 8B base)
- [ ] Evaluation framework (tool accuracy, rules adherence, narrative quality)
- [ ] 4-bit GPTQ quantization for local deployment

### Multimodal Integration (Future)
- [ ] Image generation pipeline (scene illustrations via Flux)
- [ ] Video generation for dramatic moments (experimental)

### Legal Stance
**Maximalist/conservative** for public distribution: fine-tune your own model with your own data, share code/pipeline/approach, but don't distribute weights or datasets containing copyrighted Paizo content or TOS-restricted LLM outputs.

**Priority:** Long-term

## Equipment Sub-Typing in pf2e-extraction

Weapons, armor, shields, runes, and consumables are all stored as flat `equipment`/`item` types in
search.db. This prevents gm-agent from dynamically loading weapon/rune name lists for search query
decomposition (these remain hardcoded in `_decompose_complex_query` in `search.py`).

**Needed in pf2e-extraction:**
- [ ] Add fine-grained `type` values: `weapon`, `armor`, `shield`, `rune`, `consumable`, `worn_item`, etc.
- [ ] Or add a `subtype` column to `content` for equipment classification

Once available, gm-agent can load weapon/rune names from the DB at init (like conditions and classes).

**Priority:** Low (hardcoded lists work, just won't auto-update with new content)

## Agent Integrations

- [ ] Check Pathfinder Wiki agent implementation
- [ ] Check Archives of Nethys agent implementation
- [ ] Check Paizo Forums agent implementation

## Automation Mode Enhancements

### Configurable Prompts
Allow customization of the prompts used for player chat responses and NPC turn handling.
- Template variables for actor name, content, scene state
- Per-campaign prompt customization
- Default prompts with sensible defaults

### Message Queuing
Queue incoming events to handle bursts of activity gracefully.
- Priority ordering (combat turns vs exploration chat)
- Queue depth limits, timeout/expiry for stale messages
- Backpressure signaling to Foundry

### Dry Run Mode
Test automation behavior without posting to Foundry.
- Log what would be posted
- Useful for prompt tuning and comparing configurations

## Encounter Execution

Two mutually exclusive modes per campaign. Pick one.

### Mode 1: Foundry VTT Puppeting (BUILT)

The `foundryvtt-pf2e-gm-agent` module already provides full combat automation:
- **30+ commands**: `applyDamage`, `applyCondition`, `rollCheck`, `advanceTurn`, `spawnToken`, `moveToken`, etc.
- **Bidirectional bridge**: Socket.IO (WebSocket) + HTTP polling fallback
- **Full automation mode**: agent receives `combatTurn`/`playerChat` events, issues commands back
- **ACA integration**: NPC designations, AI suggestions, persistent notes

**Deployment topology:**
- **Local Foundry + local gm-agent**: WebSocket mode (Foundry connects to gm-agent :5000)
- **Forge-hosted Foundry + local gm-agent**: Polling mode (`FOUNDRY_MODE=polling`).
  gm-agent reaches out to Forge via HTTP — no inbound ports needed on local infra.
  Foundry module exposes `/api/gm-agent/events` (long-poll) and `/api/gm-agent/command`.
  Poll interval default 2s, fine for combat-turn granularity.

**What's missing for full agent-driven combat:**
- [ ] Agent combat loop — backend receives `combatTurn` for NPC but doesn't yet
  decide and execute a full turn (search creature tactics → pick actions → issue
  commands). Currently delegates to ACA or waits for GM input.
- [ ] Action sequencing — agent needs to chain commands within a turn
  (e.g., Stride → Strike → Strike with MAP tracking)
- [ ] Tactical decision-making — use creature enrichment metadata (tactics, morale,
  behavior) to inform NPC action selection via LLM

### Mode 2: Theater of the Mind (NOT BUILT)

Agent-native encounter execution — no Foundry dependency, no grid.
For CLI/chat-only play or when Foundry isn't available.

**What TotM needs (vs Foundry):**
- **Doesn't need:** Grid positions, token movement, range calculations, walls/lighting
- **Does need:** Narrative positioning ("the archers are on the balcony"), who's engaged with whom, conditions/HP, initiative order

**Existing infrastructure:**
- `EncounterServer` — initiative, participants, HP/conditions (5 tools, stateless)
- `SubsystemServer` — VP tracking, chase/infiltration state (5 tools, campaign-scoped)
- `CreatureModifierServer` — elite/weak, templates, scaffolding (7 tools, stateless)
- Creature stats via RAG search (remaster boost handles dedup)
- AP encounter areas already extracted (locations, hazards, treasure, read-aloud)

**Missing pieces:**
- [ ] **Narrative positioning layer** — lightweight "zones" instead of grid squares
  - Zone examples: "melee range", "balcony", "behind cover", "flanking"
  - Track which participants are in which zone
  - No distance math — GM adjudicates movement between zones
- [ ] **Turn orchestration** — agent drives initiative loop
  - Prompt NPC actions based on tactics/personality from creature metadata
  - Present player turns with relevant context (conditions, nearby threats)
  - Track reactions and free actions within the round
- [ ] **Encounter lifecycle** — start/end hooks
  - `start_encounter` pulls creature stats + AP encounter area context
  - Auto-surface tactics from creature enrichment metadata
  - `end_encounter` logs XP, treasure, updates AP progress
- [ ] **Condition automation** — persistent damage, frightened reduction, dying/recovery
  - Tick conditions at start/end of turn per PF2e rules
  - Agent shouldn't have to remember to reduce Frightened manually
- [ ] **Action economy tracking** — 3 actions + reaction per turn
  - Track MAP (multiple attack penalty) state within a turn
  - Flag illegal action sequences (e.g., 4 actions)

**Design decisions (deferred):**
- How much rules automation vs GM adjudication? (PF2e has complex action interactions)
- Should the agent roll dice or just suggest DCs and let the GM roll?
- How to handle AoE/splash in zones without precise positioning?
- Integrate with dice server or keep combat dice separate?

**Priority:** Low (only needed if Foundry puppeting doesn't cover the use case)

## Analytics & Tuning

### Response Time Analytics
- Dashboard for latency percentiles
- Alert on degraded performance
- Correlation with tool usage

### Tool Usage Analytics
**Current:** TurnMetadata includes `tool_count` and `tool_usage` dict.

**Future:**
- Per-tool success/failure rates
- Identify unused or underutilized tools
