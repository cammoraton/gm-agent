"""Two-phase split pipeline: orchestrate with small model, narrate with big model.

The small model drives everything — RAG lookups, dice rolls, encounter evaluation —
then calls ``synthesize`` to hand off to the big model for narrative generation.
The narrator is toolless and produces a grounded response from reference material.

Even with the same model for both phases, this improves quality because:
1. Reduced cognitive load per call (full tools vs no tools)
2. Forced grounding -- narrator sees retrieved content first
3. Different temperature/sampling for each phase
4. Different system prompts optimized for each role
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import json

from .config import (
    RETRIEVAL_TEMPERATURE,
    SYNTHESIS_TEMPERATURE,
    ORCHESTRATOR_REASONING_EFFORT,
    NARRATOR_REASONING_EFFORT,
)
from .mcp.base import ToolDef, ToolResult
from .mcp.client import MCPClient
from .models.base import LLMBackend, Message, ToolCall


def _stable_args_key(args: dict) -> str:
    """Produce a stable string key for a tool-call args dict."""
    return json.dumps(args, sort_keys=True, default=str)


# Regex that matches Ollama special tokens like <|start|>, <|channel|>, <|call|>, etc.
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]*\|>')

# ---------------------------------------------------------------------------
# Query-classification regexes — used by both _detect_categories() and the
# auto-supplement helpers inside SplitPipeline.
# ---------------------------------------------------------------------------

_ENCOUNTER_RE = re.compile(r"\bencounter\b", re.IGNORECASE)
_LEVEL_RE = re.compile(r"\blevel[s]?\s+(\d+)\b", re.IGNORECASE)
_PARTY_SIZE_RE = re.compile(r"\b(\d+)\s+(?:level|player|character|pc)\b", re.IGNORECASE)
_NPC_LIST_RE = re.compile(
    r"\b(?:relationship\s+map|npc\s+web|who(?:'s|\s+are)\s+(?:the\s+)?(?:key\s+)?npcs?|"
    r"all\s+(?:the\s+)?npcs?|community\s+(?:map|dynamics)|list\s+(?:all\s+)?npcs?|"
    r"faction(?:s)?\s+(?:in|inside|within|compete|competing)|"
    r"living\s+ecosystem|group(?:s)?\s+(?:compete|competing|inside|within)|"
    r"who\s+(?:controls?|runs?|occupies?|holds?)\s+(?:the\s+)?(?:dungeon|keep|tower|area))\b",
    re.IGNORECASE,
)
_SESSION_RECAP_RE = re.compile(
    r"\b(?:last\s+session|previous\s+session|what\s+(?:happened|did\s+we\s+do)|"
    r"session\s+summary|session\s+recap|what\s+(?:has\s+)?happened|before|previously|recap|"
    r"what\s+(?:have|has)\s+(?:we|our|the\s+party)\s+"
    r"(?:encountered|explored|discovered|found|done|seen|accomplished|covered|cleared)|"
    r"what\s+quests?\s+(?:are\s+we|do\s+we\s+have|are\s+we\s+tracking)|"
    r"how\s+(?:does|do|did)\s+\w[\w\s]{0,30}(?:feel|think|view|regard|treat|react)\s+"
    r"(?:about|toward|towards|to)\s+us|"
    r"(?:what|how)\s+is\s+\w[\w\s]{0,20}(?:'s|s')\s+(?:attitude|relationship|opinion|"
    r"disposition|standing|feelings?)\s+(?:toward|towards|to|about|with|for)\s+"
    r"(?:us|the\s+party|our\s+party)|"
    r"how\s+(?:have|has)\s+(?:we|our|the\s+party)\s+(?:left\s+things?|left\s+it|"
    r"left\s+off|stood?)\s+with)\b",
    re.IGNORECASE,
)
_PARTY_KNOWLEDGE_RE = re.compile(
    r"\bwhat\s+do\s+(?:we|the\s+party)\s+know\b|"
    r"\bwhat\s+has\s+(?:the\s+)?party\s+learned\b|"
    r"\bwhat\s+(?:do\s+)?(?:we|the\s+party)\s+(?:currently\s+)?know\b",
    re.IGNORECASE,
)
# Campaign-state tool names — results from these contain session narrative,
# not raw DB entity data, so they don't count toward the session-recap guard.
_SESSION_TOOLS = frozenset({
    "get_session_recap", "get_session_summary", "search_history",
    "query_npc_knowledge", "query_party_knowledge", "what_will_npc_share",
})

# Regex that matches raw XML function-call blocks output by some models when they
# try to call tools as text instead of via structured API.  Catches both the
# standard <function_calls>…</function_calls> form and the DSML variant used by
# some OpenRouter-served models (<｜DSML｜function_calls>…).  Everything from the
# opening tag to end-of-text is stripped so the caller can treat the response
# as empty and fall through to the retry / hard-fallback path.
_FUNCTION_CALL_XML_RE = re.compile(r'<.{0,12}function_calls[\s\S]*', re.DOTALL)


def _strip_special_tokens(text: str) -> str:
    """Remove Ollama tokenizer artifacts and raw XML tool-call output."""
    text = _SPECIAL_TOKEN_RE.sub('', text)
    text = _FUNCTION_CALL_XML_RE.sub('', text)
    return text.strip()


# Minimum tool calls before the orchestrator is allowed to synthesize.
# If it tries to synthesize with fewer results, the call is rejected and
# it's told to keep searching (unless results are clearly sufficient).
# 2 is the floor: even simple questions benefit from at least a lookup + rules check.
_MIN_TOOLS_BEFORE_SYNTHESIZE = 2

# Maximum reflection rounds between tool-gathering passes and synthesis handoff.
# Set to 0 to disable reflection entirely (model synthesizes after first pass).
_MAX_REFLECTION_ROUNDS = 0

# ---------------------------------------------------------------------------
# Meta-tools
# ---------------------------------------------------------------------------

SYNTHESIZE_TOOL = ToolDef(
    name="synthesize",
    description=(
        "Hand off to the narrator to produce the final response. "
        "Call this AFTER you have gathered all necessary information "
        "and executed all required actions (dice rolls, encounter evaluation, etc.)."
    ),
    parameters=[],
)


# ---------------------------------------------------------------------------
# Orchestrator system prompt (Phase 1)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Orchestrator prompt — composable sections
#
# Structure per query:
#   _PROMPT_CORE          — always: role, workflow, tool names
#   _CHAT_MODE_HEADER     — chat mode: "pure RAG oracle, no session memory"
#   _CAMPAIGN_MODE_HEADER — campaign mode: memory systems + how they work
#   _PROMPT_COMMON        — always: key distinctions, AP tips, search strategy
#   _SECTION_*            — injected when category detected (see _detect_categories)
# ---------------------------------------------------------------------------

_PROMPT_CORE = """\
You are a Pathfinder 2e tool orchestrator.

Your job: call tools to gather information, then call `synthesize` ONCE to hand off \
to the narrator. Do NOT answer the question yourself.

Workflow:
1. Call information-gathering tools (2 minimum; 3-4 for deep lore or AP narrative)
2. Call action tools (dice, encounters) if needed
3. Call `synthesize` when you have sufficient information

LOOK UP everything — your parametric knowledge of PF2e is unreliable. \
The tools have actual rules text. Synthesize as soon as results cover the question. \
If 2 calls fully answer it, synthesize immediately.

CRITICAL — avoid thinking loops: If you have made ≥2 tool calls and find yourself \
stuck deliberating rather than calling a tool, call synthesize immediately. \
Making multiple browse_book calls for different AP volumes is correct research, \
not a loop. The signal for a loop is: thinking without acting.

Tools:
- lookup(type, name, book?): by name — types: creature, spell, item, location, \
hazard, npc, encounter, class
- search_rules: game mechanics, conditions, actions (rulebooks only)
- search_content: general search; scope="lore" for world lore/regions/deities/history; \
scope="guidance" for GM advice; types= to filter by entity type
- search_pages: raw page text, narrative passages, flavor text
- list_entities: bulk inventories — "all NPCs in X", "all creatures", "all deities"
- browse_book: book/chapter structure, summaries, and open threads
- evaluate_encounter(party_level, party_size, creature_levels): encounter difficulty
- scaffold_creature/scaffold_hazard/apply_elite_weak: encounter building blocks"""


_CHAT_MODE_HEADER = """
Mode: Pure information oracle — no session memory exists.
You have access to the PF2e content database (rules, creatures, NPCs, spells, \
items, AP text, world lore) and encounter/dice tools. There is no campaign history, \
NPC memory, or session state — every answer comes from the database or encounter math."""


_CAMPAIGN_MODE_HEADER = """
Mode: Campaign manager for an ongoing Pathfinder 2e game.

You have persistent memory systems built up across sessions. \
Use these BEFORE searching the RAG database for questions about this party's \
specific experience:

NPC Knowledge → query_npc_knowledge(character_name)
  Facts about named NPCs: personality, relationships, what they know and will share.
  Seeded from AP books during campaign prep. Updated by the crunch pipeline after \
each session as NPCs learn new things and relationships change.
  Empty result = NPC not yet seeded or not yet encountered. \
Fall back to lookup(type="npc").

Party Knowledge → query_party_knowledge(query)
  What this specific party has learned — seeded from Players Guides, updated as they \
discover things in play. For "what do we know about X?" questions.

Session Recap → get_session_recap()
  Rolling narrative summary of COMPLETED sessions: what happened, who was met, \
what was found, what was cleared, what mysteries were uncovered. \
  For "last session" / "previously" / "what have we done?" questions.
  NOT for current-session events — those are already in the Conversation History \
above. Read the Conversation History directly; do not call a tool for it.

Subsystem State → get_subsystem_state(type)
  Persistent state machines for tracked activities: hexploration hex progress, \
influence tracks, kingdom building turns, chase positions, infiltration points.
  Only populated when start_subsystem was called during play. \
Empty = not yet initialized. Fall back to Conversation History if the answer is there.
  Do NOT call start_subsystem in response to a retrospective query.

Use RAG (lookup/search_*) for rules, world lore, and creatures not yet in memory."""


_PROMPT_COMMON = """
Key tool distinctions:
- Story NPCs → lookup(type="npc"). Bestiary creatures → lookup(type="creature"). \
NEVER use search_content(types="npc") for stat blocks.
- Major world figures (Tar-Baphon, Iomedae, Aroden, etc.) → BOTH \
lookup(type="npc") AND search_content(scope="lore"). The former has AP biography \
and appearance metadata; the latter has deity/faction data. Using only one is incomplete.
- "Which APs feature X?" → lookup(type="npc") — NPC entries contain AP appearance \
metadata. search_content will not return a complete cross-AP list.
- Creatures are NOT in "Player Core" — use Monster Core or Bestiary.
- Dragon taxonomy / creature families → list_entities(type="creature_family") or \
search_content(types="creature_family"). NOT search_content(scope="lore").
- Enumerating a category ("all dragons", "what deities exist", "list all X") → \
ALWAYS list_entities, NOT search_content (search returns incomplete results).
- City districts / named quarters → search_content(types="landmark"), no book filter.
- Cities and towns → search_content(types="settlement") AND search_pages — entity \
summaries are brief; actual detail is in page text.
- Deities → search_content(scope="lore") or list_entities(type="deity").
- Classes → lookup(type="class") for the class entry; search_rules for specific \
abilities. No book filter — class data spans Player Core and Player Core 2.
- Faction / NPC relationship dynamics → search_content(types="relationship", book=X) \
AND search_content(types="faction", book=X). Both types, always.
- Boxed text / read-aloud for AP rooms → lookup(type="location", name=X, book=X). \
NOT search_pages (that returns raw text, not the formatted read-aloud field).
- A deity result when you searched for a person = wrong entity. Try name variants.
- A MISMATCH (result directly contradicts what you searched for) → retry lookup with \
an alternate name form (nickname, surname only). Do NOT fall back to search_content.

NPC lookups:
- Roleplay / voice / reaction → lookup(type="npc", name=X, book=AP). Not search_content.
- AP-specific NPCs → ALWAYS add book= to disambiguate from same-name entities elsewhere.
- Name variants: remove title prefixes ("Mayor X" → "X"); try nickname forms \
("Grandmother X" → "Granny X"); try surname alone.
- NEVER truncate names with "…" — use the full name or it will fail silently.

AP questions:
- Keep queries SHORT (1-3 terms). Do NOT embed AP names in query text.
- Scope to an AP only when the question is clearly about that specific AP.
- browse_book(book="AP Name") to see structure and volumes.
- Multi-volume APs: use abbreviated references — "Season of Ghosts 1", "Kingmaker". \
The tool resolves to full titles. NEVER use "…" in book names.
- AP chapters have REAL names, not "Chapter 1". If browse_book(chapter="Chapter N") \
fails, it lists available names — use one on the next call.

Rules questions:
- ALWAYS use search_rules. Do not rely on training knowledge.
- Conditions in a tactical scenario → look them up with search_rules BEFORE advising.
- Keep queries SHORT (1-3 key terms). No book names in the query string.
- AP subsystem rules (kingdom building, influence, hexploration) are NOT in \
search_rules — use browse_book to find the subsystem chapter, then browse it.
- Core rules split across Player Core + Player Core 2 — no book filter for base rules.
- Truncated result (no DCs, no outcome table) → follow up with search_pages for full text.
- If search_rules returns nothing: retry with shorter terms, then try search_pages.

Remaster:
- Dragon renames → search_rules("remaster dragon renames"). Not from memory.
- Other term changes (flat-footed, spell level, AoO) → search_rules("remaster overview").

Search strategy:
- Thin results → try different query, different tool, remove type filter, or search_pages.
- Do NOT call the same tool+args twice (duplicates are already skipped, but avoid them).
- "Error:" in result → do NOT retry the same call. Move on or try a different tool.
- "(No results in 'X' — showing results from all books)" → scoped query found nothing. \
Change query or remove the type filter.
- Narrative content (plot, relationships, events) → search_pages. No type filters.
- Comparisons → look up BOTH sides separately. Rules adjudication → look up EACH ability.
- INVALID search_content types (return zero results): "event", "adventure", "book", \
"article", "chapter", "other". For narrative events: omit types filter.

NEVER produce a text response without calling tools. ALWAYS search first."""


# ---------------------------------------------------------------------------
# Composable category sections — injected based on _detect_categories()
# ---------------------------------------------------------------------------

_SECTION_MULTI_VOLUME = """
AP narrative questions (story arc, book-to-book progression, unresolved threads, \
tone, NPC development across books):
- "Unresolved threads from Book N" / "what carries forward?" → \
browse_book(book="EXACT Individual Book Title") returns an "Open/Unresolved Threads" \
section — that IS the answer; synthesize it directly. Use the exact individual title, \
NOT the series name (e.g. "Season of Ghosts 1 of 4 - The Summer that Never Was", not \
"Season of Ghosts"). CRITICAL: after listing Book N's threads, do NOT add a column or \
section about "how Book N+1 treats them" unless you have retrieved Book N+1 data — \
that is fabrication. List Book N threads only; note you haven't searched the next book.
- Arc/overview / chapter summary → browse_book("EXACT Title") gives chapter TOC + \
summaries. Then search_pages for specific plot mechanics or events not in summaries. \
NEVER synthesize from browse_book summaries alone — they describe themes, not specifics.
- AP-spanning mystery ("how does the mystery unfold?", "central mystery", "what does \
the party learn across all books?") → browse_book for EACH volume (exact titles). \
A 4-book AP needs 4 calls. Then 1-2 search_pages targeting specific named threads.
- NPC arc across books ("how does X change?", "X in Book 1 vs Book 2") → \
lookup(type="npc", name=X, book="Book 1") THEN lookup(type="npc", name=X, book="Book 2"). \
Character info is stored per-volume — a single lookup misses the arc."""

_SECTION_NPC_LIST = """
NPC inventory and relationship map questions ("who are the NPCs", "key NPCs in X", \
"relationship map", "important characters", "who does the party meet"):
- ALWAYS call list_entities(type="npc", book="Exact Book Title") FIRST. The book= \
parameter must exactly match the title — use browse_book first to confirm it if unsure. \
Kingmaker is one book ("Kingmaker"); Season of Ghosts has separate volumes.
- list_entities gives names only. For relationship maps or personalities, follow up \
with lookup(type="npc") for the 5-8 most plot-relevant characters individually.
- Then search_pages(query="[NPC] relationship allies enemies", book=X) — entity \
entries rarely state explicit relationships; page text often does.
- Report only NPCs and relationships actually found in tool results."""

_SECTION_ENCOUNTER = """
Encounter and combat questions:
- Tactical running ("how should I run X?", "round N of combat", "fighting Y tactically", \
"boss + minions"): FIRST look up stat blocks via lookup(type="creature", name=X) or \
lookup(type="npc", name=X). Tactical advice MUST be grounded in actual stat block \
abilities — do NOT advise from training memory. Also search_rules for any conditions \
involved (flanking, wounded, frightened, etc.).
- Difficulty/balance ("is this fair?", "good challenge for level Y?"): ALWAYS call \
evaluate_encounter(party_level, party_size, creature_levels). creature_levels is a list.
- Elite/Weak template: ALWAYS call apply_elite_weak(creature_name, adjustment) BEFORE \
looking up stats manually.
- scaffold_creature(level, type, role?, size?, name?): type="creature" (default), \
"troop" (mob/unit mechanics), or "swarm". Use for goblin mobs, cultist groups, swarms.
- scaffold_hazard(level, type?): type="trap" (default) or "haunt".
- Always look up creature levels via lookup(type="creature") before evaluate_encounter."""

_SECTION_CONTENT_SENS = """
Content sensitivity/safety questions ("body horror in X", "sensitive themes", \
"content warnings", "can I run this for players sensitive to Y"):
- MANDATORY: First call MUST be browse_book for each relevant AP volume. \
Do NOT start with keyword searches — chapter summaries describe what actually happens.
- FORBIDDEN: Do NOT search for "body horror", "gore", "blood", "violence" — the \
database does not tag by sensitivity. These queries return nothing useful.
- After browse_book, add search_content(scope="guidance", query="content sensitivity") \
for general GM advice on handling difficult themes."""

_SECTION_SESSION_STATE = """
Session history and party state questions ("what happened last session?", \
"what quests are we tracking?", "what have we accomplished?", "previously"):
- get_session_recap() → rolling narrative of completed sessions. For "last session" \
or "before" questions.
- search_history(query=X) → search across multiple older sessions. Use only when the \
question spans many past sessions not visible in the current Conversation History.
- Current-session events are in the Conversation History — read it directly. \
Do NOT call a tool for things that happened in this session."""

_SECTION_SUBSYSTEM = """
Subsystem state questions ("what hexes have we explored?", "influence progress", \
"where are we in the chase?", "kingdom stats", "infiltration points"):
- get_subsystem_state(type=X) where type is one of: "hexploration", "influence", \
"chase", "kingdom", "infiltration", "research", or the specific subsystem ID.
- Empty result = subsystem not yet initialized. Check Conversation History for \
manually tracked state. Do NOT call start_subsystem in response to a retrospective query."""


def _detect_categories(query: str) -> frozenset[str]:
    """Detect which prompt sections to inject based on query content.

    Pure regex — no LLM call. Returns a frozenset of category tags.
    Unknown/uncategorised queries get only CORE + MODE_HEADER + COMMON,
    which already covers the vast majority of questions.
    """
    q = query.lower()
    cats: set[str] = set()

    if re.search(
        r"how does.{0,40}unfold|story arc|book.to.book|across.{0,10}book|"
        r"across.{0,10}volume|unresolved thread|open thread|what carries forward|"
        r"central mystery|how does.{0,30}mystery|ap.spanning|"
        r"book \d.{0,20}book \d|\ball \d+ book",
        q,
    ):
        cats.add("multi_volume")

    if _NPC_LIST_RE.search(q):
        cats.add("npc_list")

    if re.search(
        r"\bencounter\b|\bcombat\b|\bfighting\b|\btactically\b|"
        r"\bround \d\b|\bstat.?block\b|\brun this\b",
        q,
    ) and re.search(
        r"level \d|challeng|difficult|fair fight|balanc|"
        r"build.{0,15}encounter|design.{0,15}encounter|create.{0,15}encounter|"
        r"scaffold|elite|weak template|"
        r"how.{0,20}run|tactically|bodyguard|flanking|wounded|initiative|"
        r"boss.{0,10}fight|minion|horde|swarm",
        q,
    ):
        cats.add("encounter")

    if re.search(
        r"body horror|content warning|sensitive to|warn.{0,10}player|"
        r"disturbing|uncomfortable|mature theme|how do i handle.{0,20}theme|"
        r"safe to run|trigger warning",
        q,
    ):
        cats.add("content_sens")

    if _SESSION_RECAP_RE.search(q):
        cats.add("session_state")

    if re.search(
        r"hexplor|hex.{0,10}explor|explor.{0,10}hex|kingdom.{0,10}stat|"
        r"influence.{0,10}track|chase.{0,10}position|infiltration.{0,10}point|"
        r"subsystem.{0,10}state|what.{0,20}explored|how far.{0,10}along|"
        r"tracking.{0,10}progress",
        q,
    ):
        cats.add("subsystem")

    return frozenset(cats)


# Backward-compat alias: full prompt with all sections (campaign mode)
ORCHESTRATOR_PROMPT = (
    _PROMPT_CORE
    + _CAMPAIGN_MODE_HEADER
    + _PROMPT_COMMON
    + _SECTION_MULTI_VOLUME
    + _SECTION_NPC_LIST
    + _SECTION_ENCOUNTER
    + _SECTION_CONTENT_SENS
    + _SECTION_SESSION_STATE
    + _SECTION_SUBSYSTEM
)

# Kept for any external code that referenced the old names
_ORCHESTRATOR_PROMPT_BASE = _PROMPT_CORE + _CHAT_MODE_HEADER + _PROMPT_COMMON
_CAMPAIGN_TOOLS_SECTION = _SECTION_SESSION_STATE + "\n" + _SECTION_SUBSYSTEM


# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------


def _make_reflection_prompt(n_results: int) -> str:
    """Prompt injected between research passes to encourage self-assessment.

    The orchestrator reads this after calling synthesize. It must make at
    least one NEW tool call (not a repeat of something already searched)
    before synthesize is accepted again — so only call synthesize if you
    have genuinely new information to add first.
    """
    return (
        f"You have {n_results} result(s). Quick self-check before handing off:\n\n"
        "Is there a specific thing — an NPC, rule, location, or creature — "
        "that you haven't looked up yet and that would meaningfully improve the answer? "
        "If yes, look it up now (1–2 calls), then synthesize. "
        "If your coverage is solid, synthesize now.\n\n"
        "Note: duplicate tool calls (same tool + same args as before) are skipped automatically — "
        "only search for things you haven't already retrieved."
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Structured output from a retrieval cycle."""

    tool_calls: list[tuple[str, dict, str]]  # (name, args, result_text)
    duration_ms: float
    model: str
    thinking: list[str] = field(default_factory=list)  # per-iteration reasoning traces


@dataclass
class SplitResult:
    """Full output from the two-phase pipeline."""

    response: str
    retrieval_rounds: list[RetrievalResult]
    synthesis_model: str
    total_duration_ms: float
    reflection_rounds_used: int = 0
    synthesis_thinking: str | None = None  # narrator reasoning trace


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SplitPipeline:
    """Two-phase agent: orchestrate with small model, narrate with big model.

    The small model handles all tool calls (RAG, dice, encounters) then
    calls ``synthesize`` to hand off.  The big model receives all tool results
    as reference material and produces a grounded narrative response.
    """

    def __init__(
        self,
        retrieval_llm: LLMBackend,
        synthesis_llm: LLMBackend,
        mcp: MCPClient,
        synthesis_system_prompt: str,
        verbose: bool = False,
        campaign_mode: bool = False,
    ):
        self.retrieval_llm = retrieval_llm
        self.synthesis_llm = synthesis_llm
        self.mcp = mcp
        self.synthesis_system_prompt = synthesis_system_prompt
        self.verbose = verbose
        self._campaign_mode = campaign_mode

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_orchestrator_prompt(self, query: str) -> str:
        """Assemble a focused orchestrator prompt for this specific query.

        Starts with CORE + mode header + COMMON (always present), then
        appends only the category sections relevant to the detected query
        type.  This keeps the prompt short for simple queries and adds
        depth only when the query needs it.
        """
        categories = _detect_categories(query)

        parts = [
            _PROMPT_CORE,
            _CAMPAIGN_MODE_HEADER if self._campaign_mode else _CHAT_MODE_HEADER,
            _PROMPT_COMMON,
        ]

        if "multi_volume" in categories:
            parts.append(_SECTION_MULTI_VOLUME)
        if "npc_list" in categories:
            parts.append(_SECTION_NPC_LIST)
        if "encounter" in categories:
            parts.append(_SECTION_ENCOUNTER)
        if "content_sens" in categories:
            parts.append(_SECTION_CONTENT_SENS)
        # Campaign-only sections
        if self._campaign_mode and "session_state" in categories:
            parts.append(_SECTION_SESSION_STATE)
        if self._campaign_mode and "subsystem" in categories:
            parts.append(_SECTION_SUBSYSTEM)

        if self.verbose and categories:
            print(f"  [Prompt] categories={set(categories)}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        extra_context: str = "",
        max_orchestration_iterations: int = 5,
    ) -> SplitResult:
        """Run the full two-phase pipeline.

        Args:
            query: The user's question.
            conversation_history: Recent conversation turns for context.
            extra_context: Extra text prepended to the user message in
                narration (e.g. campaign context).
            max_orchestration_iterations: Max tool-call loops in orchestration.

        Returns:
            A :class:`SplitResult` with the final response and metadata.
        """
        t0 = time.time()

        # Build recent turns for pronoun resolution
        recent_turns: list[Message] = []
        if conversation_history:
            recent_turns = [
                m for m in conversation_history[-6:]
                if m.role in ("user", "assistant")
            ]

        # Phase 1: Orchestration (all tools + synthesize + reflection)
        orchestration, reflection_rounds_used = self._orchestrate(
            query, recent_turns, max_orchestration_iterations,
        )
        all_retrievals = [orchestration]

        if self.verbose:
            n = len(orchestration.tool_calls)
            reflect_str = (
                f", {reflection_rounds_used} reflection round(s)"
                if reflection_rounds_used else ""
            )
            print(f"\n[Orchestrator] {n} tool call(s) in {orchestration.duration_ms:.0f}ms{reflect_str}")

        # Phase 2: Narration (toolless)
        response, synthesis_thinking = self._narrate(
            query,
            all_retrievals,
            conversation_history,
            extra_context,
        )

        total_ms = (time.time() - t0) * 1000
        return SplitResult(
            response=response,
            retrieval_rounds=all_retrievals,
            synthesis_model=self.synthesis_llm.get_model_name(),
            total_duration_ms=total_ms,
            reflection_rounds_used=reflection_rounds_used,
            synthesis_thinking=synthesis_thinking,
        )

    # ------------------------------------------------------------------
    # Phase 1: Orchestration
    # ------------------------------------------------------------------

    def _orchestrate(
        self,
        query: str,
        recent_turns: list[Message],
        max_iterations: int,
    ) -> tuple[RetrievalResult, int]:
        """Phase 1: Tool orchestration at low temperature.

        The orchestrator sees ALL MCP tools plus ``synthesize``.  When it calls
        ``synthesize``, it enters a reflection loop (up to _MAX_REFLECTION_ROUNDS)
        where it can assess its own findings and optionally make more tool calls
        before the final handoff to narration.

        Returns:
            (RetrievalResult, reflection_rounds_used)
        """
        t0 = time.time()
        tools = self.mcp.list_tools() + [SYNTHESIZE_TOOL]

        messages: list[Message] = [
            Message(role="system", content=self._build_orchestrator_prompt(query)),
        ]
        messages.extend(recent_turns)
        messages.append(Message(role="user", content=query))

        results: list[tuple[str, dict, str]] = []
        seen_calls: set[str] = set()
        thinking_traces: list[str] = []
        reflection_rounds_used = 0

        # Outer loop: each iteration is one research pass (possibly after a reflection).
        # Pass 0 is the initial research; passes 1..N are post-reflection continuations.
        # In a reflection round, the orchestrator must make ≥1 new tool call before
        # synthesize is accepted again — immediate re-synthesize is rejected.
        for reflection_round in range(_MAX_REFLECTION_ROUNDS + 1):
            synthesize_called = False
            reflection_new_calls = 0  # Tool calls made in this specific reflection pass

            for iteration in range(max_iterations):
                try:
                    response = self.retrieval_llm.chat(
                        messages,
                        tools=tools,
                        temperature=RETRIEVAL_TEMPERATURE,
                        thinking={"effort": ORCHESTRATOR_REASONING_EFFORT} if ORCHESTRATOR_REASONING_EFFORT else None,
                    )
                except Exception as exc:
                    # Handle malformed tool calls (e.g. None id/name from small models)
                    if self.verbose:
                        print(f"  [Error] Orchestrator LLM error: {exc}")
                    response = None

                runaway_thinking = False
                if response is not None and response.thinking:
                    thinking_traces.append(response.thinking)
                    # Detect runaway reasoning loop: a single thinking trace is very
                    # long AND we have enough results to synthesize. We don't nudge
                    # (the model ignores nudges while mid-loop); instead we set a flag
                    # and force a synthesize handoff after this iteration's tool calls
                    # finish executing.
                    if (
                        len(response.thinking) > 2000
                        and len(results) >= _MIN_TOOLS_BEFORE_SYNTHESIZE
                        and not synthesize_called
                    ):
                        runaway_thinking = True
                        if self.verbose:
                            print(
                                f"  [Loop] Thinking {len(response.thinking)} chars "
                                f"with {len(results)} results — will force handoff"
                            )

                if response is None or not response.tool_calls:
                    if reflection_round == 0 and iteration == 0 and not results:
                        # First-ever iteration with zero tool calls — force a fallback
                        # search so the narrator has SOMETHING to work with
                        if self.verbose:
                            print("  [Fallback] Orchestrator returned no tools — auto-searching")
                        fallback_result = self.mcp.call_tool(
                            "search_content", {"query": query, "limit": 5},
                        )
                        fallback_text = fallback_result.to_string()
                        results.append(("search_content", {"query": query, "limit": 5}, fallback_text))
                    elif len(results) < _MIN_TOOLS_BEFORE_SYNTHESIZE and iteration < max_iterations - 1:
                        # Too few results and model stopped calling tools —
                        # nudge it to keep searching instead of implicit handoff
                        if self.verbose:
                            print(f"  [Nudge] Only {len(results)} tool call(s) — prompting for more")
                        messages.append(Message(
                            role="user",
                            content=(
                                "You have made {n} tool call(s). If you already have "
                                "sufficient information to answer the question fully, "
                                "call synthesize. Otherwise, make 1-2 more targeted "
                                "searches with different queries or tool types."
                            ).format(n=len(results)),
                        ))
                        continue
                    # Implicit handoff — exit both loops
                    synthesize_called = False
                    break

                # Add assistant message with tool calls
                messages.append(Message(
                    role="assistant",
                    content=_strip_special_tokens(response.text),
                    tool_calls=response.tool_calls,
                ))

                for tc in response.tool_calls:
                    # Strip Ollama tokenizer artifacts like "synthesize<|channel|>commentary"
                    tool_name = tc.name.split("<|")[0].strip()
                    if tool_name == "synthesize":
                        # Reject premature synthesize — force more searching
                        if len(results) < _MIN_TOOLS_BEFORE_SYNTHESIZE and iteration < max_iterations - 1:
                            if self.verbose:
                                print(f"  [Reject] synthesize with only {len(results)} tool call(s) — need more")
                            messages.append(Message(
                                role="tool",
                                content=(
                                    "Only {n} tool call(s) so far. If you have "
                                    "enough to answer the question, call synthesize. "
                                    "If not, make 1-2 more targeted searches — look up "
                                    "specific entities, rules, or locations. Try a "
                                    "different tool type if initial results were thin."
                                ).format(n=len(results)),
                                tool_call_id=tc.id,
                            ))
                            continue

                        # In a reflection round, require at least 1 new tool call first.
                        # This prevents the model from immediately re-synthesizing
                        # without actually filling any gaps identified in the reflection.
                        if reflection_round > 0 and reflection_new_calls == 0:
                            if self.verbose:
                                print("  [Reject] reflection synthesize with no new tool calls — must search first")
                            messages.append(Message(
                                role="tool",
                                content=(
                                    "You must make at least one tool call to fill a gap "
                                    "before synthesizing in a reflection round. "
                                    "Look up a specific NPC, rule, location, or creature "
                                    "that you haven't searched for yet."
                                ),
                                tool_call_id=tc.id,
                            ))
                            continue

                        # Reject synthesize when ALL results so far are browse_book
                        # and no specific content was fetched — force a search_pages call
                        non_browse = [n for n, _a, _t in results if n != "browse_book"]
                        if len(results) >= 2 and not non_browse:
                            if self.verbose:
                                print(
                                    f"  [Reject] synthesize with only {len(results)} browse_book call(s) "
                                    "— no specific content fetched"
                                )
                            messages.append(Message(
                                role="tool",
                                content=(
                                    "You have only called browse_book ({n} time(s)) and no specific "
                                    "content tools. Before synthesizing, call search_pages(query='[key "
                                    "term from summaries]', book='...') or lookup(type='npc')/lookup(type='creature') "
                                    "to fetch actual plot/NPC/encounter content. "
                                    "browse_book gives chapter titles and themes only — "
                                    "it cannot answer questions about specific NPCs, plot events, "
                                    "encounter mechanics, or how a mystery resolves."
                                ).format(n=len(results)),
                                tool_call_id=tc.id,
                            ))
                            continue

                        synthesize_called = True
                        messages.append(Message(
                            role="tool",
                            content="Synthesize acknowledged.",
                            tool_call_id=tc.id,
                        ))
                        continue

                    # Dedup
                    call_key = f"{tool_name}|{_stable_args_key(tc.args)}"
                    if call_key in seen_calls:
                        prev_text = next(
                            (text for name, args, text in results
                             if f"{name}|{_stable_args_key(args)}" == call_key),
                            "(duplicate call skipped)",
                        )
                        if self.verbose:
                            print(f"  {tool_name}({tc.args}) -> [dedup skipped]")
                        messages.append(Message(
                            role="tool",
                            content=prev_text,
                            tool_call_id=tc.id,
                        ))
                        continue

                    seen_calls.add(call_key)
                    tool_result = self.mcp.call_tool(tool_name, tc.args)
                    result_text = tool_result.to_string()
                    results.append((tool_name, tc.args, result_text))
                    reflection_new_calls += 1  # Track calls made in this reflection pass

                    if self.verbose:
                        preview = result_text[:100]
                        print(f"  {tool_name}({tc.args}) -> {preview}...")

                    messages.append(Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=tc.id,
                    ))

                # After all tool calls in this iteration have run, check if
                # we flagged a runaway thinking trace. Force handoff now —
                # don't let the model generate another massive thinking block.
                if runaway_thinking:
                    if self.verbose:
                        print(f"  [Loop] Forcing synthesize handoff ({len(results)} results)")
                    synthesize_called = True

                if synthesize_called:
                    break
            # end inner loop

            # If synthesize wasn't called (implicit handoff) or this was the last
            # allowed reflection pass, we're done — hand off to narrator.
            passes_remaining = _MAX_REFLECTION_ROUNDS - reflection_round
            if not synthesize_called or passes_remaining == 0:
                break

            # Synthesize was called and we still have reflection passes left.
            # Give the orchestrator a chance to assess its own work.
            reflection_rounds_used += 1
            if self.verbose:
                print(
                    f"  [Reflect] Pass {reflection_round + 1} complete "
                    f"({len(results)} results so far, {passes_remaining} reflection pass(es) remaining)"
                )
            messages.append(Message(
                role="user",
                content=_make_reflection_prompt(len(results)),
            ))
        # end outer loop

        # Auto-supplement: encounter-building queries need level-appropriate creatures.
        # If the orchestrator searched for creatures but never filtered by level, inject
        # a list_entities call so the narrator has grounded options to work from.
        self._supplement_encounter_creatures(query, results)
        self._supplement_npc_list(query, results)
        self._supplement_session_recap(query, results)
        self._supplement_party_knowledge(query, results)

        duration_ms = (time.time() - t0) * 1000
        return RetrievalResult(
            tool_calls=results,
            duration_ms=duration_ms,
            model=self.retrieval_llm.get_model_name(),
            thinking=thinking_traces,
        ), reflection_rounds_used

    # ------------------------------------------------------------------
    # Phase 2: Narration
    # ------------------------------------------------------------------

    def _narrate(
        self,
        query: str,
        retrievals: list[RetrievalResult],
        conversation_history: list[Message] | None,
        extra_context: str,
    ) -> tuple[str, str | None]:
        """Phase 2: Toolless grounded response generation.

        Returns:
            (response_text, thinking_trace_or_None)
        """
        context_text = self._format_all_retrievals(retrievals)

        messages: list[Message] = [
            Message(role="system", content=self.synthesis_system_prompt),
        ]

        if conversation_history:
            messages.extend(conversation_history)

        user_content = query
        if context_text:
            user_content += f"\n\n{context_text}"
        if extra_context:
            user_content = f"{extra_context}\n\n{user_content}"
        messages.append(Message(role="user", content=user_content))

        _narrator_thinking_param = {"effort": NARRATOR_REASONING_EFFORT} if NARRATOR_REASONING_EFFORT else None
        response = self.synthesis_llm.chat(
            messages,
            tools=None,
            temperature=SYNTHESIS_TEMPERATURE,
            thinking=_narrator_thinking_param,
        )

        synthesis_thinking = response.thinking

        # Strip Ollama special tokens (<|start|>, <|channel|>, <|call|>, etc.)
        # that leak into output when the synthesis model confuses itself with the
        # orchestrator role and tries to emit tool-routing tokens.
        cleaned_text = _strip_special_tokens(response.text)

        # Guard against empty response
        if not cleaned_text:
            if self.verbose:
                print("[Narrator] Empty response — retrying")

            messages.append(Message(
                role="user",
                content=(
                    "Please answer my original question now based on the "
                    "Reference Material above. Do not request more information."
                ),
            ))
            response = self.synthesis_llm.chat(
                messages, tools=None, temperature=SYNTHESIS_TEMPERATURE,
                thinking=_narrator_thinking_param,
            )
            cleaned_text = _strip_special_tokens(response.text)
            if response.thinking:
                synthesis_thinking = response.thinking

        # Hard fallback
        if not cleaned_text:
            return (
                "I wasn't able to formulate a response from the available search results. "
                "The retrieved content may not directly address your question. "
                "Please try rephrasing or ask about a more specific aspect."
            ), synthesis_thinking

        return cleaned_text, synthesis_thinking

    # ------------------------------------------------------------------
    # Auto-supplement helpers
    # ------------------------------------------------------------------

    # Module-level regexes referenced as class attributes for backward compat
    _ENCOUNTER_RE = _ENCOUNTER_RE  # noqa: SIM111
    _LEVEL_RE = _LEVEL_RE
    _PARTY_SIZE_RE = _PARTY_SIZE_RE
    _NPC_LIST_RE = _NPC_LIST_RE
    _SESSION_RECAP_RE = _SESSION_RECAP_RE
    _PARTY_KNOWLEDGE_RE = _PARTY_KNOWLEDGE_RE
    _SESSION_TOOLS = _SESSION_TOOLS

    def _supplement_encounter_creatures(
        self,
        query: str,
        results: list[tuple[str, dict, str]],
    ) -> None:
        """Inject suggest_encounter + level-filtered creature list if the orchestrator missed them.

        Only fires when:
        - The query is an encounter-building request (contains "encounter")
        - The query mentions a party level
        - suggest_encounter was not already called
        """
        if not self._ENCOUNTER_RE.search(query):
            return
        m = self._LEVEL_RE.search(query)
        if not m:
            return
        level = int(m.group(1))

        # Check what the orchestrator already called
        called = {name for name, _args, _text in results}
        has_level_filter = any(
            "min_level" in args or "max_level" in args
            for _name, args, _text in results
        )

        # Determine book context from existing tool calls (used for both supplements)
        book = next(
            (args["book"] for _name, args, _text in results if args.get("book")),
            None,
        )

        # Inject suggest_encounter if not already called — gives XP budget + creature roster
        if "suggest_encounter" not in called:
            ps_m = self._PARTY_SIZE_RE.search(query)
            party_size = int(ps_m.group(1)) if ps_m else 4
            suggest_args: dict = {"party_level": level, "party_size": party_size, "target_threat": "moderate"}
            if book:
                suggest_args["book"] = book
            if self.verbose:
                print(f"  [Supplement] Injecting suggest_encounter(level={level}, size={party_size}, book={book!r})")
            tool_result = self.mcp.call_tool("suggest_encounter", suggest_args)
            results.append(("suggest_encounter", suggest_args, tool_result.to_string()))

        # Inject level-filtered creature list if not already present
        if not has_level_filter:
            list_args: dict = {
                "type": "creature",
                "min_level": level - 1,
                "max_level": level + 2,
                "limit": 20,
            }
            if book:
                list_args["book"] = book
            if self.verbose:
                print(f"  [Supplement] Injecting level {level-1}–{level+2} creature list")
            tool_result = self.mcp.call_tool("list_entities", list_args)
            results.append(("list_entities", list_args, tool_result.to_string()))

    def _supplement_npc_list(
        self,
        query: str,
        results: list[tuple[str, dict, str]],
    ) -> None:
        """Inject list_entities(type="npc") for relationship-map / NPC-inventory queries.

        The orchestrator tends to call search_content repeatedly for these queries
        instead of doing a bulk NPC list, which leaves large parts of the community
        invisible to the narrator.  Only fires when:
        - The query looks like a relationship map or NPC inventory request
        - No list_entities(type="npc") was already called
        - A book can be inferred from existing results or the query
        """
        if not self._NPC_LIST_RE.search(query):
            return

        # Skip if orchestrator already fetched an NPC list
        already_listed = any(
            name == "list_entities" and args.get("type") == "npc"
            for name, args, _text in results
        )
        if already_listed:
            return

        # Infer book from existing tool results or from the query itself
        book = next(
            (args["book"] for _name, args, _text in results if args.get("book")),
            None,
        )
        list_args: dict = {"type": "npc", "limit": 40}
        if book:
            list_args["book"] = book
        if self.verbose:
            print(f"  [Supplement] Injecting NPC list (book={book!r})")
        tool_result = self.mcp.call_tool("list_entities", list_args)
        results.append(("list_entities", list_args, tool_result.to_string()))

    def _supplement_session_recap(
        self,
        query: str,
        results: list[tuple[str, dict, str]],
    ) -> None:
        """Inject get_session_recap for queries asking about prior session events.

        The orchestrator tends to use RAG tools for "what happened last session"
        queries instead of calling get_session_recap, leaving session continuity
        invisible to the narrator.  Only fires when:
        - The query looks like a prior-session / recap request
        - No session-state tool was already called
        - get_session_recap is available (campaign mode only)
        """
        if not self._SESSION_RECAP_RE.search(query):
            return

        already_called = any(
            name in self._SESSION_TOOLS for name, _args, _text in results
        )
        if already_called:
            return

        # Only fire in campaign mode (tool must be available)
        available = {t.name for t in self.mcp.list_tools()}
        if "get_session_recap" not in available:
            return

        if self.verbose:
            print("  [Supplement] Injecting get_session_recap for prior-session query")
        tool_result = self.mcp.call_tool("get_session_recap", {})
        results.append(("get_session_recap", {}, tool_result.to_string()))

    def _supplement_party_knowledge(
        self,
        query: str,
        results: list[tuple[str, dict, str]],
    ) -> None:
        """Inject query_party_knowledge for "what do we know about X" queries.

        The orchestrator tends to use RAG tools for party-knowledge questions
        instead of querying the campaign knowledge store, surfacing world lore
        instead of what the party has actually learned.  Only fires when:
        - The query looks like a party knowledge request
        - No knowledge tool was already called
        - query_party_knowledge is available (campaign mode only)
        """
        if not self._PARTY_KNOWLEDGE_RE.search(query):
            return

        already_called = any(
            name in self._SESSION_TOOLS for name, _args, _text in results
        )
        if already_called:
            return

        available = {t.name for t in self.mcp.list_tools()}
        if "query_party_knowledge" not in available:
            return

        if self.verbose:
            print("  [Supplement] Injecting query_party_knowledge for party-knowledge query")
        tool_result = self.mcp.call_tool("query_party_knowledge", {"query": query})
        results.append(("query_party_knowledge", {"query": query}, tool_result.to_string()))

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_retrieval(retrieval: RetrievalResult) -> str:
        """Format a single retrieval cycle's results as context text."""
        if not retrieval.tool_calls:
            return "No results found."
        sections = []
        for name, args, text in retrieval.tool_calls:
            arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            sections.append(f"### {name}({arg_str})\n{text}")
        return "\n\n".join(sections)

    def _format_all_retrievals(self, retrievals: list[RetrievalResult]) -> str:
        """Format all retrieval rounds as reference material."""
        all_text = "\n\n".join(
            self._format_retrieval(r) for r in retrievals if r.tool_calls
        )
        if not all_text:
            return ""

        # Detect sparse results: same entity returned multiple times counts once.
        all_results = [text for r in retrievals for _, _, text in r.tool_calls]
        all_tool_names = [name for r in retrievals for name, _, _ in r.tool_calls]
        seen: set[str] = set()
        unique_results: list[str] = []
        for text in all_results:
            key = text[:300]
            if key not in seen:
                seen.add(key)
                unique_results.append(text)
        unique_chars = sum(len(t) for t in unique_results)

        header = "## Reference Material"
        if len(unique_results) <= 4 and unique_chars < 1500:
            header += (
                "\n\n> **LIMITED COVERAGE — DO NOT EXTRAPOLATE**: Only {n} distinct "
                "result(s) found ({c} chars of unique content). "
                "Your training memory likely has much more information about this topic "
                "than the {n} result(s) below — you MUST ignore that memory entirely. "
                "Write a SHORT response covering ONLY what the {n} result(s) explicitly "
                "state. Do NOT expand lore, add context, or infer additional details "
                "beyond what the results say. If you find yourself writing content not "
                "in those results, STOP and replace it with an honest gap statement."
            ).format(n=len(unique_results), c=unique_chars)
        elif all(n == "browse_book" for n in all_tool_names) and len(all_tool_names) >= 2:
            _combined_lower = all_text.lower()
            _has_threads = (
                "open/unresolved threads" in _combined_lower
                or "open threads (unresolved" in _combined_lower
            )
            _has_npc_list = "key entities:" in _combined_lower or "key npcs:" in _combined_lower
            if _has_threads or _has_npc_list:
                header += (
                    "\n\n> **BROWSE SUMMARIES WITH SPECIFIC SECTIONS**: Results include "
                    "chapter/book summaries AND specific sections (e.g. Open/Unresolved "
                    "Threads, key entity lists). Synthesize those specific sections directly "
                    "— they ARE the answer. "
                    "CRITICAL: Do NOT add information about subsequent books (e.g. 'how "
                    "Book N+1 treats/resolves this thread') unless you have retrieved "
                    "Book N+1 data above — that is fabrication. List what the retrieved "
                    "book contains; note that subsequent books were not searched."
                )
            else:
                header += (
                    "\n\n> **SUMMARIES ONLY — NO SPECIFIC CONTENT RETRIEVED**: All results "
                    "below are book/chapter structural summaries. No specific NPC records, "
                    "encounter descriptions, or page text was retrieved. "
                    "These summaries describe THEMES and STRUCTURE only — they do NOT contain "
                    "specific plot events, NPC names, encounter mechanics, how a mystery "
                    "resolves, or what the party actually does. "
                    "You MUST NOT infer, extrapolate, or invent those specifics from summary text. "
                    "State what the summaries tell you (themes, chapter titles, tone), "
                    "then clearly say you don't have the specific details requested."
                )
        elif (
            len(all_tool_names) >= 3
            and "search_pages" not in all_tool_names
            and any(n == "browse_book" for n in all_tool_names)
        ):
            header += (
                "\n\n> **STRUCTURED DATA ONLY** — no raw page text was retrieved. "
                "Entity entries and chapter summaries may lack narrative detail: "
                "NPC relationships, social dynamics, specific plot revelations, "
                "and cross-story connections are in page text, not entity records. "
                "If the question asks for something not explicitly stated above "
                "(e.g. how NPCs relate to each other, exact plot beats, what the "
                "party specifically discovers), state that gap clearly. "
                "Do NOT fill narrative gaps from general knowledge."
            )

        return f"{header}\n\n{all_text}"
