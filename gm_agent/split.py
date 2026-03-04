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
from dataclasses import dataclass

import json

from .config import RETRIEVAL_TEMPERATURE, SYNTHESIS_TEMPERATURE
from .mcp.base import ToolDef, ToolResult
from .mcp.client import MCPClient
from .models.base import LLMBackend, Message, ToolCall


def _stable_args_key(args: dict) -> str:
    """Produce a stable string key for a tool-call args dict."""
    return json.dumps(args, sort_keys=True, default=str)


# Regex that matches Ollama special tokens like <|start|>, <|channel|>, <|call|>, etc.
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]*\|>')


def _strip_special_tokens(text: str) -> str:
    """Remove Ollama tokenizer artifacts (<|...|>) from model output text."""
    return _SPECIAL_TOKEN_RE.sub('', text).strip()


# Minimum tool calls before the orchestrator is allowed to synthesize.
# If it tries to synthesize with fewer results, the call is rejected and
# it's told to keep searching.
_MIN_TOOLS_BEFORE_SYNTHESIZE = 3

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

ORCHESTRATOR_PROMPT = """\
You are a Pathfinder 2e tool orchestrator.

Your job: call tools to gather information, then call `synthesize` to hand off \
to the narrator. Do NOT answer the question yourself.

Workflow:
1. Call information-gathering tools (3 calls minimum; 4-5 for deep lore/AP narrative)
2. Call action tools (dice, encounters) if needed
3. Call `synthesize` when done

ALWAYS call at least 3 tools before synthesize. LOOK UP everything -- your \
parametric knowledge of PF2e is unreliable. The tools have actual rules text.

Be FOCUSED. Target 3 well-targeted results before synthesizing. If results \
are thin, try different queries or tools — avoid redundant calls.

Tool selection:
- lookup_creature/lookup_npc/lookup_spell/lookup_item/lookup_hazard: by name
- search_rules: game mechanics, conditions, actions (rulebooks only)
- search_content: general search with type/book filters
- search_lore: world lore, locations, regions, deities, history — does NOT return NPCs or creature ecology/taxonomy
- search_guidance: GM advice and best practices
- search_pages: raw page text, narrative passages
- list_entities: inventories ("all NPCs in X", "all creatures", "all deities")
- browse_book: book/chapter structure and summaries

Key distinctions:
- NPCs -> lookup_npc (NOT lookup_creature)
- Combat monsters/stat blocks → ALWAYS lookup_creature(name="X"). Do NOT use \
search_content(types="npc") for creature stat blocks — that returns story-character \
NPCs, not bestiary stat blocks. For a combat scenario involving a will-o'-wisp, \
boar, or any bestiary creature: lookup_creature("will-o'-wisp"), NOT \
search_content(types="npc", query="will-o'-wisp"). And do NOT scope creature \
lookups to a specific AP book (lookup_creature does its own book resolution).
- Major world figures (Tar-Baphon, Iomedae, Aroden, Razmir, etc.) are NPCs/deities — \
call BOTH lookup_npc(name="X") AND search_lore(query="X"). lookup_npc holds the narrative \
biography and AP appearance data; search_lore holds the deity/faction entry. Using only \
one may give an incomplete or misleading picture. Do NOT use lookup_creature — it retrieves \
stat blocks, missing biography and AP context.
- For "which APs feature X?" or "where does X appear across the game?" → lookup_npc \
is essential — NPC entries contain AP appearance metadata. search_content and search_lore \
will NOT return a complete list of AP appearances for a named figure.
- Creatures -> lookup_creature; NOT in "Player Core" -- use Monster Core, Bestiary, etc.
- Creature families and dragon taxonomy -> list_entities(type="creature_family", name_filter="X") or search_content(types="creature_family", query="X"). Do NOT use search_lore for creature ecology, dragon lore, or "where do X creatures come from" — search_lore returns geography/history, not creature entries.
- Enumerating a category ("list all X types", "what kinds of X exist?", "what dragons are in PF2e?") → \
ALWAYS use list_entities, NOT search_content. The search will return incomplete results. \
Examples: "list all dragons" → list_entities(type="creature_family", name_filter="dragon"), \
"what deities" → list_entities(type="deity"), "all hazard types" → list_entities(type="hazard")
- Named districts, quarters, and neighborhoods WITHIN a city (e.g. "Precipice Quarter", \
"The Coins", "Eastgate district") → search_content(types="landmark"). These are \
NOT settlements — they are sub-city features indexed as landmarks. \
Do NOT add a book= filter — a city landmark may be indexed in any sourcebook \
(e.g., the Precipice Quarter may appear in Knights of Lastwall, not just the \
Absalom book). Search without book scope, then filter by relevance.
- Cities, towns, villages ("what is Sandpoint like?", "tell me about Absalom") → \
search_content(query="X", types="settlement") AND search_pages(query="X"). \
Entity summaries are brief; the actual detail is in page text.
- For comparison questions ("X vs Y", "debating between X and Y") → search EACH \
location separately; do not stop if the first one returns no results. Try \
search_content(types="settlement"), search_lore, and search_pages for each topic.
- Deities -> search_lore or list_entities(type="deity")
- Classes -> search WITHOUT book filter (spread across Player Core + Player Core 2)
- NPC inventory — ANY of these phrasings triggers this rule: "who are the NPCs", \
"key NPCs in", "important characters in", "relationship map of NPCs", "list of \
characters", "NPC at the start of", "characters in Book X", "who does the party \
meet in" → ALWAYS call list_entities(type="npc", book="Exact Book Title") FIRST, \
THEN look up specific NPCs. NEVER use search_content(types="npc") for these — \
it returns the wrong book's NPCs. The book= parameter must exactly match the \
book title — use browse_book first to confirm it. NOTE: not all APs are split \
by volume. Kingmaker is a single book (book="Kingmaker"), NOT "Kingmaker Book 1". \
Season of Ghosts has separate volumes ("Season of Ghosts 1 of 4 - The Summer \
that Never Was", etc.). When unsure, call browse_book(book="AP Name") first.
- For RELATIONSHIP MAP or DETAILED NPC ANALYSIS questions: list_entities gives \
you names only. You MUST follow up with lookup_npc for the key characters — \
the narrator cannot describe relationships or personalities from a name alone. \
After list_entities returns 20+ names, pick the 5-8 most plot-relevant ones and \
look them up individually. Then use search_pages(query="[NPC name] relationship \
allies enemies faction", book="AP Title") to find narrative text describing social \
dynamics — entity entries rarely state explicit relationships, but page text often \
does. Your relationship map should contain ONLY NPCs and relationships you \
actually found in tool results — do not fill in others from memory.

AP questions:
- Use the `book` parameter to scope to the correct AP
- Keep queries SHORT: 1-3 terms. Do NOT put AP names in query text
- Only scope to an AP when the question is CLEARLY about that specific AP
- For general questions (lore, rules, "who is X"), do NOT scope to any AP
- browse_book(book="AP Name") to see AP structure and volumes
- Search for concrete nouns (names, places), not abstract concepts ("mystery")
- Multi-volume APs: use abbreviated references — browse_book("Season of Ghosts 1"), \
browse_book("Season of Ghosts 2"), browse_book("Kingmaker") etc. The tool resolves \
these to the correct full title. Do NOT guess the subtitle from memory.
- NEVER truncate book names with "…" or "..." — use the abbreviated form \
("Season of Ghosts 4", not "Season of Ghosts 4..."). Ellipsis in a book name \
will cause the lookup to fail silently.
- AP chapters have REAL names ("The Stolen Lands"), NOT "Chapter 1"/"Chapter 2". \
If browse_book(chapter="Chapter 2") fails it will list the available chapter names — \
use one of those names on the next call.

AP NARRATIVE questions (story arc, book-to-book progression, unresolved threads, \
tone, NPC development across books):
- For arc/overview questions ("summarize Book N", "story progression", "what carries \
forward into Book N+1?", "unresolved threads", "open plot threads"): \
browse_book(book="AP Book Title") — the output includes an "Open/Unresolved Threads" \
section with thread NAMES. For DETAILS about a specific thread (what the party \
discovers, how events unfold), follow up with \
search_pages(query="[thread name or key term]", book="AP Title"). \
Thread names are in browse_book; thread details live in raw page text. \
Do NOT use browse_book for specific-detail questions \
(NPC personalities, room descriptions) — use search_content/search_pages.
- For AP-spanning questions ("how does the mystery unfold?", "what does the party \
learn across the AP?", "what is the central mystery?", "how does X unfold?"):
  browse_book for EACH volume using abbreviated references \
  ("Season of Ghosts 1", "Season of Ghosts 2", etc.). A 4-book AP needs 4 calls.
  CRITICAL: After browsing all volumes, ALWAYS follow up with at least 1-2 \
  search_pages calls targeting specific named threads, events, or characters you \
  found in the summaries. browse_book gives themes and thread NAMES; search_pages \
  gives the actual plot mechanics and how events unfold. Never synthesize an AP \
  mystery answer from browse_book summaries alone — the summaries describe THEMES \
  (horror, ghosts, curses) not specific mechanics (who the villain is, how the \
  mystery resolves, what the party actually does).
- For NPC development/arc questions ("how does X change across books?", "X in Book 1 \
vs Book 2"): lookup_npc(name="X", book="AP: Book 1") THEN lookup_npc(name="X", \
book="AP: Book 2") etc. Character info is stored per-volume — a single lookup misses \
the arc. Try name variants if needed (e.g., "Granny X" for "Grandmother X").

Content sensitivity/safety questions ("how do I handle X theme?", "what body horror is in \
this AP?", "can I run this for players who are sensitive to Y?", "what content should I \
warn players about?"):
- MANDATORY: Your FIRST call MUST be browse_book for each relevant AP volume. \
Do NOT start with keyword searches. The chapter summaries describe what actually \
happens — they are the only reliable source for content assessment.
- FORBIDDEN: Do NOT use search_pages or search_content with keywords like "body \
horror", "gore", "blood", "disturbing", "violence" — the DB does not tag content \
by sensitivity. These searches return nothing useful and waste tool calls.
- After browse_book, supplement with search_guidance("content sensitivity") or \
search_guidance("safety tools") for general GM advice on handling difficult content.

Remaster questions ("what changed in the remaster?", "what's the remaster equivalent of X?", \
"list all Y types including remaster changes"):
- ALWAYS call search_rules("remaster dragon renames") for dragon taxonomy questions — \
creature_family searches alone won't show the rename table (Red→Cinder, Blue→Stormcrown, \
Green→Horned, Black→Bog, White→Rime, etc.). Do NOT organize dragon categories from memory.
- For other remaster term changes (flat-footed, spell level, attack of opportunity, etc.): \
call search_rules("remaster overview"). Your training knowledge of term changes is unreliable.

Rules questions ("how does X work?", "can I do Y?", "what is the rule for Z?"):
- ALWAYS use search_rules. Do NOT rely on your own PF2e knowledge — it may be \
outdated, wrong, or a D&D holdover. The tools have the actual rules text.
- Tactical/combat scenarios: if the scenario mentions a condition (flanked, \
frightened, off-guard, wounded, etc.), look it up with search_rules or \
search_content(types="condition") BEFORE advising on tactics. The mechanical \
effect determines the correct tactical answer.
- Keep search_rules queries SHORT (1-3 key terms). Do NOT include book or AP names \
in the query text — book names belong in the book= parameter, not the query.
- For AP-specific subsystem rules (kingdom building, influence, chases, hexploration): \
these are NOT reliably in search_rules. PRIMARY approach: browse_book(book="Kingmaker") \
→ find the subsystem chapter name → browse_book(book="Kingmaker", chapter="Chapter Name") \
for page-by-page subsystem detail. search_content(types="subsystem", book="AP") returns \
individual entries (Kingdom Events, turn phases) but NOT the full rule set. \
search_rules is for core rulebook mechanics, not AP-specific systems.
- If search_rules returns no results, retry with shorter/different key terms. \
If still thin, try search_content(types="rule") or search_pages. Exhausting \
retries beats reporting the rule is unknowable.
- For "how does X work in play" or "explain X for someone coming from 5e" style \
questions about core mechanics (actions, reactions, conditions, skill checks): \
search_rules("X") AND search_pages(query="X", book="Player Core"). The rules text \
in the DB is often split across entity entries (brief) and raw page text (full). \
Example: "three actions" → search_rules("your turn actions") + search_pages("single \
actions reactions free actions", book="Player Core") to get the complete explanation.
- Core rules/actions/feats are SPLIT across Player Core and Player Core 2 — \
NEVER scope a core rules search to a single book. Omit the book= filter for \
anything from the base rulebooks. "Tumble Through" is in Player Core 2; \
"Opportune Backstab" is in Player Core; searching only one will miss content.
- If a rules result is present but appears TRUNCATED (no DCs, no outcome table, \
no critical success/failure text), follow up with search_pages(query="rule name", \
book="Player Core") to get the full page text including tables.

NPC lookups:
- For roleplay requests ("write X's greeting", "roleplay X meeting the party", \
"voice X's reaction"): use lookup_npc(name="X", book="AP Title") directly. \
Do NOT use search_content or search_lore — they return the wrong results for \
creative/roleplay tasks. lookup_npc gives personality, motivation, and voice.
- For questions about how an NPC will react, their demeanor, or what they would do \
("how would X react to Y?", "X's demeanor toward the party", "what should X do?", \
"how does X change across the AP?") → ALWAYS call lookup_npc(name="X") first. \
NPC behavior answers require knowing the NPC's actual personality traits, even when \
the question is phrased as GM advice rather than a data lookup.
- For AP-specific NPCs ("the Stag Lord", "Grandmother Hu", "Oleg Leveton"), \
ALWAYS scope the lookup: lookup_npc(name="X", book="AP Title"). A generic \
search_content(types="npc", query="Stag Lord") may return a different entity with \
the same name from another source. The book= parameter disambiguates.
- If lookup_npc returns no match or returns only a deity/organization when you \
searched for a person, try name variants:
  · Remove title prefixes: "Mayor X" → try "X"; "Captain X" → try "X"
  · Nickname forms: "Grandmother X" → try "Granny X"; "Doctor X" → try just "X"
  · Surname only: if full name returns wrong entries, try surname alone
- NEVER truncate or abbreviate a name with "…" or "..." in a lookup call. \
Always use the full name: lookup_npc(name="Tar-Baphon") NOT lookup_npc(name="Tar-…"). \
A truncated name will fail or return a completely wrong entity.
- A deity result is NOT an NPC result. If you searched for an NPC and got a deity, \
the NPC was not found — try variants or note the gap.
- A MISMATCH is when a result's key attributes directly contradict the character as \
described in the question (e.g., you searched for a village elder matriarch and got \
a young male fishmonger). A mismatch means the lookup hit the WRONG entity. \
When this happens: your NEXT call MUST be lookup_npc with an alternate name form \
(nickname, shortened name, just surname). Do NOT fall back to search_content — \
that will return a pile of unrelated matches. lookup_npc with a variant is the fix.

Use natural type names for search_content (e.g. "creature", "npc", "spell", "feat", \
"condition", "hazard", "settlement", "deity") — the tool maps synonyms automatically. \
For NPC searches use search_content(types="npc", ...) or lookup_npc.

Rules adjudication ("can X do Y?"):
- Look up EACH ability/action mentioned separately

Search strategy:
- If results are thin or irrelevant, try: different query, different tool, \
remove type filter, remove book filter, or search_pages as last resort
- Do NOT call the same tool+type combination more than twice. If two \
search_content(types="landmark", ...) calls both return nothing useful, \
switch to search_pages(query="X") — raw page text often contains what \
typed-entity searches miss (encounter rooms, unnamed shrines, minor locations).
- If a tool call result contains "Error:" — do not retry the same call. \
It will fail again. Move on with what you have or try a different tool.
- A result prefixed "(No results in 'X' — showing results from all books)" means \
your book-scoped query found NOTHING in that book. The fallback results are from \
other books and will not answer your question — change your query or remove the type filter.
- For plot, story, NPC relationships, narrative events, or cross-AP/lore connections: \
use search_pages — raw page text contains narrative detail that entity searches miss. \
Do NOT use type filters for narrative content. Type filters are for mechanical \
lookups (spells, feats, conditions).
- Do NOT repeat the same tool call with the same arguments
- For comparisons, look up BOTH sides separately

NEVER produce a text response without calling tools. ALWAYS search first."""


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


@dataclass
class SplitResult:
    """Full output from the two-phase pipeline."""

    response: str
    retrieval_rounds: list[RetrievalResult]
    synthesis_model: str
    total_duration_ms: float
    reflection_rounds_used: int = 0


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
    ):
        self.retrieval_llm = retrieval_llm
        self.synthesis_llm = synthesis_llm
        self.mcp = mcp
        self.synthesis_system_prompt = synthesis_system_prompt
        self.verbose = verbose

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
        response = self._narrate(
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
            Message(role="system", content=ORCHESTRATOR_PROMPT),
        ]
        messages.extend(recent_turns)
        messages.append(Message(role="user", content=query))

        results: list[tuple[str, dict, str]] = []
        seen_calls: set[str] = set()
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
                    )
                except Exception as exc:
                    # Handle malformed tool calls (e.g. None id/name from small models)
                    if self.verbose:
                        print(f"  [Error] Orchestrator LLM error: {exc}")
                    response = None

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
                                "You have only made {n} tool call(s). "
                                "Search for more information before handing off. "
                                "Try different queries or tools."
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
                                    "Not enough reference material gathered yet "
                                    "({n} tool call(s) so far). Keep researching — "
                                    "look up specific entities, rules, or locations "
                                    "relevant to the question. Try different queries "
                                    "or tools if initial results were thin."
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
                                    "term from summaries]', book='...') or lookup_npc/lookup_creature "
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

        duration_ms = (time.time() - t0) * 1000
        return RetrievalResult(
            tool_calls=results,
            duration_ms=duration_ms,
            model=self.retrieval_llm.get_model_name(),
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
    ) -> str:
        """Phase 2: Toolless grounded response generation.

        Returns:
            The narrator's response text.
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

        response = self.synthesis_llm.chat(
            messages,
            tools=None,
            temperature=SYNTHESIS_TEMPERATURE,
        )

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
            )
            cleaned_text = _strip_special_tokens(response.text)

        # Hard fallback
        if not cleaned_text:
            context_text = self._format_all_retrievals(retrievals)
            if context_text:
                return "Here's what I found from the database:\n\n" + context_text
            return "I wasn't able to formulate a response. Please try rephrasing your question."

        return cleaned_text

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
