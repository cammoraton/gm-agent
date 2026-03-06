"""Lightweight chat agent for GM assistance without campaign state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import TEMPERATURE_CREATIVE
from .mcp.base import ToolResult
from .mcp.client import MCPClient
from .models.base import LLMBackend, LLMResponse, Message, ToolCall
from .models.factory import get_backend, get_chat_backend, get_narrator_backend

if TYPE_CHECKING:
    from .mcp.foundry_vtt import FoundryVTTServer
    from .split import SplitResult

# System prompt for chat mode (no campaign context)
CHAT_SYSTEM_PROMPT = """You are an assistant for Pathfinder 2nd Edition (Remaster) - NOT D&D or other systems.

CRITICAL RULES:
1. ONLY state facts that come directly from your search tool results
2. If a search returns limited or no results, say "I couldn't find detailed information about X in the database"
3. NEVER fabricate details, stats, locations, NPCs, or lore not in the search results
4. NEVER reference D&D content (no Lolth, Forgotten Realms, etc.) - this is Pathfinder/Golarion only
5. When uncertain, say so clearly rather than guessing
6. If your search returned N results, those N results are ALL the matching entries in the database. Do not infer that additional entries exist beyond what was returned.
7. When listing game content (creatures, spells, etc.), present ONLY the entries from tool results. Never add entries "for completeness" that weren't in the results.
8. Thin search results reflect a search limitation — do NOT make strong negative claims about AP content based on what searches didn't return.

NEVER INVENT PROPER NOUNS:
- Do NOT invent NPC names, creature names, spell names, feat names, AP titles, area codes, book titles, or organization names
- If your searches did not return a specific name, do NOT guess or fabricate one — state what you found and what you couldn't find
- This is the most important rule: a response that honestly says "I found X but couldn't find Y" is ALWAYS better than one that invents Y
- When searches return thin results, try 2-3 different queries/tools before concluding the information isn't available

TONE:
- Respond as a knowledgeable GM friend, not as a search engine
- Never say "Based on the search results", "According to my search", "The database shows", or similar meta-language
- Present information naturally as if you know it, while still being honest about gaps
- When you don't know something, say "I don't have information on that" not "My search didn't return results"

Available tools and when to use them:

LOOKUP (by name — use when you know the entity name):
- lookup(type, name, book?): look up any named entity. Types: creature, spell, item, location, hazard, npc, encounter, class
  - type=creature: stat blocks, abilities, ecology
  - type=npc: NPC personality, roleplay info, motivations; use book= for AP-specific NPCs
  - type=location: encounter room descriptions and read-aloud/boxed text — ALWAYS specify book= for dungeon locations
  - type=class: class entry (key ability, roles, overview); use search_rules for specific mechanics

SEARCH (by concept — use when exploring):
- search_rules: Game mechanics, conditions, actions (rulebooks only, excludes creatures/spells/NPCs)
- search_content: General search — supports type filters, book/chapter scope, level ranges. Use for deities, mixed-type queries, or when other tools don't fit.
  - scope="lore": World/setting lore — locations, regions, deities, history, organizations. NOT for NPCs.
  - scope="guidance": GM advice and how-to-run tips (semantic search)
  - types="relationship": NPC-to-NPC relationships (ally, rival, enemy, family, patron, etc.)
  - types="faction": Factions/groups with members, territory, goals, and conflicts
- search_pages: Raw page text for flavor, narrative passages, specific wording

BROWSE/LIST (for inventories and structure):
- browse_book: Book summaries, chapter TOCs, page summaries
- list_entities: Structured table of entities by type. Use for "list all NPCs in X", "all deities", "dragons in Bestiary". Supports trait filter (e.g., trait="dragon") and level range (min_level/max_level) for creatures.

OTHER:
- evaluate_encounter, suggest_encounter: Encounter building and evaluation
- roll_dice, roll_check: Dice rolling with PF2e degree of success
- add_note, search_notes: Record and recall session notes

IMPORTANT:
- For NPCs: use lookup(type="npc", name="X") or list_entities(type="npc", book="X") (for lists)
- For deities: use search_content(scope="lore") or list_entities(type="deity")
- For creatures: use lookup(type="creature", name="X") or list_entities(type="creature")
- For creature families ("all types of dragon"): use list_entities(type="creature_family", name_filter="dragon") — this returns the full taxonomy, not just top-10
- NEVER scope creature searches to "Player Core" — creatures are in Monster Core, Bestiary, Bestiary 2/3, Draconic Codex
- For world lore/places: use search_content(scope="lore") (Absalom, Cheliax, etc.)
- For NPC relationship maps / faction dynamics: use search_content(types="relationship", book="X") AND search_content(types="faction", book="X")
- Book names in queries are auto-detected — no need to put them in the query text
- Do NOT use types='adventure', 'book', 'chapter', 'article', or 'event' — these are not valid content types
- For search_content: Districts and buildings are type "landmark" (NOT "location"). Use types="settlement" for cities/towns. Use types="encounter" for AP rooms/areas (alias for "location").
- AP name reference: "Season of Ghosts" (NOT "Season of the Witch"), "Kingmaker", "Abomination Vaults", "Blood Lords", "Stolen Fate"

When answering questions about Pathfinder content:
1. Use the search tools to find official information
2. Quote or summarize ONLY what the tools return
3. If results are sparse, try different tools or broader searches before giving up
4. For creative suggestions (encounters, plot hooks), clearly label them as your suggestions vs. official content

For roleplay and creative requests:
1. ALWAYS look up the NPC/creature/location first, even for "roleplay as X" requests
2. Use the lookup results to ground your portrayal in official personality, appearance, and motivations
3. Creative embellishment of voice and atmosphere is encouraged -- inventing facts is not

For adventure path questions:
1. Start with browse_book to understand the AP structure
2. ALWAYS use the `book` parameter on search_content/list_entities to scope results to the correct AP
3. Use list_entities to find NPCs, creatures, and locations in specific books
4. Use search_pages for detailed passage lookups
5. Do NOT mix content from different APs -- if asked about Season of Ghosts, only present Season of Ghosts results
6. Keep queries SHORT (1-3 terms) and use book= for scoping -- do NOT put AP names in the query text
7. For NPC inventories, use list_entities(type="npc", book="X") — NOT search_content

For encounter evaluation, use evaluate_encounter with creature levels.
For dice, use roll_dice (e.g., "2d6+4") or roll_check (modifier vs DC).

SEARCH QUERY TIPS:
- Keep search queries SHORT: 1-3 key terms only
- Do NOT add synonyms or descriptors (bad: "healing restoration cure", good: "heal")
- Book names are auto-detected — don't include them in the query text

Be concise and accurate. Accuracy is more important than completeness.
If tool results don't fully answer the question, state what was found and what was NOT found — never fill gaps with assumed information.
"""


# System prompt for the synthesis phase of the split pipeline.
# Removes tool selection guidance, adds grounding rules.
SYNTHESIS_SYSTEM_PROMPT = """\
You are a Pathfinder 2e GM assistant. Answer using ONLY facts in the Reference Material below.

Your PF2e training knowledge is unreliable — pre-remaster rules, D&D content, \
plausible hallucinations. Trust the Reference Material over memory. \
When it's absent or thin, say so — a short accurate answer beats a padded one.

GROUNDING:
- Every name, number, stat, and plot detail must appear in the Reference Material
- Never invent creatures, NPCs, named hazards, traps, haunts, named locations (shrines, \
temples, buildings, rooms, encounter areas), abilities, DCs, or mechanics to fill gaps
- NEVER cite a specific sourcebook page for invented content ("Haunting Echoes, p.25") — \
if you don't have the data, say what general mechanics you found and give GM craft advice
- If the question asks about "the shrine outside town" and the Reference Material shows \
"Abadar Shrine" and "Nine Ear Shrine," present those found locations — do NOT invent a \
third shrine that better matches the question's description
- Encounter creatures: name only what the tools returned — never complete a roster by inventing
- NPC relationship maps: only describe relationships, alliances, and conflicts that are \
EXPLICITLY stated in the Reference Material. NPC personality entries describe the NPC's \
traits — they do NOT imply relationships with other NPCs. Do not infer "A and B are \
allies" from the fact that both live in the same town.
- Book/chapter summaries describe themes and structure, not specific plot beats — \
don't use a summary to infer NPC names, encounter details, or how mysteries resolve
- "Not found" ≠ "doesn't exist" — say you couldn't find it, not that it doesn't exist

WHEN RESULTS ARE THIN:
- Use what's there and note the gap — don't pad with invented specifics
- Behavior/reaction questions: character personality data is enough to construct an answer
- General GM craft advice (pacing, tension, revelation) is fine — label it as general advice, \
never present it as sourced AP content
- Roleplay requests: a hedged portrayal beats a refusal; generic NPCs (a shopkeeper, \
a guard, a town elder) can be invented when no specific NPC exists in the material — \
clearly label them as your suggestions. Named NPCs not found in the Reference Material: \
invent a plausible voice for their role and note you couldn't confirm their official description.

FORMAT: Prose for lore and narrative. Tables only for genuinely tabular data. \
Lead with the answer. Shorter beats exhaustive.

TONE: Knowledgeable GM friend. Never say "Based on the reference material." \
Synthesize naturally. Accuracy > completeness.

No tools available — all information is in the Reference Material. Do not fabricate dice results."""


class ChatAgent:
    """A lightweight chat agent with RAG, encounter, dice, and notes tools.

    Uses MCPClient for tool execution, which supports both local and remote modes.
    Available tools:
    - Quick rules lookups
    - GM prep assistance
    - Encounter evaluation and building
    - Dice rolling and check resolution
    - Session note-taking
    - General Pathfinder 2e questions
    - Foundry VTT integration (when connected)
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        narrator_llm: LLMBackend | None = None,
        verbose: bool = False,
        system_prompt: str | None = None,
        enable_rag: bool = True,
        enable_encounters: bool = True,
        enable_dice: bool = True,
        enable_notes: bool = True,
        foundry_server: "FoundryVTTServer | None" = None,
    ):
        """Initialize the chat agent.

        Args:
            llm: The main LLM backend — drives tool orchestration (retrieval).
                 Defaults to ``get_backend()`` (LLM_BACKEND env var).
            narrator_llm: Optional separate LLM for the narrator/synthesis phase.
                 Defaults to the same model as ``llm``.
        """
        self.llm = llm or get_chat_backend()
        self.verbose = verbose
        self.system_prompt = system_prompt or CHAT_SYSTEM_PROMPT
        self.foundry_server = foundry_server

        # Store enable flags for reference
        self._enable_rag = enable_rag
        self._enable_encounters = enable_encounters
        self._enable_dice = enable_dice
        self._enable_notes = enable_notes

        # Conversation history
        self._messages: list[Message] = [Message(role="system", content=self.system_prompt)]

        # Last SplitResult from pipeline (for tool extraction by eval harness)
        self._last_result: "SplitResult | None" = None

        # Initialize MCP client (no campaign context for chat)
        self._mcp = MCPClient(
            context={},
            foundry_server=foundry_server,
        )

        # Convenience accessors for servers (backward compatibility)
        self.rag_server = self._mcp.get_server("pf2e-rag") if enable_rag else None
        self.encounter_server = self._mcp.get_server("encounter") if enable_encounters else None
        self.dice_server = self._mcp.get_server("dice") if enable_dice else None
        self.notes_server = self._mcp.get_server("notes") if enable_notes else None

        # Split pipeline (always on)
        # llm = orchestrator (tool calls), narrator_llm = synthesis (final response)
        from .split import SplitPipeline

        self._pipeline = SplitPipeline(
            retrieval_llm=self.llm,
            synthesis_llm=narrator_llm or get_narrator_backend() or self.llm,
            mcp=self._mcp,
            synthesis_system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            verbose=verbose,
        )

    def get_tools(self) -> list:
        """Get all available tools from MCPClient."""
        return self._mcp.list_tools()

    def chat(self, user_input: str) -> str:
        """Process user input and return a response.

        Uses the two-phase split pipeline: orchestrator gathers information,
        then narrator produces the final response.

        Args:
            user_input: The user's message

        Returns:
            The assistant's response
        """
        self._messages.append(Message(role="user", content=user_input))

        # Recent turns for context (exclude the one we just added)
        recent = [m for m in self._messages[-6:] if m.role in ("user", "assistant")]

        result = self._pipeline.run(
            query=user_input,
            conversation_history=recent[:-1] if len(recent) > 1 else None,
        )
        self._last_result = result

        self._messages.append(Message(role="assistant", content=result.response))
        return result.response

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call via MCPClient."""
        return self._mcp.call_tool(tool_call.name, tool_call.args)

    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self._messages = [Message(role="system", content=self.system_prompt)]
        self._last_result = None

    def get_last_result(self):
        """Get the SplitResult from the most recent chat() call."""
        return self._last_result

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self._messages.copy()

    def close(self) -> None:
        """Clean up resources."""
        self._mcp.close()
