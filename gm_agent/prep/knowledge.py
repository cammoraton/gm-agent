"""Knowledge synthesis functions for campaign prep.

Each function follows the same pattern:
1. Query search.db for relevant entities
2. Group/batch entities for LLM calls
3. Call LLM with synthesis prompt
4. Parse structured JSON output into KnowledgeEntry objects
5. Write to KnowledgeStore
6. Log each step via PrepLogger
"""

import json
import logging
import re
import time
from collections.abc import Callable

from ..models.base import LLMBackend, LLMResponse, Message
from ..rag.search import PathfinderSearch
from ..storage.knowledge import KnowledgeStore
from .log import PrepLogEntry, PrepLogger
from .prompts import (
    GENERATE_BACKGROUND_SYSTEM,
    GENERATE_BACKGROUND_USER,
    NPC_KNOWLEDGE_SYSTEM,
    NPC_KNOWLEDGE_USER,
    PARTY_KNOWLEDGE_SYSTEM,
    PARTY_KNOWLEDGE_USER,
    SUBSYSTEM_KNOWLEDGE_SYSTEM,
    SUBSYSTEM_KNOWLEDGE_USER,
    WORLD_CONTEXT_SYSTEM,
    WORLD_CONTEXT_USER,
    WORLD_CONTEXT_QUERY_SYSTEM,
    WORLD_CONTEXT_QUERY_USER,
)

logger = logging.getLogger(__name__)

# Party knowledge character ID
PARTY_KNOWLEDGE_ID = "__party__"

# Categories relevant for party knowledge extraction
PARTY_LORE_CATEGORIES = [
    "location",
    "settlement",
    "region",
    "organization",
    "deity",
    "npc",
    "guidance",
]

# Categories relevant for world context extraction
WORLD_CONTEXT_CATEGORIES = [
    "location",
    "settlement",
    "region",
    "organization",
    "deity",
    "landmark",
    "faction",
    "historical_event",
]

# Max entities per LLM batch to stay within context limits
BATCH_SIZE = 15

# Thinking config for synthesis calls
THINKING_CONFIG = {"type": "enabled", "budget_tokens": 4096}

# Canonical subsystem names → detection keyword lists
SUBSYSTEM_KEYWORDS: dict[str, list[str]] = {
    "kingdom": ["kingdom building", "kingdom turn", "kingdom activities", "civic activities"],
    "influence": ["influence subsystem", "influence encounter", "influence points"],
    "chase": ["chase subsystem", "chase points", "chase obstacle"],
    "infiltration": ["infiltration subsystem", "infiltration points", "awareness points"],
    "research": ["research subsystem", "research points", "learn more"],
    "hexploration": ["hexploration", "hex exploration", "explore hex"],
    "reputation": ["reputation subsystem", "reputation points"],
}

# Content types relevant for subsystem rule extraction
SUBSYSTEM_CONTENT_TYPES = ["subsystem", "game_mechanic", "action", "table", "guidance"]


def _format_entities_text(entities: list[dict]) -> str:
    """Format a batch of entities into text for LLM prompts."""
    parts = []
    for e in entities:
        header = f"[{e['type']}] {e['name']}"
        if e.get("page"):
            header += f" (p.{e['page']})"
        parts.append(f"{header}\n{e['content'][:2000]}")
    return "\n\n---\n\n".join(parts)


def _parse_json_array(text: str) -> list[dict]:
    """Extract and parse a JSON array from LLM response text.

    Handles cases where the LLM wraps the array in markdown code fences.
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse JSON array from LLM response")
        return []


def _call_llm(
    llm: LLMBackend,
    system_prompt: str,
    user_prompt: str,
    use_thinking: bool = True,
) -> LLMResponse:
    """Make an LLM call with optional thinking enabled."""
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    thinking = THINKING_CONFIG if use_thinking else None
    return llm.chat(messages, thinking=thinking)


def _slugify_name(name: str) -> str:
    """Convert an NPC name to a character_id slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _get_tokens(usage: dict | None) -> tuple[int, int]:
    """Extract prompt and completion token counts from usage dict."""
    if not usage:
        return 0, 0
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def seed_party_knowledge(
    search: PathfinderSearch,
    knowledge: KnowledgeStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    books: list[dict],
    campaign_id: str,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Seed party baseline knowledge from Players Guides and setting books.

    Returns count of knowledge entries created.
    """
    total = 0

    for book_info in books:
        book_name = book_info["name"]
        logger.info("Seeding party knowledge from: %s", book_name)

        entities = search.list_entities(
            book=book_name,
            category=PARTY_LORE_CATEGORIES,
        )

        if not entities:
            logger.warning("No entities found in %s for party knowledge", book_name)
            continue

        total_batches = (len(entities) + BATCH_SIZE - 1) // BATCH_SIZE
        book_total = 0
        skipped_total = 0

        # Process in batches
        for batch_idx, i in enumerate(range(0, len(entities), BATCH_SIZE), 1):
            batch = entities[i : i + BATCH_SIZE]
            entities_text = _format_entities_text(batch)

            user_prompt = PARTY_KNOWLEDGE_USER.format(
                book_name=book_name,
                entities_text=entities_text,
            )

            start = time.monotonic()
            response = _call_llm(llm, PARTY_KNOWLEDGE_SYSTEM, user_prompt)
            duration_ms = (time.monotonic() - start) * 1000

            entries = _parse_json_array(response.text)
            output_entries = []

            skipped = 0
            for entry in entries:
                content = entry.get("content", "")
                if not content:
                    continue

                # Dedup check
                if knowledge.has_similar_knowledge(PARTY_KNOWLEDGE_ID, content):
                    skipped += 1
                    continue

                tags = entry.get("tags", [])
                tags.extend(["players_guide", book_name])
                # Deduplicate tags
                tags = list(dict.fromkeys(tags))

                ke = knowledge.add_knowledge(
                    character_id=PARTY_KNOWLEDGE_ID,
                    character_name="Party",
                    content=content,
                    knowledge_type="fact",
                    sharing_condition="free",
                    source=f"prep:{book_name}",
                    importance=entry.get("importance", 5),
                    decay_rate=0.0,
                    tags=tags,
                )
                output_entries.append(ke.to_dict())
                total += 1
                book_total += 1

            skipped_total += skipped

            if skipped:
                logger.info("Party knowledge: skipped %d duplicate(s) from %s", skipped, book_name)

            if on_progress:
                on_progress(
                    f"Party knowledge: batch {batch_idx}/{total_batches} from "
                    f"{book_name} ({book_total} entries so far)"
                )

            # Log for training data
            logger_.log(PrepLogEntry(
                step="party_knowledge",
                campaign_id=campaign_id,
                book=book_name,
                entity=None,
                input_context=entities_text,
                system_prompt=PARTY_KNOWLEDGE_SYSTEM,
                thinking=response.thinking,
                output=output_entries,
                model=llm.get_model_name(),
                duration_ms=duration_ms,
                token_usage=response.usage,
            ))

        if on_progress:
            on_progress(
                f"  → {book_total} entries from {book_name}"
                + (f" ({skipped_total} duplicates skipped)" if skipped_total else "")
            )

    logger.info("Party knowledge seeded: %d entries", total)
    return total


def seed_npc_knowledge(
    search: PathfinderSearch,
    knowledge: KnowledgeStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    books: list[dict],
    campaign_id: str,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Seed NPC knowledge from AP content.

    Returns count of knowledge entries created.
    """
    total = 0

    for book_info in books:
        book_name = book_info["name"]
        logger.info("Seeding NPC knowledge from: %s", book_name)

        npcs = search.list_entities(
            book=book_name,
            category="npc",
        )

        if not npcs:
            logger.warning("No NPCs found in %s", book_name)
            continue

        total_npcs = len(npcs)

        for npc_idx, npc in enumerate(npcs, 1):
            npc_name = npc["name"]
            character_id = _slugify_name(npc_name)
            entity_text = _format_entities_text([npc])

            # Get surrounding page context for richer synthesis
            page_results = search.search_pages(
                npc_name, book=book_name, top_k=3
            )
            page_context = "\n\n".join(
                f"[p.{p['page_number']}] {p['snippet']}" for p in page_results
            ) if page_results else "(No additional page context found)"

            user_prompt = NPC_KNOWLEDGE_USER.format(
                book_name=book_name,
                npc_name=npc_name,
                entity_text=entity_text,
                page_context=page_context,
            )

            start = time.monotonic()
            response = _call_llm(llm, NPC_KNOWLEDGE_SYSTEM, user_prompt)
            duration_ms = (time.monotonic() - start) * 1000

            entries = _parse_json_array(response.text)
            output_entries = []

            skipped = 0
            for entry in entries:
                content = entry.get("content", "")
                if not content:
                    continue

                # Dedup check
                if knowledge.has_similar_knowledge(character_id, content):
                    skipped += 1
                    continue

                tags = entry.get("tags", [])
                tags.extend(["npc", book_name])
                tags = list(dict.fromkeys(tags))

                knowledge_type = entry.get("knowledge_type", "fact")
                sharing_condition = entry.get("sharing_condition", "free")

                ke = knowledge.add_knowledge(
                    character_id=character_id,
                    character_name=npc_name,
                    content=content,
                    knowledge_type=knowledge_type,
                    sharing_condition=sharing_condition,
                    source=f"prep:{book_name}",
                    importance=entry.get("importance", 5),
                    decay_rate=0.0,
                    tags=tags,
                )
                output_entries.append(ke.to_dict())
                total += 1

            if skipped:
                logger.info("NPC %s: skipped %d duplicate(s)", npc_name, skipped)

            prompt_tok, comp_tok = _get_tokens(response.usage)
            if on_progress:
                on_progress(
                    f"NPC {npc_idx}/{total_npcs}: {npc_name} — "
                    f"{len(output_entries)} entries ({duration_ms / 1000:.1f}s"
                    + (f", {prompt_tok + comp_tok} tokens" if prompt_tok + comp_tok else "")
                    + ")"
                )

            logger_.log(PrepLogEntry(
                step="npc_knowledge",
                campaign_id=campaign_id,
                book=book_name,
                entity=npc_name,
                input_context=entity_text + "\n\n" + page_context,
                system_prompt=NPC_KNOWLEDGE_SYSTEM,
                thinking=response.thinking,
                output=output_entries,
                model=llm.get_model_name(),
                duration_ms=duration_ms,
                token_usage=response.usage,
            ))

    logger.info("NPC knowledge seeded: %d entries", total)
    return total


def seed_world_context(
    search: PathfinderSearch,
    knowledge: KnowledgeStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    books: list[dict],
    campaign_id: str,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Seed world context from setting books.

    Returns count of knowledge entries created.
    """
    total = 0

    for book_info in books:
        book_name = book_info["name"]
        logger.info("Seeding world context from: %s", book_name)

        entities = search.list_entities(
            book=book_name,
            category=WORLD_CONTEXT_CATEGORIES,
        )

        if not entities:
            logger.warning("No entities found in %s for world context", book_name)
            continue

        total_batches = (len(entities) + BATCH_SIZE - 1) // BATCH_SIZE
        book_total = 0
        skipped_total = 0

        for batch_idx, i in enumerate(range(0, len(entities), BATCH_SIZE), 1):
            batch = entities[i : i + BATCH_SIZE]
            entities_text = _format_entities_text(batch)

            user_prompt = WORLD_CONTEXT_USER.format(
                book_name=book_name,
                entities_text=entities_text,
            )

            start = time.monotonic()
            response = _call_llm(llm, WORLD_CONTEXT_SYSTEM, user_prompt)
            duration_ms = (time.monotonic() - start) * 1000

            entries = _parse_json_array(response.text)
            output_entries = []

            skipped = 0
            for entry in entries:
                content = entry.get("content", "")
                if not content:
                    continue

                # Dedup check
                if knowledge.has_similar_knowledge(PARTY_KNOWLEDGE_ID, content):
                    skipped += 1
                    continue

                tags = entry.get("tags", [])
                tags.extend(["setting", book_name])
                tags = list(dict.fromkeys(tags))

                ke = knowledge.add_knowledge(
                    character_id=PARTY_KNOWLEDGE_ID,
                    character_name="World",
                    content=content,
                    knowledge_type="fact",
                    sharing_condition="free",
                    source=f"prep:{book_name}",
                    importance=entry.get("importance", 4),
                    decay_rate=0.0,
                    tags=tags,
                )
                output_entries.append(ke.to_dict())
                total += 1
                book_total += 1

            skipped_total += skipped

            if skipped:
                logger.info("World context: skipped %d duplicate(s) from %s", skipped, book_name)

            if on_progress:
                on_progress(
                    f"World context: batch {batch_idx}/{total_batches} from "
                    f"{book_name} ({book_total} entries so far)"
                )

            logger_.log(PrepLogEntry(
                step="world_context",
                campaign_id=campaign_id,
                book=book_name,
                entity=None,
                input_context=entities_text,
                system_prompt=WORLD_CONTEXT_SYSTEM,
                thinking=response.thinking,
                output=output_entries,
                model=llm.get_model_name(),
                duration_ms=duration_ms,
                token_usage=response.usage,
            ))

        if on_progress:
            on_progress(
                f"  → {book_total} entries from {book_name}"
                + (f" ({skipped_total} duplicates skipped)" if skipped_total else "")
            )

    logger.info("World context seeded: %d entries", total)
    return total


def seed_world_context_by_query(
    search: PathfinderSearch,
    knowledge: KnowledgeStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    search_terms: list[str],
    campaign_id: str,
    campaign_background: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Seed world context using targeted search queries.

    Instead of bulk-loading entities from entire setting books, searches
    across ALL books for specific terms relevant to the campaign.

    Args:
        search: PathfinderSearch instance.
        knowledge: KnowledgeStore to write entries to.
        llm: LLM backend for synthesis.
        logger_: Prep logger for training data.
        search_terms: List of search terms (e.g. ["Stolen Lands", "Brevoy"]).
        campaign_id: Campaign identifier.
        campaign_background: Optional campaign background for LLM context.
        on_progress: Optional callback for progress messages.

    Returns:
        Count of knowledge entries created.
    """
    total = 0

    # Collect entities from search, dedup by name
    seen_names: set[str] = set()
    all_entities: list[dict] = []

    if on_progress:
        on_progress(f"World context: searching {len(search_terms)} terms across all books...")

    for term in search_terms:
        results = search.search(term, top_k=10)
        for r in results:
            name = r.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                all_entities.append(r)

    if on_progress:
        on_progress(f"World context: found {len(all_entities)} unique entities from {len(search_terms)} search terms")

    if not all_entities:
        logger.warning("No entities found for search terms: %s", search_terms)
        return 0

    # Process in batches
    total_batches = (len(all_entities) + BATCH_SIZE - 1) // BATCH_SIZE
    skipped_total = 0

    for batch_idx, i in enumerate(range(0, len(all_entities), BATCH_SIZE), 1):
        batch = all_entities[i : i + BATCH_SIZE]
        entities_text = _format_entities_text(batch)

        # Collect source books for this batch
        batch_books = list(dict.fromkeys(e.get("book", "Unknown") for e in batch))

        user_prompt = WORLD_CONTEXT_QUERY_USER.format(
            campaign_background=campaign_background or "(No background provided)",
            search_terms=", ".join(search_terms),
            entities_text=entities_text,
        )

        start = time.monotonic()
        response = _call_llm(llm, WORLD_CONTEXT_QUERY_SYSTEM, user_prompt)
        duration_ms = (time.monotonic() - start) * 1000

        entries = _parse_json_array(response.text)
        output_entries = []

        skipped = 0
        for entry in entries:
            content = entry.get("content", "")
            if not content:
                continue

            # Dedup check
            if knowledge.has_similar_knowledge(PARTY_KNOWLEDGE_ID, content):
                skipped += 1
                continue

            tags = entry.get("tags", [])
            tags.extend(["setting", "world_context"])
            tags = list(dict.fromkeys(tags))

            ke = knowledge.add_knowledge(
                character_id=PARTY_KNOWLEDGE_ID,
                character_name="World",
                content=content,
                knowledge_type="fact",
                sharing_condition="free",
                source="prep:search_queries",
                importance=entry.get("importance", 4),
                decay_rate=0.0,
                tags=tags,
            )
            output_entries.append(ke.to_dict())
            total += 1

        skipped_total += skipped

        if skipped:
            logger.info("World context (query): skipped %d duplicate(s)", skipped)

        if on_progress:
            on_progress(
                f"World context: batch {batch_idx}/{total_batches} "
                f"({total} entries so far, sources: {', '.join(batch_books)})"
            )

        logger_.log(PrepLogEntry(
            step="world_context_query",
            campaign_id=campaign_id,
            book=", ".join(batch_books),
            entity=None,
            input_context=entities_text,
            system_prompt=WORLD_CONTEXT_QUERY_SYSTEM,
            thinking=response.thinking,
            output=output_entries,
            model=llm.get_model_name(),
            duration_ms=duration_ms,
            token_usage=response.usage,
        ))

    if on_progress:
        on_progress(
            f"  → {total} world context entries from {len(search_terms)} search terms"
            + (f" ({skipped_total} duplicates skipped)" if skipped_total else "")
        )

    logger.info("World context (query) seeded: %d entries from %d terms", total, len(search_terms))
    return total


# =========================================================================
# Subsystem detection and knowledge seeding
# =========================================================================


def detect_subsystems(
    search: PathfinderSearch,
    books: list[dict],
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Detect which PF2e subsystems an adventure path uses.

    Scans AP books for subsystem keywords using page-level search.
    Only considers adventure books.

    Returns:
        List of detected subsystem type names (e.g. ["kingdom", "influence"]).
    """
    # Filter to adventure books only
    ap_books = [b for b in books if b.get("book_type") == "adventure"]
    if not ap_books:
        return []

    detected: list[str] = []

    for subsystem_type, keywords in SUBSYSTEM_KEYWORDS.items():
        found = False
        for keyword in keywords:
            if found:
                break
            for book_info in ap_books:
                results = search.search_pages(keyword, book=book_info["name"], top_k=1)
                if results:
                    detected.append(subsystem_type)
                    found = True
                    if on_progress:
                        on_progress(f"Detected subsystem: {subsystem_type} (via '{keyword}' in {book_info['name']})")
                    break

    return detected


def seed_subsystem_knowledge(
    search: PathfinderSearch,
    knowledge: KnowledgeStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    ap_books: list[dict],
    subsystem_types: list[str],
    campaign_id: str,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Seed subsystem rules knowledge from AP and core rulebook content.

    One LLM call per subsystem type. Gathers content from three sources:
    1. Entities with subsystem content types from AP books
    2. Search across all books for core rules (pulls from GM Core etc.)
    3. Page-level context from AP books

    Returns count of knowledge entries created.
    """
    total = 0

    for subsystem_type in subsystem_types:
        keywords = SUBSYSTEM_KEYWORDS.get(subsystem_type, [subsystem_type])
        primary_keyword = keywords[0]

        # 1. Entities from AP books with subsystem content types
        all_entities: list[dict] = []
        seen_names: set[str] = set()
        for book_info in ap_books:
            entities = search.list_entities(
                book=book_info["name"],
                include_types=SUBSYSTEM_CONTENT_TYPES,
            )
            for e in entities:
                name = e.get("name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_entities.append(e)

        # 2. Search across all books for core rules
        for keyword in keywords[:2]:  # First two keywords
            results = search.search(keyword, include_types=SUBSYSTEM_CONTENT_TYPES)
            for r in results:
                name = r.get("name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_entities.append(r)

        # 3. Page-level context from AP books
        page_parts: list[str] = []
        for book_info in ap_books:
            for keyword in keywords[:2]:
                page_results = search.search_pages(keyword, book=book_info["name"], top_k=3)
                for p in page_results:
                    page_parts.append(
                        f"[{book_info['name']} p.{p['page_number']}] {p['snippet']}"
                    )

        if not all_entities and not page_parts:
            if on_progress:
                on_progress(f"Subsystem {subsystem_type}: no content found, skipping")
            continue

        # Build combined text
        entities_text = _format_entities_text(all_entities) if all_entities else ""
        if page_parts:
            page_text = "\n\n".join(page_parts)
            if entities_text:
                entities_text += f"\n\n--- Page Context ---\n\n{page_text}"
            else:
                entities_text = page_text

        ap_name = ", ".join(b["name"] for b in ap_books)
        user_prompt = SUBSYSTEM_KNOWLEDGE_USER.format(
            subsystem_type=subsystem_type,
            ap_name=ap_name,
            entities_text=entities_text,
        )

        start = time.monotonic()
        response = _call_llm(llm, SUBSYSTEM_KNOWLEDGE_SYSTEM, user_prompt)
        duration_ms = (time.monotonic() - start) * 1000

        entries = _parse_json_array(response.text)
        output_entries = []

        skipped = 0
        for entry in entries:
            content = entry.get("content", "")
            if not content:
                continue

            # Dedup check
            if knowledge.has_similar_knowledge(PARTY_KNOWLEDGE_ID, content):
                skipped += 1
                continue

            tags = entry.get("tags", [])
            tags.extend(["subsystem", subsystem_type])
            tags = list(dict.fromkeys(tags))

            ke = knowledge.add_knowledge(
                character_id=PARTY_KNOWLEDGE_ID,
                character_name="Subsystem Rules",
                content=content,
                knowledge_type="fact",
                sharing_condition="free",
                source=f"prep:subsystem:{subsystem_type}",
                importance=entry.get("importance", 8),
                decay_rate=0.0,
                tags=tags,
            )
            output_entries.append(ke.to_dict())
            total += 1

        if skipped:
            logger.info("Subsystem %s: skipped %d duplicate(s)", subsystem_type, skipped)

        prompt_tok, comp_tok = _get_tokens(response.usage)
        if on_progress:
            on_progress(
                f"Subsystem {subsystem_type}: {len(output_entries)} entries "
                f"({duration_ms / 1000:.1f}s"
                + (f", {prompt_tok + comp_tok} tokens" if prompt_tok + comp_tok else "")
                + ")"
            )

        logger_.log(PrepLogEntry(
            step="subsystem_knowledge",
            campaign_id=campaign_id,
            book=ap_name,
            entity=subsystem_type,
            input_context=entities_text,
            system_prompt=SUBSYSTEM_KNOWLEDGE_SYSTEM,
            thinking=response.thinking,
            output=output_entries,
            model=llm.get_model_name(),
            duration_ms=duration_ms,
            token_usage=response.usage,
        ))

    logger.info("Subsystem knowledge seeded: %d entries", total)
    return total


# =========================================================================
# NPC name deduplication
# =========================================================================

# Titles to strip when comparing names
_TITLE_PREFIXES = [
    "lord ", "lady ", "king ", "queen ", "captain ", "baron ", "baroness ",
    "grand ranger ", "grand ", "sir ", "madame ", "chief ", "mayor ",
    "elder ", "prince ", "princess ",
]

# Suffixes that indicate a distinct entity (not a duplicate)
_DISTINCT_SUFFIXES = ["'s ", " fetch", " guards", " warriors", " scouts",
                      " cultists", " cutthroats", " heralds"]


def _normalize_for_compare(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip titles."""
    n = name.lower().strip()
    for prefix in _TITLE_PREFIXES:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return n


def _is_distinct_entity(name_a: str, name_b: str) -> bool:
    """Check if two names likely refer to distinct entities."""
    a_lower = name_a.lower()
    b_lower = name_b.lower()
    for suffix in _DISTINCT_SUFFIXES:
        # "Vordakai's Fetch" vs "Vordakai" → distinct
        if suffix in a_lower and suffix not in b_lower:
            return True
        if suffix in b_lower and suffix not in a_lower:
            return True
    return False


def _names_match(name_a: str, name_b: str) -> bool:
    """Determine if two NPC names refer to the same character.

    Handles:
    - Substring at word boundary: "Oleg" matches "Oleg Leveton" but not "Kob Moleg"
    - Title variants: "Lady Jamandi Aldori" matches "Jamandi Aldori"
    - Plural/singular: "Troll King" matches "Troll Kings"
    - Typos: "Maager Varn" matches "Maegar Varn" (edit distance ≤ 2, same first word)
    """
    if name_a == name_b:
        return True

    # Filter out distinct entities (fetches, guards, possessives)
    if _is_distinct_entity(name_a, name_b):
        return False

    norm_a = _normalize_for_compare(name_a)
    norm_b = _normalize_for_compare(name_b)

    if norm_a == norm_b:
        return True

    # Order by length: shorter, longer
    short, long = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)

    # Short name must be at least 4 chars to avoid coincidental matches
    if len(short) < 4:
        return False

    # Plural check: "troll king" vs "troll kings"
    if long == short + "s" or short == long + "s":
        return True

    # Substring at word boundary
    if short in long:
        idx = long.index(short)
        # Must start at beginning or after a space
        if idx == 0 or long[idx - 1] == " ":
            # Must end at end or before a space
            end = idx + len(short)
            if end == len(long) or long[end] == " ":
                return True

    # Edit distance for multi-word names with similar structure
    # Catches typos like "Maager Varn" / "Maegar Varn", "Ilraith Valadhkani" / "Valadkhani"
    words_a = norm_a.split()
    words_b = norm_b.split()
    if (words_a and words_b and len(words_a) == len(words_b)
            and len(words_a) >= 2 and len(norm_a) > 6):
        # At least one word must match exactly (first or last name)
        if words_a[0] == words_b[0] or words_a[-1] == words_b[-1]:
            dist = sum(1 for a, b in zip(norm_a, norm_b) if a != b)
            dist += abs(len(norm_a) - len(norm_b))
            if dist <= 2:
                return True

    return False


def _pick_canonical_name(names: list[tuple[str, str, int]]) -> tuple[str, str]:
    """Pick the best canonical name from a group of duplicates.

    Args:
        names: List of (character_id, character_name, entry_count) tuples.

    Returns:
        (canonical_character_id, canonical_character_name)

    Prefers: longest name (most specific), then most entries as tiebreaker.
    """
    # Sort by name length desc, then entry count desc
    sorted_names = sorted(names, key=lambda x: (len(x[1]), x[2]), reverse=True)
    winner = sorted_names[0]
    return winner[0], winner[1]


def deduplicate_npc_names(
    knowledge: "KnowledgeStore",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Merge duplicate NPC names in the knowledge store.

    Finds NPC name variants that refer to the same character and merges
    their knowledge entries under the canonical (longest) name.

    Returns:
        Dict with merge statistics: {"merges": int, "entries_moved": int, "groups": list}
    """
    conn = knowledge._get_conn()

    # Get all NPC character_ids (exclude __party__)
    rows = conn.execute("""
        SELECT character_id, character_name, COUNT(*) as cnt
        FROM knowledge
        WHERE character_id != '__party__'
        GROUP BY character_id
        ORDER BY character_name
    """).fetchall()

    npcs = [(r[0], r[1], r[2]) for r in rows]

    if on_progress:
        on_progress(f"Dedup: checking {len(npcs)} NPC names for duplicates...")

    # Build merge groups using union-find
    parent: dict[str, str] = {cid: cid for cid, _, _ in npcs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (id_a, name_a, _) in enumerate(npcs):
        for id_b, name_b, _ in npcs[i + 1:]:
            if find(id_a) == find(id_b):
                continue  # Already in same group
            if _names_match(name_a, name_b):
                union(id_a, id_b)

    # Collect groups
    groups: dict[str, list[tuple[str, str, int]]] = {}
    for cid, cname, cnt in npcs:
        root = find(cid)
        groups.setdefault(root, []).append((cid, cname, cnt))

    # Filter to groups with actual duplicates
    merge_groups = {k: v for k, v in groups.items() if len(v) > 1}

    total_merges = 0
    total_moved = 0
    group_details = []

    for _root, members in merge_groups.items():
        canonical_id, canonical_name = _pick_canonical_name(members)

        # Move entries from non-canonical to canonical
        moved = 0
        merged_from = []
        for cid, cname, cnt in members:
            if cid == canonical_id:
                continue
            cursor = conn.execute(
                "UPDATE knowledge SET character_id = ?, character_name = ? WHERE character_id = ?",
                (canonical_id, canonical_name, cid),
            )
            moved += cursor.rowcount
            merged_from.append(f"{cname} ({cnt})")

        if moved > 0:
            total_merges += 1
            total_moved += moved
            group_details.append({
                "canonical": canonical_name,
                "merged_from": merged_from,
                "entries_moved": moved,
            })

            if on_progress:
                on_progress(
                    f"  Merged {', '.join(merged_from)} → {canonical_name} ({moved} entries)"
                )

    conn.commit()

    if on_progress:
        on_progress(f"Dedup complete: {total_merges} merges, {total_moved} entries moved")

    return {
        "merges": total_merges,
        "entries_moved": total_moved,
        "groups": group_details,
    }


def resolve_ap_books(
    search: PathfinderSearch,
    ap_name: str,
) -> list[dict]:
    """Resolve an AP name to all its books (handles multi-book APs).

    "Curtain Call" → 3 books, "Kingmaker" → 1 book.
    Returns list of dicts with name, book_type, summary, chapters.
    """
    all_books = search.list_books_with_summaries()

    # Find books whose name contains the AP name (case-insensitive)
    ap_lower = ap_name.lower()
    matches = [
        b for b in all_books
        if ap_lower in b.get("book", "").lower()
    ]

    if not matches:
        # Fall back to resolve_book_name for single match
        exact = search.resolve_book_name(ap_name)
        if exact:
            summary = search.get_book_summary(exact)
            matches = [{"book": exact, **(summary or {})}]

    # Enrich with chapter data
    result = []
    for book_info in matches:
        book_name = book_info["book"]
        chapters = search.list_chapters(book_name)
        chapter_summaries = []
        for ch in chapters:
            ch_summary = search.get_chapter_summary(book_name, ch["chapter"])
            if ch_summary:
                chapter_summaries.append({
                    "chapter": ch["chapter"],
                    "page_range": f"{ch['page_start']}-{ch['page_end']}",
                    "summary": ch_summary.get("summary", "")[:500],
                })

        result.append({
            "name": book_name,
            "book_type": book_info.get("book_type", "adventure"),
            "total_pages": book_info.get("total_pages", 0),
            "summary": book_info.get("summary", ""),
            "chapters": chapter_summaries,
        })

    return result


def generate_background(
    search: PathfinderSearch,
    llm: LLMBackend,
    ap_name: str,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Generate a campaign background and search terms from AP book summaries.

    Handles single-book APs (Kingmaker) and multi-book APs (Curtain Call → 3 books).

    Args:
        search: PathfinderSearch instance.
        llm: LLM backend for generation.
        ap_name: Adventure path name (e.g., "Kingmaker", "Curtain Call").
        on_progress: Optional callback for progress messages.

    Returns:
        Dict with "background" and "search_terms" keys.
        Returns empty dict if no books found.
    """
    # Resolve all books for this AP
    books = resolve_ap_books(search, ap_name)

    if not books:
        logger.warning("No books found for AP: %s", ap_name)
        return {}

    if on_progress:
        on_progress(f"Generating background from {len(books)} book(s):")
        for b in books:
            on_progress(f"  {b['name']} ({b['total_pages']} pages)")

    # Build book summaries text
    parts = []
    for b in books:
        part = f"## {b['name']}\n"
        part += f"Pages: {b['total_pages']}\n"
        if b["summary"]:
            # Truncate long summaries to fit context
            part += f"\nBook Summary:\n{b['summary'][:1500]}\n"
        if b["chapters"]:
            part += "\nChapters:\n"
            for ch in b["chapters"]:
                part += f"- {ch['chapter']} (pp.{ch['page_range']})\n"
                if ch["summary"]:
                    part += f"  {ch['summary'][:300]}\n"
        parts.append(part)

    book_summaries_text = "\n\n".join(parts)

    user_prompt = GENERATE_BACKGROUND_USER.format(
        ap_name=ap_name,
        book_count=len(books),
        book_summaries=book_summaries_text,
    )

    if on_progress:
        on_progress("Calling LLM to generate background...")

    response = _call_llm(llm, GENERATE_BACKGROUND_SYSTEM, user_prompt)

    # Parse JSON response
    text = response.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            bg = result.get("background", "")
            terms = result.get("search_terms", [])
            if on_progress:
                on_progress(f"Generated background ({len(bg)} chars) with {len(terms)} search terms")
            return {"background": bg, "search_terms": terms}
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return {
                    "background": result.get("background", ""),
                    "search_terms": result.get("search_terms", []),
                }
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse generate_background response")
    return {"background": response.text, "search_terms": []}
