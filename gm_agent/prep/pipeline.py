"""PrepPipeline orchestrator for campaign knowledge initialization."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..models.base import LLMBackend
from ..rag.search import PathfinderSearch
from ..storage.knowledge import KnowledgeStore
from .knowledge import (
    deduplicate_npc_names,
    detect_subsystems,
    generate_background,
    resolve_ap_books,
    seed_npc_knowledge,
    seed_party_knowledge,
    seed_subsystem_knowledge,
    seed_world_context,
    seed_world_context_by_query,
)
from .log import PrepLogger

logger = logging.getLogger(__name__)


@dataclass
class PrepResult:
    """Result summary from a prep pipeline run."""

    campaign_id: str
    books_resolved: list[dict] = field(default_factory=list)
    party_knowledge_count: int = 0
    npc_knowledge_count: int = 0
    subsystem_knowledge_count: int = 0
    world_context_count: int = 0
    duration_ms: float = 0.0
    party_duration_ms: float = 0.0
    npc_duration_ms: float = 0.0
    subsystem_duration_ms: float = 0.0
    world_duration_ms: float = 0.0
    subsystems_detected: list[str] = field(default_factory=list)
    npc_dedup_merges: int = 0
    npc_dedup_entries_moved: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return (
            self.party_knowledge_count
            + self.npc_knowledge_count
            + self.subsystem_knowledge_count
            + self.world_context_count
        )

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


class PrepPipeline:
    """Orchestrates campaign knowledge initialization.

    Resolves books, runs synthesis steps (party, NPC, world), and
    collects results into KnowledgeStore with JSONL training logs.
    """

    def __init__(
        self,
        campaign_id: str,
        llm: LLMBackend,
        search: PathfinderSearch,
        knowledge_base_dir: Path | None = None,
        logger_base_dir: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ):
        self.campaign_id = campaign_id
        self.llm = llm
        self.search = search
        self.knowledge = KnowledgeStore(campaign_id, base_dir=knowledge_base_dir)
        self.logger = PrepLogger(campaign_id, base_dir=logger_base_dir)
        self.on_progress = on_progress

    def _aggregate_tokens(self, result: PrepResult) -> None:
        """Aggregate token counts from prep log entries."""
        entries = self.logger.read()
        prompt_total = 0
        completion_total = 0
        for entry in entries:
            usage = entry.token_usage or {}
            prompt_total += usage.get("prompt_tokens", 0)
            completion_total += usage.get("completion_tokens", 0)
        result.total_prompt_tokens = prompt_total
        result.total_completion_tokens = completion_total

    def run(
        self,
        books: list[str],
        search_terms: list[str] | None = None,
        campaign_background: str = "",
        skip_steps: list[str] | None = None,
    ) -> PrepResult:
        """Run full campaign prep pipeline.

        Args:
            books: List of book names (fuzzy-matched against search.db).
            search_terms: Optional list of search terms for query-based world context.
            campaign_background: Optional campaign background text for world context.
            skip_steps: Optional list of steps to skip ("party", "npc", "subsystem", "world").

        Returns:
            PrepResult with counts and timing.
        """
        start = time.monotonic()
        result = PrepResult(campaign_id=self.campaign_id)
        skip = set(skip_steps or [])

        # Resolve books
        try:
            resolved = self.resolve_books(books)
            result.books_resolved = resolved
        except Exception as e:
            result.errors.append(f"Book resolution failed: {e}")
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        if self.on_progress:
            for b in resolved:
                self.on_progress(f"  Book: {b['name']} ({b['book_type']}, {b['entity_count']} entities)")

        # Split by book type
        pgs = [b for b in resolved if b["book_type"] == "players_guide"]
        aps = [b for b in resolved if b["book_type"] == "adventure"]
        settings = [b for b in resolved if b["book_type"] == "setting"]

        # Also feed PGs into party knowledge (they're the primary source)
        party_books = pgs + settings

        # Step 1: Party knowledge
        if party_books and "party" not in skip:
            if self.on_progress:
                self.on_progress("\n--- Party Knowledge ---")
            try:
                step_start = time.monotonic()
                result.party_knowledge_count = seed_party_knowledge(
                    self.search,
                    self.knowledge,
                    self.llm,
                    self.logger,
                    party_books,
                    self.campaign_id,
                    on_progress=self.on_progress,
                )
                result.party_duration_ms = (time.monotonic() - step_start) * 1000
            except Exception as e:
                logger.error("Party knowledge seeding failed: %s", e)
                result.errors.append(f"Party knowledge failed: {e}")

        # Step 2: NPC knowledge
        if aps and "npc" not in skip:
            if self.on_progress:
                self.on_progress("\n--- NPC Knowledge ---")
            try:
                step_start = time.monotonic()
                result.npc_knowledge_count = seed_npc_knowledge(
                    self.search,
                    self.knowledge,
                    self.llm,
                    self.logger,
                    aps,
                    self.campaign_id,
                    on_progress=self.on_progress,
                )
                result.npc_duration_ms = (time.monotonic() - step_start) * 1000
            except Exception as e:
                logger.error("NPC knowledge seeding failed: %s", e)
                result.errors.append(f"NPC knowledge failed: {e}")

        # Step 2b: NPC name deduplication (always runs if we have NPC knowledge)
        if "dedup" not in skip:
            try:
                dedup_result = deduplicate_npc_names(
                    self.knowledge,
                    on_progress=self.on_progress,
                )
                if dedup_result["merges"]:
                    result.npc_dedup_merges = dedup_result["merges"]
                    result.npc_dedup_entries_moved = dedup_result["entries_moved"]
            except Exception as e:
                logger.error("NPC name dedup failed: %s", e)
                result.errors.append(f"NPC dedup failed: {e}")

        # Step 2c: Subsystem knowledge
        if aps and "subsystem" not in skip:
            if self.on_progress:
                self.on_progress("\n--- Subsystem Knowledge ---")
            try:
                step_start = time.monotonic()
                detected = detect_subsystems(
                    self.search, resolved, on_progress=self.on_progress,
                )
                result.subsystems_detected = detected
                if detected:
                    result.subsystem_knowledge_count = seed_subsystem_knowledge(
                        self.search,
                        self.knowledge,
                        self.llm,
                        self.logger,
                        aps,
                        detected,
                        self.campaign_id,
                        on_progress=self.on_progress,
                    )
                elif self.on_progress:
                    self.on_progress("No subsystems detected in AP books")
                result.subsystem_duration_ms = (time.monotonic() - step_start) * 1000
            except Exception as e:
                logger.error("Subsystem knowledge seeding failed: %s", e)
                result.errors.append(f"Subsystem knowledge failed: {e}")

        # Step 3: World context
        if "world" not in skip:
            # Use query-based approach if search terms provided
            if search_terms:
                if self.on_progress:
                    self.on_progress("\n--- World Context (query-based) ---")
                try:
                    step_start = time.monotonic()
                    result.world_context_count = seed_world_context_by_query(
                        self.search,
                        self.knowledge,
                        self.llm,
                        self.logger,
                        search_terms,
                        self.campaign_id,
                        campaign_background=campaign_background,
                        on_progress=self.on_progress,
                    )
                    result.world_duration_ms = (time.monotonic() - step_start) * 1000
                except Exception as e:
                    logger.error("World context (query) seeding failed: %s", e)
                    result.errors.append(f"World context (query) failed: {e}")
            elif settings:
                if self.on_progress:
                    self.on_progress("\n--- World Context (book-based) ---")
                try:
                    step_start = time.monotonic()
                    result.world_context_count = seed_world_context(
                        self.search,
                        self.knowledge,
                        self.llm,
                        self.logger,
                        settings,
                        self.campaign_id,
                        on_progress=self.on_progress,
                    )
                    result.world_duration_ms = (time.monotonic() - step_start) * 1000
                except Exception as e:
                    logger.error("World context seeding failed: %s", e)
                    result.errors.append(f"World context failed: {e}")

        result.duration_ms = (time.monotonic() - start) * 1000

        # Aggregate token counts from log entries
        self._aggregate_tokens(result)

        self.knowledge.close()
        return result

    def resolve_books(self, books: list[str]) -> list[dict]:
        """Resolve book names to exact DB names with book_type.

        Also auto-discovers Players Guides: for each AP book, checks
        if a matching Players Guide exists.
        """
        resolved = []
        seen = set()

        for name in books:
            exact_name = self.search.resolve_book_name(name)
            if not exact_name:
                logger.warning("Could not resolve book: %s", name)
                continue

            if exact_name in seen:
                continue
            seen.add(exact_name)

            summary = self.search.get_book_summary(exact_name)
            book_type = summary.get("book_type", "unknown") if summary else "unknown"
            # Books with "Players Guide" in the name but no summary
            if book_type == "unknown" and "players guide" in exact_name.lower():
                book_type = "players_guide"
            entity_count = 0

            # Count entities in this book
            try:
                entities = self.search.list_entities(book=exact_name, limit=10000)
                entity_count = len(entities)
            except Exception:
                pass

            resolved.append({
                "name": exact_name,
                "book_type": book_type,
                "entity_count": entity_count,
            })

            # Auto-discover Players Guide for AP books
            if book_type == "adventure":
                pg_name = self.search.resolve_book_name(f"{name} Players Guide")
                if pg_name and pg_name not in seen:
                    seen.add(pg_name)
                    pg_summary = self.search.get_book_summary(pg_name)
                    pg_type = pg_summary.get("book_type", "players_guide") if pg_summary else "players_guide"
                    pg_count = 0
                    try:
                        pg_entities = self.search.list_entities(book=pg_name, limit=10000)
                        pg_count = len(pg_entities)
                    except Exception:
                        pass
                    resolved.append({
                        "name": pg_name,
                        "book_type": pg_type,
                        "entity_count": pg_count,
                    })

        return resolved
