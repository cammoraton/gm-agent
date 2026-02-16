"""CrunchPipeline orchestrator for session post-processing."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models.base import LLMBackend
from ..storage.campaign import CampaignStore
from ..storage.dialogue import DialogueStore
from ..storage.history import HistoryIndex
from ..storage.knowledge import KnowledgeStore
from ..storage.schemas import Session
from .log import PrepLogger
from .session import extract_dialogue, extract_events, update_arc, update_knowledge

logger = logging.getLogger(__name__)

ALL_STEPS = ("events", "dialogue", "knowledge", "arc")


@dataclass
class CrunchResult:
    """Result summary from a crunch pipeline run."""

    campaign_id: str
    session_id: str
    events_count: int = 0
    dialogue_count: int = 0
    knowledge_count: int = 0
    arc_updated: bool = False
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return self.events_count + self.dialogue_count + self.knowledge_count


class CrunchPipeline:
    """Orchestrates session post-processing.

    Runs event extraction, dialogue extraction, knowledge updates, and
    arc progress on a completed session. Results are written to the
    campaign's HistoryIndex, DialogueStore, KnowledgeStore, and
    CampaignStore respectively.
    """

    def __init__(
        self,
        campaign_id: str,
        llm: LLMBackend,
        knowledge_base_dir: Path | None = None,
        logger_base_dir: Path | None = None,
        campaigns_dir: Path | None = None,
    ):
        self.campaign_id = campaign_id
        self.llm = llm
        self.history = HistoryIndex(campaign_id, base_dir=knowledge_base_dir)
        self.dialogue = DialogueStore(campaign_id, base_dir=knowledge_base_dir)
        self.knowledge = KnowledgeStore(campaign_id, base_dir=knowledge_base_dir)
        self.campaign_store = CampaignStore(base_dir=campaigns_dir)
        self.logger = PrepLogger(campaign_id, base_dir=logger_base_dir)

    def run(
        self,
        session: Session,
        steps: list[str] | None = None,
        skip_steps: list[str] | None = None,
    ) -> CrunchResult:
        """Run session post-processing.

        Args:
            session: The completed session to process.
            steps: Specific steps to run (default: all). Options:
                "events", "dialogue", "knowledge", "arc"
            skip_steps: Steps to skip (applied after steps filter).

        Returns:
            CrunchResult with counts and timing.
        """
        start = time.monotonic()
        result = CrunchResult(campaign_id=self.campaign_id, session_id=session.id)

        # Determine which steps to run
        active_steps = set(steps) if steps else set(ALL_STEPS)
        if skip_steps:
            active_steps -= set(skip_steps)

        # Idempotent: delete existing data for this session
        if "events" in active_steps:
            self.history.delete_session_events(session.id)
        if "dialogue" in active_steps:
            self.dialogue.delete_session_dialogues(session.id)

        # Step 1: Extract events
        events_summary = ""
        if "events" in active_steps:
            try:
                result.events_count = extract_events(
                    session, self.history, self.llm, self.logger, self.campaign_id
                )
                # Build events summary for arc update
                events = self.history.get_session_events(session.id)
                events_summary = "\n".join(
                    f"- [{e.importance}] {e.event}" for e in events
                )
            except Exception as e:
                logger.error("Event extraction failed: %s", e)
                result.errors.append(f"Event extraction failed: {e}")

        # Step 2: Extract dialogue
        if "dialogue" in active_steps:
            try:
                result.dialogue_count = extract_dialogue(
                    session, self.dialogue, self.llm, self.logger, self.campaign_id
                )
            except Exception as e:
                logger.error("Dialogue extraction failed: %s", e)
                result.errors.append(f"Dialogue extraction failed: {e}")

        # Step 3: Update knowledge
        if "knowledge" in active_steps:
            try:
                result.knowledge_count = update_knowledge(
                    session, self.knowledge, self.llm, self.logger, self.campaign_id
                )
            except Exception as e:
                logger.error("Knowledge update failed: %s", e)
                result.errors.append(f"Knowledge update failed: {e}")

        # Step 4: Update arc
        if "arc" in active_steps:
            try:
                campaign = self.campaign_store.get(self.campaign_id)
                if campaign:
                    update_arc(
                        session,
                        campaign,
                        self.campaign_store,
                        self.llm,
                        self.logger,
                        events_summary,
                    )
                    result.arc_updated = True
                else:
                    result.errors.append(
                        f"Campaign '{self.campaign_id}' not found for arc update"
                    )
            except Exception as e:
                logger.error("Arc update failed: %s", e)
                result.errors.append(f"Arc update failed: {e}")

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    def close(self):
        """Close all store connections."""
        self.history.close()
        self.dialogue.close()
        self.knowledge.close()
