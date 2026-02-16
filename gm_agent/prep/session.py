"""Session post-processing synthesis functions.

Each function follows the same pattern:
1. Format session transcript for LLM
2. LLM synthesis with appropriate prompts
3. Parse structured output
4. Write to target store
5. Log step via PrepLogger
"""

import logging
import re
import time
from datetime import datetime

from ..models.base import LLMBackend, LLMResponse, Message
from ..storage.campaign import CampaignStore
from ..storage.dialogue import DialogueStore
from ..storage.history import HistoryIndex
from ..storage.knowledge import KnowledgeStore
from ..storage.schemas import Campaign, Session
from .knowledge import _call_llm, _parse_json_array
from .log import PrepLogEntry, PrepLogger
from .session_prompts import (
    ARC_PROGRESS_SYSTEM,
    ARC_PROGRESS_USER,
    DIALOGUE_EXTRACTION_SYSTEM,
    DIALOGUE_EXTRACTION_USER,
    EVENT_EXTRACTION_SYSTEM,
    EVENT_EXTRACTION_USER,
    KNOWLEDGE_UPDATE_SYSTEM,
    KNOWLEDGE_UPDATE_USER,
)

logger = logging.getLogger(__name__)

# Party knowledge character ID (matches prep/knowledge.py)
PARTY_KNOWLEDGE_ID = "__party__"


def _format_transcript(session: Session) -> str:
    """Format session turns into a readable transcript for LLM consumption."""
    if not session.turns:
        return "(Empty session — no turns recorded)"

    parts = []
    for i, turn in enumerate(session.turns, 1):
        parts.append(f"[Turn {i}]")
        parts.append(f"Player: {turn.player_input}")
        parts.append(f"GM: {turn.gm_response}")
        parts.append("")

    return "\n".join(parts)


def _slugify_name(name: str) -> str:
    """Convert a character name to a slug ID."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def extract_events(
    session: Session,
    history: HistoryIndex,
    llm: LLMBackend,
    logger_: PrepLogger,
    campaign_id: str,
) -> int:
    """Extract events from session transcript into HistoryIndex.

    Returns count of events logged.
    """
    transcript = _format_transcript(session)
    if not session.turns:
        return 0

    user_prompt = EVENT_EXTRACTION_USER.format(transcript=transcript)

    start = time.monotonic()
    response = _call_llm(llm, EVENT_EXTRACTION_SYSTEM, user_prompt)
    duration_ms = (time.monotonic() - start) * 1000

    entries = _parse_json_array(response.text)
    total = 0
    output_entries = []

    for entry in entries:
        event_text = entry.get("event", "")
        if not event_text:
            continue

        importance = entry.get("importance", "session")
        if importance not in ("session", "arc", "campaign"):
            importance = "session"

        tags = entry.get("tags", [])

        history.log_event(
            session_id=session.id,
            event=event_text,
            importance=importance,
            tags=tags,
        )
        output_entries.append(entry)
        total += 1

    logger_.log(PrepLogEntry(
        step="event_extraction",
        campaign_id=campaign_id,
        book="",
        entity=None,
        input_context=transcript[:4000],
        system_prompt=EVENT_EXTRACTION_SYSTEM,
        thinking=response.thinking,
        output=output_entries,
        model=llm.get_model_name(),
        duration_ms=duration_ms,
        token_usage=response.usage,
    ))

    logger.info("Extracted %d events from session %s", total, session.id)
    return total


def extract_dialogue(
    session: Session,
    dialogue_store: DialogueStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    campaign_id: str,
) -> int:
    """Extract notable NPC dialogue into DialogueStore.

    Returns count of dialogue entries logged.
    """
    transcript = _format_transcript(session)
    if not session.turns:
        return 0

    user_prompt = DIALOGUE_EXTRACTION_USER.format(transcript=transcript)

    start = time.monotonic()
    response = _call_llm(llm, DIALOGUE_EXTRACTION_SYSTEM, user_prompt)
    duration_ms = (time.monotonic() - start) * 1000

    entries = _parse_json_array(response.text)
    total = 0
    output_entries = []

    for entry in entries:
        content = entry.get("content", "")
        if not content:
            continue

        character_name = entry.get("character_name", "Unknown")
        character_id = entry.get("character_id", _slugify_name(character_name))
        dialogue_type = entry.get("dialogue_type", "statement")
        flagged = entry.get("flagged", False)

        dialogue_store.log_dialogue(
            character_id=character_id,
            character_name=character_name,
            session_id=session.id,
            content=content,
            dialogue_type=dialogue_type,
            flagged=bool(flagged),
        )
        output_entries.append(entry)
        total += 1

    logger_.log(PrepLogEntry(
        step="dialogue_extraction",
        campaign_id=campaign_id,
        book="",
        entity=None,
        input_context=transcript[:4000],
        system_prompt=DIALOGUE_EXTRACTION_SYSTEM,
        thinking=response.thinking,
        output=output_entries,
        model=llm.get_model_name(),
        duration_ms=duration_ms,
        token_usage=response.usage,
    ))

    logger.info("Extracted %d dialogue entries from session %s", total, session.id)
    return total


def update_knowledge(
    session: Session,
    knowledge: KnowledgeStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    campaign_id: str,
) -> int:
    """Update NPC/party knowledge based on session events.

    Returns count of knowledge entries added.
    """
    transcript = _format_transcript(session)
    if not session.turns:
        return 0

    # Get existing knowledge for context to avoid duplicates
    existing = knowledge.query_knowledge(limit=50)
    existing_text = "\n".join(
        f"- [{e.character_name}] {e.content}" for e in existing
    ) if existing else "(No existing knowledge)"

    user_prompt = KNOWLEDGE_UPDATE_USER.format(
        transcript=transcript,
        existing_knowledge=existing_text,
    )

    start = time.monotonic()
    response = _call_llm(llm, KNOWLEDGE_UPDATE_SYSTEM, user_prompt)
    duration_ms = (time.monotonic() - start) * 1000

    entries = _parse_json_array(response.text)
    total = 0
    output_entries = []

    for entry in entries:
        content = entry.get("content", "")
        if not content:
            continue

        character_name = entry.get("character_name", "Party")
        character_id = entry.get("character_id", PARTY_KNOWLEDGE_ID)
        knowledge_type = entry.get("knowledge_type", "fact")
        importance = entry.get("importance", 5)
        tags = entry.get("tags", [])
        tags.append(f"session:{session.id}")
        tags = list(dict.fromkeys(tags))

        ke = knowledge.add_knowledge(
            character_id=character_id,
            character_name=character_name,
            content=content,
            knowledge_type=knowledge_type,
            sharing_condition="free",
            source=f"session:{session.id}",
            importance=importance,
            decay_rate=0.0,
            tags=tags,
        )
        output_entries.append(ke.to_dict())
        total += 1

    logger_.log(PrepLogEntry(
        step="knowledge_update",
        campaign_id=campaign_id,
        book="",
        entity=None,
        input_context=transcript[:4000],
        system_prompt=KNOWLEDGE_UPDATE_SYSTEM,
        thinking=response.thinking,
        output=output_entries,
        model=llm.get_model_name(),
        duration_ms=duration_ms,
        token_usage=response.usage,
    ))

    logger.info("Added %d knowledge entries from session %s", total, session.id)
    return total


def update_arc(
    session: Session,
    campaign: Campaign,
    campaign_store: CampaignStore,
    llm: LLMBackend,
    logger_: PrepLogger,
    events_summary: str,
) -> str:
    """Update campaign.current_arc based on session.

    Returns the new arc text.
    """
    current_arc = campaign.current_arc or "(No arc established yet)"
    session_summary = session.summary or "(No session summary available)"

    user_prompt = ARC_PROGRESS_USER.format(
        current_arc=current_arc,
        session_summary=session_summary,
        events_summary=events_summary or "(No events extracted)",
    )

    start = time.monotonic()
    response = _call_llm(llm, ARC_PROGRESS_SYSTEM, user_prompt)
    duration_ms = (time.monotonic() - start) * 1000

    new_arc = response.text.strip()

    # Store arc version history before updating
    arc_history = campaign.preferences.get("arc_history", [])
    if campaign.current_arc:
        arc_history.append({
            "session_id": session.id,
            "arc_text": campaign.current_arc,
            "timestamp": datetime.now().isoformat(),
        })
    campaign.preferences["arc_history"] = arc_history

    # Update campaign
    campaign.current_arc = new_arc
    campaign_store.update(campaign)

    logger_.log(PrepLogEntry(
        step="arc_update",
        campaign_id=campaign.id,
        book="",
        entity=None,
        input_context=user_prompt[:4000],
        system_prompt=ARC_PROGRESS_SYSTEM,
        thinking=response.thinking,
        output=[{"arc_text": new_arc}],
        model=llm.get_model_name(),
        duration_ms=duration_ms,
        token_usage=response.usage,
    ))

    logger.info("Updated arc for campaign %s", campaign.id)
    return new_arc
