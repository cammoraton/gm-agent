"""Extract fiction tree data into campaign knowledge entries.

Converts fiction tree nodes from generation games (Microscope, Ex Novo,
Delve, Ex Umbra) into KnowledgeStore entries for the GM agent to use
during play. Pure data transforms — no LLM required. Idempotent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.fiction_tree import FictionTreeStore
    from ..storage.knowledge import KnowledgeStore

logger = logging.getLogger(__name__)

PARTY_ID = "__party__"
PARTY_NAME = "World"


def _add_if_new(
    knowledge_store: KnowledgeStore,
    content: str,
    knowledge_type: str = "fact",
    sharing_condition: str = "free",
    source: str = "",
    importance: int = 5,
    tags: list[str] | None = None,
) -> bool:
    """Add knowledge if no similar entry already exists. Returns True if added."""
    if not content or not content.strip():
        return False
    if knowledge_store.has_similar_knowledge(PARTY_ID, content):
        return False
    knowledge_store.add_knowledge(
        character_id=PARTY_ID,
        character_name=PARTY_NAME,
        content=content,
        knowledge_type=knowledge_type,
        sharing_condition=sharing_condition,
        source=source,
        importance=importance,
        tags=tags or [],
    )
    return True


def extract_from_microscope(
    fiction_store: FictionTreeStore,
    knowledge_store: KnowledgeStore,
    root_id: str,
) -> int:
    """Extract knowledge from a Microscope timeline.

    Returns the number of knowledge entries added.
    """
    root = fiction_store.get_node(root_id)
    if root is None:
        logger.warning("Root node %s not found", root_id)
        return 0

    source = f"microscope:{root.title}"
    count = 0

    # Root big picture / summary -> party knowledge
    if root.summary or root.content:
        text = root.summary or root.content
        if _add_if_new(knowledge_store, text, knowledge_type="fact",
                       source=source, importance=8, tags=["microscope", "big_picture"]):
            count += 1

    # Palette yes/no -> world constraints
    for item in root.palette_yes:
        if _add_if_new(knowledge_store, f"In this world: {item}",
                       knowledge_type="fact", source=source, importance=5,
                       tags=["microscope", "palette", "yes"]):
            count += 1

    for item in root.palette_no:
        if _add_if_new(knowledge_store, f"NOT in this world: {item}",
                       knowledge_type="fact", source=source, importance=5,
                       tags=["microscope", "palette", "no"]):
            count += 1

    # Periods -> era knowledge
    periods = fiction_store.get_children(root_id)
    for period in periods:
        period_text = period.title
        if period.summary:
            period_text = f"{period.title}: {period.summary}"
        tone_note = f" (tone: {period.tone})" if period.tone else ""
        if _add_if_new(knowledge_store, f"Era — {period_text}{tone_note}",
                       knowledge_type="fact", source=source, importance=6,
                       tags=["microscope", "period"]):
            count += 1

        # Events
        events = fiction_store.get_children(period.id)
        for event in events:
            event_text = event.title
            if event.summary:
                event_text = f"{event.title}: {event.summary}"
            if _add_if_new(knowledge_store, f"Historical event — {event_text}",
                           knowledge_type="fact", source=source, importance=5,
                           tags=["microscope", "event"]):
                count += 1

            # Scenes
            scenes = fiction_store.get_children(event.id)
            for scene in scenes:
                scene_text = scene.title
                if scene.summary:
                    scene_text = f"{scene.title}: {scene.summary}"
                if _add_if_new(knowledge_store, scene_text,
                               knowledge_type="fact", source=source, importance=4,
                               tags=["microscope", "scene"]):
                    count += 1

    return count


def extract_from_settlement(
    fiction_store: FictionTreeStore,
    knowledge_store: KnowledgeStore,
    root_id: str,
) -> int:
    """Extract knowledge from an Ex Novo settlement.

    Returns the number of knowledge entries added.
    """
    root = fiction_store.get_node(root_id)
    if root is None:
        logger.warning("Root node %s not found", root_id)
        return 0

    source = f"ex_novo:{root.title}"
    count = 0

    # Settlement root -> location knowledge
    root_text = root.title
    if root.summary:
        root_text = f"{root.title}: {root.summary}"
    if _add_if_new(knowledge_store, f"Settlement — {root_text}",
                   knowledge_type="fact", source=source, importance=7,
                   tags=["ex_novo", "settlement"]):
        count += 1

    # Children: factions, districts, landmarks, resources, problems
    children = fiction_store.get_children(root_id)
    for child in children:
        title_lower = child.title.lower()
        child_text = child.title
        if child.summary:
            child_text = f"{child.title}: {child.summary}"

        # Detect factions by keywords in title/tags
        if any(kw in title_lower for kw in ("guild", "faction", "order", "clan", "cult", "gang")):
            if _add_if_new(knowledge_store, f"Organization — {child_text}",
                           knowledge_type="fact", source=source, importance=6,
                           tags=["ex_novo", "faction"]):
                count += 1
        elif any(kw in title_lower for kw in ("district", "quarter", "ward", "neighborhood")):
            if _add_if_new(knowledge_store, f"District — {child_text}",
                           knowledge_type="fact", source=source, importance=5,
                           tags=["ex_novo", "district"]):
                count += 1
        elif any(kw in title_lower for kw in ("resource", "trade", "export", "mine")):
            if _add_if_new(knowledge_store, f"Resource — {child_text}",
                           knowledge_type="fact", source=source, importance=4,
                           tags=["ex_novo", "resource"]):
                count += 1
        elif any(kw in title_lower for kw in ("problem", "threat", "danger", "conflict")):
            if _add_if_new(knowledge_store, f"Problem — {child_text}",
                           knowledge_type="fact", source=source, importance=5,
                           tags=["ex_novo", "problem"]):
                count += 1
        else:
            # Default: landmark/location
            if _add_if_new(knowledge_store, f"Landmark — {child_text}",
                           knowledge_type="fact", source=source, importance=4,
                           tags=["ex_novo", "landmark"]):
                count += 1

    return count


def extract_from_dungeon(
    fiction_store: FictionTreeStore,
    knowledge_store: KnowledgeStore,
    root_id: str,
) -> int:
    """Extract knowledge from a Delve/Ex Umbra dungeon.

    Returns the number of knowledge entries added.
    """
    root = fiction_store.get_node(root_id)
    if root is None:
        logger.warning("Root node %s not found", root_id)
        return 0

    system = root.source_system or "delve"
    source = f"{system}:{root.title}"
    count = 0

    # Dungeon root -> location knowledge
    root_text = root.title
    if root.summary:
        root_text = f"{root.title}: {root.summary}"
    if _add_if_new(knowledge_store, f"Dungeon — {root_text}",
                   knowledge_type="fact", source=source, importance=6,
                   tags=[system, "dungeon"]):
        count += 1

    # Rooms
    rooms = fiction_store.get_children(root_id)
    for room in rooms:
        room_text = room.title
        if room.summary:
            room_text = f"{room.title}: {room.summary}"
        if _add_if_new(knowledge_store, f"Room — {room_text}",
                       knowledge_type="fact", source=source, importance=4,
                       tags=[system, "room"]):
            count += 1

        # Room details
        details = fiction_store.get_children(room.id)
        for detail in details:
            detail_text = detail.title
            if detail.summary:
                detail_text = f"{detail.title}: {detail.summary}"
            if _add_if_new(knowledge_store, detail_text,
                           knowledge_type="fact", source=source, importance=3,
                           tags=[system, "detail"]):
                count += 1

    return count


def extract_fiction_knowledge(
    fiction_store: FictionTreeStore,
    knowledge_store: KnowledgeStore,
    root_id: str,
) -> int:
    """Auto-detect source system and dispatch to the right extractor.

    Returns the number of knowledge entries added.
    """
    root = fiction_store.get_node(root_id)
    if root is None:
        logger.warning("Root node %s not found", root_id)
        return 0

    system = (root.source_system or "").lower()

    if system == "microscope":
        return extract_from_microscope(fiction_store, knowledge_store, root_id)
    elif system == "ex_novo":
        return extract_from_settlement(fiction_store, knowledge_store, root_id)
    elif system in ("delve", "ex_umbra"):
        return extract_from_dungeon(fiction_store, knowledge_store, root_id)
    else:
        logger.warning("Unknown source system '%s' for node %s", system, root_id)
        return 0
