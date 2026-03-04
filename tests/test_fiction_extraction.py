"""Tests for fiction_extraction.py — extract fiction tree data to knowledge."""

from pathlib import Path

import pytest

from gm_agent.prep.fiction_extraction import (
    extract_from_microscope,
    extract_from_settlement,
    extract_from_dungeon,
    extract_fiction_knowledge,
    PARTY_ID,
)
from gm_agent.storage.fiction_tree import FictionTreeStore, FictionNode
from gm_agent.storage.knowledge import KnowledgeStore


@pytest.fixture
def campaigns_dir(tmp_path: Path) -> Path:
    d = tmp_path / "campaigns"
    d.mkdir()
    (d / "extract-test").mkdir()
    return d


@pytest.fixture
def fiction_store(campaigns_dir: Path) -> FictionTreeStore:
    s = FictionTreeStore("extract-test", base_dir=campaigns_dir)
    yield s
    s.close()


@pytest.fixture
def knowledge_store(campaigns_dir: Path) -> KnowledgeStore:
    s = KnowledgeStore("extract-test", base_dir=campaigns_dir)
    yield s
    s.close()


# ===========================================================================
# Microscope extraction
# ===========================================================================


def _create_microscope(fiction_store: FictionTreeStore) -> str:
    root = fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        title="The Age of Heroes",
        summary="A sweeping history of heroes and villains.",
        zoom_level="era",
        source_system="microscope",
        palette_yes=["dragons", "magic"],
        palette_no=["firearms"],
    ))
    p1 = fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="The Rise of the Kingdom",
        summary="A golden age begins.",
        zoom_level="period",
        tone="light",
        source_system="microscope",
    ))
    e1 = fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=p1.id,
        title="The Battle of Dragon Peak",
        summary="Heroes defeat a dragon at the mountain.",
        zoom_level="event",
        tone="dark",
        source_system="microscope",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=e1.id,
        title="The Dragon Falls",
        summary="The final blow.",
        zoom_level="scene",
        source_system="microscope",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=p1.id,
        title="The Discovery of the Ancient Tomb",
        zoom_level="event",
        source_system="microscope",
    ))
    return root.id


class TestMicroscopeExtraction:
    def test_basic_extraction(self, fiction_store, knowledge_store):
        root_id = _create_microscope(fiction_store)
        count = extract_from_microscope(fiction_store, knowledge_store, root_id)
        assert count >= 5  # root + 2 palette_yes + 1 palette_no + period + events + scene

    def test_palette_extracted(self, fiction_store, knowledge_store):
        root_id = _create_microscope(fiction_store)
        extract_from_microscope(fiction_store, knowledge_store, root_id)

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["palette"])
        assert len(entries) == 3  # 2 yes + 1 no
        contents = [e.content for e in entries]
        assert any("dragons" in c for c in contents)
        assert any("NOT" in c and "firearms" in c for c in contents)

    def test_periods_extracted(self, fiction_store, knowledge_store):
        root_id = _create_microscope(fiction_store)
        extract_from_microscope(fiction_store, knowledge_store, root_id)

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["period"])
        assert len(entries) >= 1
        assert any("Rise of the Kingdom" in e.content for e in entries)

    def test_events_extracted(self, fiction_store, knowledge_store):
        root_id = _create_microscope(fiction_store)
        extract_from_microscope(fiction_store, knowledge_store, root_id)

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["event"])
        assert len(entries) >= 2

    def test_idempotent(self, fiction_store, knowledge_store):
        root_id = _create_microscope(fiction_store)
        count1 = extract_from_microscope(fiction_store, knowledge_store, root_id)
        count2 = extract_from_microscope(fiction_store, knowledge_store, root_id)
        assert count1 > 0
        assert count2 == 0

    def test_missing_root(self, fiction_store, knowledge_store):
        count = extract_from_microscope(fiction_store, knowledge_store, "nonexistent")
        assert count == 0


# ===========================================================================
# Settlement extraction
# ===========================================================================


def _create_settlement(fiction_store: FictionTreeStore) -> str:
    root = fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        title="Riverside",
        summary="A bustling riverside town.",
        zoom_level="era",
        source_system="ex_novo",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="Market District",
        summary="The commercial heart of town.",
        zoom_level="detail",
        source_system="ex_novo",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="The Thieves Guild",
        summary="A shadowy faction controlling the docks.",
        zoom_level="detail",
        source_system="ex_novo",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="Iron Mine Resource",
        summary="Rich iron deposits.",
        zoom_level="detail",
        source_system="ex_novo",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="Orc Threat Problem",
        summary="Orc raiders threaten trade routes.",
        zoom_level="detail",
        source_system="ex_novo",
    ))
    return root.id


class TestSettlementExtraction:
    def test_basic_extraction(self, fiction_store, knowledge_store):
        root_id = _create_settlement(fiction_store)
        count = extract_from_settlement(fiction_store, knowledge_store, root_id)
        assert count >= 5  # root + district + faction + resource + problem

    def test_faction_detected(self, fiction_store, knowledge_store):
        root_id = _create_settlement(fiction_store)
        extract_from_settlement(fiction_store, knowledge_store, root_id)

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["faction"])
        assert len(entries) >= 1
        assert any("Thieves Guild" in e.content for e in entries)

    def test_district_detected(self, fiction_store, knowledge_store):
        root_id = _create_settlement(fiction_store)
        extract_from_settlement(fiction_store, knowledge_store, root_id)

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["district"])
        assert len(entries) >= 1

    def test_idempotent(self, fiction_store, knowledge_store):
        root_id = _create_settlement(fiction_store)
        count1 = extract_from_settlement(fiction_store, knowledge_store, root_id)
        count2 = extract_from_settlement(fiction_store, knowledge_store, root_id)
        assert count1 > 0
        assert count2 == 0


# ===========================================================================
# Dungeon extraction
# ===========================================================================


def _create_dungeon(fiction_store: FictionTreeStore, system: str = "delve") -> str:
    root = fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        title="The Forgotten Mines",
        summary="An abandoned dwarven mine.",
        zoom_level="era",
        source_system=system,
    ))
    r1 = fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="Collapsed Entrance",
        summary="A trap-filled entrance.",
        zoom_level="detail",
        source_system=system,
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=r1.id,
        title="Hidden mechanism",
        summary="A lever behind loose stones.",
        zoom_level="detail",
        source_system=system,
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="extract-test",
        parent_id=root.id,
        title="The Forge",
        zoom_level="detail",
        source_system=system,
    ))
    return root.id


class TestDungeonExtraction:
    def test_basic_extraction(self, fiction_store, knowledge_store):
        root_id = _create_dungeon(fiction_store)
        count = extract_from_dungeon(fiction_store, knowledge_store, root_id)
        assert count >= 4  # root + 2 rooms + 1 detail

    def test_rooms_tagged(self, fiction_store, knowledge_store):
        root_id = _create_dungeon(fiction_store)
        extract_from_dungeon(fiction_store, knowledge_store, root_id)

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["room"])
        assert len(entries) >= 2

    def test_ex_umbra_system(self, fiction_store, knowledge_store):
        root_id = _create_dungeon(fiction_store, system="ex_umbra")
        count = extract_from_dungeon(fiction_store, knowledge_store, root_id)
        assert count >= 1

        entries = knowledge_store.query_knowledge(character_id=PARTY_ID, tags=["ex_umbra"])
        assert len(entries) >= 1

    def test_idempotent(self, fiction_store, knowledge_store):
        root_id = _create_dungeon(fiction_store)
        count1 = extract_from_dungeon(fiction_store, knowledge_store, root_id)
        count2 = extract_from_dungeon(fiction_store, knowledge_store, root_id)
        assert count1 > 0
        assert count2 == 0


# ===========================================================================
# Auto-dispatch
# ===========================================================================


class TestExtractFictionKnowledge:
    def test_dispatch_microscope(self, fiction_store, knowledge_store):
        root_id = _create_microscope(fiction_store)
        count = extract_fiction_knowledge(fiction_store, knowledge_store, root_id)
        assert count > 0

    def test_dispatch_ex_novo(self, fiction_store, knowledge_store):
        root_id = _create_settlement(fiction_store)
        count = extract_fiction_knowledge(fiction_store, knowledge_store, root_id)
        assert count > 0

    def test_dispatch_delve(self, fiction_store, knowledge_store):
        root_id = _create_dungeon(fiction_store, system="delve")
        count = extract_fiction_knowledge(fiction_store, knowledge_store, root_id)
        assert count > 0

    def test_dispatch_ex_umbra(self, fiction_store, knowledge_store):
        root_id = _create_dungeon(fiction_store, system="ex_umbra")
        count = extract_fiction_knowledge(fiction_store, knowledge_store, root_id)
        assert count > 0

    def test_unknown_system(self, fiction_store, knowledge_store):
        root = fiction_store.add_node(FictionNode(
            campaign_id="extract-test",
            title="Unknown",
            zoom_level="era",
            source_system="unknown_game",
        ))
        count = extract_fiction_knowledge(fiction_store, knowledge_store, root.id)
        assert count == 0

    def test_missing_root(self, fiction_store, knowledge_store):
        count = extract_fiction_knowledge(fiction_store, knowledge_store, "nonexistent")
        assert count == 0
