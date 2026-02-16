"""Tests for PropagationBus — cross-system knowledge propagation."""

import pytest

from gm_agent.propagation import PropagationBus
from gm_agent.storage.characters import CharacterStore
from gm_agent.storage.factions import FactionStore
from gm_agent.storage.knowledge import KnowledgeStore
from gm_agent.storage.locations import LocationStore
from gm_agent.storage.schemas import Secret


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge(tmp_path):
    return KnowledgeStore("test-campaign", base_dir=tmp_path)


@pytest.fixture
def factions(tmp_path):
    return FactionStore("test-campaign", base_dir=tmp_path)


@pytest.fixture
def locations(tmp_path):
    return LocationStore("test-campaign", base_dir=tmp_path)


@pytest.fixture
def characters(tmp_path):
    return CharacterStore("test-campaign", base_dir=tmp_path)


@pytest.fixture
def bus(knowledge, factions, locations, characters):
    return PropagationBus(
        knowledge=knowledge,
        factions=factions,
        locations=locations,
        characters=characters,
    )


def _make_secret(**kwargs) -> Secret:
    defaults = dict(
        id="secret-1",
        campaign_id="test-campaign",
        content="The mayor is secretly a vampire.",
        importance="major",
        known_by_character_ids=[],
        known_by_faction_ids=[],
    )
    defaults.update(kwargs)
    return Secret(**defaults)


# ---------------------------------------------------------------------------
# KnowledgeStore: has_similar_knowledge
# ---------------------------------------------------------------------------


class TestHasSimilarKnowledge:
    def test_exact_match(self, knowledge):
        knowledge.add_knowledge("npc-1", "NPC", "The sky is blue.")
        assert knowledge.has_similar_knowledge("npc-1", "The sky is blue.")

    def test_case_insensitive(self, knowledge):
        knowledge.add_knowledge("npc-1", "NPC", "The sky is blue.")
        assert knowledge.has_similar_knowledge("npc-1", "THE SKY IS BLUE.")

    def test_whitespace_normalized(self, knowledge):
        knowledge.add_knowledge("npc-1", "NPC", "  The sky is blue.  ")
        assert knowledge.has_similar_knowledge("npc-1", "The sky is blue.")

    def test_no_match(self, knowledge):
        knowledge.add_knowledge("npc-1", "NPC", "The sky is blue.")
        assert not knowledge.has_similar_knowledge("npc-1", "The grass is green.")

    def test_different_character(self, knowledge):
        knowledge.add_knowledge("npc-1", "NPC", "The sky is blue.")
        assert not knowledge.has_similar_knowledge("npc-2", "The sky is blue.")

    def test_substring_match(self, knowledge):
        knowledge.add_knowledge(
            "npc-1", "NPC",
            "The mayor of Sandpoint is secretly a vampire who feeds on travelers at night"
        )
        # Longer version contains the shorter as a substring
        assert knowledge.has_similar_knowledge(
            "npc-1",
            "The mayor of Sandpoint is secretly a vampire who feeds on travelers at night and has been doing so for years"
        )


# ---------------------------------------------------------------------------
# KnowledgeStore: copy_knowledge
# ---------------------------------------------------------------------------


class TestCopyKnowledge:
    def test_basic_copy(self, knowledge):
        original = knowledge.add_knowledge("npc-1", "Source NPC", "Secret info", importance=8)
        copy = knowledge.copy_knowledge(original.id, "npc-2", "Target NPC", "from_npc-1")
        assert copy is not None
        assert copy.character_id == "npc-2"
        assert copy.content == "Secret info"
        assert copy.source == "from_npc-1"
        assert copy.importance == 8

    def test_duplicate_returns_none(self, knowledge):
        original = knowledge.add_knowledge("npc-1", "Source", "Secret info")
        knowledge.add_knowledge("npc-2", "Target", "Secret info")
        copy = knowledge.copy_knowledge(original.id, "npc-2", "Target", "copy")
        assert copy is None

    def test_missing_source_returns_none(self, knowledge):
        result = knowledge.copy_knowledge(9999, "npc-2", "Target", "copy")
        assert result is None


# ---------------------------------------------------------------------------
# PropagationBus: on_secret_revealed
# ---------------------------------------------------------------------------


class TestSecretRevealed:
    def test_party_gets_knowledge(self, bus, knowledge):
        secret = _make_secret()
        count = bus.on_secret_revealed(secret, revealer="Ameiko")
        assert count >= 1

        entries = knowledge.query_knowledge(character_id="__party__")
        assert len(entries) == 1
        assert "vampire" in entries[0].content
        assert entries[0].knowledge_type == "secret"
        assert entries[0].source == "revealed_by_Ameiko"

    def test_npc_knowers_get_knowledge(self, bus, knowledge, characters):
        char = characters.create("Ameiko Kaijitsu", character_type="npc")
        secret = _make_secret(known_by_character_ids=[char.id])
        count = bus.on_secret_revealed(secret)
        assert count >= 2  # party + NPC

        npc_entries = knowledge.query_knowledge(character_id=char.id)
        assert len(npc_entries) == 1

    def test_faction_members_get_knowledge(self, bus, knowledge, factions, characters):
        char1 = characters.create("Guard A", character_type="npc")
        char2 = characters.create("Guard B", character_type="npc")
        faction = factions.create("Town Guard")
        factions.add_member(faction.id, char1.id)
        factions.add_member(faction.id, char2.id)

        secret = _make_secret(known_by_faction_ids=[faction.id])
        count = bus.on_secret_revealed(secret)
        # party + 2 faction members
        assert count == 3

    def test_importance_mapping(self, bus, knowledge):
        for imp, expected in [("minor", 4), ("major", 7), ("critical", 10)]:
            secret = _make_secret(
                id=f"secret-{imp}",
                content=f"Secret {imp} content",
                importance=imp,
            )
            bus.on_secret_revealed(secret)

        entries = knowledge.query_knowledge(character_id="__party__", limit=10)
        importances = {e.content.split()[1]: e.importance for e in entries}
        assert importances["minor"] == 4
        assert importances["major"] == 7
        assert importances["critical"] == 10

    def test_idempotent(self, bus, knowledge):
        secret = _make_secret()
        bus.on_secret_revealed(secret)
        bus.on_secret_revealed(secret)

        entries = knowledge.query_knowledge(character_id="__party__")
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# PropagationBus: on_faction_knowledge_added
# ---------------------------------------------------------------------------


class TestFactionKnowledgeAdded:
    def test_members_get_copy(self, bus, knowledge, factions, characters):
        char = characters.create("Guard", character_type="npc")
        faction = factions.create("Guards")
        factions.add_member(faction.id, char.id)

        entry = knowledge.add_knowledge("source", "Source", "Guard patrol schedule")
        count = bus.on_faction_knowledge_added(faction.id, entry.id)
        assert count == 1

        guard_entries = knowledge.query_knowledge(character_id=char.id)
        assert len(guard_entries) == 1
        assert guard_entries[0].source == f"faction:{faction.id}"

    def test_no_factions_no_op(self, knowledge):
        bus = PropagationBus(knowledge=knowledge)
        assert bus.on_faction_knowledge_added("x", 1) == 0


# ---------------------------------------------------------------------------
# PropagationBus: on_npc_joins_faction
# ---------------------------------------------------------------------------


class TestNPCJoinsFaction:
    def test_inherits_shared_knowledge(self, bus, knowledge, factions, characters):
        # Create knowledge and add to faction's shared_knowledge
        entry = knowledge.add_knowledge("source", "Source", "Secret handshake")
        faction = factions.create("Thieves Guild")
        factions.add_shared_knowledge(faction.id, str(entry.id))

        char = characters.create("New Thief", character_type="npc")
        count = bus.on_npc_joins_faction(char.id, char.name, faction.id)
        assert count == 1

        entries = knowledge.query_knowledge(character_id=char.id)
        assert len(entries) == 1
        assert entries[0].content == "Secret handshake"

    def test_no_faction_returns_zero(self, bus):
        assert bus.on_npc_joins_faction("x", "X", "nonexistent") == 0


# ---------------------------------------------------------------------------
# PropagationBus: propagate_location_knowledge
# ---------------------------------------------------------------------------


class TestLocationPropagation:
    def test_npc_gets_location_knowledge(self, bus, knowledge, locations):
        entry = knowledge.add_knowledge("source", "Source", "The tavern has a secret cellar")
        loc = locations.create("Rusty Dragon")
        locations.add_common_knowledge(loc.id, str(entry.id))

        count = bus.propagate_location_knowledge("npc-1", "Ameiko", loc.id)
        assert count == 1

        entries = knowledge.query_knowledge(character_id="npc-1")
        assert entries[0].source == f"location:{loc.id}"

    def test_no_locations_no_op(self, knowledge):
        bus = PropagationBus(knowledge=knowledge)
        assert bus.propagate_location_knowledge("npc-1", "NPC", "loc-1") == 0

    def test_duplicate_prevented(self, bus, knowledge, locations):
        entry = knowledge.add_knowledge("source", "Source", "Known info")
        loc = locations.create("Tavern")
        locations.add_common_knowledge(loc.id, str(entry.id))

        bus.propagate_location_knowledge("npc-1", "NPC", loc.id)
        count = bus.propagate_location_knowledge("npc-1", "NPC", loc.id)
        assert count == 0


# ---------------------------------------------------------------------------
# PropagationBus: graceful no-op with missing stores
# ---------------------------------------------------------------------------


class TestMissingStores:
    def test_secret_without_factions(self, knowledge):
        bus = PropagationBus(knowledge=knowledge)
        secret = _make_secret(known_by_faction_ids=["some-faction"])
        # Should still create party knowledge, just skip factions
        count = bus.on_secret_revealed(secret)
        assert count == 1  # party only
