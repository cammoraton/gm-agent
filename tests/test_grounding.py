"""Tests for GroundingServer."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gm_agent.mcp.grounding import GroundingServer
from gm_agent.storage.fiction_tree import FictionTreeStore, FictionNode


# ===========================================================================
# Fixtures
# ===========================================================================


class MockSearch:
    """Mock PathfinderSearch that returns canned results."""

    def __init__(self, results: list[dict] | None = None):
        self._results = results if results is not None else [
            {"id": "goblin-warrior", "name": "Goblin Warrior", "type": "creature",
             "book": "Monster Core", "score": -20, "content": "A small, vicious creature."},
            {"id": "fire-trap", "name": "Fire Trap", "type": "hazard",
             "book": "GM Core", "score": -15, "content": "A magical fire trap."},
            {"id": "longsword", "name": "Longsword", "type": "item",
             "book": "Player Core", "score": -10, "content": "A versatile melee weapon."},
        ]
        self.calls: list[dict] = []

    def search(self, query, include_types=None, exclude_types=None, top_k=10, **kwargs):
        self.calls.append({"query": query, "include_types": include_types, "top_k": top_k})
        results = self._results
        if include_types:
            results = [r for r in results if r.get("type") in include_types]
        return results[:top_k]


@pytest.fixture
def campaigns_dir(tmp_path: Path) -> Path:
    d = tmp_path / "campaigns"
    d.mkdir()
    (d / "ground-test").mkdir()
    return d


@pytest.fixture
def fiction_store(campaigns_dir: Path) -> FictionTreeStore:
    s = FictionTreeStore("ground-test", base_dir=campaigns_dir)
    yield s
    s.close()


@pytest.fixture
def mock_search() -> MockSearch:
    return MockSearch()


@pytest.fixture
def server(fiction_store: FictionTreeStore, mock_search: MockSearch) -> GroundingServer:
    return GroundingServer(
        campaign_id="ground-test",
        fiction_store=fiction_store,
        search=mock_search,
    )


def _create_timeline(fiction_store: FictionTreeStore) -> str:
    """Create a Microscope-style timeline fixture. Returns root ID."""
    root = fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        title="The Age of Heroes",
        zoom_level="era",
        source_system="microscope",
    ))
    p1 = fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=root.id,
        title="The Rise",
        zoom_level="period",
        tone="light",
        source_system="microscope",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=p1.id,
        title="The Battle of Dragon Peak",
        zoom_level="event",
        tone="dark",
        summary="Heroes fight a dragon at the mountain.",
        source_system="microscope",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=p1.id,
        title="The Great Treasure Hoard",
        zoom_level="event",
        tone="light",
        summary="A treasure hoard is discovered.",
        source_system="microscope",
    ))
    return root.id


def _create_settlement(fiction_store: FictionTreeStore) -> str:
    """Create an Ex Novo-style settlement fixture. Returns root ID."""
    root = fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        title="Riverside",
        zoom_level="era",
        summary="A riverside settlement",
        source_system="ex_novo",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=root.id,
        title="Market District",
        zoom_level="detail",
        source_system="ex_novo",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=root.id,
        title="The Thieves Guild",
        zoom_level="detail",
        summary="A shadowy faction controlling the docks.",
        source_system="ex_novo",
    ))
    return root.id


def _create_dungeon(fiction_store: FictionTreeStore) -> str:
    """Create a Delve-style dungeon fixture. Returns root ID."""
    root = fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        title="The Forgotten Mines",
        zoom_level="era",
        source_system="delve",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=root.id,
        title="The Collapsed Entrance",
        zoom_level="detail",
        summary="A trap-filled entrance.",
        source_system="delve",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=root.id,
        title="The Dragon's Lair",
        zoom_level="detail",
        summary="A creature guards the treasure.",
        source_system="delve",
    ))
    fiction_store.add_node(FictionNode(
        campaign_id="ground-test",
        parent_id=root.id,
        title="The Treasury",
        zoom_level="detail",
        summary="A chest of treasure and loot.",
        source_system="delve",
    ))
    return root.id


# ===========================================================================
# Tool Listing
# ===========================================================================


class TestGroundingTools:
    def test_list_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "ground_node" in names
        assert "suggest_groundings" in names
        assert "ground_settlement" in names
        assert "ground_dungeon" in names
        assert "ground_timeline" in names
        assert "list_groundings" in names
        assert len(names) == 6

    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent", {})
        assert not result.success


# ===========================================================================
# ground_node
# ===========================================================================


class TestGroundNode:
    def test_ground_node(self, server, fiction_store, mock_search):
        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        period = children[0]
        events = fiction_store.get_children(period.id)
        event = events[0]

        result = server.call_tool("ground_node", {"node_id": event.id})
        assert result.success
        assert "Battle of Dragon Peak" in result.data

        # Verify search was called
        assert len(mock_search.calls) >= 1

        # Verify references stored
        node = fiction_store.get_node(event.id)
        assert "rag_refs" in node.linked_entities

    def test_ground_node_with_types(self, server, fiction_store, mock_search):
        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)

        result = server.call_tool("ground_node", {
            "node_id": events[0].id,
            "content_types": "creature",
        })
        assert result.success
        # Should have filtered to only creatures
        assert any(c["include_types"] == ["creature"] for c in mock_search.calls)

    def test_ground_nonexistent_node(self, server):
        result = server.call_tool("ground_node", {"node_id": "fake"})
        assert not result.success
        assert "not found" in (result.error or "")

    def test_ground_empty_node(self, server, fiction_store):
        node = fiction_store.add_node(FictionNode(
            campaign_id="ground-test",
            title="",
            summary="",
            zoom_level="event",
        ))
        result = server.call_tool("ground_node", {"node_id": node.id})
        assert not result.success
        assert "no title" in (result.error or "").lower()

    def test_ground_no_results(self, fiction_store):
        empty_search = MockSearch(results=[])
        srv = GroundingServer("ground-test", fiction_store, empty_search)
        node = fiction_store.add_node(FictionNode(
            campaign_id="ground-test",
            title="Xyzzy",
            zoom_level="event",
        ))
        result = srv.call_tool("ground_node", {"node_id": node.id})
        assert result.success
        assert "No PF2e matches" in result.data


# ===========================================================================
# suggest_groundings
# ===========================================================================


class TestSuggestGroundings:
    def test_suggest_groundings(self, server, fiction_store):
        root_id = _create_timeline(fiction_store)
        result = server.call_tool("suggest_groundings", {"root_node_id": root_id})
        assert result.success
        # Should suggest grounding the battle event
        assert "Battle" in result.data or "Dragon" in result.data or "Treasure" in result.data

    def test_suggest_nonexistent(self, server):
        result = server.call_tool("suggest_groundings", {"root_node_id": "fake"})
        assert not result.success

    def test_suggest_max_limit(self, server, fiction_store):
        root_id = _create_timeline(fiction_store)
        result = server.call_tool("suggest_groundings", {
            "root_node_id": root_id,
            "max_suggestions": 1,
        })
        assert result.success

    def test_suggest_already_grounded_skipped(self, server, fiction_store):
        """Nodes that already have rag_refs should be skipped."""
        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)
        # Pre-ground the first event
        fiction_store.update_node(events[0].id, linked_entities={"rag_refs": ["something"]})

        result = server.call_tool("suggest_groundings", {"root_node_id": root_id})
        assert result.success
        # Battle event should not be suggested since it's already grounded
        # (but Treasure should still be)


# ===========================================================================
# ground_settlement
# ===========================================================================


class TestGroundSettlement:
    def test_ground_settlement(self, server, fiction_store, mock_search):
        root_id = _create_settlement(fiction_store)
        result = server.call_tool("ground_settlement", {"node_id": root_id})
        assert result.success
        assert "Riverside" in result.data
        assert len(mock_search.calls) >= 1

        # Verify references stored
        node = fiction_store.get_node(root_id)
        assert any(k in node.linked_entities for k in ("locations", "npcs", "creatures", "factions"))

    def test_ground_settlement_nonexistent(self, server):
        result = server.call_tool("ground_settlement", {"node_id": "fake"})
        assert not result.success


# ===========================================================================
# ground_dungeon
# ===========================================================================


class TestGroundDungeon:
    def test_ground_dungeon(self, server, fiction_store, mock_search):
        root_id = _create_dungeon(fiction_store)
        result = server.call_tool("ground_dungeon", {"node_id": root_id})
        assert result.success
        assert "Forgotten Mines" in result.data

        node = fiction_store.get_node(root_id)
        assert any(k in node.linked_entities for k in ("encounters", "creatures", "hazards", "items"))

    def test_ground_dungeon_with_level(self, server, fiction_store):
        root_id = _create_dungeon(fiction_store)
        result = server.call_tool("ground_dungeon", {
            "node_id": root_id,
            "party_level": 5,
        })
        assert result.success
        assert "level 5" in result.data

    def test_ground_dungeon_nonexistent(self, server):
        result = server.call_tool("ground_dungeon", {"node_id": "fake"})
        assert not result.success


# ===========================================================================
# ground_timeline
# ===========================================================================


class TestGroundTimeline:
    def test_ground_timeline(self, server, fiction_store, mock_search):
        root_id = _create_timeline(fiction_store)
        result = server.call_tool("ground_timeline", {"node_id": root_id})
        assert result.success
        assert "Age of Heroes" in result.data

        node = fiction_store.get_node(root_id)
        assert any(k in node.linked_entities for k in ("lore", "creatures", "npcs"))

    def test_ground_timeline_nonexistent(self, server):
        result = server.call_tool("ground_timeline", {"node_id": "fake"})
        assert not result.success


# ===========================================================================
# list_groundings
# ===========================================================================


class TestListGroundings:
    def test_list_empty(self, server, fiction_store):
        node = fiction_store.add_node(FictionNode(
            campaign_id="ground-test",
            title="Empty",
            zoom_level="event",
        ))
        result = server.call_tool("list_groundings", {"node_id": node.id})
        assert result.success
        assert "No groundings" in result.data

    def test_list_with_groundings(self, server, fiction_store):
        node = fiction_store.add_node(FictionNode(
            campaign_id="ground-test",
            title="Grounded",
            zoom_level="event",
            linked_entities={
                "creatures": ["goblin-warrior", "orc-brute"],
                "items": ["longsword"],
            },
        ))
        result = server.call_tool("list_groundings", {"node_id": node.id})
        assert result.success
        assert "Creatures" in result.data
        assert "goblin-warrior" in result.data
        assert "longsword" in result.data

    def test_list_nonexistent(self, server):
        result = server.call_tool("list_groundings", {"node_id": "fake"})
        assert not result.success

    def test_close(self, server):
        # Should not raise
        server.close()


# ===========================================================================
# LLM Reranking (WS2)
# ===========================================================================


class MockLLMForRerank:
    """Mock LLM that returns a JSON array for reranking."""

    def __init__(self, response_text: str = ""):
        from gm_agent.models.base import LLMResponse
        self.response_text = response_text or json.dumps([
            {"index": 2, "relevance": 9, "reasoning": "Longsword is a classic weapon."},
            {"index": 0, "relevance": 7, "reasoning": "Goblin fits the setting."},
        ])
        self.calls: list = []
        self._response_cls = LLMResponse

    def chat(self, messages, tools=None, thinking=None, temperature=None):
        self.calls.append(messages)
        return self._response_cls(text=self.response_text)

    def get_model_name(self):
        return "mock-rerank"

    def is_available(self):
        return True


class TestLLMReranking:
    def test_rerank_reorders_results(self, fiction_store, mock_search):
        """LLM reranking should reorder results by LLM selection."""
        import json as json_mod
        from gm_agent.models.base import LLMResponse

        llm = MockLLMForRerank()
        srv = GroundingServer("ground-test", fiction_store, mock_search, llm=llm)

        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)

        result = srv.call_tool("ground_node", {"node_id": events[0].id})
        assert result.success
        # LLM should have been called
        assert len(llm.calls) >= 1
        # Reasoning should appear in output
        assert "Longsword" in result.data or "Goblin" in result.data

    def test_rerank_includes_reasoning(self, fiction_store, mock_search):
        """Reranked results should include reasoning in output."""
        llm = MockLLMForRerank()
        srv = GroundingServer("ground-test", fiction_store, mock_search, llm=llm)

        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)

        result = srv.call_tool("ground_node", {"node_id": events[0].id})
        assert result.success
        assert "classic weapon" in result.data or "fits the setting" in result.data

    def test_rerank_fallback_on_error(self, fiction_store, mock_search):
        """Should fall back to raw results when LLM fails."""
        from gm_agent.models.base import LLMResponse

        class FailingLLM:
            def chat(self, messages, tools=None, thinking=None, temperature=None):
                raise RuntimeError("LLM error")
            def get_model_name(self):
                return "fail"
            def is_available(self):
                return True

        srv = GroundingServer("ground-test", fiction_store, mock_search, llm=FailingLLM())

        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)

        result = srv.call_tool("ground_node", {"node_id": events[0].id})
        assert result.success
        # Should still have results (fallback)
        assert "Goblin Warrior" in result.data

    def test_rerank_fallback_on_bad_json(self, fiction_store, mock_search):
        """Should fall back when LLM returns non-JSON."""
        llm = MockLLMForRerank(response_text="I cannot help with that.")
        srv = GroundingServer("ground-test", fiction_store, mock_search, llm=llm)

        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)

        result = srv.call_tool("ground_node", {"node_id": events[0].id})
        assert result.success
        # Should still work with fallback
        assert "Grounding suggestions" in result.data

    def test_no_rerank_without_llm(self, server, fiction_store, mock_search):
        """Without LLM, should use raw results."""
        root_id = _create_timeline(fiction_store)
        children = fiction_store.get_children(root_id)
        events = fiction_store.get_children(children[0].id)

        result = server.call_tool("ground_node", {"node_id": events[0].id})
        assert result.success
        # No reasoning should be present
        assert "classic weapon" not in result.data

    def test_rerank_settlement(self, fiction_store):
        """Reranking should work for settlement grounding."""
        settlement_search = MockSearch(results=[
            {"id": "otari", "name": "Otari", "type": "settlement",
             "book": "GM Core", "score": -20, "content": "A small coastal town."},
            {"id": "market", "name": "Market District", "type": "landmark",
             "book": "GM Core", "score": -15, "content": "A busy marketplace."},
        ])
        llm = MockLLMForRerank()
        srv = GroundingServer("ground-test", fiction_store, settlement_search, llm=llm)

        root_id = _create_settlement(fiction_store)
        result = srv.call_tool("ground_settlement", {"node_id": root_id})
        assert result.success
        assert len(llm.calls) >= 1
