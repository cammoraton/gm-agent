"""Tests for FictionTreeStore and FictionTreeServer."""

import json
from pathlib import Path

import pytest

from gm_agent.storage.fiction_tree import FictionTreeStore, FictionNode
from gm_agent.mcp.fiction_tree import FictionTreeServer


CAMPAIGN_ID = "test-fiction"


@pytest.fixture
def store(tmp_path: Path) -> FictionTreeStore:
    """Create a FictionTreeStore with a temp directory."""
    s = FictionTreeStore(CAMPAIGN_ID, base_dir=tmp_path)
    yield s
    s.close()


@pytest.fixture
def server(tmp_path: Path) -> FictionTreeServer:
    """Create a FictionTreeServer with a temp directory."""
    from unittest.mock import patch
    with patch("gm_agent.mcp.fiction_tree.CAMPAIGNS_DIR", tmp_path):
        srv = FictionTreeServer(CAMPAIGN_ID)
        yield srv
        srv.close()


# ===========================================================================
# FictionNode model
# ===========================================================================


class TestFictionNode:
    def test_default_values(self):
        node = FictionNode()
        assert node.zoom_level == "era"
        assert node.tone is None
        assert node.tags == []
        assert node.source_system == "manual"
        assert node.parent_id is None
        assert len(node.id) == 16

    def test_custom_values(self):
        node = FictionNode(
            title="The Age of Wonders",
            zoom_level="period",
            tone="light",
            source_system="microscope",
        )
        assert node.title == "The Age of Wonders"
        assert node.zoom_level == "period"
        assert node.tone == "light"


# ===========================================================================
# FictionTreeStore CRUD
# ===========================================================================


class TestFictionTreeStoreCRUD:
    def test_add_and_get_node(self, store: FictionTreeStore):
        node = FictionNode(
            title="Root Era",
            zoom_level="era",
            tone="dark",
            summary="The beginning.",
            content="Long description of the era.",
        )
        created = store.add_node(node)
        assert created.id == node.id
        assert created.campaign_id == CAMPAIGN_ID

        fetched = store.get_node(node.id)
        assert fetched is not None
        assert fetched.title == "Root Era"
        assert fetched.tone == "dark"

    def test_get_nonexistent(self, store: FictionTreeStore):
        assert store.get_node("nonexistent") is None

    def test_update_node(self, store: FictionTreeStore):
        node = store.add_node(FictionNode(title="Original", zoom_level="period"))
        updated = store.update_node(node.id, title="Renamed", tone="light")
        assert updated is not None
        assert updated.title == "Renamed"
        assert updated.tone == "light"

        # Verify persisted
        fetched = store.get_node(node.id)
        assert fetched.title == "Renamed"

    def test_update_nonexistent(self, store: FictionTreeStore):
        assert store.update_node("nonexistent", title="X") is None

    def test_update_tags(self, store: FictionTreeStore):
        node = store.add_node(FictionNode(title="Tagged", zoom_level="event"))
        store.update_node(node.id, tags=["war", "magic"])
        fetched = store.get_node(node.id)
        assert fetched.tags == ["war", "magic"]

    def test_update_no_fields(self, store: FictionTreeStore):
        node = store.add_node(FictionNode(title="NoChange", zoom_level="event"))
        result = store.update_node(node.id)
        assert result is not None
        assert result.title == "NoChange"

    def test_update_ignores_protected_fields(self, store: FictionTreeStore):
        node = store.add_node(FictionNode(title="Protected", zoom_level="era"))
        original_id = node.id
        result = store.update_node(node.id, id="new-id")
        assert result.id == original_id

    def test_delete_node(self, store: FictionTreeStore):
        node = store.add_node(FictionNode(title="ToDelete", zoom_level="era"))
        assert store.delete_node(node.id) is True
        assert store.get_node(node.id) is None

    def test_delete_nonexistent(self, store: FictionTreeStore):
        assert store.delete_node("nonexistent") is False

    def test_delete_reparents_children(self, store: FictionTreeStore):
        root = store.add_node(FictionNode(title="Root", zoom_level="era"))
        child = store.add_node(FictionNode(title="Child", zoom_level="period", parent_id=root.id))
        grandchild = store.add_node(FictionNode(title="Grandchild", zoom_level="event", parent_id=child.id))

        store.delete_node(child.id, cascade=False)
        fetched_gc = store.get_node(grandchild.id)
        assert fetched_gc.parent_id == root.id

    def test_delete_cascade(self, store: FictionTreeStore):
        root = store.add_node(FictionNode(title="Root", zoom_level="era"))
        child = store.add_node(FictionNode(title="Child", zoom_level="period", parent_id=root.id))
        grandchild = store.add_node(FictionNode(title="Grandchild", zoom_level="event", parent_id=child.id))

        store.delete_node(root.id, cascade=True)
        assert store.get_node(root.id) is None
        assert store.get_node(child.id) is None
        assert store.get_node(grandchild.id) is None


# ===========================================================================
# FictionTreeStore Navigation
# ===========================================================================


class TestFictionTreeStoreNavigation:
    def _build_tree(self, store: FictionTreeStore):
        """Build a 3-level tree: root -> 2 children -> 1 grandchild."""
        root = store.add_node(FictionNode(title="Timeline", zoom_level="era"))
        c1 = store.add_node(FictionNode(title="Dark Ages", zoom_level="period", parent_id=root.id, sort_order=0))
        c2 = store.add_node(FictionNode(title="Renaissance", zoom_level="period", parent_id=root.id, sort_order=1))
        gc = store.add_node(FictionNode(title="The Plague", zoom_level="event", parent_id=c1.id))
        return root, c1, c2, gc

    def test_get_roots(self, store: FictionTreeStore):
        root, _, _, _ = self._build_tree(store)
        roots = store.get_roots()
        assert len(roots) == 1
        assert roots[0].id == root.id

    def test_get_children(self, store: FictionTreeStore):
        root, c1, c2, _ = self._build_tree(store)
        children = store.get_children(root.id)
        assert len(children) == 2
        assert children[0].title == "Dark Ages"
        assert children[1].title == "Renaissance"

    def test_get_children_empty(self, store: FictionTreeStore):
        root, _, c2, _ = self._build_tree(store)
        children = store.get_children(c2.id)
        assert children == []

    def test_get_ancestors(self, store: FictionTreeStore):
        _, c1, _, gc = self._build_tree(store)
        ancestors = store.get_ancestors(gc.id)
        assert len(ancestors) == 2
        assert ancestors[0].title == "Timeline"  # root
        assert ancestors[1].title == "Dark Ages"  # parent

    def test_get_ancestors_root(self, store: FictionTreeStore):
        root, _, _, _ = self._build_tree(store)
        ancestors = store.get_ancestors(root.id)
        assert ancestors == []

    def test_get_siblings(self, store: FictionTreeStore):
        _, c1, c2, _ = self._build_tree(store)
        siblings = store.get_siblings(c1.id)
        assert len(siblings) == 2
        titles = [s.title for s in siblings]
        assert "Dark Ages" in titles
        assert "Renaissance" in titles

    def test_reorder_siblings(self, store: FictionTreeStore):
        root, c1, c2, _ = self._build_tree(store)
        store.reorder_siblings([c2.id, c1.id])
        children = store.get_children(root.id)
        assert children[0].title == "Renaissance"
        assert children[1].title == "Dark Ages"


# ===========================================================================
# FictionTreeStore Search
# ===========================================================================


class TestFictionTreeStoreSearch:
    def test_fts_search(self, store: FictionTreeStore):
        store.add_node(FictionNode(title="The Great War", zoom_level="event", content="A devastating conflict"))
        store.add_node(FictionNode(title="Peace Treaty", zoom_level="event", content="End of hostilities"))

        results = store.search("war")
        assert len(results) >= 1
        assert any("War" in r.title for r in results)

    def test_fts_search_with_zoom_filter(self, store: FictionTreeStore):
        store.add_node(FictionNode(title="War Era", zoom_level="era", content="An age of conflict"))
        store.add_node(FictionNode(title="War Event", zoom_level="event", content="A specific battle"))

        results = store.search("war", zoom_level="event")
        assert all(r.zoom_level == "event" for r in results)

    def test_search_no_results(self, store: FictionTreeStore):
        results = store.search("nonexistent_xyz_123")
        assert results == []

    def test_by_tag(self, store: FictionTreeStore):
        store.add_node(FictionNode(title="Tagged", zoom_level="era", tags=["magic", "war"]))
        store.add_node(FictionNode(title="Untagged", zoom_level="era"))

        results = store.by_tag("magic")
        assert len(results) == 1
        assert results[0].title == "Tagged"

    def test_by_source_system(self, store: FictionTreeStore):
        store.add_node(FictionNode(title="From Microscope", zoom_level="era", source_system="microscope"))
        store.add_node(FictionNode(title="Manual", zoom_level="era", source_system="manual"))

        results = store.by_source_system("microscope")
        assert len(results) == 1
        assert results[0].title == "From Microscope"


# ===========================================================================
# FictionTreeStore Bulk
# ===========================================================================


class TestFictionTreeStoreBulk:
    def test_export_subtree(self, store: FictionTreeStore):
        root = store.add_node(FictionNode(title="Root", zoom_level="era"))
        child = store.add_node(FictionNode(title="Child", zoom_level="period", parent_id=root.id))

        exported = store.export_subtree(root.id)
        assert len(exported) == 2
        assert exported[0]["title"] == "Root"
        assert exported[1]["title"] == "Child"

    def test_import_subtree(self, store: FictionTreeStore):
        # Create source tree
        root = store.add_node(FictionNode(title="Source", zoom_level="era"))
        child = store.add_node(FictionNode(title="Source Child", zoom_level="period", parent_id=root.id))

        exported = store.export_subtree(root.id)

        # Import under a new parent
        new_root = store.add_node(FictionNode(title="New Root", zoom_level="era"))
        imported = store.import_subtree(exported, new_parent_id=new_root.id)

        assert len(imported) == 2
        # IDs should be different
        assert imported[0].id != root.id
        assert imported[1].id != child.id

    def test_get_tree_summary(self, store: FictionTreeStore):
        root = store.add_node(FictionNode(title="Timeline", zoom_level="era"))
        store.add_node(FictionNode(title="Dark Ages", zoom_level="period", tone="dark", parent_id=root.id, summary="Bad times"))
        store.add_node(FictionNode(title="Renaissance", zoom_level="period", tone="light", parent_id=root.id, summary="Good times"))

        summary = store.get_tree_summary(root.id)
        assert "Timeline" in summary
        assert "Dark Ages" in summary
        assert "[Dark]" in summary
        assert "Renaissance" in summary
        assert "[Light]" in summary

    def test_get_tree_summary_no_root(self, store: FictionTreeStore):
        store.add_node(FictionNode(title="Root A", zoom_level="era"))
        store.add_node(FictionNode(title="Root B", zoom_level="era"))

        summary = store.get_tree_summary()
        assert "Root A" in summary
        assert "Root B" in summary


# ===========================================================================
# FictionTreeServer Tools
# ===========================================================================


class TestFictionTreeServer:
    def test_list_tools(self, server: FictionTreeServer):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "browse_fiction_tree" in names
        assert "get_fiction_node" in names
        assert "search_fiction" in names
        assert "add_fiction_node" in names
        assert "update_fiction_node" in names
        assert "link_fiction_node" in names
        assert len(names) == 6

    def test_browse_empty(self, server: FictionTreeServer):
        result = server.call_tool("browse_fiction_tree", {})
        assert result.success
        assert "empty" in result.data.lower()

    def test_add_and_browse(self, server: FictionTreeServer):
        add_result = server.call_tool("add_fiction_node", {
            "title": "The Age of Stars",
            "zoom_level": "era",
            "tone": "light",
            "summary": "A bright beginning",
        })
        assert add_result.success
        assert "Age of Stars" in add_result.data

        browse_result = server.call_tool("browse_fiction_tree", {})
        assert browse_result.success
        assert "Age of Stars" in browse_result.data

    def test_add_child_and_browse(self, server: FictionTreeServer):
        root = server.call_tool("add_fiction_node", {
            "title": "Timeline",
            "zoom_level": "era",
        })
        # Extract ID from result
        root_id = root.data.split("ID:** ")[1].split("\n")[0].strip()

        server.call_tool("add_fiction_node", {
            "title": "Dark Period",
            "zoom_level": "period",
            "parent_id": root_id,
            "tone": "dark",
        })

        result = server.call_tool("browse_fiction_tree", {"node_id": root_id})
        assert result.success
        assert "Dark Period" in result.data

    def test_get_fiction_node(self, server: FictionTreeServer):
        add_result = server.call_tool("add_fiction_node", {
            "title": "The Great Cataclysm",
            "zoom_level": "event",
            "tone": "dark",
            "content": "The world was forever changed.",
            "tags": '["destruction", "magic"]',
        })
        node_id = add_result.data.split("ID:** ")[1].split("\n")[0].strip()

        result = server.call_tool("get_fiction_node", {"node_id": node_id})
        assert result.success
        assert "Great Cataclysm" in result.data
        assert "dark" in result.data.lower()
        assert "forever changed" in result.data

    def test_get_nonexistent_node(self, server: FictionTreeServer):
        result = server.call_tool("get_fiction_node", {"node_id": "nonexistent"})
        assert not result.success

    def test_search_fiction(self, server: FictionTreeServer):
        server.call_tool("add_fiction_node", {
            "title": "The Dragon Wars",
            "zoom_level": "event",
            "content": "Dragons fought for dominion",
        })
        server.call_tool("add_fiction_node", {
            "title": "Peace of Elders",
            "zoom_level": "event",
            "content": "Peace finally came",
        })

        result = server.call_tool("search_fiction", {"query": "dragon"})
        assert result.success
        assert "Dragon Wars" in result.data

    def test_update_fiction_node(self, server: FictionTreeServer):
        add_result = server.call_tool("add_fiction_node", {
            "title": "Original Title",
            "zoom_level": "period",
        })
        node_id = add_result.data.split("ID:** ")[1].split("\n")[0].strip()

        result = server.call_tool("update_fiction_node", {
            "node_id": node_id,
            "title": "Updated Title",
            "tone": "dark",
        })
        assert result.success
        assert "Updated Title" in result.data

    def test_update_nonexistent(self, server: FictionTreeServer):
        result = server.call_tool("update_fiction_node", {
            "node_id": "nonexistent",
            "title": "X",
        })
        assert not result.success

    def test_update_no_fields(self, server: FictionTreeServer):
        result = server.call_tool("update_fiction_node", {"node_id": "x"})
        assert not result.success

    def test_link_fiction_node(self, server: FictionTreeServer):
        add_result = server.call_tool("add_fiction_node", {
            "title": "Battle Scene",
            "zoom_level": "scene",
        })
        node_id = add_result.data.split("ID:** ")[1].split("\n")[0].strip()

        result = server.call_tool("link_fiction_node", {
            "node_id": node_id,
            "link_type": "characters",
            "entity_ids": '["npc-001", "npc-002"]',
        })
        assert result.success
        assert "2 characters" in result.data

    def test_link_nonexistent_node(self, server: FictionTreeServer):
        result = server.call_tool("link_fiction_node", {
            "node_id": "nonexistent",
            "link_type": "characters",
            "entity_ids": '["x"]',
        })
        assert not result.success

    def test_unknown_tool(self, server: FictionTreeServer):
        result = server.call_tool("nonexistent_tool", {})
        assert not result.success

    def test_browse_children_of_nonexistent_parent(self, server: FictionTreeServer):
        result = server.call_tool("browse_fiction_tree", {"node_id": "nonexistent"})
        assert result.success
        assert "No children" in result.data
