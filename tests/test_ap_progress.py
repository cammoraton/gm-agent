"""Tests for AP progression tracking — Phase 5A."""

import pytest

from gm_agent.storage.ap_progress import APProgressStore, APProgressEntry


class TestAPProgressStore:
    """Tests for the SQLite-based AP progress store."""

    @pytest.fixture
    def store(self, tmp_path):
        s = APProgressStore("test-campaign", base_dir=tmp_path)
        yield s
        s.close()

    def test_mark_complete_encounter(self, store):
        entry = store.mark_complete(
            name="Goblin Ambush",
            entry_type="encounter",
            book="AV Book 1",
            xp_awarded=80,
        )
        assert entry.id is not None
        assert entry.name == "Goblin Ambush"
        assert entry.entry_type == "encounter"
        assert entry.xp_awarded == 80
        assert entry.completed is True

    def test_mark_complete_area(self, store):
        entry = store.mark_complete(
            name="Room B12",
            entry_type="area",
            book="AV Book 1",
        )
        assert entry.entry_type == "area"
        assert entry.xp_awarded == 0

    def test_mark_complete_milestone(self, store):
        entry = store.mark_complete(
            name="Rescued the Mayor",
            entry_type="milestone",
            xp_awarded=120,
            notes="Party convinced bandits to surrender",
        )
        assert entry.entry_type == "milestone"
        assert entry.xp_awarded == 120

    def test_get_progress_all(self, store):
        store.mark_complete("Encounter A", "encounter", book="Book 1", xp_awarded=40)
        store.mark_complete("Encounter B", "encounter", book="Book 1", xp_awarded=60)
        store.mark_complete("Area C", "area", book="Book 2")

        entries = store.get_progress()
        assert len(entries) == 3

    def test_get_progress_by_book(self, store):
        store.mark_complete("Encounter A", "encounter", book="Book 1", xp_awarded=40)
        store.mark_complete("Area B", "area", book="Book 2")

        entries = store.get_progress(book="Book 1")
        assert len(entries) == 1
        assert entries[0]["name"] == "Encounter A"

    def test_get_book_progress(self, store):
        store.mark_complete("E1", "encounter", book="Book 1", xp_awarded=40)
        store.mark_complete("E2", "encounter", book="Book 1", xp_awarded=60)
        store.mark_complete("A1", "area", book="Book 1")

        summary = store.get_book_progress("Book 1")
        assert summary["book"] == "Book 1"
        assert summary["total_entries"] == 3
        assert summary["total_xp"] == 100
        assert summary["types"]["encounter"]["count"] == 2

    def test_total_xp(self, store):
        store.mark_complete("E1", "encounter", xp_awarded=40)
        store.mark_complete("E2", "encounter", xp_awarded=80)
        store.mark_complete("M1", "milestone", xp_awarded=120)

        assert store.total_xp() == 240

    def test_total_xp_empty(self, store):
        assert store.total_xp() == 0

    def test_list_incomplete_empty(self, store):
        """All entries are created completed, so incomplete starts empty."""
        entries = store.list_incomplete()
        assert len(entries) == 0


class TestAPProgressTool:
    """Tests for the ap_progress compound tool via CampaignStateServer."""

    @pytest.fixture
    def setup(self, tmp_path):
        from pathlib import Path
        from gm_agent.mcp.campaign_state import CampaignStateServer
        from gm_agent.storage.campaign import CampaignStore
        from gm_agent.storage.session import SessionStore
        import gm_agent.mcp.campaign_state as cs_module

        campaigns_dir = tmp_path / "campaigns"
        campaigns_dir.mkdir()

        campaign_store_local = CampaignStore(base_dir=campaigns_dir)
        session_store_local = SessionStore(base_dir=campaigns_dir)
        campaign = campaign_store_local.create(name="AP Test", background="Test")
        session_store_local.start(campaign.id)

        original_cs = cs_module.campaign_store
        original_ss = cs_module.session_store
        original_cd = cs_module.CAMPAIGNS_DIR

        cs_module.campaign_store = campaign_store_local
        cs_module.session_store = session_store_local
        cs_module.CAMPAIGNS_DIR = campaigns_dir

        server = CampaignStateServer(campaign.id)

        yield {"server": server, "campaign": campaign}

        server.close()
        cs_module.campaign_store = original_cs
        cs_module.session_store = original_ss
        cs_module.CAMPAIGNS_DIR = original_cd

    def test_complete_encounter(self, setup):
        server = setup["server"]
        result = server.call_tool("ap_progress", {
            "action": "complete_encounter",
            "name": "Goblin Ambush",
            "xp": 80,
            "book": "AV Book 1",
        })
        assert result.success
        assert "Goblin Ambush" in result.data
        assert "+80 XP" in result.data

    def test_explore_area(self, setup):
        server = setup["server"]
        result = server.call_tool("ap_progress", {
            "action": "explore_area",
            "name": "Room B12",
        })
        assert result.success
        assert "Room B12" in result.data

    def test_milestone(self, setup):
        server = setup["server"]
        result = server.call_tool("ap_progress", {
            "action": "milestone",
            "name": "Cleared Level 1",
            "xp": 120,
        })
        assert result.success
        assert "Cleared Level 1" in result.data

    def test_get_progress(self, setup):
        server = setup["server"]
        server.call_tool("ap_progress", {
            "action": "complete_encounter", "name": "E1", "xp": 40,
        })
        server.call_tool("ap_progress", {
            "action": "explore_area", "name": "A1",
        })

        result = server.call_tool("ap_progress", {"action": "get_progress"})
        assert result.success
        assert "2 entries" in result.data
        assert "E1" in result.data

    def test_missing_name_returns_error(self, setup):
        server = setup["server"]
        result = server.call_tool("ap_progress", {
            "action": "complete_encounter",
        })
        assert not result.success
        assert "name" in result.error.lower()

    def test_unknown_action(self, setup):
        server = setup["server"]
        result = server.call_tool("ap_progress", {"action": "explode"})
        assert not result.success
        assert "Unknown" in result.error
