"""Tests for treasure tracking — Phase 5B."""

import pytest

from gm_agent.storage.treasure import TreasureStore, TreasureEntry, TREASURE_BY_LEVEL


class TestTreasureStore:
    """Tests for the SQLite-based treasure store."""

    @pytest.fixture
    def store(self, tmp_path):
        s = TreasureStore("test-campaign", base_dir=tmp_path)
        yield s
        s.close()

    def test_add_item(self, store):
        entry = store.add_item(
            item_name="+1 Longsword",
            value_gp=100.0,
            item_level=4,
            source="Goblin Ambush",
        )
        assert entry.id is not None
        assert entry.item_name == "+1 Longsword"
        assert entry.value_gp == 100.0
        assert entry.item_level == 4
        assert entry.holder == "party"

    def test_distribute_item(self, store):
        entry = store.add_item("+1 Longsword", value_gp=100.0)
        success = store.distribute_item(entry.id, "Valeros")
        assert success

        items = store.list_items(holder="Valeros")
        assert len(items) == 1
        assert items[0]["item_name"] == "+1 Longsword"

    def test_distribute_nonexistent(self, store):
        success = store.distribute_item(999, "Valeros")
        assert not success

    def test_sell_item(self, store):
        entry = store.add_item("Potion of Healing", value_gp=50.0)
        sale_value = store.sell_item(entry.id)
        assert sale_value == 25.0  # Half price

        items = store.list_items(holder="sold")
        assert len(items) == 1

    def test_sell_nonexistent(self, store):
        sale_value = store.sell_item(999)
        assert sale_value == 0.0

    def test_get_party_wealth(self, store):
        store.add_item("Sword", value_gp=100.0)
        store.add_item("Shield", value_gp=50.0)

        wealth = store.get_party_wealth()
        assert wealth["total_items"] == 2
        assert wealth["total_value_gp"] == 150.0

    def test_get_party_wealth_with_sold(self, store):
        e1 = store.add_item("Sword", value_gp=100.0)
        store.add_item("Shield", value_gp=50.0)
        store.sell_item(e1.id)

        wealth = store.get_party_wealth()
        assert wealth["total_items"] == 1  # Only shield (unsold)
        assert wealth["sold_income_gp"] == 50.0  # Half of 100
        assert wealth["effective_wealth_gp"] == 100.0  # 50 (shield) + 50 (sold income)

    def test_get_wealth_by_level(self, store):
        store.add_item("Gold coins", value_gp=100.0)

        comparison = store.get_wealth_by_level(party_level=3, party_size=4)
        assert comparison["party_level"] == 3
        assert comparison["current_wealth_gp"] == 100.0
        assert comparison["expected_wealth_gp"] == 75.0 * 4  # 300 gp
        assert comparison["difference_gp"] == 100.0 - 300.0

    def test_list_items_all(self, store):
        store.add_item("A", value_gp=10.0)
        store.add_item("B", value_gp=20.0)

        items = store.list_items()
        assert len(items) == 2

    def test_list_items_by_holder(self, store):
        e1 = store.add_item("A", value_gp=10.0)
        store.add_item("B", value_gp=20.0)
        store.distribute_item(e1.id, "Ezren")

        party_items = store.list_items(holder="party")
        assert len(party_items) == 1
        assert party_items[0]["item_name"] == "B"


class TestTreasureByLevelTable:
    """Verify the treasure-by-level table."""

    def test_table_has_all_levels(self):
        for level in range(1, 21):
            assert level in TREASURE_BY_LEVEL, f"Missing treasure entry for level {level}"

    def test_values_increase_with_level(self):
        for level in range(2, 21):
            assert TREASURE_BY_LEVEL[level]["total"] > TREASURE_BY_LEVEL[level - 1]["total"]


class TestTreasureTool:
    """Tests for the treasure compound tool via CampaignStateServer."""

    @pytest.fixture
    def setup(self, tmp_path):
        from gm_agent.mcp.campaign_state import CampaignStateServer
        from gm_agent.storage.campaign import CampaignStore
        from gm_agent.storage.session import SessionStore
        import gm_agent.mcp.campaign_state as cs_module

        campaigns_dir = tmp_path / "campaigns"
        campaigns_dir.mkdir()

        campaign_store_local = CampaignStore(base_dir=campaigns_dir)
        session_store_local = SessionStore(base_dir=campaigns_dir)
        campaign = campaign_store_local.create(name="Treasure Test", background="Test")
        session_store_local.start(campaign.id)

        original_cs = cs_module.campaign_store
        original_ss = cs_module.session_store
        original_cd = cs_module.CAMPAIGNS_DIR

        cs_module.campaign_store = campaign_store_local
        cs_module.session_store = session_store_local
        cs_module.CAMPAIGNS_DIR = campaigns_dir

        server = CampaignStateServer(campaign.id)

        yield {"server": server}

        server.close()
        cs_module.campaign_store = original_cs
        cs_module.session_store = original_ss
        cs_module.CAMPAIGNS_DIR = original_cd

    def test_log_item(self, setup):
        server = setup["server"]
        result = server.call_tool("treasure", {
            "action": "log",
            "item_name": "+1 Longsword",
            "value_gp": 100,
            "item_level": 4,
            "source": "Goblin Chief",
        })
        assert result.success
        assert "+1 Longsword" in result.data
        assert "100" in result.data
        assert "ID:" in result.data

    def test_log_requires_name(self, setup):
        server = setup["server"]
        result = server.call_tool("treasure", {"action": "log"})
        assert not result.success
        assert "item_name" in result.error

    def test_distribute_item(self, setup):
        server = setup["server"]
        # Log first
        r1 = server.call_tool("treasure", {
            "action": "log", "item_name": "Sword", "value_gp": 100,
        })
        # Extract ID
        import re
        item_id = int(re.search(r"ID:\s*(\d+)", r1.data).group(1))

        r2 = server.call_tool("treasure", {
            "action": "distribute", "item_id": item_id, "character": "Valeros",
        })
        assert r2.success
        assert "Valeros" in r2.data

    def test_sell_item(self, setup):
        server = setup["server"]
        r1 = server.call_tool("treasure", {
            "action": "log", "item_name": "Gem", "value_gp": 50,
        })
        import re
        item_id = int(re.search(r"ID:\s*(\d+)", r1.data).group(1))

        r2 = server.call_tool("treasure", {
            "action": "sell", "item_id": item_id,
        })
        assert r2.success
        assert "25.0 gp" in r2.data

    def test_wealth_summary(self, setup):
        server = setup["server"]
        server.call_tool("treasure", {
            "action": "log", "item_name": "Gold", "value_gp": 200,
        })
        result = server.call_tool("treasure", {"action": "wealth"})
        assert result.success
        assert "200.0 gp" in result.data

    def test_by_level(self, setup):
        server = setup["server"]
        server.call_tool("treasure", {
            "action": "log", "item_name": "Gold", "value_gp": 100,
        })
        result = server.call_tool("treasure", {
            "action": "by_level", "party_level": 1, "party_size": 4,
        })
        assert result.success
        assert "Level 1" in result.data
        assert "Expected" in result.data

    def test_unknown_action(self, setup):
        server = setup["server"]
        result = server.call_tool("treasure", {"action": "melt"})
        assert not result.success
        assert "Unknown" in result.error
