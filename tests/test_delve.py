"""Tests for DelveSystem and DelveServer."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gm_agent.systems import get_system, SYSTEM_REGISTRY
from gm_agent.systems.delve.servers import DelveServer
from gm_agent.systems.delve.tables import ROOM_TYPES, UNIT_TYPES, DEPTH_SCALING
from gm_agent.systems.shared.cards import CardDeck
from gm_agent.storage.fiction_tree import FictionTreeStore
import gm_agent.systems as systems_mod


@pytest.fixture(autouse=True)
def _ensure_discovered():
    systems_mod._discovered = False
    systems_mod._discover_systems()
    yield
    systems_mod._discovered = False


@pytest.fixture
def campaigns_dir(tmp_path: Path) -> Path:
    d = tmp_path / "campaigns"
    d.mkdir()
    (d / "delve-test").mkdir()
    return d


@pytest.fixture
def server(campaigns_dir: Path) -> DelveServer:
    with patch("gm_agent.systems.delve.servers.CAMPAIGNS_DIR", campaigns_dir):
        srv = DelveServer("delve-test")
        yield srv
        srv.close()


@pytest.fixture
def fiction_store(campaigns_dir: Path) -> FictionTreeStore:
    s = FictionTreeStore("delve-test", base_dir=campaigns_dir)
    yield s
    s.close()


def _setup_hold(server: DelveServer, **overrides) -> dict:
    args = {"hold_name": "Irondeep", "resources": 20, "trade_goods": 10}
    args.update(overrides)
    return server.call_tool("delve_setup", args)


# ===========================================================================
# System Registration
# ===========================================================================


class TestDelveSystem:
    def test_registered(self):
        assert "delve" in SYSTEM_REGISTRY

    def test_metadata(self):
        system = get_system("delve")
        assert system.name == "delve"
        assert system.display_name == "Delve"
        assert system.category == "generation"

    def test_prompt(self):
        system = get_system("delve")
        prompt = system.system_prompt()
        assert "Delve" in prompt

    def test_servers_no_campaign(self):
        system = get_system("delve")
        assert system.servers({}) == []

    def test_servers_with_campaign(self, campaigns_dir):
        system = get_system("delve")
        with patch("gm_agent.systems.delve.servers.CAMPAIGNS_DIR", campaigns_dir):
            servers = system.servers({"campaign_id": "delve-test"})
            assert len(servers) == 1
            assert isinstance(servers[0], DelveServer)


# ===========================================================================
# Tool Listing
# ===========================================================================


class TestToolListing:
    def test_list_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "delve_setup" in names
        assert "explore" in names
        assert "resolve_combat" in names
        assert "trade" in names
        assert "build_room" in names
        assert "recruit_unit" in names
        assert "build_trap" in names
        assert "get_hold_state" in names
        assert "descend" in names
        assert "end_delve" in names
        assert "add_virtual_player" in names
        assert "virtual_player_turn" in names
        assert "remove_virtual_player" in names
        assert len(names) == 13


# ===========================================================================
# Setup
# ===========================================================================


class TestSetup:
    def test_setup(self, server):
        result = _setup_hold(server)
        assert result.success
        assert "Irondeep" in result.data

    def test_setup_state(self, server):
        _setup_hold(server)
        assert server.state["hold_name"] == "Irondeep"
        assert server.state["phase"] == "active"
        assert server.state["depth"] == 1
        assert server.state["resources"] == 20
        assert server.state["trade_goods"] == 10
        assert server.state["deck_remaining"] == 52

    def test_setup_creates_fiction_node(self, server, fiction_store):
        _setup_hold(server)
        roots = fiction_store.get_roots()
        assert len(roots) == 1
        assert roots[0].title == "Irondeep"

    def test_setup_twice_fails(self, server):
        _setup_hold(server)
        result = _setup_hold(server)
        assert not result.success
        assert "already active" in (result.error or "")

    def test_custom_resources(self, server):
        _setup_hold(server, resources=30, trade_goods=15)
        assert server.state["resources"] == 30
        assert server.state["trade_goods"] == 15


# ===========================================================================
# Exploration
# ===========================================================================


class TestExplore:
    def test_explore_draws_card(self, server):
        _setup_hold(server)
        result = server.call_tool("explore", {})
        assert result.success
        assert server.state["deck_remaining"] == 51
        assert server.state["turn_number"] == 1

    def test_explore_records_discovery(self, server):
        _setup_hold(server)
        server.call_tool("explore", {})
        assert len(server.state["discoveries"]) == 1
        assert server.state["discoveries"][0]["depth"] == 1

    def test_explore_writes_fiction_node(self, server, fiction_store):
        _setup_hold(server)
        server.call_tool("explore", {})
        root_id = server.state["history_root_id"]
        children = fiction_store.get_children(root_id)
        assert len(children) >= 1

    def test_explore_no_session(self, server):
        result = server.call_tool("explore", {})
        assert not result.success

    def test_multiple_explorations(self, server):
        _setup_hold(server)
        for _ in range(5):
            result = server.call_tool("explore", {})
            assert result.success
        assert server.state["deck_remaining"] == 47
        assert server.state["turn_number"] == 5


# ===========================================================================
# Combat
# ===========================================================================


class TestCombat:
    def test_combat_victory(self, server):
        _setup_hold(server)
        # Recruit a unit stronger than the enemy
        server.call_tool("recruit_unit", {"unit_type": "soldier", "name": "Guard"})
        result = server.call_tool("resolve_combat", {"enemy_str": 1})
        assert result.success
        assert "Victory" in result.data

    def test_combat_defeat(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "miner", "name": "Digger"})
        result = server.call_tool("resolve_combat", {"enemy_str": 100})
        assert result.success
        assert "Defeat" in result.data

    def test_combat_stalemate(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "soldier", "name": "Guard"})
        # soldier has STR 2
        result = server.call_tool("resolve_combat", {"enemy_str": 2})
        assert result.success
        assert "Stalemate" in result.data

    def test_combat_logged(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "soldier", "name": "Guard"})
        server.call_tool("resolve_combat", {"enemy_str": 1})
        assert len(server.state["combat_log"]) == 1

    def test_combat_defeat_loses_unit(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "miner", "name": "Digger"})
        assert len(server.state["units"]) == 1
        server.call_tool("resolve_combat", {"enemy_str": 100})
        assert len(server.state["units"]) == 0

    def test_combat_no_session(self, server):
        result = server.call_tool("resolve_combat", {"enemy_str": 5})
        assert not result.success

    def test_combat_with_specific_units(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "soldier", "name": "Champion"})
        server.call_tool("recruit_unit", {"unit_type": "miner", "name": "Digger"})
        result = server.call_tool("resolve_combat", {
            "enemy_str": 1,
            "party_units": '["Champion"]',
        })
        assert result.success
        assert "Victory" in result.data


# ===========================================================================
# Trade
# ===========================================================================


class TestTrade:
    def test_trade(self, server):
        _setup_hold(server)
        result = server.call_tool("trade", {
            "trade_goods_spent": 5,
            "resources_gained": 3,
        })
        assert result.success
        assert server.state["trade_goods"] == 5
        assert server.state["resources"] == 23

    def test_trade_insufficient_goods(self, server):
        _setup_hold(server)
        result = server.call_tool("trade", {
            "trade_goods_spent": 100,
            "resources_gained": 50,
        })
        assert not result.success
        assert "Not enough" in result.error

    def test_trade_no_session(self, server):
        result = server.call_tool("trade", {
            "trade_goods_spent": 1,
            "resources_gained": 1,
        })
        assert not result.success


# ===========================================================================
# Building
# ===========================================================================


class TestBuildRoom:
    def test_build_room(self, server):
        _setup_hold(server)
        result = server.call_tool("build_room", {
            "room_type": "quarters",
            "name": "Miner's Quarters",
        })
        assert result.success
        assert "Miner's Quarters" in result.data
        assert len(server.state["rooms"]) == 1
        assert server.state["resources"] == 20 - ROOM_TYPES["quarters"]["cost"]

    def test_build_unknown_room(self, server):
        _setup_hold(server)
        result = server.call_tool("build_room", {"room_type": "nonexistent"})
        assert not result.success

    def test_build_room_insufficient_resources(self, server):
        _setup_hold(server, resources=1)
        result = server.call_tool("build_room", {"room_type": "treasury"})
        assert not result.success
        assert "Not enough" in result.error

    def test_treasury_expands_caps(self, server):
        _setup_hold(server, resources=50)
        server.call_tool("build_room", {"room_type": "treasury"})
        assert server.state["resource_cap"] == 75
        assert server.state["trade_goods_cap"] == 75

    def test_build_room_no_session(self, server):
        result = server.call_tool("build_room", {"room_type": "quarters"})
        assert not result.success


class TestRecruitUnit:
    def test_recruit_unit(self, server):
        _setup_hold(server)
        result = server.call_tool("recruit_unit", {
            "unit_type": "soldier",
            "name": "Guard",
        })
        assert result.success
        assert len(server.state["units"]) == 1
        assert server.state["units"][0]["str"] == UNIT_TYPES["soldier"]["str"]

    def test_recruit_unknown_unit(self, server):
        _setup_hold(server)
        result = server.call_tool("recruit_unit", {"unit_type": "dragon"})
        assert not result.success

    def test_recruit_insufficient_resources(self, server):
        _setup_hold(server, resources=0)
        result = server.call_tool("recruit_unit", {"unit_type": "soldier"})
        assert not result.success

    def test_recruit_no_session(self, server):
        result = server.call_tool("recruit_unit", {"unit_type": "soldier"})
        assert not result.success


class TestBuildTrap:
    def test_build_trap(self, server):
        _setup_hold(server)
        result = server.call_tool("build_trap", {
            "trap_type": "pit",
            "level": 2,
        })
        assert result.success
        assert len(server.state["traps"]) == 1
        assert server.state["traps"][0]["type"] == "pit"
        assert server.state["traps"][0]["level"] == 2
        # Cost = level * 2 = 4
        assert server.state["resources"] == 16

    def test_build_trap_insufficient_resources(self, server):
        _setup_hold(server, resources=1)
        result = server.call_tool("build_trap", {"trap_type": "pit", "level": 3})
        assert not result.success

    def test_build_trap_no_session(self, server):
        result = server.call_tool("build_trap", {"trap_type": "pit"})
        assert not result.success


# ===========================================================================
# Descend
# ===========================================================================


class TestDescend:
    def test_descend(self, server):
        _setup_hold(server)
        result = server.call_tool("descend", {})
        assert result.success
        assert server.state["depth"] == 2
        assert server.state["deck_remaining"] == 52  # reshuffled

    def test_descend_multiple(self, server):
        _setup_hold(server)
        server.call_tool("descend", {})
        server.call_tool("descend", {})
        assert server.state["depth"] == 3

    def test_descend_no_session(self, server):
        result = server.call_tool("descend", {})
        assert not result.success


# ===========================================================================
# Hold State
# ===========================================================================


class TestGetState:
    def test_get_state(self, server):
        _setup_hold(server)
        result = server.call_tool("get_hold_state", {})
        assert result.success
        assert "Irondeep" in result.data
        assert "Resources" in result.data

    def test_get_state_with_units(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "soldier", "name": "Guard"})
        result = server.call_tool("get_hold_state", {})
        assert "Guard" in result.data
        assert "STR" in result.data

    def test_get_state_no_session(self, server):
        result = server.call_tool("get_hold_state", {})
        assert result.success
        assert "No active" in result.data


# ===========================================================================
# End
# ===========================================================================


class TestEnd:
    def test_end_delve(self, server):
        _setup_hold(server)
        server.call_tool("recruit_unit", {"unit_type": "soldier", "name": "Guard"})
        server.call_tool("explore", {})
        result = server.call_tool("end_delve", {})
        assert result.success
        assert "Ended" in result.data
        assert server.state["phase"] == "ended"

    def test_end_no_session(self, server):
        result = server.call_tool("end_delve", {})
        assert result.success
        assert "No active" in result.data

    def test_can_restart_after_end(self, server):
        _setup_hold(server)
        server.call_tool("end_delve", {})
        result = _setup_hold(server, hold_name="New Hold")
        assert result.success


# ===========================================================================
# Unknown tool
# ===========================================================================


class TestUnknown:
    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.error
