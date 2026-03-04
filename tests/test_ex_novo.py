"""Tests for ExNovoSystem and SettlementServer."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gm_agent.systems import get_system, SYSTEM_REGISTRY
from gm_agent.systems.ex_novo.servers import SettlementServer
from gm_agent.systems.ex_novo.tables import (
    AGE_TABLE, SIZE_TABLE, INFLUENCE_TABLE,
    GEOGRAPHY_TABLE, FEATURES_TABLE,
    LOCATION_TABLE, DECISION_TABLE,
    PARADIGM_TABLE, RELATIONSHIP_TABLE,
    EVENT_TABLE,
    roll_d6, roll_2d6, roll_d666,
)
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
    (d / "novo-test").mkdir()
    return d


@pytest.fixture
def server(campaigns_dir: Path) -> SettlementServer:
    with patch("gm_agent.systems.ex_novo.servers.CAMPAIGNS_DIR", campaigns_dir):
        srv = SettlementServer("novo-test")
        yield srv
        srv.close()


@pytest.fixture
def fiction_store(campaigns_dir: Path) -> FictionTreeStore:
    s = FictionTreeStore("novo-test", base_dir=campaigns_dir)
    yield s
    s.close()


def _setup_settlement(server: SettlementServer) -> dict:
    result = server.call_tool("ex_novo_setup", {"name": "Thornwall"})
    return result


# ===========================================================================
# System Registration
# ===========================================================================


class TestExNovoSystem:
    def test_registered(self):
        assert "ex_novo" in SYSTEM_REGISTRY

    def test_metadata(self):
        system = get_system("ex_novo")
        assert system.name == "ex_novo"
        assert system.display_name == "Ex Novo"
        assert system.category == "generation"

    def test_prompt(self):
        system = get_system("ex_novo")
        prompt = system.system_prompt()
        assert "Ex Novo" in prompt
        assert "settlement" in prompt.lower()

    def test_servers_no_campaign(self):
        system = get_system("ex_novo")
        servers = system.servers({})
        assert servers == []

    def test_servers_with_campaign(self, campaigns_dir):
        system = get_system("ex_novo")
        with patch("gm_agent.systems.ex_novo.servers.CAMPAIGNS_DIR", campaigns_dir):
            servers = system.servers({"campaign_id": "novo-test"})
            assert len(servers) == 1
            assert isinstance(servers[0], SettlementServer)


# ===========================================================================
# Tables
# ===========================================================================


class TestTables:
    def test_d6_tables_structure(self):
        # Check that whatever entries exist have valid structure (int key, str value)
        for table in (AGE_TABLE, SIZE_TABLE, INFLUENCE_TABLE):
            assert len(table) > 0
            for k, v in table.items():
                assert isinstance(k, int)
                assert isinstance(v, str)

    def test_2d6_tables_structure(self):
        for table in (GEOGRAPHY_TABLE, FEATURES_TABLE, LOCATION_TABLE,
                      DECISION_TABLE, PARADIGM_TABLE, RELATIONSHIP_TABLE):
            assert len(table) > 0
            for k, v in table.items():
                assert isinstance(k, int)
                assert isinstance(v, str)

    def test_event_table_has_entries(self):
        assert len(EVENT_TABLE) > 0
        for code, entry in EVENT_TABLE.items():
            assert "text" in entry
            assert "effect" in entry

    def test_roll_d6_range(self):
        results = {roll_d6() for _ in range(200)}
        assert results <= set(range(1, 7))
        assert len(results) > 1  # not always the same

    def test_roll_2d6_range(self):
        results = {roll_2d6() for _ in range(200)}
        assert results <= set(range(2, 13))
        assert len(results) > 1

    def test_roll_d666_format(self):
        for _ in range(50):
            code = roll_d666()
            # D666 = 3 dice concatenated: each digit is 1-6
            digits = str(code)
            assert len(digits) == 3
            for d in digits:
                assert d in "123456"


# ===========================================================================
# Tool Listing
# ===========================================================================


class TestToolListing:
    def test_list_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "ex_novo_setup" in names
        assert "add_faction" in names
        assert "add_district" in names
        assert "add_landmark" in names
        assert "roll_event" in names
        assert "resolve_event" in names
        assert "advance_phase" in names
        assert "get_settlement_state" in names
        assert "browse_settlement_history" in names
        assert "end_settlement" in names
        assert "add_virtual_player" in names
        assert "virtual_player_turn" in names
        assert "remove_virtual_player" in names
        assert len(names) == 13


# ===========================================================================
# Setup
# ===========================================================================


class TestSetup:
    def test_setup_creates_settlement(self, server):
        result = _setup_settlement(server)
        assert result.success
        assert "Thornwall" in result.data

    def test_setup_has_scale(self, server):
        result = _setup_settlement(server)
        assert "Scale" in result.data

    def test_setup_has_terrain(self, server):
        result = _setup_settlement(server)
        assert "Terrain" in result.data

    def test_setup_has_purpose(self, server):
        result = _setup_settlement(server)
        assert "Purpose" in result.data

    def test_setup_has_power(self, server):
        result = _setup_settlement(server)
        assert "Power" in result.data

    def test_setup_state_saved(self, server):
        _setup_settlement(server)
        assert server.state is not None
        assert server.state["name"] == "Thornwall"
        assert server.state["phase"] == "founding"

    def test_setup_creates_fiction_nodes(self, server, fiction_store):
        _setup_settlement(server)
        roots = fiction_store.get_roots()
        assert len(roots) == 1
        assert roots[0].title == "Thornwall"

    def test_setup_twice_fails(self, server):
        _setup_settlement(server)
        result = server.call_tool("ex_novo_setup", {"name": "Riverdale"})
        assert not result.success
        assert "already active" in (result.error or "")


# ===========================================================================
# Factions and Districts
# ===========================================================================


class TestFactions:
    def test_add_faction(self, server):
        _setup_settlement(server)
        result = server.call_tool("add_faction", {"name": "The Merchants Guild"})
        assert result.success
        assert "Merchants Guild" in result.data

    def test_faction_stored(self, server):
        _setup_settlement(server)
        server.call_tool("add_faction", {"name": "The Merchants Guild", "power_tokens": 3})
        assert len(server.state["factions"]) == 1
        assert server.state["factions"][0]["name"] == "The Merchants Guild"
        assert server.state["factions"][0]["power_tokens"] == 3

    def test_add_faction_no_settlement(self, server):
        result = server.call_tool("add_faction", {"name": "Test"})
        assert not result.success

    def test_multiple_factions(self, server):
        _setup_settlement(server)
        server.call_tool("add_faction", {"name": "Guild A"})
        server.call_tool("add_faction", {"name": "Guild B"})
        assert len(server.state["factions"]) == 2


class TestDistricts:
    def test_add_district(self, server):
        _setup_settlement(server)
        result = server.call_tool("add_district", {"name": "Market Quarter"})
        assert result.success
        assert "Market Quarter" in result.data

    def test_district_stored(self, server):
        _setup_settlement(server)
        server.call_tool("add_district", {"name": "Market Quarter", "citizen_tokens": 5})
        assert len(server.state["districts"]) == 1
        assert server.state["districts"][0]["citizen_tokens"] == 5

    def test_add_district_no_settlement(self, server):
        result = server.call_tool("add_district", {"name": "Test"})
        assert not result.success


class TestLandmarks:
    def test_add_landmark(self, server):
        _setup_settlement(server)
        server.call_tool("add_district", {"name": "Market Quarter"})
        result = server.call_tool("add_landmark", {
            "name": "Grand Bazaar",
            "district": "Market Quarter",
        })
        assert result.success
        assert "Grand Bazaar" in result.data

    def test_landmark_in_district(self, server):
        _setup_settlement(server)
        server.call_tool("add_district", {"name": "Market Quarter"})
        server.call_tool("add_landmark", {
            "name": "Grand Bazaar",
            "district": "Market Quarter",
        })
        district = server.state["districts"][0]
        assert "Grand Bazaar" in district["landmarks"]

    def test_landmark_in_fiction_tree(self, server, fiction_store):
        _setup_settlement(server)
        server.call_tool("add_landmark", {"name": "Old Tower"})
        # The root + founding node + landmark node
        root_id = server.state["history_root_id"]
        children = fiction_store.get_children(root_id)
        landmark_nodes = [n for n in children if n.title == "Old Tower"]
        assert len(landmark_nodes) == 1


# ===========================================================================
# Events
# ===========================================================================


class TestEvents:
    def test_roll_event(self, server):
        _setup_settlement(server)
        result = server.call_tool("roll_event", {})
        assert result.success
        assert "Event Roll" in result.data

    def test_roll_event_increments_turn(self, server):
        _setup_settlement(server)
        server.call_tool("roll_event", {})
        assert server.state["turn_number"] == 1
        server.call_tool("roll_event", {})
        assert server.state["turn_number"] == 2

    def test_roll_event_no_settlement(self, server):
        result = server.call_tool("roll_event", {})
        assert not result.success

    def test_resolve_event_add_resource(self, server):
        _setup_settlement(server)
        # Find an event with add_resource effect
        add_resource_code = None
        for code, entry in EVENT_TABLE.items():
            if entry["effect"] == "add_resource":
                add_resource_code = code
                break
        if add_resource_code is None:
            pytest.skip("No add_resource event in table")

        result = server.call_tool("resolve_event", {
            "event_code": add_resource_code,
            "narrative": "Found a vein of iron ore",
            "effect_target": "Iron",
        })
        assert result.success
        assert "Iron" in server.state["resources"]

    def test_resolve_event_add_problem(self, server):
        _setup_settlement(server)
        add_problem_code = None
        for code, entry in EVENT_TABLE.items():
            if entry["effect"] == "add_problem":
                add_problem_code = code
                break
        if add_problem_code is None:
            pytest.skip("No add_problem event in table")

        result = server.call_tool("resolve_event", {
            "event_code": add_problem_code,
            "narrative": "Disease spreads through the settlement",
            "effect_target": "Plague",
        })
        assert result.success
        assert "Plague" in server.state["problems"]

    def test_resolve_event_writes_fiction_node(self, server, fiction_store):
        _setup_settlement(server)
        code = next(iter(EVENT_TABLE.keys()))
        server.call_tool("resolve_event", {
            "event_code": code,
            "narrative": "Something happened",
        })
        root_id = server.state["history_root_id"]
        children = fiction_store.get_children(root_id)
        event_nodes = [n for n in children if n.zoom_level == "event"]
        assert len(event_nodes) >= 1

    def test_resolve_event_no_settlement(self, server):
        result = server.call_tool("resolve_event", {
            "event_code": 111,
            "narrative": "Test",
        })
        assert not result.success


# ===========================================================================
# Phase Advancement
# ===========================================================================


class TestPhases:
    def test_advance_phase(self, server):
        _setup_settlement(server)
        assert server.state["phase"] == "founding"
        result = server.call_tool("advance_phase", {})
        assert result.success
        assert server.state["phase"] == "development"

    def test_advance_to_topping_out(self, server):
        _setup_settlement(server)
        server.call_tool("advance_phase", {})
        result = server.call_tool("advance_phase", {})
        assert result.success
        assert server.state["phase"] == "topping_out"

    def test_advance_past_topping_out(self, server):
        _setup_settlement(server)
        server.call_tool("advance_phase", {})
        server.call_tool("advance_phase", {})
        result = server.call_tool("advance_phase", {})
        assert result.success
        assert "Already at final phase" in result.data

    def test_advance_no_settlement(self, server):
        result = server.call_tool("advance_phase", {})
        assert not result.success


# ===========================================================================
# State and History
# ===========================================================================


class TestState:
    def test_get_state(self, server):
        _setup_settlement(server)
        result = server.call_tool("get_settlement_state", {})
        assert result.success
        assert "Thornwall" in result.data

    def test_get_state_shows_factions(self, server):
        _setup_settlement(server)
        server.call_tool("add_faction", {"name": "The Watch"})
        result = server.call_tool("get_settlement_state", {})
        assert "The Watch" in result.data

    def test_get_state_no_settlement(self, server):
        result = server.call_tool("get_settlement_state", {})
        assert result.success
        assert "No active" in result.data

    def test_browse_history(self, server):
        _setup_settlement(server)
        result = server.call_tool("browse_settlement_history", {})
        assert result.success

    def test_browse_history_no_settlement(self, server):
        result = server.call_tool("browse_settlement_history", {})
        assert not result.success


# ===========================================================================
# End Settlement
# ===========================================================================


class TestEnd:
    def test_end_settlement(self, server):
        _setup_settlement(server)
        server.call_tool("add_faction", {"name": "Guild A"})
        server.call_tool("add_district", {"name": "Harbor"})
        result = server.call_tool("end_settlement", {})
        assert result.success
        assert "Finalized" in result.data
        assert server.state["phase"] == "ended"

    def test_end_no_settlement(self, server):
        result = server.call_tool("end_settlement", {})
        assert result.success
        assert "No active" in result.data

    def test_can_restart_after_end(self, server):
        _setup_settlement(server)
        server.call_tool("end_settlement", {})
        result = server.call_tool("ex_novo_setup", {"name": "New Town"})
        assert result.success


# ===========================================================================
# Unknown tool
# ===========================================================================


class TestUnknown:
    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.error
