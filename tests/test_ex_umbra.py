"""Tests for ExUmbraSystem and DungeonServer."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gm_agent.systems import get_system, SYSTEM_REGISTRY
from gm_agent.systems.ex_umbra.servers import DungeonServer
from gm_agent.systems.ex_umbra.tables import (
    TREMOR_TABLE, DIFFICULTY_SETTINGS,
    ADJECTIVE_PAIRS, ARCHITECTURE_TYPES,
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
    (d / "umbra-test").mkdir()
    return d


@pytest.fixture
def server(campaigns_dir: Path) -> DungeonServer:
    with patch("gm_agent.systems.ex_umbra.servers.CAMPAIGNS_DIR", campaigns_dir):
        srv = DungeonServer("umbra-test")
        yield srv
        srv.close()


@pytest.fixture
def fiction_store(campaigns_dir: Path) -> FictionTreeStore:
    s = FictionTreeStore("umbra-test", base_dir=campaigns_dir)
    yield s
    s.close()


def _setup_dungeon(server: DungeonServer, **overrides) -> dict:
    args = {
        "dungeon_name": "Shadow Crypts",
        "size": 6,
        "difficulty": 2,
        "aspects": '["undead", "traps"]',
    }
    args.update(overrides)
    return server.call_tool("ex_umbra_setup", args)


# ===========================================================================
# System Registration
# ===========================================================================


class TestExUmbraSystem:
    def test_registered(self):
        assert "ex_umbra" in SYSTEM_REGISTRY

    def test_metadata(self):
        system = get_system("ex_umbra")
        assert system.name == "ex_umbra"
        assert system.display_name == "Ex Umbra"
        assert system.category == "generation"

    def test_prompt(self):
        system = get_system("ex_umbra")
        prompt = system.system_prompt()
        assert "Ex Umbra" in prompt

    def test_servers_no_campaign(self):
        system = get_system("ex_umbra")
        assert system.servers({}) == []

    def test_servers_with_campaign(self, campaigns_dir):
        system = get_system("ex_umbra")
        with patch("gm_agent.systems.ex_umbra.servers.CAMPAIGNS_DIR", campaigns_dir):
            servers = system.servers({"campaign_id": "umbra-test"})
            assert len(servers) == 1
            assert isinstance(servers[0], DungeonServer)


# ===========================================================================
# Tables
# ===========================================================================


class TestTables:
    def test_tremor_table_structure(self):
        # Check that whatever entries exist have the correct schema
        assert len(TREMOR_TABLE) > 0
        for i, entry in TREMOR_TABLE.items():
            assert isinstance(i, int)
            assert "text" in entry
            assert "effect" in entry

    def test_difficulty_settings(self):
        assert len(DIFFICULTY_SETTINGS) > 0
        for i, setting in DIFFICULTY_SETTINGS.items():
            assert isinstance(i, int)
            assert "name" in setting
            assert "threat_mult" in setting
            assert "reward_mult" in setting

    def test_adjective_pairs(self):
        assert len(ADJECTIVE_PAIRS) > 0
        for pair in ADJECTIVE_PAIRS:
            assert len(pair) == 2

    def test_architecture_types(self):
        assert len(ARCHITECTURE_TYPES) > 0
        for atype in ARCHITECTURE_TYPES:
            assert isinstance(atype, str)


# ===========================================================================
# Tool Listing
# ===========================================================================


class TestToolListing:
    def test_list_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "ex_umbra_setup" in names
        assert "set_foundation" in names
        assert "play_dungeon_card" in names
        assert "spend_tokens" in names
        assert "exploration_turn" in names
        assert "tremor_turn" in names
        assert "heart_turn" in names
        assert "get_dungeon_state" in names
        assert "cleanup" in names
        assert "end_dungeon" in names
        assert "add_virtual_player" in names
        assert "virtual_player_turn" in names
        assert "remove_virtual_player" in names
        assert len(names) == 13


# ===========================================================================
# Setup
# ===========================================================================


class TestSetup:
    def test_setup(self, server):
        result = _setup_dungeon(server)
        assert result.success
        assert "Shadow Crypts" in result.data

    def test_setup_state(self, server):
        _setup_dungeon(server)
        s = server.state
        assert s["dungeon_name"] == "Shadow Crypts"
        assert s["size"] == 6
        assert s["difficulty"] == 2
        assert s["aspects"] == ["undead", "traps"]
        assert s["phase"] == "planning"

    def test_setup_pools(self, server):
        _setup_dungeon(server)
        s = server.state
        # size=6, difficulty=2 → threat_mult=1.0, reward_mult=1.0
        # threat = int(6 * 2 * 1.0) = 12
        assert s["dungeon_pool"]["threat"] == 12
        assert s["dungeon_pool"]["reward"] == 12
        assert s["heart_pool"]["threat"] == 0
        assert s["heart_pool"]["reward"] == 0

    def test_setup_turn_clock(self, server):
        _setup_dungeon(server)
        # turn_clock = size * 2 = 12
        assert server.state["turn_clock"] == 12
        assert server.state["turns_remaining"] == 12

    def test_setup_creates_fiction_node(self, server, fiction_store):
        _setup_dungeon(server)
        roots = fiction_store.get_roots()
        assert len(roots) == 1
        assert roots[0].title == "Shadow Crypts"

    def test_setup_twice_fails(self, server):
        _setup_dungeon(server)
        result = _setup_dungeon(server)
        assert not result.success
        assert "already active" in (result.error or "")

    def test_setup_validates_size(self, server):
        result = server.call_tool("ex_umbra_setup", {
            "dungeon_name": "Test",
            "size": 20,
            "difficulty": 1,
            "aspects": '["test"]',
        })
        assert not result.success
        assert "Size" in (result.error or "")

    def test_setup_validates_difficulty(self, server):
        result = server.call_tool("ex_umbra_setup", {
            "dungeon_name": "Test",
            "size": 4,
            "difficulty": 5,
            "aspects": '["test"]',
        })
        assert not result.success
        assert "Difficulty" in (result.error or "")

    def test_setup_validates_aspects(self, server):
        result = server.call_tool("ex_umbra_setup", {
            "dungeon_name": "Test",
            "size": 4,
            "difficulty": 1,
            "aspects": "[]",
        })
        assert not result.success
        assert "aspect" in (result.error or "").lower()

    def test_higher_difficulty_more_tokens(self, server, campaigns_dir):
        # Difficulty 4 should produce more tokens than difficulty 2
        with patch("gm_agent.systems.ex_umbra.servers.CAMPAIGNS_DIR", campaigns_dir):
            srv4 = DungeonServer("umbra-test")
            result = srv4.call_tool("ex_umbra_setup", {
                "dungeon_name": "Hard Dungeon",
                "size": 6,
                "difficulty": 4,
                "aspects": '["fire"]',
            })
            assert result.success
            # difficulty=4, threat_mult=2.0 → int(6*2*2.0) = 24
            assert srv4.state["dungeon_pool"]["threat"] == 24
            srv4.close()


# ===========================================================================
# Foundation
# ===========================================================================


class TestFoundation:
    def test_set_foundation(self, server):
        _setup_dungeon(server)
        result = server.call_tool("set_foundation", {
            "guide_lines": '["Main corridor runs north-south", "Three branching halls"]',
            "initial_rooms": '[{"name": "Entrance Hall", "type": "hall"}]',
        })
        assert result.success
        assert server.state["phase"] == "foundation"
        assert len(server.state["guide_lines"]) == 2
        assert len(server.state["architecture"]) == 1

    def test_set_foundation_no_dungeon(self, server):
        result = server.call_tool("set_foundation", {
            "guide_lines": "[]",
            "initial_rooms": "[]",
        })
        assert not result.success


# ===========================================================================
# Dungeon Cards
# ===========================================================================


class TestDungeonCards:
    def test_play_card(self, server):
        _setup_dungeon(server)
        result = server.call_tool("play_dungeon_card", {
            "aspect": "undead",
            "room_name": "Bone Gallery",
            "room_type": "gallery",
            "description": "A long hall lined with ossuary niches",
        })
        assert result.success
        assert "Bone Gallery" in result.data
        assert len(server.state["architecture"]) == 1

    def test_play_card_moves_tokens(self, server):
        _setup_dungeon(server)
        initial_threat = server.state["dungeon_pool"]["threat"]
        initial_reward = server.state["dungeon_pool"]["reward"]
        server.call_tool("play_dungeon_card", {
            "aspect": "undead",
            "room_name": "Chamber",
            "room_type": "chamber",
        })
        # Each card moves 1 threat and 1 reward to heart pool
        assert server.state["dungeon_pool"]["threat"] == initial_threat - 1
        assert server.state["dungeon_pool"]["reward"] == initial_reward - 1
        assert server.state["heart_pool"]["threat"] == 1
        assert server.state["heart_pool"]["reward"] == 1

    def test_play_card_writes_fiction_node(self, server, fiction_store):
        _setup_dungeon(server)
        server.call_tool("play_dungeon_card", {
            "aspect": "traps",
            "room_name": "Trap Hall",
            "room_type": "corridor",
        })
        root_id = server.state["history_root_id"]
        children = fiction_store.get_children(root_id)
        assert any(n.title == "Trap Hall" for n in children)

    def test_play_card_no_dungeon(self, server):
        result = server.call_tool("play_dungeon_card", {
            "aspect": "test",
            "room_name": "Test",
            "room_type": "chamber",
        })
        assert not result.success


# ===========================================================================
# Token Spending
# ===========================================================================


class TestTokenSpending:
    def test_spend_threat_tokens(self, server):
        _setup_dungeon(server)
        result = server.call_tool("spend_tokens", {
            "pool": "dungeon",
            "token_type": "threat",
            "amount": 3,
            "target": "Bone Gallery",
            "description": "Skeleton guardians",
        })
        assert result.success
        assert len(server.state["threats"]) == 1
        assert server.state["threats"][0]["tokens"] == 3

    def test_spend_reward_tokens(self, server):
        _setup_dungeon(server)
        result = server.call_tool("spend_tokens", {
            "pool": "dungeon",
            "token_type": "reward",
            "amount": 2,
            "target": "Treasure Room",
        })
        assert result.success
        assert len(server.state["rewards"]) == 1

    def test_spend_too_many_tokens(self, server):
        _setup_dungeon(server)
        result = server.call_tool("spend_tokens", {
            "pool": "dungeon",
            "token_type": "threat",
            "amount": 999,
            "target": "Test",
        })
        assert not result.success
        assert "Not enough" in (result.error or "")

    def test_spend_invalid_pool(self, server):
        _setup_dungeon(server)
        result = server.call_tool("spend_tokens", {
            "pool": "invalid",
            "token_type": "threat",
            "amount": 1,
            "target": "Test",
        })
        assert not result.success

    def test_spend_invalid_token_type(self, server):
        _setup_dungeon(server)
        result = server.call_tool("spend_tokens", {
            "pool": "dungeon",
            "token_type": "invalid",
            "amount": 1,
            "target": "Test",
        })
        assert not result.success

    def test_spend_no_dungeon(self, server):
        result = server.call_tool("spend_tokens", {
            "pool": "dungeon",
            "token_type": "threat",
            "amount": 1,
            "target": "Test",
        })
        assert not result.success


# ===========================================================================
# Exploration Turns
# ===========================================================================


class TestExplorationTurn:
    def test_exploration_turn(self, server):
        _setup_dungeon(server)
        result = server.call_tool("exploration_turn", {
            "description": "The party ventures deeper into the crypts",
        })
        assert result.success
        assert server.state["turns_remaining"] == 11

    def test_exploration_turn_advances_phase(self, server):
        _setup_dungeon(server)
        server.call_tool("exploration_turn", {"description": "Exploring"})
        assert server.state["phase"] == "discovery"

    def test_exploration_turn_exhausted(self, server):
        _setup_dungeon(server, size=2)
        # turn_clock = 4 for size=2
        for i in range(4):
            server.call_tool("exploration_turn", {"description": f"Turn {i+1}"})
        result = server.call_tool("exploration_turn", {"description": "One more"})
        assert not result.success
        assert "No exploration turns" in (result.error or "")

    def test_exploration_turn_no_dungeon(self, server):
        result = server.call_tool("exploration_turn", {"description": "Test"})
        assert not result.success


# ===========================================================================
# Tremor Turns
# ===========================================================================


class TestTremorTurn:
    def test_tremor_turn(self, server):
        _setup_dungeon(server)
        result = server.call_tool("tremor_turn", {})
        assert result.success
        assert "Tremor" in result.data
        assert "d20" in result.data.lower()

    def test_tremor_decrements_remaining(self, server):
        _setup_dungeon(server)
        initial = server.state["tremors_remaining"]
        server.call_tool("tremor_turn", {})
        assert server.state["tremors_remaining"] == initial - 1

    def test_tremor_exhausted(self, server):
        _setup_dungeon(server, size=2)
        # Use up all tremors
        while server.state["tremors_remaining"] > 0:
            server.call_tool("tremor_turn", {})
        result = server.call_tool("tremor_turn", {})
        assert not result.success
        assert "No tremors" in (result.error or "")

    def test_tremor_no_dungeon(self, server):
        result = server.call_tool("tremor_turn", {})
        assert not result.success


# ===========================================================================
# Heart Turn
# ===========================================================================


class TestHeartTurn:
    def test_heart_turn(self, server):
        _setup_dungeon(server)
        # Play a card to move tokens to heart pool
        server.call_tool("play_dungeon_card", {
            "aspect": "undead",
            "room_name": "Antechamber",
            "room_type": "chamber",
        })
        result = server.call_tool("heart_turn", {
            "name": "The Heart of Darkness",
            "description": "A pulsing core of necrotic energy",
        })
        assert result.success
        assert "Heart Chamber" in result.data
        assert server.state["heart_pool"]["threat"] == 0
        assert server.state["heart_pool"]["reward"] == 0
        assert server.state["phase"] == "cleanup"

    def test_heart_turn_creates_threats_and_rewards(self, server):
        _setup_dungeon(server)
        # Play multiple cards to build heart pool
        for i in range(3):
            server.call_tool("play_dungeon_card", {
                "aspect": "undead",
                "room_name": f"Room {i}",
                "room_type": "chamber",
            })
        server.call_tool("heart_turn", {
            "name": "Dark Heart",
            "description": "The core",
        })
        # Should have threats and rewards from heart pool
        heart_threats = [t for t in server.state["threats"] if t["location"] == "Dark Heart"]
        heart_rewards = [r for r in server.state["rewards"] if r["location"] == "Dark Heart"]
        assert len(heart_threats) == 1
        assert len(heart_rewards) == 1

    def test_heart_turn_writes_fiction_node(self, server, fiction_store):
        _setup_dungeon(server)
        server.call_tool("play_dungeon_card", {
            "aspect": "traps",
            "room_name": "Hall",
            "room_type": "hall",
        })
        server.call_tool("heart_turn", {
            "name": "The Core",
            "description": "Ancient power resides here",
        })
        root_id = server.state["history_root_id"]
        children = fiction_store.get_children(root_id)
        assert any(n.title == "The Core" for n in children)

    def test_heart_turn_no_dungeon(self, server):
        result = server.call_tool("heart_turn", {
            "name": "Test",
            "description": "Test",
        })
        assert not result.success


# ===========================================================================
# Dungeon State
# ===========================================================================


class TestGetState:
    def test_get_state(self, server):
        _setup_dungeon(server)
        result = server.call_tool("get_dungeon_state", {})
        assert result.success
        assert "Shadow Crypts" in result.data
        assert "Dungeon Pool" in result.data

    def test_get_state_shows_architecture(self, server):
        _setup_dungeon(server)
        server.call_tool("play_dungeon_card", {
            "aspect": "undead",
            "room_name": "Bone Gallery",
            "room_type": "gallery",
        })
        result = server.call_tool("get_dungeon_state", {})
        assert "Bone Gallery" in result.data

    def test_get_state_shows_threats(self, server):
        _setup_dungeon(server)
        server.call_tool("spend_tokens", {
            "pool": "dungeon",
            "token_type": "threat",
            "amount": 2,
            "target": "Guardian",
        })
        result = server.call_tool("get_dungeon_state", {})
        assert "Guardian" in result.data

    def test_get_state_no_dungeon(self, server):
        result = server.call_tool("get_dungeon_state", {})
        assert result.success
        assert "No active" in result.data


# ===========================================================================
# Cleanup and End
# ===========================================================================


class TestCleanup:
    def test_cleanup(self, server):
        _setup_dungeon(server)
        result = server.call_tool("cleanup", {
            "notes": "Connected all dead ends",
        })
        assert result.success
        assert server.state["phase"] == "cleanup"

    def test_cleanup_no_dungeon(self, server):
        result = server.call_tool("cleanup", {})
        assert not result.success


class TestEnd:
    def test_end_dungeon(self, server):
        _setup_dungeon(server)
        result = server.call_tool("end_dungeon", {})
        assert result.success
        assert "Finalized" in result.data
        assert server.state["phase"] == "ended"

    def test_end_no_dungeon(self, server):
        result = server.call_tool("end_dungeon", {})
        assert result.success
        assert "No active" in result.data

    def test_can_restart_after_end(self, server):
        _setup_dungeon(server)
        server.call_tool("end_dungeon", {})
        result = _setup_dungeon(server, dungeon_name="New Dungeon")
        assert result.success


# ===========================================================================
# Unknown tool
# ===========================================================================


class TestUnknown:
    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.error
