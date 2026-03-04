"""Tests for MicroscopeSystem and MicroscopeSessionServer."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gm_agent.systems import get_system, SYSTEM_REGISTRY
from gm_agent.systems.microscope.servers import MicroscopeSessionServer
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
    (d / "micro-test").mkdir()
    return d


@pytest.fixture
def server(campaigns_dir: Path) -> MicroscopeSessionServer:
    with patch("gm_agent.systems.microscope.servers.CAMPAIGNS_DIR", campaigns_dir):
        srv = MicroscopeSessionServer("micro-test")
        yield srv
        srv.close()


@pytest.fixture
def fiction_store(campaigns_dir: Path) -> FictionTreeStore:
    s = FictionTreeStore("micro-test", base_dir=campaigns_dir)
    yield s
    s.close()


def _setup_session(server: MicroscopeSessionServer) -> dict:
    """Helper: run setup and return the result."""
    result = server.call_tool("microscope_setup", {
        "big_picture": "The rise and fall of a galactic empire",
        "start_period": "The First Colony",
        "start_tone": "light",
        "end_period": "The Final Silence",
        "end_tone": "dark",
        "players": '["Alice", "Bob", "Charlie"]',
    })
    return result


# ===========================================================================
# System Registration
# ===========================================================================


class TestMicroscopeSystem:
    def test_registered(self):
        assert "microscope" in SYSTEM_REGISTRY

    def test_metadata(self):
        system = get_system("microscope")
        assert system.name == "microscope"
        assert system.display_name == "Microscope"
        assert system.category == "generation"

    def test_prompt(self):
        system = get_system("microscope")
        prompt = system.system_prompt()
        assert "Microscope" in prompt
        assert "Big Picture" in prompt

    def test_servers_no_campaign(self):
        system = get_system("microscope")
        servers = system.servers({"campaign_id": None})
        assert servers == []

    def test_servers_with_campaign(self, campaigns_dir):
        with patch("gm_agent.systems.microscope.servers.CAMPAIGNS_DIR", campaigns_dir):
            system = get_system("microscope")
            servers = system.servers({"campaign_id": "micro-test"})
            assert len(servers) == 1
            assert isinstance(servers[0], MicroscopeSessionServer)


# ===========================================================================
# Tool listing
# ===========================================================================


class TestMicroscopeTools:
    def test_list_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "microscope_setup" in names
        assert "set_palette" in names
        assert "set_focus" in names
        assert "create_period" in names
        assert "create_event" in names
        assert "create_scene" in names
        assert "resolve_scene" in names
        assert "add_legacy" in names
        assert "browse_history" in names
        assert "get_microscope_state" in names
        assert "end_microscope" in names
        assert len(names) == 25
        # New tools
        assert "list_seeds" in names
        assert "use_seed" in names
        assert "consult_oracle" in names
        assert "start_chronicle" in names
        assert "chronicle_entry" in names
        assert "start_echo" in names
        assert "resolve_echo" in names
        assert "start_union" in names
        assert "union_moment" in names
        assert "end_minigame" in names


# ===========================================================================
# Session Lifecycle
# ===========================================================================


class TestMicroscopeSetup:
    def test_setup_creates_session(self, server):
        result = _setup_session(server)
        assert result.success
        assert "galactic empire" in result.data
        assert "First Colony" in result.data
        assert "Final Silence" in result.data

    def test_setup_creates_fiction_nodes(self, server, fiction_store):
        _setup_session(server)
        roots = fiction_store.get_roots()
        assert len(roots) == 1
        assert "galactic empire" in roots[0].title

        periods = fiction_store.get_children(roots[0].id)
        assert len(periods) == 2
        assert periods[0].title == "The First Colony"
        assert periods[0].tone == "light"
        assert periods[1].title == "The Final Silence"
        assert periods[1].tone == "dark"

    def test_setup_twice_fails(self, server):
        _setup_session(server)
        result = _setup_session(server)
        assert not result.success
        assert "already active" in (result.error or "")

    def test_state_after_setup(self, server):
        _setup_session(server)
        result = server.call_tool("get_microscope_state", {})
        assert result.success
        assert "setup" in result.data
        assert "galactic empire" in result.data


# ===========================================================================
# Palette
# ===========================================================================


class TestMicroscopePalette:
    def test_add_yes(self, server):
        _setup_session(server)
        result = server.call_tool("set_palette", {"entry": "Aliens exist", "allowed": True})
        assert result.success
        assert "YES" in result.data

    def test_add_no(self, server):
        _setup_session(server)
        result = server.call_tool("set_palette", {"entry": "No time travel", "allowed": False})
        assert result.success
        assert "NO" in result.data

    def test_palette_in_state(self, server):
        _setup_session(server)
        server.call_tool("set_palette", {"entry": "FTL travel", "allowed": True})
        server.call_tool("set_palette", {"entry": "No magic", "allowed": False})
        result = server.call_tool("get_microscope_state", {})
        assert "FTL travel" in result.data
        assert "No magic" in result.data

    def test_no_session(self, server):
        result = server.call_tool("set_palette", {"entry": "X", "allowed": True})
        assert not result.success


# ===========================================================================
# Focus
# ===========================================================================


class TestMicroscopeFocus:
    def test_set_focus(self, server):
        _setup_session(server)
        result = server.call_tool("set_focus", {"focus": "Technology"})
        assert result.success
        assert "Technology" in result.data

    def test_focus_advances_phase(self, server):
        _setup_session(server)
        server.call_tool("set_focus", {"focus": "Technology"})
        assert server.state["phase"] == "play"

    def test_no_session(self, server):
        result = server.call_tool("set_focus", {"focus": "X"})
        assert not result.success


# ===========================================================================
# Create Period
# ===========================================================================


class TestMicroscopeCreatePeriod:
    def test_create_period(self, server, fiction_store):
        _setup_session(server)
        root_id = server.state["history_root_id"]

        result = server.call_tool("create_period", {
            "title": "The Great Expansion",
            "tone": "light",
            "description": "Humanity spreads across the stars.",
        })
        assert result.success
        assert "Great Expansion" in result.data

        # Should now have 3 periods
        periods = fiction_store.get_children(root_id)
        assert len(periods) == 3

    def test_create_period_advances_turn(self, server):
        _setup_session(server)
        initial_turn = server.state["current_turn"]
        server.call_tool("create_period", {
            "title": "New Period",
            "tone": "dark",
        })
        assert server.state["current_turn"] == initial_turn + 1

    def test_invalid_tone(self, server):
        _setup_session(server)
        result = server.call_tool("create_period", {
            "title": "X",
            "tone": "neutral",
        })
        assert not result.success

    def test_no_session(self, server):
        result = server.call_tool("create_period", {"title": "X", "tone": "light"})
        assert not result.success


# ===========================================================================
# Create Event
# ===========================================================================


class TestMicroscopeCreateEvent:
    def test_create_event(self, server, fiction_store):
        _setup_session(server)
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        period_id = periods[0].id

        result = server.call_tool("create_event", {
            "period_id": period_id,
            "title": "First Contact",
            "tone": "light",
            "description": "Humanity meets the Elari.",
        })
        assert result.success
        assert "First Contact" in result.data

        events = fiction_store.get_children(period_id)
        assert len(events) == 1
        assert events[0].title == "First Contact"

    def test_invalid_period(self, server):
        _setup_session(server)
        result = server.call_tool("create_event", {
            "period_id": "nonexistent",
            "title": "X",
            "tone": "light",
        })
        assert not result.success

    def test_invalid_tone(self, server):
        _setup_session(server)
        result = server.call_tool("create_event", {
            "period_id": "x",
            "title": "X",
            "tone": "mixed",
        })
        assert not result.success


# ===========================================================================
# Create / Resolve Scene
# ===========================================================================


class TestMicroscopeScene:
    def _setup_event(self, server, fiction_store):
        _setup_session(server)
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        period_id = periods[0].id
        server.call_tool("create_event", {
            "period_id": period_id,
            "title": "The Meeting",
            "tone": "light",
        })
        events = fiction_store.get_children(period_id)
        return events[0].id

    def test_create_scene(self, server, fiction_store):
        event_id = self._setup_event(server, fiction_store)
        result = server.call_tool("create_scene", {
            "event_id": event_id,
            "question": "Did the ambassador survive?",
            "setting": "The orbital station",
            "characters": '["Ambassador Kira", "Captain Sol"]',
        })
        assert result.success
        assert "survive" in result.data

    def test_resolve_scene(self, server, fiction_store):
        event_id = self._setup_event(server, fiction_store)
        create_result = server.call_tool("create_scene", {
            "event_id": event_id,
            "question": "Did the treaty hold?",
        })
        scene_id = create_result.data.split("ID:** ")[1].split("\n")[0].strip()

        result = server.call_tool("resolve_scene", {
            "scene_id": scene_id,
            "answer": "Yes, but at a great cost.",
            "description": "The treaty was signed in blood.",
        })
        assert result.success
        assert "great cost" in result.data

    def test_invalid_event(self, server):
        _setup_session(server)
        result = server.call_tool("create_scene", {
            "event_id": "nonexistent",
            "question": "X",
        })
        assert not result.success

    def test_resolve_invalid_scene(self, server):
        _setup_session(server)
        result = server.call_tool("resolve_scene", {
            "scene_id": "nonexistent",
            "answer": "X",
        })
        assert not result.success


# ===========================================================================
# Legacy
# ===========================================================================


class TestMicroscopeLegacy:
    def test_add_legacy(self, server):
        _setup_session(server)
        result = server.call_tool("add_legacy", {
            "name": "The Void Crown",
            "legacy_type": "institution",
            "owner": "Alice",
        })
        assert result.success
        assert "Void Crown" in result.data

    def test_update_legacy(self, server):
        _setup_session(server)
        server.call_tool("add_legacy", {"name": "X", "legacy_type": "character"})
        result = server.call_tool("add_legacy", {"name": "X", "legacy_type": "place", "owner": "Bob"})
        assert result.success
        assert "Updated" in result.data

    def test_legacy_in_state(self, server):
        _setup_session(server)
        server.call_tool("add_legacy", {"name": "The Library", "legacy_type": "place"})
        result = server.call_tool("get_microscope_state", {})
        assert "Library" in result.data


# ===========================================================================
# Browse History
# ===========================================================================


class TestMicroscopeBrowseHistory:
    def test_browse_empty(self, server):
        # No session
        result = server.call_tool("browse_history", {})
        assert not result.success

    def test_browse_after_setup(self, server):
        _setup_session(server)
        result = server.call_tool("browse_history", {"expand": False})
        assert result.success
        assert "First Colony" in result.data
        assert "Final Silence" in result.data

    def test_browse_expanded(self, server, fiction_store):
        _setup_session(server)
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        server.call_tool("create_event", {
            "period_id": periods[0].id,
            "title": "The Discovery",
            "tone": "light",
        })
        result = server.call_tool("browse_history", {"expand": True})
        assert result.success
        assert "Discovery" in result.data


# ===========================================================================
# End Session
# ===========================================================================


class TestMicroscopeEnd:
    def test_end_session(self, server):
        _setup_session(server)
        result = server.call_tool("end_microscope", {})
        assert result.success
        assert "Ended" in result.data
        assert server.state["phase"] == "ended"

    def test_end_no_session(self, server):
        result = server.call_tool("end_microscope", {})
        assert result.success

    def test_can_restart_after_end(self, server):
        _setup_session(server)
        server.call_tool("end_microscope", {})
        # Should be able to start a new session
        result = server.call_tool("microscope_setup", {
            "big_picture": "A new history",
            "start_period": "Beginning",
            "start_tone": "light",
            "end_period": "End",
            "end_tone": "dark",
        })
        assert result.success


# ===========================================================================
# Unknown tool
# ===========================================================================


# ===========================================================================
# Seeds
# ===========================================================================


class TestMicroscopeSeeds:
    def test_list_seeds(self, server):
        from gm_agent.systems.microscope.seeds import SEEDS
        result = server.call_tool("list_seeds", {})
        assert result.success
        # Check that whatever seeds are loaded appear in the listing
        for key in SEEDS:
            assert key in result.data

    def test_use_seed(self, server, fiction_store):
        from gm_agent.systems.microscope.seeds import SEEDS
        if not SEEDS:
            return  # No seeds loaded (empty placeholder)
        seed_key = next(iter(SEEDS))
        seed = SEEDS[seed_key]
        result = server.call_tool("use_seed", {
            "seed_name": seed_key,
            "players": '["Alice", "Bob"]',
        })
        assert result.success
        assert "Seed" in result.data

        # Verify session was created with the seed's content
        assert server.state is not None
        assert server.state["big_picture"] == seed.big_picture
        assert server.state["current_focus"] == seed.suggested_focus

    def test_use_seed_creates_fiction_nodes(self, server, fiction_store):
        from gm_agent.systems.microscope.seeds import SEEDS
        if not SEEDS:
            return  # No seeds loaded (empty placeholder)
        # Use the second seed if available, otherwise the first
        keys = list(SEEDS)
        seed_key = keys[1] if len(keys) > 1 else keys[0]
        seed = SEEDS[seed_key]
        server.call_tool("use_seed", {"seed_name": seed_key})
        roots = fiction_store.get_roots()
        assert len(roots) >= 1
        periods = fiction_store.get_children(roots[-1].id)
        titles = [p.title for p in periods]
        assert seed.start_period in titles
        assert seed.end_period in titles

    def test_use_unknown_seed(self, server):
        result = server.call_tool("use_seed", {"seed_name": "nonexistent"})
        assert not result.success
        assert "Unknown seed" in (result.error or "")

    def test_use_seed_twice_fails(self, server):
        from gm_agent.systems.microscope.seeds import SEEDS
        if not SEEDS:
            return
        keys = list(SEEDS)
        server.call_tool("use_seed", {"seed_name": keys[0]})
        second_key = keys[1] if len(keys) > 1 else keys[0]
        result = server.call_tool("use_seed", {"seed_name": second_key})
        assert not result.success
        assert "already active" in (result.error or "")


# ===========================================================================
# Oracles
# ===========================================================================


class TestMicroscopeOracles:
    def test_list_oracles(self, server):
        from gm_agent.systems.microscope.oracles import ORACLES
        result = server.call_tool("consult_oracle", {"oracle_name": ""})
        assert result.success
        for key in ORACLES:
            assert key in result.data

    def test_consult_oracle(self, server):
        from gm_agent.systems.microscope.oracles import ORACLES
        if not ORACLES:
            return
        oracle_key = next(iter(ORACLES))
        oracle = ORACLES[oracle_key]
        result = server.call_tool("consult_oracle", {"oracle_name": oracle_key})
        assert result.success
        assert oracle.name in result.data

    def test_consult_unknown_oracle(self, server):
        result = server.call_tool("consult_oracle", {"oracle_name": "nonexistent"})
        assert not result.success
        assert "Unknown oracle" in (result.error or "")

    def test_consult_each_oracle(self, server):
        from gm_agent.systems.microscope.oracles import ORACLES
        for key in ORACLES:
            result = server.call_tool("consult_oracle", {"oracle_name": key})
            assert result.success, f"Oracle {key} failed"


# ===========================================================================
# Mini-game: Chronicle
# ===========================================================================


class TestMicroscopeChronicle:
    def test_start_chronicle(self, server):
        _setup_session(server)
        server.call_tool("add_legacy", {"name": "The Void Crown", "legacy_type": "institution"})
        result = server.call_tool("start_chronicle", {"legacy_name": "The Void Crown"})
        assert result.success
        assert "Void Crown" in result.data
        assert server.state["active_minigame"] == "chronicle"

    def test_chronicle_entry(self, server, fiction_store):
        _setup_session(server)
        server.call_tool("start_chronicle", {"legacy_name": "The Void Crown"})
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        period_id = periods[0].id

        result = server.call_tool("chronicle_entry", {
            "period_id": period_id,
            "title": "The Crown is Forged",
            "tone": "light",
            "description": "A master smith creates the Crown.",
        })
        assert result.success
        assert "Crown is Forged" in result.data
        assert "chronicle" in result.data.lower() or "Legacy" in result.data

        # Verify fiction node was created with chronicle tag
        events = fiction_store.get_children(period_id)
        chronicle_events = [e for e in events if "chronicle:" in str(e.tags)]
        assert len(chronicle_events) == 1

    def test_chronicle_entry_no_active_chronicle(self, server, fiction_store):
        _setup_session(server)
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        result = server.call_tool("chronicle_entry", {
            "period_id": periods[0].id,
            "title": "X",
            "tone": "light",
        })
        assert not result.success
        assert "No active Chronicle" in (result.error or "")

    def test_start_chronicle_no_session(self, server):
        result = server.call_tool("start_chronicle", {"legacy_name": "X"})
        assert not result.success

    def test_end_chronicle(self, server):
        _setup_session(server)
        server.call_tool("start_chronicle", {"legacy_name": "X"})
        result = server.call_tool("end_minigame", {})
        assert result.success
        assert "Chronicle" in result.data or "chronicle" in result.data
        assert server.state.get("active_minigame") is None


# ===========================================================================
# Mini-game: Echo
# ===========================================================================


class TestMicroscopeEcho:
    def _create_scene(self, server, fiction_store):
        """Helper to create a scene for Echo tests."""
        _setup_session(server)
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        period_id = periods[0].id

        server.call_tool("create_event", {
            "period_id": period_id,
            "title": "The Meeting",
            "tone": "light",
        })
        events = fiction_store.get_children(period_id)
        event_id = events[0].id

        create_result = server.call_tool("create_scene", {
            "event_id": event_id,
            "question": "Did the ambassador survive?",
        })
        scene_id = create_result.data.split("ID:** ")[1].split("\n")[0].strip()
        return scene_id

    def test_start_echo(self, server, fiction_store):
        scene_id = self._create_scene(server, fiction_store)
        result = server.call_tool("start_echo", {
            "scene_id": scene_id,
            "new_perspective": "The ambassador's bodyguard",
        })
        assert result.success
        assert "Echo Started" in result.data
        assert "bodyguard" in result.data
        assert server.state["active_minigame"] == "echo"

    def test_resolve_echo(self, server, fiction_store):
        scene_id = self._create_scene(server, fiction_store)
        server.call_tool("start_echo", {
            "scene_id": scene_id,
            "new_perspective": "The enemy spy",
        })

        result = server.call_tool("resolve_echo", {
            "answer": "No, the ambassador was assassinated.",
            "description": "The spy's poison worked.",
        })
        assert result.success
        assert "Echo Resolved" in result.data
        assert server.state.get("active_minigame") is None

    def test_start_echo_no_session(self, server):
        result = server.call_tool("start_echo", {"scene_id": "x", "new_perspective": "y"})
        assert not result.success

    def test_start_echo_invalid_scene(self, server):
        _setup_session(server)
        result = server.call_tool("start_echo", {
            "scene_id": "nonexistent",
            "new_perspective": "Someone",
        })
        assert not result.success

    def test_resolve_echo_without_active(self, server):
        _setup_session(server)
        result = server.call_tool("resolve_echo", {"answer": "X"})
        assert not result.success
        assert "No active Echo" in (result.error or "")


# ===========================================================================
# Mini-game: Union
# ===========================================================================


class TestMicroscopeUnion:
    def test_start_union(self, server):
        _setup_session(server)
        result = server.call_tool("start_union", {
            "character_a": "Queen Aria",
            "character_b": "General Mox",
        })
        assert result.success
        assert "Queen Aria" in result.data
        assert "General Mox" in result.data
        assert server.state["active_minigame"] == "union"

    def test_union_moment(self, server, fiction_store):
        _setup_session(server)
        server.call_tool("start_union", {
            "character_a": "Queen Aria",
            "character_b": "General Mox",
        })
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        period_id = periods[0].id

        result = server.call_tool("union_moment", {
            "period_id": period_id,
            "title": "First Meeting",
            "tone": "light",
            "description": "They meet at the peace summit.",
        })
        assert result.success
        assert "First Meeting" in result.data

        # Verify fiction node created with union tag
        events = fiction_store.get_children(period_id)
        union_events = [e for e in events if "union:" in str(e.tags)]
        assert len(union_events) == 1

    def test_union_moment_no_active_union(self, server, fiction_store):
        _setup_session(server)
        root_id = server.state["history_root_id"]
        periods = fiction_store.get_children(root_id)
        result = server.call_tool("union_moment", {
            "period_id": periods[0].id,
            "title": "X",
            "tone": "light",
        })
        assert not result.success
        assert "No active Union" in (result.error or "")

    def test_start_union_no_session(self, server):
        result = server.call_tool("start_union", {
            "character_a": "A",
            "character_b": "B",
        })
        assert not result.success

    def test_start_union_missing_chars(self, server):
        _setup_session(server)
        result = server.call_tool("start_union", {"character_a": "A", "character_b": ""})
        assert not result.success

    def test_end_union(self, server):
        _setup_session(server)
        server.call_tool("start_union", {
            "character_a": "A",
            "character_b": "B",
        })
        result = server.call_tool("end_minigame", {})
        assert result.success
        assert server.state.get("active_minigame") is None


# ===========================================================================
# Mini-game guards
# ===========================================================================


class TestMicroscopeMinigameGuards:
    def test_no_overlap(self, server):
        """Cannot start a new mini-game while one is active."""
        _setup_session(server)
        server.call_tool("start_chronicle", {"legacy_name": "X"})
        result = server.call_tool("start_union", {
            "character_a": "A",
            "character_b": "B",
        })
        assert not result.success
        assert "already active" in (result.error or "")

    def test_end_no_minigame(self, server):
        _setup_session(server)
        result = server.call_tool("end_minigame", {})
        assert result.success
        assert "No active" in result.data

    def test_end_minigame_no_session(self, server):
        result = server.call_tool("end_minigame", {})
        assert not result.success


# ===========================================================================
# Unknown tool
# ===========================================================================


class TestMicroscopeUnknown:
    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent_tool", {})
        assert not result.success
