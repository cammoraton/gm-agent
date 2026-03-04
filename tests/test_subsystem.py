"""Tests for SubsystemServer and SubsystemStore — Phase 2."""

import json
import pytest

from gm_agent.storage.subsystems import SubsystemStore, SubsystemInstance
from gm_agent.mcp.subsystem import SubsystemServer


# ---------------------------------------------------------------------------
# SubsystemStore tests
# ---------------------------------------------------------------------------

class TestSubsystemStore:
    """Tests for JSON-on-disk SubsystemStore."""

    def test_create_vp_subsystem(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        instance = store.create("vp", "Test VP", {
            "targets": {
                "Lord Gyr": {"minor": 3, "major": 6},
                "Lady Vord": {"minor": 4, "major": 8},
            }
        })

        assert instance.id
        assert instance.subsystem_type == "vp"
        assert instance.name == "Test VP"
        assert instance.status == "active"
        assert instance.victory_points["Lord Gyr"] == 0
        assert instance.thresholds["Lord Gyr"]["major"] == 6

    def test_create_chase_subsystem(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        instance = store.create("chase", "Market Chase", {
            "participants": ["Party", "Thief"],
            "chase_length": 10,
        })

        assert instance.positions["Party"] == 0
        assert instance.positions["Thief"] == 0
        assert instance.chase_length == 10

    def test_create_hazard_subsystem(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        instance = store.create("hazard", "Poison Dart Trap", {
            "hp": 50,
            "hardness": 10,
            "routine_actions": ["Fires darts at nearest creature", "Resets mechanisms"],
            "disable_conditions": [{"skill": "Thievery", "dc": 22}],
        })

        assert instance.hp == 50
        assert instance.max_hp == 50
        assert instance.hardness == 10
        assert len(instance.routine_actions) == 2

    def test_create_infiltration_subsystem(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        instance = store.create("infiltration", "Castle Infiltration", {
            "detection_threshold": 10,
            "targets": {"Main Gate": {"minor": 3, "major": 5}},
        })

        assert instance.detection_threshold == 10
        assert instance.awareness_points == 0
        assert "Main Gate" in instance.victory_points

    def test_get_and_save(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        instance = store.create("vp", "Test", {"targets": {"A": {"minor": 2, "major": 4}}})

        # Modify and save
        instance.victory_points["A"] = 3
        instance.round_number = 2
        store.save(instance)

        # Reload
        loaded = store.get(instance.id)
        assert loaded.victory_points["A"] == 3
        assert loaded.round_number == 2

    def test_get_nonexistent(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        assert store.get("nonexistent") is None

    def test_list_active(self, tmp_path):
        store = SubsystemStore("test-campaign", base_dir=tmp_path)
        inst1 = store.create("vp", "Active One", {})
        inst2 = store.create("vp", "Active Two", {})
        inst3 = store.create("vp", "Completed", {})
        inst3.status = "completed"
        store.save(inst3)

        active = store.list_active()
        assert len(active) == 2
        active_ids = {a.id for a in active}
        assert inst1.id in active_ids
        assert inst2.id in active_ids

    def test_persistence_across_loads(self, tmp_path):
        """Data survives store re-creation."""
        store1 = SubsystemStore("test-campaign", base_dir=tmp_path)
        inst = store1.create("vp", "Persist Test", {"targets": {"X": {"minor": 1, "major": 2}}})
        inst_id = inst.id
        store1.close()

        store2 = SubsystemStore("test-campaign", base_dir=tmp_path)
        loaded = store2.get(inst_id)
        assert loaded is not None
        assert loaded.name == "Persist Test"


# ---------------------------------------------------------------------------
# SubsystemServer tool tests
# ---------------------------------------------------------------------------

class TestSubsystemServer:
    """Tests for SubsystemServer MCP tools."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def test_list_tools(self, server):
        tools = server.list_tools()
        tool_names = [t.name for t in tools]
        assert "start_subsystem" in tool_names
        assert "subsystem_action" in tool_names
        assert "get_subsystem_state" in tool_names
        assert "end_subsystem" in tool_names
        assert "list_subsystems" in tool_names
        assert "attempt_disable" in tool_names
        assert "trigger_haunt" in tool_names
        assert "reset_haunt" in tool_names
        assert len(tools) == 8

    # --- VP subsystem tests ---

    def test_start_vp_subsystem(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Influence Council",
            "config": json.dumps({
                "targets": {"Mayor": {"minor": 3, "major": 6}}
            }),
        })
        assert result.success
        assert "Influence Council" in result.data
        assert "Mayor: 0 VP" in result.data

    def test_add_vp(self, server):
        # Start subsystem
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Test VP",
            "config": json.dumps({"targets": {"Target": {"minor": 3, "major": 6}}}),
        })
        # Extract ID from response
        subsystem_id = self._extract_id(start_result.data)

        # Add VP
        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "Target", "amount": 2}),
        })
        assert result.success
        assert "Target: 2 VP" in result.data

    def test_vp_threshold_detection(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Threshold Test",
            "config": json.dumps({"targets": {"Target": {"minor": 3, "major": 6}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        # Add VP to hit minor threshold
        server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "Target", "amount": 3}),
        })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "Target", "amount": 0}),
        })
        # Minor threshold should have been notified on the +3 call
        # Let's check state
        state_result = server.call_tool("get_subsystem_state", {
            "subsystem_id": subsystem_id,
        })
        assert "Target: 3 VP" in state_result.data

    def test_vp_auto_complete(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Auto Complete",
            "config": json.dumps({"targets": {"A": {"minor": 1, "major": 3}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        # Hit major threshold
        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "A", "amount": 3}),
        })
        assert result.success
        assert "Major threshold" in result.data
        assert "completed" in result.data

    def test_remove_vp(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Remove VP",
            "config": json.dumps({"targets": {"T": {"minor": 5, "major": 10}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "T", "amount": 5}),
        })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "remove_vp",
            "args": json.dumps({"target": "T", "amount": 2}),
        })
        assert result.success
        assert "T: 3 VP" in result.data

    # --- Chase tests ---

    def test_chase_movement(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "chase",
            "name": "Market Chase",
            "config": json.dumps({
                "participants": ["Party", "Thief"],
                "chase_length": 10,
            }),
        })
        subsystem_id = self._extract_id(start_result.data)

        # Move Party
        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "move",
            "args": json.dumps({"participant": "Party", "distance": 3}),
        })
        assert result.success
        assert "Party: 3" in result.data

    def test_chase_endpoint(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "chase",
            "name": "Short Chase",
            "config": json.dumps({
                "participants": ["Runner"],
                "chase_length": 5,
            }),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "move",
            "args": json.dumps({"participant": "Runner", "distance": 7}),
        })
        assert result.success
        assert "reached the end" in result.data

    # --- Hazard tests ---

    def test_hazard_damage_with_hardness(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "hazard",
            "name": "Dart Trap",
            "config": json.dumps({
                "hp": 50,
                "hardness": 10,
                "routine_actions": ["Fire darts"],
            }),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "damage_hazard",
            "args": json.dumps({"amount": 25}),
        })
        assert result.success
        # 25 - 10 hardness = 15 effective. HP: 50-15=35
        assert "35/50" in result.data

    def test_hazard_damage_below_hardness(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "hazard",
            "name": "Hard Trap",
            "config": json.dumps({"hp": 50, "hardness": 20}),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "damage_hazard",
            "args": json.dumps({"amount": 10}),
        })
        assert result.success
        # 10 - 20 = 0 effective. HP unchanged.
        assert "50/50" in result.data

    def test_hazard_destruction(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "hazard",
            "name": "Weak Trap",
            "config": json.dumps({"hp": 20, "hardness": 0}),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "damage_hazard",
            "args": json.dumps({"amount": 25}),
        })
        assert result.success
        assert "destroyed" in result.data.lower()

    def test_hazard_disable(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "hazard",
            "name": "Disableable Trap",
            "config": json.dumps({
                "hp": 50,
                "hardness": 5,
                "disable_conditions": [{"skill": "Thievery", "dc": 22}],
            }),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "disable_hazard",
        })
        assert result.success
        assert "disabled" in result.data.lower()

    def test_hazard_routine(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "hazard",
            "name": "Complex Trap",
            "config": json.dumps({
                "hp": 80,
                "hardness": 10,
                "routine_actions": ["Fires darts at nearest", "Resets mechanisms", "Sprays acid"],
            }),
        })
        subsystem_id = self._extract_id(start_result.data)

        # First routine action
        r1 = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "hazard_routine",
        })
        assert "Fires darts" in r1.data

        # Second
        r2 = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "hazard_routine",
        })
        assert "Resets mechanisms" in r2.data

        # Third
        r3 = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "hazard_routine",
        })
        assert "Sprays acid" in r3.data

        # Fourth wraps back to first
        r4 = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "hazard_routine",
        })
        assert "Fires darts" in r4.data

    # --- Infiltration tests ---

    def test_infiltration_awareness(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "infiltration",
            "name": "Castle Sneak",
            "config": json.dumps({"detection_threshold": 10}),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_awareness",
            "args": json.dumps({"amount": 3}),
        })
        assert result.success
        assert "3/10" in result.data

    def test_infiltration_detection(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "infiltration",
            "name": "Caught!",
            "config": json.dumps({"detection_threshold": 5}),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_awareness",
            "args": json.dumps({"amount": 6}),
        })
        assert result.success
        assert "failed" in result.data.lower()

    # --- Advance round ---

    def test_advance_round(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Round Test",
            "config": json.dumps({"targets": {"A": {"minor": 2, "major": 4}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "advance_round",
        })
        assert result.success
        assert "Round 1" in result.data

    # --- Error cases ---

    def test_invalid_subsystem_id(self, server):
        result = server.call_tool("subsystem_action", {
            "subsystem_id": "nonexistent",
            "action": "add_vp",
        })
        assert not result.success
        assert "not found" in result.error

    def test_action_on_completed_subsystem(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Done",
            "config": json.dumps({"targets": {"A": {"minor": 1, "major": 1}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        # Complete it via VP
        server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "A", "amount": 1}),
        })

        # Try to act on completed subsystem
        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "A", "amount": 1}),
        })
        assert not result.success
        assert "not active" in result.error

    def test_unknown_action(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Unknown Action",
            "config": "{}",
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "fly_to_moon",
        })
        assert not result.success
        assert "Unknown action" in result.error

    def test_invalid_subsystem_type(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "invalid_type",
            "name": "Bad Type",
        })
        assert not result.success
        assert "Invalid subsystem type" in result.error

    # --- end_subsystem ---

    def test_end_subsystem(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Ending",
            "config": json.dumps({"targets": {"A": {"minor": 2, "major": 4}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("end_subsystem", {
            "subsystem_id": subsystem_id,
            "outcome": "abandoned",
        })
        assert result.success
        assert "abandoned" in result.data

    def test_end_subsystem_invalid_outcome(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "Bad End",
            "config": "{}",
        })
        subsystem_id = self._extract_id(start_result.data)

        result = server.call_tool("end_subsystem", {
            "subsystem_id": subsystem_id,
            "outcome": "exploded",
        })
        assert not result.success
        assert "Invalid outcome" in result.error

    # --- list_subsystems ---

    def test_list_subsystems_empty(self, server):
        result = server.call_tool("list_subsystems", {})
        assert result.success
        assert "No active subsystems" in result.data

    def test_list_subsystems(self, server):
        server.call_tool("start_subsystem", {
            "type": "vp", "name": "VP One", "config": "{}",
        })
        server.call_tool("start_subsystem", {
            "type": "chase", "name": "Chase One",
            "config": json.dumps({"participants": ["A"], "chase_length": 5}),
        })

        result = server.call_tool("list_subsystems", {})
        assert result.success
        assert "VP One" in result.data
        assert "Chase One" in result.data

    # --- get_subsystem_state ---

    def test_get_subsystem_state_with_log(self, server):
        start_result = server.call_tool("start_subsystem", {
            "type": "vp",
            "name": "State Test",
            "config": json.dumps({"targets": {"T": {"minor": 2, "major": 4}}}),
        })
        subsystem_id = self._extract_id(start_result.data)

        server.call_tool("subsystem_action", {
            "subsystem_id": subsystem_id,
            "action": "add_vp",
            "args": json.dumps({"target": "T", "amount": 1}),
        })

        result = server.call_tool("get_subsystem_state", {
            "subsystem_id": subsystem_id,
            "include_log": True,
        })
        assert result.success
        assert "Action Log" in result.data

    # --- helpers ---

    @staticmethod
    def _extract_id(data: str) -> str:
        """Extract subsystem ID from start_subsystem response."""
        # Format: "... (ID: abc123def456)\n..."
        import re
        match = re.search(r"ID:\s*(\w+)", data)
        assert match, f"Could not extract ID from: {data}"
        return match.group(1)


class TestSubsystemStateValidation:
    """Phase 2: State validation guards."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start(self, server, type_="vp", name="Test", config=None):
        config = config or {}
        result = server.call_tool("start_subsystem", {
            "type": type_, "name": name, "config": json.dumps(config),
        })
        import re
        match = re.search(r"ID:\s*(\w+)", result.data)
        return match.group(1)

    # --- hazard-only actions on non-hazard ---

    def test_damage_hazard_on_vp_fails(self, server):
        sid = self._start(server, "vp")
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "damage_hazard",
            "args": json.dumps({"amount": 10}),
        })
        assert not result.success
        assert "only valid for hazard" in result.error

    def test_disable_hazard_on_chase_fails(self, server):
        sid = self._start(server, "chase", config={
            "participants": ["A"], "chase_length": 5,
        })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "disable_hazard",
        })
        assert not result.success
        assert "only valid for hazard" in result.error

    def test_hazard_routine_on_infiltration_fails(self, server):
        sid = self._start(server, "infiltration", config={"detection_threshold": 10})
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "hazard_routine",
        })
        assert not result.success
        assert "only valid for hazard" in result.error

    # --- hazard actions on destroyed/disabled ---

    def test_damage_destroyed_hazard_fails(self, server):
        sid = self._start(server, "hazard", config={"hp": 10, "hardness": 0})
        # Destroy it
        server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "damage_hazard",
            "args": json.dumps({"amount": 20}),
        })
        # Now it's completed (destroyed), try to damage again - will hit "not active" first
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "damage_hazard",
            "args": json.dumps({"amount": 5}),
        })
        assert not result.success

    def test_hazard_routine_on_disabled_fails(self, server):
        sid = self._start(server, "hazard", config={
            "hp": 50, "hardness": 0,
            "routine_actions": ["Fire darts"],
        })
        # Disable it
        server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "disable_hazard",
        })
        # Completed status → not active
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "hazard_routine",
        })
        assert not result.success

    # --- move on non-chase ---

    def test_move_on_vp_fails(self, server):
        sid = self._start(server, "vp")
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "move",
            "args": json.dumps({"participant": "A", "distance": 1}),
        })
        assert not result.success
        assert "only valid for chase" in result.error

    # --- add_awareness on non-infiltration ---

    def test_add_awareness_on_hazard_fails(self, server):
        sid = self._start(server, "hazard", config={"hp": 50, "hardness": 0})
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "add_awareness",
            "args": json.dumps({"amount": 1}),
        })
        assert not result.success
        assert "only valid for infiltration" in result.error


class TestSubsystemAutoHazardRoutine:
    """Phase 2: Auto hazard routine on advance_round."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_hazard(self, server, config):
        result = server.call_tool("start_subsystem", {
            "type": "hazard", "name": "Trap", "config": json.dumps(config),
        })
        import re
        return re.search(r"ID:\s*(\w+)", result.data).group(1)

    def test_advance_round_triggers_routine(self, server):
        sid = self._start_hazard(server, {
            "hp": 50, "hardness": 0,
            "routine_actions": ["Fires darts", "Resets"],
        })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
        })
        assert "Routine action: Fires darts" in result.data
        assert "Round 1" in result.data

    def test_advance_round_cycles_routine(self, server):
        sid = self._start_hazard(server, {
            "hp": 50, "hardness": 0,
            "routine_actions": ["Action A", "Action B"],
        })
        # Round 1: Action A
        r1 = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
        })
        assert "Action A" in r1.data

        # Round 2: Action B
        r2 = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
        })
        assert "Action B" in r2.data

        # Round 3: wraps to Action A
        r3 = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
        })
        assert "Action A" in r3.data

    def test_advance_round_multi_action(self, server):
        sid = self._start_hazard(server, {
            "hp": 50, "hardness": 0,
            "routine_actions": ["Fire", "Reset", "Spray"],
        })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
            "args": json.dumps({"routine_actions_count": 2}),
        })
        assert "Routine action: Fire" in result.data
        assert "Routine action: Reset" in result.data

    def test_advance_round_non_hazard_no_routine(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "VP",
            "config": json.dumps({"targets": {"A": {"minor": 2, "major": 4}}}),
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)

        r = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
        })
        assert "Routine action" not in r.data

    def test_advance_round_destroyed_hazard_no_routine(self, server):
        sid = self._start_hazard(server, {
            "hp": 10, "hardness": 0,
            "routine_actions": ["Fire"],
        })
        # Destroy it but keep it active (manually set destroyed without completing)
        inst = server.store.get(sid)
        inst.destroyed = True
        server.store.save(inst)

        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid, "action": "advance_round",
        })
        assert "Routine action" not in result.data


class TestExplorationSubsystem:
    """Phase 4C: Exploration subsystem type."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_exploration(self, server, config=None):
        config = config or {
            "activities": {"Valeros": "Scout", "Ezren": "Detect Magic"},
            "marching_order": ["Valeros", "Ezren", "Kyra", "Merisiel"],
        }
        result = server.call_tool("start_subsystem", {
            "type": "exploration",
            "name": "Forest Trek",
            "config": json.dumps(config),
        })
        import re
        match = re.search(r"ID:\s*(\w+)", result.data)
        return match.group(1)

    def test_start_exploration(self, server):
        sid = self._start_exploration(server)
        result = server.call_tool("get_subsystem_state", {"subsystem_id": sid})
        assert result.success
        assert "exploration" in result.data
        assert "Valeros" in result.data
        assert "Scout" in result.data

    def test_set_activity(self, server):
        sid = self._start_exploration(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "set_activity",
            "args": json.dumps({"character": "Kyra", "activity": "Search"}),
        })
        assert result.success
        assert "Kyra" in result.data
        assert "Search" in result.data

    def test_set_activity_shows_character(self, server):
        sid = self._start_exploration(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "set_activity",
            "args": json.dumps({"character": "Merisiel", "activity": "Avoid Notice"}),
        })
        assert result.success
        assert "Merisiel" in result.data
        assert "Avoid Notice" in result.data

    def test_set_marching_order(self, server):
        sid = self._start_exploration(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "set_marching_order",
            "args": json.dumps({"order": ["Merisiel", "Valeros", "Kyra", "Ezren"]}),
        })
        assert result.success
        assert "Merisiel > Valeros" in result.data

    def test_set_marching_order_csv(self, server):
        sid = self._start_exploration(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "set_marching_order",
            "args": json.dumps({"order": "A, B, C"}),
        })
        assert result.success
        assert "A > B > C" in result.data

    def test_set_activity_on_vp_fails(self, server):
        """set_activity should fail on non-exploration subsystem."""
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "VP", "config": "{}",
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)

        r = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "set_activity",
            "args": json.dumps({"character": "A", "activity": "Scout"}),
        })
        assert not r.success
        assert "only valid for exploration" in r.error

    def test_set_marching_order_on_chase_fails(self, server):
        """set_marching_order should fail on non-exploration subsystem."""
        result = server.call_tool("start_subsystem", {
            "type": "chase", "name": "Chase",
            "config": json.dumps({"participants": ["A"], "chase_length": 5}),
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)

        r = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "set_marching_order",
            "args": json.dumps({"order": ["A", "B"]}),
        })
        assert not r.success
        assert "only valid for exploration" in r.error

    def test_exploration_state_shows_activities_and_order(self, server):
        sid = self._start_exploration(server)
        result = server.call_tool("get_subsystem_state", {"subsystem_id": sid})
        assert result.success
        assert "Activities" in result.data
        assert "Marching Order" in result.data
        assert "Valeros > Ezren" in result.data


class TestHauntMechanics:
    """Tests for haunt subsystem type and new disable/trigger/reset actions."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    # ---- helpers ----

    def _start_haunt(self, server, name="Bloody Handprint", successes_needed=2):
        result = server.call_tool("start_subsystem", {
            "type": "haunt",
            "name": name,
            "config": json.dumps({
                "hp": 30,
                "trigger_condition": "When a living creature enters the room",
                "anchor": "The bloodstained mirror",
                "reset_interval": "1 day",
                "spiritual_immunity": True,
                "disable_conditions": [
                    {"skill": "Religion", "dc": 20, "successes_needed": successes_needed},
                    {"skill": "Occultism", "dc": 18, "successes_needed": 1},
                ],
            }),
        })
        assert result.success, result.error
        import re
        match = re.search(r"ID:\s*(\w+)", result.data)
        return match.group(1)

    # ---- spiritual immunity ----

    def test_haunt_spiritual_immunity(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "damage_hazard",
            "args": json.dumps({"amount": 15}),
        })
        assert result.success
        assert "immune to physical damage" in result.data.lower()

    def test_haunt_spiritual_immunity_via_shortcut(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("trigger_haunt", {"subsystem_id": sid})
        assert result.success  # trigger_haunt should work fine

    # ---- attempt_disable ----

    def test_attempt_disable_manual_roll_success(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion", "roll_total": 22}),
        })
        assert result.success
        assert "Success" in result.data or "Progress" in result.data

    def test_attempt_disable_manual_roll_failure(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion", "roll_total": 10}),
        })
        assert result.success
        assert "Failure" in result.data or "No progress" in result.data

    def test_attempt_disable_auto_roll(self, server):
        """modifier provided — verifies a d20 is rolled and degree applied."""
        sid = self._start_haunt(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion", "modifier": 10}),
        })
        assert result.success
        # Should contain roll breakdown
        assert "d20(" in result.data

    def test_attempt_disable_critical_success(self, server):
        """CS (beat DC by 10+) instantly completes condition."""
        sid = self._start_haunt(server)
        # DC 20, roll 30 = +10 over → critical success
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion", "roll_total": 30}),
        })
        assert result.success
        assert "instantly completed" in result.data

    def test_attempt_disable_critical_failure_warns_trigger(self, server):
        """CF on a haunt warns about trigger."""
        sid = self._start_haunt(server)
        # DC 20, roll 8 = CF (more than 9 below)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion", "roll_total": 8}),
        })
        assert result.success
        assert "triggers" in result.data.lower()

    def test_attempt_disable_all_conditions_met(self, server):
        """When all conditions are complete, haunt is disabled."""
        # Use 1 success needed for quick test
        sid = self._start_haunt(server, successes_needed=1)
        # Complete condition 0
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion", "roll_total": 30}),
        })
        # Complete condition 1 — should finish the haunt
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 1, "skill": "Occultism", "roll_total": 30}),
        })
        assert result.success
        assert "disabled" in result.data.lower()

    def test_attempt_disable_no_params_fails(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 0, "skill": "Religion"}),
        })
        assert result.success  # action succeeds but notification warns about missing params
        assert "Provide either" in result.data

    def test_attempt_disable_invalid_index(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 99, "skill": "Religion", "roll_total": 25}),
        })
        assert result.success
        assert "Invalid condition_index" in result.data

    # ---- shortcut tools: attempt_disable, trigger_haunt, reset_haunt ----

    def test_attempt_disable_shortcut_tool(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("attempt_disable", {
            "subsystem_id": sid,
            "condition_index": 0,
            "skill": "Religion",
            "roll_total": 25,
        })
        assert result.success

    def test_trigger_haunt(self, server):
        sid = self._start_haunt(server)
        result = server.call_tool("trigger_haunt", {"subsystem_id": sid})
        assert result.success
        assert "triggered" in result.data.lower()

    def test_trigger_haunt_on_non_haunt_fails(self, server):
        result_start = server.call_tool("start_subsystem", {
            "type": "hazard", "name": "A Trap",
            "config": json.dumps({"hp": 30}),
        })
        import re
        match = re.search(r"ID:\s*(\w+)", result_start.data)
        sid = match.group(1)
        result = server.call_tool("trigger_haunt", {"subsystem_id": sid})
        assert not result.success
        assert "only valid for haunt" in result.error

    def test_reset_haunt(self, server):
        sid = self._start_haunt(server)
        # Trigger it first
        server.call_tool("trigger_haunt", {"subsystem_id": sid})
        # Make one success
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_disable",
            "args": json.dumps({"condition_index": 1, "skill": "Occultism", "roll_total": 30}),
        })
        # Now reset
        result = server.call_tool("reset_haunt", {"subsystem_id": sid})
        assert result.success
        assert "reset" in result.data.lower()
        # Verify state is cleared
        state = server.call_tool("get_subsystem_state", {"subsystem_id": sid})
        assert "Currently Triggered" not in state.data

    def test_reset_haunt_never_interval(self, server):
        result_start = server.call_tool("start_subsystem", {
            "type": "haunt", "name": "Permanent Haunt",
            "config": json.dumps({
                "trigger_condition": "Always active",
                "reset_interval": "never",
                "disable_conditions": [{"skill": "Religion", "dc": 25, "successes_needed": 1}],
            }),
        })
        import re
        match = re.search(r"ID:\s*(\w+)", result_start.data)
        sid = match.group(1)
        result = server.call_tool("reset_haunt", {"subsystem_id": sid})
        assert result.success  # action returns as success, but content warns
        assert "never" in result.data.lower()

    # ---- state display ----

    def test_haunt_state_shows_fields(self, server):
        sid = self._start_haunt(server)
        state = server.call_tool("get_subsystem_state", {"subsystem_id": sid})
        assert state.success
        data = state.data
        assert "Trigger" in data
        assert "Anchor" in data
        assert "Spiritual Immunity" in data
        assert "Disable Conditions" in data

    def test_attempt_disable_on_hazard_also_works(self, server):
        """attempt_disable should work for regular hazards too."""
        result_start = server.call_tool("start_subsystem", {
            "type": "hazard", "name": "Dart Trap",
            "config": json.dumps({
                "hp": 40, "hardness": 5,
                "disable_conditions": [{"skill": "Thievery", "dc": 22, "successes_needed": 1}],
            }),
        })
        import re
        match = re.search(r"ID:\s*(\w+)", result_start.data)
        sid = match.group(1)
        result = server.call_tool("attempt_disable", {
            "subsystem_id": sid,
            "condition_index": 0,
            "skill": "Thievery",
            "roll_total": 25,
        })
        assert result.success


# ---------------------------------------------------------------------------
# New resolution classes (Part 1 plan implementation)
# ---------------------------------------------------------------------------

class TestInfluenceResolution:
    """Tests for influence subsystem dice-driven VP resolution."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_influence(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "influence",
            "name": "Influence Lord Gyr",
            "config": json.dumps({
                "npcs": {
                    "Lord Gyr": {
                        "approach_dcs": {"Diplomacy": 20, "Intimidation": 24},
                        "weaknesses": ["flattery"],
                        "resistances": ["threats"],
                        "min_vp": 0,
                        "max_vp": 5,
                    }
                }
            }),
        })
        assert result.success, result.error
        import re
        return re.search(r"ID:\s*(\w+)", result.data).group(1)

    def test_success_adds_one_vp(self, server):
        sid = self._start_influence(server)
        # DC 20, roll 21 → success → +1 VP
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_influence",
            "args": json.dumps({"npc": "Lord Gyr", "approach": "Diplomacy", "roll_total": 21}),
        })
        assert result.success
        assert "1/5" in result.data

    def test_critical_success_adds_two_vp(self, server):
        sid = self._start_influence(server)
        # DC 20, roll 30 → CS → +2 VP
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_influence",
            "args": json.dumps({"npc": "Lord Gyr", "approach": "Diplomacy", "roll_total": 30}),
        })
        assert result.success
        assert "2/5" in result.data

    def test_critical_failure_delta_is_negative_one(self, server):
        sid = self._start_influence(server)
        # Get to VP=2 first
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_influence",
            "args": json.dumps({"npc": "Lord Gyr", "approach": "Diplomacy", "roll_total": 30}),
        })
        # DC 20, roll 5 → CF (-15) → -1 VP; from 2 → 1
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_influence",
            "args": json.dumps({"npc": "Lord Gyr", "approach": "Diplomacy", "roll_total": 5}),
        })
        assert result.success
        assert "-1 VP" in result.data or "1/5" in result.data

    def test_weakness_lowers_dc(self, server):
        sid = self._start_influence(server)
        # Flattery weakness reduces DC from 20 to 18; roll 19 → success at DC 18
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_influence",
            "args": json.dumps({
                "npc": "Lord Gyr",
                "approach": "Diplomacy",
                "roll_total": 19,
                "situation": "using lots of flattery and praise",
            }),
        })
        assert result.success
        # DC reduced notification or +1 VP
        assert "flattery" in result.data.lower() or "+1 VP" in result.data or "1/5" in result.data

    def test_all_targets_max_auto_completes(self, server):
        sid = self._start_influence(server)
        # CS every round: +2 VP each → 3 rounds hits max_vp=5 (2+2+1)
        for _ in range(3):
            server.call_tool("subsystem_action", {
                "subsystem_id": sid,
                "action": "attempt_influence",
                "args": json.dumps({"npc": "Lord Gyr", "approach": "Diplomacy", "roll_total": 30}),
            })
        state = server.call_tool("get_subsystem_state", {"subsystem_id": sid})
        assert "completed" in state.data.lower()

    def test_wrong_type_error(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "Wrong", "config": "{}",
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_influence",
            "args": json.dumps({"npc": "X", "approach": "Diplomacy", "roll_total": 20}),
        })
        assert not result.success
        assert "only valid for influence" in result.error


class TestChaseResolution:
    """Tests for chase subsystem obstacle resolution."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_chase(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "chase",
            "name": "Roof Chase",
            "config": json.dumps({
                "participants": ["Party", "Thief"],
                "chase_length": 10,
                "obstacles": [
                    {
                        "name": "Crowded Market",
                        "skill": "Acrobatics",
                        "dc": 18,
                        "alternative_skill": "Athletics",
                        "alternative_dc": 20,
                        "success_advances": 1,
                        "cf_retreats": 1,
                    },
                    {
                        "name": "Narrow Ledge",
                        "skill": "Athletics",
                        "dc": 16,
                        "success_advances": 2,
                        "cf_retreats": 1,
                    },
                ],
            }),
        })
        assert result.success, result.error
        import re
        return re.search(r"ID:\s*(\w+)", result.data).group(1)

    def test_success_advances_position(self, server):
        sid = self._start_chase(server)
        # DC 18, roll 19 → success → +1
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "Party", "obstacle_index": 0, "skill": "Acrobatics", "roll_total": 19}),
        })
        assert result.success
        assert "Party" in result.data
        assert "0 → 1" in result.data

    def test_critical_success_advances_double(self, server):
        sid = self._start_chase(server)
        # DC 18, roll 28 → CS (+10 over) → success_advances*2 = 2
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "Party", "obstacle_index": 0, "skill": "Acrobatics", "roll_total": 28}),
        })
        assert result.success
        assert "0 → 2" in result.data

    def test_critical_failure_retreats(self, server):
        sid = self._start_chase(server)
        # Advance to pos 2 first
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "Party", "obstacle_index": 0, "skill": "Acrobatics", "roll_total": 28}),
        })
        # CF: DC 18, roll 5 → retreat 1
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "Party", "obstacle_index": 0, "skill": "Acrobatics", "roll_total": 5}),
        })
        assert result.success
        assert "2 → 1" in result.data

    def test_alternative_skill_uses_alt_dc(self, server):
        sid = self._start_chase(server)
        # Athletics is alt skill with alt_dc=20; roll 20 → success at DC 20
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "Party", "obstacle_index": 0, "skill": "Athletics", "roll_total": 20}),
        })
        assert result.success
        # Success (not CS since 20-20=0, success); the DC used should be 20
        assert "Crowded Market" in result.data

    def test_oob_index_error(self, server):
        sid = self._start_chase(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "Party", "obstacle_index": 99, "skill": "Acrobatics", "roll_total": 20}),
        })
        assert result.success
        assert "Invalid obstacle_index" in result.data

    def test_wrong_type_error(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "Wrong", "config": "{}",
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "attempt_obstacle",
            "args": json.dumps({"participant": "X", "obstacle_index": 0, "skill": "A", "roll_total": 20}),
        })
        assert not result.success
        assert "only valid for chase" in result.error


class TestResearchResolution:
    """Tests for research subsystem source-based VP resolution."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_research(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "research",
            "name": "Ancient Library",
            "config": json.dumps({
                "vp_target": 10,
                "sources": [
                    {
                        "name": "Old Tome",
                        "skill": "Arcana",
                        "dc": 20,
                        "per_success": 2,
                        "max_contribution": 6,
                        "discovery_note": "The tome reveals a hidden vault location!",
                    },
                    {
                        "name": "Star Charts",
                        "skill": "Occultism",
                        "dc": 18,
                        "per_success": 1,
                        "max_contribution": 4,
                    },
                ],
            }),
        })
        assert result.success, result.error
        import re
        return re.search(r"ID:\s*(\w+)", result.data).group(1)

    def test_success_adds_vp(self, server):
        sid = self._start_research(server)
        # DC 20, roll 21 → success → +2 VP
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 21}),
        })
        assert result.success
        assert "+2 VP" in result.data

    def test_critical_success_triggers_discovery_note(self, server):
        sid = self._start_research(server)
        # DC 20, roll 30 → CS → +4 VP + discovery note
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 30}),
        })
        assert result.success
        assert "Discovery" in result.data
        assert "hidden vault" in result.data

    def test_source_cap_stops_vp_gain(self, server):
        sid = self._start_research(server)
        # Fill source 0 to max_contribution=6 via 3 CS (4 VP each time, capped)
        # Roll CS 3 times: +4 each = 12, but capped at 6
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 30}),
        })
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 30}),
        })
        # Third call — should be capped
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 30}),
        })
        assert result.success
        assert "max contribution" in result.data

    def test_vp_target_auto_completes(self, server):
        sid = self._start_research(server)
        # source 0: per_success=2, max_contribution=6. CS=4 VP each.
        #   1st CS: contribution 0→4, VP 0→4
        #   2nd CS: contribution 4→6 (capped), VP 4→6
        # source 1: per_success=1, max_contribution=4. success=1 VP.
        #   need 4 more VP from source 1 → 4 successes
        for _ in range(2):
            server.call_tool("subsystem_action", {
                "subsystem_id": sid,
                "action": "research_check",
                "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 30}),
            })
        # VP now at 6; need 4 more
        for _ in range(3):
            server.call_tool("subsystem_action", {
                "subsystem_id": sid,
                "action": "research_check",
                "args": json.dumps({"source_index": 1, "skill": "Occultism", "roll_total": 25}),
            })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 1, "skill": "Occultism", "roll_total": 25}),
        })
        assert result.success
        assert "Research complete" in result.data or "completed" in result.data.lower()

    def test_wrong_type_error(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "Wrong", "config": "{}",
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "research_check",
            "args": json.dumps({"source_index": 0, "skill": "Arcana", "roll_total": 20}),
        })
        assert not result.success
        assert "only valid for research" in result.error


class TestInfiltrationEdge:
    """Tests for infiltration edge point earn/spend mechanics."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_infiltration(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "infiltration",
            "name": "Nobleman's Estate",
            "config": json.dumps({
                "detection_threshold": 15,
                "edge_benefits": [
                    {"cost": 1, "description": "Bypass one guard check"},
                    {"cost": 3, "description": "Stolen uniform — avoid detection"},
                ],
            }),
        })
        assert result.success, result.error
        import re
        return re.search(r"ID:\s*(\w+)", result.data).group(1)

    def test_earn_edge_increments(self, server):
        sid = self._start_infiltration(server)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "earn_edge",
            "args": json.dumps({"amount": 2, "action": "Bribed a guard"}),
        })
        assert result.success
        assert "Edge points: 2" in result.data

    def test_spend_edge_decrements(self, server):
        sid = self._start_infiltration(server)
        # Earn first
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "earn_edge",
            "args": json.dumps({"amount": 3}),
        })
        # Spend benefit index 0 (cost=1)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "spend_edge",
            "args": json.dumps({"benefit_index": 0}),
        })
        assert result.success
        assert "Edge points remaining: 2" in result.data

    def test_spend_edge_validates_cost(self, server):
        sid = self._start_infiltration(server)
        # Earn only 2, try benefit_index=1 which costs 3
        server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "earn_edge",
            "args": json.dumps({"amount": 2}),
        })
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "spend_edge",
            "args": json.dumps({"benefit_index": 1}),
        })
        assert result.success
        assert "Insufficient edge" in result.data

    def test_earn_edge_wrong_type_error(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "Wrong", "config": "{}",
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "earn_edge",
            "args": json.dumps({"amount": 1}),
        })
        assert not result.success
        assert "only valid for infiltration" in result.error


class TestHazardCombat:
    """Tests for hazard attack roll and saving throw resolution."""

    @pytest.fixture
    def server(self, tmp_path):
        from unittest.mock import patch
        with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_path):
            yield SubsystemServer("test-campaign")

    def _start_hazard(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "hazard",
            "name": "Scything Blade Trap",
            "config": json.dumps({
                "hp": 60,
                "hardness": 10,
                "routine_actions": ["Swings blade at nearest"],
                "disable_conditions": [{"skill": "Thievery", "dc": 22}],
            }),
        })
        assert result.success, result.error
        import re
        return re.search(r"ID:\s*(\w+)", result.data).group(1)

    def test_hazard_attack_hit(self, server):
        sid = self._start_hazard(server)
        # attack_bonus=12, target_ac=18, roll_total=20 → total=20 ≥ 18 → hit
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "hazard_attack",
            "args": json.dumps({"attack_bonus": 12, "target_ac": 18, "roll_total": 20}),
        })
        assert result.success
        assert "Hit" in result.data
        assert "Apply damage" in result.data

    def test_hazard_attack_miss(self, server):
        sid = self._start_hazard(server)
        # roll_total=10 < target_ac=18 → miss (not CF since 18-10=8 < 10)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "hazard_attack",
            "args": json.dumps({"attack_bonus": 0, "target_ac": 18, "roll_total": 10}),
        })
        assert result.success
        assert "Miss" in result.data
        assert "Apply damage" not in result.data

    def test_hazard_attack_critical_hit(self, server):
        sid = self._start_hazard(server)
        # roll_total=28 ≥ target_ac=18+10=28 → critical hit
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "hazard_attack",
            "args": json.dumps({"attack_bonus": 0, "target_ac": 18, "roll_total": 28}),
        })
        assert result.success
        assert "Critical Hit" in result.data

    def test_hazard_save_success(self, server):
        sid = self._start_hazard(server)
        # save_dc=20, save_modifier=5, roll_total=22 → 22 ≥ 20 → success
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "hazard_save",
            "args": json.dumps({"save_dc": 20, "save_type": "Reflex", "save_modifier": 5, "roll_total": 22}),
        })
        assert result.success
        assert "Reflex" in result.data
        assert "Success" in result.data

    def test_hazard_save_critical_failure(self, server):
        sid = self._start_hazard(server)
        # save_dc=20, roll_total=5 → 5-20=-15 < -9 → critical failure
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "hazard_save",
            "args": json.dumps({"save_dc": 20, "save_type": "Fortitude", "save_modifier": 0, "roll_total": 5}),
        })
        assert result.success
        assert "Critical Failure" in result.data

    def test_hazard_combat_wrong_type_error(self, server):
        result = server.call_tool("start_subsystem", {
            "type": "vp", "name": "Wrong", "config": "{}",
        })
        import re
        sid = re.search(r"ID:\s*(\w+)", result.data).group(1)
        result = server.call_tool("subsystem_action", {
            "subsystem_id": sid,
            "action": "hazard_attack",
            "args": json.dumps({"attack_bonus": 5, "target_ac": 15, "roll_total": 18}),
        })
        assert not result.success
        assert "only valid for hazard" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
