"""Tests for CampaignStateServer MCP server."""

from pathlib import Path

import pytest

from gm_agent.mcp.campaign_state import CampaignStateServer
from gm_agent.storage.campaign import CampaignStore
from gm_agent.storage.session import SessionStore
from gm_agent.storage.schemas import SceneState


@pytest.fixture
def campaign_state_setup(tmp_path: Path):
    """Set up campaign, session, and CampaignStateServer."""
    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()

    campaign_store_local = CampaignStore(base_dir=campaigns_dir)
    session_store_local = SessionStore(base_dir=campaigns_dir)

    campaign = campaign_store_local.create(
        name="State Test Campaign",
        background="Test background",
    )
    session = session_store_local.start(campaign.id)

    # Patch the stores where they are used (in campaign_state module)
    import gm_agent.mcp.campaign_state as cs_module

    original_campaign_store = cs_module.campaign_store
    original_session_store = cs_module.session_store
    original_campaigns_dir = cs_module.CAMPAIGNS_DIR

    cs_module.campaign_store = campaign_store_local
    cs_module.session_store = session_store_local
    cs_module.CAMPAIGNS_DIR = campaigns_dir

    server = CampaignStateServer(campaign.id)

    yield {
        "server": server,
        "campaign": campaign,
        "session": session,
        "campaign_store": campaign_store_local,
        "session_store": session_store_local,
        "campaigns_dir": campaigns_dir,
    }

    server.close()

    # Restore original stores
    cs_module.campaign_store = original_campaign_store
    cs_module.session_store = original_session_store
    cs_module.CAMPAIGNS_DIR = original_campaigns_dir


class TestCampaignStateServerTools:
    """Tests for CampaignStateServer tool definitions."""

    def test_list_tools(self, campaign_state_setup):
        """Server should list all expected tools."""
        server = campaign_state_setup["server"]
        tools = server.list_tools()

        tool_names = [t.name for t in tools]
        assert "update_scene" in tool_names
        assert "advance_time" in tool_names
        assert "log_event" in tool_names
        assert "search_history" in tool_names
        assert "get_scene" in tool_names
        assert "get_session_summary" in tool_names
        assert "update_session_summary" in tool_names
        assert "get_preferences" in tool_names
        assert "update_preferences" in tool_names

    def test_get_tool(self, campaign_state_setup):
        """Server should find tools by name."""
        server = campaign_state_setup["server"]

        tool = server.get_tool("update_scene")
        assert tool is not None
        assert tool.name == "update_scene"

        tool = server.get_tool("nonexistent")
        assert tool is None


class TestUpdateScene:
    """Tests for update_scene tool."""

    def test_update_location(self, campaign_state_setup):
        """update_scene should change location."""
        server = campaign_state_setup["server"]
        session_store = campaign_state_setup["session_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool("update_scene", {"location": "Dragon's Lair"})

        assert result.success
        session = session_store.get_current(campaign.id)
        assert session.scene_state.location == "Dragon's Lair"

    def test_update_npcs(self, campaign_state_setup):
        """update_scene should change NPCs present."""
        server = campaign_state_setup["server"]
        session_store = campaign_state_setup["session_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool(
            "update_scene",
            {"npcs_present": "Sheriff, Innkeeper, Mysterious Stranger"},
        )

        assert result.success
        session = session_store.get_current(campaign.id)
        assert "Sheriff" in session.scene_state.npcs_present
        assert "Innkeeper" in session.scene_state.npcs_present

    def test_update_time(self, campaign_state_setup):
        """update_scene should change time of day."""
        server = campaign_state_setup["server"]
        session_store = campaign_state_setup["session_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool("update_scene", {"time_of_day": "midnight"})

        assert result.success
        session = session_store.get_current(campaign.id)
        assert session.scene_state.time_of_day == "midnight"

    def test_update_conditions(self, campaign_state_setup):
        """update_scene should change conditions."""
        server = campaign_state_setup["server"]
        session_store = campaign_state_setup["session_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool(
            "update_scene",
            {"conditions": "dark, raining, cold"},
        )

        assert result.success
        session = session_store.get_current(campaign.id)
        assert "dark" in session.scene_state.conditions
        assert "raining" in session.scene_state.conditions

    def test_update_multiple_fields(self, campaign_state_setup):
        """update_scene should update multiple fields at once."""
        server = campaign_state_setup["server"]
        session_store = campaign_state_setup["session_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool(
            "update_scene",
            {
                "location": "Market Square",
                "time_of_day": "noon",
                "npcs_present": "Merchant, Guard",
                "conditions": "sunny, crowded",
                "notes": "Festival day",
            },
        )

        assert result.success
        session = session_store.get_current(campaign.id)
        assert session.scene_state.location == "Market Square"
        assert session.scene_state.time_of_day == "noon"
        assert "Festival" in session.scene_state.notes


class TestAdvanceTime:
    """Tests for advance_time tool."""

    def test_advance_hours(self, campaign_state_setup):
        """advance_time should handle hour advancement."""
        server = campaign_state_setup["server"]

        # Set initial time
        server.call_tool("update_scene", {"time_of_day": "morning"})

        result = server.call_tool("advance_time", {"amount": "4 hours"})

        assert result.success
        assert "afternoon" in result.data.lower() or "4 hours" in result.data.lower()

    def test_advance_overnight(self, campaign_state_setup):
        """advance_time should handle overnight rest."""
        server = campaign_state_setup["server"]

        server.call_tool("update_scene", {"time_of_day": "evening"})

        result = server.call_tool(
            "advance_time",
            {"amount": "overnight", "activity": "resting"},
        )

        assert result.success
        assert "morning" in result.data.lower()

    def test_advance_logs_event(self, campaign_state_setup):
        """advance_time should log an event."""
        server = campaign_state_setup["server"]

        server.call_tool("advance_time", {"amount": "1 day", "activity": "traveling"})

        # Search for the logged event
        result = server.call_tool("search_history", {"query": "time advanced"})

        assert result.success
        assert "1 day" in result.data or "traveling" in result.data.lower()


class TestLogEvent:
    """Tests for log_event tool."""

    def test_log_session_event(self, campaign_state_setup):
        """log_event should record session-level event."""
        server = campaign_state_setup["server"]

        result = server.call_tool(
            "log_event",
            {
                "event": "The party found a hidden passage",
                "importance": "session",
            },
        )

        assert result.success
        assert "logged" in result.data.lower()

    def test_log_arc_event(self, campaign_state_setup):
        """log_event should record arc-level event."""
        server = campaign_state_setup["server"]

        result = server.call_tool(
            "log_event",
            {
                "event": "The villain revealed their identity",
                "importance": "arc",
                "tags": "villain, revelation, plot",
            },
        )

        assert result.success

    def test_log_campaign_event(self, campaign_state_setup):
        """log_event should record campaign-level event."""
        server = campaign_state_setup["server"]

        result = server.call_tool(
            "log_event",
            {
                "event": "The ancient artifact was destroyed",
                "importance": "campaign",
                "tags": "artifact, destruction, major",
            },
        )

        assert result.success

    def test_log_event_requires_text(self, campaign_state_setup):
        """log_event should require event text."""
        server = campaign_state_setup["server"]

        result = server.call_tool("log_event", {"event": ""})

        assert not result.success
        assert "required" in result.error.lower()


class TestSearchHistory:
    """Tests for search_history tool."""

    def test_search_finds_events(self, campaign_state_setup):
        """search_history should find logged events."""
        server = campaign_state_setup["server"]

        # Log some events
        server.call_tool("log_event", {"event": "Found the golden key"})
        server.call_tool("log_event", {"event": "Fought the dragon"})

        result = server.call_tool("search_history", {"query": "golden key"})

        assert result.success
        assert "golden key" in result.data.lower()

    def test_search_filters_importance(self, campaign_state_setup):
        """search_history should filter by importance."""
        server = campaign_state_setup["server"]

        server.call_tool(
            "log_event",
            {"event": "Minor event", "importance": "session"},
        )
        server.call_tool(
            "log_event",
            {"event": "Major event", "importance": "campaign"},
        )

        result = server.call_tool(
            "search_history",
            {"query": "event", "importance": "campaign"},
        )

        assert result.success
        assert "Major" in result.data
        assert "Minor" not in result.data

    def test_search_empty_results(self, campaign_state_setup):
        """search_history should handle no results."""
        server = campaign_state_setup["server"]

        result = server.call_tool(
            "search_history",
            {"query": "nonexistent xyz123"},
        )

        assert result.success
        assert "no events" in result.data.lower()


class TestGetScene:
    """Tests for get_scene tool."""

    def test_get_scene(self, campaign_state_setup):
        """get_scene should return current scene state."""
        server = campaign_state_setup["server"]

        # Set up scene
        server.call_tool(
            "update_scene",
            {
                "location": "Tavern",
                "time_of_day": "evening",
                "npcs_present": "Bartender",
                "conditions": "warm, noisy",
            },
        )

        result = server.call_tool("get_scene", {})

        assert result.success
        assert "Tavern" in result.data
        assert "evening" in result.data
        assert "Bartender" in result.data


class TestSessionSummary:
    """Tests for session summary tools."""

    def test_get_empty_summary(self, campaign_state_setup):
        """get_session_summary should handle no summary."""
        server = campaign_state_setup["server"]

        result = server.call_tool("get_session_summary", {})

        assert result.success
        assert "no summary" in result.data.lower() or "just started" in result.data.lower()

    def test_update_and_get_summary(self, campaign_state_setup):
        """update_session_summary should update summary."""
        server = campaign_state_setup["server"]

        update_result = server.call_tool(
            "update_session_summary",
            {"summary": "The party explored the dungeon and found treasure."},
        )

        assert update_result.success

        get_result = server.call_tool("get_session_summary", {})

        assert get_result.success
        assert "explored the dungeon" in get_result.data


class TestPreferences:
    """Tests for preference tools."""

    def test_get_default_preferences(self, campaign_state_setup):
        """get_preferences should return defaults."""
        server = campaign_state_setup["server"]

        result = server.call_tool("get_preferences", {})

        assert result.success
        assert "rag" in result.data.lower()
        assert "uncertainty" in result.data.lower()

    def test_update_rag_aggressiveness(self, campaign_state_setup):
        """update_preferences should change RAG aggressiveness."""
        server = campaign_state_setup["server"]
        campaign_store = campaign_state_setup["campaign_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool(
            "update_preferences",
            {"rag_aggressiveness": "aggressive"},
        )

        assert result.success
        assert "aggressive" in result.data.lower()

        # Verify in campaign
        updated = campaign_store.get(campaign.id)
        assert updated.preferences.get("rag_aggressiveness") == "aggressive"

    def test_update_uncertainty_mode(self, campaign_state_setup):
        """update_preferences should change uncertainty mode."""
        server = campaign_state_setup["server"]
        campaign_store = campaign_state_setup["campaign_store"]
        campaign = campaign_state_setup["campaign"]

        result = server.call_tool(
            "update_preferences",
            {"uncertainty_mode": "introspective"},
        )

        assert result.success

        updated = campaign_store.get(campaign.id)
        assert updated.preferences.get("uncertainty_mode") == "introspective"

    def test_invalid_preference_value(self, campaign_state_setup):
        """update_preferences should ignore invalid values."""
        server = campaign_state_setup["server"]

        result = server.call_tool(
            "update_preferences",
            {"rag_aggressiveness": "invalid_value"},
        )

        assert result.success
        assert "no valid" in result.data.lower()


class TestUnknownTool:
    """Tests for unknown tool handling."""

    def test_unknown_tool_error(self, campaign_state_setup):
        """Server should return error for unknown tool."""
        server = campaign_state_setup["server"]

        result = server.call_tool("nonexistent_tool", {})

        assert not result.success
        assert "unknown tool" in result.error.lower()
