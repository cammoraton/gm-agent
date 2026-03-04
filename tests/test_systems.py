"""Tests for GameSystem ABC, registry, and PF2eSystem wrapper."""

from typing import Any

import pytest

import gm_agent.systems as systems_mod
from gm_agent.systems import (
    GameSystem,
    SYSTEM_REGISTRY,
    register_system,
    get_system,
    list_systems,
    _discover_systems,
)
from gm_agent.mcp.base import MCPServer, ToolDef, ToolResult


@pytest.fixture(autouse=True)
def _ensure_pf2e_discovered():
    """Ensure PF2e system is discovered for every test."""
    # Force re-discovery
    systems_mod._discovered = False
    _discover_systems()
    yield
    # Reset for clean state
    systems_mod._discovered = False


# ---------------------------------------------------------------------------
# Test GameSystem ABC
# ---------------------------------------------------------------------------


class _DummyServer(MCPServer):
    """Minimal server for testing."""

    def list_tools(self) -> list[ToolDef]:
        return [ToolDef(name="dummy_tool", description="test", parameters=[])]

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data="ok")

    def close(self):
        pass


class ConcreteSystem(GameSystem):
    """Concrete subclass for testing the ABC."""

    name = "test_system"
    display_name = "Test System"
    category = "generation"

    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        return [_DummyServer()]

    def system_prompt(self) -> str:
        return "You are a test facilitator."


def test_abc_cannot_instantiate():
    """GameSystem ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GameSystem()


def test_concrete_system_instantiates():
    """A concrete subclass can be instantiated."""
    system = ConcreteSystem()
    assert system.name == "test_system"
    assert system.display_name == "Test System"
    assert system.category == "generation"


def test_concrete_system_servers():
    """Concrete system returns servers."""
    system = ConcreteSystem()
    servers = system.servers({})
    assert len(servers) == 1
    assert isinstance(servers[0], MCPServer)


def test_concrete_system_prompt():
    """Concrete system returns a prompt."""
    system = ConcreteSystem()
    assert "test facilitator" in system.system_prompt()


def test_default_stores():
    """Default stores() returns empty dict."""
    system = ConcreteSystem()
    assert system.stores("campaign-123") == {}


def test_default_prep_pipeline():
    """Default prep_pipeline() returns None."""
    system = ConcreteSystem()
    assert system.prep_pipeline() is None


# ---------------------------------------------------------------------------
# Test Registry
# ---------------------------------------------------------------------------


def test_register_system_decorator():
    """@register_system adds the class to SYSTEM_REGISTRY."""
    @register_system
    class MySystem(GameSystem):
        name = "_test_register"
        display_name = "Test Register"
        category = "generation"

        def servers(self, context):
            return []

        def system_prompt(self):
            return ""

    assert "_test_register" in SYSTEM_REGISTRY
    assert SYSTEM_REGISTRY["_test_register"] is MySystem
    # Clean up
    del SYSTEM_REGISTRY["_test_register"]


def test_get_system_returns_instance():
    """get_system() returns an instance of the registered class."""
    @register_system
    class MySystem2(GameSystem):
        name = "_test_get"
        display_name = "Test Get"
        category = "generation"

        def servers(self, context):
            return []

        def system_prompt(self):
            return "hello"

    instance = get_system("_test_get")
    assert isinstance(instance, MySystem2)
    assert instance.system_prompt() == "hello"
    # Clean up
    del SYSTEM_REGISTRY["_test_get"]


def test_get_system_unknown_raises():
    """get_system() raises KeyError for unknown systems."""
    with pytest.raises(KeyError, match="Unknown game system"):
        get_system("nonexistent_system_xyz")


def test_list_systems():
    """list_systems() returns info about all registered systems."""
    systems = list_systems()
    # PF2eSystem is auto-registered on import
    names = [s["name"] for s in systems]
    assert "pf2e" in names


# ---------------------------------------------------------------------------
# Test PF2eSystem
# ---------------------------------------------------------------------------


def test_pf2e_system_registered():
    """PF2eSystem is registered as 'pf2e'."""
    assert "pf2e" in SYSTEM_REGISTRY


def test_pf2e_system_metadata():
    """PF2eSystem has correct metadata."""
    system = get_system("pf2e")
    assert system.name == "pf2e"
    assert system.display_name == "Pathfinder 2e (Remaster)"
    assert system.category == "rpg"


def test_pf2e_system_prompt():
    """PF2eSystem returns the GM_SYSTEM_PROMPT."""
    from gm_agent.config import GM_SYSTEM_PROMPT
    system = get_system("pf2e")
    assert system.system_prompt() == GM_SYSTEM_PROMPT


def test_pf2e_system_servers_no_campaign(mock_pathfinder_search):
    """PF2eSystem creates stateless servers without campaign_id."""
    system = get_system("pf2e")
    servers = system.servers({"campaign_id": None})
    # Should have RAG, Encounter, CreatureModifier (3 stateless)
    assert len(servers) == 3
    type_names = [type(s).__name__ for s in servers]
    assert "PF2eRAGServer" in type_names
    assert "EncounterServer" in type_names
    assert "CreatureModifierServer" in type_names


def test_pf2e_system_servers_with_campaign(mock_pathfinder_search, tmp_campaigns_dir):
    """PF2eSystem creates PF2e-specific servers with campaign_id.

    Core servers (CampaignStateServer, NPCKnowledgeServer) are now created
    by MCPClient directly, so PF2eSystem returns 6: 3 stateless + 3 campaign.
    """
    from unittest.mock import patch
    campaign_id = "test-pf2e"
    (tmp_campaigns_dir / campaign_id).mkdir()
    with patch("gm_agent.systems.pf2e.servers.subsystem.CAMPAIGNS_DIR", tmp_campaigns_dir), \
         patch("gm_agent.systems.pf2e.storage.subsystems.CAMPAIGNS_DIR", tmp_campaigns_dir):
        system = get_system("pf2e")
        servers = system.servers({"campaign_id": campaign_id, "llm": None})
        # 3 stateless + 3 campaign-scoped (CharacterRunner, NPCBuilder, Subsystem) = 6
        assert len(servers) == 6


def test_pf2e_campaign_plugins(tmp_campaigns_dir):
    """PF2eSystem.campaign_plugins() returns PF2eCampaignToolPlugin."""
    system = get_system("pf2e")
    plugins = system.campaign_plugins("test-campaign")
    assert len(plugins) == 1
    assert type(plugins[0]).__name__ == "PF2eCampaignToolPlugin"


def test_base_campaign_plugins_empty():
    """Default campaign_plugins() returns empty list."""
    system = get_system("microscope")
    plugins = system.campaign_plugins("test-campaign")
    assert plugins == []


# ---------------------------------------------------------------------------
# Test Campaign model extension
# ---------------------------------------------------------------------------


def test_campaign_default_game_systems():
    """Campaign model defaults to ['pf2e'] game_systems."""
    from gm_agent.storage.schemas import Campaign
    campaign = Campaign(id="test", name="Test")
    assert campaign.game_systems == ["pf2e"]
    assert campaign.primary_system == "pf2e"


def test_campaign_custom_game_systems():
    """Campaign model accepts custom game_systems."""
    from gm_agent.storage.schemas import Campaign
    campaign = Campaign(
        id="worldbuilding",
        name="Worldbuilding",
        game_systems=["microscope", "ex_novo"],
        primary_system="microscope",
    )
    assert campaign.game_systems == ["microscope", "ex_novo"]
    assert campaign.primary_system == "microscope"


def test_campaign_backward_compat():
    """Existing campaigns without game_systems field still work."""
    from gm_agent.storage.schemas import Campaign
    data = {
        "id": "old-campaign",
        "name": "Old Campaign",
        "background": "Some background",
    }
    campaign = Campaign(**data)
    assert campaign.game_systems == ["pf2e"]
    assert campaign.primary_system == "pf2e"
