"""Tests for Character Runner MCP server."""

from datetime import datetime
from pathlib import Path

import pytest

from gm_agent.models.base import LLMBackend, LLMResponse
from gm_agent.storage.characters import CharacterStore
from gm_agent.storage.schemas import CharacterProfile
from gm_agent.mcp.character_runner import CharacterRunnerServer


class MockLLM(LLMBackend):
    """Mock LLM for testing character runner."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text
        self.calls: list[tuple] = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return LLMResponse(text=self.response_text, tool_calls=[])

    def get_model_name(self) -> str:
        return "mock-llm"

    def is_available(self) -> bool:
        return True


class TestCharacterProfile:
    """Tests for CharacterProfile schema."""

    def test_default_values(self):
        """CharacterProfile should have sensible defaults."""
        profile = CharacterProfile(
            id="test-id",
            campaign_id="test-campaign",
            name="Test NPC",
        )
        assert profile.character_type == "npc"
        assert profile.personality == ""
        assert profile.knowledge == []
        assert profile.goals == []
        assert profile.intelligence == "average"
        assert profile.risk_tolerance == "medium"

    def test_npc_profile(self):
        """CharacterProfile should store NPC-specific fields."""
        profile = CharacterProfile(
            id="innkeeper",
            campaign_id="test",
            name="Old Barkeep",
            character_type="npc",
            personality="Gruff but fair",
            knowledge=["Local gossip", "Secret passages"],
            goals=["Run the tavern", "Protect regulars"],
            secrets=["Former adventurer"],
            speech_patterns="Short sentences, lots of sighing",
        )
        assert profile.name == "Old Barkeep"
        assert "Local gossip" in profile.knowledge
        assert "Former adventurer" in profile.secrets

    def test_monster_profile(self):
        """CharacterProfile should store monster-specific fields."""
        profile = CharacterProfile(
            id="goblin-chief",
            campaign_id="test",
            name="Goblin Chief",
            character_type="monster",
            intelligence="low",
            instincts=["territorial", "cowardly"],
            morale="Flees at half HP",
            goals=["Protect the warren"],
        )
        assert profile.character_type == "monster"
        assert profile.intelligence == "low"
        assert "territorial" in profile.instincts

    def test_player_profile(self):
        """CharacterProfile should store player emulation fields."""
        profile = CharacterProfile(
            id="test-player",
            campaign_id="test",
            name="Cautious Chris",
            character_type="player",
            playstyle="Always checks for traps",
            risk_tolerance="low",
            party_role="Scout",
            quirks=["Never splits the party", "Hoards potions"],
        )
        assert profile.character_type == "player"
        assert profile.risk_tolerance == "low"
        assert "Hoards potions" in profile.quirks


class TestCharacterStore:
    """Tests for CharacterStore."""

    @pytest.fixture
    def character_store(self, tmp_path: Path) -> CharacterStore:
        """Create a CharacterStore with temp storage."""
        campaign_dir = tmp_path / "test-campaign"
        campaign_dir.mkdir(parents=True)
        return CharacterStore("test-campaign", base_dir=tmp_path)

    def test_create_character(self, character_store: CharacterStore):
        """create should store character profile."""
        profile = character_store.create(
            name="Test NPC",
            character_type="npc",
            personality="Friendly",
        )

        assert profile.id is not None
        assert profile.name == "Test NPC"
        assert profile.personality == "Friendly"

    def test_get_character(self, character_store: CharacterStore):
        """get should retrieve character by ID."""
        created = character_store.create(name="Findable NPC")

        found = character_store.get(created.id)

        assert found is not None
        assert found.name == "Findable NPC"

    def test_get_by_name(self, character_store: CharacterStore):
        """get_by_name should find character case-insensitively."""
        character_store.create(name="Sheriff Hemlock")

        found = character_store.get_by_name("sheriff hemlock")

        assert found is not None
        assert found.name == "Sheriff Hemlock"

    def test_get_nonexistent(self, character_store: CharacterStore):
        """get should return None for missing character."""
        result = character_store.get("nonexistent")
        assert result is None

    def test_list_characters(self, character_store: CharacterStore):
        """list should return all characters."""
        character_store.create(name="NPC 1", character_type="npc")
        character_store.create(name="Monster 1", character_type="monster")
        character_store.create(name="Player 1", character_type="player")

        all_chars = character_store.list()

        assert len(all_chars) == 3

    def test_list_by_type(self, character_store: CharacterStore):
        """list should filter by character type."""
        character_store.create(name="NPC 1", character_type="npc")
        character_store.create(name="NPC 2", character_type="npc")
        character_store.create(name="Monster 1", character_type="monster")

        npcs = character_store.list(character_type="npc")
        monsters = character_store.list(character_type="monster")

        assert len(npcs) == 2
        assert len(monsters) == 1

    def test_update_character(self, character_store: CharacterStore):
        """update should modify character profile."""
        profile = character_store.create(name="Updatable", personality="Old personality")

        profile.personality = "New personality"
        character_store.update(profile)

        reloaded = character_store.get(profile.id)
        assert reloaded.personality == "New personality"

    def test_delete_character(self, character_store: CharacterStore):
        """delete should remove character."""
        profile = character_store.create(name="Deletable")

        result = character_store.delete(profile.id)

        assert result is True
        assert character_store.get(profile.id) is None

    def test_delete_nonexistent(self, character_store: CharacterStore):
        """delete should return False for missing character."""
        result = character_store.delete("nonexistent")
        assert result is False

    def test_persistence(self, tmp_path: Path):
        """Characters should persist across store instances."""
        campaign_dir = tmp_path / "persist-campaign"
        campaign_dir.mkdir(parents=True)

        # Create with first store instance
        store1 = CharacterStore("persist-campaign", base_dir=tmp_path)
        store1.create(name="Persistent NPC", personality="Memorable")

        # Read with second store instance
        store2 = CharacterStore("persist-campaign", base_dir=tmp_path)
        found = store2.get_by_name("Persistent NPC")

        assert found is not None
        assert found.personality == "Memorable"


class TestCharacterRunnerServer:
    """Tests for CharacterRunnerServer."""

    @pytest.fixture
    def runner_setup(self, tmp_path: Path):
        """Set up CharacterRunnerServer with mock LLM."""
        campaign_dir = tmp_path / "runner-campaign"
        campaign_dir.mkdir(parents=True)

        mock_llm = MockLLM("Test NPC response")

        # Patch the CAMPAIGNS_DIR
        import gm_agent.mcp.character_runner as cr_module

        original_dir = cr_module.CAMPAIGNS_DIR
        cr_module.CAMPAIGNS_DIR = tmp_path

        server = CharacterRunnerServer("runner-campaign", llm=mock_llm)

        yield {
            "server": server,
            "mock_llm": mock_llm,
            "tmp_path": tmp_path,
        }

        cr_module.CAMPAIGNS_DIR = original_dir

    def test_list_tools(self, runner_setup):
        """Server should list all expected tools."""
        server = runner_setup["server"]
        tools = server.list_tools()

        tool_names = [t.name for t in tools]
        assert "run_npc" in tool_names
        assert "run_monster" in tool_names
        assert "emulate_player" in tool_names
        assert "create_character" in tool_names
        assert "get_character" in tool_names
        assert "list_characters" in tool_names

    def test_create_character_tool(self, runner_setup):
        """create_character tool should create character profiles."""
        server = runner_setup["server"]

        result = server.call_tool(
            "create_character",
            {
                "name": "Test Innkeeper",
                "character_type": "npc",
                "personality": "Cheerful and helpful",
                "knowledge": "Local rumors, Best dishes",
                "goals": "Keep customers happy",
            },
        )

        assert result.success
        assert "Test Innkeeper" in result.data

    def test_create_character_duplicate(self, runner_setup):
        """create_character should reject duplicates."""
        server = runner_setup["server"]

        server.call_tool("create_character", {"name": "Unique NPC", "character_type": "npc"})
        result = server.call_tool(
            "create_character", {"name": "Unique NPC", "character_type": "npc"}
        )

        assert not result.success
        assert "already exists" in result.error

    def test_get_character_tool(self, runner_setup):
        """get_character tool should return profile details."""
        server = runner_setup["server"]

        server.call_tool(
            "create_character",
            {
                "name": "Sheriff",
                "character_type": "npc",
                "personality": "Stern but fair",
            },
        )

        result = server.call_tool("get_character", {"name": "Sheriff"})

        assert result.success
        assert "Sheriff" in result.data
        assert "Stern but fair" in result.data

    def test_get_character_not_found(self, runner_setup):
        """get_character should error for missing character."""
        server = runner_setup["server"]

        result = server.call_tool("get_character", {"name": "Nobody"})

        assert not result.success
        assert "not found" in result.error

    def test_list_characters_tool(self, runner_setup):
        """list_characters tool should list all characters."""
        server = runner_setup["server"]

        server.call_tool("create_character", {"name": "NPC1", "character_type": "npc"})
        server.call_tool("create_character", {"name": "Monster1", "character_type": "monster"})

        result = server.call_tool("list_characters", {})

        assert result.success
        assert "NPC1" in result.data
        assert "Monster1" in result.data

    def test_list_characters_by_type(self, runner_setup):
        """list_characters should filter by type."""
        server = runner_setup["server"]

        server.call_tool("create_character", {"name": "NPC1", "character_type": "npc"})
        server.call_tool("create_character", {"name": "Monster1", "character_type": "monster"})

        result = server.call_tool("list_characters", {"character_type": "monster"})

        assert result.success
        assert "Monster1" in result.data
        assert "NPC1" not in result.data

    def test_run_npc(self, runner_setup):
        """run_npc should embody NPC and return response."""
        server = runner_setup["server"]
        mock_llm = runner_setup["mock_llm"]
        mock_llm.response_text = "Aye, I've heard tell of strange lights in the old mine."

        server.call_tool(
            "create_character",
            {
                "name": "Old Miner",
                "character_type": "npc",
                "personality": "Superstitious and wary",
                "knowledge": "Mining lore, Local legends",
            },
        )

        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Old Miner",
                "player_input": "Have you seen anything strange lately?",
                "context": "Dusty tavern, evening",
            },
        )

        assert result.success
        assert "Old Miner" in result.data
        assert "strange lights" in result.data

        # Verify LLM was called with proper system prompt
        assert len(mock_llm.calls) == 1
        messages = mock_llm.calls[0][0]
        system_msg = messages[0].content
        assert "Old Miner" in system_msg
        assert "Superstitious" in system_msg

    def test_run_npc_not_found(self, runner_setup):
        """run_npc should error if NPC doesn't exist."""
        server = runner_setup["server"]

        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Nonexistent NPC",
                "player_input": "Hello?",
            },
        )

        assert not result.success
        assert "not found" in result.error

    def test_run_npc_wrong_type(self, runner_setup):
        """run_npc should reject non-NPC characters."""
        server = runner_setup["server"]

        server.call_tool("create_character", {"name": "Goblin", "character_type": "monster"})

        result = server.call_tool(
            "run_npc",
            {
                "npc_name": "Goblin",
                "player_input": "Hello",
            },
        )

        assert not result.success
        assert "monster" in result.error

    def test_run_monster(self, runner_setup):
        """run_monster should return tactical decisions."""
        server = runner_setup["server"]
        mock_llm = runner_setup["mock_llm"]
        mock_llm.response_text = (
            "The goblin retreats to alert its allies, using the shadows for cover."
        )

        server.call_tool(
            "create_character",
            {
                "name": "Scout Goblin",
                "character_type": "monster",
                "intelligence": "low",
                "instincts": "cowardly, alert allies",
                "morale": "Flees when outnumbered",
            },
        )

        result = server.call_tool(
            "run_monster",
            {
                "monster_name": "Scout Goblin",
                "situation": "Party of 4 adventurers spotted, goblin is alone",
            },
        )

        assert result.success
        assert "Scout Goblin" in result.data
        assert "retreats" in result.data

    def test_run_monster_wrong_type(self, runner_setup):
        """run_monster should reject non-monster characters."""
        server = runner_setup["server"]

        server.call_tool("create_character", {"name": "Barkeep", "character_type": "npc"})

        result = server.call_tool(
            "run_monster",
            {
                "monster_name": "Barkeep",
                "situation": "Combat",
            },
        )

        assert not result.success
        assert "npc" in result.error

    def test_emulate_player(self, runner_setup):
        """emulate_player should act as player character."""
        server = runner_setup["server"]
        mock_llm = runner_setup["mock_llm"]
        mock_llm.response_text = "I carefully check the chest for traps before opening it."

        server.call_tool(
            "create_character",
            {
                "name": "Cautious Carl",
                "character_type": "player",
                "playstyle": "Always checks for traps",
                "risk_tolerance": "low",
                "party_role": "Scout",
            },
        )

        result = server.call_tool(
            "emulate_player",
            {
                "player_name": "Cautious Carl",
                "scenario": "The party finds a treasure chest in the dungeon",
            },
        )

        assert result.success
        assert "Cautious Carl" in result.data
        assert "traps" in result.data

    def test_emulate_player_wrong_type(self, runner_setup):
        """emulate_player should reject non-player characters."""
        server = runner_setup["server"]

        server.call_tool("create_character", {"name": "Guard", "character_type": "npc"})

        result = server.call_tool(
            "emulate_player",
            {
                "player_name": "Guard",
                "scenario": "Test",
            },
        )

        assert not result.success
        assert "npc" in result.error

    def test_run_npc_no_llm(self, tmp_path: Path):
        """run_npc should error without LLM configured."""
        campaign_dir = tmp_path / "no-llm-campaign"
        campaign_dir.mkdir(parents=True)

        import gm_agent.mcp.character_runner as cr_module

        original_dir = cr_module.CAMPAIGNS_DIR
        cr_module.CAMPAIGNS_DIR = tmp_path

        server = CharacterRunnerServer("no-llm-campaign", llm=None)
        server.call_tool("create_character", {"name": "Test", "character_type": "npc"})

        result = server.call_tool("run_npc", {"npc_name": "Test", "player_input": "Hi"})

        assert not result.success
        assert "No LLM" in result.error

        cr_module.CAMPAIGNS_DIR = original_dir

    def test_unknown_tool(self, runner_setup):
        """Server should return error for unknown tool."""
        server = runner_setup["server"]

        result = server.call_tool("nonexistent_tool", {})

        assert not result.success
        assert "Unknown tool" in result.error
