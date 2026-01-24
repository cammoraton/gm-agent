"""Tests for the NPC Builder MCP server."""

import json
from unittest.mock import Mock, MagicMock, patch
import pytest

from gm_agent.mcp.npc_builder import NPCBuilderServer, EXTRACTION_PROMPT
from gm_agent.mcp.base import ToolResult
from gm_agent.storage.schemas import CharacterProfile
from gm_agent.models.base import Message, LLMResponse


class TestNPCBuilderServer:
    """Tests for NPCBuilderServer."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM backend."""
        llm = Mock()
        # Mock a successful extraction response
        llm.chat.return_value = LLMResponse(
            text='```json\n{"character_type": "npc", "personality": "Cunning and ambitious", "speech_patterns": "Speaks in riddles", "knowledge": ["Secret passage location"], "goals": ["Gain power"], "secrets": ["Working for enemies"]}\n```',
            tool_calls=[]
        )
        return llm

    @pytest.fixture
    def mock_rag(self):
        """Create a mock RAG server."""
        with patch("gm_agent.mcp.npc_builder.PF2eRAGServer") as MockRAG:
            rag_instance = MockRAG.return_value
            # Mock successful RAG lookups
            rag_instance.call_tool.return_value = ToolResult(
                success=True,
                data="Test NPC data from Pathfinder content"
            )
            yield MockRAG

    @pytest.fixture
    def server(self, mock_llm, mock_rag, tmp_path):
        """Create an NPCBuilderServer with mocked dependencies."""
        with patch("gm_agent.mcp.npc_builder.CAMPAIGNS_DIR", tmp_path):
            server = NPCBuilderServer(campaign_id="test-campaign", llm=mock_llm)
            return server

    def test_initialization(self, server):
        """Test server initializes correctly."""
        assert server.campaign_id == "test-campaign"
        assert server.llm is not None
        assert server._character_store is not None
        assert server._rag_server is not None

    def test_list_tools(self, server):
        """Test listing available tools."""
        tools = server.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "build_npc_profile"
        assert "NPC or monster" in tools[0].description

    def test_build_npc_profile_no_llm(self, tmp_path):
        """Test that building without LLM returns error."""
        with patch("gm_agent.mcp.npc_builder.CAMPAIGNS_DIR", tmp_path):
            server = NPCBuilderServer(campaign_id="test-campaign", llm=None)
            result = server.call_tool("build_npc_profile", {"name": "Test NPC"})

        assert not result.success
        assert "LLM backend required" in result.error

    def test_build_npc_profile_no_rag_data(self, server, mock_rag):
        """Test that building with no RAG data returns error."""
        # Mock RAG server to return no data
        mock_rag.return_value.call_tool.return_value = ToolResult(
            success=False,
            data=""
        )

        result = server.call_tool("build_npc_profile", {"name": "Unknown NPC"})

        assert not result.success
        assert "No information found" in result.error

    def test_build_npc_profile_success(self, server):
        """Test successful NPC profile building."""
        result = server.call_tool("build_npc_profile", {"name": "Voz Lirayne"})

        assert result.success
        assert "Character profile created" in result.data or "Character profile updated" in result.data
        assert "Voz Lirayne" in result.data

        # Verify character was saved
        character = server._character_store.get_by_name("Voz Lirayne")
        assert character is not None
        assert character.name == "Voz Lirayne"
        assert character.personality == "Cunning and ambitious"
        assert character.speech_patterns == "Speaks in riddles"
        assert "Secret passage location" in character.knowledge
        assert "Gain power" in character.goals
        assert "Working for enemies" in character.secrets

    def test_build_npc_profile_already_exists(self, server):
        """Test that building when profile exists returns info without rebuilding."""
        # Create profile first
        server.call_tool("build_npc_profile", {"name": "Voz Lirayne"})

        # Try to build again
        result = server.call_tool("build_npc_profile", {"name": "Voz Lirayne"})

        assert result.success
        assert "already exists" in result.data
        assert "force_rebuild" in result.data

    def test_build_npc_profile_force_rebuild(self, server, mock_llm):
        """Test force rebuild updates existing profile."""
        # Create profile first
        server.call_tool("build_npc_profile", {"name": "Test NPC"})

        # Mock LLM to return different data
        mock_llm.chat.return_value = LLMResponse(
            text='```json\n{"character_type": "npc", "personality": "Updated personality", "speech_patterns": "New speech", "knowledge": [], "goals": [], "secrets": []}\n```',
            tool_calls=[]
        )

        # Force rebuild
        result = server.call_tool(
            "build_npc_profile",
            {"name": "Test NPC", "force_rebuild": True}
        )

        assert result.success
        assert "updated" in result.data

        # Verify profile was updated
        character = server._character_store.get_by_name("Test NPC")
        assert character.personality == "Updated personality"
        assert character.speech_patterns == "New speech"

    def test_build_monster_profile(self, server, mock_llm):
        """Test building a monster profile with monster-specific fields."""
        # Mock LLM to return monster data
        mock_llm.chat.return_value = LLMResponse(
            text='```json\n{"character_type": "monster", "personality": "Aggressive predator", "speech_patterns": "", "knowledge": [], "goals": ["Hunt prey"], "secrets": [], "intelligence": "low", "instincts": ["Attack on sight"], "morale": "Flees at half HP"}\n```',
            tool_calls=[]
        )

        result = server.call_tool("build_npc_profile", {"name": "Goblin Warrior"})

        assert result.success
        character = server._character_store.get_by_name("Goblin Warrior")
        assert character.character_type == "monster"
        assert character.intelligence == "low"
        assert "Attack on sight" in character.instincts
        assert character.morale == "Flees at half HP"

    def test_gather_rag_data(self, server):
        """Test RAG data gathering."""
        data = server._gather_rag_data("Test Character")

        # Should have made calls to RAG server
        assert server._rag_server.call_tool.called
        assert data  # Should have some data
        assert "Test NPC data" in data or "Creature Data" in data or "Content Search" in data

    def test_extract_character_data_json_parsing(self, server, mock_llm):
        """Test extraction handles various JSON formats."""
        # Test with markdown code block
        mock_llm.chat.return_value = LLMResponse(
            text='```json\n{"character_type": "npc", "personality": "Test", "speech_patterns": "", "knowledge": [], "goals": [], "secrets": []}\n```',
            tool_calls=[]
        )

        data = server._extract_character_data("Test", "RAG data")
        assert data["personality"] == "Test"

        # Test without markdown
        mock_llm.chat.return_value = LLMResponse(
            text='{"character_type": "npc", "personality": "Test2", "speech_patterns": "", "knowledge": [], "goals": [], "secrets": []}',
            tool_calls=[]
        )

        data = server._extract_character_data("Test", "RAG data")
        assert data["personality"] == "Test2"

    def test_extract_character_data_fallback(self, server, mock_llm):
        """Test extraction fallback for invalid JSON."""
        # Mock LLM to return invalid JSON
        mock_llm.chat.return_value = LLMResponse(
            text="This is not JSON at all",
            tool_calls=[]
        )

        data = server._extract_character_data("Test NPC", "RAG data")

        # Should return fallback data
        assert data["character_type"] == "npc"
        assert "auto-generated" in data["personality"]
        assert "Failed to parse" in data["notes"]

    def test_unknown_tool(self, server):
        """Test calling unknown tool returns error."""
        result = server.call_tool("unknown_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error

    def test_close_cleanup(self, server):
        """Test server cleanup."""
        server.close()
        # Should not raise exception


class TestNPCBuilderIntegration:
    """Integration tests for NPC Builder."""

    def test_npc_builder_via_mcp_client(self, tmp_path):
        """Test NPC Builder through MCP client."""
        from gm_agent.mcp.client import MCPClient
        from gm_agent.models.base import LLMResponse
        from gm_agent.storage.campaign import CampaignStore

        # Create mock LLM
        mock_llm = Mock()
        mock_llm.chat.return_value = LLMResponse(
            text='```json\n{"character_type": "npc", "personality": "Brave warrior", "speech_patterns": "Direct", "knowledge": ["Combat tactics"], "goals": ["Protect town"], "secrets": []}\n```',
            tool_calls=[]
        )

        # Create a test campaign with proper patching
        with patch("gm_agent.mcp.npc_builder.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.characters.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.mcp.npc_builder.PF2eRAGServer") as MockRAG:
                    MockRAG.return_value.call_tool.return_value = ToolResult(
                        success=True,
                        data="Warrior character data"
                    )

                    # Create campaign manually in tmp_path
                    campaign_store = CampaignStore(base_dir=tmp_path)
                    campaign = campaign_store.create("Test NPC Builder Campaign")
                    campaign_id = campaign.id

                    client = MCPClient(
                        mode="local",
                        context={"campaign_id": campaign_id, "llm": mock_llm}
                    )

                    # Verify tool is available
                    tools = client.list_tools()
                    tool_names = [t.name for t in tools]
                    assert "build_npc_profile" in tool_names

                    # Call the tool
                    result = client.call_tool("build_npc_profile", {"name": "Test Warrior"})

                    assert result.success
                    assert "Test Warrior" in result.data


class TestExtractionPrompt:
    """Tests for the extraction prompt."""

    def test_extraction_prompt_exists(self):
        """Test that extraction prompt is defined."""
        assert EXTRACTION_PROMPT
        assert "JSON" in EXTRACTION_PROMPT
        assert "personality" in EXTRACTION_PROMPT
        assert "character_type" in EXTRACTION_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
