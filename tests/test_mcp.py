"""Integration tests for MCP tools and RAG wrapper."""

import pytest

from gm_agent.mcp.base import ToolDef, ToolParameter, ToolResult, MCPServer

# Import the mock from conftest
from tests.conftest import MockMCPServer, MockPathfinderSearch


class TestToolParameter:
    """Tests for ToolParameter model."""

    def test_required_parameter(self):
        """ToolParameter should default to required=True."""
        param = ToolParameter(
            name="query",
            type="string",
            description="The search query",
        )
        assert param.required is True
        assert param.default is None

    def test_optional_parameter(self):
        """ToolParameter should support optional parameters with defaults."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_all_types(self):
        """ToolParameter should accept various type strings."""
        for type_name in ["string", "integer", "boolean", "number", "array"]:
            param = ToolParameter(
                name="test",
                type=type_name,
                description="Test param",
            )
            assert param.type == type_name


class TestToolDef:
    """Tests for ToolDef model."""

    def test_basic_tool(self):
        """ToolDef should store name, description, and parameters."""
        tool = ToolDef(
            name="lookup_creature",
            description="Look up a creature by name",
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="The creature name",
                )
            ],
        )
        assert tool.name == "lookup_creature"
        assert "creature" in tool.description
        assert len(tool.parameters) == 1

    def test_to_ollama_format_simple(self):
        """ToolDef.to_ollama_format should convert simple tool."""
        tool = ToolDef(
            name="lookup_creature",
            description="Look up a creature",
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Creature name",
                )
            ],
        )

        ollama_format = tool.to_ollama_format()

        assert ollama_format["type"] == "function"
        assert ollama_format["function"]["name"] == "lookup_creature"
        assert ollama_format["function"]["description"] == "Look up a creature"
        assert ollama_format["function"]["parameters"]["type"] == "object"
        assert "name" in ollama_format["function"]["parameters"]["properties"]
        assert "name" in ollama_format["function"]["parameters"]["required"]

    def test_to_ollama_format_multiple_params(self):
        """ToolDef.to_ollama_format should handle multiple parameters."""
        tool = ToolDef(
            name="search_content",
            description="Search content",
            parameters=[
                ToolParameter(name="query", type="string", description="Query"),
                ToolParameter(
                    name="types",
                    type="string",
                    description="Types filter",
                    required=False,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                    default=10,
                ),
            ],
        )

        ollama_format = tool.to_ollama_format()

        properties = ollama_format["function"]["parameters"]["properties"]
        required = ollama_format["function"]["parameters"]["required"]

        assert len(properties) == 3
        assert "query" in properties
        assert "types" in properties
        assert "limit" in properties

        # Only required params in required list
        assert required == ["query"]
        assert "types" not in required
        assert "limit" not in required

    def test_to_ollama_format_no_params(self):
        """ToolDef.to_ollama_format should handle tools with no parameters."""
        tool = ToolDef(
            name="get_status",
            description="Get current status",
            parameters=[],
        )

        ollama_format = tool.to_ollama_format()

        assert ollama_format["function"]["parameters"]["properties"] == {}
        assert ollama_format["function"]["parameters"]["required"] == []


class TestToolResult:
    """Tests for ToolResult model."""

    def test_success_result(self):
        """ToolResult should store successful results."""
        result = ToolResult(success=True, data="Goblin info here")
        assert result.success is True
        assert result.data == "Goblin info here"
        assert result.error is None

    def test_error_result(self):
        """ToolResult should store error results."""
        result = ToolResult(success=False, error="Tool not found")
        assert result.success is False
        assert result.data is None
        assert result.error == "Tool not found"

    def test_to_string_success(self):
        """ToolResult.to_string should return data for success."""
        result = ToolResult(success=True, data="The answer is 42")
        assert result.to_string() == "The answer is 42"

    def test_to_string_error(self):
        """ToolResult.to_string should return error message for failure."""
        result = ToolResult(success=False, error="Connection failed")
        assert result.to_string() == "Error: Connection failed"

    def test_to_string_list_data(self):
        """ToolResult.to_string should join list data."""
        result = ToolResult(success=True, data=["Item 1", "Item 2", "Item 3"])
        output = result.to_string()
        assert "Item 1" in output
        assert "Item 2" in output
        assert "Item 3" in output

    def test_to_string_non_string_data(self):
        """ToolResult.to_string should convert non-string data."""
        result = ToolResult(success=True, data={"key": "value"})
        output = result.to_string()
        assert "key" in output


class TestMockMCPServer:
    """Tests for MockMCPServer (validates test infrastructure)."""

    def test_list_tools_returns_five(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return 5 tools."""
        tools = mock_mcp_server.list_tools()
        assert len(tools) == 7

        tool_names = [t.name for t in tools]
        assert "lookup_creature" in tool_names
        assert "lookup_spell" in tool_names
        assert "lookup_item" in tool_names
        assert "search_rules" in tool_names
        assert "search_content" in tool_names

    def test_get_tool_by_name(self, mock_mcp_server: MockMCPServer):
        """MCPServer.get_tool should find tool by name."""
        tool = mock_mcp_server.get_tool("lookup_creature")
        assert tool is not None
        assert tool.name == "lookup_creature"

    def test_get_tool_nonexistent(self, mock_mcp_server: MockMCPServer):
        """MCPServer.get_tool should return None for missing tool."""
        tool = mock_mcp_server.get_tool("nonexistent")
        assert tool is None

    def test_call_lookup_creature(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return mock creature data."""
        result = mock_mcp_server.call_tool("lookup_creature", {"name": "goblin"})

        assert result.success is True
        assert "Goblin" in result.data
        assert "creature" in result.data

    def test_call_lookup_spell(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return mock spell data."""
        result = mock_mcp_server.call_tool("lookup_spell", {"name": "fireball"})

        assert result.success is True
        assert "Fireball" in result.data
        assert "spell" in result.data

    def test_call_lookup_item(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return mock item data."""
        result = mock_mcp_server.call_tool("lookup_item", {"name": "longsword"})

        assert result.success is True
        assert "Longsword" in result.data

    def test_call_search_rules(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return mock rules data."""
        result = mock_mcp_server.call_tool("search_rules", {"query": "flanking"})

        assert result.success is True
        assert "Flanking" in result.data

    def test_call_search_content(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return mock search results."""
        result = mock_mcp_server.call_tool(
            "search_content",
            {"query": "fire", "types": "spell", "limit": 5},
        )

        assert result.success is True

    def test_call_unknown_tool(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should return error for unknown tool."""
        result = mock_mcp_server.call_tool("unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_calls_are_recorded(self, mock_mcp_server: MockMCPServer):
        """MockMCPServer should record all calls."""
        mock_mcp_server.call_tool("lookup_creature", {"name": "goblin"})
        mock_mcp_server.call_tool("lookup_spell", {"name": "fireball"})

        assert len(mock_mcp_server.calls) == 2
        assert mock_mcp_server.calls[0] == ("lookup_creature", {"name": "goblin"})
        assert mock_mcp_server.calls[1] == ("lookup_spell", {"name": "fireball"})

    def test_custom_tool_results(self, mock_mcp_with_errors: MockMCPServer):
        """MockMCPServer should use custom tool results."""
        result = mock_mcp_with_errors.call_tool("lookup_creature", {"name": "goblin"})

        assert result.success is False
        assert "Database connection failed" in result.error


class TestPF2eRAGServerWithMock:
    """Tests for PF2eRAGServer using mocked PathfinderSearch."""

    def test_list_tools(self, mock_pathfinder_search: MockPathfinderSearch):
        """PF2eRAGServer should list all tools."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        tools = server.list_tools()

        assert len(tools) == 18
        tool_names = [t.name for t in tools]
        assert "lookup_creature" in tool_names
        assert "list_entities" in tool_names
        assert "get_player_advice" in tool_names
        assert "lookup_spell" in tool_names
        assert "lookup_item" in tool_names
        assert "lookup_npc" in tool_names
        assert "search_rules" in tool_names
        assert "search_content" in tool_names
        assert "search_pages" in tool_names
        assert "get_db_stats" in tool_names
        assert "find_page" in tool_names
        assert "browse_book" in tool_names

    def test_tool_definitions_valid(self, mock_pathfinder_search: MockPathfinderSearch):
        """All PF2eRAGServer tools should have valid definitions."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        tools = server.list_tools()

        for tool in tools:
            # Should be convertible to Ollama format
            ollama_format = tool.to_ollama_format()
            assert ollama_format["type"] == "function"
            assert "name" in ollama_format["function"]
            assert "description" in ollama_format["function"]
            assert "parameters" in ollama_format["function"]

    def test_lookup_creature_calls_search(self, mock_pathfinder_search: MockPathfinderSearch):
        """lookup_creature should call PathfinderSearch correctly."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_creature", {"name": "goblin"})

        assert result.success is True
        assert len(mock_pathfinder_search.calls) >= 1
        # First call should be for creature lookup
        query, kwargs = mock_pathfinder_search.calls[0]
        assert query == "goblin"

    def test_lookup_spell_calls_search(self, mock_pathfinder_search: MockPathfinderSearch):
        """lookup_spell should call PathfinderSearch with spell types."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_spell", {"name": "fireball"})

        assert result.success is True
        assert len(mock_pathfinder_search.calls) >= 1

    def test_lookup_item_calls_search(self, mock_pathfinder_search: MockPathfinderSearch):
        """lookup_item should call PathfinderSearch with item types."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_item", {"name": "longsword"})

        assert result.success is True

    def test_search_rules_calls_search(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_rules should call PathfinderSearch with rules filters."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_rules", {"query": "flanking", "limit": 5})

        assert result.success is True

    def test_search_content_with_types(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content should parse comma-separated types."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "fire", "types": "spell,cantrip", "limit": 5},
        )

        assert result.success is True

    def test_search_content_without_types(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content should work without type filter."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "healing", "limit": 10},
        )

        assert result.success is True

    def test_unknown_tool_error(self, mock_pathfinder_search: MockPathfinderSearch):
        """PF2eRAGServer should return error for unknown tool."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_format_results(self, mock_pathfinder_search: MockPathfinderSearch):
        """PF2eRAGServer should format results with headers including book."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_creature", {"name": "goblin"})

        # Should contain formatted result with book info
        assert result.success is True
        output = result.data if isinstance(result.data, str) else result.to_string()
        assert "Goblin" in output
        assert "Core Rulebook" in output  # book field from mock

    def test_search_pages(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_pages should call PathfinderSearch.search_pages."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_pages", {"query": "goblin tactics"})

        assert result.success is True
        output = result.data if isinstance(result.data, str) else result.to_string()
        assert "Core Rulebook" in output

    def test_get_db_stats(self, mock_pathfinder_search: MockPathfinderSearch):
        """get_db_stats should return formatted statistics."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("get_db_stats", {})

        assert result.success is True
        output = result.data if isinstance(result.data, str) else result.to_string()
        assert "1,000" in output  # total_entities
        assert "schema v4" in output

    def test_lookup_npc(self, mock_pathfinder_search: MockPathfinderSearch):
        """lookup_npc should search both NPC and creature categories and merge results."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_npc", {"name": "Goblin"})

        assert result.success is True
        # Should have searched multiple times (npc category, creature category, pages)
        assert len(mock_pathfinder_search.calls) >= 2

    def test_get_tool_method(self, mock_pathfinder_search: MockPathfinderSearch):
        """PF2eRAGServer.get_tool should find tools by name."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")

        tool = server.get_tool("lookup_creature")
        assert tool is not None
        assert tool.name == "lookup_creature"

        tool = server.get_tool("nonexistent")
        assert tool is None

    def test_find_page(self, mock_pathfinder_search: MockPathfinderSearch):
        """find_page should return page references for a term."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("find_page", {"term": "Fireball"})

        assert result.success is True
        output = result.data
        assert "Fireball" in output
        assert "p.326" in output
        assert "Player Core" in output

    def test_find_page_with_book(self, mock_pathfinder_search: MockPathfinderSearch):
        """find_page should accept optional book filter."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("find_page", {"term": "Fireball", "book": "Player Core"})

        assert result.success is True

    def test_browse_book_list(self, mock_pathfinder_search: MockPathfinderSearch):
        """browse_book with no args lists all books."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("browse_book", {})

        assert result.success is True
        output = result.data
        assert "Player Core" in output
        assert "GM Core" in output
        assert "Monster Core" in output
        assert "Available Books" in output

    def test_browse_book_filter_by_type(self, mock_pathfinder_search: MockPathfinderSearch):
        """browse_book with book_type filters the book list."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("browse_book", {"book_type": "bestiary"})

        assert result.success is True
        output = result.data
        assert "Monster Core" in output
        assert "Player Core" not in output

    def test_browse_book_chapters(self, mock_pathfinder_search: MockPathfinderSearch):
        """browse_book with book shows book summary + chapter TOC."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("browse_book", {"book": "Player Core"})

        assert result.success is True
        output = result.data
        assert "Player Core" in output
        assert "Table of Contents" in output
        assert "Introduction" in output
        assert "Spells" in output

    def test_browse_book_chapter_summary(self, mock_pathfinder_search: MockPathfinderSearch):
        """browse_book with book+chapter shows chapter summary + page summaries."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "browse_book",
            {"book": "Player Core", "chapter": "Spells"},
        )

        assert result.success is True
        output = result.data
        # Chapter summary info
        assert "Spells" in output
        assert "280" in output
        assert "420" in output
        # Page-level summaries
        assert "p.280" in output
        assert "p.281" in output

    def test_browse_book_resolves_name(self, mock_pathfinder_search: MockPathfinderSearch):
        """browse_book resolves user-friendly book names."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        # Should resolve "Player" to "Player Core" via contains match
        result = server.call_tool("browse_book", {"book": "Player"})

        assert result.success is True
        output = result.data
        assert "Player Core" in output


class TestToolDefConversions:
    """Tests for converting ToolDef between formats."""

    def test_ollama_format_parameter_types(self):
        """Ollama format should preserve parameter type information."""
        tool = ToolDef(
            name="test_tool",
            description="Test",
            parameters=[
                ToolParameter(name="str_param", type="string", description="String"),
                ToolParameter(name="int_param", type="integer", description="Integer"),
                ToolParameter(name="bool_param", type="boolean", description="Boolean"),
            ],
        )

        ollama = tool.to_ollama_format()
        props = ollama["function"]["parameters"]["properties"]

        assert props["str_param"]["type"] == "string"
        assert props["int_param"]["type"] == "integer"
        assert props["bool_param"]["type"] == "boolean"

    def test_ollama_format_descriptions_preserved(self):
        """Ollama format should preserve parameter descriptions."""
        tool = ToolDef(
            name="search",
            description="Search for content",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to execute",
                )
            ],
        )

        ollama = tool.to_ollama_format()
        query_prop = ollama["function"]["parameters"]["properties"]["query"]

        assert query_prop["description"] == "The search query to execute"

    def test_all_tools_convertible(self, mock_mcp_server: MockMCPServer):
        """All tools from MCP server should be convertible to Ollama format."""
        tools = mock_mcp_server.list_tools()

        for tool in tools:
            ollama = tool.to_ollama_format()

            # Validate structure
            assert "type" in ollama
            assert "function" in ollama
            assert "name" in ollama["function"]
            assert "description" in ollama["function"]
            assert "parameters" in ollama["function"]
            assert "type" in ollama["function"]["parameters"]
            assert "properties" in ollama["function"]["parameters"]
            assert "required" in ollama["function"]["parameters"]


class TestMetadataFiltering:
    """Tests for metadata-aware search (level, level_range, traits)."""

    @pytest.fixture
    def search_with_creatures(self, mock_pathfinder_search: MockPathfinderSearch):
        """Set up MockPathfinderSearch with diverse creature results."""
        mock_pathfinder_search._search_results = [
            {
                "name": "Zombie Shambler",
                "type": "creature",
                "category": "creature",
                "source": "Monster Core",
                "book": "Monster Core",
                "book_type": "bestiary",
                "page": 10,
                "content": "A shambling undead creature",
                "metadata": {"level": -1, "traits": ["undead", "mindless"]},
                "score": 8.0,
            },
            {
                "name": "Skeletal Champion",
                "type": "creature",
                "category": "creature",
                "source": "Monster Core",
                "book": "Monster Core",
                "book_type": "bestiary",
                "page": 20,
                "content": "An undead skeletal warrior",
                "metadata": {"level": 5, "traits": ["undead"]},
                "score": 7.0,
            },
            {
                "name": "Vampire Count",
                "type": "creature",
                "category": "creature",
                "source": "Monster Core",
                "book": "Monster Core",
                "book_type": "bestiary",
                "page": 30,
                "content": "A powerful vampire noble",
                "metadata": {"level": 9, "traits": ["undead", "vampire"]},
                "score": 6.0,
            },
            {
                "name": "Fire Giant",
                "type": "creature",
                "category": "creature",
                "source": "Monster Core",
                "book": "Monster Core",
                "book_type": "bestiary",
                "page": 40,
                "content": "A massive fire-breathing giant",
                "metadata": {"level": 10, "traits": ["giant", "fire"]},
                "score": 5.0,
            },
            {
                "name": "Innkeeper",
                "type": "creature",
                "category": "creature",
                "source": "NPC Core",
                "book": "NPC Core",
                "book_type": "npc",
                "page": 50,
                "content": "A common innkeeper NPC",
                "metadata": {"level": 1, "traits": ["human", "humanoid"]},
                "score": 4.0,
            },
        ]
        return mock_pathfinder_search

    def test_search_level_exact(self, search_with_creatures):
        """search() with level=5 returns only level 5 entities."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "undead", "level": 5},
        )
        assert result.success is True
        # Check that the mock was called with level param
        query, kwargs = search_with_creatures.calls[-1]
        assert kwargs.get("level") == 5
        # Verify filtering worked
        assert "Skeletal Champion" in result.data
        assert "Zombie Shambler" not in result.data
        assert "Vampire Count" not in result.data

    def test_search_level_range(self, search_with_creatures):
        """search() with level_range filters to level range."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "creature", "level_range": "5-10"},
        )
        assert result.success is True
        query, kwargs = search_with_creatures.calls[-1]
        assert kwargs.get("level_range") == (5, 10)
        # Should include level 5, 9, and 10
        assert "Skeletal Champion" in result.data
        assert "Vampire Count" in result.data
        assert "Fire Giant" in result.data
        # Should exclude level -1 and 1
        assert "Zombie Shambler" not in result.data
        assert "Innkeeper" not in result.data

    def test_search_traits_single(self, search_with_creatures):
        """search() with traits filters to entities with that trait."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "creature", "traits": "undead"},
        )
        assert result.success is True
        query, kwargs = search_with_creatures.calls[-1]
        assert kwargs.get("traits") == ["undead"]
        # All undead
        assert "Zombie Shambler" in result.data
        assert "Skeletal Champion" in result.data
        assert "Vampire Count" in result.data
        # Not undead
        assert "Fire Giant" not in result.data
        assert "Innkeeper" not in result.data

    def test_search_traits_multiple(self, search_with_creatures):
        """search() with multiple traits requires ALL to match."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "creature", "traits": "undead,vampire"},
        )
        assert result.success is True
        query, kwargs = search_with_creatures.calls[-1]
        assert kwargs.get("traits") == ["undead", "vampire"]
        # Only vampire count has both
        assert "Vampire Count" in result.data
        assert "Skeletal Champion" not in result.data

    def test_search_level_plus_traits(self, search_with_creatures):
        """search() with combined level + traits filters."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "creature", "level": 5, "traits": "undead"},
        )
        assert result.success is True
        # Only level 5 undead = Skeletal Champion
        assert "Skeletal Champion" in result.data
        assert "Zombie Shambler" not in result.data
        assert "Vampire Count" not in result.data

    def test_search_level_range_plus_traits(self, search_with_creatures):
        """search() with combined level_range + traits filters."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool(
            "search_content",
            {"query": "creature", "level_range": "1-8", "traits": "undead"},
        )
        assert result.success is True
        # Undead in range 1-8: Skeletal Champion (5)
        assert "Skeletal Champion" in result.data
        # Zombie Shambler is level -1, outside range
        assert "Zombie Shambler" not in result.data
        # Vampire Count is level 9, outside range
        assert "Vampire Count" not in result.data

    def test_search_content_tool_has_new_params(self, mock_pathfinder_search):
        """search_content tool definition should include level, level_range, traits params."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        tool = server.get_tool("search_content")
        param_names = [p.name for p in tool.parameters]
        assert "level" in param_names
        assert "level_range" in param_names
        assert "traits" in param_names

    def test_search_no_metadata_filters_unchanged(self, mock_pathfinder_search):
        """search_content without metadata filters should work as before."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {"query": "goblin"})
        assert result.success is True
        assert "Goblin" in result.data


class TestPartialNameMatchBoost:
    """Tests for partial name-match boost in search scoring."""

    @pytest.fixture
    def search_with_innkeeper(self, mock_pathfinder_search: MockPathfinderSearch):
        """Set up mock with an Innkeeper creature and generic NPC stat blocks."""
        mock_pathfinder_search._search_results = [
            {
                "name": "Innkeeper",
                "type": "creature",
                "category": "creature",
                "source": "NPC Core",
                "book": "NPC Core",
                "book_type": "npc",
                "page": 50,
                "content": "A common innkeeper NPC who runs a tavern",
                "metadata": {"level": 1},
                "score": 4.0,
            },
            {
                "name": "Commoner",
                "type": "creature",
                "category": "creature",
                "source": "NPC Core",
                "book": "NPC Core",
                "book_type": "npc",
                "page": 10,
                "content": "A generic commoner. Can be used as innkeeper, merchant, or farmer.",
                "metadata": {"level": -1},
                "score": 8.0,
            },
        ]
        return mock_pathfinder_search

    def test_partial_name_match_params_passed(self, search_with_innkeeper):
        """Verify search is called — partial name boost is SQL-level in real search."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {"query": "innkeeper"})
        assert result.success is True
        # Both results should be present (mock doesn't apply SQL-level boost)
        assert "Innkeeper" in result.data
        assert "Commoner" in result.data


class TestRemasterOnly:
    """Tests for the remaster_only parameter on search_content."""

    def test_search_content_has_remaster_only_param(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content tool should have remaster_only parameter."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        tool = server.get_tool("search_content")
        param_names = [p.name for p in tool.parameters]
        assert "remaster_only" in param_names

    def test_remaster_only_passes_is_remaster(self, mock_pathfinder_search: MockPathfinderSearch):
        """remaster_only=True should pass is_remaster=True to search."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {
            "query": "dragon",
            "remaster_only": True,
        })
        assert result.success is True
        # Verify the call was made with is_remaster=True
        assert len(mock_pathfinder_search.calls) > 0
        _, kwargs = mock_pathfinder_search.calls[0]
        assert kwargs.get("is_remaster") is True

    def test_remaster_only_false_does_not_filter(self, mock_pathfinder_search: MockPathfinderSearch):
        """remaster_only=False or absent should not pass is_remaster."""
        from gm_agent.mcp.pf2e_rag import PF2eRAGServer

        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {
            "query": "dragon",
        })
        assert result.success is True
        _, kwargs = mock_pathfinder_search.calls[0]
        assert "is_remaster" not in kwargs


class TestInferEntityTypes:
    """Tests for _infer_entity_types() type inference for list queries."""

    def test_no_list_intent(self):
        from gm_agent.rag.search import _infer_entity_types

        assert _infer_entity_types("dragon") is None
        assert _infer_entity_types("fire damage") is None
        assert _infer_entity_types("how does flanking work") is None

    def test_types_of_dragon(self):
        from gm_agent.rag.search import _infer_entity_types

        result = _infer_entity_types("types of dragon")
        assert result is not None
        assert "creature" in result
        assert "creature_family" in result

    def test_kinds_of_undead(self):
        from gm_agent.rag.search import _infer_entity_types

        result = _infer_entity_types("kinds of undead")
        assert result is not None
        assert "creature" in result

    def test_list_of_elementals(self):
        from gm_agent.rag.search import _infer_entity_types

        result = _infer_entity_types("list of elemental creatures")
        assert result is not None
        assert "creature" in result

    def test_all_the_demons(self):
        from gm_agent.rag.search import _infer_entity_types

        result = _infer_entity_types("what are the different types of demon")
        assert result is not None
        assert "creature" in result

    def test_creature_family_name_match(self):
        from gm_agent.rag.search import _infer_entity_types

        families = {"drake", "linnorm", "sphinx"}
        result = _infer_entity_types("types of drake", creature_family_names=families)
        assert result is not None
        assert "creature" in result

    def test_no_creature_keyword_no_match(self):
        from gm_agent.rag.search import _infer_entity_types

        # "types of damage" has list intent but no creature keyword
        result = _infer_entity_types("types of damage")
        assert result is None

    def test_what_are_the_different_types(self):
        from gm_agent.rag.search import _infer_entity_types

        result = _infer_entity_types("what are the different types of giant")
        assert result is not None
        assert "creature" in result


class TestBoostTuning:
    """Verify game_mechanic and guidance boosts were lowered."""

    def test_game_mechanic_boost_lowered(self):
        from gm_agent.rag.search import PathfinderSearch
        assert PathfinderSearch.DEFAULT_TYPE_BOOST["game_mechanic"] == 10

    def test_guidance_boost_lowered(self):
        from gm_agent.rag.search import PathfinderSearch
        assert PathfinderSearch.DEFAULT_TYPE_BOOST["guidance"] == 12

    def test_creature_boost_still_high(self):
        from gm_agent.rag.search import PathfinderSearch
        assert PathfinderSearch.DEFAULT_TYPE_BOOST["creature"] == 25

    def test_spell_boost_still_high(self):
        from gm_agent.rag.search import PathfinderSearch
        assert PathfinderSearch.DEFAULT_TYPE_BOOST["spell"] == 20


class TestInformalSynonyms:
    """Tests for informal synonym expansion (aliases normally loaded from DB)."""

    @pytest.fixture(autouse=True)
    def _patch_remaster_aliases(self):
        """Populate REMASTER_ALIASES with test data (normally loaded from DB at runtime)."""
        import gm_agent.rag.search as search_mod
        test_aliases = {
            "grandmother": "granny",
            "grandma": "granny",
        }
        orig_aliases = dict(search_mod.REMASTER_ALIASES)
        orig_index = search_mod._SORTED_ALIAS_INDEX
        search_mod.REMASTER_ALIASES.update(test_aliases)
        search_mod._SORTED_ALIAS_INDEX = search_mod._build_sorted_alias_index(
            search_mod.REMASTER_ALIASES
        )
        yield
        search_mod.REMASTER_ALIASES.clear()
        search_mod.REMASTER_ALIASES.update(orig_aliases)
        search_mod._SORTED_ALIAS_INDEX = orig_index

    def test_grandmother_expands_to_granny(self):
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("grandmother")
        assert any("granny" in q.lower() for q in queries)

    def test_granny_expands_to_grandmother(self):
        """Reverse alias: granny → grandmother."""
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("granny")
        assert any("grandmother" in q.lower() for q in queries)

    def test_grandma_expands_to_granny(self):
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("grandma")
        assert any("granny" in q.lower() for q in queries)

    def test_multi_word_per_word_expansion(self):
        """'Grandmother Hu' should expand to 'granny Hu' via per-word expansion."""
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("Grandmother Hu")
        assert any("granny" in q.lower() for q in queries)

    def test_informal_synonyms_loaded(self):
        from gm_agent.rag.search import REMASTER_ALIASES
        assert "grandmother" in REMASTER_ALIASES
        assert REMASTER_ALIASES["grandmother"] == "granny"


class TestCreatureFamilyAliases:
    """Tests for creature family alias expansion (aliases normally loaded from DB)."""

    _DRAGON_ALIASES = {
        "red dragon": "cinder dragon",
        "blue dragon": "stormcrown dragon",
        "black dragon": "rive dragon",
        "green dragon": "jungle dragon",
        "white dragon": "tundra dragon",
        "gold dragon": "vizier dragon",
        "silver dragon": "sovereign dragon",
        "bronze dragon": "sea dragon",
        "brass dragon": "desert dragon",
        "copper dragon": "mirage dragon",
    }

    @pytest.fixture(autouse=True)
    def _patch_creature_family_aliases(self):
        """Populate CREATURE_FAMILY_ALIASES with test data (normally loaded from DB at runtime)."""
        import gm_agent.rag.search as search_mod
        orig = dict(search_mod.CREATURE_FAMILY_ALIASES)
        search_mod.CREATURE_FAMILY_ALIASES.update(self._DRAGON_ALIASES)
        yield
        search_mod.CREATURE_FAMILY_ALIASES.clear()
        search_mod.CREATURE_FAMILY_ALIASES.update(orig)

    def test_creature_family_aliases_loaded(self):
        from gm_agent.rag.search import CREATURE_FAMILY_ALIASES
        assert len(CREATURE_FAMILY_ALIASES) > 0
        assert "red dragon" in CREATURE_FAMILY_ALIASES
        assert CREATURE_FAMILY_ALIASES["red dragon"] == "cinder dragon"

    def test_red_dragon_expands_to_cinder(self):
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("red dragon")
        assert any("cinder dragon" in q.lower() for q in queries)

    def test_cinder_dragon_expands_to_red(self):
        """Reverse: cinder dragon → red dragon."""
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("cinder dragon")
        assert any("red dragon" in q.lower() for q in queries)

    def test_compound_query_adult_red_dragon(self):
        """'adult red dragon' should expand to 'adult cinder dragon'."""
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("adult red dragon")
        assert any("adult cinder dragon" in q.lower() for q in queries)

    def test_blue_dragon_expands_to_stormcrown(self):
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("blue dragon")
        assert any("stormcrown dragon" in q.lower() for q in queries)

    def test_gold_dragon_expands_to_vizier(self):
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("gold dragon")
        assert any("vizier dragon" in q.lower() for q in queries)

    def test_all_ten_families_present(self):
        from gm_agent.rag.search import CREATURE_FAMILY_ALIASES
        expected = {
            "red dragon", "blue dragon", "black dragon", "green dragon",
            "white dragon", "gold dragon", "silver dragon", "bronze dragon",
            "brass dragon", "copper dragon",
        }
        assert expected.issubset(set(CREATURE_FAMILY_ALIASES.keys()))

    def test_non_dragon_query_unaffected(self):
        """Queries without dragon families shouldn't get dragon expansions."""
        from gm_agent.rag.search import expand_query_aliases
        queries = expand_query_aliases("goblin warrior")
        # Should only have the original query (+ any other alias expansions)
        assert not any("dragon" in q.lower() for q in queries)


class TestChapterScoping:
    """Tests for chapter param on search tools."""

    def test_search_content_has_chapter_param(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content tool should have chapter parameter."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        tool = server.get_tool("search_content")
        param_names = [p.name for p in tool.parameters]
        assert "chapter" in param_names

    def test_search_lore_has_book_param(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_lore tool should have book parameter."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        tool = server.get_tool("search_lore")
        param_names = [p.name for p in tool.parameters]
        assert "book" in param_names
        assert "chapter" in param_names

    def test_search_guidance_has_book_param(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_guidance tool should have book parameter."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        tool = server.get_tool("search_guidance")
        param_names = [p.name for p in tool.parameters]
        assert "book" in param_names
        assert "chapter" in param_names

    def test_search_pages_has_chapter_param(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_pages tool should have chapter parameter."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        tool = server.get_tool("search_pages")
        param_names = [p.name for p in tool.parameters]
        assert "chapter" in param_names

    def test_search_lore_passes_book(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_lore with book= should pass book to PathfinderSearch."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_lore", {
            "query": "body horror",
            "book": "Season of Ghosts",
        })
        assert result.success is True
        # Verify book was passed through
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        assert kwargs.get("book") == "Season of Ghosts"

    def test_search_guidance_passes_book(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_guidance with book= should pass book to PathfinderSearch."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_guidance", {
            "query": "running encounters",
            "book": "GM Core",
        })
        assert result.success is True
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        assert kwargs.get("book") == "GM Core"

    def test_search_content_passes_chapter(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content with chapter= should pass chapter to PathfinderSearch."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {
            "query": "encounters",
            "book": "Player Core",
            "chapter": "Chapter 2",
        })
        assert result.success is True
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        assert kwargs.get("chapter") == "Chapter 2"

    def test_search_pages_passes_chapter(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_pages with chapter= should pass chapter to search_pages."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_pages", {
            "query": "goblin tactics",
            "book": "Player Core",
            "chapter": "Combat",
        })
        assert result.success is True
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        assert kwargs.get("chapter") == "Combat"

    def test_search_lore_passes_chapter(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_lore with chapter= should pass chapter to PathfinderSearch."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_lore", {
            "query": "history",
            "book": "Player Core",
            "chapter": "Introduction",
        })
        assert result.success is True
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        assert kwargs.get("chapter") == "Introduction"

    def test_search_lore_excludes_npc_but_includes_deity(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_lore should NOT include 'npc' but SHOULD include 'deity'."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        server.call_tool("search_lore", {"query": "Willowshore elders"})
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        cats = kwargs.get("category", [])
        assert "npc" not in cats
        assert "deity" in cats

    def test_search_lore_includes_place_types(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_lore should include region, settlement, landmark, historical_event."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        server.call_tool("search_lore", {"query": "Absalom"})
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        cats = kwargs.get("category", [])
        for expected in ["region", "settlement", "landmark", "historical_event"]:
            assert expected in cats, f"Expected '{expected}' in lore categories"

    def test_lookup_creature_filters_irrelevant_families(self, mock_pathfinder_search: MockPathfinderSearch):
        """lookup_creature should not prepend families whose name doesn't match query words."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        # Configure mock: creature search returns NPC, family search returns irrelevant beetle
        mock_pathfinder_search._search_results = None
        original_search = mock_pathfinder_search.search

        call_count = [0]
        def mock_search(query, **kwargs):
            call_count[0] += 1
            doc_type = kwargs.get("doc_type")
            if doc_type == "creature":
                return [{"name": "Stag Lord", "type": "creature", "category": "creature",
                         "source": "AP", "book": "AP", "book_type": "adventure_path",
                         "page": 1, "content": "A bandit leader", "metadata": {}, "score": 10.0}]
            elif doc_type == "creature_family":
                return [{"name": "Beetle", "type": "creature_family", "category": "creature",
                         "source": "Monster Core", "book": "Monster Core", "book_type": "bestiary",
                         "page": 5, "content": "Stag beetle family", "metadata": {}, "score": 8.0}]
            return original_search(query, **kwargs)

        mock_pathfinder_search.search = mock_search
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_creature", {"name": "Stag Lord"})
        assert result.success is True
        # Beetle should NOT appear because "Beetle" doesn't contain "stag" or "lord"
        assert "Beetle" not in result.data[:200]

    def test_lookup_creature_includes_relevant_families(self, mock_pathfinder_search: MockPathfinderSearch):
        """lookup_creature should prepend families whose name matches query words."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        mock_pathfinder_search._search_results = None
        original_search = mock_pathfinder_search.search

        def mock_search(query, **kwargs):
            doc_type = kwargs.get("doc_type")
            if doc_type == "creature":
                return [{"name": "Red Dragon", "type": "creature", "category": "creature",
                         "source": "Monster Core", "book": "Monster Core", "book_type": "bestiary",
                         "page": 1, "content": "A fearsome dragon", "metadata": {}, "score": 10.0}]
            elif doc_type == "creature_family":
                return [{"name": "Dragon", "type": "creature_family", "category": "creature",
                         "source": "Monster Core", "book": "Monster Core", "book_type": "bestiary",
                         "page": 2, "content": "Dragon family overview", "metadata": {}, "score": 8.0}]
            return original_search(query, **kwargs)

        mock_pathfinder_search.search = mock_search
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("lookup_creature", {"name": "dragon"})
        assert result.success is True
        # Dragon family SHOULD appear because "Dragon" contains "dragon"
        assert "Dragon" in result.data

    def test_search_content_adventure_type_expansion(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content with types='adventure' should expand to location/landmark/npc/etc."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {"query": "dragon", "types": "adventure"})
        assert result.success is True
        # Should have called search with expanded types
        assert len(mock_pathfinder_search.calls) >= 1
        _, kwargs = mock_pathfinder_search.calls[0]
        types = kwargs.get("include_types", [])
        assert "location" in types
        assert "landmark" in types
        assert "npc" in types

    def test_search_content_book_type_redirects_to_browse_book(self, mock_pathfinder_search: MockPathfinderSearch):
        """search_content with types='book' should redirect to browse_book, not call search."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        result = server.call_tool("search_content", {"query": "dragon", "types": "book"})
        assert result.success is True
        # Should not have called the search engine — browse_book returns early
        assert len(mock_pathfinder_search.calls) == 0

    def test_tool_count_updated(self, mock_pathfinder_search: MockPathfinderSearch):
        """PF2eRAGServer should still have the expected number of tools."""
        from gm_agent.systems.pf2e.servers.pf2e_rag import PF2eRAGServer
        server = PF2eRAGServer(db_path="/fake/path")
        tools = server.list_tools()
        assert len(tools) == 18  # unchanged — no new tools, just new params


class TestAutoDetection:
    """Tests for auto-detection of book/chapter references in query text."""

    # Test book name index for detect_book_in_query
    BOOK_INDEX = [
        # Sorted by length descending (longest first)
        ("season of ghosts 1 of 4 - the summer that never was", "Season of Ghosts 1 of 4 - The Summer that Never Was"),
        ("lost omens absalom, city of lost omens", "Lost Omens Absalom, City of Lost Omens"),
        ("abomination vaults", "Abomination Vaults"),
        ("season of ghosts", None),  # Series prefix
        ("blood lords", None),  # Series prefix
        ("player core", "Player Core"),
        ("monster core", "Monster Core"),
        ("gm core", "GM Core"),
        ("kingmaker", "Kingmaker"),
    ]

    # --- detect_book_in_query ---

    def test_detect_book_ap_series(self):
        """Detects AP series name and strips it from query."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("Season of Ghosts body horror", self.BOOK_INDEX)
        assert book == "season of ghosts"
        assert cleaned == "body horror"

    def test_detect_book_ap_series_with_book_n(self):
        """Detects AP series + Book N and combines them."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("Season of Ghosts Book 1 NPCs", self.BOOK_INDEX)
        assert book == "season of ghosts book 1"
        assert cleaned == "NPCs"

    def test_detect_book_single_word(self):
        """Detects single-word book name like Kingmaker."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("Kingmaker kingdom building", self.BOOK_INDEX)
        assert book == "Kingmaker"
        assert cleaned == "kingdom building"

    def test_detect_book_multiword(self):
        """Detects multi-word book name."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("Player Core healing spells", self.BOOK_INDEX)
        assert book == "Player Core"
        assert cleaned == "healing spells"

    def test_detect_book_no_match(self):
        """Returns None for queries without book names."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("dragons in caves", self.BOOK_INDEX)
        assert book is None
        assert cleaned == "dragons in caves"

    def test_detect_book_entire_query(self):
        """Returns None when book name IS the entire query."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("Season of Ghosts", self.BOOK_INDEX)
        assert book is None
        assert cleaned == "Season of Ghosts"

    def test_detect_book_case_insensitive(self):
        """Detection is case-insensitive."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("PLAYER CORE healing", self.BOOK_INDEX)
        assert book == "Player Core"
        assert cleaned == "healing"

    def test_detect_book_word_boundary(self):
        """Book name must match on word boundaries."""
        from gm_agent.rag.search import detect_book_in_query
        # "kingmaker" shouldn't match inside "kingmakers"
        book, cleaned = detect_book_in_query("kingmakers guild", self.BOOK_INDEX)
        assert book is None

    def test_detect_book_preserves_remainder_case(self):
        """Remaining query preserves original casing."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("GM Core Encounter Building Tips", self.BOOK_INDEX)
        assert book == "GM Core"
        assert cleaned == "Encounter Building Tips"

    def test_detect_book_longest_match_wins(self):
        """Longer book names match before shorter ones."""
        from gm_agent.rag.search import detect_book_in_query
        # "abomination vaults" should match, not just "vaults"
        book, cleaned = detect_book_in_query("Abomination Vaults traps", self.BOOK_INDEX)
        assert book == "Abomination Vaults"
        assert cleaned == "traps"

    def test_detect_book_in_middle(self):
        """Detects book name in the middle of a query."""
        from gm_agent.rag.search import detect_book_in_query
        book, cleaned = detect_book_in_query("encounters in Kingmaker adventure", self.BOOK_INDEX)
        assert book == "Kingmaker"
        assert cleaned == "encounters in adventure"

    # --- detect_chapter_in_query ---

    def test_detect_chapter_numbered(self):
        """Detects 'Chapter N' and strips it."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("Chapter 2 encounters")
        assert chapter == "Chapter 2"
        assert cleaned == "encounters"

    def test_detect_chapter_ch_abbreviation(self):
        """Detects 'Ch N' abbreviation."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("Ch 3 traps and hazards")
        assert chapter == "Chapter 3"
        assert cleaned == "traps and hazards"

    def test_detect_chapter_ch_dot(self):
        """Detects 'Ch. N' abbreviation."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("Ch. 5 finale")
        assert chapter == "Chapter 5"
        assert cleaned == "finale"

    def test_detect_chapter_strips_book_n(self):
        """Strips 'Book N' references even without chapter."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("Book 1 encounters")
        assert chapter is None
        assert cleaned == "encounters"

    def test_detect_chapter_both_book_and_chapter(self):
        """Strips both 'Book N' and 'Chapter N'."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("Book 1 Chapter 2 encounters")
        assert chapter == "Chapter 2"
        assert cleaned == "encounters"

    def test_detect_chapter_no_match(self):
        """Returns None when no chapter reference found."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("goblin encounters")
        assert chapter is None
        assert cleaned == "goblin encounters"

    def test_detect_chapter_case_insensitive(self):
        """Chapter detection is case-insensitive."""
        from gm_agent.rag.search import detect_chapter_in_query
        chapter, cleaned = detect_chapter_in_query("CHAPTER 4 boss fight")
        assert chapter == "Chapter 4"
        assert cleaned == "boss fight"

    # --- Combined detection ---

    def test_combined_book_and_chapter(self):
        """Book and chapter detection work together."""
        from gm_agent.rag.search import detect_book_in_query, detect_chapter_in_query
        book, cleaned = detect_book_in_query("Kingmaker Chapter 2 encounters", self.BOOK_INDEX)
        assert book == "Kingmaker"
        chapter, final = detect_chapter_in_query(cleaned)
        assert chapter == "Chapter 2"
        assert final == "encounters"

    def test_combined_ap_book_n_chapter(self):
        """AP series + Book N + Chapter N all detected."""
        from gm_agent.rag.search import detect_book_in_query, detect_chapter_in_query
        book, cleaned = detect_book_in_query(
            "Season of Ghosts Book 1 Chapter 2 encounters", self.BOOK_INDEX
        )
        assert book == "season of ghosts book 1"
        chapter, final = detect_chapter_in_query(cleaned)
        assert chapter == "Chapter 2"
        assert final == "encounters"
