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

        assert len(tools) == 16
        tool_names = [t.name for t in tools]
        assert "lookup_creature" in tool_names
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
