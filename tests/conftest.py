"""Root pytest configuration for GM Agent tests."""

from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from gm_agent.mcp.base import MCPServer, ToolDef, ToolParameter, ToolResult
from gm_agent.models.base import LLMBackend, LLMResponse, Message, StreamChunk, ToolCall
from gm_agent.storage.campaign import CampaignStore
from gm_agent.storage.session import SessionStore
from gm_agent.storage.schemas import (
    Campaign,
    Session,
    Turn,
    SceneState,
    PartyMember,
    ToolCallRecord,
)


def pytest_addoption(parser):
    """Add --run-integration option to pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests against real Ollama server",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires --run-integration)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is provided."""
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------


class MockLLMBackend(LLMBackend):
    """Mock LLM backend for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or [
            LLMResponse(text="Mock response", tool_calls=[], finish_reason="stop")
        ]
        self._call_index = 0
        self.calls: list[tuple[list[Message], list[ToolDef] | None]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        thinking: dict | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        if self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            self._call_index += 1
            return response
        return self.responses[-1]

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Iterator[StreamChunk]:
        response = self.chat(messages, tools)
        yield StreamChunk(
            delta=response.text,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    def get_model_name(self) -> str:
        return "mock-model"

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Mock MCP Server
# ---------------------------------------------------------------------------


class MockMCPServer(MCPServer):
    """Mock MCP server for testing."""

    def __init__(self, tool_results: dict[str, ToolResult] | None = None):
        self._tool_results = tool_results or {}
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="lookup_creature",
                description="Look up a creature by name",
                parameters=[
                    ToolParameter(name="name", type="string", description="Creature name")
                ],
            ),
            ToolDef(
                name="lookup_spell",
                description="Look up a spell by name",
                parameters=[
                    ToolParameter(name="name", type="string", description="Spell name")
                ],
            ),
            ToolDef(
                name="lookup_item",
                description="Look up an item by name",
                parameters=[
                    ToolParameter(name="name", type="string", description="Item name")
                ],
            ),
            ToolDef(
                name="search_rules",
                description="Search rules and mechanics",
                parameters=[
                    ToolParameter(name="query", type="string", description="Rules query"),
                ],
            ),
            ToolDef(
                name="search_content",
                description="General search across content",
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
            ),
            ToolDef(
                name="search_lore",
                description="Search for world lore",
                parameters=[
                    ToolParameter(name="query", type="string", description="Lore query"),
                ],
            ),
            ToolDef(
                name="search_guidance",
                description="Search for GM guidance",
                parameters=[
                    ToolParameter(name="query", type="string", description="Guidance query"),
                ],
            ),
        ]

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))

        if name in self._tool_results:
            return self._tool_results[name]

        # Default mock responses
        defaults = {
            "lookup_creature": ToolResult(
                success=True,
                data="**Goblin** (creature) - Core Rulebook\nA small green humanoid",
            ),
            "lookup_spell": ToolResult(
                success=True,
                data="**Fireball** (spell) - Core Rulebook\nA burst of fire",
            ),
            "lookup_item": ToolResult(
                success=True,
                data="**Longsword** (equipment) - Core Rulebook\n1d8 slashing damage",
            ),
            "search_rules": ToolResult(
                success=True,
                data="**Flanking** (rule) - Core Rulebook\nYou are off-guard when flanked",
            ),
            "search_content": ToolResult(
                success=True,
                data="**Mock Result** (spell) - Core Rulebook\nMock content",
            ),
            "search_lore": ToolResult(
                success=True,
                data="**Absalom** (location) - World Guide\nCity at the center of the world",
            ),
            "search_guidance": ToolResult(
                success=True,
                data="**Running Combat** (guidance) - GM Core\nTips for running combat",
            ),
        }
        if name in defaults:
            return defaults[name]

        return ToolResult(success=False, error=f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Mock MCP Client
# ---------------------------------------------------------------------------


class MockMCPClient:
    """Mock MCP client for testing GMAgent.

    Returns tools from all 3 server categories (RAG, campaign state,
    character runner) to match real MCPClient behavior.
    """

    # Campaign state tools (representative subset)
    _CAMPAIGN_STATE_TOOLS = [
        ToolDef(name="update_scene", description="Update scene state", parameters=[
            ToolParameter(name="location", type="string", description="Location", required=False),
        ]),
        ToolDef(name="advance_time", description="Advance time of day", parameters=[
            ToolParameter(name="time_of_day", type="string", description="Time"),
        ]),
        ToolDef(name="log_event", description="Log a session event", parameters=[
            ToolParameter(name="description", type="string", description="Event description"),
        ]),
        ToolDef(name="log_dialogue", description="Log NPC dialogue", parameters=[
            ToolParameter(name="npc_name", type="string", description="NPC name"),
        ]),
        ToolDef(name="search_history", description="Search session history", parameters=[
            ToolParameter(name="query", type="string", description="Query"),
        ]),
        ToolDef(name="get_scene", description="Get current scene state", parameters=[]),
        ToolDef(name="get_session_summary", description="Get session summary", parameters=[]),
        ToolDef(name="update_session_summary", description="Update session summary", parameters=[
            ToolParameter(name="summary", type="string", description="Summary"),
        ]),
        ToolDef(name="get_preferences", description="Get campaign preferences", parameters=[]),
    ]

    # Character runner tools
    _CHARACTER_RUNNER_TOOLS = [
        ToolDef(name="run_npc", description="Run an NPC", parameters=[
            ToolParameter(name="character_id", type="string", description="Character ID"),
        ]),
        ToolDef(name="run_monster", description="Run a monster", parameters=[
            ToolParameter(name="character_id", type="string", description="Character ID"),
        ]),
        ToolDef(name="emulate_player", description="Emulate a player", parameters=[
            ToolParameter(name="character_id", type="string", description="Character ID"),
        ]),
        ToolDef(name="create_character", description="Create a character", parameters=[
            ToolParameter(name="name", type="string", description="Name"),
        ]),
        ToolDef(name="get_character", description="Get character profile", parameters=[
            ToolParameter(name="character_id", type="string", description="Character ID"),
        ]),
        ToolDef(name="list_characters", description="List characters", parameters=[]),
    ]

    def __init__(self, **kwargs):
        self._server = MockMCPServer()
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[ToolDef]:
        """Return tools from all server categories (RAG + campaign state + character runner)."""
        return (
            self._server.list_tools()
            + self._CAMPAIGN_STATE_TOOLS
            + self._CHARACTER_RUNNER_TOOLS
        )

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        return self._server.call_tool(name, args)

    def get_server(self, name: str) -> MCPServer | None:
        return self._server

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Mock PathfinderSearch
# ---------------------------------------------------------------------------


class MockPathfinderSearch:
    """Mock PathfinderSearch for testing PF2eRAGServer without a real database."""

    def __init__(self, db_path: str = "fake.db"):
        self.db_path = db_path
        self.calls: list[tuple[str, dict]] = []

    def search(self, query: str, **kwargs) -> list[dict]:
        self.calls.append((query, kwargs))
        return [
            {
                "name": "Goblin",
                "type": "creature",
                "category": "creature",
                "source": "Core Rulebook",
                "book": "Core Rulebook",
                "book_type": "rulebook",
                "page": 42,
                "content": "A small green humanoid creature",
                "metadata": {},
                "score": 10.0,
            }
        ]

    def search_pages(self, query: str, **kwargs) -> list[dict]:
        self.calls.append((query, kwargs))
        return [
            {
                "book": "Core Rulebook",
                "page_number": 100,
                "chapter": "Combat",
                "snippet": "...matching text...",
                "score": 5.0,
            }
        ]

    def get_stats(self) -> dict:
        return {
            "total_entities": 1000,
            "total_pages": 500,
            "total_embeddings": 2000,
            "by_category": {"creature": 200, "spell": 150},
            "by_book_type": {"rulebook": 600, "bestiary": 400},
            "by_book": {"Core Rulebook": 500},
            "schema_version": 4,
        }

    def get_book_summary(self, book: str) -> dict | None:
        return {
            "book": book,
            "total_pages": 460,
            "chapter_count": 11,
            "summary": "A comprehensive rulebook for Pathfinder 2E players.",
            "chapters": [
                {"name": "Introduction", "page_range": [1, 20]},
                {"name": "Ancestries & Backgrounds", "page_range": [21, 80]},
                {"name": "Spells", "page_range": [280, 420]},
            ],
        }

    def resolve_book_name(self, query: str) -> str | None:
        books = {
            "Player Core": "Player Core",
            "GM Core": "GM Core",
            "Monster Core": "Monster Core",
        }
        # Exact match
        if query in books:
            return books[query]
        # Contains match
        query_lower = query.lower()
        for name in books:
            if query_lower in name.lower():
                return name
        return None

    def list_books_with_summaries(self, book_type: str = None) -> list[dict]:
        books = [
            {
                "book": "Player Core",
                "total_pages": 460,
                "chapter_count": 11,
                "summary": "A comprehensive rulebook for Pathfinder 2E players.",
                "book_type": "rulebook",
            },
            {
                "book": "GM Core",
                "total_pages": 336,
                "chapter_count": 8,
                "summary": "The Game Master's guide to running Pathfinder 2E.",
                "book_type": "rulebook",
            },
            {
                "book": "Monster Core",
                "total_pages": 360,
                "chapter_count": 12,
                "summary": "Creatures for Pathfinder 2E.",
                "book_type": "bestiary",
            },
        ]
        if book_type:
            books = [b for b in books if b.get("book_type") == book_type]
        return books

    def get_chapter_summary(self, book: str, chapter: str) -> dict | None:
        return {
            "chapter": "Spells",
            "page_start": 280,
            "page_end": 420,
            "page_count": 141,
            "summary": "Contains all spells available in the Player Core.",
            "keywords": ["spell", "cantrip", "focus spell", "tradition"],
            "entities": ["Fireball", "Heal", "Magic Missile"],
        }

    def list_chapters(self, book: str) -> list[dict]:
        return [
            {"chapter": "Introduction", "page_start": 1, "page_end": 20, "page_count": 20},
            {"chapter": "Ancestries & Backgrounds", "page_start": 21, "page_end": 80, "page_count": 60},
            {"chapter": "Spells", "page_start": 280, "page_end": 420, "page_count": 141},
        ]

    def get_page_summary(self, book: str, page_number: int) -> dict | None:
        return {
            "page_number": page_number,
            "chapter": "Spells",
            "page_type": "spell_page",
            "summary": "Contains the spells Fireball, Fire Shield, and Fire Ray.",
            "keywords": ["fire", "evocation"],
            "rules_referenced": ["area of effect", "saving throw"],
            "entities_on_page": ["Fireball", "Fire Shield", "Fire Ray"],
            "gm_notes": [],
        }

    def get_page_summaries_for_chapter(self, book: str, chapter: str) -> list[dict]:
        return [
            {
                "page_number": 280,
                "chapter": "Spells",
                "page_type": "spell_page",
                "summary": "Introduction to spellcasting in PF2E.",
                "entities_on_page": ["Spellcasting", "Traditions"],
            },
            {
                "page_number": 281,
                "chapter": "Spells",
                "page_type": "spell_page",
                "summary": "Spell lists for arcane tradition.",
                "entities_on_page": ["Arcane Spell List"],
            },
        ]

    def find_page_for_term(self, term: str, book: str = None) -> list[dict]:
        return [
            {
                "name": "Fireball",
                "type": "spell",
                "book": "Player Core",
                "page_number": 326,
                "source": "entity",
            }
        ]

    def list_entities(
        self,
        book: str | None = None,
        book_type: str | None = None,
        category: str | list[str] | None = None,
        include_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        return [
            {
                "name": "Goblin",
                "type": "creature",
                "category": "creature",
                "source": "Core Rulebook",
                "book": "Core Rulebook",
                "book_type": "rulebook",
                "page": 42,
                "content": "A small green humanoid creature",
                "metadata": {},
            }
        ]

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_campaigns_dir(tmp_path: Path) -> Path:
    """Create temporary campaigns directory."""
    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()
    return campaigns_dir


@pytest.fixture
def campaign_store(tmp_campaigns_dir: Path) -> CampaignStore:
    """Create CampaignStore with temporary directory."""
    return CampaignStore(tmp_campaigns_dir)


@pytest.fixture
def session_store(tmp_campaigns_dir: Path) -> SessionStore:
    """Create SessionStore with temporary directory."""
    return SessionStore(tmp_campaigns_dir)


@pytest.fixture
def mock_llm_backend() -> MockLLMBackend:
    """Create a mock LLM backend."""
    return MockLLMBackend()


@pytest.fixture
def mock_mcp_server() -> MockMCPServer:
    """Create a mock MCP server."""
    return MockMCPServer()


@pytest.fixture
def mock_mcp_with_errors() -> MockMCPServer:
    """Create a mock MCP server that returns errors."""
    return MockMCPServer(
        tool_results={
            "lookup_creature": ToolResult(
                success=False, error="Database connection failed"
            ),
        }
    )


@pytest.fixture
def mock_pathfinder_search():
    """Patch PathfinderSearch constructor to return MockPathfinderSearch."""
    mock = MockPathfinderSearch()
    with patch("gm_agent.rag.search.PathfinderSearch.__init__", lambda self, **kwargs: None):
        with patch.object(
            MockPathfinderSearch, "__init__", lambda self, **kwargs: None
        ):
            # Patch the PathfinderSearch class in the pf2e_rag module
            with patch(
                "gm_agent.mcp.pf2e_rag.PathfinderSearch",
                return_value=mock,
            ):
                yield mock


# ---------------------------------------------------------------------------
# Schema Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scene_state() -> SceneState:
    """Create a sample SceneState for testing."""
    return SceneState(
        location="Sandpoint Market Square",
        npcs_present=["Sheriff Hemlock", "Ameiko Kaijitsu"],
        time_of_day="morning",
        conditions=["crowded", "sunny"],
        notes="Festival grounds are being set up.",
    )


@pytest.fixture
def sample_party_member() -> PartyMember:
    """Create a sample PartyMember for testing."""
    return PartyMember(
        name="Valeros",
        ancestry="Human",
        class_name="Fighter",
        level=5,
        player_name="Alice",
        notes="Equipped with a +1 longsword and heavy armor.",
    )


@pytest.fixture
def sample_party() -> list[PartyMember]:
    """Create a sample party of 3 members for testing."""
    return [
        PartyMember(
            name="Valeros",
            ancestry="Human",
            class_name="Fighter",
            level=5,
            player_name="Alice",
            notes="Tank and melee damage dealer.",
        ),
        PartyMember(
            name="Seoni",
            ancestry="Half-Elf",
            class_name="Sorcerer",
            level=5,
            player_name="Bob",
            notes="Arcane blaster.",
        ),
        PartyMember(
            name="Kyra",
            ancestry="Half-Orc",
            class_name="Cleric",
            level=5,
            player_name="Charlie",
            notes="Healer and support.",
        ),
    ]


@pytest.fixture
def sample_tool_call_record() -> ToolCallRecord:
    """Create a sample ToolCallRecord for testing."""
    return ToolCallRecord(
        name="lookup_creature",
        args={"name": "goblin"},
        result="**Goblin** (creature) - Core Rulebook\nA small green humanoid.",
    )


@pytest.fixture
def sample_turn(sample_tool_call_record: ToolCallRecord) -> Turn:
    """Create a sample Turn with tool calls for testing."""
    return Turn(
        player_input="What is a goblin?",
        gm_response="A goblin is a small, malicious humanoid creature.",
        tool_calls=[sample_tool_call_record],
    )


@pytest.fixture
def sample_campaign(sample_party: list[PartyMember]) -> Campaign:
    """Create a full sample Campaign for testing."""
    return Campaign(
        id="rise-of-the-runelords",
        name="Rise of the Runelords",
        background="An ancient evil stirs beneath the town of Sandpoint.",
        current_arc="Chapter 1: Burnt Offerings",
        party=sample_party,
        preferences={"combat_style": "theater_of_mind"},
    )


@pytest.fixture
def sample_session(sample_scene_state: SceneState) -> Session:
    """Create a sample Session with one turn for testing."""
    return Session(
        id="abc12345",
        campaign_id="rise-of-the-runelords",
        turns=[
            Turn(
                player_input="What is a goblin?",
                gm_response="A goblin is a small, malicious creature.",
            ),
        ],
        scene_state=sample_scene_state,
    )


@pytest.fixture
def sample_session_with_many_turns() -> Session:
    """Create a Session with 20 turns for pagination testing."""
    turns = [
        Turn(
            player_input=f"Player action {i}",
            gm_response=f"GM response {i}",
        )
        for i in range(1, 21)
    ]
    return Session(
        id="many-turns-session",
        campaign_id="rise-of-the-runelords",
        turns=turns,
    )


@pytest.fixture
def minimal_campaign() -> Campaign:
    """Create a minimal Campaign with only required fields."""
    return Campaign(
        id="minimal-test",
        name="Minimal Test Campaign",
    )


@pytest.fixture
def minimal_session() -> Session:
    """Create a minimal Session with only required fields."""
    return Session(
        id="minimal-session",
        campaign_id="minimal-test",
    )


@pytest.fixture
def persisted_campaign(campaign_store: CampaignStore) -> Campaign:
    """Create a campaign that exists on disk via CampaignStore."""
    return campaign_store.create(name="Persisted Campaign")


@pytest.fixture
def mock_llm_with_tool_call() -> MockLLMBackend:
    """Create a MockLLMBackend that returns a tool call then a text response."""
    return MockLLMBackend(
        responses=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_creature",
                        args={"name": "goblin"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                text="A goblin is a small, malicious humanoid creature.",
                tool_calls=[],
                finish_reason="stop",
            ),
        ]
    )


@pytest.fixture
def mock_llm_multiple_tool_calls() -> MockLLMBackend:
    """Create a MockLLMBackend that returns two tool calls then a text response."""
    return MockLLMBackend(
        responses=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_creature",
                        args={"name": "goblin"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="lookup_spell",
                        args={"name": "fireball"},
                    ),
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                text="Goblins are weak against fireball.",
                tool_calls=[],
                finish_reason="stop",
            ),
        ]
    )
