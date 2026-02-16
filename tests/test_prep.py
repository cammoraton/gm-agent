"""Tests for campaign prep system."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from gm_agent.models.base import LLMResponse, Message
from gm_agent.prep.knowledge import (
    PARTY_KNOWLEDGE_ID,
    SUBSYSTEM_CONTENT_TYPES,
    SUBSYSTEM_KEYWORDS,
    _format_entities_text,
    _names_match,
    _normalize_for_compare,
    _parse_json_array,
    _slugify_name,
    deduplicate_npc_names,
    detect_subsystems,
    generate_background,
    resolve_ap_books,
    seed_npc_knowledge,
    seed_party_knowledge,
    seed_subsystem_knowledge,
    seed_world_context,
    seed_world_context_by_query,
)
from gm_agent.prep.log import PrepLogEntry, PrepLogger
from gm_agent.prep.pipeline import PrepPipeline, PrepResult
from gm_agent.storage.knowledge import KnowledgeStore

# Import test fixtures/mocks — these are in conftest.py which pytest
# auto-loads, but we need them as classes for direct use
from tests.conftest import MockLLMBackend, MockPathfinderSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_search(entities=None, npcs=None, book_summary=None):
    """Create a MockPathfinderSearch with customizable list_entities."""
    mock = MockPathfinderSearch()

    if entities is not None:
        mock.list_entities = lambda **kwargs: entities

    if npcs is not None:
        _orig = mock.list_entities if entities is not None else lambda **kwargs: []

        def _list_entities(**kwargs):
            cat = kwargs.get("category")
            if cat == "npc" or (isinstance(cat, list) and "npc" in cat):
                return npcs
            if entities is not None:
                return entities
            return _orig(**kwargs)

        mock.list_entities = _list_entities

    if book_summary is not None:
        mock.get_book_summary = lambda book: book_summary

    return mock


def _make_llm_with_json(json_data: list[dict], thinking: str | None = None) -> MockLLMBackend:
    """Create a MockLLMBackend that returns a JSON array."""
    text = json.dumps(json_data)
    return MockLLMBackend(
        responses=[LLMResponse(text=text, thinking=thinking, usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})]
    )


# ---------------------------------------------------------------------------
# PrepLogger tests
# ---------------------------------------------------------------------------


class TestPrepLogger:
    def test_write_read_cycle(self, tmp_path):
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        entry = PrepLogEntry(
            step="party_knowledge",
            campaign_id="test-campaign",
            book="Player Core",
            entity=None,
            input_context="Some entity text",
            system_prompt="System prompt here",
            thinking="I think this is important because...",
            output=[{"content": "A fact", "importance": 7}],
            model="mock-model",
            duration_ms=1234.5,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        logger.log(entry)
        logger.log(entry)

        entries = logger.read()
        assert len(entries) == 2
        assert entries[0].step == "party_knowledge"
        assert entries[0].book == "Player Core"
        assert entries[0].thinking == "I think this is important because..."
        assert entries[0].duration_ms == 1234.5

    def test_read_empty(self, tmp_path):
        logger = PrepLogger("nonexistent", base_dir=tmp_path)
        assert logger.read() == []

    def test_log_creates_directory(self, tmp_path):
        logger = PrepLogger("new-campaign", base_dir=tmp_path)
        entry = PrepLogEntry(
            step="test",
            campaign_id="new-campaign",
            book="Test",
            entity=None,
            input_context="",
            system_prompt="",
            thinking=None,
            output=[],
            model="test",
            duration_ms=0,
        )
        logger.log(entry)
        assert logger.log_path.exists()

    def test_entry_json_roundtrip(self):
        entry = PrepLogEntry(
            step="npc_knowledge",
            campaign_id="test",
            book="AP Vol 1",
            entity="Boss NPC",
            input_context="NPC text",
            system_prompt="prompt",
            thinking=None,
            output=[{"content": "fact"}],
            model="gpt-4",
            duration_ms=500.0,
            token_usage={"prompt_tokens": 50},
        )
        json_str = entry.to_json()
        restored = PrepLogEntry.from_json(json_str)
        assert restored.step == entry.step
        assert restored.entity == "Boss NPC"
        assert restored.thinking is None
        assert restored.token_usage == {"prompt_tokens": 50}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_format_entities_text(self):
        entities = [
            {"name": "Goblin", "type": "creature", "page": 42, "content": "A small creature"},
            {"name": "Otari", "type": "settlement", "page": 10, "content": "A small town"},
        ]
        text = _format_entities_text(entities)
        assert "[creature] Goblin (p.42)" in text
        assert "[settlement] Otari (p.10)" in text
        assert "A small creature" in text
        assert "---" in text

    def test_format_entities_no_page(self):
        entities = [{"name": "Test", "type": "rule", "page": None, "content": "Rule text"}]
        text = _format_entities_text(entities)
        assert "[rule] Test" in text
        assert "(p." not in text

    def test_parse_json_array_plain(self):
        result = _parse_json_array('[{"content": "hello"}]')
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_parse_json_array_with_fences(self):
        text = '```json\n[{"content": "fenced"}]\n```'
        result = _parse_json_array(text)
        assert len(result) == 1
        assert result[0]["content"] == "fenced"

    def test_parse_json_array_embedded(self):
        text = 'Here is the result: [{"content": "embedded"}] and more text'
        result = _parse_json_array(text)
        assert len(result) == 1

    def test_parse_json_array_invalid(self):
        result = _parse_json_array("not json at all")
        assert result == []

    def test_parse_json_array_object(self):
        result = _parse_json_array('{"key": "value"}')
        assert result == []

    def test_slugify_name(self):
        assert _slugify_name("Merisiel") == "merisiel"
        assert _slugify_name("Lord Gyr of Absalom") == "lord-gyr-of-absalom"
        assert _slugify_name("  Test NPC  ") == "test-npc"


# ---------------------------------------------------------------------------
# seed_party_knowledge tests
# ---------------------------------------------------------------------------


class TestSeedPartyKnowledge:
    def test_basic_seeding(self, tmp_path):
        entities = [
            {
                "name": "Otari",
                "type": "settlement",
                "category": "location",
                "source": "Test PG",
                "book": "Test PG",
                "book_type": "players_guide",
                "page": 5,
                "content": "A small fishing town on the coast of Absalom.",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "Otari is a small fishing town.", "importance": 7, "tags": ["location"]},
            {"content": "The town is known for its lumber trade.", "importance": 5, "tags": ["economy"]},
        ]

        mock_search = _make_mock_search(entities=entities)
        llm = _make_llm_with_json(json_output, thinking="Analyzing the settlement...")
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_party_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "Test PG", "book_type": "players_guide"}],
            campaign_id="test-campaign",
        )

        assert count == 2

        # Verify knowledge entries
        entries = knowledge.query_knowledge(character_id=PARTY_KNOWLEDGE_ID)
        assert len(entries) == 2
        assert entries[0].character_name == "Party"
        assert entries[0].sharing_condition == "free"
        assert entries[0].decay_rate == 0.0
        assert "players_guide" in entries[0].tags

        # Verify log
        log_entries = prep_logger.read()
        assert len(log_entries) == 1
        assert log_entries[0].step == "party_knowledge"
        assert log_entries[0].thinking == "Analyzing the settlement..."

        knowledge.close()

    def test_no_entities(self, tmp_path):
        mock_search = _make_mock_search(entities=[])
        llm = _make_llm_with_json([])
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_party_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "Empty Book", "book_type": "players_guide"}],
            campaign_id="test-campaign",
        )

        assert count == 0
        assert llm._call_index == 0  # No LLM calls made
        knowledge.close()

    def test_empty_content_filtered(self, tmp_path):
        entities = [
            {
                "name": "Test",
                "type": "guidance",
                "category": "guidance",
                "source": "PG",
                "book": "PG",
                "book_type": "players_guide",
                "page": 1,
                "content": "Test content",
                "metadata": {},
            },
        ]

        # LLM returns entries with some empty content
        json_output = [
            {"content": "", "importance": 5, "tags": []},
            {"content": "Valid fact", "importance": 6, "tags": ["valid"]},
        ]

        mock_search = _make_mock_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_party_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "PG", "book_type": "players_guide"}],
            campaign_id="test-campaign",
        )

        assert count == 1  # Empty content entry filtered
        knowledge.close()


# ---------------------------------------------------------------------------
# seed_npc_knowledge tests
# ---------------------------------------------------------------------------


class TestSeedNPCKnowledge:
    def test_basic_npc_seeding(self, tmp_path):
        npcs = [
            {
                "name": "Tamily Tanderveil",
                "type": "npc",
                "category": "npc",
                "source": "AP Vol 1",
                "book": "AP Vol 1",
                "book_type": "adventure",
                "page": 15,
                "content": "Tamily is the owner of the Otari Market.",
                "metadata": {},
            },
        ]

        json_output = [
            {
                "content": "Tamily runs the general store in Otari.",
                "knowledge_type": "fact",
                "sharing_condition": "free",
                "importance": 6,
                "tags": ["merchant"],
            },
            {
                "content": "Tamily secretly worries about monsters in the basement.",
                "knowledge_type": "secret",
                "sharing_condition": "trust",
                "importance": 8,
                "tags": ["quest_hook"],
            },
        ]

        mock_search = _make_mock_search(npcs=npcs)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_npc_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "AP Vol 1", "book_type": "adventure"}],
            campaign_id="test-campaign",
        )

        assert count == 2

        # Verify entries have correct character_id
        entries = knowledge.query_knowledge(character_id="tamily-tanderveil")
        assert len(entries) == 2

        # Check knowledge types are preserved
        types = {e.knowledge_type for e in entries}
        assert "fact" in types
        assert "secret" in types

        # Check sharing conditions
        conditions = {e.sharing_condition for e in entries}
        assert "free" in conditions
        assert "trust" in conditions

        knowledge.close()

    def test_no_npcs(self, tmp_path):
        mock_search = _make_mock_search(npcs=[])
        llm = _make_llm_with_json([])
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_npc_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "AP Vol 1", "book_type": "adventure"}],
            campaign_id="test-campaign",
        )

        assert count == 0
        knowledge.close()


# ---------------------------------------------------------------------------
# seed_world_context tests
# ---------------------------------------------------------------------------


class TestSeedWorldContext:
    def test_basic_world_seeding(self, tmp_path):
        entities = [
            {
                "name": "Absalom",
                "type": "settlement",
                "category": "location",
                "source": "World Guide",
                "book": "World Guide",
                "book_type": "setting",
                "page": 10,
                "content": "The great city at the center of the world.",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "Absalom is the largest city in the Inner Sea region.", "importance": 7, "tags": ["geography"]},
        ]

        mock_search = _make_mock_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_world_context(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "World Guide", "book_type": "setting"}],
            campaign_id="test-campaign",
        )

        assert count == 1

        entries = knowledge.query_knowledge(character_id=PARTY_KNOWLEDGE_ID, tags=["setting"])
        assert len(entries) == 1
        assert entries[0].character_name == "World"
        assert "setting" in entries[0].tags

        knowledge.close()


# ---------------------------------------------------------------------------
# list_entities tests (in-memory SQLite)
# ---------------------------------------------------------------------------


class TestListEntities:
    @pytest.fixture
    def search_db(self, tmp_path):
        """Create an in-memory-like SQLite DB for testing list_entities."""
        db_path = str(tmp_path / "test_search.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE content (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                category TEXT,
                book TEXT,
                book_type TEXT,
                page INTEGER,
                content TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE content_fts USING fts5(
                name, type, category, content,
                content='content', content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TABLE pages (
                id INTEGER PRIMARY KEY,
                book TEXT,
                page_number INTEGER,
                chapter TEXT,
                content TEXT
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE pages_fts USING fts5(
                book, chapter, content,
                content='pages', content_rowid='id'
            )
        """)
        conn.execute("""
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY,
                source_type TEXT,
                source_id TEXT,
                book TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                chunk_text TEXT,
                embedding BLOB,
                model_name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE book_summaries (
                book TEXT PRIMARY KEY,
                book_type TEXT,
                total_pages INTEGER,
                chapter_count INTEGER,
                summary TEXT,
                chapters TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE schema_meta (version INTEGER)
        """)
        conn.execute("INSERT INTO schema_meta VALUES (5)")

        # Insert test data
        test_entities = [
            ("goblin-1", "Goblin Warrior", "creature", "creature", "Monster Core", "bestiary", 42, "A goblin warrior", "{}"),
            ("goblin-2", "Goblin Pyro", "creature", "creature", "Monster Core", "bestiary", 43, "A goblin pyromancer", "{}"),
            ("fireball-1", "Fireball", "spell", "spell", "Player Core", "rulebook", 300, "A burst of fire", "{}"),
            ("otari-1", "Otari", "settlement", "location", "AV PG", "players_guide", 5, "Small fishing town", "{}"),
            ("boss-npc", "Boss Villain", "npc", "npc", "AV Vol 1", "adventure", 60, "The big bad", "{}"),
        ]

        conn.executemany(
            "INSERT INTO content VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            test_entities,
        )
        conn.commit()
        conn.close()
        return db_path

    def test_list_all(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities()
        assert len(results) == 5
        search.close()

    def test_filter_by_book(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(book="Monster Core")
        assert len(results) == 2
        assert all(r["book"] == "Monster Core" for r in results)
        search.close()

    def test_filter_by_book_type(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(book_type="bestiary")
        assert len(results) == 2
        search.close()

    def test_filter_by_category(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(category="npc")
        assert len(results) == 1
        assert results[0]["name"] == "Boss Villain"
        search.close()

    def test_filter_by_category_list(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(category=["creature", "spell"])
        assert len(results) == 3
        search.close()

    def test_filter_by_include_types(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(include_types=["settlement"])
        assert len(results) == 1
        assert results[0]["name"] == "Otari"
        search.close()

    def test_filter_by_exclude_types(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(exclude_types=["creature", "npc"])
        assert len(results) == 2  # fireball + otari
        search.close()

    def test_limit(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(limit=2)
        assert len(results) == 2
        search.close()

    def test_combined_filters(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(book_type="bestiary", category="creature")
        assert len(results) == 2
        search.close()

    def test_no_results(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(book="Nonexistent Book")
        assert len(results) == 0
        search.close()

    def test_result_structure(self, search_db):
        from gm_agent.rag.search import PathfinderSearch

        search = PathfinderSearch(db_path=search_db)
        results = search.list_entities(category="spell")
        assert len(results) == 1
        r = results[0]
        assert r["name"] == "Fireball"
        assert r["type"] == "spell"
        assert r["category"] == "spell"
        assert r["book"] == "Player Core"
        assert r["source"] == "Player Core"  # backward compat alias
        assert r["book_type"] == "rulebook"
        assert r["page"] == 300
        assert "content" in r
        assert "metadata" in r
        assert "score" not in r  # list_entities doesn't return scores
        search.close()


# ---------------------------------------------------------------------------
# PrepPipeline tests
# ---------------------------------------------------------------------------


class TestPrepPipeline:
    def test_resolve_books(self, tmp_path):
        mock_search = MockPathfinderSearch()

        # Make get_book_summary return type info
        def get_book_summary(book):
            types = {
                "Player Core": "rulebook",
                "GM Core": "rulebook",
                "Monster Core": "bestiary",
            }
            return {"book_type": types.get(book, "unknown")}

        mock_search.get_book_summary = get_book_summary

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=MockLLMBackend(),
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        resolved = pipeline.resolve_books(["Player Core", "GM Core"])
        assert len(resolved) == 2
        names = {b["name"] for b in resolved}
        assert "Player Core" in names
        assert "GM Core" in names

    def test_resolve_books_dedup(self, tmp_path):
        mock_search = MockPathfinderSearch()
        mock_search.get_book_summary = lambda book: {"book_type": "rulebook"}

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=MockLLMBackend(),
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        resolved = pipeline.resolve_books(["Player Core", "Player Core"])
        assert len(resolved) == 1

    def test_resolve_books_unknown(self, tmp_path):
        mock_search = MockPathfinderSearch()
        pipeline = PrepPipeline(
            campaign_id="test",
            llm=MockLLMBackend(),
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        resolved = pipeline.resolve_books(["Totally Fake Book"])
        assert len(resolved) == 0

    def test_end_to_end(self, tmp_path):
        """Test full pipeline with all mocked dependencies."""
        # Set up mock search that returns entities by category
        mock_search = MockPathfinderSearch()

        pg_entities = [
            {
                "name": "Otari",
                "type": "settlement",
                "category": "location",
                "source": "AV PG",
                "book": "AV PG",
                "book_type": "players_guide",
                "page": 5,
                "content": "Small fishing town",
                "metadata": {},
            },
        ]

        npc_entities = [
            {
                "name": "Tamily",
                "type": "npc",
                "category": "npc",
                "source": "AV Vol 1",
                "book": "AV Vol 1",
                "book_type": "adventure",
                "page": 15,
                "content": "Shopkeeper",
                "metadata": {},
            },
        ]

        setting_entities = [
            {
                "name": "Absalom",
                "type": "settlement",
                "category": "location",
                "source": "World Guide",
                "book": "World Guide",
                "book_type": "setting",
                "page": 10,
                "content": "Great city",
                "metadata": {},
            },
        ]

        def list_entities(**kwargs):
            book = kwargs.get("book")
            cat = kwargs.get("category")
            if book == "AV PG":
                return pg_entities
            if book == "AV Vol 1":
                if cat == "npc" or (isinstance(cat, list) and "npc" in cat):
                    return npc_entities
                return []
            if book == "World Guide":
                return setting_entities
            return []

        mock_search.list_entities = list_entities

        book_types = {
            "AV PG": "players_guide",
            "AV Vol 1": "adventure",
            "World Guide": "setting",
        }

        def resolve_book_name(name):
            return name if name in book_types else None

        def get_book_summary(book):
            bt = book_types.get(book)
            if bt:
                return {"book_type": bt}
            return None

        mock_search.resolve_book_name = resolve_book_name
        mock_search.get_book_summary = get_book_summary

        # LLM returns one entry per call — use distinct content to avoid dedup
        party_json = json.dumps([{"content": "A party fact", "importance": 5, "tags": ["test"]}])
        npc_json_response = json.dumps([
            {"content": "NPC fact", "knowledge_type": "fact", "sharing_condition": "free", "importance": 6, "tags": ["npc"]}
        ])
        world_json = json.dumps([{"content": "A world fact", "importance": 4, "tags": ["setting"]}])

        # We need multiple responses: party (1 batch), npc (1 npc), world (1 batch)
        llm = MockLLMBackend(responses=[
            LLMResponse(text=party_json, usage={}),
            LLMResponse(text=npc_json_response, usage={}),
            LLMResponse(text=world_json, usage={}),
        ])

        pipeline = PrepPipeline(
            campaign_id="test-campaign",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["AV PG", "AV Vol 1", "World Guide"], skip_steps=["subsystem"])

        assert result.campaign_id == "test-campaign"
        assert len(result.books_resolved) == 3
        assert result.party_knowledge_count >= 1
        assert result.npc_knowledge_count >= 1
        assert result.world_context_count >= 1
        assert result.total_count >= 3
        assert result.duration_ms > 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# PrepResult tests
# ---------------------------------------------------------------------------


class TestPrepResult:
    def test_total_count(self):
        result = PrepResult(
            campaign_id="test",
            party_knowledge_count=10,
            npc_knowledge_count=25,
            world_context_count=15,
        )
        assert result.total_count == 50

    def test_defaults(self):
        result = PrepResult(campaign_id="test")
        assert result.total_count == 0
        assert result.errors == []
        assert result.books_resolved == []

    def test_timing_fields(self):
        result = PrepResult(
            campaign_id="test",
            party_duration_ms=1000.0,
            npc_duration_ms=2000.0,
            world_duration_ms=500.0,
        )
        assert result.party_duration_ms == 1000.0
        assert result.npc_duration_ms == 2000.0
        assert result.world_duration_ms == 500.0

    def test_token_fields(self):
        result = PrepResult(
            campaign_id="test",
            total_prompt_tokens=1000,
            total_completion_tokens=500,
        )
        assert result.total_tokens == 1500

    def test_token_defaults_zero(self):
        result = PrepResult(campaign_id="test")
        assert result.total_prompt_tokens == 0
        assert result.total_completion_tokens == 0
        assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# LLMResponse thinking field tests
# ---------------------------------------------------------------------------


class TestLLMResponseThinking:
    def test_thinking_field_default_none(self):
        response = LLMResponse(text="hello")
        assert response.thinking is None

    def test_thinking_field_set(self):
        response = LLMResponse(text="hello", thinking="I reasoned about this")
        assert response.thinking == "I reasoned about this"

    def test_thinking_field_serialization(self):
        response = LLMResponse(text="hello", thinking="reasoning")
        d = response.model_dump()
        assert d["thinking"] == "reasoning"

        restored = LLMResponse.model_validate(d)
        assert restored.thinking == "reasoning"


# ---------------------------------------------------------------------------
# Progress callback tests
# ---------------------------------------------------------------------------


class TestProgressCallbacks:
    def test_party_progress_callback(self, tmp_path):
        entities = [
            {
                "name": f"Entity {i}",
                "type": "settlement",
                "category": "location",
                "source": "PG",
                "book": "PG",
                "book_type": "players_guide",
                "page": i,
                "content": f"Content for entity {i}",
                "metadata": {},
            }
            for i in range(3)
        ]

        json_output = [
            {"content": "A fact", "importance": 5, "tags": ["test"]},
        ]

        mock_search = _make_mock_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        messages = []
        seed_party_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "PG", "book_type": "players_guide"}],
            campaign_id="test-campaign",
            on_progress=messages.append,
        )

        assert len(messages) >= 1
        # Should have batch progress + summary
        assert any("batch" in m for m in messages)
        assert any("→" in m for m in messages)
        knowledge.close()

    def test_npc_progress_callback(self, tmp_path):
        npcs = [
            {
                "name": "Test NPC",
                "type": "npc",
                "category": "npc",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 10,
                "content": "An NPC",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "NPC knows stuff", "knowledge_type": "fact",
             "sharing_condition": "free", "importance": 5, "tags": []},
        ]

        mock_search = _make_mock_search(npcs=npcs)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        messages = []
        seed_npc_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "AP", "book_type": "adventure"}],
            campaign_id="test-campaign",
            on_progress=messages.append,
        )

        assert len(messages) >= 1
        assert any("NPC 1/1" in m for m in messages)
        assert any("Test NPC" in m for m in messages)
        knowledge.close()

    def test_world_context_progress_callback(self, tmp_path):
        entities = [
            {
                "name": "Absalom",
                "type": "settlement",
                "category": "location",
                "source": "WG",
                "book": "WG",
                "book_type": "setting",
                "page": 10,
                "content": "A great city",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "Absalom is big", "importance": 6, "tags": ["geography"]},
        ]

        mock_search = _make_mock_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        messages = []
        seed_world_context(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "WG", "book_type": "setting"}],
            campaign_id="test-campaign",
            on_progress=messages.append,
        )

        assert len(messages) >= 1
        assert any("batch" in m for m in messages)
        knowledge.close()

    def test_no_progress_callback_doesnt_crash(self, tmp_path):
        """Ensure on_progress=None (default) works fine."""
        entities = [
            {
                "name": "Test",
                "type": "settlement",
                "category": "location",
                "source": "PG",
                "book": "PG",
                "book_type": "players_guide",
                "page": 1,
                "content": "Content",
                "metadata": {},
            },
        ]

        mock_search = _make_mock_search(entities=entities)
        llm = _make_llm_with_json([{"content": "Fact", "importance": 5, "tags": []}])
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        # No on_progress param — should work without error
        count = seed_party_knowledge(
            mock_search, knowledge, llm, prep_logger,
            books=[{"name": "PG", "book_type": "players_guide"}],
            campaign_id="test-campaign",
        )
        assert count == 1
        knowledge.close()


# ---------------------------------------------------------------------------
# seed_world_context_by_query tests
# ---------------------------------------------------------------------------


class TestSeedWorldContextByQuery:
    def test_basic_query_seeding(self, tmp_path):
        """Test query-based world context with search results."""
        mock_search = MockPathfinderSearch()

        # Override search to return specific results for each term
        search_results = {
            "Stolen Lands": [
                {
                    "name": "Stolen Lands",
                    "type": "region",
                    "category": "location",
                    "source": "World Guide",
                    "book": "World Guide",
                    "book_type": "setting",
                    "page": 30,
                    "content": "A lawless frontier south of Brevoy.",
                    "metadata": {},
                    "score": 10.0,
                },
            ],
            "Brevoy": [
                {
                    "name": "Brevoy",
                    "type": "region",
                    "category": "location",
                    "source": "World Guide",
                    "book": "World Guide",
                    "book_type": "setting",
                    "page": 25,
                    "content": "A northern nation on the brink of civil war.",
                    "metadata": {},
                    "score": 8.0,
                },
            ],
        }

        def mock_search_fn(query, **kwargs):
            return search_results.get(query, [])

        mock_search.search = mock_search_fn

        json_output = [
            {"content": "The Stolen Lands are a lawless frontier.", "importance": 8, "tags": ["region"]},
            {"content": "Brevoy teeters on the brink of civil war.", "importance": 7, "tags": ["politics"]},
        ]

        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_world_context_by_query(
            mock_search, knowledge, llm, prep_logger,
            search_terms=["Stolen Lands", "Brevoy"],
            campaign_id="test-campaign",
            campaign_background="A campaign in the Stolen Lands.",
        )

        assert count == 2

        entries = knowledge.query_knowledge(character_id=PARTY_KNOWLEDGE_ID)
        assert len(entries) == 2
        assert all(e.character_name == "World" for e in entries)
        assert all("world_context" in e.tags for e in entries)
        assert all(e.source == "prep:search_queries" for e in entries)

        # Verify log entries
        log_entries = prep_logger.read()
        assert len(log_entries) == 1
        assert log_entries[0].step == "world_context_query"

        knowledge.close()

    def test_deduplicates_search_results(self, tmp_path):
        """Test that duplicate entity names across search terms are deduped."""
        mock_search = MockPathfinderSearch()

        # Both terms return the same entity
        same_entity = {
            "name": "Brevoy",
            "type": "region",
            "category": "location",
            "source": "World Guide",
            "book": "World Guide",
            "book_type": "setting",
            "page": 25,
            "content": "A northern nation.",
            "metadata": {},
            "score": 8.0,
        }

        mock_search.search = lambda query, **kwargs: [same_entity]

        json_output = [
            {"content": "Brevoy fact", "importance": 6, "tags": ["region"]},
        ]

        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_world_context_by_query(
            mock_search, knowledge, llm, prep_logger,
            search_terms=["Brevoy", "Rostland"],
            campaign_id="test-campaign",
        )

        # Only 1 LLM call because Brevoy was deduped
        assert llm._call_index == 1
        assert count == 1
        knowledge.close()

    def test_no_results_returns_zero(self, tmp_path):
        """Test that no search results returns 0."""
        mock_search = MockPathfinderSearch()
        mock_search.search = lambda query, **kwargs: []

        llm = _make_llm_with_json([])
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_world_context_by_query(
            mock_search, knowledge, llm, prep_logger,
            search_terms=["Nonexistent Place"],
            campaign_id="test-campaign",
        )

        assert count == 0
        assert llm._call_index == 0  # No LLM calls
        knowledge.close()

    def test_progress_callback(self, tmp_path):
        """Test that progress callback is called during query-based seeding."""
        mock_search = MockPathfinderSearch()
        mock_search.search = lambda query, **kwargs: [
            {
                "name": query,
                "type": "region",
                "category": "location",
                "source": "WG",
                "book": "WG",
                "book_type": "setting",
                "page": 1,
                "content": f"Info about {query}",
                "metadata": {},
                "score": 5.0,
            }
        ]

        json_output = [{"content": "A fact", "importance": 5, "tags": []}]
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        messages = []
        seed_world_context_by_query(
            mock_search, knowledge, llm, prep_logger,
            search_terms=["Term1", "Term2"],
            campaign_id="test-campaign",
            on_progress=messages.append,
        )

        assert len(messages) >= 2
        assert any("searching" in m for m in messages)
        assert any("unique entities" in m for m in messages)
        knowledge.close()


# ---------------------------------------------------------------------------
# Pipeline timing and token tracking tests
# ---------------------------------------------------------------------------


class TestPipelineTimingAndTokens:
    def _make_pipeline_mocks(self, tmp_path):
        """Set up a pipeline with mocks for a simple end-to-end test."""
        mock_search = MockPathfinderSearch()

        pg_entities = [
            {
                "name": "Town",
                "type": "settlement",
                "category": "location",
                "source": "PG",
                "book": "PG",
                "book_type": "players_guide",
                "page": 1,
                "content": "A town",
                "metadata": {},
            },
        ]

        def list_entities(**kwargs):
            book = kwargs.get("book")
            if book == "PG":
                return pg_entities
            return []

        mock_search.list_entities = list_entities

        book_types = {"PG": "players_guide"}
        mock_search.resolve_book_name = lambda name: name if name in book_types else None
        mock_search.get_book_summary = lambda book: {"book_type": book_types.get(book, "unknown")}

        party_json = json.dumps([{"content": "A party fact", "importance": 5, "tags": []}])
        llm = MockLLMBackend(responses=[
            LLMResponse(text=party_json, usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}),
        ])

        return mock_search, llm

    def test_per_step_timing(self, tmp_path):
        mock_search, llm = self._make_pipeline_mocks(tmp_path)

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["PG"])

        assert result.party_duration_ms > 0
        assert result.duration_ms > 0
        assert result.party_duration_ms <= result.duration_ms

    def test_token_aggregation(self, tmp_path):
        mock_search, llm = self._make_pipeline_mocks(tmp_path)

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["PG"])

        assert result.total_prompt_tokens == 200
        assert result.total_completion_tokens == 80
        assert result.total_tokens == 280

    def test_pipeline_on_progress(self, tmp_path):
        mock_search, llm = self._make_pipeline_mocks(tmp_path)

        messages = []
        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
            on_progress=messages.append,
        )

        result = pipeline.run(["PG"])

        assert result.party_knowledge_count == 1
        # Pipeline emits book info + step headers + synthesis progress
        assert len(messages) >= 2
        assert any("Book:" in m for m in messages)
        assert any("Party Knowledge" in m for m in messages)


# ---------------------------------------------------------------------------
# Pipeline skip_steps tests
# ---------------------------------------------------------------------------


class TestPipelineSkipSteps:
    def test_skip_npc(self, tmp_path):
        """Test that skipping NPC step doesn't run NPC seeding."""
        mock_search = MockPathfinderSearch()

        entities = [
            {
                "name": "Town",
                "type": "settlement",
                "category": "location",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 1,
                "content": "A town",
                "metadata": {},
            },
        ]
        npcs = [
            {
                "name": "Boss",
                "type": "npc",
                "category": "npc",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 10,
                "content": "Big bad",
                "metadata": {},
            },
        ]

        def list_entities(**kwargs):
            cat = kwargs.get("category")
            if cat == "npc" or (isinstance(cat, list) and "npc" in cat):
                return npcs
            return entities

        mock_search.list_entities = list_entities
        mock_search.resolve_book_name = lambda name: "AP" if name == "AP" else None
        mock_search.get_book_summary = lambda book: {"book_type": "adventure"}

        llm = MockLLMBackend(responses=[
            LLMResponse(text="[]", usage={}),
        ])

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["AP"], skip_steps=["npc"])

        # NPC step was skipped
        assert result.npc_knowledge_count == 0
        assert result.npc_duration_ms == 0.0

    def test_skip_party_and_world(self, tmp_path):
        mock_search = MockPathfinderSearch()
        mock_search.resolve_book_name = lambda name: "PG" if name == "PG" else None
        mock_search.get_book_summary = lambda book: {"book_type": "players_guide"}

        llm = MockLLMBackend(responses=[
            LLMResponse(text="[]", usage={}),
        ])

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["PG"], skip_steps=["party", "world"])

        assert result.party_knowledge_count == 0
        assert result.world_context_count == 0
        # No LLM calls at all for a PG with party+world skipped
        assert llm._call_index == 0

    def test_query_based_world_context_in_pipeline(self, tmp_path):
        """Test that pipeline uses query-based world context when search_terms given."""
        mock_search = MockPathfinderSearch()
        mock_search.resolve_book_name = lambda name: "AP" if name == "AP" else None
        mock_search.get_book_summary = lambda book: {"book_type": "adventure"}

        # search() returns results for query-based world context
        mock_search.search = lambda query, **kwargs: [
            {
                "name": query,
                "type": "region",
                "category": "location",
                "source": "WG",
                "book": "WG",
                "book_type": "setting",
                "page": 1,
                "content": f"Info about {query}",
                "metadata": {},
                "score": 5.0,
            }
        ]

        # list_entities returns nothing for NPC (skip NPC for simplicity)
        mock_search.list_entities = lambda **kwargs: []

        world_json = json.dumps([{"content": "World fact", "importance": 6, "tags": ["region"]}])
        llm = MockLLMBackend(responses=[
            LLMResponse(text=world_json, usage={"prompt_tokens": 100, "completion_tokens": 40}),
        ])

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(
            ["AP"],
            search_terms=["Stolen Lands", "Brevoy"],
            campaign_background="A campaign in the north.",
            skip_steps=["npc", "subsystem"],
        )

        # Party has no books (AP is not PG/setting), so only world context runs
        assert result.world_context_count == 1
        assert result.world_duration_ms > 0


# ---------------------------------------------------------------------------
# resolve_ap_books tests
# ---------------------------------------------------------------------------


class TestResolveAPBooks:
    def test_single_book_ap(self):
        """Test resolving a single-book AP like Kingmaker."""
        mock_search = MockPathfinderSearch()

        # Override list_books_with_summaries to include Kingmaker
        mock_search.list_books_with_summaries = lambda **kwargs: [
            {
                "book": "Kingmaker",
                "book_type": "adventure",
                "total_pages": 642,
                "summary": "A massive AP",
            },
        ]

        mock_search.list_chapters = lambda book: [
            {"chapter": "Chapter 1", "page_start": 1, "page_end": 100, "page_count": 100},
        ]
        mock_search.get_chapter_summary = lambda book, chapter: {
            "summary": "The adventure begins.",
        }

        result = resolve_ap_books(mock_search, "Kingmaker")

        assert len(result) == 1
        assert result[0]["name"] == "Kingmaker"
        assert result[0]["total_pages"] == 642
        assert len(result[0]["chapters"]) == 1

    def test_multi_book_ap(self):
        """Test resolving a multi-book AP like Curtain Call (3 books)."""
        mock_search = MockPathfinderSearch()

        mock_search.list_books_with_summaries = lambda **kwargs: [
            {"book": "Curtain Call 1 of 3 - Stage Fright", "book_type": "adventure", "total_pages": 100, "summary": "Act 1"},
            {"book": "Curtain Call 2 of 3 - Skinsaw Man", "book_type": "adventure", "total_pages": 100, "summary": "Act 2"},
            {"book": "Curtain Call 3 of 3 - Bring the House Down", "book_type": "adventure", "total_pages": 100, "summary": "Act 3"},
            {"book": "Kingmaker", "book_type": "adventure", "total_pages": 642, "summary": "Other AP"},
        ]

        mock_search.list_chapters = lambda book: [
            {"chapter": "Chapter 1", "page_start": 1, "page_end": 50, "page_count": 50},
        ]
        mock_search.get_chapter_summary = lambda book, chapter: {"summary": "A chapter."}

        result = resolve_ap_books(mock_search, "Curtain Call")

        assert len(result) == 3
        names = [b["name"] for b in result]
        assert all("Curtain Call" in n for n in names)

    def test_no_match_falls_back(self):
        """Test fallback to resolve_book_name for unknown AP names."""
        mock_search = MockPathfinderSearch()
        mock_search.list_books_with_summaries = lambda **kwargs: []
        # resolve_book_name returns None for unknown books by default
        mock_search.resolve_book_name = lambda name: None

        result = resolve_ap_books(mock_search, "Nonexistent AP")
        assert result == []


# ---------------------------------------------------------------------------
# generate_background tests
# ---------------------------------------------------------------------------


class TestGenerateBackground:
    def test_basic_generation(self):
        """Test generating background from AP book summaries."""
        mock_search = MockPathfinderSearch()

        mock_search.list_books_with_summaries = lambda **kwargs: [
            {
                "book": "Kingmaker",
                "book_type": "adventure",
                "total_pages": 642,
                "summary": "An adventure about settling the Stolen Lands.",
            },
        ]

        mock_search.list_chapters = lambda book: [
            {"chapter": "Chapter 1", "page_start": 1, "page_end": 100, "page_count": 100},
        ]
        mock_search.get_chapter_summary = lambda book, chapter: {
            "summary": "Charter from the Swordlords to explore the Stolen Lands.",
        }

        bg_json = json.dumps({
            "background": "The Stolen Lands await adventurers brave enough to tame them.",
            "search_terms": ["Stolen Lands", "Brevoy", "Swordlords"],
        })

        llm = MockLLMBackend(responses=[
            LLMResponse(text=bg_json, usage={"prompt_tokens": 500, "completion_tokens": 200}),
        ])

        result = generate_background(mock_search, llm, "Kingmaker")

        assert result["background"] == "The Stolen Lands await adventurers brave enough to tame them."
        assert "Stolen Lands" in result["search_terms"]
        assert len(result["search_terms"]) == 3

    def test_no_books_returns_empty(self):
        """Test that no books returns empty dict."""
        mock_search = MockPathfinderSearch()
        mock_search.list_books_with_summaries = lambda **kwargs: []
        mock_search.resolve_book_name = lambda name: None

        llm = MockLLMBackend()

        result = generate_background(mock_search, llm, "Nonexistent")
        assert result == {}

    def test_progress_callback(self):
        """Test that progress callback is called during generation."""
        mock_search = MockPathfinderSearch()

        mock_search.list_books_with_summaries = lambda **kwargs: [
            {"book": "Test AP", "book_type": "adventure", "total_pages": 100, "summary": "Test"},
        ]
        mock_search.list_chapters = lambda book: []
        mock_search.get_chapter_summary = lambda book, chapter: None

        bg_json = json.dumps({"background": "A background.", "search_terms": ["term1"]})
        llm = MockLLMBackend(responses=[
            LLMResponse(text=bg_json, usage={}),
        ])

        messages = []
        generate_background(mock_search, llm, "Test AP", on_progress=messages.append)

        assert len(messages) >= 2
        assert any("Generating" in m for m in messages)

    def test_markdown_fenced_response(self):
        """Test that markdown-fenced JSON is handled correctly."""
        mock_search = MockPathfinderSearch()

        mock_search.list_books_with_summaries = lambda **kwargs: [
            {"book": "Test AP", "book_type": "adventure", "total_pages": 100, "summary": "Test"},
        ]
        mock_search.list_chapters = lambda book: []
        mock_search.get_chapter_summary = lambda book, chapter: None

        bg_json = '```json\n{"background": "Fenced background.", "search_terms": ["term"]}\n```'
        llm = MockLLMBackend(responses=[
            LLMResponse(text=bg_json, usage={}),
        ])

        result = generate_background(mock_search, llm, "Test AP")
        assert result["background"] == "Fenced background."


# ---------------------------------------------------------------------------
# NPC knowledge hints in context tests
# ---------------------------------------------------------------------------


class TestNPCKnowledgeHints:
    def test_hints_with_npcs_present(self):
        """Test that NPC knowledge hints are injected when NPCs are in scene."""
        from gm_agent.context import _build_npc_knowledge_hints
        from gm_agent.storage.schemas import SceneState, Session

        session = Session(
            id="test",
            campaign_id="test",
            scene_state=SceneState(
                location="Market Square",
                npcs_present=["Tamily Tanderveil", "Wrin Sivinxi"],
            ),
        )

        hints = _build_npc_knowledge_hints(session)
        assert hints is not None
        assert "Tamily Tanderveil" in hints
        assert "Wrin Sivinxi" in hints
        assert "what_will_npc_share" in hints

    def test_no_hints_without_npcs(self):
        """Test that no hints are generated when no NPCs present."""
        from gm_agent.context import _build_npc_knowledge_hints
        from gm_agent.storage.schemas import SceneState, Session

        session = Session(
            id="test",
            campaign_id="test",
            scene_state=SceneState(location="Empty Room", npcs_present=[]),
        )

        hints = _build_npc_knowledge_hints(session)
        assert hints is None

    def test_hints_in_full_context(self):
        """Test that NPC hints appear in the full context build."""
        from gm_agent.context import build_context
        from gm_agent.storage.schemas import Campaign, SceneState, Session

        campaign = Campaign(id="test", name="Test Campaign")
        session = Session(
            id="test",
            campaign_id="test",
            scene_state=SceneState(
                location="Tavern",
                npcs_present=["Zzamas"],
            ),
        )

        messages = build_context(campaign, session)
        system_msg = messages[0].content
        assert "NPC Knowledge Available" in system_msg
        assert "Zzamas" in system_msg


# ---------------------------------------------------------------------------
# NPC name matching heuristics tests
# ---------------------------------------------------------------------------


class TestNamesMatch:
    """Test the _names_match heuristic for NPC name deduplication."""

    # True positives — should match
    def test_first_name_matches_full_name(self):
        assert _names_match("Oleg", "Oleg Leveton")

    def test_title_prefix_stripped(self):
        assert _names_match("Jamandi Aldori", "Lady Jamandi Aldori")

    def test_both_titles_stripped(self):
        assert _names_match("Lady Jamandi", "Lady Jamandi Aldori")

    def test_typo_same_first_word(self):
        assert _names_match("Maager Varn", "Maegar Varn")

    def test_typo_transposition(self):
        assert _names_match("Ilraith Valadhkani", "Ilraith Valadkhani")

    def test_plural_singular(self):
        assert _names_match("Troll King", "Troll Kings")

    def test_epithet_matches_name(self):
        assert _names_match("Vordakai", "Cyclops lich Vordakai")

    def test_dovan_full_name(self):
        assert _names_match("Dovan", "Dovan from Nisroch")

    def test_normalize_title_lord(self):
        assert _names_match("Lord Terrion Numesti", "Terrion Numesti")

    def test_same_name(self):
        assert _names_match("Oleg Leveton", "Oleg Leveton")

    # True negatives — should NOT match
    def test_distinct_fetch_entity(self):
        assert not _names_match("Vordakai", "Vordakai's Fetch")

    def test_distinct_guards(self):
        assert not _names_match("Annamede", "Annamede's Guards")

    def test_substring_in_middle_of_word(self):
        """'Oleg' should not match 'Kob Moleg' (not at word boundary)."""
        assert not _names_match("Oleg", "Kob Moleg")

    def test_unrelated_short_edit_distance(self):
        """Corax and Foras share no first word — should not match."""
        assert not _names_match("Corax", "Foras")

    def test_unrelated_similar_length(self):
        assert not _names_match("Grabbles", "Gribbler")

    def test_short_name_ignored(self):
        """Names under 4 chars should not substring-match."""
        assert not _names_match("Aga", "Agai")

    def test_different_first_word_no_typo_match(self):
        assert not _names_match("Garum", "Garuum")  # different spelling, not a typo with same first word


# ---------------------------------------------------------------------------
# NPC name deduplication integration tests
# ---------------------------------------------------------------------------


class TestDeduplicateNPCNames:
    def _seed_npcs(self, knowledge, entries: list[tuple[str, str, str]]):
        """Helper: seed NPC knowledge entries. entries = [(char_id, char_name, content)]"""
        for cid, cname, content in entries:
            knowledge.add_knowledge(
                character_id=cid,
                character_name=cname,
                content=content,
                knowledge_type="fact",
                sharing_condition="free",
                source="test",
                importance=5,
            )

    def test_merges_first_name_into_full_name(self, tmp_path):
        knowledge = KnowledgeStore("test", base_dir=tmp_path)
        self._seed_npcs(knowledge, [
            ("oleg", "Oleg", "Oleg runs a trading post."),
            ("oleg", "Oleg", "Oleg is gruff but honest."),
            ("oleg-leveton", "Oleg Leveton", "Oleg Leveton owns Oleg's Trading Post."),
            ("oleg-leveton", "Oleg Leveton", "Oleg Leveton came from Restov."),
        ])

        result = deduplicate_npc_names(knowledge)

        assert result["merges"] == 1
        assert result["entries_moved"] == 2

        # All entries should now be under "Oleg Leveton"
        entries = knowledge.query_knowledge(character_id="oleg-leveton")
        assert len(entries) == 4
        assert all(e.character_name == "Oleg Leveton" for e in entries)

        # Nothing left under "oleg"
        entries = knowledge.query_knowledge(character_id="oleg")
        assert len(entries) == 0

        knowledge.close()

    def test_does_not_merge_distinct_entities(self, tmp_path):
        knowledge = KnowledgeStore("test", base_dir=tmp_path)
        self._seed_npcs(knowledge, [
            ("vordakai", "Vordakai", "An ancient cyclops lich."),
            ("vordakais-fetch", "Vordakai's Fetch", "A fey duplicate of Vordakai."),
        ])

        result = deduplicate_npc_names(knowledge)

        assert result["merges"] == 0
        assert result["entries_moved"] == 0
        knowledge.close()

    def test_merges_title_variants(self, tmp_path):
        knowledge = KnowledgeStore("test", base_dir=tmp_path)
        self._seed_npcs(knowledge, [
            ("jamandi-aldori", "Jamandi Aldori", "A swordlord."),
            ("lady-jamandi-aldori", "Lady Jamandi Aldori", "Lady Jamandi leads the expedition."),
            ("lady-jamandi", "Lady Jamandi", "Lady Jamandi is stern but fair."),
        ])

        result = deduplicate_npc_names(knowledge)

        # All three should merge into "Lady Jamandi Aldori" (longest)
        assert result["merges"] == 1
        assert result["entries_moved"] == 2

        entries = knowledge.query_knowledge(character_id="lady-jamandi-aldori")
        assert len(entries) == 3
        knowledge.close()

    def test_no_duplicates_returns_zero(self, tmp_path):
        knowledge = KnowledgeStore("test", base_dir=tmp_path)
        self._seed_npcs(knowledge, [
            ("alice", "Alice", "A wizard."),
            ("bob-the-fighter", "Bob the Fighter", "A fighter."),
        ])

        result = deduplicate_npc_names(knowledge)
        assert result["merges"] == 0
        assert result["entries_moved"] == 0
        knowledge.close()

    def test_progress_callback(self, tmp_path):
        knowledge = KnowledgeStore("test", base_dir=tmp_path)
        self._seed_npcs(knowledge, [
            ("oleg", "Oleg", "Fact 1."),
            ("oleg-leveton", "Oleg Leveton", "Fact 2."),
        ])

        messages = []
        deduplicate_npc_names(knowledge, on_progress=messages.append)

        assert any("Dedup" in m for m in messages)
        assert any("Merged" in m or "complete" in m for m in messages)
        knowledge.close()

    def test_skips_party_knowledge(self, tmp_path):
        """Ensure __party__ entries are never touched by dedup."""
        knowledge = KnowledgeStore("test", base_dir=tmp_path)
        knowledge.add_knowledge(
            character_id="__party__",
            character_name="Party",
            content="Party knows things.",
            source="test",
        )
        self._seed_npcs(knowledge, [
            ("oleg", "Oleg", "A fact."),
        ])

        result = deduplicate_npc_names(knowledge)
        assert result["merges"] == 0

        # Party entry untouched
        entries = knowledge.query_knowledge(character_id="__party__")
        assert len(entries) == 1
        knowledge.close()


# ---------------------------------------------------------------------------
# detect_subsystems tests
# ---------------------------------------------------------------------------


class TestDetectSubsystems:
    def _make_search_with_pages(self, page_hits: dict[str, list[str]]):
        """Create a mock search where page_hits maps keyword→list of books that have it."""
        mock_search = MockPathfinderSearch()

        def search_pages(query, **kwargs):
            book = kwargs.get("book")
            query_lower = query.lower()
            for keyword, books in page_hits.items():
                if keyword.lower() in query_lower or query_lower in keyword.lower():
                    if book in books:
                        return [{"book": book, "page_number": 1, "chapter": "Ch1", "snippet": f"...{keyword}...", "score": 5.0}]
            return []

        mock_search.search_pages = search_pages
        return mock_search

    def test_detects_kingdom(self):
        mock_search = self._make_search_with_pages({
            "kingdom building": ["Kingmaker"],
        })

        result = detect_subsystems(
            mock_search,
            books=[{"name": "Kingmaker", "book_type": "adventure"}],
        )

        assert "kingdom" in result

    def test_detects_multiple_subsystems(self):
        mock_search = self._make_search_with_pages({
            "kingdom building": ["Kingmaker"],
            "influence subsystem": ["Kingmaker"],
            "hexploration": ["Kingmaker"],
        })

        result = detect_subsystems(
            mock_search,
            books=[{"name": "Kingmaker", "book_type": "adventure"}],
        )

        assert "kingdom" in result
        assert "influence" in result
        assert "hexploration" in result

    def test_no_false_positives(self):
        """AP without subsystems returns empty list."""
        mock_search = self._make_search_with_pages({})

        result = detect_subsystems(
            mock_search,
            books=[{"name": "Simple AP", "book_type": "adventure"}],
        )

        assert result == []

    def test_ignores_non_adventure_books(self):
        """Setting/rulebook books should be ignored for detection."""
        mock_search = self._make_search_with_pages({
            "kingdom building": ["GM Core"],
        })

        result = detect_subsystems(
            mock_search,
            books=[
                {"name": "GM Core", "book_type": "rulebook"},
                {"name": "World Guide", "book_type": "setting"},
            ],
        )

        assert result == []

    def test_progress_callback(self):
        mock_search = self._make_search_with_pages({
            "chase subsystem": ["AP Vol 1"],
        })

        messages = []
        detect_subsystems(
            mock_search,
            books=[{"name": "AP Vol 1", "book_type": "adventure"}],
            on_progress=messages.append,
        )

        assert len(messages) >= 1
        assert any("chase" in m for m in messages)


# ---------------------------------------------------------------------------
# seed_subsystem_knowledge tests
# ---------------------------------------------------------------------------


class TestSeedSubsystemKnowledge:
    def _make_subsystem_search(self, entities=None, page_results=None, search_results=None):
        """Create a mock search for subsystem knowledge seeding."""
        mock_search = MockPathfinderSearch()

        mock_search.list_entities = lambda **kwargs: entities or []
        mock_search.search = lambda query, **kwargs: search_results or []
        mock_search.search_pages = lambda query, **kwargs: page_results or []

        return mock_search

    def test_basic_seeding(self, tmp_path):
        entities = [
            {
                "name": "Kingdom Turn",
                "type": "subsystem",
                "category": "game_mechanic",
                "source": "Kingmaker",
                "book": "Kingmaker",
                "book_type": "adventure",
                "page": 520,
                "content": "Each kingdom turn follows these phases...",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "Kingdom turns have 4 phases: Upkeep, Commerce, Activities, Events.", "importance": 9, "tags": ["kingdom"]},
            {"content": "The DC for kingdom checks equals 14 + kingdom level.", "importance": 8, "tags": ["kingdom", "dc"]},
        ]

        mock_search = self._make_subsystem_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_subsystem_knowledge(
            mock_search, knowledge, llm, prep_logger,
            ap_books=[{"name": "Kingmaker", "book_type": "adventure"}],
            subsystem_types=["kingdom"],
            campaign_id="test-campaign",
        )

        assert count == 2

        entries = knowledge.query_knowledge(character_id=PARTY_KNOWLEDGE_ID, tags=["subsystem"])
        assert len(entries) == 2
        assert all(e.character_name == "Subsystem Rules" for e in entries)
        assert all(e.sharing_condition == "free" for e in entries)
        assert all("kingdom" in e.tags for e in entries)
        assert entries[0].source == "prep:subsystem:kingdom"

        # Verify log
        log_entries = prep_logger.read()
        assert len(log_entries) == 1
        assert log_entries[0].step == "subsystem_knowledge"
        assert log_entries[0].entity == "kingdom"

        knowledge.close()

    def test_multiple_subsystem_types(self, tmp_path):
        """Each subsystem type gets its own LLM call."""
        entities = [
            {
                "name": "Subsystem Rule",
                "type": "game_mechanic",
                "category": "game_mechanic",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 100,
                "content": "Rules for this subsystem.",
                "metadata": {},
            },
        ]

        kingdom_output = [{"content": "Kingdom turns have 4 phases.", "importance": 8, "tags": ["kingdom"]}]
        influence_output = [{"content": "Influence uses Discovery and Influence actions.", "importance": 8, "tags": ["influence"]}]

        mock_search = self._make_subsystem_search(entities=entities)
        # Need two responses (one per subsystem type)
        llm = MockLLMBackend(responses=[
            LLMResponse(text=json.dumps(kingdom_output), usage={"prompt_tokens": 100, "completion_tokens": 50}),
            LLMResponse(text=json.dumps(influence_output), usage={"prompt_tokens": 100, "completion_tokens": 50}),
        ])
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_subsystem_knowledge(
            mock_search, knowledge, llm, prep_logger,
            ap_books=[{"name": "AP", "book_type": "adventure"}],
            subsystem_types=["kingdom", "influence"],
            campaign_id="test-campaign",
        )

        assert count == 2  # 1 per type
        assert llm._call_index == 2  # One LLM call per type
        knowledge.close()

    def test_no_content_skips_gracefully(self, tmp_path):
        """When no content found for a subsystem type, skip without LLM call."""
        mock_search = self._make_subsystem_search()
        llm = _make_llm_with_json([])
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = seed_subsystem_knowledge(
            mock_search, knowledge, llm, prep_logger,
            ap_books=[{"name": "AP", "book_type": "adventure"}],
            subsystem_types=["kingdom"],
            campaign_id="test-campaign",
        )

        assert count == 0
        assert llm._call_index == 0  # No LLM calls
        knowledge.close()

    def test_dedup_skips_duplicates(self, tmp_path):
        """Duplicate content is skipped via has_similar_knowledge."""
        entities = [
            {
                "name": "Rule",
                "type": "subsystem",
                "category": "game_mechanic",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 1,
                "content": "A rule.",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "Kingdom turns have 4 phases.", "importance": 8, "tags": ["kingdom"]},
        ]

        mock_search = self._make_subsystem_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        # Pre-seed with the same content
        knowledge.add_knowledge(
            character_id=PARTY_KNOWLEDGE_ID,
            character_name="Subsystem Rules",
            content="Kingdom turns have 4 phases.",
            source="prep:subsystem:kingdom",
        )

        count = seed_subsystem_knowledge(
            mock_search, knowledge, llm, prep_logger,
            ap_books=[{"name": "AP", "book_type": "adventure"}],
            subsystem_types=["kingdom"],
            campaign_id="test-campaign",
        )

        assert count == 0  # Duplicate skipped
        knowledge.close()

    def test_progress_callback(self, tmp_path):
        entities = [
            {
                "name": "Rule",
                "type": "subsystem",
                "category": "game_mechanic",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 1,
                "content": "A rule.",
                "metadata": {},
            },
        ]

        json_output = [
            {"content": "A subsystem rule.", "importance": 8, "tags": []},
        ]

        mock_search = self._make_subsystem_search(entities=entities)
        llm = _make_llm_with_json(json_output)
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        prep_logger = PrepLogger("test-campaign", base_dir=tmp_path)

        messages = []
        seed_subsystem_knowledge(
            mock_search, knowledge, llm, prep_logger,
            ap_books=[{"name": "AP", "book_type": "adventure"}],
            subsystem_types=["kingdom"],
            campaign_id="test-campaign",
            on_progress=messages.append,
        )

        assert len(messages) >= 1
        assert any("kingdom" in m for m in messages)
        knowledge.close()


# ---------------------------------------------------------------------------
# Pipeline subsystem step tests
# ---------------------------------------------------------------------------


class TestPipelineSubsystemStep:
    def test_subsystem_step_runs(self, tmp_path):
        """Subsystem step detects and seeds in the pipeline."""
        mock_search = MockPathfinderSearch()

        # search_pages detects kingdom
        def search_pages(query, **kwargs):
            if "kingdom" in query.lower():
                return [{"book": "AP", "page_number": 500, "chapter": "Kingdom", "snippet": "...kingdom building...", "score": 5.0}]
            return []

        mock_search.search_pages = search_pages

        entities = [
            {
                "name": "Kingdom Rules",
                "type": "subsystem",
                "category": "game_mechanic",
                "source": "AP",
                "book": "AP",
                "book_type": "adventure",
                "page": 500,
                "content": "Kingdom building rules.",
                "metadata": {},
            },
        ]
        mock_search.list_entities = lambda **kwargs: entities
        mock_search.resolve_book_name = lambda name: "AP" if name == "AP" else None
        mock_search.get_book_summary = lambda book: {"book_type": "adventure"}
        mock_search.search = lambda query, **kwargs: []

        subsystem_json = json.dumps([
            {"content": "Kingdom turn phases.", "importance": 9, "tags": ["kingdom"]},
        ])
        llm = MockLLMBackend(responses=[
            LLMResponse(text=subsystem_json, usage={"prompt_tokens": 100, "completion_tokens": 50}),
        ])

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["AP"], skip_steps=["npc", "party", "world"])

        assert result.subsystems_detected == ["kingdom"]
        assert result.subsystem_knowledge_count == 1
        assert result.subsystem_duration_ms > 0
        assert result.total_count == 1

    def test_skip_subsystem(self, tmp_path):
        """--skip subsystem skips the step entirely."""
        mock_search = MockPathfinderSearch()
        mock_search.resolve_book_name = lambda name: "AP" if name == "AP" else None
        mock_search.get_book_summary = lambda book: {"book_type": "adventure"}
        mock_search.list_entities = lambda **kwargs: []

        llm = MockLLMBackend(responses=[])

        pipeline = PrepPipeline(
            campaign_id="test",
            llm=llm,
            search=mock_search,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
        )

        result = pipeline.run(["AP"], skip_steps=["npc", "party", "world", "subsystem"])

        assert result.subsystems_detected == []
        assert result.subsystem_knowledge_count == 0
        assert result.subsystem_duration_ms == 0.0
