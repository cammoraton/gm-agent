"""Tests for FTS5 history index."""

from datetime import datetime
from pathlib import Path

import pytest

from gm_agent.storage.history import HistoryIndex, HistoryEvent


class TestHistoryEvent:
    """Tests for HistoryEvent model."""

    def test_default_values(self):
        """HistoryEvent should have sensible defaults."""
        event = HistoryEvent(
            session_id="test-session",
            event="Something happened",
        )
        assert event.session_id == "test-session"
        assert event.event == "Something happened"
        assert event.importance == "session"
        assert event.tags == []
        assert event.turn_number is None
        assert event.id is None
        assert isinstance(event.timestamp, datetime)

    def test_all_fields(self):
        """HistoryEvent should store all fields."""
        event = HistoryEvent(
            id=1,
            session_id="test-session",
            turn_number=5,
            event="Major discovery",
            importance="campaign",
            tags=["discovery", "plot"],
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
        )
        assert event.id == 1
        assert event.turn_number == 5
        assert event.importance == "campaign"
        assert event.tags == ["discovery", "plot"]

    def test_to_searchable_text(self):
        """HistoryEvent should generate searchable text."""
        event = HistoryEvent(
            session_id="test",
            event="Found the ancient key",
            tags=["key", "discovery"],
        )
        text = event.to_searchable_text()
        assert "ancient key" in text
        assert "key" in text
        assert "discovery" in text


class TestHistoryIndex:
    """Tests for HistoryIndex FTS5 database."""

    @pytest.fixture
    def history_index(self, tmp_path: Path) -> HistoryIndex:
        """Create a HistoryIndex with temp storage."""
        campaign_dir = tmp_path / "test-campaign"
        campaign_dir.mkdir(parents=True)
        index = HistoryIndex("test-campaign", base_dir=tmp_path)
        yield index
        index.close()

    def test_creates_database(self, history_index: HistoryIndex):
        """HistoryIndex should create database file."""
        assert history_index.db_path.exists()

    def test_log_event(self, history_index: HistoryIndex):
        """log_event should store event in database."""
        event = history_index.log_event(
            session_id="session-1",
            event="The party entered the dungeon",
            importance="session",
            tags=["dungeon", "exploration"],
            turn_number=1,
        )

        assert event.id is not None
        assert event.session_id == "session-1"
        assert event.event == "The party entered the dungeon"

    def test_search_finds_event(self, history_index: HistoryIndex):
        """search should find events matching query."""
        history_index.log_event(
            session_id="session-1",
            event="The party found a magical sword",
            tags=["sword", "magic", "treasure"],
        )
        history_index.log_event(
            session_id="session-1",
            event="The party fought goblins",
            tags=["combat", "goblin"],
        )

        results = history_index.search("sword")

        assert len(results) == 1
        assert "sword" in results[0].event.lower()

    def test_search_by_tags(self, history_index: HistoryIndex):
        """search should find events by tags."""
        history_index.log_event(
            session_id="session-1",
            event="An event with magic",
            tags=["magic", "spell"],
        )

        results = history_index.search("spell")

        assert len(results) == 1

    def test_search_filters_by_importance(self, history_index: HistoryIndex):
        """search should filter by importance level."""
        history_index.log_event(
            session_id="session-1",
            event="Minor event",
            importance="session",
        )
        history_index.log_event(
            session_id="session-1",
            event="Major plot event",
            importance="campaign",
        )

        session_results = history_index.search("event", importance="session")
        campaign_results = history_index.search("event", importance="campaign")

        assert len(session_results) == 1
        assert "Minor" in session_results[0].event
        assert len(campaign_results) == 1
        assert "Major" in campaign_results[0].event

    def test_search_filters_by_session(self, history_index: HistoryIndex):
        """search should filter by session ID."""
        history_index.log_event(
            session_id="session-1",
            event="Event in session 1",
        )
        history_index.log_event(
            session_id="session-2",
            event="Event in session 2",
        )

        results = history_index.search("event", session_id="session-1")

        assert len(results) == 1
        assert "session 1" in results[0].event

    def test_search_respects_limit(self, history_index: HistoryIndex):
        """search should respect limit parameter."""
        for i in range(10):
            history_index.log_event(
                session_id="session-1",
                event=f"Event number {i}",
            )

        results = history_index.search("event", limit=5)

        assert len(results) == 5

    def test_search_orders_by_timestamp(self, history_index: HistoryIndex):
        """search should return newest events first."""
        history_index.log_event(session_id="s1", event="First event")
        history_index.log_event(session_id="s1", event="Second event")
        history_index.log_event(session_id="s1", event="Third event")

        results = history_index.search("event")

        assert "Third" in results[0].event
        assert "First" in results[-1].event

    def test_get_recent(self, history_index: HistoryIndex):
        """get_recent should return most recent events."""
        for i in range(5):
            history_index.log_event(
                session_id="session-1",
                event=f"Event {i}",
            )

        results = history_index.get_recent(limit=3)

        assert len(results) == 3
        assert "Event 4" in results[0].event

    def test_get_recent_filters_importance(self, history_index: HistoryIndex):
        """get_recent should filter by importance."""
        history_index.log_event(
            session_id="s1",
            event="Session event",
            importance="session",
        )
        history_index.log_event(
            session_id="s1",
            event="Arc event",
            importance="arc",
        )

        arc_results = history_index.get_recent(importance="arc")

        assert len(arc_results) == 1
        assert "Arc" in arc_results[0].event

    def test_get_session_events(self, history_index: HistoryIndex):
        """get_session_events should return all events for session."""
        history_index.log_event(session_id="target", event="Event 1")
        history_index.log_event(session_id="target", event="Event 2")
        history_index.log_event(session_id="other", event="Other event")

        results = history_index.get_session_events("target")

        assert len(results) == 2

    def test_delete_session_events(self, history_index: HistoryIndex):
        """delete_session_events should remove all events for session."""
        history_index.log_event(session_id="to-delete", event="Event 1")
        history_index.log_event(session_id="to-delete", event="Event 2")
        history_index.log_event(session_id="keep", event="Keep this")

        deleted = history_index.delete_session_events("to-delete")

        assert deleted == 2
        assert len(history_index.get_session_events("to-delete")) == 0
        assert len(history_index.get_session_events("keep")) == 1

    def test_get_stats(self, history_index: HistoryIndex):
        """get_stats should return index statistics."""
        history_index.log_event(session_id="s1", event="E1", importance="session")
        history_index.log_event(session_id="s1", event="E2", importance="arc")
        history_index.log_event(session_id="s2", event="E3", importance="session")

        stats = history_index.get_stats()

        assert stats["total_events"] == 3
        assert stats["by_importance"]["session"] == 2
        assert stats["by_importance"]["arc"] == 1
        assert stats["sessions_with_events"] == 2

    def test_empty_search(self, history_index: HistoryIndex):
        """search with empty query should still filter correctly."""
        history_index.log_event(session_id="s1", event="E1", importance="session")
        history_index.log_event(session_id="s1", event="E2", importance="arc")

        results = history_index.search("", importance="arc")

        assert len(results) == 1

    def test_fts5_special_characters(self, history_index: HistoryIndex):
        """search should handle special characters gracefully."""
        history_index.log_event(
            session_id="s1",
            event="The +5 longsword of fire!",
        )

        # Should not crash with special characters
        results = history_index.search("longsword")
        assert len(results) == 1

    def test_context_manager(self, tmp_path: Path):
        """HistoryIndex should work as context manager."""
        campaign_dir = tmp_path / "ctx-campaign"
        campaign_dir.mkdir()

        with HistoryIndex("ctx-campaign", base_dir=tmp_path) as index:
            index.log_event(session_id="s1", event="Test event")
            results = index.search("test")
            assert len(results) == 1

    def test_reopen_preserves_data(self, tmp_path: Path):
        """Data should persist across index reopenings."""
        campaign_dir = tmp_path / "persist-campaign"
        campaign_dir.mkdir()

        # Write data
        index1 = HistoryIndex("persist-campaign", base_dir=tmp_path)
        index1.log_event(session_id="s1", event="Persisted event")
        index1.close()

        # Read data with new instance
        index2 = HistoryIndex("persist-campaign", base_dir=tmp_path)
        results = index2.search("persisted")
        index2.close()

        assert len(results) == 1
        assert "Persisted" in results[0].event
