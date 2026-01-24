"""Tests for event queue system."""

import pytest
import time
from unittest.mock import Mock

from gm_agent.event_queue import QueuedEvent, EventQueue, QueueWorker


class TestQueuedEvent:
    """Tests for QueuedEvent schema."""

    def test_event_creation(self):
        """Test creating a QueuedEvent."""
        event = QueuedEvent(
            event_id="evt-001",
            event_type="playerChat",
            priority=0,
            payload={"actorName": "Bob", "content": "Hello"},
        )

        assert event.event_id == "evt-001"
        assert event.event_type == "playerChat"
        assert event.priority == 0
        assert event.payload["actorName"] == "Bob"

    def test_event_defaults(self):
        """Test QueuedEvent default values."""
        event = QueuedEvent(
            event_id="evt-001",
            event_type="playerChat",
            payload={},
        )

        assert event.priority == 0
        assert event.queued_at is not None
        assert event.expires_at is None

    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry is set."""
        event = QueuedEvent(
            event_id="evt-001",
            event_type="playerChat",
            payload={},
        )

        assert not event.is_expired()

    def test_is_expired_future(self):
        """Test is_expired with future expiry."""
        from datetime import datetime, timedelta

        event = QueuedEvent(
            event_id="evt-001",
            event_type="playerChat",
            payload={},
            expires_at=datetime.now() + timedelta(seconds=10),
        )

        assert not event.is_expired()

    def test_is_expired_past(self):
        """Test is_expired with past expiry."""
        from datetime import datetime, timedelta

        event = QueuedEvent(
            event_id="evt-001",
            event_type="playerChat",
            payload={},
            expires_at=datetime.now() - timedelta(seconds=10),
        )

        assert event.is_expired()


class TestEventQueue:
    """Tests for EventQueue."""

    def test_create_queue_in_memory(self):
        """Test creating in-memory queue."""
        queue = EventQueue("test-campaign", use_redis=False)

        assert queue.campaign_id == "test-campaign"
        assert not queue.use_redis
        assert queue.size() == 0

    def test_push_event(self):
        """Test pushing an event to queue."""
        queue = EventQueue("test-campaign", use_redis=False)

        event_id = queue.push(
            event_type="playerChat",
            payload={"content": "Hello"},
            priority=0,
        )

        assert event_id.startswith("evt-")
        assert queue.size() == 1

    def test_push_multiple_priorities(self):
        """Test pushing events with different priorities."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("playerChat", {}, priority=0)
        queue.push("combatTurn", {}, priority=1)
        queue.push("urgent", {}, priority=2)

        assert queue.size() == 3
        assert queue.size(priority=0) == 1
        assert queue.size(priority=1) == 1
        assert queue.size(priority=2) == 1

    def test_pop_priority_order(self):
        """Test popping events in priority order."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("exploration", {"type": "low"}, priority=0)
        queue.push("combat", {"type": "medium"}, priority=1)
        queue.push("urgent", {"type": "high"}, priority=2)

        # Should pop in priority order: 2, 1, 0
        event1 = queue.pop()
        assert event1.priority == 2
        assert event1.payload["type"] == "high"

        event2 = queue.pop()
        assert event2.priority == 1
        assert event2.payload["type"] == "medium"

        event3 = queue.pop()
        assert event3.priority == 0
        assert event3.payload["type"] == "low"

        assert queue.pop() is None

    def test_pop_fifo_within_priority(self):
        """Test FIFO ordering within same priority."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("first", {"order": 1}, priority=0)
        queue.push("second", {"order": 2}, priority=0)
        queue.push("third", {"order": 3}, priority=0)

        event1 = queue.pop()
        assert event1.payload["order"] == 1

        event2 = queue.pop()
        assert event2.payload["order"] == 2

        event3 = queue.pop()
        assert event3.payload["order"] == 3

    def test_max_depth_limit(self):
        """Test queue rejects events when max depth reached."""
        queue = EventQueue("test-campaign", max_depth=3, use_redis=False)

        queue.push("event1", {}, priority=0)
        queue.push("event2", {}, priority=0)
        queue.push("event3", {}, priority=0)

        # Fourth push should fail
        with pytest.raises(ValueError, match="Queue full"):
            queue.push("event4", {}, priority=0)

    def test_max_exploration_depth(self):
        """Test exploration events dropped when limit reached."""
        queue = EventQueue(
            "test-campaign",
            max_depth=10,
            max_exploration_depth=2,
            use_redis=False,
        )

        queue.push("explore1", {"order": 1}, priority=0)
        queue.push("explore2", {"order": 2}, priority=0)

        # Third exploration event should drop oldest
        queue.push("explore3", {"order": 3}, priority=0)

        assert queue.size(priority=0) == 2

        # Should have dropped first event
        event1 = queue.pop()
        assert event1.payload["order"] == 2

        event2 = queue.pop()
        assert event2.payload["order"] == 3

    def test_skip_expired_events(self):
        """Test queue skips expired events when popping."""
        from datetime import datetime, timedelta

        queue = EventQueue("test-campaign", use_redis=False)

        # Push expired event
        queue.push("expired", {"type": "old"}, priority=0, ttl_seconds=0)

        # Wait a moment to ensure expiry
        time.sleep(0.01)

        # Push valid event
        queue.push("valid", {"type": "new"}, priority=0)

        # Should skip expired and return valid
        event = queue.pop()
        assert event is not None
        assert event.payload["type"] == "new"

    def test_peek(self):
        """Test peeking at queue without removing events."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("event1", {}, priority=0)
        queue.push("event2", {}, priority=1)

        events = queue.peek()
        assert len(events) == 2
        assert queue.size() == 2  # Not removed

    def test_peek_by_priority(self):
        """Test peeking at specific priority."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("explore", {}, priority=0)
        queue.push("combat", {}, priority=1)

        combat_events = queue.peek(priority=1)
        assert len(combat_events) == 1
        assert combat_events[0].event_type == "combat"

    def test_clear_all(self):
        """Test clearing entire queue."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("event1", {}, priority=0)
        queue.push("event2", {}, priority=1)
        queue.push("event3", {}, priority=2)

        removed = queue.clear()
        assert removed == 3
        assert queue.size() == 0

    def test_clear_priority(self):
        """Test clearing specific priority."""
        queue = EventQueue("test-campaign", use_redis=False)

        queue.push("explore1", {}, priority=0)
        queue.push("explore2", {}, priority=0)
        queue.push("combat1", {}, priority=1)

        removed = queue.clear(priority=0)
        assert removed == 2
        assert queue.size() == 1
        assert queue.size(priority=0) == 0
        assert queue.size(priority=1) == 1

    def test_stats(self):
        """Test queue statistics."""
        queue = EventQueue("test-campaign", use_redis=False, max_depth=100)

        queue.push("explore", {}, priority=0)
        queue.push("combat", {}, priority=1)

        stats = queue.stats()
        assert stats["backend"] == "in-memory"
        assert stats["campaign_id"] == "test-campaign"
        assert stats["total_events"] == 2
        assert stats["by_priority"]["exploration"] == 1
        assert stats["by_priority"]["combat"] == 1
        assert stats["max_depth"] == 100


class TestQueueWorker:
    """Tests for QueueWorker."""

    def test_create_worker(self):
        """Test creating a queue worker."""
        queue = EventQueue("test-campaign", use_redis=False)
        process_func = Mock()

        worker = QueueWorker(queue, process_func)

        assert worker.queue == queue
        assert worker.process_func == process_func
        assert not worker._running

    def test_worker_processes_events(self):
        """Test worker processes events from queue."""
        queue = EventQueue("test-campaign", use_redis=False)
        processed_events = []

        def process_func(event):
            processed_events.append(event.event_id)

        worker = QueueWorker(queue, process_func, poll_interval=0.01)

        # Add events before starting
        event_id1 = queue.push("event1", {}, priority=0)
        event_id2 = queue.push("event2", {}, priority=0)

        # Start worker
        worker.start()

        # Wait for processing
        time.sleep(0.1)

        # Stop worker
        worker.stop()

        # Check events were processed
        assert event_id1 in processed_events
        assert event_id2 in processed_events
        assert queue.size() == 0

    def test_worker_processes_by_priority(self):
        """Test worker processes higher priority first."""
        queue = EventQueue("test-campaign", use_redis=False)
        processed_events = []

        def process_func(event):
            processed_events.append((event.event_id, event.priority))

        worker = QueueWorker(queue, process_func, poll_interval=0.01)

        # Add events in mixed priority
        queue.push("low", {}, priority=0)
        queue.push("high", {}, priority=2)
        queue.push("medium", {}, priority=1)

        worker.start()
        time.sleep(0.1)
        worker.stop()

        # Should be processed in priority order
        assert len(processed_events) == 3
        priorities = [p for _, p in processed_events]
        assert priorities == [2, 1, 0]  # High to low

    def test_worker_handles_errors(self):
        """Test worker continues after processing errors."""
        queue = EventQueue("test-campaign", use_redis=False)
        processed_count = [0]

        def process_func(event):
            processed_count[0] += 1
            if processed_count[0] == 2:
                raise ValueError("Test error")

        worker = QueueWorker(queue, process_func, poll_interval=0.01)

        queue.push("event1", {}, priority=0)
        queue.push("event2", {}, priority=0)  # Will error
        queue.push("event3", {}, priority=0)

        worker.start()
        time.sleep(0.15)
        worker.stop()

        # Should have processed all 3 despite error
        assert processed_count[0] == 3

    def test_worker_start_stop(self):
        """Test starting and stopping worker."""
        queue = EventQueue("test-campaign", use_redis=False)
        process_func = Mock()

        worker = QueueWorker(queue, process_func)

        # Start
        worker.start()
        assert worker._running

        # Stop
        worker.stop()
        assert not worker._running

    def test_worker_idempotent_start(self):
        """Test starting worker multiple times is safe."""
        queue = EventQueue("test-campaign", use_redis=False)
        process_func = Mock()

        worker = QueueWorker(queue, process_func)

        worker.start()
        worker.start()  # Should not error

        assert worker._running

        worker.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
