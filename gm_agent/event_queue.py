"""Event queue for automation mode message handling.

Provides priority-based queuing with depth limits, expiry, and both
Redis-backed (production) and in-memory (development) implementations.
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable

from pydantic import BaseModel, Field

from .config import REDIS_URL


class QueuedEvent(BaseModel):
    """Event queued for processing."""

    event_id: str
    event_type: str  # "playerChat", "combatTurn", etc.
    priority: int = 0  # 0=exploration, 1=combat, 2=urgent
    payload: dict[str, Any]
    queued_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if event has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class EventQueue:
    """Priority-based event queue with depth limits and expiry.

    Supports both Redis-backed (production) and in-memory (development) modes.
    Events are prioritized by combat urgency and processed FIFO within each priority.
    """

    def __init__(
        self,
        campaign_id: str,
        max_depth: int = 100,
        max_exploration_depth: int = 20,
        default_ttl_seconds: int = 300,  # 5 minutes
        use_redis: bool | None = None,
    ):
        """Initialize event queue.

        Args:
            campaign_id: Campaign ID for queue isolation
            max_depth: Maximum total queue size
            max_exploration_depth: Max exploration events (lower priority dropped)
            default_ttl_seconds: Default event expiry time
            use_redis: Use Redis if available (auto-detect if None)
        """
        self.campaign_id = campaign_id
        self.max_depth = max_depth
        self.max_exploration_depth = max_exploration_depth
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.Lock()

        # Try to use Redis if available
        if use_redis is None:
            use_redis = REDIS_URL is not None and REDIS_URL.startswith("redis://")

        self.use_redis = use_redis
        self._redis = None

        if use_redis:
            try:
                import redis

                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                # Test connection
                self._redis.ping()
            except Exception:
                # Fall back to in-memory
                self.use_redis = False
                self._redis = None

        # In-memory queues (used if Redis unavailable)
        if not self.use_redis:
            self._queues: dict[int, deque[QueuedEvent]] = {
                0: deque(),  # Exploration
                1: deque(),  # Combat
                2: deque(),  # Urgent
            }

    def _redis_key(self, priority: int) -> str:
        """Get Redis key for priority queue."""
        return f"automation_queue:{self.campaign_id}:{priority}"

    def push(
        self,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 0,
        ttl_seconds: int | None = None,
    ) -> str:
        """Add event to queue.

        Args:
            event_type: Type of event ("playerChat", "combatTurn", etc.)
            payload: Event data
            priority: Priority level (0=exploration, 1=combat, 2=urgent)
            ttl_seconds: Time to live in seconds (uses default if None)

        Returns:
            Event ID

        Raises:
            ValueError: If queue is full
        """
        with self._lock:
            # Check total depth
            total = self.size()
            if total >= self.max_depth:
                raise ValueError(f"Queue full (max {self.max_depth} events)")

            # Check exploration depth (priority 0 only)
            if priority == 0:
                exploration_count = self.size(priority=0)
                if exploration_count >= self.max_exploration_depth:
                    # Drop oldest exploration event
                    self._pop_oldest_exploration()

            # Create event
            event = QueuedEvent(
                event_id=f"evt-{int(time.time() * 1000)}",
                event_type=event_type,
                priority=priority,
                payload=payload,
            )

            # Set expiry
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            event.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

            # Push to queue
            if self.use_redis:
                self._redis.rpush(
                    self._redis_key(priority), event.model_dump_json()
                )
            else:
                self._queues[priority].append(event)

            return event.event_id

    def pop(self) -> QueuedEvent | None:
        """Pop highest priority non-expired event.

        Returns:
            Event or None if queue empty
        """
        with self._lock:
            # Try each priority level (highest first)
            for priority in [2, 1, 0]:
                while True:
                    event = self._pop_from_priority(priority)
                    if event is None:
                        break

                    # Check expiry
                    if event.is_expired():
                        continue  # Skip and try next

                    return event

            return None

    def _pop_from_priority(self, priority: int) -> QueuedEvent | None:
        """Pop from specific priority queue."""
        if self.use_redis:
            data = self._redis.lpop(self._redis_key(priority))
            if data is None:
                return None
            return QueuedEvent.model_validate_json(data)
        else:
            if not self._queues[priority]:
                return None
            return self._queues[priority].popleft()

    def _pop_oldest_exploration(self) -> None:
        """Remove oldest exploration event to make room."""
        if self.use_redis:
            self._redis.lpop(self._redis_key(0))
        else:
            if self._queues[0]:
                self._queues[0].popleft()

    def size(self, priority: int | None = None) -> int:
        """Get queue size.

        Args:
            priority: Specific priority level or None for total

        Returns:
            Number of events
        """
        if priority is not None:
            if self.use_redis:
                return self._redis.llen(self._redis_key(priority))
            else:
                return len(self._queues[priority])
        else:
            # Total across all priorities
            if self.use_redis:
                return sum(
                    self._redis.llen(self._redis_key(p)) for p in [0, 1, 2]
                )
            else:
                return sum(len(q) for q in self._queues.values())

    def peek(self, priority: int | None = None) -> list[QueuedEvent]:
        """View events without removing them.

        Args:
            priority: Specific priority or None for all

        Returns:
            List of events
        """
        events = []

        if priority is not None:
            priorities = [priority]
        else:
            priorities = [2, 1, 0]

        for pri in priorities:
            if self.use_redis:
                data_list = self._redis.lrange(self._redis_key(pri), 0, -1)
                for data in data_list:
                    events.append(QueuedEvent.model_validate_json(data))
            else:
                events.extend(self._queues[pri])

        return events

    def clear(self, priority: int | None = None) -> int:
        """Clear queue.

        Args:
            priority: Specific priority or None for all

        Returns:
            Number of events removed
        """
        with self._lock:
            removed = 0

            if priority is not None:
                priorities = [priority]
            else:
                priorities = [2, 1, 0]

            for pri in priorities:
                if self.use_redis:
                    removed += self._redis.llen(self._redis_key(pri))
                    self._redis.delete(self._redis_key(pri))
                else:
                    removed += len(self._queues[pri])
                    self._queues[pri].clear()

            return removed

    def stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dict with queue stats
        """
        total = self.size()
        exploration = self.size(priority=0)
        combat = self.size(priority=1)
        urgent = self.size(priority=2)

        # Count expired events
        expired = 0
        for event in self.peek():
            if event.is_expired():
                expired += 1

        return {
            "backend": "redis" if self.use_redis else "in-memory",
            "campaign_id": self.campaign_id,
            "total_events": total,
            "by_priority": {
                "exploration": exploration,
                "combat": combat,
                "urgent": urgent,
            },
            "expired_events": expired,
            "max_depth": self.max_depth,
            "max_exploration_depth": self.max_exploration_depth,
            "utilization": f"{(total / self.max_depth) * 100:.1f}%",
        }


class QueueWorker:
    """Background worker that processes events from queue."""

    def __init__(
        self,
        queue: EventQueue,
        process_func: Callable[[QueuedEvent], None],
        poll_interval: float = 0.1,
    ):
        """Initialize worker.

        Args:
            queue: EventQueue to process
            process_func: Function to call for each event
            poll_interval: Seconds between queue checks
        """
        self.queue = queue
        self.process_func = process_func
        self.poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                event = self.queue.pop()
                if event is None:
                    # Queue empty, sleep
                    time.sleep(self.poll_interval)
                    continue

                # Process event
                self.process_func(event)

            except Exception:
                # Log error but keep running
                time.sleep(self.poll_interval)
