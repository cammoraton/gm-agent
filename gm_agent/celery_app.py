"""Celery application configuration.

This module creates and configures the Celery application instance
that is shared between the API and worker processes.
"""

import os

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None  # type: ignore

# Get Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery app (if Celery is available)
if CELERY_AVAILABLE:
    celery_app = Celery(  # type: ignore
        "gm_agent",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=["gm_agent.tasks", "gm_agent.mcp_tasks"],
    )

    # Celery configuration
    celery_app.conf.update(
        # Task settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Task execution settings
        task_acks_late=True,  # Acknowledge after task completes (for reliability)
        task_reject_on_worker_lost=True,  # Requeue if worker dies
        task_time_limit=300,  # 5 minute hard limit
        task_soft_time_limit=240,  # 4 minute soft limit (raises exception)
        # Result settings
        result_expires=3600,  # Results expire after 1 hour
        # Worker settings
        worker_prefetch_multiplier=1,  # Don't prefetch (LLM tasks are long)
        worker_concurrency=4,  # Default concurrency
        # Retry settings
        task_default_retry_delay=5,  # 5 second retry delay
        task_max_retries=3,  # Max 3 retries
        # Task routing (Phase 4.2: Separate worker pools)
        task_routes={
            "gm_agent.tasks.process_player_chat_async": {"queue": "automation"},
            "gm_agent.tasks.process_npc_turn_async": {"queue": "automation"},
            "gm_agent.mcp_tasks.*": {"queue": "mcp"},
        },
        # Beat schedule for periodic tasks (state checkpointing)
        beat_schedule={
            "checkpoint-state": {
                "task": "gm_agent.tasks.checkpoint_state",
                "schedule": 30.0,  # Every 30 seconds
            },
        },
    )
else:
    celery_app = None  # type: ignore


def get_celery_app():
    """Get the Celery application instance.

    Returns:
        Celery app instance

    Raises:
        RuntimeError: If Celery is not available
    """
    if not CELERY_AVAILABLE:
        raise RuntimeError("Celery is not installed. Install with: pip install 'celery[redis]'")
    return celery_app
