"""Celery tasks for async LLM processing.

These tasks handle computationally expensive LLM operations asynchronously,
allowing the API to respond quickly while processing happens in the background.

Note: GameLoopController/Foundry integration stays synchronous because it needs
immediate WebSocket response for real-time gameplay.
"""

import json
import logging
import os
from typing import Any

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore

from gm_agent.chat import ChatAgent
from gm_agent.summarizer import RollingSummarizer
from gm_agent.storage.campaign import campaign_store
from gm_agent.storage.session import session_store

# Import celery_app conditionally
try:
    from gm_agent.celery_app import celery_app, CELERY_AVAILABLE
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None  # type: ignore

# If Celery is not available, skip all task decorators
if not CELERY_AVAILABLE:
    def task(*args, **kwargs):
        """No-op decorator when Celery is not available."""
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) is False else decorator(args[0])

    # Mock celery_app.task
    class MockCeleryApp:
        task = staticmethod(task)

    if celery_app is None:
        celery_app = MockCeleryApp()  # type: ignore

logger = logging.getLogger(__name__)

# Redis client for state checkpointing
_redis_client = None


def get_redis_client():
    """Get Redis client for state operations.

    Returns:
        Redis client instance

    Raises:
        RuntimeError: If Redis is not available
    """
    if not REDIS_AVAILABLE:
        raise RuntimeError("Redis is not installed. Install with: pip install redis")

    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url)  # type: ignore
    return _redis_client


# =============================================================================
# Chat Tasks
# =============================================================================


@celery_app.task(bind=True, max_retries=2, soft_time_limit=120)
def process_stateless_chat(self, message: str, context: dict | None = None) -> dict:
    """Process a stateless chat message asynchronously.

    This is used for rules lookups and general questions that don't need
    campaign context.

    Args:
        message: The user's message
        context: Optional context (combat state, token info, etc.)

    Returns:
        dict with 'response' key containing the assistant's response
    """
    try:
        # Format context into message if provided
        if context:
            context_str = _format_context(context)
            if context_str:
                message = f"{context_str}\n\nQuestion: {message}"

        agent = ChatAgent()
        try:
            response = agent.chat(message)
            return {"response": response, "success": True}
        finally:
            agent.close()

    except Exception as exc:
        logger.exception(f"Error processing stateless chat: {exc}")
        # Retry on failure
        raise self.retry(exc=exc, countdown=5)


@celery_app.task(bind=True, max_retries=2, soft_time_limit=180)
def process_campaign_chat(self, campaign_id: str, player_input: str) -> dict:
    """Process a campaign-aware chat message asynchronously.

    This creates a temporary GMAgent to process the turn, then closes it.
    For ongoing sessions with WebSocket integration, use the synchronous
    API endpoint instead.

    Args:
        campaign_id: The campaign ID
        player_input: The player's input

    Returns:
        dict with 'response' and 'turn_number' keys
    """
    from gm_agent.agent import GMAgent

    try:
        # Verify campaign exists
        campaign = campaign_store.get(campaign_id)
        if not campaign:
            return {"success": False, "error": f"Campaign '{campaign_id}' not found"}

        # Check for active session
        session = session_store.get_current(campaign_id)
        if not session:
            return {
                "success": False,
                "error": "No active session. Start one first.",
            }

        # Create temporary agent and process turn
        agent = GMAgent(campaign_id)
        try:
            response = agent.process_turn(player_input)
            # Get updated session state
            session = session_store.get_current(campaign_id)
            return {
                "success": True,
                "response": response,
                "turn_number": len(session.turns),
            }
        finally:
            agent.close()

    except Exception as exc:
        logger.exception(f"Error processing campaign chat: {exc}")
        raise self.retry(exc=exc, countdown=5)


# =============================================================================
# Summary Tasks
# =============================================================================


@celery_app.task(bind=True, max_retries=1, soft_time_limit=300)
def generate_session_summary(self, campaign_id: str, session_id: str) -> dict:
    """Generate a summary for a completed session.

    This is a background task that can take several minutes for long sessions.

    Args:
        campaign_id: The campaign ID
        session_id: The session ID to summarize

    Returns:
        dict with 'summary' key containing the generated summary
    """
    try:
        session = session_store.get(campaign_id, session_id)
        if not session:
            return {"success": False, "error": f"Session '{session_id}' not found"}

        if not session.turns:
            return {"success": False, "error": "Session has no turns to summarize"}

        summarizer = RollingSummarizer()
        summary = summarizer.generate_summary(session)

        # Update session with summary
        session.summary = summary
        session_store.update(session)

        return {"success": True, "summary": summary}

    except Exception as exc:
        logger.exception(f"Error generating session summary: {exc}")
        raise self.retry(exc=exc, countdown=10)


# =============================================================================
# State Checkpointing Tasks
# =============================================================================


@celery_app.task(ignore_result=True)
def checkpoint_state() -> None:
    """Periodic task to checkpoint active game loop state to Redis.

    This task is scheduled via celery beat to run every 30 seconds.
    The API worker calls register_active_loop() when loops start/stop,
    and this task persists that state to Redis for recovery.

    Note: This task runs in the Celery worker, not the API worker.
    The API worker must publish state updates to Redis for this to work.
    """
    try:
        client = get_redis_client()

        # Check if there's pending state to checkpoint
        pending_state = client.get("gm_agent:pending_checkpoint")
        if pending_state:
            # Move pending state to checkpointed state
            client.set("gm_agent:checkpointed_state", pending_state)
            client.delete("gm_agent:pending_checkpoint")
            logger.debug("State checkpointed to Redis")

    except Exception as exc:
        logger.warning(f"Failed to checkpoint state: {exc}")


@celery_app.task(ignore_result=True)
def clear_checkpoint() -> None:
    """Clear checkpointed state (called on clean shutdown)."""
    try:
        client = get_redis_client()
        client.delete("gm_agent:checkpointed_state")
        client.delete("gm_agent:pending_checkpoint")
        logger.info("Cleared checkpointed state")
    except Exception as exc:
        logger.warning(f"Failed to clear checkpoint: {exc}")


# =============================================================================
# Utility Functions
# =============================================================================


def _format_context(context: dict) -> str:
    """Format Foundry context into a string for the agent."""
    parts = []

    # Combat context
    combat = context.get("combat")
    if combat:
        parts.append("Current Combat State:")
        if combat.get("round"):
            parts.append(f"  Round: {combat['round']}")
        if combat.get("combatants"):
            parts.append("  Combatants:")
            for c in combat["combatants"]:
                name = c.get("name", "Unknown")
                hp = c.get("hp", {})
                hp_str = f" (HP: {hp.get('value', '?')}/{hp.get('max', '?')})" if hp else ""
                conditions = c.get("conditions", [])
                cond_str = f" [{', '.join(conditions)}]" if conditions else ""
                parts.append(f"    - {name}{hp_str}{cond_str}")

    # Token context
    tokens = context.get("tokens")
    if tokens:
        parts.append("Selected Tokens:")
        for token in tokens:
            name = token.get("name", "Unknown")
            actor_type = token.get("type", "")
            level = token.get("level")
            level_str = f" (Level {level})" if level else ""
            parts.append(f"  - {name}{level_str} [{actor_type}]")

    return "\n".join(parts)


# =============================================================================
# State Publishing (called from API worker)
# =============================================================================


def publish_loop_state(active_loops: dict[str, Any]) -> None:
    """Publish active game loop state for checkpointing.

    This should be called from the API worker whenever game loop state changes.
    The checkpoint_state task will pick this up and persist it.

    Args:
        active_loops: Dict mapping campaign_id to loop info:
            {
                "campaign_id": {
                    "enabled": bool,
                    "started_at": float,
                    "stats": dict
                }
            }
    """
    try:
        client = get_redis_client()
        state = {
            "timestamp": __import__("time").time(),
            "loops": {
                cid: {
                    "enabled": info.get("enabled", False),
                    "started_at": info.get("started_at"),
                }
                for cid, info in active_loops.items()
            },
        }
        client.set("gm_agent:pending_checkpoint", json.dumps(state))
    except Exception as exc:
        logger.warning(f"Failed to publish loop state: {exc}")


def get_checkpointed_state() -> dict | None:
    """Retrieve checkpointed state for recovery.

    Returns:
        Checkpointed state dict or None if no checkpoint exists
    """
    try:
        client = get_redis_client()
        state = client.get("gm_agent:checkpointed_state")
        if state:
            return json.loads(state)
        return None
    except Exception as exc:
        logger.warning(f"Failed to get checkpointed state: {exc}")
        return None


# =============================================================================
# Automation Tasks (Phase 4.2)
# =============================================================================


def _acquire_campaign_lock(campaign_id: str, timeout: int = 120) -> redis.lock.Lock:
    """Acquire campaign-level lock for automation tasks.

    Args:
        campaign_id: Campaign to lock
        timeout: Lock timeout in seconds

    Returns:
        Redis lock object (must be released by caller)
    """
    client = get_redis_client()
    lock_key = f"automation:lock:{campaign_id}"
    return client.lock(lock_key, timeout=timeout, blocking_timeout=5)


def _publish_task_progress(task_id: str, state: str, info: dict | None = None) -> None:
    """Publish task progress to Redis for polling.

    Args:
        task_id: Celery task ID
        state: Task state (PENDING, PROCESSING, SUCCESS, FAILURE)
        info: Additional info (result, error, etc.)
    """
    try:
        client = get_redis_client()
        progress_key = f"task:progress:{task_id}"
        data = {
            "state": state,
            "timestamp": __import__("time").time(),
        }
        if info:
            data.update(info)
        client.setex(progress_key, 600, json.dumps(data))  # 10 min TTL
    except Exception as exc:
        logger.warning(f"Failed to publish task progress: {exc}")


@celery_app.task(bind=True, max_retries=2, soft_time_limit=120, queue="automation")
def process_player_chat_async(
    self,
    campaign_id: str,
    player_input: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Process player chat asynchronously with campaign locking.

    This task acquires a campaign-level lock to ensure only one automation
    task runs per campaign at a time. This prevents race conditions on
    file-based storage and Redis state.

    Args:
        campaign_id: The campaign ID
        player_input: The player's input
        session_id: Optional session ID (uses current if None)
        metadata: Optional turn metadata

    Returns:
        dict with 'success', 'response', and 'turn_number' keys
    """
    from gm_agent.agent import GMAgent
    from gm_agent.storage.schemas import TurnMetadata

    task_id = self.request.id

    try:
        # Publish initial state
        _publish_task_progress(task_id, "PENDING", {"campaign_id": campaign_id})

        # Acquire campaign lock
        lock = _acquire_campaign_lock(campaign_id)
        if not lock.acquire(blocking=True):
            # Lock held by another task - retry
            logger.info(f"Campaign {campaign_id} locked, retrying task {task_id}")
            raise self.retry(countdown=2)

        try:
            _publish_task_progress(task_id, "PROCESSING", {"campaign_id": campaign_id})

            # Verify campaign exists
            campaign = campaign_store.get(campaign_id)
            if not campaign:
                result = {"success": False, "error": f"Campaign '{campaign_id}' not found"}
                _publish_task_progress(task_id, "FAILURE", result)
                return result

            # Check for active session
            session = session_store.get_current(campaign_id)
            if not session:
                result = {
                    "success": False,
                    "error": "No active session. Start one first.",
                }
                _publish_task_progress(task_id, "FAILURE", result)
                return result

            # Create metadata if provided
            turn_metadata = None
            if metadata:
                turn_metadata = TurnMetadata(**metadata)

            # Create agent and process turn
            agent = GMAgent(campaign_id)
            try:
                response = agent.process_turn(player_input, metadata=turn_metadata)

                # Get updated session state
                session = session_store.get_current(campaign_id)
                result = {
                    "success": True,
                    "response": response,
                    "turn_number": len(session.turns),
                    "campaign_id": campaign_id,
                }
                _publish_task_progress(task_id, "SUCCESS", result)
                return result

            finally:
                agent.close()

        finally:
            # Always release lock
            try:
                lock.release()
            except Exception:
                pass  # Lock may have expired

    except Exception as exc:
        logger.exception(f"Error processing player chat async: {exc}")
        result = {"success": False, "error": str(exc)}
        _publish_task_progress(task_id, "FAILURE", result)
        raise self.retry(exc=exc, countdown=5)


@celery_app.task(bind=True, max_retries=2, soft_time_limit=120, queue="automation")
def process_npc_turn_async(
    self,
    campaign_id: str,
    npc_name: str,
    prompt: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Process NPC turn asynchronously with campaign locking.

    Args:
        campaign_id: The campaign ID
        npc_name: Name of the NPC
        prompt: Optional custom prompt (uses default if None)
        metadata: Optional turn metadata

    Returns:
        dict with 'success' and 'response' keys
    """
    from gm_agent.agent import GMAgent
    from gm_agent.storage.schemas import TurnMetadata

    task_id = self.request.id

    try:
        # Publish initial state
        _publish_task_progress(
            task_id, "PENDING", {"campaign_id": campaign_id, "npc_name": npc_name}
        )

        # Acquire campaign lock
        lock = _acquire_campaign_lock(campaign_id)
        if not lock.acquire(blocking=True):
            logger.info(f"Campaign {campaign_id} locked, retrying task {task_id}")
            raise self.retry(countdown=2)

        try:
            _publish_task_progress(
                task_id, "PROCESSING", {"campaign_id": campaign_id, "npc_name": npc_name}
            )

            # Verify campaign exists
            campaign = campaign_store.get(campaign_id)
            if not campaign:
                result = {"success": False, "error": f"Campaign '{campaign_id}' not found"}
                _publish_task_progress(task_id, "FAILURE", result)
                return result

            # Check for active session
            session = session_store.get_current(campaign_id)
            if not session:
                result = {
                    "success": False,
                    "error": "No active session. Start one first.",
                }
                _publish_task_progress(task_id, "FAILURE", result)
                return result

            # Build prompt
            if prompt is None:
                from gm_agent.context import render_prompt_template

                template = (
                    "It's {actor_name}'s turn in combat. "
                    "Decide their action based on the current combat state and their "
                    "character profile. Narrate their action dramatically. "
                    "Use the appropriate tools to get combat state and execute actions."
                )
                prompt = render_prompt_template(template, {"actor_name": npc_name})

            # Create metadata
            turn_metadata = TurnMetadata(
                source="automation_async",
                event_type="combatTurn",
                actor_name=npc_name,
            )
            if metadata:
                for key, value in metadata.items():
                    setattr(turn_metadata, key, value)

            # Create agent and process turn
            agent = GMAgent(campaign_id)
            try:
                response = agent.process_turn(prompt, metadata=turn_metadata)

                result = {
                    "success": True,
                    "response": response,
                    "npc_name": npc_name,
                    "campaign_id": campaign_id,
                }
                _publish_task_progress(task_id, "SUCCESS", result)
                return result

            finally:
                agent.close()

        finally:
            # Always release lock
            try:
                lock.release()
            except Exception:
                pass

    except Exception as exc:
        logger.exception(f"Error processing NPC turn async: {exc}")
        result = {"success": False, "error": str(exc)}
        _publish_task_progress(task_id, "FAILURE", result)
        raise self.retry(exc=exc, countdown=5)


def get_task_progress(task_id: str) -> dict | None:
    """Get progress info for an async automation task.

    Args:
        task_id: Celery task ID

    Returns:
        Progress dict or None if not found
    """
    try:
        client = get_redis_client()
        progress_key = f"task:progress:{task_id}"
        data = client.get(progress_key)
        if data:
            return json.loads(data)
        return None
    except Exception as exc:
        logger.warning(f"Failed to get task progress: {exc}")
        return None
