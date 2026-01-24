"""Event-driven game loop for full automation mode."""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .context import render_prompt_template
from .event_queue import EventQueue, QueueWorker, QueuedEvent
from .models import LLMUnavailableError
from .storage.schemas import TurnMetadata

if TYPE_CHECKING:
    from .agent import GMAgent
    from .mcp.foundry_vtt import FoundryBridge


@dataclass
class PlayerMessageBatch:
    """Accumulated messages from a single player within a time window."""

    player_id: str | None
    actor_name: str
    messages: list[str] = field(default_factory=list)
    first_message_time: float = field(default_factory=time.time)


logger = logging.getLogger(__name__)


class GameLoopController:
    """Event-driven game loop for full automation mode.

    When enabled, this controller automatically responds to:
    - Player chat messages (exploration and roleplay)
    - NPC combat turns (with optional AI Combat Assistant coexistence)

    The controller subscribes to Foundry events via the bridge and
    uses the GMAgent to generate responses.
    """

    def __init__(
        self,
        campaign_id: str,
        agent: "GMAgent",
        bridge: "FoundryBridge",
        batch_window_seconds: float = 2.0,
        max_batch_size: int = 5,
        cooldown_seconds: float = 2.0,
        max_consecutive_errors: int = 5,
        dry_run: bool = False,
        enable_queue: bool = False,
        max_queue_depth: int = 100,
        max_exploration_depth: int = 20,
    ):
        """Initialize the game loop controller.

        Args:
            campaign_id: The campaign this controller is managing
            agent: GMAgent instance for processing turns
            bridge: FoundryBridge for communication with Foundry VTT
            batch_window_seconds: Time window to accumulate player messages before responding
            max_batch_size: Maximum messages per batch before immediate flush
            cooldown_seconds: Minimum time between NPC turn responses
            max_consecutive_errors: Number of consecutive errors before auto-disabling
            dry_run: If True, log responses instead of posting to Foundry
            enable_queue: If True, queue events for processing
            max_queue_depth: Maximum total queue size
            max_exploration_depth: Maximum exploration events (lower priority dropped)
        """
        self.campaign_id = campaign_id
        self.agent = agent
        self.bridge = bridge
        self.enabled = False
        self.dry_run = dry_run
        self._handlers_registered = False

        # Event queue
        self.enable_queue = enable_queue
        self._event_queue: EventQueue | None = None
        self._queue_worker: QueueWorker | None = None
        if enable_queue:
            self._event_queue = EventQueue(
                campaign_id=campaign_id,
                max_depth=max_queue_depth,
                max_exploration_depth=max_exploration_depth,
            )
            self._queue_worker = QueueWorker(
                queue=self._event_queue,
                process_func=self._process_queued_event,
            )

        # Player message batching
        self.batch_window_seconds = batch_window_seconds
        self.max_batch_size = max_batch_size
        self._player_batches: dict[str, PlayerMessageBatch] = {}
        self._player_batch_timers: dict[str, threading.Timer] = {}
        self._batch_lock = threading.RLock()  # Reentrant lock for nested calls

        # NPC turn rate limiting (separate from player batching)
        self.cooldown_seconds = cooldown_seconds
        self._npc_last_response_time: float = 0.0

        # Error threshold
        self.max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors: int = 0

        # Stats tracking
        self._response_count: int = 0
        self._player_chat_count: int = 0
        self._npc_turn_count: int = 0
        self._error_count: int = 0
        self._batched_message_count: int = 0
        self._total_processing_time_ms: float = 0.0
        self._last_response_time: float | None = None
        self._started_at: float | None = None
        self._queued_events_count: int = 0
        self._queue_full_count: int = 0

    def start(self) -> None:
        """Enable automation and register event handlers.

        Also resets the consecutive error counter to give automation
        a fresh start after being re-enabled.
        """
        if not self._handlers_registered:
            self.bridge.on_event("playerChat", self._handle_player_chat)
            self.bridge.on_event("combatTurn", self._handle_combat_turn)
            self._handlers_registered = True
            logger.info(f"Registered event handlers for campaign {self.campaign_id}")

        # Start queue worker if enabled
        if self._queue_worker:
            self._queue_worker.start()
            logger.info(f"Started queue worker for campaign {self.campaign_id}")

        self.enabled = True
        self._consecutive_errors = 0  # Fresh start on re-enable
        self._started_at = time.time()
        logger.info(f"Automation started for campaign {self.campaign_id}")

    def stop(self, flush_pending: bool = True) -> None:
        """Disable automation.

        Args:
            flush_pending: If True, process any pending message batches and queue before stopping.
                          If False, discard pending batches and queue.

        Note: Event handlers remain registered but will no-op when disabled.
        This allows for quick enable/disable without re-registering handlers.
        """
        # Stop queue worker if enabled
        if self._queue_worker:
            self._queue_worker.stop()
            logger.info(f"Stopped queue worker for campaign {self.campaign_id}")

        # Cancel all pending batch timers
        with self._batch_lock:
            for timer in self._player_batch_timers.values():
                timer.cancel()

            if flush_pending:
                # Flush any pending batches before stopping
                for player_key in list(self._player_batches.keys()):
                    self._flush_player_batch(player_key)
            else:
                # Discard pending batches
                self._player_batches.clear()

            self._player_batch_timers.clear()

        # Clear queue if not flushing
        if not flush_pending and self._event_queue:
            self._event_queue.clear()

        self.enabled = False
        logger.info(f"Automation stopped for campaign {self.campaign_id}")

    def set_bridge(self, bridge: "FoundryBridge") -> None:
        """Update the Foundry bridge reference.

        Used when Foundry reconnects and a new bridge is created.
        Event handlers will need to be re-registered on the new bridge.

        Args:
            bridge: The new FoundryBridge instance
        """
        self.bridge = bridge
        # Re-register handlers on the new bridge
        self._handlers_registered = False
        if self.enabled:
            # If we were running, re-register handlers immediately
            self.bridge.on_event("playerChat", self._handle_player_chat)
            self.bridge.on_event("combatTurn", self._handle_combat_turn)
            self._handlers_registered = True

    def _check_npc_rate_limit(self) -> bool:
        """Check if enough time has passed since the last NPC response.

        Returns:
            True if we can respond, False if still in cooldown
        """
        now = time.time()
        if now - self._npc_last_response_time < self.cooldown_seconds:
            logger.debug("Rate limited: NPC turn in cooldown")
            return False
        return True

    def _record_npc_response(self) -> None:
        """Record that an NPC response was sent (for rate limiting and stats)."""
        now = time.time()
        self._npc_last_response_time = now
        self._response_count += 1
        self._last_response_time = now
        self._npc_turn_count += 1

    def _record_player_response(self, message_count: int = 1) -> None:
        """Record that a player chat response was sent.

        Args:
            message_count: Number of messages in the batch that was processed
        """
        now = time.time()
        self._response_count += 1
        self._last_response_time = now
        self._player_chat_count += 1
        self._batched_message_count += message_count

    def get_stats(self) -> dict:
        """Get automation statistics.

        Returns:
            Dict with stats including response counts, timing, and batching info.
        """
        now = time.time()
        with self._batch_lock:
            pending_batches = len(self._player_batches)
            pending_messages = sum(len(batch.messages) for batch in self._player_batches.values())
            # Find oldest pending batch
            oldest_batch_age = None
            if self._player_batches:
                oldest_time = min(
                    batch.first_message_time for batch in self._player_batches.values()
                )
                oldest_batch_age = now - oldest_time

        # Calculate average batch size
        avg_batch_size = None
        if self._player_chat_count > 0:
            avg_batch_size = self._batched_message_count / self._player_chat_count

        stats = {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "started_at": self._started_at,
            "uptime_seconds": (now - self._started_at) if self._started_at else None,
            "response_count": self._response_count,
            "player_chat_count": self._player_chat_count,
            "npc_turn_count": self._npc_turn_count,
            "batched_message_count": self._batched_message_count,
            "average_batch_size": avg_batch_size,
            "pending_batches": pending_batches,
            "pending_messages": pending_messages,
            "oldest_batch_age_seconds": oldest_batch_age,
            "error_count": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "total_processing_time_ms": self._total_processing_time_ms,
            "last_response_time": self._last_response_time,
            "seconds_since_last_response": (
                (now - self._last_response_time) if self._last_response_time else None
            ),
            "batch_window_seconds": self.batch_window_seconds,
            "max_batch_size": self.max_batch_size,
            "cooldown_seconds": self.cooldown_seconds,
        }

        # Add queue stats if enabled
        if self._event_queue:
            stats["queue"] = self._event_queue.stats()
            stats["queued_events_count"] = self._queued_events_count
            stats["queue_full_count"] = self._queue_full_count

        return stats

    def reset_stats(self) -> None:
        """Reset all statistics counters.

        Useful for monitoring resets or testing. Does not affect
        automation state (enabled/disabled) or configuration.
        """
        self._response_count = 0
        self._player_chat_count = 0
        self._npc_turn_count = 0
        self._error_count = 0
        self._batched_message_count = 0
        self._total_processing_time_ms = 0.0
        self._last_response_time = None
        self._consecutive_errors = 0
        self._queued_events_count = 0
        self._queue_full_count = 0
        # Note: _started_at is not reset - tracks when automation was enabled

    def clear_queue(self) -> int:
        """Clear all events from the queue.

        Returns:
            Number of events removed
        """
        if not self._event_queue:
            return 0
        return self._event_queue.clear()

    def get_queue_stats(self) -> dict | None:
        """Get queue statistics.

        Returns:
            Queue stats dict or None if queue not enabled
        """
        if not self._event_queue:
            return None
        return self._event_queue.stats()

    def _record_error(self) -> None:
        """Record an error occurrence and check for auto-disable threshold.

        Increments both total error count and consecutive error count.
        If consecutive errors reach the threshold, automation is disabled
        and a notification is posted to chat.
        """
        self._error_count += 1
        self._consecutive_errors += 1

        if self._consecutive_errors >= self.max_consecutive_errors:
            logger.error(
                f"Automation disabled after {self._consecutive_errors} consecutive errors "
                f"for campaign {self.campaign_id}"
            )
            # Don't flush pending batches - the LLM is failing
            self.stop(flush_pending=False)
            self._post_response(
                f"*GM Agent automation has been disabled after {self._consecutive_errors} "
                f"consecutive errors. Please check the server logs and re-enable manually.*"
            )

    def _reset_consecutive_errors(self) -> None:
        """Reset consecutive error count after a successful response."""
        self._consecutive_errors = 0

    def _process_queued_event(self, event: QueuedEvent) -> None:
        """Process an event from the queue.

        Called by QueueWorker thread.

        Args:
            event: The event to process
        """
        if event.event_type == "playerChat":
            self._handle_player_chat_direct(event.payload)
        elif event.event_type == "combatTurn":
            self._handle_combat_turn_direct(event.payload)

    def _handle_player_chat(self, payload: dict) -> None:
        """Handle player chat event - either queue or process directly.

        Args:
            payload: Event payload containing:
                - actorName: Name of the player's character
                - content: The chat message content
                - playerId: The Foundry user ID
                - timestamp: When the message was sent
        """
        if not self.enabled:
            return

        # Queue event if queueing is enabled
        if self._event_queue:
            try:
                event_id = self._event_queue.push(
                    event_type="playerChat",
                    payload=payload,
                    priority=0,  # Exploration priority
                )
                self._queued_events_count += 1
                logger.debug(f"Queued playerChat event {event_id}")
            except ValueError as e:
                # Queue is full
                self._queue_full_count += 1
                logger.warning(f"Queue full, dropping playerChat: {e}")
        else:
            # Process directly if no queue
            self._handle_player_chat_direct(payload)

    def _handle_player_chat_direct(self, payload: dict) -> None:
        """Accumulate player chat messages for batched processing.

        Messages from the same player within the batch window are combined
        into a single prompt for more coherent responses.

        Args:
            payload: Event payload containing:
                - actorName: Name of the player's character
                - content: The chat message content
                - playerId: The Foundry user ID
                - timestamp: When the message was sent
        """
        if not self.enabled:
            return

        player_id = payload.get("playerId")
        actor_name = payload.get("actorName", "Unknown")
        content = payload.get("content", "")

        if not content:
            return

        # Use player_id as key, fall back to actor_name if no player_id
        player_key = player_id or actor_name

        with self._batch_lock:
            # Check if we have an existing batch for this player
            if player_key in self._player_batches:
                batch = self._player_batches[player_key]
                batch.messages.append(content)
                logger.debug(
                    f"Added message to batch for {actor_name} "
                    f"({len(batch.messages)}/{self.max_batch_size})"
                )

                # If we hit max batch size, flush immediately
                if len(batch.messages) >= self.max_batch_size:
                    # Cancel the timer since we're flushing now
                    if player_key in self._player_batch_timers:
                        self._player_batch_timers[player_key].cancel()
                        del self._player_batch_timers[player_key]
                    logger.debug(f"Max batch size reached for {actor_name}, flushing")
                    self._flush_player_batch(player_key)
            else:
                # Start a new batch
                self._player_batches[player_key] = PlayerMessageBatch(
                    player_id=player_id,
                    actor_name=actor_name,
                    messages=[content],
                )

                # If batch window is 0 or negative, flush immediately (synchronous mode)
                if self.batch_window_seconds <= 0:
                    logger.debug(f"Immediate flush for {actor_name} (no batching)")
                    self._flush_player_batch(player_key)
                else:
                    logger.debug(f"Started new batch for {actor_name}")
                    # Schedule a timer to flush after the window expires
                    timer = threading.Timer(
                        self.batch_window_seconds,
                        self._flush_player_batch_async,
                        args=[player_key],
                    )
                    timer.daemon = True
                    timer.start()
                    self._player_batch_timers[player_key] = timer

    def _flush_player_batch_async(self, player_key: str) -> None:
        """Timer callback to flush a player's message batch.

        This is called from a timer thread, so it acquires the lock.

        Args:
            player_key: The player identifier (player_id or actor_name)
        """
        with self._batch_lock:
            # Clean up the timer reference
            if player_key in self._player_batch_timers:
                del self._player_batch_timers[player_key]

            # Flush if batch still exists and controller is still enabled
            if player_key in self._player_batches and self.enabled:
                self._flush_player_batch(player_key)

    def _flush_player_batch(self, player_key: str) -> None:
        """Process and send response for accumulated player messages.

        Must be called while holding _batch_lock.

        Supports custom prompt template via campaign preferences:
        - "player_chat_prompt": Template with {actor_name} and {content} variables

        Args:
            player_key: The player identifier (player_id or actor_name)
        """
        if player_key not in self._player_batches:
            return

        batch = self._player_batches.pop(player_key)
        message_count = len(batch.messages)

        if not batch.messages:
            return

        # Build combined content from all messages
        if message_count == 1:
            content = batch.messages[0]
        else:
            # Multiple messages - join with newlines
            content = "\n".join(batch.messages)

        # Check for custom player chat prompt template
        custom_template = self._get_prompt_template("player_chat_prompt", None)

        if custom_template:
            # Use custom template
            prompt = render_prompt_template(
                custom_template, {"actor_name": batch.actor_name, "content": content}
            )
        else:
            # Use default format
            if message_count == 1:
                prompt = f"{batch.actor_name}: {content}"
            else:
                # Multiple messages - format each on its own line
                formatted_messages = [f"{batch.actor_name}: {msg}" for msg in batch.messages]
                prompt = "\n".join(formatted_messages)

        # Create metadata for session logging
        metadata = TurnMetadata(
            source="automation",
            event_type="playerChat",
            player_id=batch.player_id,
            actor_name=batch.actor_name,
        )

        try:
            logger.debug(f"Processing batch of {message_count} messages from {batch.actor_name}")
            start_time = time.time()
            response = self.agent.process_turn(prompt, metadata=metadata)
            processing_time_ms = (time.time() - start_time) * 1000
            self._total_processing_time_ms += processing_time_ms
            self._post_response(response)
            self._record_player_response(message_count)
            self._reset_consecutive_errors()
            logger.info(
                f"Responded to {message_count} batched message(s) from {batch.actor_name} "
                f"in {processing_time_ms:.0f}ms"
            )
        except LLMUnavailableError as e:
            logger.error(f"LLM unavailable during player chat handling: {e}")
            self._post_response(
                "*The GM Agent is temporarily unavailable (LLM backend offline). "
                "Please wait a moment and try again.*"
            )
            self._record_error()
            self._record_player_response(message_count)
        except Exception as e:
            logger.exception(f"Error handling player chat: {e}")
            self._post_response(f"*The GM Agent encountered an error: {e}*")
            self._record_error()
            self._record_player_response(message_count)

    def _handle_combat_turn(self, payload: dict) -> None:
        """Handle combat turn event - either queue or process directly.

        Args:
            payload: Event payload containing:
                - round: Current combat round
                - turn: Current turn index
                - combatant: Dict with name, isNPC, actorId flags
        """
        if not self.enabled:
            return

        # Queue event if queueing is enabled
        if self._event_queue:
            try:
                event_id = self._event_queue.push(
                    event_type="combatTurn",
                    payload=payload,
                    priority=1,  # Combat priority
                )
                self._queued_events_count += 1
                logger.debug(f"Queued combatTurn event {event_id}")
            except ValueError as e:
                # Queue is full
                self._queue_full_count += 1
                logger.warning(f"Queue full, dropping combatTurn: {e}")
        else:
            # Process directly if no queue
            self._handle_combat_turn_direct(payload)

    def _handle_combat_turn_direct(self, payload: dict) -> None:
        """Handle NPC turn in combat.

        Args:
            payload: Event payload containing:
                - round: Current combat round
                - turn: Current turn index
                - combatant: Dict with name, isNPC, actorId flags
        """
        if not self.enabled:
            return

        # Global NPC rate limiting - skip if too soon after last NPC response
        if not self._check_npc_rate_limit():
            return

        combatant = payload.get("combatant", {})
        if not combatant:
            return

        # Only handle NPC turns - player turns wait for input
        if not combatant.get("isNPC"):
            return

        npc_name = combatant.get("name", "Unknown NPC")
        actor_id = combatant.get("actorId")

        logger.debug(f"Handling combat turn for NPC: {npc_name}")

        # Check coordination mode for this specific NPC
        aca_status = self._get_aca_status_for_npc(actor_id, npc_name)

        if aca_status["controls_this_npc"]:
            # ACA is designated to control this NPC - just narrate
            logger.info(f"NPC {npc_name} controlled by ACA, providing narration only")
            self._narrate_npc_turn(npc_name, aca_status.get("turn_state"), actor_id=actor_id)
        else:
            # GM Agent has full control - decide and execute action
            logger.info(f"Running full NPC turn for {npc_name}")
            self._run_npc_turn(npc_name, actor_id=actor_id)

        self._record_npc_response()

    def _get_prompt_template(self, template_key: str, default: str) -> str:
        """Get a custom prompt template from campaign preferences.

        Args:
            template_key: The preference key for the template (e.g., "npc_combat_turn_prompt")
            default: Default template if not found in preferences

        Returns:
            The custom template if found, otherwise the default
        """
        if hasattr(self.agent, "campaign") and self.agent.campaign:
            prefs = self.agent.campaign.preferences or {}
            return prefs.get(template_key, default)
        return default

    def _post_response(self, content: str) -> None:
        """Post GM response to Foundry chat.

        In dry run mode, logs the response instead of posting to Foundry.

        Args:
            content: The response text to post
        """
        if not content:
            return

        if self.dry_run:
            logger.info(f"[DRY RUN] Would post to Foundry: {content[:200]}...")
            return

        try:
            self.bridge.send_command(
                "createChat",
                {
                    "content": content,
                    "speaker": "GM Agent",
                },
            )
        except Exception:
            # Silently fail if bridge is disconnected
            pass

    def _run_npc_turn(self, npc_name: str, actor_id: str | None = None) -> None:
        """Run a full NPC combat turn with decision-making and narration.

        The agent will decide the NPC's action based on combat state
        and character profile, then narrate the action dramatically.

        Supports custom prompt template via campaign preferences:
        - "npc_combat_turn_prompt": Template with {actor_name} variable

        Args:
            npc_name: Name of the NPC whose turn it is
            actor_id: Foundry actor ID for session logging
        """
        default_prompt = (
            "It's {actor_name}'s turn in combat. "
            "Decide their action based on the current combat state and their "
            "character profile. Narrate their action dramatically. "
            "Use the appropriate tools to get combat state and execute actions."
        )

        template = self._get_prompt_template("npc_combat_turn_prompt", default_prompt)
        prompt = render_prompt_template(template, {"actor_name": npc_name})

        # Create metadata for session logging
        metadata = TurnMetadata(
            source="automation",
            event_type="combatTurn",
            actor_name=npc_name,
            player_id=actor_id,  # Use actor_id in player_id field for NPCs
        )

        try:
            start_time = time.time()
            response = self.agent.process_turn(prompt, metadata=metadata)
            processing_time_ms = (time.time() - start_time) * 1000
            self._total_processing_time_ms += processing_time_ms
            self._post_response(response)
            self._reset_consecutive_errors()
            logger.info(f"NPC turn for {npc_name} completed in {processing_time_ms:.0f}ms")
        except LLMUnavailableError as e:
            logger.error(f"LLM unavailable during NPC turn: {e}")
            self._post_response(
                f"*{npc_name} pauses, lost in thought... "
                f"(GM Agent offline - please advance the turn manually)*"
            )
            self._record_error()
        except Exception as e:
            logger.exception(f"Error during NPC turn for {npc_name}: {e}")
            self._post_response(f"*{npc_name} hesitates, uncertain what to do. (Error: {e})*")
            self._record_error()

    def _narrate_npc_turn(
        self,
        npc_name: str,
        turn_state: dict | None = None,
        actor_id: str | None = None,
    ) -> None:
        """Narrate an NPC turn when AI Combat Assistant handles tactics.

        Provides brief narrative color for the NPC's action without
        controlling the actual mechanics (which ACA handles).

        Supports custom prompt template via campaign preferences:
        - "npc_narration_prompt": Template with {actor_name} and {context} variables

        Args:
            npc_name: Name of the NPC whose turn it is
            turn_state: Optional ACA turn state for context
            actor_id: Foundry actor ID for session logging
        """
        # Build context from turn state if available
        context_parts = []
        if turn_state:
            actions = turn_state.get("actionsRemaining", 3)
            history = turn_state.get("actionHistory", [])
            notes = turn_state.get("permanentNotes", "")

            if history:
                context_parts.append(f"Actions so far: {', '.join(history)}")
            if notes:
                context_parts.append(f"Tactical notes: {notes}")
            context_parts.append(f"Actions remaining: {actions}")

        context = " ".join(context_parts) if context_parts else ""

        default_prompt = (
            "{actor_name} is taking their turn in combat. "
            "The AI Combat Assistant is handling the tactical decisions. "
            "{context} "
            "Provide brief, evocative narrative color for their action "
            "without specifying exact mechanical details."
        )

        template = self._get_prompt_template("npc_narration_prompt", default_prompt)
        prompt = render_prompt_template(template, {"actor_name": npc_name, "context": context})

        # Create metadata for session logging (narration only mode)
        metadata = TurnMetadata(
            source="automation",
            event_type="combatTurn",
            actor_name=npc_name,
            player_id=actor_id,
        )

        try:
            start_time = time.time()
            response = self.agent.process_turn(prompt, metadata=metadata)
            processing_time_ms = (time.time() - start_time) * 1000
            self._total_processing_time_ms += processing_time_ms
            self._post_response(response)
            self._reset_consecutive_errors()
            logger.info(f"NPC narration for {npc_name} completed in {processing_time_ms:.0f}ms")
        except LLMUnavailableError as e:
            # ACA handles mechanics, so just provide minimal narrative
            logger.warning(f"LLM unavailable for NPC narration, using fallback: {e}")
            self._post_response(f"*{npc_name} acts swiftly...*")
            self._record_error()
        except Exception as e:
            logger.warning(f"Error during NPC narration for {npc_name}: {e}")
            self._post_response(f"*{npc_name} acts swiftly...*")
            self._record_error()

    def _get_aca_status_for_npc(self, actor_id: str | None, npc_name: str) -> dict:
        """Get AI Combat Assistant status for a specific NPC.

        Checks whether this NPC is designated for ACA control and
        retrieves their current turn state if available.

        Args:
            actor_id: The Foundry actor ID (optional)
            npc_name: Name of the NPC

        Returns:
            Dict with:
                - controls_this_npc: True if ACA controls this NPC
                - aca_active: True if ACA module is active
                - turn_state: Optional turn state dict
        """
        result = {
            "controls_this_npc": False,
            "aca_active": False,
            "turn_state": None,
        }

        try:
            # Get overall ACA state
            aca_state = self.bridge.send_command("getACAState", timeout=2.0)

            if not aca_state.get("active"):
                return result

            result["aca_active"] = True

            # Check if this NPC is designated for AI control
            designations = aca_state.get("designations", {})

            # Check by actor ID first, then by name
            is_ai_controlled = False
            if actor_id and designations.get(actor_id) == "ai":
                is_ai_controlled = True
            else:
                # Check by name (less reliable but fallback)
                for name_or_id, designation in designations.items():
                    if designation == "ai" and npc_name.lower() in name_or_id.lower():
                        is_ai_controlled = True
                        break

            result["controls_this_npc"] = is_ai_controlled

            # If ACA controls this NPC, get turn state for context
            if is_ai_controlled:
                try:
                    turn_state = self.bridge.send_command(
                        "getACATurnState",
                        {"actorName": npc_name},
                        timeout=2.0,
                    )
                    if not turn_state.get("error"):
                        result["turn_state"] = turn_state
                except Exception:
                    pass  # Turn state is optional context

        except Exception:
            # Default to GM Agent control if we can't check ACA
            pass

        return result

    def _is_ai_combat_assistant_active(self) -> bool:
        """Check if AI Combat Assistant module is globally active in Foundry.

        Note: For per-NPC control, use _get_aca_status_for_npc instead.

        Returns:
            True if AI Combat Assistant is active
        """
        try:
            result = self.bridge.send_command("getACAState", timeout=2.0)
            return result.get("active", False)
        except Exception:
            # Default to False if we can't check
            return False
