"""Polling-based client for remote Foundry VTT communication.

This module implements the FoundryBridgeBase interface using HTTP polling,
allowing gm-agent to communicate with remote Foundry VTT instances where
WebSocket connectivity is not possible (firewalls, NAT, reverse proxies).

In this mode, gm-agent acts as the client, polling Foundry for events
and sending commands via REST API calls.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

from .foundry_bridge_base import FoundryBridgeBase

logger = logging.getLogger(__name__)


@dataclass
class PollingConfig:
    """Configuration for polling-based Foundry communication.

    Attributes:
        base_url: Base URL of Foundry VTT (e.g., "https://foundry.example.com")
        api_key: API key for authentication
        campaign_id: Campaign identifier for this connection
        poll_interval: Seconds between event polls (default: 2.0)
        long_poll_timeout: Timeout for long-polling requests (default: 25.0)
        request_timeout: HTTP request timeout (default: 30.0)
        verify_ssl: Whether to verify SSL certificates (default: True)
        max_retries: Maximum connection retries before giving up (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
    """

    base_url: str
    api_key: str
    campaign_id: str
    poll_interval: float = 2.0
    long_poll_timeout: float = 25.0
    request_timeout: float = 30.0
    verify_ssl: bool = True
    max_retries: int = 5
    retry_delay: float = 1.0


@dataclass
class PendingRequest:
    """Tracks a pending command request awaiting response."""

    event: threading.Event
    response: dict | None = None
    error: str | None = None


class FoundryPollingClient(FoundryBridgeBase):
    """Polling-based client for remote Foundry VTT communication.

    This client polls Foundry's REST API for events and sends commands
    via HTTP POST requests. It's designed for scenarios where:
    - Foundry is hosted on cloud services (Forge, etc.)
    - Firewall/NAT prevents WebSocket connections to gm-agent
    - Reverse proxy doesn't support WebSocket upgrades

    The polling client implements the same interface as FoundryBridge,
    allowing seamless switching between communication modes.
    """

    def __init__(self, config: PollingConfig):
        """Initialize the polling client.

        Args:
            config: Polling configuration
        """
        self.config = config
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._event_handlers: dict[str, list[Callable[[dict], None]]] = {}
        self._last_event_id: str | None = None
        self._connected = False
        self._lock = threading.Lock()

        # Pending command requests
        self._pending_requests: dict[str, PendingRequest] = {}

        # Cache for graceful fallback
        self._cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = 60.0  # Cache TTL in seconds

        # HTTP session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "X-Campaign-Id": config.campaign_id,
                "Content-Type": "application/json",
            }
        )

        # Connection retry state
        self._consecutive_failures = 0

    def start(self) -> None:
        """Start the polling loop.

        Begins background polling for events from Foundry.
        Safe to call multiple times - will not start multiple threads.
        """
        if self._running:
            logger.warning("Polling client already running")
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="foundry-polling", daemon=True
        )
        self._poll_thread.start()
        logger.info(
            f"Started polling client for {self.config.base_url} "
            f"(interval: {self.config.poll_interval}s)"
        )

    def stop(self) -> None:
        """Stop the polling loop.

        Signals the polling thread to stop and waits for it to finish.
        """
        if not self._running:
            return

        self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5.0)
        self._poll_thread = None

        with self._lock:
            self._connected = False
            # Fail any pending requests
            for request_id, request in self._pending_requests.items():
                request.error = "Polling client stopped"
                request.event.set()

        logger.info("Stopped polling client")

    def is_connected(self) -> bool:
        """Check if connection to Foundry is active."""
        with self._lock:
            return self._connected

    def send_command(
        self, command: str, data: dict | None = None, timeout: float = 5.0
    ) -> dict:
        """Send a command to Foundry and wait for response.

        Args:
            command: The command name (e.g., "getScene", "updateToken")
            data: Optional data payload
            timeout: Timeout in seconds

        Returns:
            Response data from Foundry

        Raises:
            ConnectionError: If Foundry is not connected
            TimeoutError: If response not received in time
            RuntimeError: If Foundry returns an error
        """
        if not self.is_connected():
            raise ConnectionError("Foundry VTT is not connected")

        request_id = str(uuid.uuid4())
        event = threading.Event()

        with self._lock:
            self._pending_requests[request_id] = PendingRequest(event=event)

        try:
            # Send command via HTTP POST
            url = f"{self.config.base_url}/api/gm-agent/command"
            payload = {
                "requestId": request_id,
                "command": command,
                "data": data or {},
            }

            response = self._session.post(
                url,
                json=payload,
                timeout=self.config.request_timeout,
                verify=self.config.verify_ssl,
            )

            if response.status_code == 401:
                raise ConnectionError("Authentication failed - check API key")
            elif response.status_code == 404:
                raise ConnectionError(
                    "Foundry polling API not found - ensure module is configured"
                )
            elif not response.ok:
                raise RuntimeError(
                    f"Command failed with status {response.status_code}: {response.text}"
                )

            result = response.json()

            if result.get("success"):
                return result.get("data", {})
            else:
                raise RuntimeError(result.get("error", "Unknown error"))

        except requests.Timeout:
            raise TimeoutError(f"Timeout waiting for Foundry response to {command}")
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to send command: {e}")
        finally:
            with self._lock:
                self._pending_requests.pop(request_id, None)

    def on_event(self, event_type: str, handler: Callable[[dict], None]) -> None:
        """Register a handler for Foundry events.

        Args:
            event_type: Event type (e.g., "combatStart", "playerChat")
            handler: Callback function receiving event payload
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_cached(self, key: str) -> Any | None:
        """Get cached data if still valid."""
        with self._lock:
            if key not in self._cache:
                return None
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp > self._cache_ttl:
                return None
            return self._cache[key]

    def set_cached(self, key: str, value: Any) -> None:
        """Set cached data."""
        with self._lock:
            self._cache[key] = value
            self._cache_timestamps[key] = time.time()

    def _poll_loop(self) -> None:
        """Background thread that polls Foundry for events."""
        while self._running:
            try:
                self._poll_events()
                self._consecutive_failures = 0
                with self._lock:
                    if not self._connected:
                        self._connected = True
                        logger.info("Connected to Foundry VTT via polling")
            except requests.Timeout:
                # Long-poll timeout is normal - just continue
                pass
            except requests.RequestException as e:
                self._handle_poll_error(e)
            except Exception as e:
                logger.exception(f"Unexpected error in poll loop: {e}")
                self._handle_poll_error(e)

            # Sleep between polls (unless we're using long-polling timeout)
            if self._running:
                time.sleep(self.config.poll_interval)

    def _poll_events(self) -> None:
        """Poll Foundry for new events."""
        url = f"{self.config.base_url}/api/gm-agent/events"
        params = {
            "timeout": int(self.config.long_poll_timeout),
        }
        if self._last_event_id:
            params["since"] = self._last_event_id

        response = self._session.get(
            url,
            params=params,
            timeout=self.config.request_timeout + self.config.long_poll_timeout,
            verify=self.config.verify_ssl,
        )

        if response.status_code == 401:
            raise ConnectionError("Authentication failed")
        elif not response.ok:
            raise RuntimeError(f"Poll failed: {response.status_code}")

        data = response.json()
        events = data.get("events", [])
        last_event_id = data.get("lastEventId")

        if last_event_id:
            self._last_event_id = last_event_id

        # Process events
        for event in events:
            self._handle_event(event)

    def _handle_event(self, event: dict) -> None:
        """Handle an event received from Foundry.

        Args:
            event: Event dict with type, timestamp, and payload
        """
        event_type = event.get("type")
        payload = event.get("payload", {})

        if not event_type:
            return

        # Update cache based on event type
        self._update_cache_from_event(event_type, payload)

        # Notify registered handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                logger.exception(f"Error in event handler for {event_type}: {e}")

    def _update_cache_from_event(self, event_type: str, payload: dict) -> None:
        """Update cache based on Foundry events."""
        if event_type == "combatUpdate":
            self.set_cached("combat_state", payload)
        elif event_type == "sceneChange":
            self.set_cached("scene", payload)
        elif event_type == "combatEnd":
            with self._lock:
                self._cache.pop("combat_state", None)

    def _handle_poll_error(self, error: Exception) -> None:
        """Handle polling errors with exponential backoff."""
        self._consecutive_failures += 1

        with self._lock:
            if self._connected:
                self._connected = False
                logger.warning(f"Lost connection to Foundry: {error}")

        if self._consecutive_failures >= self.config.max_retries:
            logger.error(
                f"Failed to connect to Foundry after {self._consecutive_failures} attempts"
            )
            # Reset counter to allow future retries
            self._consecutive_failures = 0
            # Longer backoff after max retries
            time.sleep(self.config.retry_delay * 10)
        else:
            # Exponential backoff
            backoff = self.config.retry_delay * (2 ** (self._consecutive_failures - 1))
            logger.debug(f"Retrying in {backoff:.1f}s (attempt {self._consecutive_failures})")
            time.sleep(backoff)

    def check_connection(self) -> bool:
        """Check connection to Foundry via status endpoint.

        Returns:
            True if Foundry is reachable and API is enabled
        """
        try:
            url = f"{self.config.base_url}/api/gm-agent/status"
            response = self._session.get(
                url,
                timeout=5.0,
                verify=self.config.verify_ssl,
            )
            return response.ok
        except requests.RequestException:
            return False

    def close(self) -> None:
        """Close the client and clean up resources."""
        self.stop()
        self._session.close()
