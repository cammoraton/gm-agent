"""Abstract base class for Foundry VTT communication bridges.

This module defines the interface that all Foundry communication bridges must implement,
allowing for both WebSocket-based (local) and polling-based (remote) communication modes.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class FoundryBridgeBase(ABC):
    """Abstract base class for Foundry VTT communication bridges.

    This interface is implemented by:
    - FoundryBridge: WebSocket-based communication (Foundry connects to gm-agent)
    - FoundryPollingClient: HTTP polling communication (gm-agent polls Foundry)

    Both implementations provide the same interface for sending commands and
    receiving events from Foundry VTT.
    """

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection to Foundry is active.

        Returns:
            True if connected and ready for communication
        """
        ...

    @abstractmethod
    def send_command(
        self, command: str, data: dict | None = None, timeout: float = 5.0
    ) -> dict:
        """Send a command to Foundry and wait for response.

        Args:
            command: The command name (e.g., "getScene", "updateToken")
            data: Optional data payload for the command
            timeout: Timeout in seconds to wait for response

        Returns:
            Response data from Foundry

        Raises:
            ConnectionError: If Foundry is not connected
            TimeoutError: If response not received within timeout
            RuntimeError: If Foundry returns an error
        """
        ...

    @abstractmethod
    def on_event(self, event_type: str, handler: Callable[[dict], None]) -> None:
        """Register a handler for Foundry events.

        Args:
            event_type: Event type to handle (e.g., "combatStart", "playerChat")
            handler: Callback function receiving event payload dict
        """
        ...

    @abstractmethod
    def get_cached(self, key: str) -> Any | None:
        """Get cached data if still valid.

        Cache is used for graceful fallback when connection is temporarily unavailable.

        Args:
            key: Cache key (e.g., "combat_state", "scene")

        Returns:
            Cached value if valid, None if expired or not found
        """
        ...

    @abstractmethod
    def set_cached(self, key: str, value: Any) -> None:
        """Set cached data.

        Args:
            key: Cache key
            value: Value to cache
        """
        ...
