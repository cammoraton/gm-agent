"""Game system plugin architecture.

Provides the GameSystem ABC and a registry for pluggable game systems.
Each system declares its MCP servers, system prompt, stores, and optional
prep pipeline. The registry enables system-aware initialization of the
MCPClient and context builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..mcp.base import MCPServer
    from ..prep.pipeline import PrepPipeline


class GameSystem(ABC):
    """Abstract base class for pluggable game systems.

    Each system (PF2e, Microscope, Ex Novo, Delve, Ex Umbra, etc.)
    implements this interface to declare its capabilities.
    """

    name: str  # "pf2e", "microscope", "ex_novo", "delve", "ex_umbra"
    display_name: str  # "Pathfinder 2e (Remaster)", "Microscope"
    category: str  # "rpg" | "generation"

    @abstractmethod
    def servers(self, context: dict[str, Any]) -> list[MCPServer]:
        """Create and return MCP server instances for this system.

        Args:
            context: Dict with campaign_id, llm, campaign_books, etc.

        Returns:
            List of MCP server instances.
        """
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this game system."""
        ...

    def campaign_plugins(self, campaign_id: str, **kwargs: Any) -> list:
        """Return SystemToolPlugin instances for CampaignStateServer.

        These plugins inject system-specific tools into the shared
        CampaignStateServer. Override in subclasses that need campaign tools.
        """
        return []

    def stores(self, campaign_id: str) -> dict[str, Any]:
        """Return system-specific stores keyed by name.

        Override in subclasses that need persistent stores.
        """
        return {}

    def prep_pipeline(self) -> PrepPipeline | None:
        """Return a prep pipeline for campaign initialization, if any."""
        return None


# ---------------------------------------------------------------------------
# System Registry
# ---------------------------------------------------------------------------

SYSTEM_REGISTRY: dict[str, type[GameSystem]] = {}


def register_system(cls: type[GameSystem]) -> type[GameSystem]:
    """Class decorator to register a GameSystem subclass.

    Usage::

        @register_system
        class PF2eSystem(GameSystem):
            name = "pf2e"
            ...
    """
    SYSTEM_REGISTRY[cls.name] = cls
    return cls


def get_system(name: str) -> GameSystem:
    """Instantiate and return a GameSystem by name.

    Args:
        name: System identifier (e.g. "pf2e", "microscope").

    Raises:
        KeyError: If the system is not registered.
    """
    _discover_systems()
    if name not in SYSTEM_REGISTRY:
        raise KeyError(
            f"Unknown game system '{name}'. "
            f"Registered: {', '.join(SYSTEM_REGISTRY.keys()) or '(none)'}"
        )
    return SYSTEM_REGISTRY[name]()


def list_systems() -> list[dict[str, str]]:
    """List all registered game systems.

    Returns:
        List of dicts with name, display_name, category.
    """
    _discover_systems()
    result = []
    for cls in SYSTEM_REGISTRY.values():
        result.append({
            "name": cls.name,
            "display_name": cls.display_name,
            "category": cls.category,
        })
    return result


# ---------------------------------------------------------------------------
# Auto-discovery of built-in systems
# ---------------------------------------------------------------------------

_discovered = False


def _discover_systems() -> None:
    """Import all built-in system modules to trigger @register_system."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    # Import each built-in system package — the @register_system decorator
    # in each __init__.py adds it to SYSTEM_REGISTRY on import.
    import importlib
    for module_name in (
        "gm_agent.systems.pf2e",
        "gm_agent.systems.microscope",
        "gm_agent.systems.ex_novo",
        "gm_agent.systems.delve",
        "gm_agent.systems.ex_umbra",
    ):
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass
