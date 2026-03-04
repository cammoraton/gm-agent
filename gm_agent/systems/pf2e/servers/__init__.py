"""PF2e MCP servers — canonical location.

Imports are lazy to avoid circular dependencies with gm_agent.mcp.
Use direct imports from submodules (e.g. from .pf2e_rag import PF2eRAGServer).
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular import with gm_agent.mcp."""
    if name == "PF2eRAGServer":
        from .pf2e_rag import PF2eRAGServer
        return PF2eRAGServer
    if name == "EncounterServer":
        from .encounter import EncounterServer
        return EncounterServer
    if name == "CreatureModifierServer":
        from .creature_modifier import CreatureModifierServer
        return CreatureModifierServer
    if name == "SubsystemServer":
        from .subsystem import SubsystemServer
        return SubsystemServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
