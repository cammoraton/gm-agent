"""PF2e storage — canonical location.

Imports are lazy to avoid circular dependencies with gm_agent.storage.
Use direct imports from submodules.
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular import."""
    if name in ("APProgressStore", "APProgressEntry"):
        from . import ap_progress
        return getattr(ap_progress, name)
    if name in ("TreasureStore", "TreasureEntry", "TREASURE_BY_LEVEL"):
        from . import treasure
        return getattr(treasure, name)
    if name in ("SubsystemStore", "SubsystemInstance"):
        from . import subsystems
        return getattr(subsystems, name)
    if name in ("AnnotationStore", "AnnotationEntry", "VALID_ISSUE_TYPES", "VALID_SEVERITIES"):
        from . import annotations
        return getattr(annotations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
