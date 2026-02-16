"""JSON-on-disk persistence for encounter subsystems.

Stores subsystem instances (VP tracking, chases, hazards, influence,
infiltration) as individual JSON files under the campaign directory.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import CAMPAIGNS_DIR


class SubsystemInstance(BaseModel):
    """A running encounter subsystem instance."""

    id: str
    campaign_id: str
    subsystem_type: str  # "vp", "influence", "research", "chase", "infiltration", "hazard"
    name: str
    status: str = "active"  # "active", "completed", "failed", "abandoned"

    # Universal tracking
    round_number: int = 0
    victory_points: dict[str, int] = Field(default_factory=dict)  # target_name -> VP
    thresholds: dict[str, dict] = Field(default_factory=dict)  # target_name -> {minor: N, major: N}

    # Chase-specific
    positions: dict[str, int] = Field(default_factory=dict)  # participant -> position
    obstacles: list[dict] = Field(default_factory=list)
    chase_length: int = 0

    # Hazard-specific
    hp: int | None = None
    max_hp: int | None = None
    hardness: int = 0
    routine_actions: list[str] = Field(default_factory=list)
    routine_index: int = 0
    disable_conditions: list[dict] = Field(default_factory=list)
    destroyed: bool = False
    disabled: bool = False

    # Infiltration-specific
    awareness_points: int = 0
    detection_threshold: int = 0

    # Audit
    action_log: list[dict] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SubsystemStore:
    """JSON-on-disk storage for subsystem instances.

    Directory layout:
        data/campaigns/{campaign_id}/subsystems/{subsystem_id}.json
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self._dir = self.base_dir / campaign_id / "subsystems"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, subsystem_id: str) -> Path:
        return self._dir / f"{subsystem_id}.json"

    def create(
        self,
        subsystem_type: str,
        name: str,
        config: dict[str, Any] | None = None,
    ) -> SubsystemInstance:
        """Create a new subsystem instance.

        Args:
            subsystem_type: Type of subsystem (vp, influence, research, chase, infiltration, hazard).
            name: Human-readable name for the subsystem.
            config: Type-specific configuration dict.

        Returns:
            The newly created SubsystemInstance.
        """
        config = config or {}
        instance_id = uuid.uuid4().hex[:12]

        instance = SubsystemInstance(
            id=instance_id,
            campaign_id=self.campaign_id,
            subsystem_type=subsystem_type,
            name=name,
            config=config,
        )

        # Apply type-specific initialization from config
        self._init_from_config(instance, config)

        self.save(instance)
        return instance

    def _init_from_config(self, instance: SubsystemInstance, config: dict[str, Any]) -> None:
        """Initialize subsystem fields from config based on type."""
        st = instance.subsystem_type

        if st in ("vp", "influence", "research"):
            # VP-based subsystems: targets with thresholds
            targets = config.get("targets", {})
            for target_name, target_cfg in targets.items():
                instance.victory_points[target_name] = 0
                instance.thresholds[target_name] = {
                    "minor": target_cfg.get("minor", 3),
                    "major": target_cfg.get("major", 6),
                }

        elif st == "chase":
            participants = config.get("participants", [])
            chase_length = config.get("chase_length", 10)
            instance.chase_length = chase_length
            for p in participants:
                instance.positions[p] = config.get("start_position", 0)
            instance.obstacles = config.get("obstacles", [])

        elif st == "hazard":
            instance.hp = config.get("hp")
            instance.max_hp = config.get("hp")
            instance.hardness = config.get("hardness", 0)
            instance.routine_actions = config.get("routine_actions", [])
            instance.disable_conditions = config.get("disable_conditions", [])

        elif st == "infiltration":
            instance.detection_threshold = config.get("detection_threshold", 10)
            # VP tracking for infiltration objectives
            targets = config.get("targets", {})
            for target_name, target_cfg in targets.items():
                instance.victory_points[target_name] = 0
                instance.thresholds[target_name] = {
                    "minor": target_cfg.get("minor", 3),
                    "major": target_cfg.get("major", 6),
                }

        elif st == "exploration":
            # Store activities and marching order in config dict
            instance.config["activities"] = config.get("activities", {})
            instance.config["marching_order"] = config.get("marching_order", [])

    def get(self, subsystem_id: str) -> SubsystemInstance | None:
        """Load a subsystem instance by ID."""
        path = self._file_path(subsystem_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return SubsystemInstance.model_validate(data)

    def save(self, instance: SubsystemInstance) -> None:
        """Persist a subsystem instance to disk."""
        instance.updated_at = datetime.now()
        with open(self._file_path(instance.id), "w") as f:
            json.dump(instance.model_dump(mode="json"), f, indent=2, default=str)

    def list_active(self) -> list[SubsystemInstance]:
        """List all active subsystem instances for the campaign."""
        instances = []
        for path in self._dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            inst = SubsystemInstance.model_validate(data)
            if inst.status == "active":
                instances.append(inst)
        return instances

    def close(self) -> None:
        """No-op for JSON store (no persistent connections)."""
        pass
