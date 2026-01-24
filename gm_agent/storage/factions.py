"""Faction and organization storage."""

import json
from pathlib import Path
from datetime import datetime

from .schemas import Faction
from ..config import CAMPAIGNS_DIR


class FactionStore:
    """Store for faction and organization management.

    Factions are stored as JSON files in:
        campaigns/{campaign_id}/factions/{faction_id}.json
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.factions_dir = self.base_dir / campaign_id / "factions"
        self.factions_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        description: str = "",
        goals: list[str] | None = None,
        resources: list[str] | None = None,
    ) -> Faction:
        """Create a new faction.

        Args:
            name: Faction name
            description: Faction description
            goals: Faction goals
            resources: Faction resources

        Returns:
            The created Faction
        """
        # Generate ID from name (lowercase, replace spaces with hyphens)
        faction_id = name.lower().replace(" ", "-").replace("'", "")

        faction = Faction(
            id=faction_id,
            campaign_id=self.campaign_id,
            name=name,
            description=description,
            goals=goals or [],
            resources=resources or [],
        )

        self._save(faction)
        return faction

    def get(self, faction_id: str) -> Faction | None:
        """Get a faction by ID.

        Args:
            faction_id: Faction ID

        Returns:
            Faction or None if not found
        """
        faction_file = self.factions_dir / f"{faction_id}.json"
        if not faction_file.exists():
            return None

        with open(faction_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert datetime strings back to datetime objects
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return Faction(**data)

    def get_by_name(self, name: str) -> Faction | None:
        """Get a faction by name.

        Args:
            name: Faction name (case-insensitive)

        Returns:
            Faction or None if not found
        """
        # Try generated ID first
        faction_id = name.lower().replace(" ", "-").replace("'", "")
        faction = self.get(faction_id)
        if faction:
            return faction

        # Fall back to searching all factions
        for faction in self.list_all():
            if faction.name.lower() == name.lower():
                return faction

        return None

    def update(self, faction: Faction) -> Faction:
        """Update a faction.

        Args:
            faction: Faction to update

        Returns:
            The updated Faction
        """
        faction.updated_at = datetime.now()
        self._save(faction)
        return faction

    def delete(self, faction_id: str) -> bool:
        """Delete a faction.

        Args:
            faction_id: Faction ID

        Returns:
            True if deleted, False if not found
        """
        faction_file = self.factions_dir / f"{faction_id}.json"
        if not faction_file.exists():
            return False

        faction_file.unlink()
        return True

    def list_all(self) -> list[Faction]:
        """List all factions in the campaign.

        Returns:
            List of all Factions
        """
        factions = []
        for faction_file in self.factions_dir.glob("*.json"):
            faction_id = faction_file.stem
            faction = self.get(faction_id)
            if faction:
                factions.append(faction)

        return factions

    def add_member(self, faction_id: str, character_id: str) -> bool:
        """Add a character to a faction.

        Args:
            faction_id: Faction ID
            character_id: Character ID to add

        Returns:
            True if successful, False if faction not found
        """
        faction = self.get(faction_id)
        if not faction:
            return False

        if character_id not in faction.member_character_ids:
            faction.member_character_ids.append(character_id)
            self.update(faction)

        return True

    def remove_member(self, faction_id: str, character_id: str) -> bool:
        """Remove a character from a faction.

        Args:
            faction_id: Faction ID
            character_id: Character ID to remove

        Returns:
            True if successful, False if faction not found or member not in faction
        """
        faction = self.get(faction_id)
        if not faction:
            return False

        if character_id in faction.member_character_ids:
            faction.member_character_ids.remove(character_id)
            self.update(faction)
            return True

        return False

    def get_members(self, faction_id: str) -> list[str]:
        """Get all member character IDs for a faction.

        Args:
            faction_id: Faction ID

        Returns:
            List of character IDs, or empty list if faction not found
        """
        faction = self.get(faction_id)
        if not faction:
            return []

        return faction.member_character_ids

    def add_shared_knowledge(self, faction_id: str, knowledge_id: str) -> bool:
        """Add shared knowledge to a faction.

        Args:
            faction_id: Faction ID
            knowledge_id: Knowledge ID to share with all members

        Returns:
            True if successful, False if faction not found
        """
        faction = self.get(faction_id)
        if not faction:
            return False

        if knowledge_id not in faction.shared_knowledge:
            faction.shared_knowledge.append(knowledge_id)
            self.update(faction)

        return True

    def update_reputation(self, faction_id: str, reputation: int) -> bool:
        """Update faction reputation with the party.

        Args:
            faction_id: Faction ID
            reputation: New reputation value (-100 to +100)

        Returns:
            True if successful, False if faction not found
        """
        faction = self.get(faction_id)
        if not faction:
            return False

        # Clamp to -100 to +100
        faction.reputation_with_party = max(-100, min(100, reputation))
        self.update(faction)
        return True

    def adjust_reputation(self, faction_id: str, delta: int) -> bool:
        """Adjust faction reputation by a delta.

        Args:
            faction_id: Faction ID
            delta: Amount to adjust reputation (+/-)

        Returns:
            True if successful, False if faction not found
        """
        faction = self.get(faction_id)
        if not faction:
            return False

        new_reputation = faction.reputation_with_party + delta
        faction.reputation_with_party = max(-100, min(100, new_reputation))
        self.update(faction)
        return True

    def set_inter_faction_attitude(
        self, faction_id: str, other_faction_id: str, attitude: str
    ) -> bool:
        """Set attitude between two factions.

        Args:
            faction_id: Faction ID
            other_faction_id: Other faction ID
            attitude: Attitude (allied, friendly, neutral, unfriendly, hostile)

        Returns:
            True if successful, False if faction not found
        """
        faction = self.get(faction_id)
        if not faction:
            return False

        faction.inter_faction_attitudes[other_faction_id] = attitude
        self.update(faction)
        return True

    def _save(self, faction: Faction) -> None:
        """Save a faction to disk.

        Args:
            faction: Faction to save
        """
        faction_file = self.factions_dir / f"{faction.id}.json"

        # Convert to dict and serialize datetimes
        data = faction.model_dump()
        data["created_at"] = faction.created_at.isoformat()
        data["updated_at"] = faction.updated_at.isoformat()

        with open(faction_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
