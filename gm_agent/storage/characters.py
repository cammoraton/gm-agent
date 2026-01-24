"""Character profile storage and management."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..config import CAMPAIGNS_DIR
from .schemas import CharacterProfile


class CharacterStore:
    """Manages character profile persistence for a campaign."""

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self._characters_dir = self.base_dir / campaign_id / "characters"
        self._characters_dir.mkdir(parents=True, exist_ok=True)

    def _character_file(self, character_id: str) -> Path:
        return self._characters_dir / f"{character_id}.json"

    def create(
        self,
        name: str,
        character_type: str = "npc",
        **kwargs,
    ) -> CharacterProfile:
        """Create a new character profile.

        Args:
            name: Character name
            character_type: One of "npc", "monster", or "player"
            **kwargs: Additional profile fields

        Returns:
            The created CharacterProfile
        """
        character_id = str(uuid4())[:8]

        profile = CharacterProfile(
            id=character_id,
            campaign_id=self.campaign_id,
            name=name,
            character_type=character_type,
            **kwargs,
        )

        self._save(profile)
        return profile

    def get(self, character_id: str) -> CharacterProfile | None:
        """Get a character profile by ID."""
        character_file = self._character_file(character_id)
        if not character_file.exists():
            return None

        with open(character_file) as f:
            data = json.load(f)
        return CharacterProfile.model_validate(data)

    def get_by_name(self, name: str) -> CharacterProfile | None:
        """Get a character profile by name (case-insensitive)."""
        name_lower = name.lower()
        for profile in self.list():
            if profile.name.lower() == name_lower:
                return profile
        return None

    def list(self, character_type: str | None = None) -> list[CharacterProfile]:
        """List all character profiles, optionally filtered by type."""
        profiles = []

        for file_path in self._characters_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                profile = CharacterProfile.model_validate(data)

                if character_type is None or profile.character_type == character_type:
                    profiles.append(profile)
            except (json.JSONDecodeError, ValueError):
                continue

        return sorted(profiles, key=lambda p: p.name)

    def update(self, profile: CharacterProfile) -> CharacterProfile:
        """Update a character profile."""
        profile.updated_at = datetime.now()
        self._save(profile)
        return profile

    def delete(self, character_id: str) -> bool:
        """Delete a character profile."""
        character_file = self._character_file(character_id)
        if not character_file.exists():
            return False

        character_file.unlink()
        return True

    def _save(self, profile: CharacterProfile) -> None:
        """Save a character profile to disk."""
        with open(self._character_file(profile.id), "w") as f:
            json.dump(profile.model_dump(mode="json"), f, indent=2, default=str)


def get_character_store(campaign_id: str, base_dir: Path | None = None) -> CharacterStore:
    """Factory function to get a CharacterStore for a campaign."""
    return CharacterStore(campaign_id, base_dir)
