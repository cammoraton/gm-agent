"""Secret storage and revelation tracking."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import Secret
from ..config import CAMPAIGNS_DIR

if TYPE_CHECKING:
    pass


class SecretStore:
    """Storage for campaign secrets and revelation tracking.

    Secrets are stored as JSON files in the campaign directory.
    Each secret represents plot-critical information with tracked revelations.
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        """Initialize secret store.

        Args:
            campaign_id: The campaign ID
            base_dir: Base directory for campaigns (for testing)
        """
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.secrets_dir = self.base_dir / campaign_id / "secrets"
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        content: str,
        importance: str = "major",
        known_by_character_ids: list[str] | None = None,
        known_by_faction_ids: list[str] | None = None,
        consequences: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Secret:
        """Create a new secret.

        Args:
            content: The secret content
            importance: "minor", "major", or "critical"
            known_by_character_ids: Character IDs who know this secret
            known_by_faction_ids: Faction IDs whose members know this
            consequences: What happens when revealed
            tags: Tags for categorization

        Returns:
            The created Secret
        """
        # Generate unique ID with microseconds for uniqueness
        import random
        now = datetime.now()
        secret_id = f"secret-{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond:06d}-{random.randint(100, 999)}"

        secret = Secret(
            id=secret_id,
            campaign_id=self.campaign_id,
            content=content,
            importance=importance,
            known_by_character_ids=known_by_character_ids or [],
            known_by_faction_ids=known_by_faction_ids or [],
            consequences=consequences or [],
            tags=tags or [],
        )

        self._save(secret)
        return secret

    def get(self, secret_id: str) -> Secret | None:
        """Get a secret by ID.

        Args:
            secret_id: The secret ID

        Returns:
            The Secret or None if not found
        """
        file_path = self.secrets_dir / f"{secret_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return Secret(**data)

    def update(self, secret: Secret) -> Secret:
        """Update a secret.

        Args:
            secret: The secret to update

        Returns:
            The updated Secret
        """
        secret.updated_at = datetime.now()
        self._save(secret)
        return secret

    def delete(self, secret_id: str) -> bool:
        """Delete a secret.

        Args:
            secret_id: The secret ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.secrets_dir / f"{secret_id}.json"
        if not file_path.exists():
            return False

        file_path.unlink()
        return True

    def list_all(self, revealed: bool | None = None) -> list[Secret]:
        """List all secrets in the campaign.

        Args:
            revealed: Filter by revelation status (None = all)

        Returns:
            List of all Secrets matching filter
        """
        secrets = []
        for file_path in self.secrets_dir.glob("*.json"):
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                secret = Secret(**data)

                # Apply filter
                if revealed is not None:
                    if secret.revealed_to_party != revealed:
                        continue

                secrets.append(secret)

        return sorted(secrets, key=lambda s: s.created_at, reverse=True)

    def add_knower(self, secret_id: str, character_id: str) -> bool:
        """Add a character who knows this secret.

        Args:
            secret_id: The secret ID
            character_id: The character ID

        Returns:
            True if successful, False if secret not found
        """
        secret = self.get(secret_id)
        if not secret:
            return False

        if character_id not in secret.known_by_character_ids:
            secret.known_by_character_ids.append(character_id)
            self.update(secret)

        return True

    def add_faction_knower(self, secret_id: str, faction_id: str) -> bool:
        """Add a faction whose members know this secret.

        Args:
            secret_id: The secret ID
            faction_id: The faction ID

        Returns:
            True if successful, False if secret not found
        """
        secret = self.get(secret_id)
        if not secret:
            return False

        if faction_id not in secret.known_by_faction_ids:
            secret.known_by_faction_ids.append(faction_id)
            self.update(secret)

        return True

    def reveal_to_party(
        self,
        secret_id: str,
        session_id: str | None = None,
        turn_number: int | None = None,
        revealer: str | None = None,
        method: str | None = None,
    ) -> bool:
        """Mark a secret as revealed to the party.

        Args:
            secret_id: The secret ID
            session_id: Session where revealed
            turn_number: Turn number where revealed
            revealer: Who revealed it (character name)
            method: How it was revealed

        Returns:
            True if successful, False if secret not found
        """
        secret = self.get(secret_id)
        if not secret:
            return False

        secret.revealed_to_party = True
        secret.revelation_event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "turn_number": turn_number,
            "revealer": revealer,
            "method": method,
        }

        self.update(secret)
        return True

    def trigger_consequence(self, secret_id: str, consequence: str) -> bool:
        """Mark a consequence as triggered.

        Args:
            secret_id: The secret ID
            consequence: The consequence description

        Returns:
            True if successful, False if secret not found or consequence already triggered
        """
        secret = self.get(secret_id)
        if not secret:
            return False

        if consequence not in secret.consequences:
            return False

        if consequence in secret.triggered_consequences:
            return False  # Already triggered

        secret.triggered_consequences.append(consequence)
        self.update(secret)
        return True

    def get_untriggered_consequences(self, secret_id: str) -> list[str]:
        """Get consequences that haven't been triggered yet.

        Args:
            secret_id: The secret ID

        Returns:
            List of untriggered consequences
        """
        secret = self.get(secret_id)
        if not secret:
            return []

        return [c for c in secret.consequences if c not in secret.triggered_consequences]

    def get_character_secrets(self, character_id: str) -> list[Secret]:
        """Get all secrets known by a character.

        Args:
            character_id: The character ID

        Returns:
            List of Secrets known by this character
        """
        all_secrets = self.list_all()
        return [s for s in all_secrets if character_id in s.known_by_character_ids]

    def get_faction_secrets(self, faction_id: str) -> list[Secret]:
        """Get all secrets known by a faction.

        Args:
            faction_id: The faction ID

        Returns:
            List of Secrets known by this faction
        """
        all_secrets = self.list_all()
        return [s for s in all_secrets if faction_id in s.known_by_faction_ids]

    def _save(self, secret: Secret) -> None:
        """Save secret to disk.

        Args:
            secret: The secret to save
        """
        file_path = self.secrets_dir / f"{secret.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(secret.model_dump(mode="json"), f, indent=2, default=str)
