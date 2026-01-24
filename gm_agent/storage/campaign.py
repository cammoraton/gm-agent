"""Campaign storage and management."""

import json
import re
from pathlib import Path

from ..config import CAMPAIGNS_DIR
from .schemas import Campaign


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


class CampaignStore:
    """Manages campaign persistence."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _campaign_dir(self, campaign_id: str) -> Path:
        return self.base_dir / campaign_id

    def _campaign_file(self, campaign_id: str) -> Path:
        return self._campaign_dir(campaign_id) / "campaign.json"

    def create(self, name: str, background: str = "", **kwargs) -> Campaign:
        """Create a new campaign."""
        campaign_id = slugify(name)

        if self._campaign_file(campaign_id).exists():
            raise ValueError(f"Campaign '{campaign_id}' already exists")

        campaign = Campaign(
            id=campaign_id,
            name=name,
            background=background,
            **kwargs,
        )

        campaign_dir = self._campaign_dir(campaign_id)
        campaign_dir.mkdir(parents=True, exist_ok=True)
        (campaign_dir / "sessions").mkdir(exist_ok=True)

        self._save(campaign)
        return campaign

    def get(self, campaign_id: str) -> Campaign | None:
        """Get a campaign by ID."""
        campaign_file = self._campaign_file(campaign_id)
        if not campaign_file.exists():
            return None

        with open(campaign_file) as f:
            data = json.load(f)
        return Campaign.model_validate(data)

    def list(self) -> list[Campaign]:
        """List all campaigns."""
        campaigns = []
        if not self.base_dir.exists():
            return campaigns

        for campaign_dir in self.base_dir.iterdir():
            if campaign_dir.is_dir():
                campaign = self.get(campaign_dir.name)
                if campaign:
                    campaigns.append(campaign)

        return sorted(campaigns, key=lambda c: c.name)

    def update(self, campaign: Campaign) -> Campaign:
        """Update a campaign."""
        from datetime import datetime

        campaign.updated_at = datetime.now()
        self._save(campaign)
        return campaign

    def delete(self, campaign_id: str) -> bool:
        """Delete a campaign and all its sessions."""
        import shutil

        campaign_dir = self._campaign_dir(campaign_id)
        if not campaign_dir.exists():
            return False

        shutil.rmtree(campaign_dir)
        return True

    def _save(self, campaign: Campaign) -> None:
        """Save a campaign to disk."""
        with open(self._campaign_file(campaign.id), "w") as f:
            json.dump(campaign.model_dump(mode="json"), f, indent=2, default=str)


# Global instance
campaign_store = CampaignStore()
