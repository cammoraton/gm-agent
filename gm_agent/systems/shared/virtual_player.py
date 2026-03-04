"""Virtual player data models.

VirtualPlayerConfig represents an AI-driven participant in generation games.
VirtualPlayerDecision represents a decision made by a virtual player.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VirtualPlayerConfig(BaseModel):
    """Configuration for an AI-driven player in generation games."""

    name: str
    personality_profile: dict[str, Any] = Field(default_factory=dict)
    creative_preferences: dict[str, str] = Field(default_factory=dict)
    is_active: bool = True

    def describe(self) -> str:
        """Short human-readable description."""
        if self.personality_profile:
            from .personality import PersonalityProfile
            pp = PersonalityProfile.from_dict(self.personality_profile)
            return f"{self.name} ({pp.archetype or 'custom'})"
        return self.name


class VirtualPlayerDecision(BaseModel):
    """A decision made by a virtual player."""

    player_name: str
    decision_type: str  # "create_period", "create_event", "add_district", etc.
    content: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
