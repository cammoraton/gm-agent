"""Mixture-of-Experts personality system.

Defines ~50 composable trait dimensions that compose like a Mixture of Experts.
All processing is template-based — zero LLM cost. Usable for virtual players,
NPC enrichment, and GM agent personality tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Trait Dimensions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraitDimension:
    """A single personality trait axis (0.0 = low end, 1.0 = high end)."""

    name: str
    category: str
    description: str
    low_label: str
    high_label: str
    default: float = 0.5


# Master list of all trait dimensions, keyed by name.
TRAIT_DIMENSIONS: dict[str, TraitDimension] = {}

_TRAIT_DATA: list[tuple[str, str, str, str, str]] = [
    # Big Five
    ("openness", "big_five", "Receptivity to new ideas and experiences", "conventional", "inventive"),
    ("conscientiousness", "big_five", "Organization, discipline, and reliability", "spontaneous", "disciplined"),
    ("extraversion", "big_five", "Energy from social interaction", "reserved", "outgoing"),
    ("agreeableness", "big_five", "Tendency toward cooperation and empathy", "competitive", "cooperative"),
    ("neuroticism", "big_five", "Tendency toward emotional instability", "stable", "reactive"),
    # HEXACO extensions
    ("honesty_humility", "hexaco", "Sincerity, fairness, and modesty", "manipulative", "sincere"),
    ("emotionality", "hexaco", "Emotional sensitivity and attachment", "detached", "sensitive"),
    ("propriety", "hexaco", "Concern for social norms and decorum", "irreverent", "proper"),
    # Creative style
    ("narrative_voice", "creative", "Preference for narrative style", "terse", "flowery"),
    ("humor", "creative", "Use of humor and levity", "serious", "comedic"),
    ("drama", "creative", "Preference for dramatic tension", "understated", "theatrical"),
    ("detail_focus", "creative", "Level of descriptive detail", "broad_strokes", "granular"),
    ("pacing", "creative", "Preferred speed of narrative", "slow_burn", "action_packed"),
    ("genre_affinity", "creative", "Dark vs light genre preference", "grimdark", "hopeful"),
    # Decision making
    ("risk_appetite", "decision", "Willingness to take risks", "cautious", "bold"),
    ("planning_horizon", "decision", "Short-term vs long-term thinking", "impulsive", "strategic"),
    ("social_priority", "decision", "Individual vs group focus", "self_focused", "group_focused"),
    ("conflict_approach", "decision", "Preferred conflict resolution", "avoidant", "confrontational"),
    ("information_gathering", "decision", "How much info before deciding", "instinctive", "analytical"),
    ("flexibility", "decision", "Willingness to change plans", "rigid", "adaptable"),
    # Interpersonal
    ("empathy", "interpersonal", "Understanding others' emotions", "detached", "empathic"),
    ("assertiveness", "interpersonal", "Willingness to assert own needs", "passive", "assertive"),
    ("warmth", "interpersonal", "Interpersonal warmth", "cold", "warm"),
    ("trust", "interpersonal", "Baseline trust of others", "suspicious", "trusting"),
    ("formality", "interpersonal", "Communication formality", "casual", "formal"),
    ("directness", "interpersonal", "Communication directness", "indirect", "blunt"),
    # GM-specific
    ("rules_adherence", "gm", "How strictly rules are followed", "loose", "strict"),
    ("player_agency", "gm", "How much control players have", "directed", "sandbox"),
    ("challenge_level", "gm", "Encounter difficulty preference", "forgiving", "punishing"),
    ("narrative_focus", "gm", "Rules vs story emphasis", "mechanical", "narrative"),
    ("world_reactivity", "gm", "How much world reacts to players", "static", "reactive"),
    ("improv_comfort", "gm", "Comfort with improvisation", "scripted", "improvisational"),
    ("spotlight_sharing", "gm", "How equally attention is distributed", "star_focused", "egalitarian"),
    ("consequence_severity", "gm", "How harsh consequences are", "gentle", "harsh"),
    # Motivation
    ("curiosity", "motivation", "Drive to explore and discover", "incurious", "inquisitive"),
    ("ambition", "motivation", "Drive to achieve and advance", "content", "ambitious"),
    ("loyalty", "motivation", "Commitment to allies and causes", "independent", "devoted"),
    ("idealism", "motivation", "Belief in principles over pragmatism", "pragmatic", "idealistic"),
    ("compassion", "motivation", "Concern for others' suffering", "indifferent", "compassionate"),
    ("pride", "motivation", "Concern for reputation and honor", "humble", "proud"),
    # Behavioral
    ("verbosity", "behavioral", "Talkativeness", "laconic", "verbose"),
    ("superstition", "behavioral", "Belief in omens and fate", "skeptical", "superstitious"),
    ("patience", "behavioral", "Tolerance for waiting", "impatient", "patient"),
    ("memory_focus", "behavioral", "Tendency to recall and reference past", "present_focused", "nostalgic"),
    ("optimism", "behavioral", "Outlook on outcomes", "pessimistic", "optimistic"),
    # Archetypal
    ("mysteriousness", "archetypal", "How much is concealed", "transparent", "enigmatic"),
    ("authority", "archetypal", "Natural command presence", "meek", "commanding"),
    ("playfulness", "archetypal", "Lightheartedness in interactions", "somber", "playful"),
    ("traditionalism", "archetypal", "Respect for tradition and custom", "progressive", "traditional"),
    ("intensity", "archetypal", "Emotional and behavioral intensity", "mellow", "intense"),
]

for _name, _cat, _desc, _low, _high in _TRAIT_DATA:
    TRAIT_DIMENSIONS[_name] = TraitDimension(
        name=_name, category=_cat, description=_desc,
        low_label=_low, high_label=_high,
    )

TRAIT_CATEGORIES: dict[str, list[str]] = {}
for _td in TRAIT_DIMENSIONS.values():
    TRAIT_CATEGORIES.setdefault(_td.category, []).append(_td.name)


def get_trait(name: str) -> TraitDimension:
    """Get a trait dimension by name. Raises KeyError if not found."""
    return TRAIT_DIMENSIONS[name]


def list_traits(category: str | None = None) -> list[TraitDimension]:
    """List trait dimensions, optionally filtered by category."""
    if category:
        names = TRAIT_CATEGORIES.get(category, [])
        return [TRAIT_DIMENSIONS[n] for n in names]
    return list(TRAIT_DIMENSIONS.values())


# ---------------------------------------------------------------------------
# Personality Profile
# ---------------------------------------------------------------------------


class PersonalityProfile:
    """A named personality profile with weighted trait dimensions.

    Traits not explicitly set default to 0.5 (neutral). All operations
    are pure arithmetic — no LLM calls.
    """

    def __init__(
        self,
        name: str = "",
        traits: dict[str, float] | None = None,
        archetype: str = "",
    ):
        self.name = name
        self.traits: dict[str, float] = {}
        self.archetype = archetype
        if traits:
            for k, v in traits.items():
                self.set_weight(k, v)

    def get_weight(self, trait_name: str) -> float:
        """Get weight for a trait (default 0.5 if unset)."""
        return self.traits.get(trait_name, 0.5)

    def set_weight(self, trait_name: str, weight: float) -> None:
        """Set weight for a trait, clamped to [0.0, 1.0]."""
        self.traits[trait_name] = max(0.0, min(1.0, weight))

    def describe(self) -> str:
        """Generate a natural language paragraph from trait weights."""
        if not self.traits:
            return f"{self.name or 'This character'} has a neutral, balanced personality."

        parts = []
        for name, weight in sorted(self.traits.items()):
            td = TRAIT_DIMENSIONS.get(name)
            if td is None:
                continue
            if weight <= 0.25:
                parts.append(f"strongly {td.low_label}")
            elif weight <= 0.4:
                parts.append(f"somewhat {td.low_label}")
            elif weight >= 0.75:
                parts.append(f"strongly {td.high_label}")
            elif weight >= 0.6:
                parts.append(f"somewhat {td.high_label}")
            # 0.4-0.6 = neutral, skip

        label = self.name or "This character"
        if self.archetype:
            label = f"{label} (the {self.archetype})"

        if not parts:
            return f"{label} has a balanced personality."
        return f"{label} is {', '.join(parts)}."

    def describe_for_prompt(self) -> str:
        """Generate bullet-point personality for LLM system prompts."""
        if not self.traits:
            return "- Balanced, neutral personality"

        lines = []
        for name, weight in sorted(self.traits.items()):
            td = TRAIT_DIMENSIONS.get(name)
            if td is None:
                continue
            if weight <= 0.2:
                lines.append(f"- Very {td.low_label} ({td.description})")
            elif weight <= 0.35:
                lines.append(f"- {td.low_label.capitalize()} ({td.description})")
            elif weight >= 0.8:
                lines.append(f"- Very {td.high_label} ({td.description})")
            elif weight >= 0.65:
                lines.append(f"- {td.high_label.capitalize()} ({td.description})")
            # Skip near-neutral traits (0.35-0.65)

        if not lines:
            return "- Balanced, neutral personality"
        return "\n".join(lines)

    def merge(self, other: PersonalityProfile, weight: float = 0.5) -> PersonalityProfile:
        """Blend two profiles. weight=0.0 is all self, 1.0 is all other."""
        w = max(0.0, min(1.0, weight))
        all_keys = set(self.traits) | set(other.traits)
        merged_traits = {}
        for k in all_keys:
            a = self.get_weight(k)
            b = other.get_weight(k)
            merged_traits[k] = a * (1 - w) + b * w
        name = f"{self.name}+{other.name}" if self.name and other.name else self.name or other.name
        return PersonalityProfile(name=name, traits=merged_traits)

    def distance(self, other: PersonalityProfile) -> float:
        """Euclidean distance on the trait vector (all 50 dimensions)."""
        total = 0.0
        for name in TRAIT_DIMENSIONS:
            a = self.get_weight(name)
            b = other.get_weight(name)
            total += (a - b) ** 2
        return math.sqrt(total)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "traits": dict(self.traits),
            "archetype": self.archetype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityProfile:
        """Deserialize from dict."""
        return cls(
            name=data.get("name", ""),
            traits=data.get("traits", {}),
            archetype=data.get("archetype", ""),
        )

    @classmethod
    def from_archetype(cls, archetype_name: str) -> PersonalityProfile:
        """Create a profile from a named archetype.

        Raises KeyError if archetype not found.
        """
        from .archetypes import ARCHETYPES
        arch = ARCHETYPES[archetype_name]
        return cls(
            name=archetype_name,
            traits=dict(arch),
            archetype=archetype_name,
        )

    def __repr__(self) -> str:
        return f"PersonalityProfile(name={self.name!r}, traits={len(self.traits)}, archetype={self.archetype!r})"
