"""Pre-defined personality archetypes.

Each archetype is a dict of {trait_name: weight} for its 8-12 most
distinctive traits. Unlisted traits default to 0.5 (neutral).
"""

ARCHETYPES: dict[str, dict[str, float]] = {
    "the_sage": {
        "openness": 0.9, "curiosity": 0.9, "information_gathering": 0.9,
        "patience": 0.8, "verbosity": 0.7, "detail_focus": 0.8,
        "warmth": 0.6, "planning_horizon": 0.8, "honesty_humility": 0.7,
        "intensity": 0.4,
    },
    "the_trickster": {
        "humor": 0.9, "playfulness": 0.9, "risk_appetite": 0.8,
        "honesty_humility": 0.2, "directness": 0.3, "mysteriousness": 0.7,
        "flexibility": 0.9, "improv_comfort": 0.9, "extraversion": 0.7,
        "conscientiousness": 0.3,
    },
    "the_guardian": {
        "loyalty": 0.9, "agreeableness": 0.8, "conscientiousness": 0.8,
        "risk_appetite": 0.3, "compassion": 0.8, "social_priority": 0.8,
        "patience": 0.7, "trust": 0.6, "assertiveness": 0.7,
        "consequence_severity": 0.4,
    },
    "the_rebel": {
        "risk_appetite": 0.9, "conflict_approach": 0.8, "flexibility": 0.3,
        "traditionalism": 0.1, "assertiveness": 0.9, "honesty_humility": 0.4,
        "idealism": 0.7, "authority": 0.7, "agreeableness": 0.3,
        "intensity": 0.8,
    },
    "the_diplomat": {
        "agreeableness": 0.9, "empathy": 0.8, "warmth": 0.8,
        "directness": 0.3, "formality": 0.7, "social_priority": 0.8,
        "patience": 0.8, "conflict_approach": 0.2, "trust": 0.6,
        "verbosity": 0.6,
    },
    "the_healer": {
        "compassion": 0.95, "empathy": 0.9, "warmth": 0.9,
        "patience": 0.8, "agreeableness": 0.8, "neuroticism": 0.6,
        "social_priority": 0.8, "challenge_level": 0.3,
        "consequence_severity": 0.2, "optimism": 0.7,
    },
    "the_commander": {
        "authority": 0.9, "assertiveness": 0.9, "planning_horizon": 0.8,
        "conscientiousness": 0.8, "directness": 0.8, "risk_appetite": 0.6,
        "spotlight_sharing": 0.3, "agreeableness": 0.4,
        "intensity": 0.7, "ambition": 0.8,
    },
    "the_scholar": {
        "information_gathering": 0.95, "detail_focus": 0.9, "curiosity": 0.85,
        "conscientiousness": 0.8, "patience": 0.8, "verbosity": 0.7,
        "extraversion": 0.3, "risk_appetite": 0.3, "openness": 0.8,
        "rules_adherence": 0.8,
    },
    "the_mystic": {
        "mysteriousness": 0.9, "openness": 0.85, "superstition": 0.8,
        "narrative_voice": 0.8, "drama": 0.7, "intensity": 0.7,
        "directness": 0.3, "formality": 0.6, "curiosity": 0.7,
        "genre_affinity": 0.4,
    },
    "the_rogue": {
        "risk_appetite": 0.8, "flexibility": 0.8, "directness": 0.4,
        "honesty_humility": 0.3, "mysteriousness": 0.7, "humor": 0.6,
        "planning_horizon": 0.4, "trust": 0.3, "extraversion": 0.6,
        "playfulness": 0.6,
    },
    "the_merchant": {
        "information_gathering": 0.7, "planning_horizon": 0.7,
        "social_priority": 0.4, "ambition": 0.8, "warmth": 0.6,
        "directness": 0.5, "patience": 0.7, "trust": 0.4,
        "formality": 0.6, "verbosity": 0.6,
    },
    "the_prophet": {
        "idealism": 0.9, "intensity": 0.9, "drama": 0.8,
        "authority": 0.7, "narrative_voice": 0.8, "mysteriousness": 0.7,
        "verbosity": 0.8, "assertiveness": 0.7, "compassion": 0.7,
        "superstition": 0.7,
    },
    "the_explorer": {
        "curiosity": 0.95, "openness": 0.9, "risk_appetite": 0.8,
        "flexibility": 0.8, "pacing": 0.7, "optimism": 0.7,
        "extraversion": 0.6, "improv_comfort": 0.8,
        "detail_focus": 0.6, "patience": 0.5,
    },
    "the_artisan": {
        "detail_focus": 0.9, "conscientiousness": 0.85, "patience": 0.8,
        "openness": 0.7, "pride": 0.7, "narrative_voice": 0.6,
        "planning_horizon": 0.7, "extraversion": 0.4,
        "ambition": 0.6, "traditionalism": 0.6,
    },
    "the_noble": {
        "formality": 0.9, "authority": 0.8, "pride": 0.8,
        "traditionalism": 0.7, "directness": 0.6, "ambition": 0.7,
        "propriety": 0.8, "warmth": 0.4, "agreeableness": 0.5,
        "verbosity": 0.6,
    },
    "the_hermit": {
        "extraversion": 0.1, "patience": 0.9, "openness": 0.7,
        "mysteriousness": 0.7, "verbosity": 0.2, "social_priority": 0.2,
        "trust": 0.3, "curiosity": 0.7, "intensity": 0.4,
        "warmth": 0.4,
    },
    "the_jester": {
        "humor": 0.95, "playfulness": 0.95, "extraversion": 0.9,
        "risk_appetite": 0.7, "formality": 0.1, "directness": 0.7,
        "optimism": 0.8, "drama": 0.7, "verbosity": 0.8,
        "conscientiousness": 0.3,
    },
    "the_warrior": {
        "assertiveness": 0.8, "risk_appetite": 0.7, "loyalty": 0.7,
        "directness": 0.8, "patience": 0.4, "conflict_approach": 0.8,
        "planning_horizon": 0.5, "conscientiousness": 0.6,
        "intensity": 0.7, "pride": 0.6,
    },
    "the_mentor": {
        "empathy": 0.8, "patience": 0.9, "warmth": 0.8,
        "spotlight_sharing": 0.9, "player_agency": 0.8,
        "narrative_focus": 0.7, "verbosity": 0.6, "authority": 0.6,
        "curiosity": 0.7, "optimism": 0.7,
    },
    "the_wildcard": {
        "flexibility": 0.9, "risk_appetite": 0.9, "improv_comfort": 0.9,
        "planning_horizon": 0.2, "conscientiousness": 0.2,
        "openness": 0.9, "playfulness": 0.7, "intensity": 0.7,
        "neuroticism": 0.6, "mysteriousness": 0.6,
    },
}
