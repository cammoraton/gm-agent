"""Tests for MoE Personality System."""

import math

import pytest

from gm_agent.systems.shared.personality import (
    TraitDimension,
    TRAIT_DIMENSIONS,
    TRAIT_CATEGORIES,
    PersonalityProfile,
    get_trait,
    list_traits,
)
from gm_agent.systems.shared.archetypes import ARCHETYPES


# ===========================================================================
# Trait Dimensions
# ===========================================================================


class TestTraitDimensions:
    def test_trait_count(self):
        assert len(TRAIT_DIMENSIONS) == 50

    def test_all_categories(self):
        expected = {"big_five", "hexaco", "creative", "decision", "interpersonal",
                    "gm", "motivation", "behavioral", "archetypal"}
        assert set(TRAIT_CATEGORIES.keys()) == expected

    def test_big_five_count(self):
        assert len(TRAIT_CATEGORIES["big_five"]) == 5

    def test_get_trait(self):
        t = get_trait("openness")
        assert isinstance(t, TraitDimension)
        assert t.name == "openness"
        assert t.category == "big_five"
        assert t.low_label == "conventional"
        assert t.high_label == "inventive"

    def test_get_trait_unknown(self):
        with pytest.raises(KeyError):
            get_trait("nonexistent_trait")

    def test_list_traits_all(self):
        traits = list_traits()
        assert len(traits) == 50
        assert all(isinstance(t, TraitDimension) for t in traits)

    def test_list_traits_by_category(self):
        gm_traits = list_traits("gm")
        assert len(gm_traits) == 8
        assert all(t.category == "gm" for t in gm_traits)

    def test_list_traits_empty_category(self):
        traits = list_traits("nonexistent")
        assert traits == []

    def test_trait_fields(self):
        for name, td in TRAIT_DIMENSIONS.items():
            assert td.name == name
            assert td.category
            assert td.description
            assert td.low_label
            assert td.high_label
            assert 0.0 <= td.default <= 1.0


# ===========================================================================
# PersonalityProfile — basics
# ===========================================================================


class TestPersonalityProfile:
    def test_empty_profile(self):
        pp = PersonalityProfile()
        assert pp.name == ""
        assert pp.traits == {}
        assert pp.archetype == ""

    def test_named_profile(self):
        pp = PersonalityProfile(name="test", traits={"openness": 0.9})
        assert pp.name == "test"
        assert pp.get_weight("openness") == 0.9

    def test_get_weight_default(self):
        pp = PersonalityProfile()
        assert pp.get_weight("openness") == 0.5

    def test_set_weight(self):
        pp = PersonalityProfile()
        pp.set_weight("openness", 0.8)
        assert pp.get_weight("openness") == 0.8

    def test_set_weight_clamped(self):
        pp = PersonalityProfile()
        pp.set_weight("openness", 1.5)
        assert pp.get_weight("openness") == 1.0
        pp.set_weight("openness", -0.5)
        assert pp.get_weight("openness") == 0.0

    def test_init_clamps_weights(self):
        pp = PersonalityProfile(traits={"openness": 2.0, "warmth": -1.0})
        assert pp.get_weight("openness") == 1.0
        assert pp.get_weight("warmth") == 0.0


# ===========================================================================
# PersonalityProfile — describe
# ===========================================================================


class TestPersonalityDescribe:
    def test_describe_empty(self):
        pp = PersonalityProfile(name="Empty")
        desc = pp.describe()
        assert "neutral" in desc.lower() or "balanced" in desc.lower()

    def test_describe_high_trait(self):
        pp = PersonalityProfile(name="Bold", traits={"risk_appetite": 0.9})
        desc = pp.describe()
        assert "bold" in desc.lower()

    def test_describe_low_trait(self):
        pp = PersonalityProfile(name="Cautious", traits={"risk_appetite": 0.1})
        desc = pp.describe()
        assert "cautious" in desc.lower()

    def test_describe_neutral_trait_skipped(self):
        pp = PersonalityProfile(name="Neutral", traits={"risk_appetite": 0.5})
        desc = pp.describe()
        # Neutral traits should not generate labels
        assert "cautious" not in desc.lower()
        assert "bold" not in desc.lower()

    def test_describe_with_archetype(self):
        pp = PersonalityProfile(name="Test", archetype="the_sage")
        pp.set_weight("openness", 0.9)
        desc = pp.describe()
        assert "the_sage" in desc

    def test_describe_for_prompt_empty(self):
        pp = PersonalityProfile()
        result = pp.describe_for_prompt()
        assert "Balanced" in result

    def test_describe_for_prompt_high(self):
        pp = PersonalityProfile(traits={"openness": 0.9})
        result = pp.describe_for_prompt()
        assert "inventive" in result.lower() or "Inventive" in result

    def test_describe_for_prompt_low(self):
        pp = PersonalityProfile(traits={"openness": 0.1})
        result = pp.describe_for_prompt()
        assert "conventional" in result.lower()

    def test_describe_for_prompt_neutral_skipped(self):
        pp = PersonalityProfile(traits={"openness": 0.5})
        result = pp.describe_for_prompt()
        assert "Balanced" in result


# ===========================================================================
# PersonalityProfile — merge and distance
# ===========================================================================


class TestPersonalityMergeDistance:
    def test_merge_equal_weight(self):
        a = PersonalityProfile(name="A", traits={"openness": 0.0})
        b = PersonalityProfile(name="B", traits={"openness": 1.0})
        merged = a.merge(b, weight=0.5)
        assert abs(merged.get_weight("openness") - 0.5) < 0.001

    def test_merge_all_self(self):
        a = PersonalityProfile(name="A", traits={"openness": 0.2})
        b = PersonalityProfile(name="B", traits={"openness": 0.8})
        merged = a.merge(b, weight=0.0)
        assert abs(merged.get_weight("openness") - 0.2) < 0.001

    def test_merge_all_other(self):
        a = PersonalityProfile(name="A", traits={"openness": 0.2})
        b = PersonalityProfile(name="B", traits={"openness": 0.8})
        merged = a.merge(b, weight=1.0)
        assert abs(merged.get_weight("openness") - 0.8) < 0.001

    def test_merge_name(self):
        a = PersonalityProfile(name="A")
        b = PersonalityProfile(name="B")
        merged = a.merge(b)
        assert "A" in merged.name
        assert "B" in merged.name

    def test_merge_disjoint_traits(self):
        a = PersonalityProfile(traits={"openness": 0.9})
        b = PersonalityProfile(traits={"warmth": 0.9})
        merged = a.merge(b, weight=0.5)
        # Both traits present (unset defaults to 0.5)
        assert "openness" in merged.traits
        assert "warmth" in merged.traits

    def test_distance_identical(self):
        a = PersonalityProfile(traits={"openness": 0.5})
        b = PersonalityProfile(traits={"openness": 0.5})
        assert a.distance(b) == 0.0

    def test_distance_symmetric(self):
        a = PersonalityProfile(traits={"openness": 0.0})
        b = PersonalityProfile(traits={"openness": 1.0})
        assert abs(a.distance(b) - b.distance(a)) < 0.001

    def test_distance_max(self):
        a = PersonalityProfile(traits={t: 0.0 for t in TRAIT_DIMENSIONS})
        b = PersonalityProfile(traits={t: 1.0 for t in TRAIT_DIMENSIONS})
        # Max distance = sqrt(50)
        expected = math.sqrt(50)
        assert abs(a.distance(b) - expected) < 0.01


# ===========================================================================
# Serialization
# ===========================================================================


class TestPersonalitySerialization:
    def test_to_dict(self):
        pp = PersonalityProfile(name="Test", traits={"openness": 0.9}, archetype="sage")
        d = pp.to_dict()
        assert d["name"] == "Test"
        assert d["traits"]["openness"] == 0.9
        assert d["archetype"] == "sage"

    def test_from_dict(self):
        d = {"name": "Test", "traits": {"openness": 0.9}, "archetype": "sage"}
        pp = PersonalityProfile.from_dict(d)
        assert pp.name == "Test"
        assert pp.get_weight("openness") == 0.9
        assert pp.archetype == "sage"

    def test_roundtrip(self):
        original = PersonalityProfile(
            name="Full",
            traits={"openness": 0.9, "warmth": 0.2, "risk_appetite": 0.7},
            archetype="the_sage",
        )
        d = original.to_dict()
        restored = PersonalityProfile.from_dict(d)
        assert restored.name == original.name
        assert restored.archetype == original.archetype
        assert restored.traits == original.traits

    def test_from_dict_empty(self):
        pp = PersonalityProfile.from_dict({})
        assert pp.name == ""
        assert pp.traits == {}


# ===========================================================================
# Archetypes
# ===========================================================================


class TestArchetypes:
    def test_archetype_count(self):
        assert len(ARCHETYPES) == 20

    def test_archetype_names(self):
        expected = {
            "the_sage", "the_trickster", "the_guardian", "the_rebel",
            "the_diplomat", "the_healer", "the_commander", "the_scholar",
            "the_mystic", "the_rogue", "the_merchant", "the_prophet",
            "the_explorer", "the_artisan", "the_noble", "the_hermit",
            "the_jester", "the_warrior", "the_mentor", "the_wildcard",
        }
        assert set(ARCHETYPES.keys()) == expected

    def test_archetype_traits_valid(self):
        for name, traits in ARCHETYPES.items():
            assert isinstance(traits, dict), f"{name} is not a dict"
            assert 8 <= len(traits) <= 12, f"{name} has {len(traits)} traits (expected 8-12)"
            for trait_name, weight in traits.items():
                assert trait_name in TRAIT_DIMENSIONS, f"{name} has unknown trait {trait_name}"
                assert 0.0 <= weight <= 1.0, f"{name}.{trait_name} = {weight} out of range"

    def test_from_archetype(self):
        pp = PersonalityProfile.from_archetype("the_sage")
        assert pp.name == "the_sage"
        assert pp.archetype == "the_sage"
        assert len(pp.traits) >= 8
        assert pp.get_weight("openness") > 0.5  # Sages should be open

    def test_from_archetype_unknown(self):
        with pytest.raises(KeyError):
            PersonalityProfile.from_archetype("nonexistent")

    def test_archetype_profiles_distinct(self):
        sage = PersonalityProfile.from_archetype("the_sage")
        warrior = PersonalityProfile.from_archetype("the_warrior")
        dist = sage.distance(warrior)
        assert dist > 0.5, f"Sage and Warrior too similar: distance={dist}"

    def test_each_archetype_loadable(self):
        for name in ARCHETYPES:
            pp = PersonalityProfile.from_archetype(name)
            assert pp.name == name
            desc = pp.describe()
            assert len(desc) > 10

    def test_repr(self):
        pp = PersonalityProfile.from_archetype("the_sage")
        r = repr(pp)
        assert "the_sage" in r
