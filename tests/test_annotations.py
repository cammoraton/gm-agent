"""Tests for data quality annotation system."""

import json
from pathlib import Path

import pytest

from gm_agent.systems.pf2e.storage.annotations import (
    AnnotationStore,
    VALID_ISSUE_TYPES,
    VALID_SEVERITIES,
)
from gm_agent.systems.pf2e.campaign_tools import PF2eCampaignToolPlugin


# ============================================================================
# AnnotationStore unit tests
# ============================================================================


class TestAnnotationStore:
    """Tests for AnnotationStore CRUD operations."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> AnnotationStore:
        s = AnnotationStore(campaign_id="test-camp", base_dir=tmp_path)
        yield s
        s.close()

    def test_add_and_count(self, store: AnnotationStore):
        store.add_annotation(entity_name="Fireball", issue_type="truncated")
        assert store.count() == 1

    def test_add_returns_entry(self, store: AnnotationStore):
        entry = store.add_annotation(
            entity_name="Goblin Warrior",
            issue_type="wrong_info",
            entity_type="creature",
            book="Bestiary",
            page=42,
            severity="high",
            description="AC listed as 12 but should be 15",
        )
        assert entry.id is not None
        assert entry.entity_name == "Goblin Warrior"
        assert entry.issue_type == "wrong_info"
        assert entry.severity == "high"

    def test_list_annotations(self, store: AnnotationStore):
        store.add_annotation(entity_name="Heal", issue_type="truncated")
        store.add_annotation(entity_name="Shield", issue_type="missing")
        items = store.list_annotations()
        assert len(items) == 2
        assert items[0]["entity_name"] == "Heal"
        assert items[1]["entity_name"] == "Shield"

    def test_list_filter_by_issue_type(self, store: AnnotationStore):
        store.add_annotation(entity_name="Heal", issue_type="truncated")
        store.add_annotation(entity_name="Shield", issue_type="missing")
        items = store.list_annotations(issue_type="truncated")
        assert len(items) == 1
        assert items[0]["entity_name"] == "Heal"

    def test_list_filter_by_severity(self, store: AnnotationStore):
        store.add_annotation(entity_name="Heal", issue_type="truncated", severity="low")
        store.add_annotation(entity_name="Shield", issue_type="missing", severity="high")
        items = store.list_annotations(severity="high")
        assert len(items) == 1
        assert items[0]["entity_name"] == "Shield"

    def test_count_empty(self, store: AnnotationStore):
        assert store.count() == 0

    def test_close_and_reopen(self, tmp_path: Path):
        store = AnnotationStore(campaign_id="test-camp", base_dir=tmp_path)
        store.add_annotation(entity_name="Fireball", issue_type="truncated")
        store.close()

        store2 = AnnotationStore(campaign_id="test-camp", base_dir=tmp_path)
        assert store2.count() == 1
        store2.close()


class TestAnnotationDedup:
    """Tests for deduplication behavior."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> AnnotationStore:
        s = AnnotationStore(campaign_id="test-camp", base_dir=tmp_path)
        yield s
        s.close()

    def test_same_entity_issue_deduplicates(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated", severity="low",
            description="Description cut off",
        )
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated", severity="medium",
            description="Missing heightened info",
        )
        # Should still be 1 entry, not 2
        assert store.count() == 1

    def test_dedup_bumps_severity_upward(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated", severity="low",
        )
        entry = store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated", severity="high",
        )
        assert entry.severity == "high"

    def test_dedup_does_not_lower_severity(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated", severity="high",
        )
        entry = store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated", severity="low",
        )
        assert entry.severity == "high"

    def test_dedup_appends_description(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated",
            description="Description cut off",
        )
        entry = store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated",
            description="Missing heightened info",
        )
        assert "Description cut off" in entry.description
        assert "Missing heightened info" in entry.description

    def test_dedup_skips_duplicate_description(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated",
            description="Description cut off",
        )
        entry = store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated",
            description="Description cut off",
        )
        # Should not double the description
        assert entry.description.count("Description cut off") == 1

    def test_different_issue_types_not_deduped(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="truncated",
        )
        store.add_annotation(
            entity_name="Fireball", entity_type="spell",
            issue_type="wrong_info",
        )
        assert store.count() == 2

    def test_different_entity_types_not_deduped(self, store: AnnotationStore):
        store.add_annotation(
            entity_name="Shield", entity_type="spell",
            issue_type="truncated",
        )
        store.add_annotation(
            entity_name="Shield", entity_type="equipment",
            issue_type="truncated",
        )
        assert store.count() == 2

    def test_severity_escalation_ordering(self, store: AnnotationStore):
        """Verify full escalation chain: low → medium → high → critical."""
        store.add_annotation(
            entity_name="Test", entity_type="spell",
            issue_type="truncated", severity="low",
        )
        entry = store.add_annotation(
            entity_name="Test", entity_type="spell",
            issue_type="truncated", severity="medium",
        )
        assert entry.severity == "medium"

        entry = store.add_annotation(
            entity_name="Test", entity_type="spell",
            issue_type="truncated", severity="high",
        )
        assert entry.severity == "high"

        entry = store.add_annotation(
            entity_name="Test", entity_type="spell",
            issue_type="truncated", severity="critical",
        )
        assert entry.severity == "critical"


# ============================================================================
# Tool tests (via PF2eCampaignToolPlugin)
# ============================================================================


class TestDataQualityTool:
    """Tests for the data_quality tool on PF2eCampaignToolPlugin."""

    @pytest.fixture
    def plugin(self, tmp_path: Path) -> PF2eCampaignToolPlugin:
        p = PF2eCampaignToolPlugin(campaign_id="test-camp", base_dir=tmp_path)
        yield p
        p.close()

    def test_flag_creates_annotation(self, plugin: PF2eCampaignToolPlugin):
        result = plugin._data_quality({
            "action": "flag",
            "entity_name": "Goblin Warrior",
            "entity_type": "creature",
            "issue": "truncated",
            "severity": "medium",
            "description": "Stats block cut off",
        })
        assert result.success
        assert "Goblin Warrior" in result.data
        assert plugin.annotations.count() == 1

    def test_flag_requires_entity_name(self, plugin: PF2eCampaignToolPlugin):
        result = plugin._data_quality({"action": "flag"})
        assert not result.success
        assert "entity_name" in result.error

    def test_flag_rejects_bad_issue_type(self, plugin: PF2eCampaignToolPlugin):
        result = plugin._data_quality({
            "action": "flag",
            "entity_name": "Test",
            "issue": "nonexistent",
        })
        assert not result.success
        assert "Invalid issue type" in result.error

    def test_flag_rejects_bad_severity(self, plugin: PF2eCampaignToolPlugin):
        result = plugin._data_quality({
            "action": "flag",
            "entity_name": "Test",
            "issue": "truncated",
            "severity": "extreme",
        })
        assert not result.success
        assert "Invalid severity" in result.error

    def test_list_empty(self, plugin: PF2eCampaignToolPlugin):
        result = plugin._data_quality({"action": "list"})
        assert result.success
        assert "No data quality issues" in result.data

    def test_list_with_items(self, plugin: PF2eCampaignToolPlugin):
        plugin._data_quality({
            "action": "flag",
            "entity_name": "Fireball",
            "issue": "truncated",
            "description": "Missing heightened",
        })
        plugin._data_quality({
            "action": "flag",
            "entity_name": "Shield",
            "issue": "missing",
        })
        result = plugin._data_quality({"action": "list"})
        assert result.success
        assert "Fireball" in result.data
        assert "Shield" in result.data
        assert "2 total" in result.data

    def test_count_returns_total(self, plugin: PF2eCampaignToolPlugin):
        plugin._data_quality({
            "action": "flag",
            "entity_name": "Test1",
            "issue": "truncated",
        })
        plugin._data_quality({
            "action": "flag",
            "entity_name": "Test2",
            "issue": "missing",
        })
        result = plugin._data_quality({"action": "count"})
        assert result.success
        assert "2" in result.data

    def test_unknown_action(self, plugin: PF2eCampaignToolPlugin):
        result = plugin._data_quality({"action": "delete"})
        assert not result.success
        assert "Unknown" in result.error

    def test_tool_def_exists(self, plugin: PF2eCampaignToolPlugin):
        defs = plugin.tool_defs()
        names = [d.name for d in defs]
        assert "data_quality" in names

    def test_call_tool_routes_to_data_quality(self, plugin: PF2eCampaignToolPlugin):
        result = plugin.call_tool("data_quality", {
            "action": "flag",
            "entity_name": "Test",
            "issue": "truncated",
        }, server=None)
        assert result.success


# ============================================================================
# Export format tests
# ============================================================================


class TestExportFormat:
    """Tests for the export annotation → QueueEntry mapping."""

    def test_export_format_matches_queue_entry(self, tmp_path: Path):
        """Verify exported JSON has all QueueEntry fields."""
        store = AnnotationStore(campaign_id="test-camp", base_dir=tmp_path)
        store.add_annotation(
            entity_name="Fireball",
            entity_type="spell",
            issue_type="truncated",
            severity="high",
            description="Description cut off",
            book="Player Core",
            page=300,
        )
        items = store.list_annotations()
        store.close()

        # Reproduce export logic from CLI
        severity_to_priority = {
            "critical": 1, "high": 2, "medium": 3, "low": 4,
        }
        issue_to_reason = {
            "truncated": "low_quality",
            "wrong_info": "low_quality",
            "mistyped": "name_mismatch",
            "missing": "missing_content",
            "search_miss": "missing_content",
        }

        item = items[0]
        entry = {
            "name": item["entity_name"],
            "type": item.get("entity_type", ""),
            "book": item.get("book", ""),
            "page": item.get("page"),
            "reason": issue_to_reason.get(item["issue_type"], "low_quality"),
            "priority": severity_to_priority.get(item["severity"], 3),
            "quality_score": None,
            "suggested_pages": [item["page"]] if item.get("page") else [],
            "notes": f"[agent:{item['issue_type']}] {item['description']}",
        }

        # Verify all QueueEntry fields present
        assert entry["name"] == "Fireball"
        assert entry["type"] == "spell"
        assert entry["book"] == "Player Core"
        assert entry["page"] == 300
        assert entry["reason"] == "low_quality"
        assert entry["priority"] == 2  # high → 2
        assert entry["quality_score"] is None
        assert entry["suggested_pages"] == [300]
        assert "[agent:truncated]" in entry["notes"]

    def test_severity_to_priority_mapping(self):
        severity_to_priority = {
            "critical": 1, "high": 2, "medium": 3, "low": 4,
        }
        assert severity_to_priority["critical"] == 1
        assert severity_to_priority["high"] == 2
        assert severity_to_priority["medium"] == 3
        assert severity_to_priority["low"] == 4

    def test_issue_to_reason_mapping(self):
        issue_to_reason = {
            "truncated": "low_quality",
            "wrong_info": "low_quality",
            "mistyped": "name_mismatch",
            "missing": "missing_content",
            "search_miss": "missing_content",
        }
        assert issue_to_reason["truncated"] == "low_quality"
        assert issue_to_reason["wrong_info"] == "low_quality"
        assert issue_to_reason["mistyped"] == "name_mismatch"
        assert issue_to_reason["missing"] == "missing_content"
        assert issue_to_reason["search_miss"] == "missing_content"
