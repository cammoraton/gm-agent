"""Tests for session post-processing (crunch) system."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gm_agent.models.base import LLMResponse
from gm_agent.prep.crunch import CrunchPipeline, CrunchResult
from gm_agent.prep.session import (
    PARTY_KNOWLEDGE_ID,
    _format_transcript,
    _slugify_name,
    extract_dialogue,
    extract_events,
    update_arc,
    update_knowledge,
)
from gm_agent.prep.log import PrepLogger
from gm_agent.storage.campaign import CampaignStore
from gm_agent.storage.dialogue import DialogueStore
from gm_agent.storage.history import HistoryIndex
from gm_agent.storage.knowledge import KnowledgeStore
from gm_agent.storage.schemas import Campaign, Session, Turn

from tests.conftest import MockLLMBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    session_id: str = "test-sess",
    campaign_id: str = "test-campaign",
    turns: list[Turn] | None = None,
    summary: str = "",
) -> Session:
    """Create a test session."""
    if turns is None:
        turns = [
            Turn(
                player_input="I approach the goblin chieftain.",
                gm_response="The goblin chieftain Graak sneers at you. 'You dare enter my domain?' he growls. 'I will crush you like I crushed the merchant caravan!'",
            ),
            Turn(
                player_input="I try to persuade him to release the prisoners.",
                gm_response="Graak considers your words. 'Perhaps we can make a deal,' he says. 'Bring me the dragon's tooth from the caves to the north, and I'll release your precious merchants.' He leans forward conspiratorially. 'I've heard there's a secret entrance through the old mines.'",
            ),
        ]
    return Session(
        id=session_id,
        campaign_id=campaign_id,
        turns=turns,
        summary=summary,
    )


def _make_llm_with_json(json_data: list[dict] | str, thinking: str | None = None) -> MockLLMBackend:
    """Create a MockLLMBackend that returns JSON or plain text."""
    if isinstance(json_data, list):
        text = json.dumps(json_data)
    else:
        text = json_data
    return MockLLMBackend(
        responses=[LLMResponse(
            text=text,
            thinking=thinking,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )]
    )


# ---------------------------------------------------------------------------
# _format_transcript tests
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    def test_empty_session(self):
        session = _make_session(turns=[])
        assert "Empty session" in _format_transcript(session)

    def test_formats_turns(self):
        session = _make_session()
        text = _format_transcript(session)
        assert "[Turn 1]" in text
        assert "[Turn 2]" in text
        assert "Player: I approach the goblin" in text
        assert "GM: The goblin chieftain" in text


class TestSlugifyName:
    def test_basic(self):
        assert _slugify_name("Graak the Goblin") == "graak-the-goblin"

    def test_special_chars(self):
        assert _slugify_name("Zéphyr D'Arc") == "zphyr-darc"


# ---------------------------------------------------------------------------
# DialogueStore.delete_session_dialogues tests
# ---------------------------------------------------------------------------


class TestDeleteSessionDialogues:
    def test_deletes_session_dialogues(self, tmp_path):
        store = DialogueStore("test-campaign", base_dir=tmp_path)
        store.log_dialogue("npc1", "Graak", "sess-1", "Hello")
        store.log_dialogue("npc2", "Bob", "sess-1", "Goodbye")
        store.log_dialogue("npc1", "Graak", "sess-2", "Other session")

        deleted = store.delete_session_dialogues("sess-1")
        assert deleted == 2

        # sess-2 should be untouched
        remaining = store.search(session_id="sess-2")
        assert len(remaining) == 1
        store.close()

    def test_delete_nonexistent_session(self, tmp_path):
        store = DialogueStore("test-campaign", base_dir=tmp_path)
        deleted = store.delete_session_dialogues("no-such-session")
        assert deleted == 0
        store.close()


# ---------------------------------------------------------------------------
# extract_events tests
# ---------------------------------------------------------------------------


class TestExtractEvents:
    def test_extracts_events(self, tmp_path):
        events_json = [
            {"event": "Party confronted goblin chieftain Graak", "importance": "arc", "tags": ["goblin", "negotiation"]},
            {"event": "Graak offered a deal: dragon's tooth for prisoners", "importance": "arc", "tags": ["quest", "deal"]},
        ]
        llm = _make_llm_with_json(events_json)
        session = _make_session()
        history = HistoryIndex("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = extract_events(session, history, llm, logger, "test-campaign")

        assert count == 2
        events = history.get_session_events("test-sess")
        assert len(events) == 2
        assert events[0].importance == "arc"
        history.close()

    def test_empty_session_returns_zero(self, tmp_path):
        llm = _make_llm_with_json([])
        session = _make_session(turns=[])
        history = HistoryIndex("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = extract_events(session, history, llm, logger, "test-campaign")
        assert count == 0
        history.close()

    def test_skips_entries_without_event_text(self, tmp_path):
        events_json = [
            {"event": "Valid event", "importance": "session", "tags": []},
            {"event": "", "importance": "session", "tags": []},
            {"importance": "session"},
        ]
        llm = _make_llm_with_json(events_json)
        session = _make_session()
        history = HistoryIndex("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = extract_events(session, history, llm, logger, "test-campaign")
        assert count == 1
        history.close()

    def test_normalizes_invalid_importance(self, tmp_path):
        events_json = [
            {"event": "Something happened", "importance": "invalid_level", "tags": []},
        ]
        llm = _make_llm_with_json(events_json)
        session = _make_session()
        history = HistoryIndex("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        extract_events(session, history, llm, logger, "test-campaign")
        events = history.get_session_events("test-sess")
        assert events[0].importance == "session"
        history.close()

    def test_logs_to_prep_logger(self, tmp_path):
        events_json = [{"event": "Test event", "importance": "session", "tags": []}]
        llm = _make_llm_with_json(events_json)
        session = _make_session()
        history = HistoryIndex("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        extract_events(session, history, llm, logger, "test-campaign")

        entries = logger.read()
        assert len(entries) == 1
        assert entries[0].step == "event_extraction"
        history.close()


# ---------------------------------------------------------------------------
# extract_dialogue tests
# ---------------------------------------------------------------------------


class TestExtractDialogue:
    def test_extracts_dialogue(self, tmp_path):
        dialogue_json = [
            {
                "character_id": "graak",
                "character_name": "Graak",
                "content": "You dare enter my domain?",
                "dialogue_type": "threat",
                "flagged": False,
            },
            {
                "character_id": "graak",
                "character_name": "Graak",
                "content": "Bring me the dragon's tooth and I'll release your merchants.",
                "dialogue_type": "promise",
                "flagged": True,
            },
        ]
        llm = _make_llm_with_json(dialogue_json)
        session = _make_session()
        store = DialogueStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = extract_dialogue(session, store, llm, logger, "test-campaign")

        assert count == 2
        results = store.search(session_id="test-sess")
        assert len(results) == 2
        # Check promise is flagged
        flagged = [d for d in results if d.flagged]
        assert len(flagged) == 1
        assert flagged[0].dialogue_type == "promise"
        store.close()

    def test_maps_dialogue_types(self, tmp_path):
        dialogue_json = [
            {"character_id": "npc", "character_name": "NPC", "content": "A rumor", "dialogue_type": "rumor", "flagged": False},
            {"character_id": "npc", "character_name": "NPC", "content": "A secret", "dialogue_type": "secret", "flagged": True},
            {"character_id": "npc", "character_name": "NPC", "content": "A lie", "dialogue_type": "lie", "flagged": True},
        ]
        llm = _make_llm_with_json(dialogue_json)
        session = _make_session()
        store = DialogueStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = extract_dialogue(session, store, llm, logger, "test-campaign")
        assert count == 3

        results = store.search(session_id="test-sess")
        types = {d.dialogue_type for d in results}
        assert types == {"rumor", "secret", "lie"}
        store.close()

    def test_empty_session_returns_zero(self, tmp_path):
        llm = _make_llm_with_json([])
        session = _make_session(turns=[])
        store = DialogueStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = extract_dialogue(session, store, llm, logger, "test-campaign")
        assert count == 0
        store.close()

    def test_slugifies_name_when_no_character_id(self, tmp_path):
        dialogue_json = [
            {"character_name": "Zéphyr D'Arc", "content": "Hello", "dialogue_type": "statement", "flagged": False},
        ]
        llm = _make_llm_with_json(dialogue_json)
        session = _make_session()
        store = DialogueStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        extract_dialogue(session, store, llm, logger, "test-campaign")
        results = store.search(session_id="test-sess")
        assert results[0].character_id == "zphyr-darc"
        store.close()


# ---------------------------------------------------------------------------
# update_knowledge tests
# ---------------------------------------------------------------------------


class TestUpdateKnowledge:
    def test_updates_knowledge(self, tmp_path):
        knowledge_json = [
            {
                "character_id": "__party__",
                "character_name": "Party",
                "content": "There is a secret entrance through the old mines.",
                "knowledge_type": "rumor",
                "importance": 7,
                "tags": ["mines", "secret"],
            },
            {
                "character_id": "graak",
                "character_name": "Graak",
                "content": "The party seeks to free the prisoners.",
                "knowledge_type": "witnessed_event",
                "importance": 6,
                "tags": ["party", "prisoners"],
            },
        ]
        llm = _make_llm_with_json(knowledge_json)
        session = _make_session()
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        count = update_knowledge(session, knowledge, llm, logger, "test-campaign")

        assert count == 2
        # Check party knowledge
        party = knowledge.query_knowledge(character_id=PARTY_KNOWLEDGE_ID)
        assert len(party) == 1
        assert "secret entrance" in party[0].content
        assert party[0].source == "session:test-sess"
        # Check NPC knowledge
        npc = knowledge.query_knowledge(character_id="graak")
        assert len(npc) == 1
        knowledge.close()

    def test_includes_session_tag(self, tmp_path):
        knowledge_json = [
            {
                "character_id": "__party__",
                "character_name": "Party",
                "content": "Test knowledge",
                "knowledge_type": "fact",
                "importance": 5,
                "tags": ["test"],
            },
        ]
        llm = _make_llm_with_json(knowledge_json)
        session = _make_session()
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        update_knowledge(session, knowledge, llm, logger, "test-campaign")

        entries = knowledge.query_knowledge(character_id=PARTY_KNOWLEDGE_ID)
        assert f"session:test-sess" in entries[0].tags
        knowledge.close()

    def test_queries_existing_knowledge_for_context(self, tmp_path):
        """Verify the LLM receives existing knowledge to avoid duplicates."""
        llm = _make_llm_with_json([])
        session = _make_session()
        knowledge = KnowledgeStore("test-campaign", base_dir=tmp_path)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        # Pre-add some knowledge
        knowledge.add_knowledge(
            character_id=PARTY_KNOWLEDGE_ID,
            character_name="Party",
            content="Pre-existing fact",
            knowledge_type="fact",
        )

        update_knowledge(session, knowledge, llm, logger, "test-campaign")

        # Check that LLM was called with existing knowledge in context
        assert len(llm.calls) == 1
        messages = llm.calls[0][0]
        user_msg = messages[-1].content
        assert "Pre-existing fact" in user_msg
        knowledge.close()


# ---------------------------------------------------------------------------
# update_arc tests
# ---------------------------------------------------------------------------


class TestUpdateArc:
    def test_updates_arc(self, tmp_path):
        llm = _make_llm_with_json("The party has formed a tense alliance with the goblin chieftain Graak.")
        session = _make_session(summary="Party negotiated with goblin chieftain.")
        campaign_store = CampaignStore(base_dir=tmp_path)
        campaign = campaign_store.create("Test Campaign", background="Test")
        campaign.current_arc = "The party seeks to rescue merchants."
        campaign_store.update(campaign)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        new_arc = update_arc(
            session, campaign, campaign_store, llm, logger,
            events_summary="- [arc] Party confronted Graak",
        )

        assert "alliance" in new_arc
        # Verify campaign was updated in store
        updated = campaign_store.get(campaign.id)
        assert updated.current_arc == new_arc

    def test_preserves_context_in_prompt(self, tmp_path):
        """Verify the LLM receives current arc and events in prompt."""
        llm = _make_llm_with_json("Updated arc text.")
        session = _make_session(summary="Session summary here.")
        campaign_store = CampaignStore(base_dir=tmp_path)
        campaign = campaign_store.create("Test Campaign")
        campaign.current_arc = "Existing arc description."
        campaign_store.update(campaign)
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        update_arc(session, campaign, campaign_store, llm, logger, "- Event A\n- Event B")

        messages = llm.calls[0][0]
        user_msg = messages[-1].content
        assert "Existing arc description." in user_msg
        assert "Event A" in user_msg
        assert "Session summary here." in user_msg

    def test_handles_no_arc(self, tmp_path):
        llm = _make_llm_with_json("First arc established.")
        session = _make_session()
        campaign_store = CampaignStore(base_dir=tmp_path)
        campaign = campaign_store.create("Test Campaign")
        logger = PrepLogger("test-campaign", base_dir=tmp_path)

        new_arc = update_arc(session, campaign, campaign_store, llm, logger, "")
        assert new_arc == "First arc established."


# ---------------------------------------------------------------------------
# CrunchResult tests
# ---------------------------------------------------------------------------


class TestCrunchResult:
    def test_total_count(self):
        result = CrunchResult(
            campaign_id="test",
            session_id="sess",
            events_count=3,
            dialogue_count=5,
            knowledge_count=2,
        )
        assert result.total_count == 10

    def test_defaults(self):
        result = CrunchResult(campaign_id="test", session_id="sess")
        assert result.total_count == 0
        assert result.arc_updated is False
        assert result.errors == []


# ---------------------------------------------------------------------------
# CrunchPipeline tests
# ---------------------------------------------------------------------------


class TestCrunchPipeline:
    def _make_pipeline(self, tmp_path, llm=None):
        """Create a CrunchPipeline with all stores pointing to tmp_path."""
        if llm is None:
            llm = _make_llm_with_json([])
        pipeline = CrunchPipeline(
            campaign_id="test-campaign",
            llm=llm,
            knowledge_base_dir=tmp_path,
            logger_base_dir=tmp_path,
            campaigns_dir=tmp_path,
        )
        return pipeline

    def test_crunch_empty_session(self, tmp_path):
        # Create campaign for arc step
        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        pipeline = self._make_pipeline(tmp_path)
        session = _make_session(turns=[])

        result = pipeline.run(session)

        assert result.events_count == 0
        assert result.dialogue_count == 0
        assert result.knowledge_count == 0
        assert result.duration_ms > 0
        pipeline.close()

    def test_crunch_extracts_events(self, tmp_path):
        events_json = [{"event": "Party fought goblins", "importance": "session", "tags": ["combat"]}]
        # Need 4 LLM calls: events, dialogue, knowledge, arc
        llm = MockLLMBackend(responses=[
            LLMResponse(text=json.dumps(events_json), usage={}),  # events
            LLMResponse(text="[]", usage={}),  # dialogue
            LLMResponse(text="[]", usage={}),  # knowledge
            LLMResponse(text="Updated arc.", usage={}),  # arc
        ])

        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        pipeline = self._make_pipeline(tmp_path, llm=llm)
        session = _make_session()
        result = pipeline.run(session)

        assert result.events_count == 1
        pipeline.close()

    def test_crunch_extracts_dialogue(self, tmp_path):
        dialogue_json = [{"character_id": "npc", "character_name": "NPC", "content": "Hello", "dialogue_type": "statement", "flagged": False}]
        llm = MockLLMBackend(responses=[
            LLMResponse(text="[]", usage={}),  # events
            LLMResponse(text=json.dumps(dialogue_json), usage={}),  # dialogue
            LLMResponse(text="[]", usage={}),  # knowledge
            LLMResponse(text="Updated arc.", usage={}),  # arc
        ])

        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        pipeline = self._make_pipeline(tmp_path, llm=llm)
        session = _make_session()
        result = pipeline.run(session)

        assert result.dialogue_count == 1
        pipeline.close()

    def test_crunch_updates_knowledge(self, tmp_path):
        knowledge_json = [{"character_id": "__party__", "character_name": "Party", "content": "Learned something", "knowledge_type": "fact", "importance": 5, "tags": []}]
        llm = MockLLMBackend(responses=[
            LLMResponse(text="[]", usage={}),  # events
            LLMResponse(text="[]", usage={}),  # dialogue
            LLMResponse(text=json.dumps(knowledge_json), usage={}),  # knowledge
            LLMResponse(text="Updated arc.", usage={}),  # arc
        ])

        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        pipeline = self._make_pipeline(tmp_path, llm=llm)
        session = _make_session()
        result = pipeline.run(session)

        assert result.knowledge_count == 1
        pipeline.close()

    def test_crunch_updates_arc(self, tmp_path):
        llm = MockLLMBackend(responses=[
            LLMResponse(text="[]", usage={}),  # events
            LLMResponse(text="[]", usage={}),  # dialogue
            LLMResponse(text="[]", usage={}),  # knowledge
            LLMResponse(text="The quest continues with new allies.", usage={}),  # arc
        ])

        cs = CampaignStore(base_dir=tmp_path)
        c = cs.create("Test Campaign")
        c.current_arc = "Old arc."
        cs.update(c)

        pipeline = self._make_pipeline(tmp_path, llm=llm)
        session = _make_session()
        result = pipeline.run(session)

        assert result.arc_updated is True
        updated_campaign = cs.get(c.id)
        assert "new allies" in updated_campaign.current_arc
        pipeline.close()

    def test_step_filtering(self, tmp_path):
        """Only run specified steps."""
        events_json = [{"event": "Event", "importance": "session", "tags": []}]
        llm = MockLLMBackend(responses=[
            LLMResponse(text=json.dumps(events_json), usage={}),  # events only
        ])

        pipeline = self._make_pipeline(tmp_path, llm=llm)
        session = _make_session()
        result = pipeline.run(session, steps=["events"])

        assert result.events_count == 1
        assert result.dialogue_count == 0
        assert result.knowledge_count == 0
        assert result.arc_updated is False
        # LLM called once (only events step)
        assert len(llm.calls) == 1
        pipeline.close()

    def test_skip_steps(self, tmp_path):
        """Skip specified steps."""
        llm = MockLLMBackend(responses=[
            LLMResponse(text="[]", usage={}),  # dialogue
            LLMResponse(text="[]", usage={}),  # knowledge
            LLMResponse(text="Arc text.", usage={}),  # arc
        ])

        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        pipeline = self._make_pipeline(tmp_path, llm=llm)
        session = _make_session()
        result = pipeline.run(session, skip_steps=["events"])

        # events step was skipped
        assert result.events_count == 0
        # other steps ran (LLM called 3 times: dialogue, knowledge, arc)
        assert len(llm.calls) == 3
        pipeline.close()

    def test_idempotent_recrunch(self, tmp_path):
        """Running crunch twice produces same result by deleting first."""
        events_json = [{"event": "Event A", "importance": "session", "tags": []}]
        dialogue_json = [{"character_id": "npc", "character_name": "NPC", "content": "Hi", "dialogue_type": "statement", "flagged": False}]

        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        session = _make_session()

        # First run
        llm1 = MockLLMBackend(responses=[
            LLMResponse(text=json.dumps(events_json), usage={}),
            LLMResponse(text=json.dumps(dialogue_json), usage={}),
            LLMResponse(text="[]", usage={}),
            LLMResponse(text="Arc.", usage={}),
        ])
        p1 = CrunchPipeline("test-campaign", llm1, knowledge_base_dir=tmp_path, logger_base_dir=tmp_path, campaigns_dir=tmp_path)
        r1 = p1.run(session)

        # Second run (re-crunch) with different data
        events_json2 = [{"event": "Event B", "importance": "arc", "tags": []}]
        llm2 = MockLLMBackend(responses=[
            LLMResponse(text=json.dumps(events_json2), usage={}),
            LLMResponse(text="[]", usage={}),
            LLMResponse(text="[]", usage={}),
            LLMResponse(text="Arc 2.", usage={}),
        ])
        p2 = CrunchPipeline("test-campaign", llm2, knowledge_base_dir=tmp_path, logger_base_dir=tmp_path, campaigns_dir=tmp_path)
        r2 = p2.run(session)

        # Events should be replaced, not accumulated
        assert r2.events_count == 1
        events = p2.history.get_session_events("test-sess")
        assert len(events) == 1
        assert events[0].event == "Event B"

        # Dialogue should be replaced
        assert r2.dialogue_count == 0  # second run returned empty
        dialogues = p2.dialogue.search(session_id="test-sess")
        assert len(dialogues) == 0

        p1.close()
        p2.close()

    def test_error_isolation(self, tmp_path):
        """One step failing doesn't prevent others from running."""
        # events step will fail (LLM raises), dialogue should still work
        dialogue_json = [{"character_id": "npc", "character_name": "NPC", "content": "Hello", "dialogue_type": "statement", "flagged": False}]

        class FailOnFirstLLM(MockLLMBackend):
            def __init__(self):
                super().__init__(responses=[
                    LLMResponse(text="[]", usage={}),  # dialogue
                    LLMResponse(text="[]", usage={}),  # knowledge
                ])
                self._real_call_count = 0

            def chat(self, messages, tools=None, thinking=None):
                self._real_call_count += 1
                if self._real_call_count == 1:
                    raise RuntimeError("LLM connection failed")
                return super().chat(messages, tools, thinking=thinking)

        cs = CampaignStore(base_dir=tmp_path)
        cs.create("Test Campaign")

        llm = FailOnFirstLLM()
        pipeline = CrunchPipeline("test-campaign", llm, knowledge_base_dir=tmp_path, logger_base_dir=tmp_path, campaigns_dir=tmp_path)
        session = _make_session()

        # Skip arc step to avoid needing additional LLM response
        result = pipeline.run(session, skip_steps=["arc"])

        # events failed, but dialogue and knowledge should have continued
        assert "Event extraction failed" in result.errors[0]
        assert result.events_count == 0
        # At least one other step should have run
        assert len(llm.calls) >= 1
        pipeline.close()

    def test_full_pipeline(self, tmp_path):
        """End-to-end with all four steps producing results."""
        events_json = [
            {"event": "Combat with goblins", "importance": "session", "tags": ["combat"]},
            {"event": "Alliance with Graak", "importance": "arc", "tags": ["negotiation"]},
        ]
        dialogue_json = [
            {"character_id": "graak", "character_name": "Graak", "content": "A promise was made", "dialogue_type": "promise", "flagged": True},
        ]
        knowledge_json = [
            {"character_id": "__party__", "character_name": "Party", "content": "Secret mine entrance", "knowledge_type": "rumor", "importance": 7, "tags": ["mines"]},
            {"character_id": "graak", "character_name": "Graak", "content": "Party is formidable", "knowledge_type": "witnessed_event", "importance": 5, "tags": []},
        ]

        llm = MockLLMBackend(responses=[
            LLMResponse(text=json.dumps(events_json), usage={}),
            LLMResponse(text=json.dumps(dialogue_json), usage={}),
            LLMResponse(text=json.dumps(knowledge_json), usage={}),
            LLMResponse(text="The party forged an uneasy alliance with the goblin chieftain.", usage={}),
        ])

        cs = CampaignStore(base_dir=tmp_path)
        c = cs.create("Test Campaign")
        c.current_arc = "The party seeks to rescue merchants."
        cs.update(c)

        pipeline = CrunchPipeline("test-campaign", llm, knowledge_base_dir=tmp_path, logger_base_dir=tmp_path, campaigns_dir=tmp_path)
        session = _make_session(summary="Negotiated with goblin chieftain.")
        result = pipeline.run(session)

        assert result.events_count == 2
        assert result.dialogue_count == 1
        assert result.knowledge_count == 2
        assert result.arc_updated is True
        assert result.total_count == 5
        assert result.errors == []
        assert result.duration_ms > 0

        # Verify data was actually written
        assert len(pipeline.history.get_session_events("test-sess")) == 2
        assert len(pipeline.dialogue.search(session_id="test-sess")) == 1
        assert len(pipeline.knowledge.query_knowledge(limit=100)) == 2

        updated = cs.get(c.id)
        assert "alliance" in updated.current_arc.lower()
        pipeline.close()


# ---------------------------------------------------------------------------
# Integration: auto_crunch on end_session
# ---------------------------------------------------------------------------


class TestAutoCrunch:
    def test_auto_crunch_on_end_session(self, tmp_path):
        """Verify auto_crunch parameter triggers CrunchPipeline."""
        from gm_agent.storage.session import SessionStore

        session_st = SessionStore(base_dir=tmp_path)
        cs = CampaignStore(base_dir=tmp_path)
        campaign = cs.create("Test Campaign")

        # Start a session and add a turn
        session = session_st.start(campaign.id)
        session_st.add_turn(campaign.id, "Hello", "Greetings adventurer!")

        with patch("gm_agent.agent.session_store", session_st), \
             patch("gm_agent.agent.campaign_store", cs), \
             patch("gm_agent.prep.crunch.CrunchPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_instance.run.return_value = CrunchResult(
                campaign_id=campaign.id,
                session_id=session.id,
            )
            MockPipeline.return_value = mock_instance

            from gm_agent.agent import GMAgent

            # Create a minimal agent mock that calls end_session
            agent = object.__new__(GMAgent)
            agent.campaign = campaign
            agent.session = session_st.get_current(campaign.id)
            agent.llm = MockLLMBackend()
            agent._summarizer = None

            result = agent.end_session(auto_crunch=True)

            assert result is not None
            MockPipeline.assert_called_once()
            mock_instance.run.assert_called_once()
            mock_instance.close.assert_called_once()

    def test_auto_crunch_false_does_not_trigger(self, tmp_path):
        """Verify auto_crunch=False (default) does not trigger pipeline."""
        from gm_agent.storage.session import SessionStore

        session_st = SessionStore(base_dir=tmp_path)
        cs = CampaignStore(base_dir=tmp_path)
        campaign = cs.create("Test Campaign")

        session = session_st.start(campaign.id)
        session_st.add_turn(campaign.id, "Hello", "Greetings!")

        with patch("gm_agent.agent.session_store", session_st), \
             patch("gm_agent.agent.campaign_store", cs), \
             patch("gm_agent.prep.crunch.CrunchPipeline") as MockPipeline:

            from gm_agent.agent import GMAgent

            agent = object.__new__(GMAgent)
            agent.campaign = campaign
            agent.session = session_st.get_current(campaign.id)
            agent.llm = MockLLMBackend()
            agent._summarizer = None

            agent.end_session(auto_crunch=False)

            MockPipeline.assert_not_called()
