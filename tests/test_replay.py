"""Tests for session replay functionality (Phase 5.1)."""

import pytest
from unittest.mock import Mock, patch
import uuid

from gm_agent.replay import SessionReplayer, ReplayResult, replay_session_instant
from gm_agent.storage.campaign import campaign_store
from gm_agent.storage.session import session_store
from gm_agent.storage.schemas import Turn, TurnMetadata
from gm_agent.models.base import LLMResponse


class TestReplayResult:
    """Tests for ReplayResult class."""

    def test_replay_result_init(self):
        """Test creating a replay result."""
        result = ReplayResult(model_name="test-model", session_id="session-123")
        assert result.model_name == "test-model"
        assert result.session_id == "session-123"
        assert result.turns == []
        assert result.total_time_ms == 0.0
        assert result.total_tokens == 0
        assert result.errors == []

    def test_add_turn(self):
        """Test adding a turn to replay result."""
        result = ReplayResult(model_name="test-model", session_id="session-123")

        metadata = TurnMetadata(
            source="replay",
            model="test-model",
            processing_time_ms=100.0,
            tool_count=2,
        )

        result.add_turn(
            turn_number=1,
            player_input="Test input",
            gm_response="Test response",
            processing_time_ms=100.0,
            metadata=metadata,
        )

        assert len(result.turns) == 1
        assert result.turns[0]["turn_number"] == 1
        assert result.turns[0]["player_input"] == "Test input"
        assert result.total_time_ms == 100.0

    def test_add_error(self):
        """Test adding an error to replay result."""
        result = ReplayResult(model_name="test-model", session_id="session-123")
        result.add_error(1, "Test error")

        assert len(result.errors) == 1
        assert "Turn 1" in result.errors[0]
        assert "Test error" in result.errors[0]

    def test_to_dict(self):
        """Test converting replay result to dictionary."""
        result = ReplayResult(model_name="test-model", session_id="session-123")

        metadata = TurnMetadata(source="replay", model="test-model")
        result.add_turn(1, "Input", "Output", 50.0, metadata)
        result.add_error(2, "Error occurred")

        result_dict = result.to_dict()

        assert result_dict["model_name"] == "test-model"
        assert result_dict["session_id"] == "session-123"
        assert result_dict["total_turns"] == 1
        assert len(result_dict["errors"]) == 1


class TestSessionReplayer:
    """Tests for SessionReplayer class."""

    @pytest.fixture
    def campaign(self, tmp_path):
        """Create test campaign."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            unique_name = f"test-campaign-{uuid.uuid4().hex[:8]}"
            campaign = campaign_store.create(unique_name, "Test Campaign")
            yield campaign

    @pytest.fixture
    def session_with_turns(self, tmp_path, campaign):
        """Create test session with some turns."""
        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            session = session_store.start(campaign.id)

            # Add some turns
            for i in range(3):
                session_store.add_turn(
                    campaign_id=campaign.id,
                    player_input=f"Player input {i + 1}",
                    gm_response=f"GM response {i + 1}",
                    metadata=TurnMetadata(source="test", model="test-model"),
                )

            session = session_store.get_current(campaign.id)

            # End the session so it's archived (can be retrieved via get())
            session_store.end(campaign.id, summary="Test session")

            # Get the ended session
            session = session_store.get(campaign.id, session.id)
            yield session

    def test_init(self, campaign):
        """Test creating a session replayer."""
        replayer = SessionReplayer(campaign.id)
        assert replayer.campaign_id == campaign.id
        assert replayer.campaign.name == campaign.name

    def test_init_invalid_campaign(self):
        """Test creating replayer with invalid campaign."""
        with pytest.raises(ValueError, match="not found"):
            SessionReplayer("nonexistent-campaign")

    def test_load_session(self, tmp_path, campaign, session_with_turns):
        """Test loading a session."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.replay.session_store", session_store):
                    replayer = SessionReplayer(campaign.id)
                    session = replayer.load_session(session_with_turns.id)

                    assert session.id == session_with_turns.id
                    assert len(session.turns) == 3

    def test_load_session_not_found(self, tmp_path, campaign):
        """Test loading non-existent session."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                replayer = SessionReplayer(campaign.id)

                with pytest.raises(ValueError, match="not found"):
                    replayer.load_session("nonexistent-session")

    def test_replay_instant(self, tmp_path, campaign, session_with_turns):
        """Test replaying a session instantly."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.replay.session_store", session_store):
                    with patch("gm_agent.replay.campaign_store", campaign_store):
                        # Mock LLM backend
                        mock_llm = Mock()
                        mock_llm.get_model_name.return_value = "mock-model"
                        mock_llm.chat.return_value = LLMResponse(
                            text="Replayed response",
                            tool_calls=[],
                            finish_reason="stop",
                        )

                        replayer = SessionReplayer(campaign.id)
                        results = list(
                            replayer.replay(
                                session_with_turns.id, speed_multiplier=0.0, llm=mock_llm, verbose=False
                            )
                        )

                        # Should have 3 results (one per turn)
                        assert len(results) == 3

                        # Check first result
                        assert results[0]["turn_number"] == 1
                        assert results[0]["player_input"] == "Player input 1"
                        assert results[0]["replayed_response"] == "Replayed response"
                        assert "processing_time_ms" in results[0]

    def test_replay_with_error(self, tmp_path, campaign, session_with_turns):
        """Test replay handling errors."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.replay.session_store", session_store):
                    with patch("gm_agent.replay.campaign_store", campaign_store):
                        # Mock LLM that raises error on second turn
                        mock_llm = Mock()
                        mock_llm.get_model_name.return_value = "mock-model"
                        call_count = [0]

                        def chat_side_effect(*args, **kwargs):
                            call_count[0] += 1
                            if call_count[0] == 2:
                                raise ValueError("Test error")
                            return LLMResponse(text="Response", tool_calls=[], finish_reason="stop")

                        mock_llm.chat.side_effect = chat_side_effect

                        replayer = SessionReplayer(campaign.id)
                        results = list(replayer.replay(session_with_turns.id, speed_multiplier=0.0, llm=mock_llm))

                        # Should have 3 results
                        assert len(results) == 3

                        # First should succeed
                        assert "replayed_response" in results[0]

                        # Second should have error
                        assert "error" in results[1]
                        assert "Test error" in results[1]["error"]

    def test_compare_models(self, tmp_path, campaign, session_with_turns):
        """Test comparing different models."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.replay.session_store", session_store):
                    with patch("gm_agent.replay.campaign_store", campaign_store):
                        # Create two mock LLM backends
                        mock_llm1 = Mock()
                        mock_llm1.get_model_name.return_value = "model-1"
                        mock_llm1.chat.return_value = LLMResponse(
                            text="Response from model 1", tool_calls=[], finish_reason="stop"
                        )

                        mock_llm2 = Mock()
                        mock_llm2.get_model_name.return_value = "model-2"
                        mock_llm2.chat.return_value = LLMResponse(
                            text="Response from model 2", tool_calls=[], finish_reason="stop"
                        )

                        model_backends = [("model-1", mock_llm1), ("model-2", mock_llm2)]

                        replayer = SessionReplayer(campaign.id)
                        results = replayer.compare_models(session_with_turns.id, model_backends, verbose=False)

                        # Should have results for both models
                        assert "model-1" in results
                        assert "model-2" in results
                        assert "_summary" in results

                        # Check model-1 results
                        assert results["model-1"]["model_name"] == "model-1"
                        assert results["model-1"]["total_turns"] == 3

                        # Check model-2 results
                        assert results["model-2"]["model_name"] == "model-2"
                        assert results["model-2"]["total_turns"] == 3

                        # Check summary
                        summary = results["_summary"]
                        assert summary["models_compared"] == 2
                        assert "performance" in summary
                        assert "model-1" in summary["performance"]
                        assert "model-2" in summary["performance"]

    def test_compare_models_with_errors(self, tmp_path, campaign, session_with_turns):
        """Test model comparison when one model errors."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.replay.session_store", session_store):
                    with patch("gm_agent.replay.campaign_store", campaign_store):
                        # Good model
                        mock_llm1 = Mock()
                        mock_llm1.get_model_name.return_value = "good-model"
                        mock_llm1.chat.return_value = LLMResponse(text="Good", tool_calls=[], finish_reason="stop")

                        # Bad model (errors)
                        mock_llm2 = Mock()
                        mock_llm2.get_model_name.return_value = "bad-model"
                        mock_llm2.chat.side_effect = ValueError("Model error")

                        model_backends = [("good-model", mock_llm1), ("bad-model", mock_llm2)]

                        replayer = SessionReplayer(campaign.id)
                        results = replayer.compare_models(session_with_turns.id, model_backends, verbose=False)

                        # Good model should complete all turns
                        assert results["good-model"]["total_turns"] == 3
                        assert len(results["good-model"]["errors"]) == 0

                        # Bad model should have errors
                        assert results["bad-model"]["total_turns"] == 0
                        assert len(results["bad-model"]["errors"]) > 0

                        # Summary should show error counts
                        assert results["_summary"]["errors"]["good-model"] == 0
                        assert results["_summary"]["errors"]["bad-model"] > 0


class TestReplayConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def campaign(self, tmp_path):
        """Create test campaign."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            unique_name = f"test-campaign-{uuid.uuid4().hex[:8]}"
            campaign = campaign_store.create(unique_name, "Test Campaign")
            yield campaign

    @pytest.fixture
    def session_with_turns(self, tmp_path, campaign):
        """Create test session with turns."""
        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            session = session_store.start(campaign.id)

            for i in range(2):
                session_store.add_turn(
                    campaign_id=campaign.id,
                    player_input=f"Input {i + 1}",
                    gm_response=f"Response {i + 1}",
                    metadata=TurnMetadata(source="test"),
                )

            session = session_store.get_current(campaign.id)

            # End the session so it's archived
            session_store.end(campaign.id, summary="Test session")

            # Get the ended session
            session = session_store.get(campaign.id, session.id)
            yield session

    def test_replay_session_instant(self, tmp_path, campaign, session_with_turns):
        """Test instant replay convenience function."""
        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
                with patch("gm_agent.replay.session_store", session_store):
                    with patch("gm_agent.replay.campaign_store", campaign_store):
                        mock_llm = Mock()
                        mock_llm.get_model_name.return_value = "test-model"
                        mock_llm.chat.return_value = LLMResponse(text="Response", tool_calls=[], finish_reason="stop")

                        results = replay_session_instant(campaign.id, session_with_turns.id, llm=mock_llm)

                        assert len(results) == 2
                        assert all("replayed_response" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
