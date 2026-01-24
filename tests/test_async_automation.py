"""Tests for async automation processing (Phase 4.2)."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from gm_agent.tasks import (
    process_player_chat_async,
    process_npc_turn_async,
    get_task_progress,
    _publish_task_progress,
)


class TestTaskProgressPublishing:
    """Tests for task progress publishing."""

    @patch("gm_agent.tasks.get_redis_client")
    def test_publish_task_progress(self, mock_get_redis):
        """Test publishing task progress to Redis."""
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis

        _publish_task_progress("task-123", "PROCESSING", {"campaign_id": "test"})

        # Should set progress with TTL
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "task:progress:task-123"
        assert args[1] == 600  # 10 min TTL

    @patch("gm_agent.tasks.get_redis_client")
    def test_get_task_progress(self, mock_get_redis):
        """Test retrieving task progress."""
        import json

        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis

        progress_data = {"state": "SUCCESS", "timestamp": 123456}
        mock_redis.get.return_value = json.dumps(progress_data)

        result = get_task_progress("task-123")

        assert result["state"] == "SUCCESS"
        assert result["timestamp"] == 123456
        mock_redis.get.assert_called_once_with("task:progress:task-123")

    @patch("gm_agent.tasks.get_redis_client")
    def test_get_task_progress_not_found(self, mock_get_redis):
        """Test retrieving progress for non-existent task."""
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None

        result = get_task_progress("task-nonexistent")

        assert result is None


class TestCampaignLocking:
    """Tests for campaign-level task locking."""

    @patch("gm_agent.tasks.get_redis_client")
    def test_acquire_campaign_lock(self, mock_get_redis):
        """Test acquiring campaign lock."""
        from gm_agent.tasks import _acquire_campaign_lock

        mock_redis = Mock()
        mock_lock = Mock()
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        lock = _acquire_campaign_lock("test-campaign")

        mock_redis.lock.assert_called_once_with(
            "automation:lock:test-campaign", timeout=120, blocking_timeout=5
        )
        assert lock == mock_lock


class TestPlayerChatAsync:
    """Tests for async player chat processing."""

    @pytest.fixture
    def campaign(self, tmp_path):
        """Create test campaign."""
        from gm_agent.storage.campaign import campaign_store
        import uuid

        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            unique_name = f"test-campaign-{uuid.uuid4().hex[:8]}"
            campaign = campaign_store.create(unique_name, "Test Campaign")
            yield campaign

    @pytest.fixture
    def session(self, tmp_path, campaign):
        """Create test session."""
        from gm_agent.storage.session import session_store

        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            session = session_store.start(campaign.id)
            yield session

    @patch("gm_agent.tasks.get_redis_client")
    @patch("gm_agent.tasks._publish_task_progress")
    def test_process_player_chat_async_success(
        self, mock_publish, mock_get_redis, tmp_path, campaign, session
    ):
        """Test successful async player chat processing."""
        from gm_agent.storage.campaign import campaign_store
        from gm_agent.storage.session import session_store

        # Mock Redis lock
        mock_redis = Mock()
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        # Mock GMAgent
        with patch("gm_agent.agent.GMAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent.process_turn.return_value = "Test response"
            mock_agent_class.return_value = mock_agent

            with patch("gm_agent.tasks.campaign_store", campaign_store):
                with patch("gm_agent.tasks.session_store", session_store):
                    # Create mock task with request id
                    mock_self = Mock()
                    mock_self.request.id = "task-123"

                    result = process_player_chat_async(
                        mock_self, campaign.id, "What is a goblin?"
                    )

        assert result["success"] is True
        assert result["response"] == "Test response"
        assert "turn_number" in result

        # Verify lock was acquired and released
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()

        # Verify agent was used
        mock_agent.process_turn.assert_called_once()
        mock_agent.close.assert_called_once()

    @patch("gm_agent.tasks.get_redis_client")
    @patch("gm_agent.tasks._publish_task_progress")
    def test_process_player_chat_async_campaign_not_found(
        self, mock_publish, mock_get_redis, tmp_path
    ):
        """Test async processing with non-existent campaign."""
        from gm_agent.storage.campaign import campaign_store

        mock_redis = Mock()
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        with patch("gm_agent.tasks.campaign_store", campaign_store):
            mock_self = Mock()
            mock_self.request.id = "task-123"

            result = process_player_chat_async(
                mock_self, "nonexistent-campaign", "Test input"
            )

        assert result["success"] is False
        assert "not found" in result["error"]

    @patch("gm_agent.tasks.get_redis_client")
    @patch("gm_agent.tasks._publish_task_progress")
    def test_process_player_chat_async_no_session(
        self, mock_publish, mock_get_redis, tmp_path, campaign
    ):
        """Test async processing with no active session."""
        from gm_agent.storage.campaign import campaign_store
        from gm_agent.storage.session import session_store

        mock_redis = Mock()
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        with patch("gm_agent.tasks.campaign_store", campaign_store):
            with patch("gm_agent.tasks.session_store", session_store):
                mock_self = Mock()
                mock_self.request.id = "task-123"

                result = process_player_chat_async(mock_self, campaign.id, "Test input")

        assert result["success"] is False
        assert "No active session" in result["error"]

    @patch("gm_agent.tasks.get_redis_client")
    def test_process_player_chat_async_lock_held(self, mock_get_redis):
        """Test async processing when campaign lock is held."""
        from gm_agent.storage.campaign import campaign_store

        # Mock lock that can't be acquired
        mock_redis = Mock()
        mock_lock = Mock()
        mock_lock.acquire.return_value = False
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        mock_self = Mock()
        mock_self.request.id = "task-123"
        mock_self.retry.side_effect = Exception("Retrying")

        with patch("gm_agent.tasks.campaign_store", campaign_store):
            with pytest.raises(Exception, match="Retrying"):
                process_player_chat_async(mock_self, "test-campaign", "Test input")

        # Should attempt retry (called twice: once for lock, once for exception handling)
        assert mock_self.retry.call_count >= 1


class TestNpcTurnAsync:
    """Tests for async NPC turn processing."""

    @pytest.fixture
    def campaign(self, tmp_path):
        """Create test campaign."""
        from gm_agent.storage.campaign import campaign_store
        import uuid

        with patch("gm_agent.storage.campaign.CAMPAIGNS_DIR", tmp_path):
            unique_name = f"test-campaign-{uuid.uuid4().hex[:8]}"
            campaign = campaign_store.create(unique_name, "Test Campaign")
            yield campaign

    @pytest.fixture
    def session(self, tmp_path, campaign):
        """Create test session."""
        from gm_agent.storage.session import session_store

        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            session = session_store.start(campaign.id)
            yield session

    @patch("gm_agent.tasks.get_redis_client")
    @patch("gm_agent.tasks._publish_task_progress")
    def test_process_npc_turn_async_success(
        self, mock_publish, mock_get_redis, tmp_path, campaign, session
    ):
        """Test successful async NPC turn processing."""
        from gm_agent.storage.campaign import campaign_store
        from gm_agent.storage.session import session_store

        mock_redis = Mock()
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        with patch("gm_agent.agent.GMAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent.process_turn.return_value = "NPC acts dramatically!"
            mock_agent_class.return_value = mock_agent

            with patch("gm_agent.tasks.campaign_store", campaign_store):
                with patch("gm_agent.tasks.session_store", session_store):
                    mock_self = Mock()
                    mock_self.request.id = "task-123"

                    result = process_npc_turn_async(
                        mock_self, campaign.id, "Goblin Warrior"
                    )

        assert result["success"] is True
        assert result["response"] == "NPC acts dramatically!"
        assert result["npc_name"] == "Goblin Warrior"

        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()
        mock_agent.process_turn.assert_called_once()
        mock_agent.close.assert_called_once()

    @patch("gm_agent.tasks.get_redis_client")
    @patch("gm_agent.tasks._publish_task_progress")
    def test_process_npc_turn_async_custom_prompt(
        self, mock_publish, mock_get_redis, tmp_path, campaign, session
    ):
        """Test async NPC turn with custom prompt."""
        from gm_agent.storage.campaign import campaign_store
        from gm_agent.storage.session import session_store

        mock_redis = Mock()
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock
        mock_get_redis.return_value = mock_redis

        custom_prompt = "Custom combat prompt for NPC"

        with patch("gm_agent.agent.GMAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent.process_turn.return_value = "Custom response"
            mock_agent_class.return_value = mock_agent

            with patch("gm_agent.tasks.campaign_store", campaign_store):
                with patch("gm_agent.tasks.session_store", session_store):
                    mock_self = Mock()
                    mock_self.request.id = "task-123"

                    result = process_npc_turn_async(
                        mock_self, campaign.id, "Goblin", prompt=custom_prompt
                    )

        # Verify custom prompt was used
        call_args = mock_agent.process_turn.call_args[0]
        assert call_args[0] == custom_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
