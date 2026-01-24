"""Tests for Ollama LLM backend."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from gm_agent.models import LLMUnavailableError, OllamaBackend
from gm_agent.models.base import Message


class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        with patch("gm_agent.models.ollama.ollama.Client"):
            backend = OllamaBackend()
            assert backend.max_retries == 3
            assert backend.retry_delay == 1.0
            assert backend._available is True

    def test_init_custom_retries(self):
        """Test initialization with custom retry settings."""
        with patch("gm_agent.models.ollama.ollama.Client"):
            backend = OllamaBackend(max_retries=5, retry_delay=2.0)
            assert backend.max_retries == 5
            assert backend.retry_delay == 2.0

    def test_is_available_success(self):
        """Test is_available returns True when server responds."""
        with patch("gm_agent.models.ollama.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": []}
            mock_client_class.return_value = mock_client

            backend = OllamaBackend()
            assert backend.is_available() is True
            assert backend._available is True

    def test_is_available_connection_error(self):
        """Test is_available returns False on connection error."""
        with patch("gm_agent.models.ollama.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.list.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            backend = OllamaBackend()
            assert backend.is_available() is False
            assert backend._available is False

    def test_chat_success(self):
        """Test successful chat call."""
        with patch("gm_agent.models.ollama.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.return_value = {
                "message": {"content": "Hello!"},
                "done": True,
            }
            mock_client_class.return_value = mock_client

            backend = OllamaBackend()
            messages = [Message(role="user", content="Hi")]
            response = backend.chat(messages)

            assert response.text == "Hello!"
            assert backend._available is True

    def test_chat_retry_on_connection_error(self):
        """Test chat retries on connection error."""
        with patch("gm_agent.models.ollama.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            # Fail twice, succeed on third attempt
            mock_client.chat.side_effect = [
                httpx.ConnectError("Connection refused"),
                httpx.ConnectError("Connection refused"),
                {"message": {"content": "Success!"}, "done": True},
            ]
            mock_client_class.return_value = mock_client

            backend = OllamaBackend(max_retries=3, retry_delay=0.01)
            messages = [Message(role="user", content="Hi")]
            response = backend.chat(messages)

            assert response.text == "Success!"
            assert mock_client.chat.call_count == 3

    def test_chat_raises_after_max_retries(self):
        """Test chat raises LLMUnavailableError after max retries."""
        with patch("gm_agent.models.ollama.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            backend = OllamaBackend(max_retries=3, retry_delay=0.01)
            messages = [Message(role="user", content="Hi")]

            with pytest.raises(LLMUnavailableError) as exc_info:
                backend.chat(messages)

            assert "3 attempts" in str(exc_info.value)
            assert mock_client.chat.call_count == 3
            assert backend._available is False

    def test_chat_timeout_error_triggers_retry(self):
        """Test that timeout errors also trigger retry."""
        with patch("gm_agent.models.ollama.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.side_effect = [
                httpx.TimeoutException("Timeout"),
                {"message": {"content": "Success!"}, "done": True},
            ]
            mock_client_class.return_value = mock_client

            backend = OllamaBackend(max_retries=3, retry_delay=0.01)
            messages = [Message(role="user", content="Hi")]
            response = backend.chat(messages)

            assert response.text == "Success!"
            assert mock_client.chat.call_count == 2
