"""Unit tests for backend factory."""

from unittest.mock import patch

import pytest

from gm_agent.models.base import LLMBackend


class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_list_backends(self):
        """list_backends should return all available backends."""
        from gm_agent.models.factory import list_backends

        backends = list_backends()

        assert isinstance(backends, list)
        assert "ollama" in backends
        assert "openai" in backends
        assert "anthropic" in backends
        assert "openrouter" in backends
        assert len(backends) == 4

    def test_get_backend_ollama(self):
        """get_backend should return OllamaBackend for 'ollama'."""
        from gm_agent.models.factory import get_backend
        from gm_agent.models.ollama import OllamaBackend

        backend = get_backend("ollama")

        assert isinstance(backend, OllamaBackend)
        assert isinstance(backend, LLMBackend)

    def test_get_backend_openai(self):
        """get_backend should return OpenAIBackend for 'openai'."""
        from gm_agent.models.factory import get_backend
        from gm_agent.models.openai import OpenAIBackend

        backend = get_backend("openai")

        assert isinstance(backend, OpenAIBackend)
        assert isinstance(backend, LLMBackend)

    def test_get_backend_anthropic(self):
        """get_backend should return AnthropicBackend for 'anthropic'."""
        from gm_agent.models.factory import get_backend
        from gm_agent.models.anthropic import AnthropicBackend

        backend = get_backend("anthropic")

        assert isinstance(backend, AnthropicBackend)
        assert isinstance(backend, LLMBackend)

    def test_get_backend_openrouter(self):
        """get_backend should return OpenAIBackend with OpenRouter config for 'openrouter'."""
        from gm_agent.models.factory import get_backend
        from gm_agent.models.openai import OpenAIBackend

        backend = get_backend("openrouter")

        assert isinstance(backend, OpenAIBackend)
        # OpenRouter uses OpenAI SDK with different base URL
        assert "openrouter.ai" in backend._base_url

    def test_get_backend_invalid(self):
        """get_backend should raise ValueError for unknown backend."""
        from gm_agent.models.factory import get_backend

        with pytest.raises(ValueError) as excinfo:
            get_backend("nonexistent")

        assert "Unknown backend" in str(excinfo.value)
        assert "nonexistent" in str(excinfo.value)

    def test_get_backend_default_from_env(self):
        """get_backend should use LLM_BACKEND env var when name is None."""
        from gm_agent.models.factory import get_backend
        from gm_agent.models.ollama import OllamaBackend

        with patch("gm_agent.models.factory.LLM_BACKEND", "ollama"):
            backend = get_backend(None)
            assert isinstance(backend, OllamaBackend)

    def test_get_backend_default_openai(self):
        """get_backend should use LLM_BACKEND env var for openai."""
        from gm_agent.models.factory import get_backend
        from gm_agent.models.openai import OpenAIBackend

        with patch("gm_agent.models.factory.LLM_BACKEND", "openai"):
            backend = get_backend()
            assert isinstance(backend, OpenAIBackend)

    def test_openrouter_config(self):
        """OpenRouter backend should use correct config."""
        with patch("gm_agent.models.factory.OPENROUTER_API_KEY", "sk-or-test"):
            with patch("gm_agent.models.factory.OPENROUTER_MODEL", "anthropic/claude-3-haiku"):
                from gm_agent.models.factory import get_backend

                backend = get_backend("openrouter")

                assert backend._api_key == "sk-or-test"
                assert backend.model == "anthropic/claude-3-haiku"
                assert "openrouter.ai" in backend._base_url


class TestBackendInterface:
    """Tests that all backends implement the required interface."""

    @pytest.mark.parametrize("backend_name", ["ollama", "openai", "anthropic"])
    def test_backend_has_required_methods(self, backend_name):
        """All backends should have required abstract methods."""
        from gm_agent.models.factory import get_backend

        backend = get_backend(backend_name)

        # Check required methods exist
        assert hasattr(backend, "chat")
        assert hasattr(backend, "get_model_name")
        assert hasattr(backend, "is_available")

        # Check they're callable
        assert callable(backend.chat)
        assert callable(backend.get_model_name)
        assert callable(backend.is_available)

    @pytest.mark.parametrize("backend_name", ["ollama", "openai", "anthropic"])
    def test_get_model_name_returns_string(self, backend_name):
        """All backends should return a string model name."""
        from gm_agent.models.factory import get_backend

        backend = get_backend(backend_name)
        model_name = backend.get_model_name()

        assert isinstance(model_name, str)
        assert len(model_name) > 0

    @pytest.mark.parametrize("backend_name", ["ollama", "openai", "anthropic"])
    def test_is_available_returns_bool(self, backend_name):
        """All backends should return a boolean for is_available."""
        from gm_agent.models.factory import get_backend

        backend = get_backend(backend_name)
        available = backend.is_available()

        assert isinstance(available, bool)


class TestModuleExports:
    """Tests for module-level exports."""

    def test_factory_exports_from_models_init(self):
        """Factory functions should be importable from models package."""
        from gm_agent.models import get_backend, list_backends

        assert callable(get_backend)
        assert callable(list_backends)

    def test_all_backends_importable(self):
        """All backend classes should be importable from models package."""
        from gm_agent.models import (
            LLMBackend,
            OllamaBackend,
            OpenAIBackend,
            AnthropicBackend,
        )

        assert LLMBackend is not None
        assert OllamaBackend is not None
        assert OpenAIBackend is not None
        assert AnthropicBackend is not None

    def test_error_class_importable(self):
        """LLMUnavailableError should be importable from models package."""
        from gm_agent.models import LLMUnavailableError

        assert LLMUnavailableError is not None
        assert issubclass(LLMUnavailableError, Exception)
