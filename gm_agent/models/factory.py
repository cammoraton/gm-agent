"""Backend factory for LLM backends."""

from typing import Type

from ..config import (
    LLM_BACKEND,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    RETRIEVAL_BACKEND,
    RETRIEVAL_MODEL,
)
from .base import LLMBackend


def _get_backend_class(name: str) -> Type[LLMBackend]:
    """Get the backend class for a given name.

    Uses lazy imports to avoid loading unnecessary dependencies.
    """
    if name == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend
    elif name == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend
    elif name == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend
    elif name == "openrouter":
        from .openai import OpenAIBackend

        return OpenAIBackend
    else:
        raise ValueError(f"Unknown backend: {name}. Available: {list_backends()}")


def get_backend(name: str | None = None) -> LLMBackend:
    """Get an LLM backend instance by name.

    Args:
        name: Backend name (ollama, openai, anthropic, openrouter).
              Defaults to LLM_BACKEND env var.

    Returns:
        Configured LLMBackend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    name = name or LLM_BACKEND

    backend_class = _get_backend_class(name)

    # Special handling for OpenRouter (uses OpenAI SDK with different config)
    if name == "openrouter":
        return backend_class(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    return backend_class()


def _get_backend_with_model(name: str, model: str | None) -> LLMBackend:
    """Get a backend instance, optionally overriding the model.

    Args:
        name: Backend name (ollama, openai, anthropic, openrouter).
        model: Optional model override. None uses the backend's default.

    Returns:
        Configured LLMBackend instance.
    """
    backend_class = _get_backend_class(name)

    if name == "openrouter":
        return backend_class(
            model=model or OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    if model:
        return backend_class(model=model)
    return backend_class()


def get_retrieval_backend() -> LLMBackend:
    """Get backend for retrieval phase (tool selection) in split mode."""
    name = RETRIEVAL_BACKEND or LLM_BACKEND
    model = RETRIEVAL_MODEL
    if name == "openrouter" and not model:
        model = OPENROUTER_MODEL
    return _get_backend_with_model(name, model)


def list_backends() -> list[str]:
    """List available backend names."""
    return ["ollama", "openai", "anthropic", "openrouter"]
