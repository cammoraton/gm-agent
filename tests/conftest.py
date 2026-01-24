"""Root pytest configuration for GM Agent tests."""

import pytest


def pytest_addoption(parser):
    """Add --run-integration option to pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests against real Ollama server",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires --run-integration)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is provided."""
    if config.getoption("--run-integration"):
        # --run-integration given, don't skip
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
