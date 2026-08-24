"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest

from financial_ml.config import load_config, reset_config_cache


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Ensure each test sees a fresh config load.

    ``load_config`` is memoized at module level so the application reads
    YAML once per process. Tests that mutate the on-disk config (none yet,
    but the door is open) need deterministic reloads.
    """
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def app_config():
    """Convenience fixture exposing the cached ``AppConfig``."""
    return load_config()
