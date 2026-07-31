"""pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from core.config.settings import reset_settings


@pytest.fixture(autouse=True)
def reset_settings_singleton() -> None:
    """Reset the Settings singleton before every test for isolation."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture()
def sample_pdf_path() -> str:
    """Return path to a fixture PDF for ingestion tests."""
    return "tests/fixtures/sample_legal.pdf"
