"""Shared fixtures for spriteforge tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# Auto-load .env from project root (gitignored — never pushed to GitHub).
# This provides AZURE_AI_PROJECT_ENDPOINT and other env vars for integration tests.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from spriteforge.models import (
    PaletteColor,
    PaletteConfig,
)

# ---------------------------------------------------------------------------
# Auto-skip integration tests when Azure credentials are unavailable
# ---------------------------------------------------------------------------


def _azure_credentials_available() -> bool:
    """Check whether Azure AI Foundry credentials are available.

    Returns True when:
    0. Integration tests are explicitly enabled, AND
    1. The AZURE_AI_PROJECT_ENDPOINT env var is set, AND
    2. DefaultAzureCredential can obtain a token.
    """
    if os.environ.get("SPRITEFORGE_RUN_INTEGRATION", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    if not endpoint:
        return False
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]

        cred = DefaultAzureCredential()
        # Request a token for Azure Cognitive Services scope to validate auth
        cred.get_token("https://cognitiveservices.azure.com/.default")
        return True
    except Exception:  # noqa: BLE001
        return False


# Cache the check once per session so we don't re-auth on every test.
_AZURE_AVAILABLE: bool | None = None


def _is_azure_available() -> bool:
    global _AZURE_AVAILABLE  # noqa: PLW0603
    if _AZURE_AVAILABLE is None:
        _AZURE_AVAILABLE = _azure_credentials_available()
    return _AZURE_AVAILABLE


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip integration tests when Azure is not available."""
    if _is_azure_available():
        return
    skip_marker = pytest.mark.skip(
        reason=(
            "Integration test skipped: set SPRITEFORGE_RUN_INTEGRATION=1 and "
            "ensure AZURE_AI_PROJECT_ENDPOINT is set and DefaultAzureCredential can authenticate."
        )
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Azure fixtures (only used by integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def azure_project_endpoint() -> str:
    """Return the Azure AI Foundry project endpoint from the environment."""
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    if not endpoint:
        pytest.skip("AZURE_AI_PROJECT_ENDPOINT not set")
    return endpoint


@pytest.fixture(scope="session")
def azure_credential() -> Any:
    """Return a DefaultAzureCredential for Azure AI Foundry access."""
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]

        return DefaultAzureCredential()
    except Exception as exc:
        pytest.skip(f"DefaultAzureCredential unavailable: {exc}")


# ---------------------------------------------------------------------------
# Palette fixtures (used by unit tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_palette() -> PaletteConfig:
    """A minimal palette with two named colors for testing."""
    return PaletteConfig(
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
            PaletteColor(element="Hair", symbol="h", r=220, g=185, b=90),
        ],
    )
