"""Shared fixtures for spriteforge tests."""

from __future__ import annotations

import pytest

from spriteforge.models import (
    PaletteColor,
    PaletteConfig,
)


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
