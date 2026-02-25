"""Tests for spriteforge.palette — symbol-to-RGBA mapping and palette swapping."""

from __future__ import annotations

import pytest

from spriteforge.models import PaletteColor, PaletteConfig
from spriteforge.palette import (
    build_palette_map,
    swap_palette_grid,
    validate_grid_symbols,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def alt_palette() -> PaletteConfig:
    """An alternate palette with the same color names but different symbols/RGB."""
    return PaletteConfig(
        outline=PaletteColor(element="Outline", symbol="O", r=10, g=10, b=10),
        colors=[
            PaletteColor(element="Skin", symbol="S", r=200, g=170, b=140),
            PaletteColor(element="Hair", symbol="H", r=180, g=150, b=60),
        ],
    )


# ---------------------------------------------------------------------------
# build_palette_map
# ---------------------------------------------------------------------------


class TestBuildPaletteMap:
    """Tests for the build_palette_map function."""

    def test_build_palette_map_includes_all_symbols(
        self, simple_palette: PaletteConfig
    ) -> None:
        mapping = build_palette_map(simple_palette)
        assert "." in mapping  # transparent
        assert "O" in mapping  # outline
        assert "s" in mapping  # Skin
        assert "h" in mapping  # Hair
        assert len(mapping) == 4

    def test_build_palette_map_transparent_is_zero_alpha(
        self, simple_palette: PaletteConfig
    ) -> None:
        mapping = build_palette_map(simple_palette)
        assert mapping["."] == (0, 0, 0, 0)

    def test_build_palette_map_outline_uses_config_rgba(
        self, simple_palette: PaletteConfig
    ) -> None:
        mapping = build_palette_map(simple_palette)
        assert mapping["O"] == (20, 40, 40, 255)

    def test_build_palette_map_color_values(
        self, simple_palette: PaletteConfig
    ) -> None:
        mapping = build_palette_map(simple_palette)
        assert mapping["s"] == (235, 210, 185, 255)
        assert mapping["h"] == (220, 185, 90, 255)


# ---------------------------------------------------------------------------
# validate_grid_symbols
# ---------------------------------------------------------------------------


class TestValidateGridSymbols:
    """Tests for the validate_grid_symbols function."""

    def test_validate_grid_symbols_valid_grid(
        self, simple_palette: PaletteConfig
    ) -> None:
        grid = ["..Os", "sh.."]
        result = validate_grid_symbols(grid, simple_palette)
        assert result == []

    def test_validate_grid_symbols_invalid_symbols(
        self, simple_palette: PaletteConfig
    ) -> None:
        grid = ["..X?", "s..."]
        result = validate_grid_symbols(grid, simple_palette)
        assert sorted(result) == ["?", "X"]

    def test_validate_grid_symbols_empty_grid(
        self, simple_palette: PaletteConfig
    ) -> None:
        result = validate_grid_symbols([], simple_palette)
        assert result == []

    def test_validate_grid_symbols_duplicates_reported_once(
        self, simple_palette: PaletteConfig
    ) -> None:
        grid = ["XXXX"]
        result = validate_grid_symbols(grid, simple_palette)
        assert result == ["X"]


# ---------------------------------------------------------------------------
# swap_palette_grid
# ---------------------------------------------------------------------------


class TestSwapPaletteGrid:
    """Tests for the swap_palette_grid function."""

    def test_swap_palette_grid_changes_symbols(
        self, simple_palette: PaletteConfig, alt_palette: PaletteConfig
    ) -> None:
        grid = ["sh"]
        result = swap_palette_grid(grid, simple_palette, alt_palette)
        assert result == ["SH"]

    def test_swap_palette_grid_preserves_transparent(
        self, simple_palette: PaletteConfig, alt_palette: PaletteConfig
    ) -> None:
        grid = ["..s."]
        result = swap_palette_grid(grid, simple_palette, alt_palette)
        assert result[0][0] == "."
        assert result[0][1] == "."
        assert result[0][3] == "."

    def test_swap_palette_grid_preserves_outline(
        self, simple_palette: PaletteConfig, alt_palette: PaletteConfig
    ) -> None:
        grid = ["OsO"]
        result = swap_palette_grid(grid, simple_palette, alt_palette)
        assert result[0][0] == "O"
        assert result[0][2] == "O"

    def test_swap_palette_grid_mismatched_names(
        self, simple_palette: PaletteConfig
    ) -> None:
        bad_palette = PaletteConfig(
            colors=[
                PaletteColor(element="Eyes", symbol="e", r=0, g=0, b=0),
            ],
        )
        with pytest.raises(ValueError, match="missing color names"):
            swap_palette_grid(["s"], simple_palette, bad_palette)

    def test_swap_palette_grid_full_row(
        self, simple_palette: PaletteConfig, alt_palette: PaletteConfig
    ) -> None:
        grid = ["..OOsshh..", "OOsshh..OO"]
        result = swap_palette_grid(grid, simple_palette, alt_palette)
        assert result == ["..OOSSHH..", "OOSSHH..OO"]

    def test_swap_palette_grid_unknown_symbol_warns(
        self,
        simple_palette: PaletteConfig,
        alt_palette: PaletteConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        result = swap_palette_grid(["s?"], simple_palette, alt_palette)
        assert result == ["S?"]
        assert "unknown symbols" in caplog.text
