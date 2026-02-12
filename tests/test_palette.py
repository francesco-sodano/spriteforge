"""Tests for spriteforge.palette — symbol-to-RGBA mapping and palette swapping."""

from __future__ import annotations

import pytest

from spriteforge.models import PaletteColor, PaletteConfig
from spriteforge.palette import (
    DRUNN_OUTLINE_RGBA,
    DRUNN_P1_COLORS,
    SYLARA_OUTLINE_RGBA,
    SYLARA_P1_COLORS,
    THERON_OUTLINE_RGBA,
    THERON_P1_COLORS,
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


# ---------------------------------------------------------------------------
# Default palette constants
# ---------------------------------------------------------------------------


class TestSylaraP1Constants:
    """Tests for the SYLARA_P1_COLORS constant."""

    def test_sylara_p1_constants_correct(self) -> None:
        # Verify the constant matches the spritesheet spec from
        # docs_assets/spritesheet_instructions_sylara.md
        names = [name for name, _sym, _rgba in SYLARA_P1_COLORS]
        assert "Skin" in names
        assert "Hair" in names
        assert "Eyes" in names

    def test_sylara_p1_has_ten_colors(self) -> None:
        assert len(SYLARA_P1_COLORS) == 10

    def test_sylara_p1_skin_values(self) -> None:
        skin = next(c for c in SYLARA_P1_COLORS if c[0] == "Skin")
        assert skin[1] == "s"
        assert skin[2] == (235, 210, 185, 255)

    def test_sylara_p1_all_symbols_unique(self) -> None:
        symbols = [sym for _name, sym, _rgba in SYLARA_P1_COLORS]
        assert len(symbols) == len(set(symbols))


# ---------------------------------------------------------------------------
# Theron P1 palette constants
# ---------------------------------------------------------------------------


class TestTheronP1Constants:
    """Tests for the THERON_P1_COLORS constant.

    Source of truth: docs_assets/spritesheet_instructions_theron.md
    """

    def test_theron_p1_has_eight_colors(self) -> None:
        assert len(THERON_P1_COLORS) == 8

    def test_theron_p1_skin_values(self) -> None:
        skin = next(c for c in THERON_P1_COLORS if c[0] == "Skin")
        assert skin[1] == "s"
        assert skin[2] == (210, 170, 130, 255)

    def test_theron_p1_hair_values(self) -> None:
        hair = next(c for c in THERON_P1_COLORS if c[0] == "Hair")
        assert hair[1] == "h"
        assert hair[2] == (60, 40, 25, 255)

    def test_theron_p1_all_symbols_unique(self) -> None:
        symbols = [sym for _name, sym, _rgba in THERON_P1_COLORS]
        assert len(symbols) == len(set(symbols))

    def test_theron_p1_all_names_present(self) -> None:
        names = {name for name, _sym, _rgba in THERON_P1_COLORS}
        expected = {
            "Skin",
            "Hair",
            "Breastplate",
            "Tunic / Cloak",
            "Leather",
            "Steel trim",
            "Emberfang blade",
            "Boots",
        }
        assert names == expected

    def test_theron_outline_rgba(self) -> None:
        # "dark brown or black" per spritesheet_instructions_theron.md
        assert len(THERON_OUTLINE_RGBA) == 4
        assert THERON_OUTLINE_RGBA[3] == 255  # fully opaque


# ---------------------------------------------------------------------------
# Drunn P1 palette constants
# ---------------------------------------------------------------------------


class TestDrunnP1Constants:
    """Tests for the DRUNN_P1_COLORS constant.

    Source of truth: docs_assets/spritesheet_instructions_drunn.md
    """

    def test_drunn_p1_has_ten_colors(self) -> None:
        assert len(DRUNN_P1_COLORS) == 10

    def test_drunn_p1_skin_values(self) -> None:
        skin = next(c for c in DRUNN_P1_COLORS if c[0] == "Skin")
        assert skin[1] == "s"
        assert skin[2] == (190, 145, 110, 255)

    def test_drunn_p1_beard_values(self) -> None:
        beard = next(c for c in DRUNN_P1_COLORS if c[0] == "Beard/hair")
        assert beard[1] == "h"
        assert beard[2] == (180, 70, 20, 255)

    def test_drunn_p1_all_symbols_unique(self) -> None:
        symbols = [sym for _name, sym, _rgba in DRUNN_P1_COLORS]
        assert len(symbols) == len(set(symbols))

    def test_drunn_p1_all_names_present(self) -> None:
        names = {name for name, _sym, _rgba in DRUNN_P1_COLORS}
        expected = {
            "Skin",
            "Beard/hair",
            "Helm/breastplate",
            "Pauldrons",
            "Chainmail",
            "Leather belt/pants",
            "Boots",
            "Axe heads (steel)",
            "Axe hafts (wood)",
            "Red accent",
        }
        assert names == expected

    def test_drunn_outline_rgba(self) -> None:
        # "dark brown or black" per spritesheet_instructions_drunn.md
        assert len(DRUNN_OUTLINE_RGBA) == 4
        assert DRUNN_OUTLINE_RGBA[3] == 255  # fully opaque


# ---------------------------------------------------------------------------
# Outline RGBA constants
# ---------------------------------------------------------------------------


class TestOutlineConstants:
    """Tests for per-character outline RGBA constants."""

    def test_sylara_outline_is_dark_teal(self) -> None:
        # "dark teal or black" per spritesheet_instructions_sylara.md
        assert SYLARA_OUTLINE_RGBA == (0, 80, 80, 255)

    def test_all_outlines_fully_opaque(self) -> None:
        assert SYLARA_OUTLINE_RGBA[3] == 255
        assert THERON_OUTLINE_RGBA[3] == 255
        assert DRUNN_OUTLINE_RGBA[3] == 255
