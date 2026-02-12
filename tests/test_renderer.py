"""Tests for spriteforge.renderer — grid-to-PNG rendering."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)
from spriteforge.palette import build_palette_map
from spriteforge.renderer import (
    frame_to_png_bytes,
    render_frame,
    render_row_strip,
    render_spritesheet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(symbol: str, width: int = 64, height: int = 64) -> list[str]:
    """Create a uniform grid filled with a single symbol."""
    return [symbol * width for _ in range(height)]


# ---------------------------------------------------------------------------
# render_frame
# ---------------------------------------------------------------------------


class TestRenderFrame:
    """Tests for the render_frame function."""

    def test_render_frame_basic(self, simple_palette: PaletteConfig) -> None:
        """64×64 grid of all '.' → 64×64 fully transparent image."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid(".")
        img = render_frame(grid, palette_map)

        assert img.mode == "RGBA"
        assert img.size == (64, 64)
        # Every pixel should be fully transparent
        for pixel in img.getdata():
            assert pixel == (0, 0, 0, 0)

    def test_render_frame_single_color(self, simple_palette: PaletteConfig) -> None:
        """64×64 grid of all 'O' → 64×64 image with outline color everywhere."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid("O")
        img = render_frame(grid, palette_map)

        assert img.size == (64, 64)
        outline_rgba = simple_palette.outline.rgba
        for pixel in img.getdata():
            assert pixel == outline_rgba

    def test_render_frame_mixed(self, simple_palette: PaletteConfig) -> None:
        """Grid with '.', 'O', 's' → correct pixels at correct positions."""
        palette_map = build_palette_map(simple_palette)
        # Build a grid: first row starts with ".Os", rest are all "."
        row0 = ".Os" + "." * 61
        grid = [row0] + ["." * 64 for _ in range(63)]
        img = render_frame(grid, palette_map)

        assert img.getpixel((0, 0)) == (0, 0, 0, 0)  # "."
        assert img.getpixel((1, 0)) == palette_map["O"]  # "O"
        assert img.getpixel((2, 0)) == palette_map["s"]  # "s"

    def test_render_frame_pixel_accuracy(self, simple_palette: PaletteConfig) -> None:
        """Verify specific pixel at (10, 20) matches symbol at row 20, col 10."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid(".")
        # Place 'h' at column 10, row 20
        row20 = "." * 10 + "h" + "." * 53
        grid[20] = row20
        img = render_frame(grid, palette_map)

        assert img.getpixel((10, 20)) == palette_map["h"]
        # Surrounding pixels should still be transparent
        assert img.getpixel((9, 20)) == (0, 0, 0, 0)
        assert img.getpixel((11, 20)) == (0, 0, 0, 0)

    def test_render_frame_wrong_dimensions(self, simple_palette: PaletteConfig) -> None:
        """32×32 grid → ValueError."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid(".", width=32, height=32)
        with pytest.raises(ValueError, match="64 rows"):
            render_frame(grid, palette_map)

    def test_render_frame_wrong_row_length(self, simple_palette: PaletteConfig) -> None:
        """Grid with wrong row length → ValueError."""
        palette_map = build_palette_map(simple_palette)
        grid = ["." * 32] + ["." * 64 for _ in range(63)]
        with pytest.raises(ValueError, match="64 characters"):
            render_frame(grid, palette_map)

    def test_render_frame_unknown_symbol(self, simple_palette: PaletteConfig) -> None:
        """Grid with 'X' not in palette → KeyError."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid(".")
        grid[0] = "X" + "." * 63
        with pytest.raises(KeyError, match="Unknown palette symbol"):
            render_frame(grid, palette_map)


# ---------------------------------------------------------------------------
# render_row_strip
# ---------------------------------------------------------------------------


class TestRenderRowStrip:
    """Tests for the render_row_strip function."""

    def test_render_row_strip_dimensions(self, simple_palette: PaletteConfig) -> None:
        """6 frames → strip is (896, 64) with padding."""
        palette_map = build_palette_map(simple_palette)
        frames = [_make_grid(".") for _ in range(6)]
        strip = render_row_strip(frames, palette_map, spritesheet_columns=14)

        assert strip.size == (14 * 64, 64)

    def test_render_row_strip_frame_placement(
        self, simple_palette: PaletteConfig
    ) -> None:
        """Frame N starts at x = N * 64."""
        palette_map = build_palette_map(simple_palette)
        # Frame 0: all transparent, Frame 1: all outline
        frame0 = _make_grid(".")
        frame1 = _make_grid("O")
        strip = render_row_strip([frame0, frame1], palette_map, spritesheet_columns=14)

        # Pixel at (0, 0) should be from frame0 → transparent
        assert strip.getpixel((0, 0)) == (0, 0, 0, 0)
        # Pixel at (64, 0) should be from frame1 → outline color
        assert strip.getpixel((64, 0)) == palette_map["O"]
        # Pixel at (63, 0) should still be from frame0 → transparent
        assert strip.getpixel((63, 0)) == (0, 0, 0, 0)

    def test_render_row_strip_padding_transparent(
        self, simple_palette: PaletteConfig
    ) -> None:
        """Unused columns are fully transparent."""
        palette_map = build_palette_map(simple_palette)
        frames = [_make_grid("O")]  # Only 1 frame
        strip = render_row_strip(frames, palette_map, spritesheet_columns=14)

        # Frame 0 area should be outline color
        assert strip.getpixel((0, 0)) == palette_map["O"]
        # Padding area (frame 1 onwards) should be transparent
        assert strip.getpixel((64, 0)) == (0, 0, 0, 0)
        assert strip.getpixel((895, 63)) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# render_spritesheet
# ---------------------------------------------------------------------------


class TestRenderSpritesheet:
    """Tests for the render_spritesheet function."""

    @pytest.fixture()
    def two_row_spec(self) -> SpritesheetSpec:
        return SpritesheetSpec(
            character=CharacterConfig(
                name="Hero",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=6, timing_ms=150),
                AnimationDef(name="walk", row=1, frames=8, timing_ms=100),
            ],
        )

    def test_render_spritesheet_dimensions(
        self,
        simple_palette: PaletteConfig,
        two_row_spec: SpritesheetSpec,
    ) -> None:
        """Full spec → correct total dimensions."""
        palette_map = build_palette_map(simple_palette)
        all_rows = {
            0: [_make_grid(".") for _ in range(6)],
            1: [_make_grid(".") for _ in range(8)],
        }
        sheet = render_spritesheet(all_rows, two_row_spec, palette_map)

        assert sheet.mode == "RGBA"
        assert sheet.size == (14 * 64, 2 * 64)

    def test_render_spritesheet_missing_row(
        self,
        simple_palette: PaletteConfig,
        two_row_spec: SpritesheetSpec,
    ) -> None:
        """Missing row 1 → ValueError."""
        palette_map = build_palette_map(simple_palette)
        all_rows = {
            0: [_make_grid(".") for _ in range(6)],
        }
        with pytest.raises(ValueError, match="Missing row 1"):
            render_spritesheet(all_rows, two_row_spec, palette_map)

    def test_render_spritesheet_row_placement(
        self,
        simple_palette: PaletteConfig,
        two_row_spec: SpritesheetSpec,
    ) -> None:
        """Rows are placed at correct vertical positions."""
        palette_map = build_palette_map(simple_palette)
        all_rows = {
            0: [_make_grid("O") for _ in range(6)],  # Outline color
            1: [_make_grid("s") for _ in range(8)],  # Skin color
        }
        sheet = render_spritesheet(all_rows, two_row_spec, palette_map)

        # Row 0, frame 0 area → outline color
        assert sheet.getpixel((0, 0)) == palette_map["O"]
        # Row 1, frame 0 area → skin color
        assert sheet.getpixel((0, 64)) == palette_map["s"]


# ---------------------------------------------------------------------------
# frame_to_png_bytes
# ---------------------------------------------------------------------------


class TestFrameToPngBytes:
    """Tests for the frame_to_png_bytes function."""

    def test_frame_to_png_bytes_roundtrip(self, simple_palette: PaletteConfig) -> None:
        """Render → bytes → reopen → identical pixels."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid(".")
        grid[5] = "." * 5 + "O" + "." * 58
        original = render_frame(grid, palette_map)

        png_bytes = frame_to_png_bytes(original)
        reopened = Image.open(io.BytesIO(png_bytes))

        assert reopened.mode == "RGBA"
        assert reopened.size == (64, 64)
        assert list(reopened.getdata()) == list(original.getdata())

    def test_frame_to_png_bytes_valid_png(self, simple_palette: PaletteConfig) -> None:
        """Output bytes start with PNG magic number."""
        palette_map = build_palette_map(simple_palette)
        grid = _make_grid(".")
        png_bytes = frame_to_png_bytes(render_frame(grid, palette_map))

        # PNG files start with these 8 bytes
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
