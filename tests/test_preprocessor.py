"""Tests for spriteforge.preprocessor — image resize, quantize, and auto-palette extraction."""

from __future__ import annotations

import io
import random
from pathlib import Path

import pytest
from PIL import Image

from spriteforge.models import PaletteConfig
from spriteforge.preprocessor import (
    SYMBOL_POOL,
    PreprocessResult,
    _assign_symbols,
    extract_palette_from_image,
    preprocess_reference,
    resize_reference,
    validate_reference_image,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_rgba(img: Image.Image, path: Path) -> None:
    """Save an RGBA image as PNG."""
    img.save(str(path), format="PNG")


def _make_multicolor_image(
    width: int = 128,
    height: int = 128,
    num_colors: int = 50,
    seed: int = 42,
) -> Image.Image:
    """Create a test image with many distinct opaque colors and a transparent bg."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    rng = random.Random(seed)
    colors = [
        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 255)
        for _ in range(num_colors)
    ]
    for x in range(width):
        for y in range(height // 2):
            img.putpixel((x, y), colors[(x + y) % num_colors])
    return img


def _make_few_color_image(
    width: int = 64,
    height: int = 64,
    num_colors: int = 8,
) -> Image.Image:
    """Create a test image with exactly num_colors distinct opaque colors."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    colors = [(i * 30, 50 + i * 20, 100 + i * 15, 255) for i in range(num_colors)]
    stripe_w = max(1, width // num_colors)
    for x in range(width):
        for y in range(height // 2):
            idx = min(x // stripe_w, num_colors - 1)
            img.putpixel((x, y), colors[idx])
    return img


# ---------------------------------------------------------------------------
# validate_reference_image
# ---------------------------------------------------------------------------


class TestValidateReferenceImage:
    """Tests for the validate_reference_image function."""

    def test_validate_reference_image_valid(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        path = tmp_path / "test.png"
        _save_rgba(img, path)
        result = validate_reference_image(path)
        assert result.mode == "RGBA"
        assert result.size == (128, 128)

    def test_validate_reference_image_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_reference_image("/nonexistent/path/image.png")

    def test_validate_reference_image_too_small(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        path = tmp_path / "tiny.png"
        _save_rgba(img, path)
        with pytest.raises(ValueError, match="too small"):
            validate_reference_image(path)

    def test_validate_reference_image_extreme_aspect(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (640, 64), (255, 0, 0, 255))
        path = tmp_path / "wide.png"
        _save_rgba(img, path)
        with pytest.raises(ValueError, match="aspect ratio"):
            validate_reference_image(path)

    def test_validate_reference_converts_to_rgba(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (128, 128), (255, 0, 0))
        path = tmp_path / "rgb.png"
        img.save(str(path), format="PNG")
        result = validate_reference_image(path)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# resize_reference
# ---------------------------------------------------------------------------


class TestResizeReference:
    """Tests for the resize_reference function."""

    def test_resize_reference_to_64x64(self) -> None:
        img = Image.new("RGBA", (512, 512), (255, 0, 0, 255))
        resized = resize_reference(img, 64, 64)
        assert resized.size == (64, 64)

    def test_resize_reference_already_target_size(self) -> None:
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        resized = resize_reference(img, 64, 64)
        assert resized.size == (64, 64)

    def test_resize_reference_uses_nearest_neighbor(self) -> None:
        """Verify nearest-neighbor produces hard edges (no anti-aliasing)."""
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        # Draw a sharp boundary: left half red, right half blue
        for x in range(64):
            for y in range(128):
                img.putpixel((x, y), (255, 0, 0, 255))
        for x in range(64, 128):
            for y in range(128):
                img.putpixel((x, y), (0, 0, 255, 255))

        resized = resize_reference(img, 64, 64)
        # With nearest-neighbor, mid-boundary pixel should be exactly
        # one of the two colors, NOT a blend
        mid_pixel = resized.getpixel((31, 32))
        assert mid_pixel in ((255, 0, 0, 255), (0, 0, 255, 255))


# ---------------------------------------------------------------------------
# extract_palette_from_image
# ---------------------------------------------------------------------------


class TestExtractPaletteFromImage:
    """Tests for the extract_palette_from_image function."""

    def test_extract_palette_max_colors(self) -> None:
        img = _make_multicolor_image(num_colors=50)
        palette = extract_palette_from_image(img, max_colors=16)
        # 1 outline + up to 15 regular colors ≤ 16 total opaque
        total_opaque = 1 + len(palette.colors)  # outline + colors
        assert total_opaque <= 16

    def test_extract_palette_already_few_colors(self) -> None:
        img = _make_few_color_image(num_colors=8)
        palette = extract_palette_from_image(img, max_colors=16)
        # Should NOT over-quantize: all 8 colors present
        total_opaque = 1 + len(palette.colors)
        assert total_opaque == 8

    def test_extract_palette_outline_is_darkest(self) -> None:
        img = _make_multicolor_image(num_colors=10)
        palette = extract_palette_from_image(img)
        outline_lum = (
            0.299 * palette.outline.r
            + 0.587 * palette.outline.g
            + 0.114 * palette.outline.b
        )
        for color in palette.colors:
            color_lum = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
            assert outline_lum <= color_lum + 1  # Allow small rounding

    def test_extract_palette_no_duplicate_symbols(self) -> None:
        img = _make_multicolor_image(num_colors=20)
        palette = extract_palette_from_image(img, max_colors=16)
        symbols = [palette.transparent_symbol, palette.outline.symbol]
        symbols.extend(c.symbol for c in palette.colors)
        assert len(symbols) == len(set(symbols))

    def test_extract_palette_transparent_included(self) -> None:
        img = _make_multicolor_image(num_colors=10)
        palette = extract_palette_from_image(img)
        assert palette.transparent_symbol == "."

    def test_extract_palette_custom_max_colors(self) -> None:
        img = _make_multicolor_image(num_colors=50)
        palette = extract_palette_from_image(img, max_colors=10)
        total_opaque = 1 + len(palette.colors)
        assert total_opaque <= 10


# ---------------------------------------------------------------------------
# preprocess_reference (full pipeline)
# ---------------------------------------------------------------------------


class TestPreprocessReference:
    """Tests for the preprocess_reference function."""

    def test_preprocess_reference_happy_path(self, tmp_path: Path) -> None:
        img = _make_multicolor_image(width=256, height=256, num_colors=30)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        result = preprocess_reference(str(path))
        assert isinstance(result, PreprocessResult)
        assert result.quantized_image.size == (64, 64)
        assert result.final_color_count <= 16
        assert result.original_color_count > 0

    def test_preprocess_reference_palette_valid(self, tmp_path: Path) -> None:
        img = _make_multicolor_image(width=128, height=128, num_colors=20)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        result = preprocess_reference(str(path))
        # The palette should be a valid PaletteConfig (validated by Pydantic)
        assert isinstance(result.palette, PaletteConfig)
        # No duplicate symbols
        symbols = [result.palette.transparent_symbol, result.palette.outline.symbol]
        symbols.extend(c.symbol for c in result.palette.colors)
        assert len(symbols) == len(set(symbols))

    def test_preprocess_reference_png_bytes_valid(self, tmp_path: Path) -> None:
        img = _make_multicolor_image(width=128, height=128, num_colors=10)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        result = preprocess_reference(str(path))
        # quantized_png_bytes should be openable as a PIL Image
        reopened = Image.open(io.BytesIO(result.quantized_png_bytes))
        assert reopened.size == (64, 64)

    def test_preprocess_reference_preserves_character(self, tmp_path: Path) -> None:
        """Quantized image should have non-zero opaque pixels."""
        img = _make_multicolor_image(width=128, height=128, num_colors=10)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        result = preprocess_reference(str(path))
        raw = result.quantized_image.tobytes()
        alpha_values = list(raw[3::4])
        opaque_count = sum(1 for a in alpha_values if a > 0)
        assert opaque_count > 0

    def test_preprocess_reference_custom_outline(self, tmp_path: Path) -> None:
        img = _make_few_color_image(width=128, height=128, num_colors=5)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        # Force a specific outline color
        result = preprocess_reference(str(path), outline_color=(0, 50, 100))
        # The outline should be the closest match to forced color
        outline = result.palette.outline
        assert outline.symbol == "O"

    def test_preprocess_reference_pixel_art_input(self, tmp_path: Path) -> None:
        """A 64×64 pixel art image with few colors → palette extracted directly."""
        img = _make_few_color_image(width=64, height=64, num_colors=6)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        result = preprocess_reference(str(path))
        assert result.quantized_image.size == (64, 64)
        # Should preserve all 6 colors without quantization loss
        total_opaque = 1 + len(result.palette.colors)
        assert total_opaque == 6


# ---------------------------------------------------------------------------
# _assign_symbols
# ---------------------------------------------------------------------------


class TestAssignSymbols:
    """Tests for the _assign_symbols helper function."""

    def test_assign_symbols_coverage_order(self) -> None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 100)]
        coverage = [100, 500, 200]
        # Darkest by luminance is (0, 0, 100) at index 2
        result = _assign_symbols(colors, coverage, outline_index=2)
        # Outline first
        assert result[0][1] == "O"
        assert result[0][2] == (0, 0, 100)
        # Remaining sorted by coverage: green (500) > red (100)
        assert result[1][1] == SYMBOL_POOL[0]  # "s" (highest coverage)
        assert result[1][2] == (0, 255, 0)
        assert result[2][1] == SYMBOL_POOL[1]  # "h" (second highest)
        assert result[2][2] == (255, 0, 0)

    def test_assign_symbols_outline_excluded_from_pool(self) -> None:
        colors = [(10, 10, 10), (200, 200, 200)]
        coverage = [50, 150]
        result = _assign_symbols(colors, coverage, outline_index=0)
        symbols = [entry[1] for entry in result]
        assert "O" in symbols
        # 'O' only appears once
        assert symbols.count("O") == 1
        # Remaining symbol is from pool
        assert result[1][1] == SYMBOL_POOL[0]

    def test_assign_symbols_max_symbols(self) -> None:
        """16 colors → 16 distinct symbols (1 outline + 15 pool)."""
        colors = [(i * 15, i * 10, i * 5) for i in range(16)]
        coverage = [100 - i for i in range(16)]
        result = _assign_symbols(colors, coverage, outline_index=0)
        symbols = [entry[1] for entry in result]
        assert len(symbols) == 16
        assert len(set(symbols)) == 16
        assert symbols[0] == "O"
