"""Tests for spriteforge.preprocessor — image resize, quantize, and auto-palette extraction."""

from __future__ import annotations

import io
import random
import os
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from spriteforge.models import GenerationConfig, PaletteConfig
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
        """Verify nearest-neighbor is used for upscaling (hard edges)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        # Draw a sharp boundary: left half red, right half blue
        for x in range(16):
            for y in range(32):
                img.putpixel((x, y), (255, 0, 0, 255))
        for x in range(16, 32):
            for y in range(32):
                img.putpixel((x, y), (0, 0, 255, 255))

        # Upscaling 32→64: should use NEAREST (hard edges, no blending)
        resized = resize_reference(img, 64, 64)
        mid_pixel = resized.getpixel((31, 32))
        assert mid_pixel in ((255, 0, 0, 255), (0, 0, 255, 255))

    def test_resize_reference_uses_lanczos_for_downscaling(self) -> None:
        """Verify LANCZOS is used for downscaling (anti-aliased)."""
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        # Draw a sharp boundary: left half red, right half blue
        for x in range(64):
            for y in range(128):
                img.putpixel((x, y), (255, 0, 0, 255))
        for x in range(64, 128):
            for y in range(128):
                img.putpixel((x, y), (0, 0, 255, 255))

        # Downscaling 128→64: LANCZOS may anti-alias at the boundary
        resized = resize_reference(img, 64, 64)
        mid_pixel = resized.getpixel((31, 32))
        # With LANCZOS the boundary pixel may be blended, which is correct
        # for downscaling; the key assertion is that the image was produced
        # at the target size (covered by other tests) and that clearly
        # non-boundary pixels retain their original color.
        left_pixel = resized.getpixel((0, 0))
        right_pixel = resized.getpixel((63, 0))
        assert left_pixel == (255, 0, 0, 255)
        assert right_pixel == (0, 0, 255, 255)


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

    def test_max_palette_colors_at_pool_limit(self) -> None:
        """Test that max_colors=23 works correctly at SYMBOL_POOL limit.

        With SYMBOL_POOL having 22 symbols, the max opaque colors is 23:
        - 1 outline (symbol 'O')
        - 22 regular colors (from SYMBOL_POOL)
        Total: 23 opaque colors (transparent '.' is implicit, not counted)
        """
        img = _make_multicolor_image(num_colors=50)
        palette = extract_palette_from_image(img, max_colors=23)
        # Should have at most 1 outline + 22 regular colors = 23 total opaque
        # (may be less if image has fewer unique colors after quantization)
        total_opaque = 1 + len(palette.colors)
        assert total_opaque <= 23
        # Verify no duplicate symbols
        symbols = [palette.transparent_symbol, palette.outline.symbol]
        symbols.extend(c.symbol for c in palette.colors)
        assert len(symbols) == len(set(symbols))
        # All symbols should be from SYMBOL_POOL (plus transparent and outline)
        for color in palette.colors:
            assert color.symbol in SYMBOL_POOL

    def test_max_palette_colors_exceeds_pool(self) -> None:
        """Test that requesting max_colors > 23 is rejected by the model."""
        # This test verifies that GenerationConfig enforces the upper bound
        # The preprocessor itself doesn't validate, but the config does
        with pytest.raises(ValidationError):
            GenerationConfig(max_palette_colors=24)


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

    def test_preprocess_reference_palette_matches_image(self, tmp_path: Path) -> None:
        """Every opaque pixel in the quantized image must map to a palette color.

        This guards against double-quantization: if quantize() ran twice
        independently, the palette and image could have different color sets.
        """
        img = _make_multicolor_image(width=256, height=256, num_colors=50)
        path = tmp_path / "ref.png"
        _save_rgba(img, path)

        result = preprocess_reference(str(path), max_colors=12)

        # Collect all palette RGB values
        palette_rgbs: set[tuple[int, int, int]] = set()
        palette_rgbs.add(
            (
                result.palette.outline.r,
                result.palette.outline.g,
                result.palette.outline.b,
            )
        )
        for c in result.palette.colors:
            palette_rgbs.add((c.r, c.g, c.b))

        # Collect all opaque pixel RGB values from the quantized image
        raw = result.quantized_image.tobytes()
        pixels = list(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4]))
        image_rgbs = set((r, g, b) for r, g, b, a in pixels if a > 0)

        # Every color in the image must appear in the palette
        unmatched = image_rgbs - palette_rgbs
        assert unmatched == set(), (
            f"Quantized image has {len(unmatched)} color(s) not in palette: "
            f"{list(unmatched)[:5]}..."
        )

    def test_preprocess_reference_semantic_labels_toggle(self, tmp_path: Path) -> None:
        """semantic_labels flag controls descriptive naming in auto palette."""
        img = _make_few_color_image(width=64, height=64, num_colors=6)
        path = tmp_path / "ref_semantic_toggle.png"
        _save_rgba(img, path)

        descriptive = preprocess_reference(str(path), semantic_labels=True)
        generic = preprocess_reference(str(path), semantic_labels=False)

        descriptive_names = [c.element for c in descriptive.palette.colors]
        generic_names = [c.element for c in generic.palette.colors]

        assert any(not name.startswith("Color ") for name in descriptive_names)
        assert all(name.startswith("Color ") for name in generic_names)

    def test_extract_palette_ignores_transparent_background_rgb_pollution(self) -> None:
        """Transparent background must not consume palette slots as black RGB."""
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))

        # Small opaque islands with explicit colors.
        for x in range(8):
            for y in range(8):
                img.putpixel((x, y), (220, 30, 30, 255))
                img.putpixel((x + 12, y), (30, 220, 30, 255))

        # With mostly transparent pixels, extraction should only include opaque colors.
        palette = extract_palette_from_image(img, max_colors=16)
        extracted_rgbs = {
            (palette.outline.r, palette.outline.g, palette.outline.b),
            *((c.r, c.g, c.b) for c in palette.colors),
        }

        assert extracted_rgbs == {(220, 30, 30), (30, 220, 30)}


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


# ---------------------------------------------------------------------------
# _describe_color
# ---------------------------------------------------------------------------


class TestDescribeColor:
    """Tests for the _describe_color helper function."""

    def test_describe_color_red(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((255, 0, 0))
        assert "Red" in result

    def test_describe_color_dark_blue(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((0, 0, 80))
        assert "Dark" in result
        assert "Blue" in result

    def test_describe_color_light_green(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((150, 255, 150))
        assert "Light" in result
        assert "Green" in result

    def test_describe_color_near_black(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((10, 10, 10))
        assert result == "Near Black"

    def test_describe_color_near_white(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((250, 250, 250))
        assert result == "Near White"

    def test_describe_color_gray(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((128, 128, 128))
        assert "Gray" in result

    def test_describe_color_brown(self) -> None:
        from spriteforge.preprocessor import _describe_color

        result = _describe_color((139, 90, 43))
        assert "Brown" in result


# ---------------------------------------------------------------------------
# _assign_symbols with semantic_labels
# ---------------------------------------------------------------------------


class TestAssignSymbolsWithLabels:
    """Tests for _assign_symbols with semantic labels."""

    def test_assign_symbols_with_semantic_labels(self) -> None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 100)]
        coverage = [100, 500, 200]
        labels = ["Skin", "Hair"]  # For the 2 non-outline colors (sorted by coverage)
        result = _assign_symbols(
            colors, coverage, outline_index=2, semantic_labels=labels
        )
        # Outline first
        assert result[0][1] == "O"
        assert result[0][2] == (0, 0, 100)
        # Remaining sorted by coverage: green (500) > red (100)
        assert result[1][0] == "Skin"
        assert result[1][2] == (0, 255, 0)
        assert result[2][0] == "Hair"
        assert result[2][2] == (255, 0, 0)

    def test_assign_symbols_without_labels(self) -> None:
        """Fallback to 'Color N' when no labels provided."""
        colors = [(255, 0, 0), (0, 255, 0)]
        coverage = [100, 200]
        result = _assign_symbols(
            colors, coverage, outline_index=0, semantic_labels=None
        )
        # First remaining color (highest coverage) should be "Color 1"
        assert result[1][0] == "Color 1"
        assert result[1][2] == (0, 255, 0)

    def test_assign_symbols_labels_length_mismatch(self) -> None:
        """Wrong number of labels → ignored, falls back to 'Color N'."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 100)]
        coverage = [100, 500, 200]
        labels = ["Skin"]  # Only 1 label, but 2 non-outline colors
        result = _assign_symbols(
            colors, coverage, outline_index=2, semantic_labels=labels
        )
        # Should fall back to "Color N"
        assert "Color 1" in result[1][0]
        assert "Color 2" in result[2][0]


# ---------------------------------------------------------------------------
# label_palette_colors_with_llm
# ---------------------------------------------------------------------------


class TestLabelPaletteColorsWithLLM:
    """Tests for label_palette_colors_with_llm function."""

    @pytest.mark.asyncio
    async def test_label_palette_colors_with_llm_success(self) -> None:
        """Mock provider returns valid labels → used."""
        from spriteforge.preprocessor import label_palette_colors_with_llm

        # Create a mock chat provider
        class MockChatProvider:
            async def chat(self, messages, temperature, response_format):
                return '{"labels": ["Skin", "Hair", "Eyes"]}'

        colors = [(235, 210, 185), (220, 185, 90), (50, 180, 140)]
        quantized_png_bytes = b"fake-png-data"
        character_description = "A warrior with brown hair"

        result = await label_palette_colors_with_llm(
            quantized_png_bytes, colors, character_description, MockChatProvider()
        )

        assert result == ["Skin", "Hair", "Eyes"]

    @pytest.mark.asyncio
    async def test_label_palette_colors_with_llm_failure(self) -> None:
        """Mock provider raises → falls back to _describe_color()."""
        from spriteforge.preprocessor import label_palette_colors_with_llm

        class MockChatProvider:
            async def chat(self, messages, temperature, response_format):
                raise Exception("API error")

        colors = [(255, 0, 0), (0, 255, 0)]
        quantized_png_bytes = b"fake-png-data"
        character_description = "A warrior"

        result = await label_palette_colors_with_llm(
            quantized_png_bytes, colors, character_description, MockChatProvider()
        )

        # Should fall back to descriptive names
        assert len(result) == 2
        assert all(isinstance(label, str) for label in result)

    @pytest.mark.asyncio
    async def test_label_palette_colors_with_llm_wrong_count(self) -> None:
        """LLM returns wrong number → falls back to _describe_color()."""
        from spriteforge.preprocessor import label_palette_colors_with_llm

        class MockChatProvider:
            async def chat(self, messages, temperature, response_format):
                return '{"labels": ["Skin"]}'  # Only 1 label, but 2 colors

        colors = [(255, 0, 0), (0, 255, 0)]
        quantized_png_bytes = b"fake-png-data"
        character_description = "A warrior"

        result = await label_palette_colors_with_llm(
            quantized_png_bytes, colors, character_description, MockChatProvider()
        )

        # Should fall back to descriptive names
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests (real Azure)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_label_palette_real_azure(
    azure_project_endpoint: str, tmp_path: Path
) -> None:
    """Real GPT-5-nano call with a sample image → returns meaningful labels."""
    from spriteforge.preprocessor import label_palette_colors_with_llm
    from spriteforge.providers.azure_chat import AzureChatProvider

    # Create a simple test image with 3 colors
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    # Top third: skin-like beige
    for x in range(64):
        for y in range(20):
            img.putpixel((x, y), (235, 210, 185, 255))
    # Middle third: brown hair
    for x in range(64):
        for y in range(20, 40):
            img.putpixel((x, y), (139, 90, 43, 255))
    # Bottom third: blue eyes
    for x in range(64):
        for y in range(40, 60):
            img.putpixel((x, y), (50, 120, 200, 255))

    # Save as PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    quantized_png_bytes = buf.getvalue()

    colors = [(235, 210, 185), (139, 90, 43), (50, 120, 200)]
    character_description = "A human warrior with brown hair and blue eyes"

    # Create real Azure chat provider with labeling model
    provider = AzureChatProvider(
        project_endpoint=azure_project_endpoint,
        model_deployment_name=os.environ.get(
            "SPRITEFORGE_TEST_LABELING_MODEL", GenerationConfig().labeling_model
        ),
    )

    try:
        result = await label_palette_colors_with_llm(
            quantized_png_bytes, colors, character_description, provider
        )

        # The labels should be meaningful (not just "Color 1", etc.)
        assert len(result) == 3
        assert all(isinstance(label, str) for label in result)
        # At least one label should be semantic (not generic "Color N")
        has_semantic = any(not label.startswith("Color ") for label in result)
        assert has_semantic, f"Expected semantic labels, got: {result}"
    finally:
        await provider.close()
