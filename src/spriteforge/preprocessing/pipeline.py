"""End-to-end preprocessing pipeline orchestration."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict

from spriteforge.models import PaletteConfig
from spriteforge.preprocessing.image_io import (
    resize_reference,
    validate_reference_image,
)
from spriteforge.preprocessing.palette_quantization import (
    _quantize_opaque_only,
    _unpack_rgba,
    extract_palette_from_image,
)


class PreprocessResult(BaseModel):
    """Result of preprocessing a base reference image."""

    quantized_image: Image.Image
    palette: PaletteConfig
    quantized_png_bytes: bytes
    original_color_count: int
    final_color_count: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


def preprocess_reference(
    image_path: str | Path,
    frame_width: int = 64,
    frame_height: int = 64,
    max_colors: int = 16,
    outline_color: tuple[int, int, int] | None = None,
    semantic_labels: bool = True,
) -> PreprocessResult:
    """Preprocess a base reference image for the generation pipeline."""
    img = validate_reference_image(image_path, frame_width, frame_height)

    orig_pixels = _unpack_rgba(img.tobytes())
    original_color_count = len(set((r, g, b) for r, g, b, a in orig_pixels if a > 0))

    resized = resize_reference(img, frame_width, frame_height)
    if resized.mode != "RGBA":
        resized = resized.convert("RGBA")

    alpha = resized.getchannel("A")
    resized_opaque = [
        (r, g, b) for r, g, b, a in _unpack_rgba(resized.tobytes()) if a > 0
    ]

    if len(set(resized_opaque)) > max_colors:
        quantized_image = _quantize_opaque_only(resized, alpha, max_colors)
    else:
        quantized_image = resized.copy()

    palette = extract_palette_from_image(
        quantized_image,
        max_colors,
        outline_color,
        use_descriptive_names=semantic_labels,
    )

    q_pixels = _unpack_rgba(quantized_image.tobytes())
    final_color_count = len(set((r, g, b) for r, g, b, a in q_pixels if a > 0))

    with io.BytesIO() as buf:
        quantized_image.save(buf, format="PNG")
        quantized_png_bytes = buf.getvalue()

    return PreprocessResult(
        quantized_image=quantized_image,
        palette=palette,
        quantized_png_bytes=quantized_png_bytes,
        original_color_count=original_color_count,
        final_color_count=final_color_count,
    )
