"""Palette extraction, symbol assignment, and quantization helpers."""

from __future__ import annotations

from collections import Counter

from PIL import Image

from spriteforge.models import PaletteColor, PaletteConfig
from spriteforge.preprocessing.labeling import SYMBOL_POOL, describe_color


def _unpack_rgba(raw: bytes) -> list[tuple[int, int, int, int]]:
    return list(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4]))


def _assign_symbols(
    colors: list[tuple[int, int, int]],
    coverage: list[int],
    outline_index: int,
    semantic_labels: list[str] | None = None,
) -> list[tuple[str, str, tuple[int, int, int]]]:
    """Assign palette symbols to extracted colors."""
    entries = list(zip(colors, coverage, range(len(colors))))
    outline_entry = entries[outline_index]
    remaining = [entry for i, entry in enumerate(entries) if i != outline_index]
    remaining.sort(key=lambda entry: entry[1], reverse=True)

    result: list[tuple[str, str, tuple[int, int, int]]] = [
        ("Outline", "O", outline_entry[0])
    ]

    use_labels = (
        semantic_labels is not None
        and isinstance(semantic_labels, list)
        and len(semantic_labels) == len(remaining)
    )

    for pool_idx, (color, _count, _orig_idx) in enumerate(remaining):
        if pool_idx >= len(SYMBOL_POOL):
            break
        symbol = SYMBOL_POOL[pool_idx]
        name = (
            semantic_labels[pool_idx]
            if use_labels and semantic_labels is not None
            else f"Color {pool_idx + 1}"
        )
        result.append((name, symbol, color))

    return result


def _quantize_opaque_only(
    image: Image.Image,
    alpha: Image.Image,
    max_colors: int,
    max_sample_pixels: int = 1_000_000,
) -> Image.Image:
    """Quantize an RGBA image considering only opaque pixels."""
    width, height = image.size
    raw = image.tobytes()
    alpha_data = alpha.tobytes()

    opaque_indices: list[int] = []
    opaque_rgb: list[int] = []
    for i in range(width * height):
        if alpha_data[i] > 0:
            opaque_indices.append(i)
            base = i * 4
            opaque_rgb.extend(raw[base : base + 3])

    if not opaque_indices:
        return image.copy()

    n_opaque = len(opaque_indices)
    sampled_for_palette = False
    if n_opaque > max_sample_pixels:
        step = max(1, n_opaque // max_sample_pixels)
        sampled_indices = opaque_indices[::step][:max_sample_pixels]
        sampled_rgb = bytearray()
        for pixel_idx in sampled_indices:
            base = pixel_idx * 4
            sampled_rgb.extend(raw[base : base + 3])
        opaque_rgb_for_palette = bytes(sampled_rgb)
        sampled_for_palette = True
    else:
        opaque_rgb_for_palette = bytes(opaque_rgb)

    opaque_img = Image.frombytes(
        "RGB", (len(opaque_rgb_for_palette) // 3, 1), opaque_rgb_for_palette
    )
    quantized_p = opaque_img.quantize(
        colors=max_colors, method=Image.Quantize.MEDIANCUT
    )

    if sampled_for_palette:
        full_rgb = image.convert("RGB")
        quantized_rgb_data = (
            full_rgb.quantize(palette=quantized_p).convert("RGB").tobytes()
        )
    else:
        quantized_rgb_data = quantized_p.convert("RGB").tobytes()

    out = bytearray(width * height * 4)
    if sampled_for_palette:
        for pixel_idx in opaque_indices:
            base_out = pixel_idx * 4
            base_q = pixel_idx * 3
            out[base_out] = quantized_rgb_data[base_q]
            out[base_out + 1] = quantized_rgb_data[base_q + 1]
            out[base_out + 2] = quantized_rgb_data[base_q + 2]
            out[base_out + 3] = alpha_data[pixel_idx]
    else:
        for j, pixel_idx in enumerate(opaque_indices):
            base_out = pixel_idx * 4
            base_q = j * 3
            out[base_out] = quantized_rgb_data[base_q]
            out[base_out + 1] = quantized_rgb_data[base_q + 1]
            out[base_out + 2] = quantized_rgb_data[base_q + 2]
            out[base_out + 3] = alpha_data[pixel_idx]

    return Image.frombytes("RGBA", (width, height), bytes(out))


def extract_palette_from_image(
    image: Image.Image,
    max_colors: int = 16,
    outline_color: tuple[int, int, int] | None = None,
    semantic_labels: list[str] | None = None,
    use_descriptive_names: bool = False,
) -> PaletteConfig:
    """Extract a PaletteConfig from a PIL Image."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = image.getchannel("A")

    raw = image.tobytes()
    pixels = _unpack_rgba(raw)
    opaque_pixels = [(r, g, b) for r, g, b, a in pixels if a > 0]
    unique_opaque = set(opaque_pixels)

    if len(unique_opaque) > max_colors:
        quantized_rgba = _quantize_opaque_only(image, alpha, max_colors)
        pixels_q = _unpack_rgba(quantized_rgba.tobytes())
        opaque_pixels = [(r, g, b) for r, g, b, a in pixels_q if a > 0]

    color_counts = Counter(opaque_pixels)
    colors = list(color_counts.keys())
    coverage = [color_counts[c] for c in colors]

    if not colors:
        return PaletteConfig(name="auto")

    if outline_color is not None:

        def _color_dist(c: tuple[int, int, int]) -> float:
            return sum((a - b) ** 2 for a, b in zip(c, outline_color))

        outline_index = min(range(len(colors)), key=lambda i: _color_dist(colors[i]))
    else:

        def _luminance(c: tuple[int, int, int]) -> float:
            return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

        outline_index = min(range(len(colors)), key=lambda i: _luminance(colors[i]))

    labels = semantic_labels
    if labels is None and use_descriptive_names:
        entries = list(zip(colors, coverage, range(len(colors))))
        remaining = [entry for i, entry in enumerate(entries) if i != outline_index]
        remaining.sort(key=lambda entry: entry[1], reverse=True)
        labels = [describe_color(color) for color, _count, _orig_idx in remaining]

    assignments = _assign_symbols(colors, coverage, outline_index, labels)

    outline_entry = assignments[0]
    palette_colors = [
        PaletteColor(element=name, symbol=symbol, r=rgb[0], g=rgb[1], b=rgb[2])
        for name, symbol, rgb in assignments[1:]
    ]
    outline_pc = PaletteColor(
        element=outline_entry[0],
        symbol=outline_entry[1],
        r=outline_entry[2][0],
        g=outline_entry[2][1],
        b=outline_entry[2][2],
    )

    return PaletteConfig(name="auto", outline=outline_pc, colors=palette_colors)


__all__ = [
    "SYMBOL_POOL",
    "_assign_symbols",
    "extract_palette_from_image",
    "_quantize_opaque_only",
    "_unpack_rgba",
]
