"""Grid-to-PNG renderer for converting palette-indexed grids to PIL Images."""

from __future__ import annotations

import io

from PIL import Image

from spriteforge.errors import RenderError
from spriteforge.models import FrameContext


def render_frame(
    grid: list[str],
    context: FrameContext,
) -> Image.Image:
    """Render a palette-indexed grid to an RGBA image.

    Each character in the grid is looked up in *context.palette_map* and written
    as a single pixel.  The grid coordinate ``grid[y][x]`` maps to pixel
    ``(x, y)`` in the resulting image.

    Args:
        grid: List of *context.frame_height* strings, each exactly *context.frame_width*
            characters long.  Each character is a palette symbol.
        context: Frame context containing palette_map, frame_width, and frame_height.

    Returns:
        A PIL Image of size ``(context.frame_width, context.frame_height)`` in RGBA mode.

    Raises:
        ValueError: If grid dimensions do not match expected size.
        RenderError: If a symbol in the grid is not found in context.palette_map.
    """
    frame_width = context.frame_width
    frame_height = context.frame_height
    palette_map = context.palette_map

    if len(grid) != frame_height:
        raise ValueError(f"Grid must have exactly {frame_height} rows, got {len(grid)}")
    for i, row in enumerate(grid):
        if len(row) != frame_width:
            raise ValueError(
                f"Row {i} must be exactly {frame_width} characters, got {len(row)}"
            )

    buf = bytearray(frame_width * frame_height * 4)
    offset = 0
    for y, row in enumerate(grid):
        for x, symbol in enumerate(row):
            if symbol not in palette_map:
                raise RenderError(f"Unknown palette symbol {symbol!r} at ({x}, {y})")
            r, g, b, a = palette_map[symbol]
            buf[offset] = r
            buf[offset + 1] = g
            buf[offset + 2] = b
            buf[offset + 3] = a
            offset += 4

    img = Image.frombytes("RGBA", (frame_width, frame_height), bytes(buf))
    return img


def render_row_strip(
    frames: list[list[str]],
    context: FrameContext,
) -> Image.Image:
    """Render multiple frame grids into a horizontal strip image.

    Frames are placed left-to-right.  The strip is padded with transparent
    pixels on the right to fill the full spritesheet row width.

    Args:
        frames: List of frame grids (each is list of frame_height strings).
        context: Frame context containing palette_map, spritesheet_columns,
            frame_width, and frame_height.

    Returns:
        A PIL Image of size ``(context.spritesheet_columns * context.frame_width, context.frame_height)``.
    """
    strip_width = context.spritesheet_columns * context.frame_width
    strip = Image.new("RGBA", (strip_width, context.frame_height), (0, 0, 0, 0))

    for idx, frame_grid in enumerate(frames):
        frame_img = render_frame(frame_grid, context)
        x_offset = idx * context.frame_width
        strip.paste(frame_img, (x_offset, 0))

    return strip


def frame_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL Image to PNG bytes.

    Args:
        image: The PIL Image to serialize.

    Returns:
        PNG-encoded bytes.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
