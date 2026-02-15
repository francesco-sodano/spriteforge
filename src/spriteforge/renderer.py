"""Grid-to-PNG renderer for converting palette-indexed grids to PIL Images."""

from __future__ import annotations

import io

from PIL import Image

from spriteforge.models import SpritesheetSpec


def render_frame(
    grid: list[str],
    palette_map: dict[str, tuple[int, int, int, int]],
    frame_width: int = 64,
    frame_height: int = 64,
) -> Image.Image:
    """Render a palette-indexed grid to an RGBA image.

    Each character in the grid is looked up in *palette_map* and written
    as a single pixel.  The grid coordinate ``grid[y][x]`` maps to pixel
    ``(x, y)`` in the resulting image.

    Args:
        grid: List of *frame_height* strings, each exactly *frame_width*
            characters long.  Each character is a palette symbol.
        palette_map: Mapping of single-character symbols to RGBA tuples.
        frame_width: Expected width of the grid (columns per row).
        frame_height: Expected height of the grid (number of rows).

    Returns:
        A PIL Image of size ``(frame_width, frame_height)`` in RGBA mode.

    Raises:
        ValueError: If grid dimensions do not match expected size.
        KeyError: If a symbol in the grid is not found in palette_map.
    """
    if len(grid) != frame_height:
        raise ValueError(f"Grid must have exactly {frame_height} rows, got {len(grid)}")
    for i, row in enumerate(grid):
        if len(row) != frame_width:
            raise ValueError(
                f"Row {i} must be exactly {frame_width} characters, got {len(row)}"
            )

    pixels: list[tuple[int, int, int, int]] = []
    for y, row in enumerate(grid):
        for x, symbol in enumerate(row):
            if symbol not in palette_map:
                raise KeyError(f"Unknown palette symbol {symbol!r} at ({x}, {y})")
            pixels.append(palette_map[symbol])

    img = Image.new("RGBA", (frame_width, frame_height))
    img.putdata(pixels)  # type: ignore[arg-type]
    return img


def render_row_strip(
    frames: list[list[str]],
    palette_map: dict[str, tuple[int, int, int, int]],
    spritesheet_columns: int = 14,
    frame_width: int = 64,
    frame_height: int = 64,
) -> Image.Image:
    """Render multiple frame grids into a horizontal strip image.

    Frames are placed left-to-right.  The strip is padded with transparent
    pixels on the right to fill the full spritesheet row width.

    Args:
        frames: List of frame grids (each is list of 64 strings).
        palette_map: Symbol → RGBA mapping.
        spritesheet_columns: Total columns in the spritesheet (for padding).
        frame_width: Width of each frame in pixels.
        frame_height: Height of each frame in pixels.

    Returns:
        A PIL Image of size ``(spritesheet_columns * frame_width, frame_height)``.
    """
    strip_width = spritesheet_columns * frame_width
    strip = Image.new("RGBA", (strip_width, frame_height), (0, 0, 0, 0))

    for idx, frame_grid in enumerate(frames):
        frame_img = render_frame(
            frame_grid, palette_map, frame_width=frame_width, frame_height=frame_height
        )
        x_offset = idx * frame_width
        strip.paste(frame_img, (x_offset, 0))

    return strip


def render_spritesheet(
    all_rows: dict[int, list[list[str]]],
    spec: SpritesheetSpec,
    palette_map: dict[str, tuple[int, int, int, int]],
) -> Image.Image:
    """Render all animation rows into a complete spritesheet image.

    Args:
        all_rows: Mapping of row index → list of frame grids.
        spec: The spritesheet specification.
        palette_map: Symbol → RGBA mapping.

    Returns:
        A complete spritesheet PIL Image.

    Raises:
        ValueError: If a required row is missing from all_rows.
    """
    sheet_w = spec.sheet_width
    sheet_h = spec.sheet_height
    frame_w = spec.character.frame_width
    frame_h = spec.character.frame_height
    cols = spec.character.spritesheet_columns

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    for animation in spec.animations:
        row_idx = animation.row
        if row_idx not in all_rows:
            raise ValueError(f"Missing row {row_idx} ({animation.name}) in all_rows")

        row_strip = render_row_strip(
            all_rows[row_idx],
            palette_map,
            spritesheet_columns=cols,
            frame_width=frame_w,
            frame_height=frame_h,
        )
        y_offset = row_idx * frame_h
        sheet.paste(row_strip, (0, y_offset))

    return sheet


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
