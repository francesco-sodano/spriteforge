"""Sprite row assembly into a final spritesheet PNG."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from spriteforge.models import SpritesheetSpec


def _open_image(source: bytes | str | Path) -> Image.Image:
    """Open an image from raw bytes or a file path.

    Args:
        source: Raw PNG bytes, a string path, or a ``Path`` object.

    Returns:
        A Pillow ``Image`` in RGBA mode.

    Raises:
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If *source* type is unsupported.
    """
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGBA")
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Row image not found: {p}")
        return Image.open(p).convert("RGBA")
    raise ValueError(f"Unsupported image source type: {type(source)}")


def assemble_spritesheet(
    row_images: Mapping[int, bytes | str | Path],
    spec: SpritesheetSpec,
    output_path: str | Path | None = None,
) -> Image.Image:
    """Assemble individual animation-row images into a single spritesheet.

    Each row image is pasted at the correct vertical position and padded
    on the right with transparent pixels to fill the full sheet width.

    Args:
        row_images: Mapping of row index → image source (raw bytes or
            file path). Must contain an entry for every row defined in
            *spec*.
        spec: The ``SpritesheetSpec`` describing the sheet layout.
        output_path: If provided, saves the assembled spritesheet PNG
            to this path.

    Returns:
        The assembled Pillow ``Image`` (RGBA).

    Raises:
        ValueError: If a required row is missing from *row_images*.
    """
    sheet_w = spec.sheet_width
    sheet_h = spec.sheet_height
    frame_h = spec.character.frame_height

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    for seq_idx, animation in enumerate(sorted(spec.animations, key=lambda a: a.row)):
        row_idx = animation.row
        if row_idx not in row_images:
            raise ValueError(f"Missing image for row {row_idx} ({animation.name})")

        row_img = _open_image(row_images[row_idx])

        # Validate height matches frame height
        if row_img.height != frame_h:
            raise ValueError(
                f"Row {row_idx} ({animation.name}) image height "
                f"{row_img.height}px does not match frame height {frame_h}px"
            )

        # Validate width does not exceed sheet width
        if row_img.width > sheet_w:
            raise ValueError(
                f"Row {row_idx} ({animation.name}) image width "
                f"{row_img.width}px exceeds sheet width {sheet_w}px"
            )

        y_offset = seq_idx * frame_h
        sheet.paste(row_img, (0, y_offset), row_img)

    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(str(dest), format="PNG")

    return sheet
