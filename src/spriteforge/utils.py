"""Shared utility functions for SpriteForge.

Contains helpers used across multiple modules to avoid code duplication.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image


def image_to_base64(image: Image.Image | bytes) -> str:
    """Convert a PIL Image or raw bytes to a base64-encoded string.

    Args:
        image: PIL Image object or raw PNG bytes.

    Returns:
        Base64-encoded string of the image data.
    """
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        raw = buf.getvalue()
    else:
        raw = image
    return base64.b64encode(raw).decode("ascii")


def image_to_data_url(image: Image.Image | bytes, media_type: str = "image/png") -> str:
    """Convert a PIL Image or raw bytes to a base64 data URL.

    Args:
        image: PIL Image object or raw PNG bytes.
        media_type: MIME type for the data URL.

    Returns:
        A data URL string: ``data:{media_type};base64,{encoded_data}``
    """
    b64 = image_to_base64(image)
    return f"data:{media_type};base64,{b64}"


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text.

    Handles ````` ```json ... ``` `````, ````` ``` ... ``` `````,
    and similar patterns.  Returns the content inside the outermost
    fences, or the original text if no fences are found.

    Args:
        text: Raw text potentially wrapped in code fences.

    Returns:
        Text with outermost code fences stripped.
    """
    fence_pattern = re.compile(r"```(?:\w*)\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


def compress_grid_rle(grid: list[str]) -> str:
    """Compress a grid using run-length encoding to reduce token usage.

    Each row is encoded as a series of ``<count><symbol>`` pairs.
    For example, ``"....OOHH...."`` becomes ``"4.2O2H4."``.
    Runs of length 1 omit the count (e.g., ``"O"`` stays ``"O"``).

    Args:
        grid: List of strings representing the grid rows.

    Returns:
        Newline-joined RLE-encoded rows.
    """
    lines: list[str] = []
    for row in grid:
        if not row:
            lines.append("")
            continue
        parts: list[str] = []
        current_char = row[0]
        count = 1
        for ch in row[1:]:
            if ch == current_char:
                count += 1
            else:
                parts.append(f"{count}{current_char}" if count > 1 else current_char)
                current_char = ch
                count = 1
        parts.append(f"{count}{current_char}" if count > 1 else current_char)
        lines.append("".join(parts))
    return "\n".join(lines)


def validate_grid_dimensions(
    grid: list[str], expected_rows: int, expected_cols: int
) -> str | None:
    """Check that *grid* has the expected dimensions.

    Args:
        grid: The palette-indexed grid to validate.
        expected_rows: Expected number of rows.
        expected_cols: Expected number of characters per row.

    Returns:
        ``None`` if the grid is valid; a human-readable error message otherwise.
    """
    if len(grid) != expected_rows:
        return f"Grid must have {expected_rows} rows, got {len(grid)}."
    for i, row in enumerate(grid):
        if len(row) != expected_cols:
            return f"Row {i} must have {expected_cols} characters, got {len(row)}."
    return None


def parse_json_from_llm(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling common formatting quirks.

    Strips code fences, handles trailing commas, and attempts JSON parsing.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        ValueError: If the text cannot be parsed as JSON.
    """
    cleaned = strip_code_fences(text.strip())

    # First attempt: parse as-is
    try:
        data: Any = json.loads(cleaned)
        if isinstance(data, dict):
            return data
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    except json.JSONDecodeError:
        pass

    # Second attempt: strip trailing commas (common LLM quirk)
    # Remove trailing commas before } or ]
    sanitized = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        data = json.loads(sanitized)
        if isinstance(data, dict):
            return data
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {exc}") from exc
