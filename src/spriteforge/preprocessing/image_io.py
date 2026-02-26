"""Image validation and resize helpers for preprocessing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def validate_reference_image(
    image_path: str | Path,
    frame_width: int = 64,
    frame_height: int = 64,
) -> Image.Image:
    """Load and validate a base reference image."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise ValueError(f"Cannot open image: {path}") from exc

    width, height = img.size
    if width < 32 or height < 32:
        raise ValueError(f"Image too small: {width}×{height} (minimum 32×32)")

    target_ratio = frame_width / frame_height
    image_ratio = width / height
    if image_ratio > target_ratio * 2 or image_ratio < target_ratio / 2:
        raise ValueError(
            f"Incompatible aspect ratio: image is {width}×{height} "
            f"(ratio {image_ratio:.2f}), target frame is "
            f"{frame_width}×{frame_height} (ratio {target_ratio:.2f})"
        )

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    return img


def resize_reference(
    image: Image.Image,
    target_width: int = 64,
    target_height: int = 64,
) -> Image.Image:
    """Resize a reference image to target frame dimensions."""
    if image.size == (target_width, target_height):
        return image.copy()

    is_downscaling = target_width < image.width or target_height < image.height
    resample = Image.Resampling.LANCZOS if is_downscaling else Image.Resampling.NEAREST

    return image.resize((target_width, target_height), resample=resample)
