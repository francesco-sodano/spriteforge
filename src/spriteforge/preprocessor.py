"""Image preprocessor for base reference image resize, quantize, and auto-palette extraction."""

from __future__ import annotations

import colorsys
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict

from spriteforge.errors import GenerationError
from spriteforge.logging import get_logger
from spriteforge.models import PaletteColor, PaletteConfig
from spriteforge.providers.chat import ChatProvider
from spriteforge.utils import image_to_data_url

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Color description thresholds (internal color-science constants)
# These are HSL lightness/saturation/hue boundaries used by _describe_color().
# They are implementation internals — NOT exposed in YAML config.
# ---------------------------------------------------------------------------

# Lightness thresholds (HLS 'l' value, range 0–1)
_NEAR_BLACK_L: float = 0.1      # Below this → "Near Black"
_NEAR_WHITE_L: float = 0.9      # Above this → "Near White"
_DARK_L: float = 0.3            # Below this → "Dark" qualifier
_MID_L: float = 0.6             # Below this → no qualifier; above → "Light"
_DARK_GRAY_L: float = 0.3       # Dark gray boundary (low saturation)
_GRAY_L: float = 0.6            # Gray/Light Gray boundary (low saturation)

# Saturation thresholds
_LOW_SATURATION: float = 0.15   # Below this → grayscale naming
_BROWN_MIN_S: float = 0.3       # Minimum saturation to call something brown
_GOLDEN_MIN_S: float = 0.5      # Minimum saturation for "Golden Yellow"

# Hue bucket boundaries (in degrees, hue ∈ [0, 360))
_HUE_RED_MAX: float = 15.0      # 0–15° → Red
_HUE_ORANGE_MAX: float = 45.0   # 15–45° → Orange
_HUE_YELLOW_MAX: float = 70.0   # 45–70° → Yellow
_HUE_GREEN_MAX: float = 150.0   # 70–150° → Green
_HUE_CYAN_MAX: float = 190.0    # 150–190° → Cyan
_HUE_BLUE_MAX: float = 260.0    # 190–260° → Blue
_HUE_PURPLE_MAX: float = 320.0  # 260–320° → Purple
_HUE_RED_MIN: float = 345.0     # 320–360° → Pink; ≥345° wraps back to Red

# Brown special-case thresholds (Yellow/Orange + low luminance)
_BROWN_MAX_L: float = 0.5       # l < 0.5 for brown check
_DARK_BROWN_L: float = 0.35     # l < 0.35 → "Brown"; else → "Dark Brown"

# Golden Yellow special-case luminance range
_GOLDEN_L_MIN: float = 0.45
_GOLDEN_L_MAX: float = 0.75

# Symbol priority list for auto-assignment (after O for outline).
# Skips '.' (transparent) and 'O' (outline). Starts with common pixel-art
# mnemonics: s=skin, h=hair, e=eyes, a=armor, v=vest, etc.
SYMBOL_POOL: list[str] = list("sheavbcdgiklmnprtuwxyz")


def _unpack_rgba(raw: bytes) -> list[tuple[int, int, int, int]]:
    """Unpack raw RGBA bytes into a list of (R, G, B, A) tuples.

    Args:
        raw: Raw bytes of an RGBA image (4 bytes per pixel).

    Returns:
        List of (R, G, B, A) tuples, one per pixel.
    """
    return list(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4]))


def _describe_color(rgb: tuple[int, int, int]) -> str:
    """Generate a descriptive name for an RGB color.

    Uses HSL-based hue bucketing + lightness qualifiers:
    - "Dark Red", "Light Blue", "Golden Yellow", etc.

    This is NOT a semantic label (doesn't know "skin" vs "hair") —
    it's a best-effort fallback when the LLM is unavailable.

    Args:
        rgb: (R, G, B) tuple with values 0-255.

    Returns:
        Descriptive color name string.
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    # Convert RGB to HLS (Python uses HLS, not HSL)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Special cases: near-black, near-white, low saturation
    if l < _NEAR_BLACK_L:
        return "Near Black"
    if l > _NEAR_WHITE_L:
        return "Near White"
    if s < _LOW_SATURATION:
        # Low saturation → grayscale
        if l < _DARK_GRAY_L:
            return "Dark Gray"
        elif l < _GRAY_L:
            return "Gray"
        else:
            return "Light Gray"

    # Hue-based color naming (h is in [0, 1])
    hue_deg = h * 360

    # Hue buckets
    if hue_deg < _HUE_RED_MAX or hue_deg >= _HUE_RED_MIN:
        hue_name = "Red"
    elif hue_deg < _HUE_ORANGE_MAX:
        hue_name = "Orange"
    elif hue_deg < _HUE_YELLOW_MAX:
        hue_name = "Yellow"
    elif hue_deg < _HUE_GREEN_MAX:
        hue_name = "Green"
    elif hue_deg < _HUE_CYAN_MAX:
        hue_name = "Cyan"
    elif hue_deg < _HUE_BLUE_MAX:
        hue_name = "Blue"
    elif hue_deg < _HUE_PURPLE_MAX:
        hue_name = "Purple"
    else:
        hue_name = "Pink"

    # Special case for brown (low luminance yellow/orange with moderate saturation)
    if hue_name in ("Yellow", "Orange") and l < _BROWN_MAX_L and s > _BROWN_MIN_S:
        return "Brown" if l < _DARK_BROWN_L else "Dark Brown"

    # Lightness qualifiers
    if l < _DARK_L:
        qualifier = "Dark"
    elif l < _MID_L:
        qualifier = ""  # No qualifier for medium
    else:
        qualifier = "Light"

    # Special case: golden yellow
    if hue_name == "Yellow" and s > _GOLDEN_MIN_S and _GOLDEN_L_MIN < l < _GOLDEN_L_MAX:
        return "Golden Yellow"

    # Build final name
    if qualifier:
        return f"{qualifier} {hue_name}"
    else:
        return hue_name


async def label_palette_colors_with_llm(
    quantized_png_bytes: bytes,
    colors: list[tuple[int, int, int]],
    character_description: str,
    chat_provider: ChatProvider,
) -> list[str]:
    """Label extracted palette colors using an LLM vision call.

    Sends the quantized reference image + RGB list + character description
    to a cheap LLM (e.g., gpt-5-nano) and returns semantic labels.

    Args:
        quantized_png_bytes: PNG bytes of the quantized reference image.
        colors: List of RGB tuples (in extraction order, excluding outline).
        character_description: Character visual description from config.
        chat_provider: Chat provider configured with the labeling model.

    Returns:
        List of semantic label strings, same length as colors.
        Falls back to descriptive color names if LLM fails.
    """
    # Build the color list string
    color_list_lines = []
    for idx, (r, g, b) in enumerate(colors, start=1):
        color_list_lines.append(f"{idx}. RGB({r}, {g}, {b})")
    color_list = "\n".join(color_list_lines)

    # Import the prompt here to avoid circular imports
    from spriteforge.prompts.preprocessor import PALETTE_LABELING_PROMPT

    # Build the prompt
    prompt_text = PALETTE_LABELING_PROMPT.format(
        character_description=character_description,
        color_list=color_list,
        color_count=len(colors),
    )

    # Prepare the vision message
    image_data_url = image_to_data_url(quantized_png_bytes)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]

    try:
        # Call the LLM
        response = await chat_provider.chat(
            messages=messages, temperature=0.5, response_format="json_object"
        )

        # Parse the response
        data = json.loads(response)
        if "labels" in data:
            labels = data["labels"]
        elif isinstance(data, dict) and len(data) == 1:
            # Try to extract the first value if it's an array
            first_value = next(iter(data.values()))
            if isinstance(first_value, list):
                labels = first_value
            else:
                raise ValueError("Response format not recognized")
        else:
            raise ValueError("Response format not recognized")

        # Validate label count
        if not isinstance(labels, list) or len(labels) != len(colors):
            logger.warning(
                "LLM returned wrong number of labels: %s vs %d colors. "
                "Falling back to descriptive names.",
                len(labels) if isinstance(labels, list) else "non-list",
                len(colors),
            )
            return [_describe_color(c) for c in colors]

        # Convert all labels to strings
        return [str(label) for label in labels]

    except Exception as exc:
        logger.warning(
            "LLM palette labeling failed: %s. Falling back to descriptive names.",
            exc,
        )
        return [_describe_color(c) for c in colors]


class PreprocessResult(BaseModel):
    """Result of preprocessing a base reference image.

    Attributes:
        quantized_image: PIL Image resized and quantized to N colors.
        palette: Auto-generated PaletteConfig from the quantized image.
        quantized_png_bytes: PNG bytes of the quantized image (for LLM vision input).
        original_color_count: Number of unique colors before quantization.
        final_color_count: Number of unique colors after quantization.
    """

    quantized_image: Image.Image
    palette: PaletteConfig
    quantized_png_bytes: bytes
    original_color_count: int
    final_color_count: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


def validate_reference_image(
    image_path: str | Path,
    frame_width: int = 64,
    frame_height: int = 64,
) -> Image.Image:
    """Load and validate a base reference image.

    Checks:
    - File exists and is a valid image
    - Aspect ratio is compatible with frame dimensions
      (square for square frames, or matching aspect ratio)
    - Image is not too small (at least 32×32)

    Args:
        image_path: Path to the image file.
        frame_width: Target frame width for aspect ratio check.
        frame_height: Target frame height for aspect ratio check.

    Returns:
        Loaded PIL Image in RGBA mode.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If image is invalid or incompatible.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
        img.load()  # Force load to validate
    except Exception as exc:
        raise ValueError(f"Cannot open image: {path}") from exc

    width, height = img.size

    # Minimum size check
    if width < 32 or height < 32:
        raise ValueError(f"Image too small: {width}×{height} (minimum 32×32)")

    # Aspect ratio compatibility check
    target_ratio = frame_width / frame_height
    image_ratio = width / height
    # Allow 2:1 tolerance — reject extreme aspect ratios
    if image_ratio > target_ratio * 2 or image_ratio < target_ratio / 2:
        raise ValueError(
            f"Incompatible aspect ratio: image is {width}×{height} "
            f"(ratio {image_ratio:.2f}), target frame is "
            f"{frame_width}×{frame_height} (ratio {target_ratio:.2f})"
        )

    # Convert to RGBA if needed
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    return img


def resize_reference(
    image: Image.Image,
    target_width: int = 64,
    target_height: int = 64,
) -> Image.Image:
    """Resize a reference image to target frame dimensions.

    Uses NEAREST interpolation when upscaling (preserves hard pixel-art
    edges) and LANCZOS when downscaling (properly anti-aliases to avoid
    severe aliasing and detail loss from skipping pixels).

    Args:
        image: Source PIL Image.
        target_width: Target width in pixels.
        target_height: Target height in pixels.

    Returns:
        Resized PIL Image.
    """
    if image.size == (target_width, target_height):
        return image.copy()

    # Upscaling: NEAREST preserves hard edges in pixel art.
    # Downscaling: LANCZOS properly anti-aliases, avoiding the severe
    # aliasing that NEAREST causes (e.g. 1024×1024 → 64×64 drops 99.6%
    # of pixels with NEAREST).
    is_downscaling = target_width < image.width or target_height < image.height
    resample = Image.Resampling.LANCZOS if is_downscaling else Image.Resampling.NEAREST

    return image.resize((target_width, target_height), resample=resample)


def _assign_symbols(
    colors: list[tuple[int, int, int]],
    coverage: list[int],
    outline_index: int,
    semantic_labels: list[str] | None = None,
) -> list[tuple[str, str, tuple[int, int, int]]]:
    """Assign palette symbols to extracted colors.

    Colors are sorted by pixel coverage (descending).
    The darkest color gets symbol 'O' (outline).
    Remaining colors get symbols from SYMBOL_POOL in order.

    Args:
        colors: List of RGB tuples.
        coverage: Pixel count for each color.
        outline_index: Index of the color to use as outline.
        semantic_labels: Optional list of semantic labels for colors
            (excluding outline). If provided and length matches, these
            are used instead of "Color N". Falls back to "Color N" if
            labels are invalid or wrong length.

    Returns:
        List of (name, symbol, rgb) tuples.
    """
    # Build (color, coverage, original_index) tuples
    entries = list(zip(colors, coverage, range(len(colors))))

    # Separate outline from the rest
    outline_entry = entries[outline_index]
    remaining = [e for i, e in enumerate(entries) if i != outline_index]

    # Sort remaining by coverage descending
    remaining.sort(key=lambda e: e[1], reverse=True)

    result: list[tuple[str, str, tuple[int, int, int]]] = []

    # Outline gets symbol 'O'
    result.append(("Outline", "O", outline_entry[0]))

    # Validate semantic labels
    use_labels = (
        semantic_labels is not None
        and isinstance(semantic_labels, list)
        and len(semantic_labels) == len(remaining)
    )

    # Assign pool symbols to remaining colors in coverage order
    pool_idx = 0
    for color, _count, _orig_idx in remaining:
        if pool_idx >= len(SYMBOL_POOL):
            break
        symbol = SYMBOL_POOL[pool_idx]
        if use_labels and semantic_labels is not None:
            name = semantic_labels[pool_idx]
        else:
            name = f"Color {pool_idx + 1}"
        result.append((name, symbol, color))
        pool_idx += 1

    return result


def _quantize_opaque_only(
    image: Image.Image,
    alpha: Image.Image,
    max_colors: int,
) -> Image.Image:
    """Quantize an RGBA image considering only opaque pixels.

    Avoids the color-bleeding problem that occurs when transparent pixels
    are converted to RGB (becoming black), which then participate in
    median-cut clustering, wasting palette slots and pulling edge colors
    toward black.

    Strategy: build a 1-row RGB image containing only the opaque pixels,
    quantize that, then map the reduced palette back onto the full image
    while preserving the original alpha channel.

    Args:
        image: Source RGBA image.
        alpha: Alpha channel of the source image.
        max_colors: Maximum number of colors for quantization.

    Returns:
        New RGBA image with quantized opaque pixels and original alpha.
    """
    width, height = image.size
    raw = image.tobytes()
    alpha_data = alpha.tobytes()

    # Collect indices + RGB values of opaque pixels
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

    # Build a 1×N RGB image of only opaque pixels for quantization
    opaque_img = Image.frombytes("RGB", (n_opaque, 1), bytes(opaque_rgb))
    quantized_p = opaque_img.quantize(
        colors=max_colors, method=Image.Quantize.MEDIANCUT
    )
    quantized_rgb_data = quantized_p.convert("RGB").tobytes()

    # Reconstruct RGBA output: start with fully transparent
    out = bytearray(width * height * 4)
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
) -> PaletteConfig:
    """Extract a PaletteConfig from a PIL Image.

    Quantizes the image (if needed) and maps each unique color to a symbol.

    Args:
        image: PIL Image to extract palette from (must be RGBA).
        max_colors: Maximum opaque colors to extract.
        outline_color: Optional forced outline color. If None, uses darkest.
        semantic_labels: Optional semantic labels for colors (excluding outline).
            If provided and length matches, these are used instead of "Color N".

    Returns:
        A PaletteConfig with auto-assigned symbols.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Extract alpha channel for transparency preservation
    alpha = image.getchannel("A")

    # Count unique opaque colors
    raw = image.tobytes()
    pixels = _unpack_rgba(raw)
    opaque_pixels = [(r, g, b) for r, g, b, a in pixels if a > 0]
    unique_opaque = set(opaque_pixels)

    needs_quantize = len(unique_opaque) > max_colors

    if needs_quantize:
        # Quantize only opaque pixels to avoid color bleeding from the
        # transparent background.  Transparent pixels converted to RGB
        # become black/white, which would pollute the median-cut
        # clustering and waste palette slots.
        quantized_rgba = _quantize_opaque_only(image, alpha, max_colors)

        # Re-extract opaque colors from quantized image
        raw_q = quantized_rgba.tobytes()
        pixels_q = _unpack_rgba(raw_q)
        opaque_pixels = [(r, g, b) for r, g, b, a in pixels_q if a > 0]

    # Count color frequency
    color_counts = Counter(opaque_pixels)
    colors = list(color_counts.keys())
    coverage = [color_counts[c] for c in colors]

    if not colors:
        # Fully transparent image — return minimal palette
        return PaletteConfig(name="auto")

    # Determine outline color (darkest by luminance)
    if outline_color is not None:
        # Find closest match to forced outline color
        def _color_dist(c: tuple[int, int, int]) -> float:
            return sum((a - b) ** 2 for a, b in zip(c, outline_color))

        outline_index = min(range(len(colors)), key=lambda i: _color_dist(colors[i]))
    else:
        # Use darkest color by luminance
        def _luminance(c: tuple[int, int, int]) -> float:
            # ITU-R BT.601 luma coefficients (standard for SD/pixel-art content)
            return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

        outline_index = min(range(len(colors)), key=lambda i: _luminance(colors[i]))

    # Assign symbols
    assignments = _assign_symbols(colors, coverage, outline_index, semantic_labels)

    # Build PaletteConfig
    outline_entry = assignments[0]  # Always first (outline)
    palette_colors: list[PaletteColor] = []
    for name, symbol, color_rgb in assignments[1:]:
        palette_colors.append(
            PaletteColor(
                element=name,
                symbol=symbol,
                r=color_rgb[0],
                g=color_rgb[1],
                b=color_rgb[2],
            )
        )

    outline_pc = PaletteColor(
        element=outline_entry[0],
        symbol=outline_entry[1],
        r=outline_entry[2][0],
        g=outline_entry[2][1],
        b=outline_entry[2][2],
    )

    return PaletteConfig(name="auto", outline=outline_pc, colors=palette_colors)


def preprocess_reference(
    image_path: str | Path,
    frame_width: int = 64,
    frame_height: int = 64,
    max_colors: int = 16,
    outline_color: tuple[int, int, int] | None = None,
) -> PreprocessResult:
    """Preprocess a base reference image for the generation pipeline.

    Steps:
    1. Load and validate the image (must be square or compatible aspect ratio)
    2. Resize to (frame_width, frame_height) using nearest-neighbor interpolation
    3. Quantize to max_colors using median-cut color quantization
    4. Extract unique colors and build a PaletteConfig with auto-assigned symbols
    5. Return the quantized image + palette

    Args:
        image_path: Path to the base reference PNG.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.
        max_colors: Maximum number of opaque colors to extract (default 16).
            Does not count the transparent color.
        outline_color: Optional forced outline color (R, G, B).
            If None, the darkest color in the image is used.

    Returns:
        PreprocessResult with quantized image and auto-generated palette.

    Raises:
        FileNotFoundError: If image_path doesn't exist.
        ValueError: If image is not compatible (e.g., extreme aspect ratio).
    """
    # Step 1: Load and validate
    img = validate_reference_image(image_path, frame_width, frame_height)

    # Count original unique opaque colors
    raw_orig = img.tobytes()
    orig_pixels = _unpack_rgba(raw_orig)
    original_color_count = len(set((r, g, b) for r, g, b, a in orig_pixels if a > 0))

    # Step 2: Resize
    resized = resize_reference(img, frame_width, frame_height)

    # Step 3: Quantize once (if needed)
    if resized.mode != "RGBA":
        resized = resized.convert("RGBA")

    alpha = resized.getchannel("A")
    raw_resized = resized.tobytes()
    resized_opaque = [
        (r, g, b)
        for r, g, b, a in _unpack_rgba(raw_resized)
        if a > 0
    ]
    unique_resized = set(resized_opaque)
    needs_quantize = len(unique_resized) > max_colors

    if needs_quantize:
        # Quantize only opaque pixels to prevent transparent-background
        # colors from polluting the palette (see _quantize_opaque_only).
        quantized_image = _quantize_opaque_only(resized, alpha, max_colors)
    else:
        quantized_image = resized.copy()

    # Step 4: Extract palette from the already-quantized image
    # This guarantees the palette symbols match the quantized image pixels exactly,
    # avoiding the double-quantization inconsistency.
    palette = extract_palette_from_image(quantized_image, max_colors, outline_color)

    # Count final unique opaque colors
    raw_q = quantized_image.tobytes()
    q_pixels = _unpack_rgba(raw_q)
    final_color_count = len(set((r, g, b) for r, g, b, a in q_pixels if a > 0))

    # Generate PNG bytes
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
