"""LLM and heuristic color labeling for extracted palettes."""

from __future__ import annotations

import colorsys
import json

from spriteforge.logging import get_logger
from spriteforge.providers.chat import ChatProvider
from spriteforge.utils import image_to_data_url_limited

logger = get_logger(__name__)

_NEAR_BLACK_L: float = 0.1
_NEAR_WHITE_L: float = 0.9
_DARK_L: float = 0.3
_MID_L: float = 0.6
_DARK_GRAY_L: float = 0.3
_GRAY_L: float = 0.6

_LOW_SATURATION: float = 0.15
_BROWN_MIN_S: float = 0.3
_GOLDEN_MIN_S: float = 0.5

_HUE_RED_MAX: float = 15.0
_HUE_ORANGE_MAX: float = 45.0
_HUE_YELLOW_MAX: float = 70.0
_HUE_GREEN_MAX: float = 150.0
_HUE_CYAN_MAX: float = 190.0
_HUE_BLUE_MAX: float = 260.0
_HUE_PURPLE_MAX: float = 320.0
_HUE_RED_MIN: float = 345.0

_BROWN_MAX_L: float = 0.5
_DARK_BROWN_L: float = 0.35

_GOLDEN_L_MIN: float = 0.45
_GOLDEN_L_MAX: float = 0.75

SYMBOL_POOL: list[str] = list("sheavbcdgiklmnprtuwxyz")


def describe_color(rgb: tuple[int, int, int]) -> str:
    """Generate a descriptive name for an RGB color."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    if l < _NEAR_BLACK_L:
        return "Near Black"
    if l > _NEAR_WHITE_L:
        return "Near White"
    if s < _LOW_SATURATION:
        if l < _DARK_GRAY_L:
            return "Dark Gray"
        if l < _GRAY_L:
            return "Gray"
        return "Light Gray"

    hue_deg = h * 360
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

    if hue_name in ("Yellow", "Orange") and l < _BROWN_MAX_L and s > _BROWN_MIN_S:
        return "Brown" if l < _DARK_BROWN_L else "Dark Brown"

    if l < _DARK_L:
        qualifier = "Dark"
    elif l < _MID_L:
        qualifier = ""
    else:
        qualifier = "Light"

    if hue_name == "Yellow" and s > _GOLDEN_MIN_S and _GOLDEN_L_MIN < l < _GOLDEN_L_MAX:
        return "Golden Yellow"

    if qualifier:
        return f"{qualifier} {hue_name}"
    return hue_name


async def label_palette_colors_with_llm(
    quantized_png_bytes: bytes,
    colors: list[tuple[int, int, int]],
    character_description: str,
    chat_provider: ChatProvider,
) -> list[str]:
    """Label extracted palette colors using an LLM vision call."""
    color_list = "\n".join(
        f"{idx}. RGB({r}, {g}, {b})" for idx, (r, g, b) in enumerate(colors, start=1)
    )

    from spriteforge.prompts.preprocessor import PALETTE_LABELING_PROMPT

    prompt_text = PALETTE_LABELING_PROMPT.format(
        character_description=character_description,
        color_list=color_list,
        color_count=len(colors),
    )

    image_data_url = image_to_data_url_limited(quantized_png_bytes, max_bytes=4_000_000)
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
        response = await chat_provider.chat(
            messages=messages, temperature=0.5, response_format="json_object"
        )
        data = json.loads(response)
        if "labels" in data:
            labels = data["labels"]
        elif isinstance(data, dict) and len(data) == 1:
            first_value = next(iter(data.values()))
            if isinstance(first_value, list):
                labels = first_value
            else:
                raise ValueError("Response format not recognized")
        else:
            raise ValueError("Response format not recognized")

        if not isinstance(labels, list) or len(labels) != len(colors):
            logger.warning(
                "LLM returned wrong number of labels: %s vs %d colors. Falling back to descriptive names.",
                len(labels) if isinstance(labels, list) else "non-list",
                len(colors),
            )
            return [describe_color(c) for c in colors]

        return [str(label) for label in labels]
    except Exception as exc:
        logger.warning(
            "LLM palette labeling failed: %s. Falling back to descriptive names.",
            exc,
        )
        return [describe_color(c) for c in colors]
