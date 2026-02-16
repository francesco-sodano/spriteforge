"""Grid generator — Stage 2 pixel-precise generation with Claude Opus 4.6.

Translates rough reference frames into structured JSON grids of
palette symbols using Claude Opus 4.6 with vision input via Azure AI Foundry.
Grid dimensions are configurable (default 64×64) via ``context.frame_width`` and
``context.frame_height`` passed from the character configuration.
"""

from __future__ import annotations

import json
import re
from typing import Any

from spriteforge.errors import GenerationError
from spriteforge.logging import get_logger
from spriteforge.models import (
    AnimationDef,
    FrameContext,
    GenerationConfig,
    PaletteConfig,
)
from spriteforge.prompts.generator import (
    GRID_SYSTEM_PROMPT,
    QUANTIZED_REFERENCE_SECTION,
    build_anchor_frame_prompt,
    build_frame_prompt,
)
from spriteforge.providers.chat import ChatProvider
from spriteforge.utils import image_to_data_url

logger = get_logger("generator")

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_grid_response(
    response_text: str,
    expected_rows: int = 64,
    expected_cols: int = 64,
) -> list[str]:
    """Parse the LLM's JSON response into a grid.

    Expects JSON with a ``"grid"`` key containing a list of strings.
    Also handles common LLM output quirks (markdown fences, extra whitespace).

    Args:
        response_text: Raw text response from the LLM.
        expected_rows: Expected number of rows in the grid.
        expected_cols: Expected number of columns per row.

    Returns:
        List of *expected_rows* strings, each *expected_cols* characters.

    Raises:
        GenerationError: If parsing fails or grid has wrong dimensions.
    """
    text = response_text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    # Attempt JSON parse
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse grid JSON: %s", exc)
        raise GenerationError(f"Failed to parse grid JSON: {exc}") from exc

    if not isinstance(data, dict) or "grid" not in data:
        raise GenerationError("Response JSON must contain a 'grid' key")

    grid: list[str] = data["grid"]

    if not isinstance(grid, list):
        raise GenerationError("'grid' value must be a list of strings")

    if len(grid) != expected_rows:
        raise GenerationError(f"Grid must have {expected_rows} rows, got {len(grid)}")

    for i, row in enumerate(grid):
        if not isinstance(row, str):
            raise GenerationError(f"Grid row {i} is not a string")
        if len(row) != expected_cols:
            raise GenerationError(
                f"Grid row {i} must have {expected_cols} characters, " f"got {len(row)}"
            )

    return grid


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _build_palette_map_text(palette: PaletteConfig) -> str:
    """Build a human-readable palette symbol table for inclusion in prompts."""
    lines: list[str] = []
    lines.append(f"- `{palette.transparent_symbol}` → Transparent (background)")
    lines.append(
        f"- `{palette.outline.symbol}` → {palette.outline.element} "
        f"(RGB {palette.outline.r},{palette.outline.g},{palette.outline.b})"
    )
    for color in palette.colors:
        lines.append(
            f"- `{color.symbol}` → {color.element} "
            f"(RGB {color.r},{color.g},{color.b})"
        )
    return "\n".join(lines)


def _build_system_prompt(
    palette: PaletteConfig,
    generation: GenerationConfig,
    width: int = 64,
    height: int = 64,
) -> str:
    """Build the system prompt from palette and generation config."""
    palette_map_text = _build_palette_map_text(palette)
    extra_rules = generation.rules if generation.rules else ""
    return GRID_SYSTEM_PROMPT.format(
        width=width,
        height=height,
        palette_map_text=palette_map_text,
        style=generation.style,
        facing=generation.facing,
        feet_row=generation.feet_row,
        outline_symbol=palette.outline.symbol,
        extra_rules=extra_rules,
    )


# ---------------------------------------------------------------------------
# GridGenerator class
# ---------------------------------------------------------------------------


class GridGenerator:
    """Generates pixel-precise palette-indexed grids using Claude Opus 4.6.

    Uses a ``ChatProvider`` to call the LLM with vision input,
    translating rough reference frames into structured JSON grids.
    Grid dimensions are configurable via ``frame_width`` and ``frame_height``.
    """

    def __init__(
        self,
        chat_provider: ChatProvider,
    ) -> None:
        """Initialize the grid generator.

        Args:
            chat_provider: Chat provider for LLM calls.
        """
        self._chat = chat_provider

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    async def generate_anchor_frame(
        self,
        base_reference: bytes,
        reference_frame: bytes,
        context: FrameContext,
        temperature: float = 1.0,
        additional_guidance: str = "",
    ) -> list[str]:
        """Generate the IDLE Frame 0 anchor — the identity reference for all other frames.

        This is the most important frame: it establishes the character's
        canonical appearance at pixel level. All subsequent frames are
        verified against it.

        When *context.quantized_reference* is provided (from the preprocessor), the
        LLM uses it as a pixel-level guide — "trace and refine" rather than
        "imagine from scratch".

        Args:
            base_reference: PNG bytes of the full base character reference image.
            reference_frame: PNG bytes of the rough reference for IDLE frame 0.
            context: Frame context containing palette, animation, generation config,
                frame dimensions, and optional quantized_reference.
            temperature: LLM temperature (1.0=creative, 0.3=constrained).
            additional_guidance: Extra prompt text for retry escalation.

        Returns:
            A list of *context.frame_height* strings, each *context.frame_width* characters.
        """
        palette = context.palette
        animation = context.animation
        generation = context.generation
        frame_width = context.frame_width
        frame_height = context.frame_height
        quantized_reference = context.quantized_reference

        system_prompt = _build_system_prompt(
            palette, generation, width=frame_width, height=frame_height
        )

        # Build quantized reference section
        quantized_section = ""
        if quantized_reference is not None:
            quantized_section = QUANTIZED_REFERENCE_SECTION.format(
                width=frame_width, height=frame_height
            )

        frame_desc = ""
        if animation.frame_descriptions:
            frame_desc = f"Frame description: {animation.frame_descriptions[0]}"

        animation_context = ""
        if animation.prompt_context:
            animation_context = f"Animation context: {animation.prompt_context}"

        user_text = build_anchor_frame_prompt(
            animation_name=animation.name,
            animation_context=animation_context,
            frame_description=frame_desc,
            quantized_section=quantized_section,
            additional_guidance=additional_guidance,
        )

        # Build multimodal content parts
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(base_reference)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(reference_frame)},
            },
        ]

        if quantized_reference is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(quantized_reference)},
                }
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        response_text = await self._chat.chat(
            messages, temperature=temperature, response_format="json_object"
        )
        grid = parse_grid_response(
            response_text, expected_rows=frame_height, expected_cols=frame_width
        )
        logger.debug("Grid response: %d rows, all valid symbols", len(grid))
        return grid

    async def generate_frame(
        self,
        reference_frame: bytes,
        context: FrameContext,
        frame_index: int,
        prev_frame_grid: list[str] | None = None,
        prev_frame_rendered: bytes | None = None,
        temperature: float = 1.0,
        additional_guidance: str = "",
    ) -> list[str]:
        """Generate a single pixel-precise frame grid.

        Args:
            reference_frame: PNG bytes of the rough reference for this frame.
            context: Frame context containing anchor_grid, anchor_rendered, palette,
                animation, generation config, and frame dimensions.
            frame_index: Index of this frame within the row.
            prev_frame_grid: Grid of the previous frame (for continuity).
            prev_frame_rendered: PNG bytes of the rendered previous frame.
            temperature: LLM temperature (1.0=creative, 0.3=constrained).
            additional_guidance: Extra prompt text for retry escalation.

        Returns:
            A list of *context.frame_height* strings, each *context.frame_width* characters.

        Raises:
            GenerationError: If the LLM fails to produce a valid grid.
        """
        palette = context.palette
        animation = context.animation
        generation = context.generation
        frame_width = context.frame_width
        frame_height = context.frame_height
        anchor_grid = context.anchor_grid
        anchor_rendered = context.anchor_rendered

        system_prompt = _build_system_prompt(
            palette, generation, width=frame_width, height=frame_height
        )

        # Validate context for non-anchor frame generation
        if anchor_rendered is None or anchor_grid is None:
            raise ValueError(
                "anchor_rendered and anchor_grid must be provided in context "
                "for non-anchor frame generation"
            )

        frame_desc = ""
        if animation.frame_descriptions and frame_index < len(
            animation.frame_descriptions
        ):
            frame_desc = (
                f"Frame description: {animation.frame_descriptions[frame_index]}"
            )

        animation_context = ""
        if animation.prompt_context:
            animation_context = f"Animation context: {animation.prompt_context}"

        user_text = build_frame_prompt(
            frame_index=frame_index,
            animation_name=animation.name,
            animation_context=animation_context,
            frame_description=frame_desc,
            additional_guidance=additional_guidance,
        )

        # Build multimodal content
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(reference_frame)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(anchor_rendered)},
            },
        ]

        if prev_frame_rendered is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(prev_frame_rendered)},
                }
            )

        # Add anchor grid and previous frame grid as text context
        anchor_text = (
            "The anchor frame (IDLE F0) grid for identity reference:\n"
            + "\n".join(anchor_grid)
        )
        content.append({"type": "text", "text": anchor_text})

        if prev_frame_grid is not None:
            prev_text = (
                f"Previous frame (frame {frame_index - 1}) grid "
                "for animation continuity:\n" + "\n".join(prev_frame_grid)
            )
            content.append({"type": "text", "text": prev_text})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        logger.info(
            "Generating frame: %s F%d (temp=%.1f)",
            animation.name,
            frame_index,
            temperature,
        )

        response_text = await self._chat.chat(
            messages, temperature=temperature, response_format="json_object"
        )
        grid = parse_grid_response(
            response_text, expected_rows=frame_height, expected_cols=frame_width
        )
        logger.debug("Grid response: %d rows, all valid symbols", len(grid))
        return grid
