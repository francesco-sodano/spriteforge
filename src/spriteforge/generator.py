"""Grid generator — Stage 2 pixel-precise generation with chat models.

Translates rough reference frames into structured JSON grids of
palette symbols using chat vision input via Azure AI Foundry / Azure OpenAI.
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
from spriteforge.utils import (
    compress_grid_rle,
    image_to_data_url,
    validate_grid_dimensions,
)

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

    for i, row in enumerate(grid):
        if not isinstance(row, str):
            raise GenerationError(f"Grid row {i} is not a string")

    dim_error = validate_grid_dimensions(grid, expected_rows, expected_cols)
    if dim_error is not None:
        raise GenerationError(dim_error)

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
    """Generates pixel-precise palette-indexed grids using chat deployments.

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

    async def close(self) -> None:
        """Close the underlying chat provider."""
        if hasattr(self._chat, "close"):
            await self._chat.close()

    def get_last_usage(self) -> dict[str, int] | None:
        """Return token usage from the most recent chat call, if available."""
        usage = getattr(self._chat, "last_usage", None)
        if not isinstance(usage, dict):
            return None
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

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

        .. deprecated::
            Use :meth:`generate_frame` with ``is_anchor=True`` instead.

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
        return await self.generate_frame(
            reference_frame=reference_frame,
            context=context,
            frame_index=0,
            temperature=temperature,
            additional_guidance=additional_guidance,
            is_anchor=True,
            base_reference=base_reference,
        )

    async def generate_frame(
        self,
        reference_frame: bytes,
        context: FrameContext,
        frame_index: int,
        *,
        is_anchor: bool = False,
        base_reference: bytes | None = None,
        prev_frame_grid: list[str] | None = None,
        prev_frame_rendered: bytes | None = None,
        temperature: float = 1.0,
        additional_guidance: str = "",
    ) -> list[str]:
        """Generate a single pixel-precise frame grid.

        This method handles both anchor frame generation (when ``is_anchor=True``)
        and regular frame generation. The anchor frame (IDLE Frame 0) establishes
        the character's canonical appearance and is used as reference for all
        subsequent frames.

        Args:
            reference_frame: PNG bytes of the rough reference for this frame.
            context: Frame context containing palette, animation, generation config,
                and frame dimensions. For non-anchor frames, must also contain
                anchor_grid and anchor_rendered.
            frame_index: Index of this frame within the row.
            is_anchor: If True, generates the anchor frame (IDLE Frame 0).
            base_reference: PNG bytes of the full base character reference image.
                Required when is_anchor=True, ignored otherwise.
            prev_frame_grid: Grid of the previous frame (for continuity).
                Only used when is_anchor=False.
            prev_frame_rendered: PNG bytes of the rendered previous frame.
                Only used when is_anchor=False.
            temperature: LLM temperature (1.0=creative, 0.3=constrained).
            additional_guidance: Extra prompt text for retry escalation.

        Returns:
            A list of *context.frame_height* strings, each *context.frame_width* characters.

        Raises:
            ValueError: If is_anchor=True but base_reference is None, or if
                is_anchor=False but anchor_rendered/anchor_grid are missing from context.
            GenerationError: If the LLM fails to produce a valid grid.
        """
        palette = context.palette
        animation = context.animation
        generation = context.generation
        frame_width = context.frame_width
        frame_height = context.frame_height

        system_prompt = _build_system_prompt(
            palette, generation, width=frame_width, height=frame_height
        )

        # Extract frame description
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

        # Build user prompt based on anchor vs non-anchor
        if is_anchor:
            # Validate anchor-specific requirements
            if base_reference is None:
                raise ValueError("base_reference is required when is_anchor=True")

            # Build quantized reference section
            quantized_section = ""
            if context.quantized_reference is not None:
                quantized_section = QUANTIZED_REFERENCE_SECTION.format(
                    width=frame_width, height=frame_height
                )

            user_text = build_anchor_frame_prompt(
                animation_name=animation.name,
                animation_context=animation_context,
                frame_description=frame_desc,
                quantized_section=quantized_section,
                additional_guidance=additional_guidance,
            )

            # Build multimodal content for anchor frame
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

            if context.quantized_reference is not None:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(context.quantized_reference)
                        },
                    }
                )
        else:
            # Validate context for non-anchor frame generation
            if context.anchor_rendered is None or context.anchor_grid is None:
                raise ValueError(
                    "anchor_rendered and anchor_grid must be provided in context "
                    "for non-anchor frame generation"
                )

            user_text = build_frame_prompt(
                frame_index=frame_index,
                animation_name=animation.name,
                animation_context=animation_context,
                frame_description=frame_desc,
                additional_guidance=additional_guidance,
            )

            # Build multimodal content for non-anchor frame
            content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(reference_frame)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(context.anchor_rendered)},
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
            compact = context.generation.compact_grid_context
            if compact:
                anchor_grid_text = compress_grid_rle(context.anchor_grid)
                label = "The anchor frame (IDLE F0) grid (RLE-compressed) for identity reference:\n"
            else:
                anchor_grid_text = "\n".join(context.anchor_grid)
                label = "The anchor frame (IDLE F0) grid for identity reference:\n"
            anchor_text = label + anchor_grid_text
            content.append({"type": "text", "text": anchor_text})

            if prev_frame_grid is not None:
                if compact:
                    prev_grid_text = compress_grid_rle(prev_frame_grid)
                    prev_label = (
                        f"Previous frame (frame {frame_index - 1}) grid "
                        "(RLE-compressed) for animation continuity:\n"
                    )
                else:
                    prev_grid_text = "\n".join(prev_frame_grid)
                    prev_label = (
                        f"Previous frame (frame {frame_index - 1}) grid "
                        "for animation continuity:\n"
                    )
                prev_text = prev_label + prev_grid_text
                content.append({"type": "text", "text": prev_text})

        # Log frame generation
        frame_type = "anchor" if is_anchor else "frame"
        logger.info(
            "Generating %s: %s F%d (temp=%.1f)",
            frame_type,
            animation.name,
            frame_index,
            temperature,
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
