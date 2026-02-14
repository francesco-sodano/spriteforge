"""Grid generator — Stage 2 pixel-precise generation with Claude Opus 4.6.

Translates rough reference frames into structured 64×64 JSON grids of
palette symbols using Claude Opus 4.6 with vision input via Azure AI Foundry.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from spriteforge.errors import GenerationError
from spriteforge.models import AnimationDef, GenerationConfig, PaletteConfig
from spriteforge.utils import image_to_data_url

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert pixel artist. Your job is to translate a visual reference \
image into a precise {width}×{height} grid of single-character palette symbols.

## Output format
Return **only** valid JSON with a single key `"grid"` whose value is a list of \
{height} strings, each exactly {width} characters long. Example (4×4):
{{"grid":["....","..O.",".sO.","...."]}}

## Palette symbols
{palette_map_text}

## Art style
{style}

## Rules
- The character faces **{facing}**.
- Feet touch row y≈{feet_row} (0-indexed from top).
- Every non-transparent sprite pixel must have a 1-pixel dark outline \
(symbol `{outline_symbol}`).
- The background MUST be transparent (`.`).
- Use ONLY the palette symbols listed above — no other characters.
{extra_rules}
"""

ANCHOR_FRAME_PROMPT_TEMPLATE = """\
Generate the **anchor frame** (Row 0, Frame 0) for the animation "{animation_name}".

{animation_context}

This is the single most important frame: it establishes the character's \
canonical pixel-level appearance. All subsequent frames will be verified \
against it for identity consistency.

{frame_description}

{quantized_section}
"""

QUANTIZED_REFERENCE_PROMPT_SECTION = """\
## Pixel-Level Reference Guide
The attached quantized reference image shows the character at the exact target \
resolution ({width}×{height} pixels) with a reduced color palette. Use this as \
your primary spatial guide:

- Match the character's outline, proportions, and position as closely as possible
- Map each color region in the quantized image to the corresponding palette symbol
- The quantized image's colors are approximate — use the exact palette symbols provided
- Refine details (subpixel accuracy, outline consistency, transparent background) \
that the quantization may have degraded

Think of this as "tracing" the quantized reference while applying the precise \
palette symbols and pixel-art cleanup rules.
"""

FRAME_PROMPT_TEMPLATE = """\
Generate frame **{frame_index}** for the animation "{animation_name}".

{animation_context}

{frame_description}

{additional_guidance}
"""


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
    return SYSTEM_PROMPT.format(
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
    """Generates pixel-precise 64×64 palette-indexed grids using Claude Opus 4.6.

    Uses Azure AI Foundry to call Claude Opus 4.6 with vision input,
    translating rough reference frames into structured JSON grids.
    """

    def __init__(
        self,
        project_endpoint: str | None = None,
        model_deployment: str = "claude-opus-4-6",
    ) -> None:
        """Initialize the grid generator.

        Args:
            project_endpoint: Azure AI Foundry project endpoint.
                If ``None``, reads from ``AZURE_AI_PROJECT_ENDPOINT`` env var.
            model_deployment: Model deployment name in Foundry.
        """
        self._endpoint = project_endpoint or os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT", ""
        )
        self._model_deployment = model_deployment

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    async def generate_anchor_frame(
        self,
        base_reference: bytes,
        reference_frame: bytes,
        palette: PaletteConfig,
        animation: AnimationDef,
        generation: GenerationConfig | None = None,
        quantized_reference: bytes | None = None,
    ) -> list[str]:
        """Generate the IDLE Frame 0 anchor — the identity reference for all other frames.

        This is the most important frame: it establishes the character's
        canonical appearance at pixel level. All subsequent frames are
        verified against it.

        When *quantized_reference* is provided (from the preprocessor), the
        LLM uses it as a pixel-level guide — "trace and refine" rather than
        "imagine from scratch".

        Args:
            base_reference: PNG bytes of the full base character reference image.
            reference_frame: PNG bytes of the rough reference for IDLE frame 0.
            palette: Palette config with symbol → RGBA mappings.
            animation: The IDLE animation definition.
            generation: Generation config (style, facing, feet_row, rules).
                If ``None``, defaults are used.
            quantized_reference: Optional 64×64 quantized PNG from preprocessor.

        Returns:
            A list of 64 strings, each 64 characters long (palette symbols).
        """
        gen = generation or GenerationConfig()

        system_prompt = _build_system_prompt(palette, gen)

        # Build quantized reference section
        quantized_section = ""
        if quantized_reference is not None:
            quantized_section = QUANTIZED_REFERENCE_PROMPT_SECTION.format(
                width=64, height=64
            )

        frame_desc = ""
        if animation.frame_descriptions:
            frame_desc = f"Frame description: {animation.frame_descriptions[0]}"

        animation_context = ""
        if animation.prompt_context:
            animation_context = f"Animation context: {animation.prompt_context}"

        user_text = ANCHOR_FRAME_PROMPT_TEMPLATE.format(
            animation_name=animation.name,
            animation_context=animation_context,
            frame_description=frame_desc,
            quantized_section=quantized_section,
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

        response_text = await self._call_llm(messages, temperature=1.0)
        return parse_grid_response(response_text)

    async def generate_frame(
        self,
        reference_frame: bytes,
        anchor_grid: list[str],
        anchor_rendered: bytes,
        palette: PaletteConfig,
        animation: AnimationDef,
        frame_index: int,
        generation: GenerationConfig | None = None,
        prev_frame_grid: list[str] | None = None,
        prev_frame_rendered: bytes | None = None,
        temperature: float = 1.0,
        additional_guidance: str = "",
    ) -> list[str]:
        """Generate a single pixel-precise frame grid.

        Args:
            reference_frame: PNG bytes of the rough reference for this frame.
            anchor_grid: The IDLE F0 anchor grid (for prompt context).
            anchor_rendered: PNG bytes of the rendered anchor frame.
            palette: Palette config.
            animation: Animation definition for this row.
            frame_index: Index of this frame within the row.
            generation: Generation config. If ``None``, defaults are used.
            prev_frame_grid: Grid of the previous frame (for continuity).
            prev_frame_rendered: PNG bytes of the rendered previous frame.
            temperature: LLM temperature (1.0=creative, 0.3=constrained).
            additional_guidance: Extra prompt text for retry escalation.

        Returns:
            A list of 64 strings, each 64 characters long.

        Raises:
            GenerationError: If the LLM fails to produce a valid grid.
        """
        gen = generation or GenerationConfig()

        system_prompt = _build_system_prompt(palette, gen)

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

        user_text = FRAME_PROMPT_TEMPLATE.format(
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

        response_text = await self._call_llm(messages, temperature=temperature)
        return parse_grid_response(response_text)

    # ------------------------------------------------------------------
    # Internal LLM call
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
    ) -> str:
        """Send a chat completion request to Claude Opus 4.6 via Azure AI Foundry.

        Args:
            messages: OpenAI-style messages list.
            temperature: Sampling temperature.

        Returns:
            The text content of the first choice.

        Raises:
            GenerationError: If the API call fails or returns no content.
        """
        from azure.ai.projects.aio import AIProjectClient  # type: ignore[import-untyped]
        from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-untyped]

        if not self._endpoint:
            raise GenerationError(
                "No Azure AI Foundry endpoint configured. "
                "Set AZURE_AI_PROJECT_ENDPOINT or pass project_endpoint."
            )

        credential = DefaultAzureCredential()
        try:
            project_client = AIProjectClient(
                credential=credential,
                endpoint=self._endpoint,
            )
            try:
                openai_client = project_client.get_openai_client()
                try:
                    response = await openai_client.chat.completions.create(
                        model=self._model_deployment,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=16384,
                    )
                finally:
                    await openai_client.close()
            finally:
                await project_client.close()
        finally:
            await credential.close()

        if (
            not response.choices
            or not response.choices[0].message
            or not response.choices[0].message.content
        ):
            raise GenerationError("LLM returned no content")

        return str(response.choices[0].message.content)
