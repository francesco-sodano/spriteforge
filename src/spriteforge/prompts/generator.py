"""Prompt constants and builders for Stage 2 grid generation.

Contains the system prompt template, anchor frame prompt template, quantized
reference section, and regular frame prompt template — all originally defined
inline in :pymod:`spriteforge.generator`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

GRID_SYSTEM_PROMPT: str = """\
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

# ---------------------------------------------------------------------------
# Anchor frame prompt template
# ---------------------------------------------------------------------------

ANCHOR_FRAME_PROMPT: str = """\
Generate the **anchor frame** (Row 0, Frame 0) for the animation "{animation_name}".

{animation_context}

This is the single most important frame: it establishes the character's \
canonical pixel-level appearance. All subsequent frames will be verified \
against it for identity consistency.

{frame_description}

{quantized_section}
"""

# ---------------------------------------------------------------------------
# Quantized reference prompt section
# ---------------------------------------------------------------------------

QUANTIZED_REFERENCE_SECTION: str = """\
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

# ---------------------------------------------------------------------------
# Frame prompt template
# ---------------------------------------------------------------------------

FRAME_PROMPT: str = """\
Generate frame **{frame_index}** for the animation "{animation_name}".

{animation_context}

{frame_description}

{additional_guidance}
"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_anchor_frame_prompt(
    animation_name: str,
    animation_context: str,
    frame_description: str,
    quantized_section: str,
) -> str:
    """Build the prompt for generating an anchor frame (Row 0, Frame 0).

    Args:
        animation_name: Name of the animation (e.g., "IDLE").
        animation_context: Contextual description of the animation.
        frame_description: Description of this specific frame.
        quantized_section: Pre-built quantized reference section (or empty).

    Returns:
        Formatted prompt string for anchor frame generation.
    """
    return ANCHOR_FRAME_PROMPT.format(
        animation_name=animation_name,
        animation_context=animation_context,
        frame_description=frame_description,
        quantized_section=quantized_section,
    )


def build_frame_prompt(
    frame_index: int,
    animation_name: str,
    animation_context: str,
    frame_description: str,
    additional_guidance: str,
) -> str:
    """Build the prompt for generating a subsequent animation frame.

    Args:
        frame_index: Current frame index within the animation.
        animation_name: Name of the animation (e.g., "WALK").
        animation_context: Contextual description of the animation.
        frame_description: Description of this specific frame.
        additional_guidance: Extra prompt text for retry escalation.

    Returns:
        Formatted prompt string for animation frame generation.
    """
    return FRAME_PROMPT.format(
        frame_index=frame_index,
        animation_name=animation_name,
        animation_context=animation_context,
        frame_description=frame_description,
        additional_guidance=additional_guidance,
    )
