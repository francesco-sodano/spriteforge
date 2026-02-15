"""Prompt templates for Stage 1 reference image generation.

Contains the reference strip prompt builder — originally defined inline in
:pymod:`spriteforge.providers.gpt_image`.
"""

from __future__ import annotations

from spriteforge.models import AnimationDef, CharacterConfig


def build_reference_prompt(
    animation: AnimationDef,
    character: CharacterConfig,
    character_description: str = "",
) -> str:
    """Build the prompt for reference strip generation (Stage 1).

    Args:
        animation: The animation row spec.
        character: The character spec.
        character_description: Optional detailed character description.

    Returns:
        A text prompt suitable for GPT-Image-1.5.
    """
    parts: list[str] = []

    parts.append(
        f"Create a horizontal animation strip of {animation.frames} frames "
        f"for a {character.frame_width}x{character.frame_height} pixel-art character."
    )

    parts.append(f"Character: {character.name}.")

    if character.character_class:
        parts.append(f"Class: {character.character_class}.")

    if character_description:
        parts.append(f"Description: {character_description}")

    parts.append(f"Animation: {animation.name}.")

    if animation.prompt_context:
        parts.append(f"Animation context: {animation.prompt_context}")

    parts.append(
        f"Show {animation.frames} frames arranged side by side, left to right. "
        "Each frame should clearly show the character pose for this animation step. "
        "Transparent background."
    )

    return " ".join(parts)
