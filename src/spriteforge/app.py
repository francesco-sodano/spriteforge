"""Programmatic API entry point for SpriteForge."""

from __future__ import annotations

from pathlib import Path

from spriteforge.config import load_config
from spriteforge.preprocessor import preprocess_reference
from spriteforge.workflow import create_workflow


def _default_output_path(character_name: str) -> Path:
    """Build the default output path for a character."""
    slug = character_name.strip().lower().replace(" ", "_")
    return Path(f"output/{slug}_spritesheet.png")


async def run_spriteforge(
    config_path: Path,
    base_image_path: Path,
    output_path: Path | None = None,
    debug: bool = False,
) -> Path:
    """Programmatic entry point for SpriteForge.

    Args:
        config_path: Path to character YAML config.
        base_image_path: Path to base reference image.
        output_path: Output path. Auto-generated if None.
        debug: Reserved for debug artifact support.

    Returns:
        Path to the generated spritesheet.
    """
    # Kept for forward compatibility with CLI/API contracts; debug artifact
    # persistence is wired in a follow-up without changing this signature.
    _ = debug
    spec = load_config(config_path)

    if not base_image_path.is_file():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")

    resolved_output = output_path or _default_output_path(spec.character.name)

    preprocessor = preprocess_reference if spec.generation.auto_palette else None

    async with await create_workflow(
        config=spec, preprocessor=preprocessor
    ) as workflow:
        return await workflow.run(
            base_reference_path=base_image_path,
            output_path=resolved_output,
        )
