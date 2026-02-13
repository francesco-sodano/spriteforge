"""SpriteForge — AI-powered spritesheet generator for 2D pixel-art games."""

from spriteforge.config import load_config
from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)
from spriteforge.palette import (
    build_palette_map,
    swap_palette_grid,
    validate_grid_symbols,
)
from spriteforge.preprocessor import (
    PreprocessResult,
    extract_palette_from_image,
    preprocess_reference,
    resize_reference,
    validate_reference_image,
)
from spriteforge.renderer import (
    frame_to_png_bytes,
    render_frame,
    render_row_strip,
    render_spritesheet,
)

__all__ = [
    "AnimationDef",
    "CharacterConfig",
    "GenerationConfig",
    "PaletteColor",
    "PaletteConfig",
    "PreprocessResult",
    "SpritesheetSpec",
    "build_palette_map",
    "extract_palette_from_image",
    "frame_to_png_bytes",
    "load_config",
    "preprocess_reference",
    "render_frame",
    "render_row_strip",
    "render_spritesheet",
    "resize_reference",
    "swap_palette_grid",
    "validate_grid_symbols",
    "validate_reference_image",
]
