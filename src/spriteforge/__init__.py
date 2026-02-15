"""SpriteForge — AI-powered spritesheet generator for 2D pixel-art games."""

from spriteforge.config import load_config
from spriteforge.errors import (
    ConfigError,
    GateError,
    GenerationError,
    PaletteError,
    ProviderError,
    RenderError,
    RetryExhaustedError,
    SpriteForgeError,
)
from spriteforge.gates import (
    GateVerdict,
    LLMGateChecker,
    ProgrammaticChecker,
    parse_verdict_response,
)
from spriteforge.generator import GridGenerator, parse_grid_response
from spriteforge.logging import get_logger, setup_logging
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
    label_palette_colors_with_llm,
    preprocess_reference,
    resize_reference,
    validate_reference_image,
)
from spriteforge.providers import (
    AzureChatProvider,
    ChatProvider,
    GPTImageProvider,
    ReferenceProvider,
)
from spriteforge.renderer import (
    frame_to_png_bytes,
    render_frame,
    render_row_strip,
)
from spriteforge.retry import (
    RetryConfig,
    RetryContext,
    RetryManager,
    RetryTier,
)
from spriteforge.utils import (
    image_to_base64,
    image_to_data_url,
    parse_json_from_llm,
    strip_code_fences,
)
from spriteforge.workflow import SpriteForgeWorkflow, create_workflow

__all__ = [
    "AnimationDef",
    "AzureChatProvider",
    "CharacterConfig",
    "ChatProvider",
    "ConfigError",
    "GPTImageProvider",
    "GateError",
    "GateVerdict",
    "GenerationConfig",
    "GenerationError",
    "GridGenerator",
    "LLMGateChecker",
    "PaletteColor",
    "PaletteConfig",
    "PaletteError",
    "PreprocessResult",
    "ProgrammaticChecker",
    "ProviderError",
    "ReferenceProvider",
    "RenderError",
    "RetryConfig",
    "RetryContext",
    "RetryExhaustedError",
    "RetryManager",
    "RetryTier",
    "SpritesheetSpec",
    "SpriteForgeError",
    "SpriteForgeWorkflow",
    "build_palette_map",
    "create_workflow",
    "extract_palette_from_image",
    "frame_to_png_bytes",
    "get_logger",
    "image_to_base64",
    "image_to_data_url",
    "load_config",
    "parse_grid_response",
    "parse_json_from_llm",
    "parse_verdict_response",
    "preprocess_reference",
    "render_frame",
    "render_row_strip",
    "resize_reference",
    "setup_logging",
    "strip_code_fences",
    "swap_palette_grid",
    "validate_grid_symbols",
    "validate_reference_image",
]
