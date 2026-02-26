"""SpriteForge — AI-powered spritesheet generator for 2D pixel-art games."""

from typing import Any

from spriteforge.budget import CallEstimate, CallTracker, estimate_calls
from spriteforge.config_builder import (
    MinimalActionInput,
    MinimalConfigInput,
    build_spritesheet_spec_from_minimal_input,
    serialize_spritesheet_spec_yaml,
    write_spritesheet_spec_yaml,
)
from spriteforge.config import load_config, validate_config
from spriteforge.errors import (
    BudgetExhaustedError,
    ConfigError,
    GateError,
    GenerationError,
    PaletteError,
    ProviderError,
    RowGenerationError,
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
    BudgetConfig,
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
from spriteforge.providers import ChatProvider, ReferenceProvider
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


def __getattr__(name: str) -> Any:
    """Lazy loading for Azure-dependent classes."""
    if name == "AzureChatProvider":
        from spriteforge.providers import AzureChatProvider

        return AzureChatProvider
    if name == "GPTImageProvider":
        from spriteforge.providers import GPTImageProvider

        return GPTImageProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnimationDef",
    "AzureChatProvider",
    "BudgetConfig",
    "BudgetExhaustedError",
    "CallEstimate",
    "CallTracker",
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
    "MinimalActionInput",
    "MinimalConfigInput",
    "PaletteColor",
    "PaletteConfig",
    "PaletteError",
    "PreprocessResult",
    "ProgrammaticChecker",
    "ProviderError",
    "RowGenerationError",
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
    "build_spritesheet_spec_from_minimal_input",
    "create_workflow",
    "estimate_calls",
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
    "serialize_spritesheet_spec_yaml",
    "strip_code_fences",
    "swap_palette_grid",
    "validate_config",
    "validate_grid_symbols",
    "validate_reference_image",
    "write_spritesheet_spec_yaml",
]
