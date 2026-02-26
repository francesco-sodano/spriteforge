"""Preprocessing modules split by concern."""

from spriteforge.preprocessing.image_io import (
    resize_reference,
    validate_reference_image,
)
from spriteforge.preprocessing.description import (
    deterministic_description_fallback,
    draft_character_description_from_image,
)
from spriteforge.preprocessing.labeling import (
    SYMBOL_POOL,
    label_palette_colors_with_llm,
)
from spriteforge.preprocessing.palette_quantization import (
    _assign_symbols,
    extract_palette_from_image,
)
from spriteforge.preprocessing.pipeline import PreprocessResult, preprocess_reference

__all__ = [
    "PreprocessResult",
    "SYMBOL_POOL",
    "deterministic_description_fallback",
    "draft_character_description_from_image",
    "_assign_symbols",
    "extract_palette_from_image",
    "label_palette_colors_with_llm",
    "preprocess_reference",
    "resize_reference",
    "validate_reference_image",
]
