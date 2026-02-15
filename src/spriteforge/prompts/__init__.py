"""Prompt constants and builders for SpriteForge LLM interactions.

This package consolidates all prompt strings used across the pipeline —
grid generation (Stage 2), verification gates, retry escalation, and
reference image generation (Stage 1).

All prompts are Python f-string constants or builder functions — no
template engines are used.
"""

from __future__ import annotations

from spriteforge.prompts.gates import (
    GATE_0_PROMPT,
    GATE_1_PROMPT,
    GATE_2_PROMPT,
    GATE_3A_PROMPT,
    GATE_MINUS_1_PROMPT,
    GATE_VERDICT_SCHEMA,
)
from spriteforge.prompts.generator import (
    ANCHOR_FRAME_PROMPT,
    FRAME_PROMPT,
    GRID_SYSTEM_PROMPT,
    QUANTIZED_REFERENCE_SECTION,
    build_anchor_frame_prompt,
    build_frame_prompt,
)
from spriteforge.prompts.providers import build_reference_prompt
from spriteforge.prompts.retry import (
    build_constrained_guidance,
    build_guided_guidance,
    build_soft_guidance,
)

__all__ = [
    "ANCHOR_FRAME_PROMPT",
    "FRAME_PROMPT",
    "GATE_0_PROMPT",
    "GATE_1_PROMPT",
    "GATE_2_PROMPT",
    "GATE_3A_PROMPT",
    "GATE_MINUS_1_PROMPT",
    "GATE_VERDICT_SCHEMA",
    "GRID_SYSTEM_PROMPT",
    "QUANTIZED_REFERENCE_SECTION",
    "build_anchor_frame_prompt",
    "build_constrained_guidance",
    "build_frame_prompt",
    "build_guided_guidance",
    "build_reference_prompt",
    "build_soft_guidance",
]
