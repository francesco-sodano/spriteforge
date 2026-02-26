"""Deterministic builder utilities for minimal-input SpriteForge configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    SpritesheetSpec,
)

_LOOPING_ACTION_NAMES = {"idle", "walk", "run"}
_GENERATION_FIELD_ORDER = (
    "style",
    "facing",
    "feet_row",
    "outline_width",
    "rules",
    "auto_palette",
    "max_palette_colors",
    "semantic_labels",
    "grid_model",
    "gate_model",
    "labeling_model",
    "reference_model",
    "gate_3a_max_retries",
    "fallback_regen_frames",
    "compact_grid_context",
    "max_image_bytes",
    "request_timeout_seconds",
    "max_anchor_regenerations",
    "anchor_regen_failure_ratio",
    "allow_absolute_output_path",
    "budget",
)


def _normalize_spaces(value: str) -> str:
    return " ".join(value.split())


class MinimalActionInput(BaseModel):
    """Minimal action input used to build an animation row."""

    name: str = Field(min_length=1)
    movement_description: str = Field(min_length=1)
    frames: int = Field(ge=1)
    timing_ms: int = Field(gt=0)
    loop: bool | None = None

    @field_validator("name", "movement_description")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        normalized = _normalize_spaces(value)
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class MinimalConfigInput(BaseModel):
    """Minimal input contract for deterministic config generation."""

    character_name: str = Field(min_length=1)
    base_image_path: str = Field(min_length=1)
    actions: list[MinimalActionInput] = Field(min_length=1)

    @field_validator("character_name", "base_image_path")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        normalized = _normalize_spaces(value)
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


def build_spritesheet_spec_from_minimal_input(
    minimal_input: MinimalConfigInput | dict[str, Any],
) -> SpritesheetSpec:
    """Build a deterministic ``SpritesheetSpec`` from minimal user inputs."""
    payload = (
        minimal_input
        if isinstance(minimal_input, MinimalConfigInput)
        else MinimalConfigInput(**minimal_input)
    )

    animations: list[AnimationDef] = []
    for row, action in enumerate(payload.actions):
        loop = (
            action.loop
            if action.loop is not None
            else action.name.lower() in _LOOPING_ACTION_NAMES
        )
        animations.append(
            AnimationDef(
                name=action.name,
                row=row,
                frames=action.frames,
                loop=loop,
                timing_ms=action.timing_ms,
                hit_frame=None,
                frame_descriptions=[],
                prompt_context=f"{action.name}: {action.movement_description}",
            )
        )

    return SpritesheetSpec(
        character=CharacterConfig(name=payload.character_name),
        animations=animations,
        generation=GenerationConfig(auto_palette=True),
        base_image_path=payload.base_image_path,
        output_path="",
    )


def _serialize_generation(generation: GenerationConfig) -> dict[str, Any]:
    raw = generation.model_dump()
    payload: dict[str, Any] = {}
    for key in _GENERATION_FIELD_ORDER:
        if key == "budget":
            if raw.get("budget") is not None:
                payload[key] = raw[key]
            continue
        if key in raw:
            payload[key] = raw[key]
    for key in sorted(set(raw.keys()) - set(_GENERATION_FIELD_ORDER)):
        payload[key] = raw[key]
    return payload


def serialize_spritesheet_spec_yaml(spec: SpritesheetSpec) -> str:
    """Serialize a ``SpritesheetSpec`` into deterministic project-style YAML."""
    payload: dict[str, Any] = {
        "character": {
            "name": spec.character.name,
            "class": spec.character.character_class,
            "description": spec.character.description,
            "frame_width": spec.character.frame_width,
            "frame_height": spec.character.frame_height,
            "spritesheet_columns": spec.character.spritesheet_columns,
        },
        "animations": [
            {
                "name": animation.name,
                "row": animation.row,
                "frames": animation.frames,
                "loop": animation.loop,
                "timing_ms": animation.timing_ms,
                "hit_frame": animation.hit_frame,
                "frame_descriptions": animation.frame_descriptions,
                "prompt_context": animation.prompt_context,
            }
            for animation in spec.animations
        ],
        "generation": _serialize_generation(spec.generation),
        "base_image_path": spec.base_image_path,
        "output_path": spec.output_path,
    }

    if spec.palette is not None:
        payload["palette"] = {
            "outline": {
                "symbol": spec.palette.outline.symbol,
                "name": spec.palette.outline.element,
                "rgb": list(spec.palette.outline.rgb),
            },
            "colors": [
                {
                    "symbol": color.symbol,
                    "name": color.element,
                    "rgb": list(color.rgb),
                }
                for color in spec.palette.colors
            ],
        }

    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def write_spritesheet_spec_yaml(spec: SpritesheetSpec, output_path: str | Path) -> Path:
    """Write a serialized ``SpritesheetSpec`` YAML file to disk."""
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(serialize_spritesheet_spec_yaml(spec), encoding="utf-8")
    return resolved
