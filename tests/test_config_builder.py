"""Tests for minimal-input config building and YAML serialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spriteforge.config import load_config, validate_config
from spriteforge.config_builder import (
    MinimalConfigInput,
    build_spritesheet_spec_from_minimal_input,
    serialize_spritesheet_spec_yaml,
    write_spritesheet_spec_yaml,
)


def test_build_spritesheet_spec_from_minimal_input_defaults() -> None:
    spec = build_spritesheet_spec_from_minimal_input(
        {
            "character_name": "my_character",
            "base_image_path": "base.png",
            "actions": [
                {
                    "name": "idle",
                    "movement_description": "  Breathing   in place ",
                    "frames": 6,
                    "timing_ms": 140,
                },
                {
                    "name": "attack_slash",
                    "movement_description": "Quick forward slash",
                    "frames": 8,
                    "timing_ms": 90,
                },
            ],
        }
    )

    assert spec.character.name == "my_character"
    assert spec.base_image_path == "base.png"
    assert spec.generation.auto_palette is True
    assert [anim.row for anim in spec.animations] == [0, 1]
    assert spec.animations[0].loop is True
    assert spec.animations[1].loop is False
    assert spec.animations[0].prompt_context == "idle: Breathing in place"


def test_serialized_yaml_is_deterministic_and_loadable(tmp_path: Path) -> None:
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")

    spec = build_spritesheet_spec_from_minimal_input(
        {
            "character_name": "my_character",
            "base_image_path": str(base_image),
            "actions": [
                {
                    "name": "idle",
                    "movement_description": "Breathing in place",
                    "frames": 6,
                    "timing_ms": 140,
                },
                {
                    "name": "run",
                    "movement_description": "Fast sprint cycle",
                    "frames": 8,
                    "timing_ms": 90,
                },
            ],
        }
    )

    yaml_text_a = serialize_spritesheet_spec_yaml(spec)
    yaml_text_b = serialize_spritesheet_spec_yaml(spec)
    assert yaml_text_a == yaml_text_b

    config_path = write_spritesheet_spec_yaml(spec, tmp_path / "generated.yaml")
    loaded = load_config(config_path)
    assert loaded.character.name == spec.character.name
    assert [anim.name for anim in loaded.animations] == ["idle", "run"]
    assert validate_config(config_path) == []


@pytest.mark.parametrize("field", ["name", "movement_description"])
def test_minimal_action_input_rejects_blank_text_fields(field: str) -> None:
    action = {
        "name": "idle",
        "movement_description": "Breathing in place",
        "frames": 6,
        "timing_ms": 140,
    }
    action[field] = "   "
    with pytest.raises(ValueError, match="value must not be blank"):
        build_spritesheet_spec_from_minimal_input(
            {
                "character_name": "hero",
                "base_image_path": "base.png",
                "actions": [action],
            }
        )


@pytest.mark.parametrize("field", ["character_name", "base_image_path"])
def test_minimal_config_input_rejects_blank_text_fields(field: str) -> None:
    payload = {
        "character_name": "hero",
        "base_image_path": "base.png",
        "actions": [
            {
                "name": "idle",
                "movement_description": "Breathing in place",
                "frames": 6,
                "timing_ms": 140,
            }
        ],
    }
    payload[field] = "   "
    with pytest.raises(ValueError, match="value must not be blank"):
        MinimalConfigInput(**payload)


def test_minimal_config_input_rejects_empty_actions() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        MinimalConfigInput(
            character_name="hero",
            base_image_path="base.png",
            actions=[],
        )


@pytest.mark.parametrize(
    ("frames", "timing_ms"),
    [
        (0, 140),
        (-1, 140),
        (6, 0),
        (6, -1),
    ],
)
def test_build_from_minimal_input_rejects_invalid_numeric_constraints(
    frames: int, timing_ms: int
) -> None:
    with pytest.raises(ValueError):
        build_spritesheet_spec_from_minimal_input(
            {
                "character_name": "hero",
                "base_image_path": "base.png",
                "actions": [
                    {
                        "name": "idle",
                        "movement_description": "Breathing in place",
                        "frames": frames,
                        "timing_ms": timing_ms,
                    }
                ],
            }
        )


def test_build_spritesheet_spec_respects_explicit_loop_override() -> None:
    spec = build_spritesheet_spec_from_minimal_input(
        {
            "character_name": "hero",
            "base_image_path": "base.png",
            "actions": [
                {
                    "name": "attack_slash",
                    "movement_description": "Quick slash",
                    "frames": 5,
                    "timing_ms": 90,
                    "loop": True,
                }
            ],
        }
    )

    assert spec.animations[0].loop is True


def test_serialize_palette_uses_yaml_name_fields() -> None:
    spec = build_spritesheet_spec_from_minimal_input(
        {
            "character_name": "hero",
            "base_image_path": "base.png",
            "actions": [
                {
                    "name": "idle",
                    "movement_description": "Breathing in place",
                    "frames": 6,
                    "timing_ms": 140,
                }
            ],
        }
    )
    spec.palette = load_config(
        Path(__file__).resolve().parent.parent / "configs/examples/simple_enemy.yaml"
    ).palette
    assert spec.palette is not None
    assert len(spec.palette.colors) > 0

    yaml_text = serialize_spritesheet_spec_yaml(spec)
    payload = yaml.safe_load(yaml_text)

    outline = payload["palette"]["outline"]
    assert "name" in outline
    assert "element" not in outline

    first_color = payload["palette"]["colors"][0]
    assert "name" in first_color
    assert "element" not in first_color
