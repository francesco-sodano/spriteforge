"""Tests for minimal-input config building and YAML serialization."""

from __future__ import annotations

from pathlib import Path

from spriteforge.config import load_config, validate_config
from spriteforge.config_builder import (
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
