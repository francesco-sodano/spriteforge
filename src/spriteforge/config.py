"""YAML configuration loading and validation for character spritesheet definitions."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from spriteforge.models import AnimationDef, CharacterConfig, SpritesheetSpec


def validate_config_path(path: str | Path) -> Path:
    """Resolve and validate that a config file path exists.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A resolved ``Path`` object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return resolved


def _parse_yaml(path: Path) -> dict:
    """Read and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ValueError: If the YAML is malformed or not a mapping.
    """
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Malformed YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at top level, got {type(data).__name__}"
        )

    return data


def load_config(path: str | Path) -> SpritesheetSpec:
    """Load and validate a spritesheet configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated ``SpritesheetSpec`` instance.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValidationError: If the YAML content fails Pydantic validation.
        ValueError: If the YAML is malformed or missing required sections.
    """
    resolved = validate_config_path(path)
    data = _parse_yaml(resolved)

    # --- Validate required top-level sections ---
    if "character" not in data:
        raise ValueError("Missing required 'character' section in config")
    if "animations" not in data:
        raise ValueError("Missing required 'animations' section in config")

    # --- Build CharacterConfig ---
    char_raw = data["character"].copy()

    # Map YAML 'class' → model 'character_class'
    if "class" in char_raw:
        char_raw["character_class"] = char_raw.pop("class")

    # Map YAML 'frame_size' → model 'frame_width' / 'frame_height'
    if "frame_size" in char_raw:
        fs = char_raw.pop("frame_size")
        if not isinstance(fs, list) or len(fs) != 2:
            raise ValueError(
                f"'frame_size' must be a list of [width, height], got {fs!r}"
            )
        char_raw["frame_width"] = fs[0]
        char_raw["frame_height"] = fs[1]

    character = CharacterConfig(**char_raw)

    # --- Build AnimationDef list ---
    animations: list[AnimationDef] = []
    seen_rows: set[int] = set()
    for anim_raw in data["animations"]:
        anim = AnimationDef(**anim_raw)
        if anim.row in seen_rows:
            raise ValueError(f"Duplicate row index {anim.row} in animations")
        seen_rows.add(anim.row)
        animations.append(anim)

    # Sort animations by row index
    animations.sort(key=lambda a: a.row)

    # --- Build SpritesheetSpec ---
    spec_kwargs: dict = {
        "character": character,
        "animations": animations,
    }

    if "base_image_path" in data:
        spec_kwargs["base_image_path"] = data["base_image_path"]
    if "output_path" in data:
        spec_kwargs["output_path"] = data["output_path"]

    return SpritesheetSpec(**spec_kwargs)
