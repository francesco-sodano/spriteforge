"""YAML configuration loading and validation for character spritesheet definitions."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from spriteforge.logging import get_logger
from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)

logger = get_logger("config")


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


def _parse_palette(data: dict) -> PaletteConfig:
    """Parse a YAML palette section into a PaletteConfig.

    Expected YAML shape::

        palette:
          outline:
            symbol: "O"
            name: "Outline"
            rgb: [20, 15, 10]
          colors:
            - symbol: "s"
              name: "Skin"
              rgb: [235, 210, 185]
            ...

    Args:
        data: Parsed YAML dict for the palette section.

    Returns:
        A validated PaletteConfig instance.

    Raises:
        ValueError: If palette structure is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"'palette' section must be a YAML mapping, got {type(data).__name__}"
        )

    kwargs: dict = {"name": "P1"}

    # --- Parse outline ---
    if "outline" in data:
        outline_raw = data["outline"]
        if not isinstance(outline_raw, dict):
            raise ValueError("'palette.outline' must be a mapping")
        rgb = outline_raw.get("rgb", [20, 40, 40])
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise ValueError(
                f"'palette.outline.rgb' must be a list of 3 ints, got {rgb!r}"
            )
        kwargs["outline"] = PaletteColor(
            element=outline_raw.get("name", "Outline"),
            symbol=outline_raw.get("symbol", "O"),
            r=rgb[0],
            g=rgb[1],
            b=rgb[2],
        )

    # --- Parse colors ---
    if "colors" in data:
        colors_raw = data["colors"]
        if not isinstance(colors_raw, list):
            raise ValueError("'palette.colors' must be a YAML sequence")
        colors: list[PaletteColor] = []
        for entry in colors_raw:
            if not isinstance(entry, dict):
                raise ValueError("Each palette color entry must be a mapping")
            rgb = entry.get("rgb")
            if not isinstance(rgb, list) or len(rgb) != 3:
                raise ValueError(
                    f"'palette.colors[].rgb' must be a list of 3 ints, got {rgb!r}"
                )
            colors.append(
                PaletteColor(
                    element=entry.get("name", ""),
                    symbol=entry.get("symbol", ""),
                    r=rgb[0],
                    g=rgb[1],
                    b=rgb[2],
                )
            )
        kwargs["colors"] = colors

    return PaletteConfig(**kwargs)


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

    # --- Type-check top-level sections ---
    if not isinstance(data["character"], dict):
        raise ValueError(
            "'character' section must be a YAML mapping, "
            f"got {type(data['character']).__name__}"
        )
    if not isinstance(data["animations"], list):
        raise ValueError(
            "'animations' section must be a YAML sequence, "
            f"got {type(data['animations']).__name__}"
        )

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

    # --- Build PaletteConfig from YAML palette section ---
    if "palette" in data:
        palette = _parse_palette(data["palette"])
        spec_kwargs["palettes"] = {"P1": palette}

    # --- Build GenerationConfig from YAML generation section ---
    if "generation" in data:
        gen_data = data["generation"]
        if not isinstance(gen_data, dict):
            raise ValueError(
                "'generation' section must be a YAML mapping, "
                f"got {type(gen_data).__name__}"
            )
        spec_kwargs["generation"] = GenerationConfig(**gen_data)

    if "base_image_path" in data:
        spec_kwargs["base_image_path"] = data["base_image_path"]

    # --- Output path (top-level string or nested dict) ---
    if "output_path" in data:
        spec_kwargs["output_path"] = data["output_path"]
    elif "output" in data and isinstance(data["output"], dict):
        spec_kwargs["output_path"] = data["output"].get("path", "")

    spec = SpritesheetSpec(**spec_kwargs)

    num_colors = (
        len(spec.palettes.get("P1", PaletteConfig()).colors) if spec.palettes else 0
    )
    logger.info(
        "Loaded config: %s (%d animations, %d palette colors)",
        spec.character.name,
        len(spec.animations),
        num_colors,
    )

    return spec
