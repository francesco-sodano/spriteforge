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

    Pydantic aliases handle field name translation (YAML → model).
    All type/constraint validation is delegated to Pydantic's
    ``PaletteColor`` and ``PaletteConfig`` models.

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

    if "outline" in data:
        outline_raw = data["outline"]
        if not isinstance(outline_raw, dict):
            raise ValueError("'palette.outline' must be a mapping")
        # Set defaults for outline if not provided
        outline_raw.setdefault("name", "Outline")
        outline_raw.setdefault("symbol", "O")
        kwargs["outline"] = PaletteColor(**outline_raw)

    if "colors" in data:
        colors_raw = data["colors"]
        if not isinstance(colors_raw, list):
            raise ValueError("'palette.colors' must be a YAML sequence")
        kwargs["colors"] = [PaletteColor(**entry) for entry in colors_raw]

    return PaletteConfig(**kwargs)


def load_config(path: str | Path) -> SpritesheetSpec:
    """Load and validate a spritesheet configuration from a YAML file.

    Structural validation (e.g., duplicate row indices) is handled by
    ``SpritesheetSpec``'s model validators during construction.

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
    # Pydantic aliases handle field name translation (YAML → model)
    character = CharacterConfig(**data["character"])

    # --- Build AnimationDef list ---
    animations = [AnimationDef(**anim_raw) for anim_raw in data["animations"]]

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
        spec_kwargs["palette"] = palette

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

    num_colors = len(spec.palette.colors) if spec.palette else 0
    logger.info(
        "Loaded config: %s (%d animations, %d palette colors)",
        spec.character.name,
        len(spec.animations),
        num_colors,
    )

    return spec


def validate_config(
    path: str | Path,
    *,
    check_base_image: bool = True,
) -> list[str]:
    """Validate a character config file without running the pipeline.

    Performs all checks that ``load_config()`` does (YAML parsing,
    Pydantic schema validation) plus additional semantic checks:

    - Palette symbol uniqueness (no duplicate symbols across outline + colors)
    - Animation row indices are contiguous starting from 0
    - Frame counts are > 0 and ≤ max_columns
    - Base image path exists (when ``check_base_image=True``)
    - Model deployment names are non-empty (when present)
    - Outline symbol is "O" (conventional)

    Args:
        path: Path to the YAML configuration file.
        check_base_image: Whether to verify the base image file exists.

    Returns:
        List of warning strings (empty if no warnings).
        Warnings are non-fatal issues (e.g., missing optional sections).

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If any validation check fails.
        ValidationError: If Pydantic schema validation fails.
    """
    # First, load the config to run standard validation
    # This will raise FileNotFoundError, ValueError, or ValidationError if basic checks fail
    spec = load_config(path)

    warnings: list[str] = []

    # --- Check 1: Duplicate palette symbols ---
    # This is already checked by PaletteConfig._no_duplicate_symbols() model validator
    # during load_config(), so we don't need to duplicate it here.
    # However, we can add additional checks for cross-palette uniqueness if needed later.

    # --- Check 2: Animation row indices are contiguous ---
    if spec.animations:
        rows = sorted([a.row for a in spec.animations])
        expected_rows = list(range(len(rows)))
        if rows != expected_rows:
            # Find the gaps
            missing = set(expected_rows) - set(rows)
            if missing:
                warnings.append(
                    f"Animation row indices have gaps: missing rows {sorted(missing)}"
                )

    # --- Check 3: Frame counts validation ---
    # Zero frame count is already prevented by AnimationDef field validator (frames: int = Field(..., ge=1))
    # Max columns check is already done by SpritesheetSpec._validate_animations()
    # So these are already covered by load_config()

    # --- Check 4: Base image path exists ---
    if check_base_image and spec.base_image_path:
        resolved_path = Path(spec.base_image_path)
        # If path is relative, try resolving relative to cwd first, then config file location
        if not resolved_path.is_absolute():
            # Try relative to current working directory first
            cwd_path = Path.cwd() / spec.base_image_path
            if cwd_path.is_file():
                resolved_path = cwd_path.resolve()
            else:
                # Fall back to relative to config file location
                config_path = Path(path).resolve()
                resolved_path = (config_path.parent / spec.base_image_path).resolve()

        if not resolved_path.is_file():
            raise ValueError(
                f"Base image file does not exist: {spec.base_image_path} "
                f"(resolved to {resolved_path})"
            )

    # --- Check 5: Model deployment names are non-empty ---
    if spec.generation:
        models = [
            ("grid_model", spec.generation.grid_model),
            ("gate_model", spec.generation.gate_model),
            ("labeling_model", spec.generation.labeling_model),
            ("reference_model", spec.generation.reference_model),
        ]
        for model_name, model_value in models:
            if not model_value or not model_value.strip():
                warnings.append(
                    f"Model deployment name '{model_name}' is empty or whitespace-only"
                )

    # --- Check 6: Outline symbol is "O" (conventional) ---
    if spec.palette:
        if spec.palette.outline.symbol != "O":
            warnings.append(
                f"Palette uses non-standard outline symbol "
                f"'{spec.palette.outline.symbol}' (convention is 'O')"
            )

    # --- Check 7: Excessive palette size ---
    if spec.palette:
        color_count = len(spec.palette.colors)
        if color_count > 20:
            warnings.append(
                f"Palette has {color_count} colors (>20), "
                f"which may degrade grid generation quality"
            )

    return warnings
