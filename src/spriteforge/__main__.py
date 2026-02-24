"""SpriteForge command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from spriteforge.config import load_config
from spriteforge.errors import GenerationError, ProviderError
from spriteforge.logging import setup_logging
from spriteforge.preprocessor import preprocess_reference
from spriteforge.workflow import create_workflow

logger = logging.getLogger("spriteforge")


def _max_colors(value: str) -> int:
    parsed = int(value)
    if parsed < 2 or parsed > 64:
        raise argparse.ArgumentTypeError("--max-colors must be between 2 and 64")
    return parsed


def _default_output_path(character_name: str) -> Path:
    slug = character_name.strip().lower().replace(" ", "_")
    return Path(f"output/{slug}_spritesheet.png")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="spriteforge",
        description=(
            "Generate spritesheets from any self-contained SpriteForge YAML config. "
            "See configs/template.yaml to create a new character config."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to character YAML config",
    )
    parser.add_argument(
        "--base-image",
        "-b",
        type=Path,
        help="Path to base character reference PNG (overrides base_image_path in config)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output spritesheet path",
    )
    parser.add_argument(
        "--auto-palette",
        action="store_true",
        help="Auto-extract palette from base image",
    )
    parser.add_argument(
        "--max-colors",
        type=_max_colors,
        default=16,
        help="Maximum palette colors for auto-extraction (2-64)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate artifacts",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without API calls",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO, verbose=verbose)


def _progress_callback(stage_name: str, current: int, total: int) -> None:
    logger.debug("Progress %s: %d/%d", stage_name, current, total)


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    try:
        spec = load_config(args.config)

        if args.auto_palette:
            spec.generation.auto_palette = True
        if args.max_colors is not None:
            spec.generation.max_palette_colors = args.max_colors

        if args.dry_run:
            palette_source = (
                "auto (from base image)"
                if spec.generation.auto_palette
                else "YAML config"
            )
            print(
                f"Config valid: {spec.character.name}, "
                f"{len(spec.animations)} rows, "
                f"palette: {palette_source}"
            )
            return 0

        base_image_path = args.base_image or (
            Path(spec.base_image_path) if spec.base_image_path else None
        )

        if base_image_path is None or not base_image_path.is_file():
            print(f"Error: Base image not found: {base_image_path}", file=sys.stderr)
            return 1

        output_path = args.output
        if output_path is None:
            if spec.output_path:
                output_path = Path(spec.output_path)
            else:
                output_path = _default_output_path(spec.character.name)

        preprocessor = preprocess_reference if spec.generation.auto_palette else None

        async with await create_workflow(
            config=spec,
            preprocessor=preprocessor,
        ) as workflow:
            result = await workflow.run(
                base_reference_path=base_image_path,
                output_path=output_path,
                progress_callback=_progress_callback if args.verbose else None,
            )

        print(f"Spritesheet saved: {result}")
        return 0

    except (
        FileNotFoundError,
        ValidationError,
        GenerationError,
        ProviderError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """CLI entry point. Parses args and runs the async pipeline."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
