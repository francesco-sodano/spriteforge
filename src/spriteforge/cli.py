"""Command-line interface for SpriteForge.

Provides commands for generating spritesheets, validating configs,
estimating costs, and initializing new character configs.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from spriteforge import (
    create_workflow,
    estimate_calls,
    load_config,
    setup_logging,
    validate_config,
)
from spriteforge.errors import SpriteForgeError

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbose flag."""
    if verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)


@click.group()
@click.version_option()
def main() -> None:
    """SpriteForge — AI-powered spritesheet generator for 2D pixel-art games."""
    pass


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for the spritesheet PNG (overrides config)",
)
@click.option(
    "--base-image",
    "-b",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Base character reference image (overrides config)",
)
@click.option(
    "--max-concurrent-rows",
    "-c",
    type=int,
    default=0,
    help="Maximum rows to process in parallel (0 = unlimited)",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    help="Resume from checkpoint if available",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    help="Directory for checkpoint files (default: .spriteforge/checkpoints/<character_name>)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable detailed logging",
)
def generate(
    config_path: Path,
    output: Path | None,
    base_image: Path | None,
    max_concurrent_rows: int,
    resume: bool,
    checkpoint_dir: Path | None,
    verbose: bool,
) -> None:
    """Generate a spritesheet from a character configuration.

    CONFIG_PATH: Path to the YAML configuration file (e.g., configs/theron.yaml)

    Example:

        \b
        # Basic generation
        spriteforge generate configs/theron.yaml

        \b
        # With custom output and options
        spriteforge generate configs/theron.yaml \\
            --output output/theron.png \\
            --base-image docs_assets/theron_base_reference.png \\
            --max-concurrent-rows 4 \\
            --resume \\
            --verbose
    """
    _setup_logging(verbose)

    try:
        # Load configuration
        with console.status(f"[bold blue]Loading configuration from {config_path}..."):
            config = load_config(config_path)

        console.print(
            f"[bold green]✓[/] Loaded configuration for [bold]{config.character.name}[/]"
        )
        console.print(
            f"  Animations: {len(config.animations)} rows, "
            f"{sum(a.frames for a in config.animations)} total frames"
        )

        # Resolve paths
        output_path_resolved: Path
        if output:
            output_path_resolved = output
        elif config.output_path:
            output_path_resolved = Path(config.output_path)
        else:
            output_path_resolved = Path(
                f"output/{config.character.name.lower()}_spritesheet.png"
            )

        base_image_path_resolved: Path
        if base_image:
            base_image_path_resolved = base_image
        elif config.base_image_path:
            base_image_path_resolved = Path(config.base_image_path)
        else:
            console.print(
                "[bold red]✗[/] No base image specified. "
                "Use --base-image or set base_image_path in config."
            )
            sys.exit(1)

        # Resolve checkpoint directory
        if resume or checkpoint_dir:
            if checkpoint_dir is None:
                checkpoint_dir = (
                    Path(".spriteforge") / "checkpoints" / config.character.name.lower()
                )
            console.print(f"  Checkpoint dir: {checkpoint_dir}")

        console.print(f"  Output: {output_path_resolved}")
        console.print(f"  Base image: {base_image_path_resolved}")
        console.print()

        # Run the generation pipeline
        asyncio.run(
            _run_generation(
                config=config,
                base_reference_path=base_image_path_resolved,
                output_path=output_path_resolved,
                max_concurrent_rows=max_concurrent_rows,
                checkpoint_dir=checkpoint_dir,
                verbose=verbose,
            )
        )

    except SpriteForgeError as e:
        console.print(f"[bold red]✗[/] Generation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠[/] Generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]✗[/] Unexpected error: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


async def _run_generation(
    config: Any,
    base_reference_path: Path,
    output_path: Path,
    max_concurrent_rows: int,
    checkpoint_dir: Path | None,
    verbose: bool,
) -> None:
    """Run the generation workflow with progress display."""
    total_rows = len(config.animations)

    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Create tasks
        preprocess_task = progress.add_task(
            "[cyan]Preprocessing...", total=1, start=False
        )
        row_task = progress.add_task(
            "[cyan]Generating rows...", total=total_rows, start=False
        )

        # Track current stage
        current_stage: dict[str, str | TaskID | None] = {"name": None, "task": None}

        def progress_callback(stage_name: str, current: int, total: int) -> None:
            """Update progress based on stage."""
            if stage_name == "preprocessing":
                if current_stage["name"] != "preprocessing":
                    progress.start_task(preprocess_task)
                    current_stage["name"] = "preprocessing"
                    current_stage["task"] = preprocess_task
                progress.update(preprocess_task, completed=current)

            elif stage_name == "row":
                if current_stage["name"] != "row":
                    progress.start_task(row_task)
                    current_stage["name"] = "row"
                    current_stage["task"] = row_task
                progress.update(row_task, completed=current)

        # Create workflow
        with console.status("[bold blue]Creating workflow..."):
            workflow = await create_workflow(
                config=config,
                max_concurrent_rows=max_concurrent_rows,
                checkpoint_dir=checkpoint_dir,
            )

        try:
            # Run generation
            result = await workflow.run(
                base_reference_path=base_reference_path,
                output_path=output_path,
                progress_callback=progress_callback,
            )

            console.print()
            console.print(f"[bold green]✓[/] Spritesheet generated: [bold]{result}[/]")

        finally:
            await workflow.close()


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--no-check-base-image",
    is_flag=True,
    help="Skip base image existence check",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable detailed logging",
)
def validate(
    config_path: Path,
    no_check_base_image: bool,
    verbose: bool,
) -> None:
    """Validate a character configuration without generating.

    CONFIG_PATH: Path to the YAML configuration file

    Performs comprehensive validation:
    - YAML syntax and structure
    - Pydantic schema validation
    - Palette symbol uniqueness
    - Animation row index continuity
    - Frame count constraints
    - Base image existence (optional)
    - Model deployment name validation

    Example:

        \b
        spriteforge validate configs/theron.yaml
        spriteforge validate configs/theron.yaml --no-check-base-image
    """
    _setup_logging(verbose)

    try:
        with console.status(f"[bold blue]Validating {config_path}..."):
            warnings = validate_config(
                config_path, check_base_image=not no_check_base_image
            )
            config = load_config(config_path)

        console.print(f"[bold green]✓[/] Configuration is valid")
        console.print(f"  Character: {config.character.name}")
        console.print(f"  Animations: {len(config.animations)} rows")
        console.print(f"  Total frames: {sum(a.frames for a in config.animations)}")

        if warnings:
            console.print()
            console.print(f"[bold yellow]⚠[/] {len(warnings)} warning(s):")
            for warning in warnings:
                console.print(f"  • {warning}")

    except Exception as e:
        console.print(f"[bold red]✗[/] Validation failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable detailed logging",
)
def estimate(
    config_path: Path,
    verbose: bool,
) -> None:
    """Estimate LLM call costs before generating.

    CONFIG_PATH: Path to the YAML configuration file

    Provides min/expected/max call count estimates based on:
    - Number of animations and frames
    - Configured retry limits
    - Expected retry rates (30% reference, 20% frame, 5% row)

    Example:

        \b
        spriteforge estimate configs/theron.yaml
    """
    _setup_logging(verbose)

    try:
        # Load configuration
        with console.status(f"[bold blue]Loading configuration from {config_path}..."):
            config = load_config(config_path)

        console.print(
            f"[bold green]✓[/] Loaded configuration for [bold]{config.character.name}[/]"
        )
        console.print(f"  Animations: {len(config.animations)} rows")
        console.print(f"  Total frames: {sum(a.frames for a in config.animations)}")

        # Check budget configuration
        if config.generation.budget:
            budget = config.generation.budget
            console.print()
            console.print("[bold]Budget Configuration:[/]")
            console.print(f"  max_llm_calls: {budget.max_llm_calls}")
            console.print(f"  max_retries_per_row: {budget.max_retries_per_row}")
            console.print(f"  warn_at_percentage: {budget.warn_at_percentage:.0%}")
        else:
            console.print()
            console.print(
                "[bold yellow]⚠[/] No budget configured (unlimited LLM calls)"
            )

        # Estimate calls
        with console.status("[bold blue]Estimating LLM call counts..."):
            estimate_result = estimate_calls(config)

        console.print()
        console.print("[bold cyan]═══════════════════════════════════════[/]")
        console.print("[bold]CALL ESTIMATE[/]")
        console.print("[bold cyan]═══════════════════════════════════════[/]")
        console.print()
        console.print(f"Minimum calls:  [bold]{estimate_result.min_calls:>5}[/]")
        console.print(f"Expected calls: [bold]{estimate_result.expected_calls:>5}[/]")
        console.print(f"Maximum calls:  [bold]{estimate_result.max_calls:>5}[/]")

        # Show breakdowns
        console.print()
        console.print("[bold]MINIMUM[/] (all frames pass first attempt):")
        for key, value in estimate_result.breakdown["min"].items():
            console.print(f"  {key}: {value}")

        console.print()
        console.print("[bold]EXPECTED[/] (typical scenario with some retries):")
        for key, value in estimate_result.breakdown["expected"].items():
            console.print(f"  {key}: {value}")

        console.print()
        console.print("[bold]MAXIMUM[/] (worst case with all retries):")
        for key, value in estimate_result.breakdown["max"].items():
            console.print(f"  {key}: {value}")

        # Budget check
        if config.generation.budget and config.generation.budget.max_llm_calls > 0:
            budget_limit = config.generation.budget.max_llm_calls
            console.print()
            console.print("[bold cyan]═══════════════════════════════════════[/]")
            console.print("[bold]BUDGET CHECK[/]")
            console.print("[bold cyan]═══════════════════════════════════════[/]")
            console.print(f"Budget limit: {budget_limit}")
            console.print(f"Expected calls: {estimate_result.expected_calls}")

            if estimate_result.expected_calls <= budget_limit:
                headroom = budget_limit - estimate_result.expected_calls
                console.print(
                    f"[bold green]✓[/] Expected calls fit within budget "
                    f"({headroom} calls headroom)"
                )
            else:
                overage = estimate_result.expected_calls - budget_limit
                console.print(
                    f"[bold yellow]⚠[/] Expected calls exceed budget by {overage} calls"
                )
                console.print(
                    "  Consider reducing spritesheet complexity or increasing budget"
                )

            if estimate_result.max_calls > budget_limit:
                console.print(
                    f"[bold yellow]⚠[/] Worst-case ({estimate_result.max_calls} calls) "
                    f"exceeds budget — generation may fail if many retries occur"
                )

    except Exception as e:
        console.print(f"[bold red]✗[/] Estimation failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
