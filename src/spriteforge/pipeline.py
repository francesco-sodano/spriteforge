"""Workflow orchestrator — full pipeline coordination for spritesheet generation.

Ties all pipeline stages together using plain async Python (``asyncio``):
Stage 1 (reference generation via GPT-Image-1.5) → Gate -1 (reference QC) →
Stage 2 (per-frame grid generation via configured chat deployment, with verification
gates and retry loops) → Row assembly → Gate 3A (row coherence) →
Final spritesheet assembly.

Row 0 is always processed first to extract the anchor frame (Frame 0),
which is then used as a cross-row identity reference for all subsequent
frame generation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from spriteforge.assembler import assemble_spritesheet
from spriteforge.checkpoint import CheckpointManager
from spriteforge.errors import GateError, RowGenerationError
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GenerationError, GridGenerator
from spriteforge.logging import get_logger
from spriteforge.models import (
    AnimationDef,
    PaletteConfig,
    SpritesheetSpec,
)
from spriteforge.observability import RunMetricsCollector
from spriteforge.palette import build_palette_map
from spriteforge.preprocessor import PreprocessResult, preprocess_reference
from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.prompts.providers import build_reference_prompt
from spriteforge.renderer import (
    frame_to_png_bytes,
    render_frame,
    render_row_strip,
)
from spriteforge.retry import RetryManager
from spriteforge.row_processor import RowProcessor
from spriteforge.pipeline_stages import (
    AnchorRowStage,
    PreprocessingStage,
    RemainingRowsStage,
)

logger = get_logger("workflow")


@dataclass(frozen=True)
class CredentialHandle:
    """Credential plus explicit ownership metadata.

    When ``owned_by_workflow`` is True, the workflow is responsible for
    closing the credential during ``SpriteForgeWorkflow.close()``.
    """

    credential: object
    owned_by_workflow: bool


@dataclass(frozen=True)
class AnchorRecoveryDecision:
    """Decision result for whether anchor regeneration should run."""

    should_regenerate_anchor: bool
    failure_ratio: float


@dataclass
class AnchorRecoveryState:
    """Mutable state for anchor regeneration attempts."""

    attempts: int = 0


class AnchorRecoveryPolicy:
    """Encapsulates cascade-failure policy for anchor regeneration."""

    def __init__(
        self,
        max_anchor_regenerations: int,
        failure_ratio_threshold: float,
    ) -> None:
        self.max_anchor_regenerations = max_anchor_regenerations
        self.failure_ratio_threshold = failure_ratio_threshold

    def decide(
        self,
        failed_rows: list[tuple[int, str, Exception]],
        total_non_anchor_rows: int,
        current_attempts: int,
    ) -> AnchorRecoveryDecision:
        """Return whether to regenerate anchor based on current failures."""
        failure_ratio = len(failed_rows) / total_non_anchor_rows
        should_regenerate = (
            self.max_anchor_regenerations > current_attempts
            and failure_ratio >= self.failure_ratio_threshold
            and all(idx >= 1 for idx, _name, _exc in failed_rows)
        )
        return AnchorRecoveryDecision(
            should_regenerate_anchor=should_regenerate,
            failure_ratio=failure_ratio,
        )


def _resolve_output_path(output_path: str | Path, allow_absolute: bool) -> Path:
    """Resolve output path with optional absolute-path policy enforcement."""
    out = Path(output_path)
    if out.is_absolute() and not allow_absolute:
        raise ValueError(
            "Absolute output paths are disabled. "
            "Set generation.allow_absolute_output_path=true to allow them."
        )
    if out.is_absolute():
        return out.resolve()

    if ".." in out.parts and not allow_absolute:
        raise ValueError(
            "Relative output paths must not contain parent-directory traversal "
            "segments ('..') when generation.allow_absolute_output_path is false"
        )

    return (Path.cwd().resolve() / out).resolve()


async def assemble_final_spritesheet(
    row_images: dict[int, bytes],
    config: SpritesheetSpec,
    output_path: Path,
    checkpoint_manager: CheckpointManager | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Assemble row strips into final spritesheet and finalize run state."""
    if progress_callback:
        progress_callback("assembly", 0, 1)

    assemble_spritesheet(row_images, config, output_path=output_path)

    logger.info("Spritesheet saved: %s", output_path)

    if checkpoint_manager is not None:
        checkpoint_manager.cleanup()

    if progress_callback:
        progress_callback("assembly", 1, 1)

    return output_path


class SpriteForgeWorkflow:
    """Orchestrates the full spritesheet generation pipeline.

    Uses plain async Python (asyncio) for workflow coordination.
    Row 0 (anchor) is always processed first; subsequent rows are
    generated in parallel via ``asyncio.gather`` with an optional
    concurrency cap (``max_concurrent_rows``) to respect API rate
    limits and memory constraints.

    Uses GPT-Image-1.5 for reference generation and configurable chat
    deployments for grid generation and verification.
    """

    def __init__(
        self,
        config: SpritesheetSpec,
        row_processor: RowProcessor,
        assembler: Callable[..., Awaitable[Path]] = assemble_final_spritesheet,
        preprocessor: Callable[..., PreprocessResult] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        max_concurrent_rows: int = 0,
        credential_handle: CredentialHandle | None = None,
        call_tracker: Any | None = None,
        metrics_collector: RunMetricsCollector | None = None,
    ) -> None:
        """Initialize the workflow with all required components.

        Args:
            config: The spritesheet specification loaded from YAML.
            row_processor: Row-level generation coordinator.
            assembler: Final assembly stage callable.
            preprocessor: Optional preprocessing callable (e.g.
                ``preprocess_reference``).  When provided, the base
                reference image is resized, quantized, and optionally
                auto-palette-extracted before generation begins.
            checkpoint_manager: Optional checkpoint manager for resume support.
            max_concurrent_rows: Maximum number of rows to process in
                parallel after the anchor row.  ``0`` (default) means
                unlimited — all remaining rows run concurrently.
            credential_handle: Optional credential handle with explicit
                ownership metadata. When provided with
                ``owned_by_workflow=True``, the credential is closed
                during ``close()``.
        """
        self.config = config
        self.row_processor = row_processor
        self.assembler = assembler
        self.preprocessor = preprocessor
        self.checkpoint_manager = checkpoint_manager
        self.max_concurrent_rows = max_concurrent_rows
        self.palette_map = (
            build_palette_map(config.palette) if config.palette is not None else {}
        )
        self.frame_generator = row_processor.frame_generator
        self.gate_checker = row_processor.gate_checker
        self.reference_provider = row_processor.reference_provider
        self.grid_generator = row_processor.frame_generator.grid_generator
        self._credential_handle = credential_handle
        self.call_tracker = call_tracker
        self.metrics_collector = metrics_collector
        self._closed: bool = False

    def get_run_metrics_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of current run metrics."""
        if self.metrics_collector is None:
            return {}
        return self.metrics_collector.snapshot(call_tracker=self.call_tracker)

    async def __aenter__(self) -> "SpriteForgeWorkflow":
        """Enter the async context manager. Returns self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager, closing all resources."""
        await self.close()

    async def close(self) -> None:
        """Clean up all provider resources.

        Closes resources through the row/frame processor ownership chain.
        Also closes the factory-created credential when ownership is explicit
        via ``CredentialHandle(owned_by_workflow=True)``.
        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        await self.row_processor.close()

        # Close owned credential (only if created by factory)
        if (
            self._credential_handle is not None
            and self._credential_handle.owned_by_workflow
        ):
            try:
                owned_credential = cast(Any, self._credential_handle.credential)
                await owned_credential.close()
            except Exception as e:
                logger.warning("Failed to close credential: %s", e)
        self._credential_handle = None

    async def _regenerate_anchor_row(
        self,
        base_reference: bytes,
        anchor_animation: AnimationDef,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        quantized_reference: bytes | None,
        row_images: dict[int, bytes],
    ) -> tuple[list[str], bytes]:
        """Regenerate anchor row and persist row-0 artifacts/checkpoint."""
        anchor_grid, anchor_rendered, row0_grids = (
            await self.row_processor.process_anchor_row(
                base_reference,
                anchor_animation,
                palette,
                palette_map,
                quantized_reference=quantized_reference,
            )
        )
        anchor_render_context = self.row_processor._build_frame_context(
            palette=palette,
            palette_map=palette_map,
            animation=anchor_animation,
            anchor_grid=anchor_grid,
            anchor_rendered=None,
            quantized_reference=None,
        )
        row0_strip = render_row_strip(row0_grids, anchor_render_context)
        row_images[anchor_animation.row] = frame_to_png_bytes(row0_strip)
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_row(
                row=anchor_animation.row,
                animation_name=anchor_animation.name,
                strip_bytes=row_images[anchor_animation.row],
                grids=row0_grids,
            )
        return anchor_grid, anchor_rendered

    @staticmethod
    def _reset_non_anchor_progress(
        animations: list[AnimationDef],
        row_images: dict[int, bytes],
        anchor_row: int,
    ) -> set[int]:
        """Clear non-anchor outputs and return updated completed row set."""
        for anim in animations[1:]:
            row_images.pop(anim.row, None)
        return {anchor_row}

    def _build_anchor_recovery_policy(self) -> AnchorRecoveryPolicy:
        """Construct the anchor-recovery policy from runtime configuration."""
        return AnchorRecoveryPolicy(
            max_anchor_regenerations=self.config.generation.max_anchor_regenerations,
            failure_ratio_threshold=self.config.generation.anchor_regen_failure_ratio,
        )

    @staticmethod
    def _build_anchor_recovery_state() -> AnchorRecoveryState:
        """Construct mutable state for anchor-recovery attempts."""
        return AnchorRecoveryState()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        base_reference_path: str | Path,
        output_path: str | Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> Path:
        """Execute the full spritesheet generation pipeline.

        Args:
            base_reference_path: Path to the base character reference image.
            output_path: Path to save the final spritesheet PNG.
            progress_callback: Optional callback(stage_name, current, total)
                for progress reporting.

        Returns:
            Path to the saved spritesheet PNG.

        Raises:
            GenerationError: If a frame exhausts all retries.
            ProviderError: If reference generation fails.
        """
        try:
            base_reference = Path(base_reference_path).read_bytes()
            out = _resolve_output_path(
                output_path,
                allow_absolute=self.config.generation.allow_absolute_output_path,
            )
            out.parent.mkdir(parents=True, exist_ok=True)

            total_rows = len(self.config.animations)
            if self.config.palette is None:
                raise ValueError(
                    "No palette configured. Provide a palette in the YAML config "
                    "or enable auto_palette with a preprocessor."
                )
            palette = self.config.palette
            palette_map = dict(self.palette_map)

            logger.info(
                "Starting spritesheet generation for '%s' (%d rows)",
                self.config.character.name,
                total_rows,
            )

            # ---- Check for existing checkpoints (resume support) ----
            completed_rows: set[int] = set()
            if self.checkpoint_manager is not None:
                completed_rows = self.checkpoint_manager.completed_rows()
                if completed_rows:
                    logger.info(
                        "Found %d completed checkpoint(s): %s",
                        len(completed_rows),
                        sorted(completed_rows),
                    )

            row_images: dict[int, bytes] = {}
            preprocessing_stage = PreprocessingStage(
                workflow=self,
                base_reference_path=str(base_reference_path),
                progress_callback=progress_callback,
            )
            palette, palette_map, quantized_reference = preprocessing_stage.execute(
                palette,
                palette_map,
            )

            anchor_stage = AnchorRowStage(
                workflow=self,
                base_reference=base_reference,
                total_rows=total_rows,
                progress_callback=progress_callback,
            )
            anchor_animation, anchor_grid, anchor_rendered = await anchor_stage.execute(
                palette=palette,
                palette_map=palette_map,
                quantized_reference=quantized_reference,
                completed_rows=completed_rows,
                row_images=row_images,
            )

            remaining_rows_stage = RemainingRowsStage(
                workflow=self,
                base_reference=base_reference,
                total_rows=total_rows,
                progress_callback=progress_callback,
            )
            await remaining_rows_stage.execute(
                palette=palette,
                palette_map=palette_map,
                quantized_reference=quantized_reference,
                anchor_animation=anchor_animation,
                anchor_grid=anchor_grid,
                anchor_rendered=anchor_rendered,
                completed_rows=completed_rows,
                row_images=row_images,
            )

            return await self.assembler(
                row_images=row_images,
                config=self.config,
                output_path=out,
                checkpoint_manager=self.checkpoint_manager,
                progress_callback=progress_callback,
            )
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.finish()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


# --------------------------------------------------------------------------
# Factory function for tiered model architecture
# --------------------------------------------------------------------------
