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

            # ---- Preprocessing step (optional) ----
            quantized_reference: bytes | None = None
            if self.preprocessor is not None:
                if progress_callback:
                    progress_callback("preprocessing", 0, 1)
                preprocess_result: PreprocessResult = self.preprocessor(
                    image_path=base_reference_path,
                    frame_width=self.config.character.frame_width,
                    frame_height=self.config.character.frame_height,
                    max_colors=self.config.generation.max_palette_colors,
                    semantic_labels=self.config.generation.semantic_labels,
                )
                quantized_reference = preprocess_result.quantized_png_bytes

                # If auto_palette mode, use extracted palette locally
                if self.config.generation.auto_palette:
                    palette = preprocess_result.palette
                    palette_map = build_palette_map(preprocess_result.palette)
                if progress_callback:
                    progress_callback("preprocessing", 1, 1)

            # ---- Process Row 0 (anchor row) ----
            if progress_callback:
                progress_callback("row", 0, total_rows)

            anchor_animation = self.config.animations[0]
            row_images: dict[int, bytes] = {}

            # Check if row 0 is already checkpointed
            if (
                anchor_animation.row in completed_rows
                and self.checkpoint_manager is not None
            ):
                logger.info(
                    "Loading row 0/%d from checkpoint: %s (%d frames)",
                    total_rows,
                    anchor_animation.name,
                    anchor_animation.frames,
                )
                checkpoint_data = self.checkpoint_manager.load_row(anchor_animation.row)
                if checkpoint_data is None:
                    raise RuntimeError(
                        f"Checkpoint for row {anchor_animation.row} was reported as "
                        "completed but could not be loaded"
                    )
                row0_strip_bytes, row0_grids = checkpoint_data
                # Extract anchor grid (first grid)
                anchor_grid = row0_grids[0]
                # Render anchor frame for subsequent rows
                anchor_render_context = self.row_processor._build_frame_context(
                    palette=palette,
                    palette_map=palette_map,
                    animation=anchor_animation,
                    anchor_grid=anchor_grid,
                    anchor_rendered=None,
                    quantized_reference=None,
                )
                anchor_rendered = frame_to_png_bytes(
                    render_frame(anchor_grid, anchor_render_context)
                )
                row_images[anchor_animation.row] = row0_strip_bytes
                logger.info("Row 0 (%s) loaded from checkpoint", anchor_animation.name)
            else:
                logger.info(
                    "Processing row 0/%d: %s (%d frames)",
                    total_rows,
                    anchor_animation.name,
                    anchor_animation.frames,
                )
                anchor_grid, anchor_rendered, row0_grids = (
                    await self.row_processor.process_anchor_row(
                        base_reference,
                        anchor_animation,
                        palette,
                        palette_map,
                        quantized_reference=quantized_reference,
                    )
                )

                # Render and save anchor row strip
                # Create context for rendering with the anchor grid we just generated
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

                # Save checkpoint for row 0
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save_row(
                        row=anchor_animation.row,
                        animation_name=anchor_animation.name,
                        strip_bytes=row_images[anchor_animation.row],
                        grids=row0_grids,
                    )

                logger.info(
                    "Row 0 (%s) complete: %d/%d frames generated",
                    anchor_animation.name,
                    anchor_animation.frames,
                    anchor_animation.frames,
                )

            if progress_callback:
                progress_callback("row", 1, total_rows)

            # ---- Process remaining rows (in parallel) ----
            remaining_animations = list(enumerate(self.config.animations[1:], start=1))

            if not remaining_animations:
                return await self.assembler(
                    row_images=row_images,
                    config=self.config,
                    output_path=out,
                    checkpoint_manager=self.checkpoint_manager,
                    progress_callback=progress_callback,
                )

            anchor_recovery = AnchorRecoveryPolicy(
                max_anchor_regenerations=self.config.generation.max_anchor_regenerations,
                failure_ratio_threshold=self.config.generation.anchor_regen_failure_ratio,
            )
            anchor_recovery_state = AnchorRecoveryState()

            while True:
                semaphore: asyncio.Semaphore | None = None
                if self.max_concurrent_rows > 0:
                    semaphore = asyncio.Semaphore(self.max_concurrent_rows)

                completed_count = 0
                failed_rows: list[tuple[int, str, Exception]] = []

                async def _process_one(
                    row_idx_seq: int, animation: AnimationDef
                ) -> tuple[int, AnimationDef, list[list[str]] | Exception]:
                    """Process one row and return result or exception."""
                    if semaphore is not None:
                        await semaphore.acquire()
                    try:
                        # Check if this row is already checkpointed
                        if (
                            animation.row in completed_rows
                            and self.checkpoint_manager is not None
                        ):
                            logger.info(
                                "Loading row %d/%d from checkpoint: %s (%d frames)",
                                row_idx_seq,
                                total_rows,
                                animation.name,
                                animation.frames,
                            )
                            checkpoint_data = self.checkpoint_manager.load_row(
                                animation.row
                            )
                            if checkpoint_data is None:
                                raise RuntimeError(
                                    f"Checkpoint for row {animation.row} was reported as "
                                    "completed but could not be loaded"
                                )
                            row_strip_bytes, row_grids = checkpoint_data
                            row_images[animation.row] = row_strip_bytes
                            logger.info(
                                "Row %d (%s) loaded from checkpoint",
                                row_idx_seq,
                                animation.name,
                            )
                        else:
                            logger.info(
                                "Processing row %d/%d: %s (%d frames)",
                                row_idx_seq,
                                total_rows,
                                animation.name,
                                animation.frames,
                            )

                            row_grids = await self.row_processor.process_row(
                                base_reference,
                                animation,
                                palette,
                                palette_map,
                                anchor_grid=anchor_grid,
                                anchor_rendered=anchor_rendered,
                            )

                            logger.info(
                                "Row %d (%s) complete: %d/%d frames generated",
                                row_idx_seq,
                                animation.name,
                                animation.frames,
                                animation.frames,
                            )

                        return (row_idx_seq, animation, row_grids)
                    except Exception as e:
                        logger.error(
                            "Row %d (%s) failed: %s",
                            row_idx_seq,
                            animation.name,
                            str(e),
                        )
                        return (row_idx_seq, animation, e)
                    finally:
                        if semaphore is not None:
                            semaphore.release()

                # Gather all results with exception isolation
                results = await asyncio.gather(
                    *(_process_one(idx, anim) for idx, anim in remaining_animations),
                    return_exceptions=True,
                )

                # Process results: separate successes from failures
                for result in results:
                    # Handle gather-level exceptions (shouldn't happen with our wrapper)
                    if isinstance(result, Exception):
                        logger.error("Unexpected gather exception: %s", result)
                        failed_rows.append((-1, "unknown", result))
                        continue

                    # Type narrowing: result is now tuple[int, AnimationDef, list[list[str]] | Exception]
                    assert not isinstance(result, BaseException)
                    row_idx_seq, animation, outcome = result

                    if isinstance(outcome, Exception):
                        # Row processing failed
                        failed_rows.append((row_idx_seq, animation.name, outcome))
                    else:
                        # Row processing succeeded
                        row_grids = outcome

                        # Only render if not already loaded from checkpoint
                        if animation.row not in row_images:
                            # Create context for rendering this row
                            row_render_context = (
                                self.row_processor._build_frame_context(
                                    palette=palette,
                                    palette_map=palette_map,
                                    animation=animation,
                                    anchor_grid=anchor_grid,
                                    anchor_rendered=anchor_rendered,
                                    quantized_reference=None,
                                )
                            )
                            row_strip = render_row_strip(row_grids, row_render_context)
                            row_images[animation.row] = frame_to_png_bytes(row_strip)

                            # Save checkpoint for this row
                            if self.checkpoint_manager is not None:
                                self.checkpoint_manager.save_row(
                                    row=animation.row,
                                    animation_name=animation.name,
                                    strip_bytes=row_images[animation.row],
                                    grids=row_grids,
                                )

                        completed_count += 1
                        if progress_callback:
                            progress_callback("row", 1 + completed_count, total_rows)

                if not failed_rows:
                    logger.info(
                        "All %d non-anchor rows completed successfully",
                        len(remaining_animations),
                    )
                    break

                failed_summary = "\n".join(
                    f"  - Row {idx} ({name}): {exc}" for idx, name, exc in failed_rows
                )
                logger.error(
                    "Failed to generate %d/%d rows:\n%s",
                    len(failed_rows),
                    len(remaining_animations),
                    failed_summary,
                )

                recovery_decision = anchor_recovery.decide(
                    failed_rows=failed_rows,
                    total_non_anchor_rows=len(remaining_animations),
                    current_attempts=anchor_recovery_state.attempts,
                )

                if not recovery_decision.should_regenerate_anchor:
                    # Raise an error that includes information about partial success
                    raise RowGenerationError(
                        f"Failed to generate {len(failed_rows)} of "
                        f"{len(remaining_animations)} non-anchor rows. "
                        f"Successfully generated rows: {completed_count}. "
                        f"See logs for details."
                    )

                anchor_recovery_state.attempts += 1
                logger.warning(
                    "Regenerating anchor row due to cascade failures "
                    "(%d/%d failed, ratio=%.2f, attempt %d/%d).",
                    len(failed_rows),
                    len(remaining_animations),
                    recovery_decision.failure_ratio,
                    anchor_recovery_state.attempts,
                    anchor_recovery.max_anchor_regenerations,
                )

                # Regenerate only anchor row, then retry remaining rows.
                anchor_grid, anchor_rendered = await self._regenerate_anchor_row(
                    base_reference=base_reference,
                    anchor_animation=anchor_animation,
                    palette=palette,
                    palette_map=palette_map,
                    quantized_reference=quantized_reference,
                    row_images=row_images,
                )

                # Previously generated non-anchor outputs may encode identity drift.
                completed_rows = self._reset_non_anchor_progress(
                    animations=self.config.animations,
                    row_images=row_images,
                    anchor_row=anchor_animation.row,
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


async def create_workflow(
    config: SpritesheetSpec,
    project_endpoint: str | None = None,
    credential: object | None = None,
    preprocessor: Callable[..., PreprocessResult] | None = None,
    max_concurrent_rows: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> SpriteForgeWorkflow:
    """Create a fully wired SpriteForgeWorkflow with tiered model architecture.

    Creates separate AzureChatProvider instances for each pipeline stage
    based on the model deployment names in config.generation:
    - grid_model → GridGenerator
    - gate_model → LLMGateChecker
    - reference_model → GPTImageProvider

    Chat providers share the same Azure credential instance to avoid
    multiple token fetches. GPTImageProvider also uses Entra ID bearer
    token authentication.

    Args:
        config: Spritesheet specification with model deployment names.
        project_endpoint: Azure endpoint for chat model calls. Accepts either
            Azure AI Foundry project endpoint or Azure OpenAI endpoint.
            Falls back to environment variables when not provided.
        credential: Optional shared Azure credential. If not provided,
            a DefaultAzureCredential will be created and managed by
            the workflow (closed when workflow.close() is called).
        preprocessor: Optional preprocessing callable (e.g.
            ``preprocess_reference``). When provided, the base
            reference image is resized, quantized, and optionally
            auto-palette-extracted before generation begins.
        max_concurrent_rows: Maximum number of rows to process in
            parallel after the anchor row. ``0`` (default) means
            unlimited — all remaining rows run concurrently.
        checkpoint_dir: Optional directory for saving/loading checkpoints.
            If provided, enables checkpoint/resume support. After each
            row completes Gate 3A, its strip PNG and frame grids are
            saved. On resume, completed rows are skipped.

    Returns:
        A fully initialized SpriteForgeWorkflow ready to run.

    Raises:
        ProviderError: If no endpoint is available.

    Example::

        config = load_config("configs/theron.yaml")
        async with await create_workflow(config) as workflow:
            await workflow.run(
                base_reference_path="docs_assets/theron_base_reference.png",
                output_path="output/theron_spritesheet.png",
            )
    """
    import os

    from spriteforge.providers.azure_chat import AzureChatProvider
    from spriteforge.providers.gpt_image import GPTImageProvider

    # Resolve endpoint — prefer explicit arg, then Foundry env, then OpenAI env.
    # GPT image endpoint is retained as a compatibility fallback.
    endpoint = (
        project_endpoint
        or os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
        or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        or os.environ.get("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "")
    )
    if not endpoint:
        raise ProviderError(
            "No Azure OpenAI endpoint configured. "
            "Set AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT, "
            "or pass project_endpoint."
        )

    # Create or reuse credential.
    # Explicit ownership contract:
    # - caller-provided credential => caller closes it
    # - factory-created credential => workflow closes it
    shared_credential: object
    credential_handle: CredentialHandle
    if credential is None:
        from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-untyped,import-not-found]

        shared_credential = DefaultAzureCredential()
        credential_handle = CredentialHandle(
            credential=shared_credential,
            owned_by_workflow=True,
        )
    else:
        shared_credential = credential
        credential_handle = CredentialHandle(
            credential=shared_credential,
            owned_by_workflow=False,
        )

    # Create tiered chat providers
    default_deployments = {
        "grid_model": "gpt-5.2",
        "gate_model": "gpt-5-mini",
        "labeling_model": "gpt-5-nano",
        "reference_model": "gpt-image-1.5",
    }
    for key, default_name in default_deployments.items():
        configured_name = getattr(config.generation, key)
        if configured_name == default_name:
            logger.warning(
                "Using default deployment name for %s: %s. "
                "Set generation.%s in YAML if your Azure deployment differs.",
                key,
                configured_name,
                key,
            )

    grid_provider = AzureChatProvider(
        azure_endpoint=endpoint,
        model_deployment_name=config.generation.grid_model,
        credential=shared_credential,
    )
    gate_provider = AzureChatProvider(
        azure_endpoint=endpoint,
        model_deployment_name=config.generation.gate_model,
        credential=shared_credential,
    )

    # Create reference provider (uses Entra ID bearer token authentication)
    gpt_image_endpoint = os.environ.get("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "")
    reference_provider = GPTImageProvider(
        azure_endpoint=gpt_image_endpoint or None,
        credential=shared_credential,
        model_deployment=config.generation.reference_model,
    )

    # Create components
    grid_generator = GridGenerator(chat_provider=grid_provider)
    gate_checker = LLMGateChecker(
        chat_provider=gate_provider,
        max_image_bytes=config.generation.max_image_bytes,
        request_timeout_seconds=config.generation.request_timeout_seconds,
    )
    programmatic_checker = ProgrammaticChecker()
    metrics_collector = RunMetricsCollector()
    retry_manager = RetryManager(metrics_sink=metrics_collector)

    # Create call tracker if budget is configured
    call_tracker = None
    if config.generation.budget is not None:
        from spriteforge.budget import CallTracker

        call_tracker = CallTracker(config.generation.budget)

    frame_generator = FrameGenerator(
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=retry_manager,
        generation_config=config.generation,
        call_tracker=call_tracker,
        metrics_collector=metrics_collector,
    )
    row_processor = RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=gate_checker,
        reference_provider=reference_provider,
        call_tracker=call_tracker,
        metrics_collector=metrics_collector,
    )
    checkpoint_manager = (
        CheckpointManager(Path(checkpoint_dir)) if checkpoint_dir is not None else None
    )

    # Create workflow
    workflow = SpriteForgeWorkflow(
        config=config,
        row_processor=row_processor,
        preprocessor=preprocessor,
        checkpoint_manager=checkpoint_manager,
        max_concurrent_rows=max_concurrent_rows,
        credential_handle=credential_handle,
        call_tracker=call_tracker,
        metrics_collector=metrics_collector,
    )

    return workflow
