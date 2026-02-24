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
from pathlib import Path
from typing import Any

from spriteforge.assembler import assemble_spritesheet
from spriteforge.checkpoint import CheckpointManager
from spriteforge.errors import GateError
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GenerationError, GridGenerator
from spriteforge.logging import get_logger
from spriteforge.models import (
    AnimationDef,
    PaletteConfig,
    SpritesheetSpec,
)
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
        # Owned credential created by factory; None if caller passed their own.
        # Declared here so the field is always present (no dynamic attr assignment).
        self._owned_credential: object | None = None
        self._closed: bool = False

    async def __aenter__(self) -> "SpriteForgeWorkflow":
        """Enter the async context manager. Returns self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager, closing all resources."""
        await self.close()

    async def close(self) -> None:
        """Clean up all provider resources.

        Closes resources through the row/frame processor ownership chain.
        If a credential was created by the factory, it will also be closed.
        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        await self.row_processor.close()

        # Close owned credential (only if created by factory)
        if self._owned_credential is not None:
            try:
                await self._owned_credential.close()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("Failed to close credential: %s", e)
            self._owned_credential = None

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
        base_reference = Path(base_reference_path).read_bytes()
        out = Path(output_path)
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

        if remaining_animations:
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
                        row_render_context = self.row_processor._build_frame_context(
                            palette=palette,
                            palette_map=palette_map,
                            animation=animation,
                            anchor_grid=anchor_grid,
                            anchor_rendered=anchor_rendered,
                            quantized_reference=None,
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

            # Report failures if any
            if failed_rows:
                failed_summary = "\n".join(
                    f"  - Row {idx} ({name}): {exc}" for idx, name, exc in failed_rows
                )
                logger.error(
                    "Failed to generate %d/%d rows:\n%s",
                    len(failed_rows),
                    len(remaining_animations),
                    failed_summary,
                )

                # Raise an error that includes information about partial success
                raise GateError(
                    f"Failed to generate {len(failed_rows)} of "
                    f"{len(remaining_animations)} non-anchor rows. "
                    f"Successfully generated rows: {completed_count}. "
                    f"See logs for details."
                )

            logger.info(
                "All %d non-anchor rows completed successfully",
                len(remaining_animations),
            )

        return await self.assembler(
            row_images=row_images,
            config=self.config,
            output_path=out,
            checkpoint_manager=self.checkpoint_manager,
            progress_callback=progress_callback,
        )

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

    # Create or reuse credential
    # If user provided a credential, we don't own it and won't close it
    # If we create one, we'll store it and close it in workflow.close()
    shared_credential: object
    if credential is None:
        from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-untyped,import-not-found]

        shared_credential = DefaultAzureCredential()
        owns_credential = True
    else:
        shared_credential = credential
        owns_credential = False

    # Create tiered chat providers
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
    gate_checker = LLMGateChecker(chat_provider=gate_provider)
    programmatic_checker = ProgrammaticChecker()
    retry_manager = RetryManager()

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
    )
    row_processor = RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=gate_checker,
        reference_provider=reference_provider,
        call_tracker=call_tracker,
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
    )

    # Store credential ownership info for cleanup
    workflow._owned_credential = shared_credential if owns_credential else None

    return workflow
