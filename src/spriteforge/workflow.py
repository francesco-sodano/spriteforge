"""Workflow orchestrator — full pipeline coordination for spritesheet generation.

Ties all pipeline stages together using plain async Python (``asyncio``):
Stage 1 (reference generation via GPT-Image-1.5) → Gate -1 (reference QC) →
Stage 2 (per-frame grid generation via Claude Opus 4.6, with verification
gates and retry loops) → Row assembly → Gate 3A (row coherence) →
Final spritesheet assembly.

Row 0 is always processed first to extract the anchor frame (Frame 0),
which is then used as a cross-row identity reference for all subsequent
frame generation.
"""

from __future__ import annotations

import asyncio
import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload

from PIL import Image

from spriteforge.assembler import assemble_spritesheet
from spriteforge.checkpoint import CheckpointManager
from spriteforge.errors import GateError, RetryExhaustedError
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GenerationError, GridGenerator
from spriteforge.logging import get_logger
from spriteforge.models import (
    AnimationDef,
    FrameContext,
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

logger = get_logger("workflow")


class SpriteForgeWorkflow:
    """Orchestrates the full spritesheet generation pipeline.

    Uses plain async Python (asyncio) for workflow coordination.
    Row 0 (anchor) is always processed first; subsequent rows are
    generated in parallel via ``asyncio.gather`` with an optional
    concurrency cap (``max_concurrent_rows``) to respect API rate
    limits and memory constraints.

    Both AI models (GPT-Image-1.5 for reference generation, Claude Opus 4.6
    for grid generation and verification) are accessed through the same
    Azure AI Foundry project.
    """

    def __init__(
        self,
        config: SpritesheetSpec,
        reference_provider: ReferenceProvider,
        grid_generator: GridGenerator,
        gate_checker: LLMGateChecker,
        programmatic_checker: ProgrammaticChecker,
        retry_manager: RetryManager,
        palette_map: dict[str, tuple[int, int, int, int]],
        preprocessor: Callable[..., PreprocessResult] | None = None,
        max_concurrent_rows: int = 0,
        checkpoint_dir: str | Path | None = None,
        call_tracker: Any | None = None,
        frame_generator: FrameGenerator | None = None,
    ) -> None:
        """Initialize the workflow with all required components.

        Args:
            config: The spritesheet specification loaded from YAML.
            reference_provider: Stage 1 reference image provider.
            grid_generator: Stage 2 grid generator (Claude Opus 4.6).
            gate_checker: LLM-based verification gate checker.
            programmatic_checker: Fast deterministic grid checks.
            retry_manager: Retry and escalation engine.
            palette_map: Symbol → RGBA mapping for rendering.
            preprocessor: Optional preprocessing callable (e.g.
                ``preprocess_reference``).  When provided, the base
                reference image is resized, quantized, and optionally
                auto-palette-extracted before generation begins.
            max_concurrent_rows: Maximum number of rows to process in
                parallel after the anchor row.  ``0`` (default) means
                unlimited — all remaining rows run concurrently.
            checkpoint_dir: Optional directory for saving/loading checkpoints.
                If provided, enables checkpoint/resume support. After each
                row completes Gate 3A, its strip PNG and frame grids are
                saved. On resume, completed rows are skipped.
            call_tracker: Optional CallTracker for budget enforcement.
            frame_generator: Optional FrameGenerator instance. If not provided,
                one will be created from the other components.
        """
        self.config = config
        self.reference_provider = reference_provider
        self.grid_generator = grid_generator
        self.gate_checker = gate_checker
        self.programmatic_checker = programmatic_checker
        self.retry_manager = retry_manager
        self.palette_map = palette_map
        self.preprocessor = preprocessor
        self.max_concurrent_rows = max_concurrent_rows
        self.checkpoint_manager: CheckpointManager | None = None
        if checkpoint_dir is not None:
            self.checkpoint_manager = CheckpointManager(Path(checkpoint_dir))
        self.call_tracker = call_tracker

        # Create or use provided FrameGenerator
        if frame_generator is None:
            self.frame_generator = FrameGenerator(
                grid_generator=grid_generator,
                gate_checker=gate_checker,
                programmatic_checker=programmatic_checker,
                retry_manager=retry_manager,
                generation_config=config.generation,
                call_tracker=call_tracker,
            )
        else:
            self.frame_generator = frame_generator

    async def __aenter__(self) -> "SpriteForgeWorkflow":
        """Enter the async context manager. Returns self."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager, closing all resources."""
        await self.close()

    async def close(self) -> None:
        """Clean up all provider resources.

        Closes the grid generator's chat provider, gate checker's chat provider,
        and reference provider. If a credential was created by the factory,
        it will also be closed. Safe to call multiple times.
        """
        # Close grid generator's chat provider
        if hasattr(self.grid_generator, "_chat") and hasattr(
            self.grid_generator._chat, "close"
        ):
            await self.grid_generator._chat.close()

        # Close gate checker's chat provider
        if hasattr(self.gate_checker, "_chat") and hasattr(
            self.gate_checker._chat, "close"
        ):
            await self.gate_checker._chat.close()

        # Close reference provider
        if hasattr(self.reference_provider, "close"):
            await self.reference_provider.close()

        # Close owned credential (only if created by factory)
        if hasattr(self, "_owned_credential") and self._owned_credential is not None:  # type: ignore[has-type]
            try:
                await self._owned_credential.close()  # type: ignore[has-type]
            except Exception:
                pass
            self._owned_credential = None  # type: ignore[has-type]

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
            anchor_render_context = self._build_frame_context(
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
            anchor_grid, row0_grids = await self._process_row(
                base_reference,
                anchor_animation,
                palette,
                palette_map,
                is_anchor=True,
                quantized_reference=quantized_reference,
            )

            # Render and save anchor row strip
            # Create context for rendering with the anchor grid we just generated
            anchor_render_context = self._build_frame_context(
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

                        row_grids = await self._process_row(
                            base_reference,
                            animation,
                            palette,
                            palette_map,
                            is_anchor=False,
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
                        row_render_context = self._build_frame_context(
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

        # ---- Assemble final spritesheet ----
        if progress_callback:
            progress_callback("assembly", 0, 1)

        assemble_spritesheet(row_images, self.config, output_path=out)

        logger.info("Spritesheet saved: %s", out)

        # Clean up checkpoints after successful assembly
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.cleanup()

        if progress_callback:
            progress_callback("assembly", 1, 1)

        return out

    # ------------------------------------------------------------------
    # Row processing
    # ------------------------------------------------------------------

    @overload
    async def _process_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        is_anchor: Literal[True],
        anchor_grid: list[str] | None = None,
        anchor_rendered: bytes | None = None,
        quantized_reference: bytes | None = None,
    ) -> tuple[list[str], list[list[str]]]: ...

    @overload
    async def _process_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        is_anchor: Literal[False] = False,
        anchor_grid: list[str] | None = None,
        anchor_rendered: bytes | None = None,
        quantized_reference: bytes | None = None,
    ) -> list[list[str]]: ...

    async def _process_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        is_anchor: bool = False,
        anchor_grid: list[str] | None = None,
        anchor_rendered: bytes | None = None,
        quantized_reference: bytes | None = None,
    ) -> tuple[list[str], list[list[str]]] | list[list[str]]:
        """Process a single animation row.

        Args:
            base_reference: Raw bytes of the base character reference image.
            animation: Animation definition for this row.
            palette: The palette to use for this run.
            palette_map: Symbol → RGBA mapping for rendering.
            is_anchor: If True, processes Row 0 and generates the anchor frame.
            anchor_grid: The anchor frame grid (Row 0, Frame 0). Required if is_anchor=False.
            anchor_rendered: Rendered PNG bytes of the anchor frame. Required if is_anchor=False.
            quantized_reference: Optional quantized reference image bytes. Only used if is_anchor=True.

        Returns:
            If is_anchor=True: Tuple of (anchor_grid, list_of_all_frame_grids_for_this_row).
            If is_anchor=False: List of frame grids for this row.
        """
        # Stage 1: Generate reference strip
        reference_strip = await self._generate_reference_strip(
            base_reference, animation
        )

        # Handle anchor row: generate Frame 0 with special context
        if is_anchor:
            # Generate anchor frame (Frame 0)
            ref_frame = self._extract_reference_frame(
                reference_strip,
                0,
                self.config.character.frame_width,
                self.config.character.frame_height,
            )

            # Build context for anchor frame
            anchor_context = self._build_frame_context(
                palette=palette,
                palette_map=palette_map,
                animation=animation,
                anchor_grid=None,
                anchor_rendered=None,
                quantized_reference=quantized_reference,
            )

            generated_anchor_grid = await self._generate_and_verify_frame(
                reference_frame=ref_frame,
                context=anchor_context,
                frame_index=0,
                is_anchor=True,
                base_reference=base_reference,
            )

            generated_anchor_rendered = frame_to_png_bytes(
                render_frame(generated_anchor_grid, anchor_context)
            )

            frame_grids: list[list[str]] = [generated_anchor_grid]

            # Build context for remaining frames (with anchor references)
            frame_context = self._build_frame_context(
                palette=palette,
                palette_map=palette_map,
                animation=animation,
                anchor_grid=generated_anchor_grid,
                anchor_rendered=generated_anchor_rendered,
                quantized_reference=None,
            )

            # Generate remaining frames with anchor + prev frame context
            prev_grid = generated_anchor_grid
            prev_rendered = generated_anchor_rendered
            start_frame = 1
        else:
            # Build context for this row
            frame_context = self._build_frame_context(
                palette=palette,
                palette_map=palette_map,
                animation=animation,
                anchor_grid=anchor_grid,
                anchor_rendered=anchor_rendered,
                quantized_reference=None,
            )

            frame_grids = []
            prev_grid = None
            prev_rendered = None
            start_frame = 0

        # Generate remaining frames
        for fi in range(start_frame, animation.frames):
            ref_frame = self._extract_reference_frame(
                reference_strip,
                fi,
                self.config.character.frame_width,
                self.config.character.frame_height,
            )
            grid = await self._generate_and_verify_frame(
                reference_frame=ref_frame,
                context=frame_context,
                frame_index=fi,
                prev_frame_grid=prev_grid,
                prev_frame_rendered=prev_rendered,
                base_reference=base_reference,
            )
            frame_grids.append(grid)
            prev_grid = grid
            prev_rendered = frame_to_png_bytes(render_frame(grid, frame_context))

        # Gate 3A: Validate assembled row (with retry logic)
        ref_strip_bytes = frame_to_png_bytes(reference_strip.convert("RGBA"))
        max_retries = self.config.generation.gate_3a_max_retries

        for retry_attempt in range(max_retries + 1):
            row_strip = render_row_strip(frame_grids, frame_context)
            row_strip_bytes = frame_to_png_bytes(row_strip)

            if self.call_tracker:
                self.call_tracker.increment("gate_3a")
            verdict = await self.gate_checker.gate_3a(
                row_strip_bytes, ref_strip_bytes, animation
            )

            if verdict.passed:
                break

            # Gate 3A failed
            logger.warning(
                "Gate 3A failed for %s (attempt %d/%d): %s",
                animation.name,
                retry_attempt + 1,
                max_retries + 1,
                verdict.feedback,
            )

            # If this was the last attempt, raise error
            if retry_attempt >= max_retries:
                raise GateError(
                    f"Gate 3A (row coherence) failed for '{animation.name}' "
                    f"after {max_retries + 1} attempts: {verdict.feedback}"
                )

            # Otherwise, regenerate problematic frames
            logger.info(
                "Retrying Gate 3A for %s: regenerating problematic frames",
                animation.name,
            )

            # Identify frames to regenerate based on feedback
            frames_to_regenerate = self._identify_problematic_frames(
                verdict.feedback, animation.frames
            )

            # Regenerate identified frames
            for fi in frames_to_regenerate:
                ref_frame = self._extract_reference_frame(
                    reference_strip,
                    fi,
                    self.config.character.frame_width,
                    self.config.character.frame_height,
                )

                # Determine prev frame context
                if fi == 0:
                    prev_frame_grid = None
                    prev_frame_rendered = None
                else:
                    prev_frame_grid = frame_grids[fi - 1]
                    prev_frame_rendered = frame_to_png_bytes(
                        render_frame(prev_frame_grid, frame_context)
                    )

                logger.info(
                    "Regenerating frame %d/%d for %s",
                    fi,
                    animation.frames - 1,
                    animation.name,
                )

                grid = await self._generate_and_verify_frame(
                    reference_frame=ref_frame,
                    context=frame_context,
                    frame_index=fi,
                    is_anchor=(is_anchor and fi == 0),
                    prev_frame_grid=prev_frame_grid,
                    prev_frame_rendered=prev_frame_rendered,
                    base_reference=base_reference,
                )
                frame_grids[fi] = grid

        if is_anchor:
            # Use frame_grids[0] in case Gate 3A regenerated the anchor frame
            return frame_grids[0], frame_grids
        else:
            return frame_grids

    # ------------------------------------------------------------------
    # Frame generation with retry loop
    # ------------------------------------------------------------------

    def _identify_problematic_frames(self, feedback: str, num_frames: int) -> list[int]:
        """Identify frame indices that may be problematic based on Gate 3A feedback.

        Parses the feedback string to extract frame numbers. If no specific
        frames are mentioned, returns the last 2 frames as a conservative
        fallback (since animation issues often appear in transitions).

        Args:
            feedback: The feedback string from Gate 3A failure.
            num_frames: Total number of frames in the row.

        Returns:
            List of frame indices (0-based) to regenerate.
        """
        # Try to extract frame numbers from feedback
        # Look for patterns like "frame 3", "Frame 5", "frames 2-4", etc.
        frame_pattern = r"[Ff]rame[s]?\s+(\d+)"
        matches = re.findall(frame_pattern, feedback)

        if matches:
            # Convert to 0-based indices and ensure they're valid
            indices = []
            for match in matches:
                idx = int(match)
                # Handle both 0-based and 1-based frame numbering in feedback
                if idx >= num_frames:
                    idx = idx - 1  # Assume 1-based
                if 0 <= idx < num_frames:
                    indices.append(idx)

            if indices:
                logger.info(
                    "Identified frames to regenerate from feedback: %s",
                    sorted(set(indices)),
                )
                return sorted(set(indices))

        # Fallback: regenerate trailing frames (common source of animation issues)
        n = self.config.generation.fallback_regen_frames
        fallback = list(range(max(0, num_frames - n), num_frames))
        logger.info(
            "No specific frames identified in feedback; "
            "regenerating last %d frames as fallback: %s",
            len(fallback),
            fallback,
        )
        return fallback

    def _build_frame_context(
        self,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        animation: AnimationDef,
        anchor_grid: list[str] | None = None,
        anchor_rendered: bytes | None = None,
        quantized_reference: bytes | None = None,
    ) -> FrameContext:
        """Build a FrameContext from the current workflow state.

        Args:
            palette: Palette configuration.
            palette_map: Symbol → RGBA mapping for rendering.
            animation: Animation definition for this frame's row.
            anchor_grid: Optional anchor frame grid.
            anchor_rendered: Optional anchor frame PNG bytes.
            quantized_reference: Optional quantized reference PNG bytes.

        Returns:
            A frozen FrameContext with all frame generation parameters.
        """
        return FrameContext(
            palette=palette,
            palette_map=palette_map,
            generation=self.config.generation,
            frame_width=self.config.character.frame_width,
            frame_height=self.config.character.frame_height,
            animation=animation,
            spritesheet_columns=self.config.character.spritesheet_columns,
            anchor_grid=anchor_grid,
            anchor_rendered=anchor_rendered,
            quantized_reference=quantized_reference,
        )

    async def _generate_and_verify_frame(
        self,
        reference_frame: bytes,
        context: FrameContext,
        frame_index: int,
        prev_frame_grid: list[str] | None = None,
        prev_frame_rendered: bytes | None = None,
        is_anchor: bool = False,
        base_reference: bytes | None = None,
    ) -> list[str]:
        """Generate a single frame with full verification and retry loop.

        Delegates to FrameGenerator.generate_verified_frame().

        Args:
            reference_frame: PNG bytes of the rough reference for this frame.
            context: Frame context containing palette, animation, generation config,
                frame dimensions, palette_map, and optional anchor/quantized references.
            frame_index: Index of this frame within the row.
            prev_frame_grid: Grid of the previous frame (for continuity).
            prev_frame_rendered: PNG bytes of the rendered previous frame.
            is_anchor: Whether this is the anchor frame (Row 0, Frame 0).
            base_reference: Optional base character reference PNG bytes (for anchor only).

        Returns:
            Verified frame grid (list of frame_height strings).
        """
        return await self.frame_generator.generate_verified_frame(
            reference_frame=reference_frame,
            context=context,
            frame_index=frame_index,
            prev_frame_grid=prev_frame_grid,
            prev_frame_rendered=prev_frame_rendered,
            is_anchor=is_anchor,
            base_reference=base_reference,
        )

    # ------------------------------------------------------------------
    # Reference generation with retry
    # ------------------------------------------------------------------

    async def _generate_reference_strip(
        self,
        base_reference: bytes,
        animation: AnimationDef,
    ) -> Image.Image:
        """Generate and verify a rough reference strip for an animation row.

        1. Call ReferenceProvider.generate_row_strip() (GPT-Image-1.5)
        2. Run Gate -1 (reference quality check, Claude Opus 4.6)
        3. If Gate -1 fails: retry up to 3 times
        4. If all retries fail: raise ProviderError

        Returns:
            Reference strip as PIL Image.
        """
        max_ref_retries = 3
        prompt = build_reference_prompt(
            animation,
            self.config.character,
            self.config.character.description,
        )

        for attempt in range(max_ref_retries):
            if self.call_tracker:
                self.call_tracker.increment("reference_generation")
            strip = await self.reference_provider.generate_row_strip(
                base_reference=base_reference,
                prompt=prompt,
                num_frames=animation.frames,
                frame_size=self.config.character.frame_size,
            )

            # Gate -1: Validate reference quality
            strip_bytes = frame_to_png_bytes(strip.convert("RGBA"))
            if self.call_tracker:
                self.call_tracker.increment("gate_minus_1")
            verdict = await self.gate_checker.gate_minus_1(
                strip_bytes, base_reference, animation
            )

            if verdict.passed:
                return strip

            logger.warning(
                "Gate -1 failed for %s (attempt %d/%d): %s",
                animation.name,
                attempt + 1,
                max_ref_retries,
                verdict.feedback,
            )

        raise ProviderError(
            f"Reference strip generation for '{animation.name}' "
            f"failed after {max_ref_retries} attempts."
        )

    # ------------------------------------------------------------------
    # Reference frame extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reference_frame(
        reference_strip: Image.Image,
        frame_index: int,
        frame_width: int = 64,
        frame_height: int = 64,
    ) -> bytes:
        """Crop a single frame from a reference strip and return as PNG bytes.

        Args:
            reference_strip: The full row reference strip image.
            frame_index: Which frame to extract (0-based).
            frame_width: Width of each frame.
            frame_height: Height of each frame.

        Returns:
            PNG bytes of the cropped frame.

        Raises:
            ValueError: If frame_index is out of bounds for the strip.
        """
        max_frames = reference_strip.width // frame_width
        if frame_index < 0 or frame_index >= max_frames:
            raise ValueError(
                f"Frame index {frame_index} out of bounds for strip "
                f"with {max_frames} frames (width={reference_strip.width}, "
                f"frame_width={frame_width})."
            )

        left = frame_index * frame_width
        top = 0
        right = left + frame_width
        bottom = top + frame_height

        cropped = reference_strip.crop((left, top, right, bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Parallel gate execution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


# --------------------------------------------------------------------------
# Factory function for tiered model architecture
# --------------------------------------------------------------------------


async def create_workflow(
    config: SpritesheetSpec,
    project_endpoint: str | None = None,
    credential: Any | None = None,
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
    multiple token fetches. GPTImageProvider uses API key authentication
    and reads credentials from environment variables.

    Args:
        config: Spritesheet specification with model deployment names.
        project_endpoint: Azure AI Foundry endpoint. Falls back to
            AZURE_AI_PROJECT_ENDPOINT env var if not provided.
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

    # Resolve endpoint — prefer AZURE_OPENAI_ENDPOINT, fall back to
    # AZURE_AI_PROJECT_ENDPOINT (legacy) or AZURE_OPENAI_GPT_IMAGE_ENDPOINT
    endpoint = (
        project_endpoint
        or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        or os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
        or os.environ.get("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "")
    )
    if not endpoint:
        raise ProviderError(
            "No Azure OpenAI endpoint configured. "
            "Set AZURE_OPENAI_ENDPOINT or pass project_endpoint."
        )

    # Create or reuse credential
    # If user provided a credential, we don't own it and won't close it
    # If we create one, we'll store it and close it in workflow.close()
    shared_credential: Any
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

    # Build palette map (use palette, or will be replaced by preprocessor if auto_palette)
    palette_map: dict[str, tuple[int, int, int, int]]
    if config.palette is not None:
        palette_map = build_palette_map(config.palette)
    else:
        # Empty palette map — will be filled by preprocessor if auto_palette enabled
        palette_map = {}

    # Create workflow
    workflow = SpriteForgeWorkflow(
        config=config,
        reference_provider=reference_provider,
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=retry_manager,
        palette_map=palette_map,
        preprocessor=preprocessor,
        max_concurrent_rows=max_concurrent_rows,
        checkpoint_dir=checkpoint_dir,
        call_tracker=call_tracker,
    )

    # Store credential ownership info for cleanup
    workflow._owned_credential = shared_credential if owns_credential else None  # type: ignore[attr-defined]

    return workflow
