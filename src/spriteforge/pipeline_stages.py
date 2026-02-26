"""Explicit pipeline stage classes used by SpriteForgeWorkflow."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from spriteforge.errors import RowGenerationError
from spriteforge.logging import get_logger
from spriteforge.models import AnimationDef, PaletteConfig
from spriteforge.palette import build_palette_map
from spriteforge.renderer import frame_to_png_bytes, render_frame, render_row_strip

logger = get_logger("pipeline_stages")


class PreprocessingStage:
    """Optional preprocessing stage (resize/quantize/palette extraction)."""

    def __init__(
        self,
        workflow: Any,
        base_reference_path: str,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        self.workflow = workflow
        self.base_reference_path = base_reference_path
        self.progress_callback = progress_callback

    def execute(
        self,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
    ) -> tuple[PaletteConfig, dict[str, tuple[int, int, int, int]], bytes | None]:
        """Run preprocessing and return possibly-updated palette data."""
        quantized_reference: bytes | None = None
        if self.workflow.preprocessor is None:
            return palette, palette_map, quantized_reference

        if self.progress_callback:
            self.progress_callback("preprocessing", 0, 1)

        preprocess_result = self.workflow.preprocessor(
            image_path=self.base_reference_path,
            frame_width=self.workflow.config.character.frame_width,
            frame_height=self.workflow.config.character.frame_height,
            max_colors=self.workflow.config.generation.max_palette_colors,
            semantic_labels=self.workflow.config.generation.semantic_labels,
        )
        quantized_reference = preprocess_result.quantized_png_bytes

        if self.workflow.config.generation.auto_palette:
            palette = preprocess_result.palette
            palette_map = build_palette_map(preprocess_result.palette)

        if self.progress_callback:
            self.progress_callback("preprocessing", 1, 1)

        return palette, palette_map, quantized_reference


class AnchorRowStage:
    """Process row 0 to establish the anchor identity frame."""

    def __init__(
        self,
        workflow: Any,
        base_reference: bytes,
        total_rows: int,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        self.workflow = workflow
        self.base_reference = base_reference
        self.total_rows = total_rows
        self.progress_callback = progress_callback

    async def execute(
        self,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        quantized_reference: bytes | None,
        completed_rows: set[int],
        row_images: dict[int, bytes],
    ) -> tuple[AnimationDef, list[str], bytes]:
        """Generate or load anchor row and return anchor artifacts."""
        if self.progress_callback:
            self.progress_callback("row", 0, self.total_rows)

        anchor_animation = self.workflow.config.animations[0]

        if (
            anchor_animation.row in completed_rows
            and self.workflow.checkpoint_manager is not None
        ):
            logger.info(
                "Loading row 0/%d from checkpoint: %s (%d frames)",
                self.total_rows,
                anchor_animation.name,
                anchor_animation.frames,
            )
            checkpoint_data = self.workflow.checkpoint_manager.load_row(
                anchor_animation.row
            )
            if checkpoint_data is None:
                raise RuntimeError(
                    f"Checkpoint for row {anchor_animation.row} was reported as "
                    "completed but could not be loaded"
                )
            row0_strip_bytes, row0_grids = checkpoint_data
            anchor_grid = row0_grids[0]
            anchor_render_context = self.workflow.row_processor._build_frame_context(
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
                self.total_rows,
                anchor_animation.name,
                anchor_animation.frames,
            )
            anchor_grid, anchor_rendered, row0_grids = (
                await self.workflow.row_processor.process_anchor_row(
                    self.base_reference,
                    anchor_animation,
                    palette,
                    palette_map,
                    quantized_reference=quantized_reference,
                )
            )

            anchor_render_context = self.workflow.row_processor._build_frame_context(
                palette=palette,
                palette_map=palette_map,
                animation=anchor_animation,
                anchor_grid=anchor_grid,
                anchor_rendered=None,
                quantized_reference=None,
            )
            row0_strip = render_row_strip(row0_grids, anchor_render_context)
            row_images[anchor_animation.row] = frame_to_png_bytes(row0_strip)

            if self.workflow.checkpoint_manager is not None:
                self.workflow.checkpoint_manager.save_row(
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

        if self.progress_callback:
            self.progress_callback("row", 1, self.total_rows)

        return anchor_animation, anchor_grid, anchor_rendered


class RemainingRowsStage:
    """Generate non-anchor rows in parallel with anchor-recovery policy."""

    def __init__(
        self,
        workflow: Any,
        base_reference: bytes,
        total_rows: int,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        self.workflow = workflow
        self.base_reference = base_reference
        self.total_rows = total_rows
        self.progress_callback = progress_callback

    async def execute(
        self,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        quantized_reference: bytes | None,
        anchor_animation: AnimationDef,
        anchor_grid: list[str],
        anchor_rendered: bytes,
        completed_rows: set[int],
        row_images: dict[int, bytes],
    ) -> None:
        remaining_animations = list(
            enumerate(self.workflow.config.animations[1:], start=1)
        )

        if not remaining_animations:
            return

        anchor_recovery = self.workflow._build_anchor_recovery_policy()
        anchor_recovery_state = self.workflow._build_anchor_recovery_state()

        while True:
            semaphore: asyncio.Semaphore | None = None
            if self.workflow.max_concurrent_rows > 0:
                semaphore = asyncio.Semaphore(self.workflow.max_concurrent_rows)

            completed_count = 0
            failed_rows: list[tuple[int, str, Exception]] = []

            async def _process_one(
                row_idx_seq: int, animation: AnimationDef
            ) -> tuple[int, AnimationDef, list[list[str]] | Exception]:
                if semaphore is not None:
                    await semaphore.acquire()
                try:
                    if (
                        animation.row in completed_rows
                        and self.workflow.checkpoint_manager is not None
                    ):
                        logger.info(
                            "Loading row %d/%d from checkpoint: %s (%d frames)",
                            row_idx_seq,
                            self.total_rows,
                            animation.name,
                            animation.frames,
                        )
                        checkpoint_data = self.workflow.checkpoint_manager.load_row(
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
                            self.total_rows,
                            animation.name,
                            animation.frames,
                        )

                        row_grids = await self.workflow.row_processor.process_row(
                            self.base_reference,
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
                except Exception as exc:
                    logger.error(
                        "Row %d (%s) failed: %s",
                        row_idx_seq,
                        animation.name,
                        str(exc),
                    )
                    return (row_idx_seq, animation, exc)
                finally:
                    if semaphore is not None:
                        semaphore.release()

            results = await asyncio.gather(
                *(_process_one(idx, anim) for idx, anim in remaining_animations),
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.error("Unexpected gather exception: %s", result)
                    failed_rows.append((-1, "unknown", result))
                    continue

                assert not isinstance(result, BaseException)
                row_idx_seq, animation, outcome = result

                if isinstance(outcome, Exception):
                    failed_rows.append((row_idx_seq, animation.name, outcome))
                else:
                    row_grids = outcome
                    if animation.row not in row_images:
                        row_render_context = (
                            self.workflow.row_processor._build_frame_context(
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

                        if self.workflow.checkpoint_manager is not None:
                            self.workflow.checkpoint_manager.save_row(
                                row=animation.row,
                                animation_name=animation.name,
                                strip_bytes=row_images[animation.row],
                                grids=row_grids,
                            )

                    completed_count += 1
                    if self.progress_callback:
                        self.progress_callback(
                            "row", 1 + completed_count, self.total_rows
                        )

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

            anchor_grid, anchor_rendered = await self.workflow._regenerate_anchor_row(
                base_reference=self.base_reference,
                anchor_animation=anchor_animation,
                palette=palette,
                palette_map=palette_map,
                quantized_reference=quantized_reference,
                row_images=row_images,
            )

            completed_rows = self.workflow._reset_non_anchor_progress(
                animations=self.workflow.config.animations,
                row_images=row_images,
                anchor_row=anchor_animation.row,
            )
