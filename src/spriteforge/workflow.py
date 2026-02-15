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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from spriteforge.assembler import assemble_spritesheet
from spriteforge.errors import GateError
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GenerationError, GridGenerator
from spriteforge.logging import get_logger
from spriteforge.models import AnimationDef, PaletteConfig, SpritesheetSpec
from spriteforge.palette import build_palette_map
from spriteforge.preprocessor import PreprocessResult, preprocess_reference
from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.providers.gpt_image import build_reference_prompt
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
        palette = self._get_palette()

        logger.info(
            "Starting spritesheet generation for '%s' (%d rows)",
            self.config.character.name,
            total_rows,
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

            # If auto_palette mode, replace palette with extracted one
            if self.config.generation.auto_palette:
                self.config.palettes["P1"] = preprocess_result.palette
                self.palette_map = build_palette_map(preprocess_result.palette)
                palette = preprocess_result.palette
            if progress_callback:
                progress_callback("preprocessing", 1, 1)

        # ---- Process Row 0 (anchor row) ----
        if progress_callback:
            progress_callback("row", 0, total_rows)

        anchor_animation = self.config.animations[0]
        logger.info(
            "Processing row 0/%d: %s (%d frames)",
            total_rows,
            anchor_animation.name,
            anchor_animation.frames,
        )
        anchor_grid, row0_grids = await self._process_anchor_row(
            base_reference,
            anchor_animation,
            palette,
            quantized_reference=quantized_reference,
        )

        # Render and save anchor row strip
        anchor_rendered = frame_to_png_bytes(
            render_frame(
                anchor_grid,
                self.palette_map,
                frame_width=self.config.character.frame_width,
                frame_height=self.config.character.frame_height,
            )
        )
        row_images: dict[int, bytes] = {}
        row0_strip = render_row_strip(
            row0_grids,
            self.palette_map,
            spritesheet_columns=self.config.character.spritesheet_columns,
            frame_width=self.config.character.frame_width,
            frame_height=self.config.character.frame_height,
        )
        row_images[anchor_animation.row] = frame_to_png_bytes(row0_strip)

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
            completed_lock = asyncio.Lock()

            async def _process_one(row_idx_seq: int, animation: AnimationDef) -> None:
                nonlocal completed_count

                async def _inner() -> None:
                    nonlocal completed_count
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
                        anchor_grid,
                        anchor_rendered,
                        palette,
                    )

                    row_strip = render_row_strip(
                        row_grids,
                        self.palette_map,
                        spritesheet_columns=self.config.character.spritesheet_columns,
                        frame_width=self.config.character.frame_width,
                        frame_height=self.config.character.frame_height,
                    )
                    row_images[animation.row] = frame_to_png_bytes(row_strip)

                    logger.info(
                        "Row %d (%s) complete: %d/%d frames generated",
                        row_idx_seq,
                        animation.name,
                        animation.frames,
                        animation.frames,
                    )

                    if progress_callback:
                        async with completed_lock:
                            completed_count += 1
                            progress_callback("row", 1 + completed_count, total_rows)

                if semaphore is not None:
                    async with semaphore:
                        await _inner()
                else:
                    await _inner()

            await asyncio.gather(
                *(_process_one(idx, anim) for idx, anim in remaining_animations)
            )

        # ---- Assemble final spritesheet ----
        if progress_callback:
            progress_callback("assembly", 0, 1)

        assemble_spritesheet(row_images, self.config, output_path=out)

        logger.info("Spritesheet saved: %s", out)

        if progress_callback:
            progress_callback("assembly", 1, 1)

        return out

    # ------------------------------------------------------------------
    # Row processing
    # ------------------------------------------------------------------

    async def _process_anchor_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        palette: PaletteConfig,
        quantized_reference: bytes | None = None,
    ) -> tuple[list[str], list[list[str]]]:
        """Process Row 0 — generates the anchor frame first.

        Returns:
            Tuple of (anchor_grid, list_of_all_frame_grids_for_this_row).
        """
        # Stage 1: Generate reference strip
        reference_strip = await self._generate_reference_strip(
            base_reference, animation
        )

        # Generate anchor frame (Frame 0)
        ref_frame = self._extract_reference_frame(
            reference_strip,
            0,
            self.config.character.frame_width,
            self.config.character.frame_height,
        )

        anchor_grid = await self._generate_and_verify_frame(
            reference_frame=ref_frame,
            anchor_grid=None,
            anchor_rendered=None,
            palette=palette,
            animation=animation,
            frame_index=0,
            is_anchor=True,
            base_reference=base_reference,
            quantized_reference=quantized_reference,
        )

        anchor_rendered = frame_to_png_bytes(
            render_frame(
                anchor_grid,
                self.palette_map,
                frame_width=self.config.character.frame_width,
                frame_height=self.config.character.frame_height,
            )
        )

        frame_grids: list[list[str]] = [anchor_grid]

        # Generate remaining frames with anchor + prev frame context
        prev_grid = anchor_grid
        prev_rendered = anchor_rendered
        for fi in range(1, animation.frames):
            ref_frame = self._extract_reference_frame(
                reference_strip,
                fi,
                self.config.character.frame_width,
                self.config.character.frame_height,
            )
            grid = await self._generate_and_verify_frame(
                reference_frame=ref_frame,
                anchor_grid=anchor_grid,
                anchor_rendered=anchor_rendered,
                palette=palette,
                animation=animation,
                frame_index=fi,
                prev_frame_grid=prev_grid,
                prev_frame_rendered=prev_rendered,
                base_reference=base_reference,
            )
            frame_grids.append(grid)
            prev_grid = grid
            prev_rendered = frame_to_png_bytes(
                render_frame(
                    grid,
                    self.palette_map,
                    frame_width=self.config.character.frame_width,
                    frame_height=self.config.character.frame_height,
                )
            )

        # Gate 3A: Validate assembled row
        row_strip = render_row_strip(
            frame_grids,
            self.palette_map,
            spritesheet_columns=self.config.character.spritesheet_columns,
            frame_width=self.config.character.frame_width,
            frame_height=self.config.character.frame_height,
        )
        row_strip_bytes = frame_to_png_bytes(row_strip)
        ref_strip_bytes = frame_to_png_bytes(reference_strip.convert("RGBA"))

        verdict = await self.gate_checker.gate_3a(
            row_strip_bytes, ref_strip_bytes, animation
        )
        if not verdict.passed:
            logger.warning(
                "Gate 3A failed for %s: %s", animation.name, verdict.feedback
            )
            raise GateError(
                f"Gate 3A (row coherence) failed for '{animation.name}': "
                f"{verdict.feedback}"
            )

        return anchor_grid, frame_grids

    async def _process_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        anchor_grid: list[str],
        anchor_rendered: bytes,
        palette: PaletteConfig,
    ) -> list[list[str]]:
        """Process a single animation row (rows after Row 0).

        Returns:
            List of frame grids for this row.
        """
        # Stage 1: Generate reference strip
        reference_strip = await self._generate_reference_strip(
            base_reference, animation
        )

        frame_grids: list[list[str]] = []
        prev_grid: list[str] | None = None
        prev_rendered: bytes | None = None

        for fi in range(animation.frames):
            ref_frame = self._extract_reference_frame(
                reference_strip,
                fi,
                self.config.character.frame_width,
                self.config.character.frame_height,
            )
            grid = await self._generate_and_verify_frame(
                reference_frame=ref_frame,
                anchor_grid=anchor_grid,
                anchor_rendered=anchor_rendered,
                palette=palette,
                animation=animation,
                frame_index=fi,
                prev_frame_grid=prev_grid,
                prev_frame_rendered=prev_rendered,
                base_reference=base_reference,
            )
            frame_grids.append(grid)
            prev_grid = grid
            prev_rendered = frame_to_png_bytes(
                render_frame(
                    grid,
                    self.palette_map,
                    frame_width=self.config.character.frame_width,
                    frame_height=self.config.character.frame_height,
                )
            )

        # Gate 3A: Validate assembled row
        row_strip = render_row_strip(
            frame_grids,
            self.palette_map,
            spritesheet_columns=self.config.character.spritesheet_columns,
            frame_width=self.config.character.frame_width,
            frame_height=self.config.character.frame_height,
        )
        row_strip_bytes = frame_to_png_bytes(row_strip)
        ref_strip_bytes = frame_to_png_bytes(reference_strip.convert("RGBA"))

        verdict = await self.gate_checker.gate_3a(
            row_strip_bytes, ref_strip_bytes, animation
        )
        if not verdict.passed:
            logger.warning(
                "Gate 3A failed for %s: %s", animation.name, verdict.feedback
            )
            raise GateError(
                f"Gate 3A (row coherence) failed for '{animation.name}': "
                f"{verdict.feedback}"
            )

        return frame_grids

    # ------------------------------------------------------------------
    # Frame generation with retry loop
    # ------------------------------------------------------------------

    async def _generate_and_verify_frame(
        self,
        reference_frame: bytes,
        anchor_grid: list[str] | None,
        anchor_rendered: bytes | None,
        palette: PaletteConfig,
        animation: AnimationDef,
        frame_index: int,
        prev_frame_grid: list[str] | None = None,
        prev_frame_rendered: bytes | None = None,
        is_anchor: bool = False,
        base_reference: bytes | None = None,
        quantized_reference: bytes | None = None,
    ) -> list[str]:
        """Generate a single frame with full verification and retry loop.

        1. Generate frame via GridGenerator (Claude Opus 4.6)
        2. Run programmatic checks (fast-fail)
        3. Render grid to PNG
        4. Run LLM gates (Gate 0, Gate 1, optionally Gate 2) in parallel
        5. If any gate fails: record failure, escalate, retry
        6. After max failures: raise GenerationError

        Returns:
            Verified frame grid (list of 64 strings).
        """
        frame_id = f"row{animation.row}_frame{frame_index}"
        retry_ctx = self.retry_manager.create_context(frame_id)

        while self.retry_manager.should_retry(retry_ctx):
            attempt = retry_ctx.current_attempt + 1
            temperature = self.retry_manager.get_temperature(attempt)
            guidance = ""
            if retry_ctx.current_attempt > 0:
                guidance = self.retry_manager.build_escalated_guidance(retry_ctx)

            # Generate the grid
            if is_anchor:
                grid = await self.grid_generator.generate_anchor_frame(
                    base_reference=base_reference or b"",
                    reference_frame=reference_frame,
                    palette=palette,
                    animation=animation,
                    generation=self.config.generation,
                    quantized_reference=quantized_reference,
                )
            else:
                grid = await self.grid_generator.generate_frame(
                    reference_frame=reference_frame,
                    anchor_grid=anchor_grid or [],
                    anchor_rendered=anchor_rendered or b"",
                    palette=palette,
                    animation=animation,
                    frame_index=frame_index,
                    generation=self.config.generation,
                    prev_frame_grid=prev_frame_grid,
                    prev_frame_rendered=prev_frame_rendered,
                    temperature=temperature,
                    additional_guidance=guidance,
                )

            # Programmatic checks (fast-fail)
            prog_verdicts = self.programmatic_checker.run_all(
                grid,
                palette,
                frame_width=self.config.character.frame_width,
                frame_height=self.config.character.frame_height,
            )
            prog_failures = [v for v in prog_verdicts if not v.passed]
            if prog_failures:
                retry_ctx = self.retry_manager.record_failure(
                    retry_ctx, prog_failures, grid=grid
                )
                continue

            # Render grid to PNG
            frame_img = render_frame(
                grid,
                self.palette_map,
                frame_width=self.config.character.frame_width,
                frame_height=self.config.character.frame_height,
            )
            frame_bytes = frame_to_png_bytes(frame_img)

            # Run LLM gates in parallel
            llm_verdicts = await self._run_gates_parallel(
                frame_rendered=frame_bytes,
                anchor_rendered=anchor_rendered,
                reference_frame=reference_frame,
                prev_frame_rendered=prev_frame_rendered,
                frame_index=frame_index,
                animation=animation,
                is_anchor=is_anchor,
            )

            llm_failures = [v for v in llm_verdicts if not v.passed]
            if llm_failures:
                retry_ctx = self.retry_manager.record_failure(
                    retry_ctx, llm_failures, grid=grid
                )
                continue

            # All checks passed
            return grid

        # Exhausted all retries
        raise GenerationError(
            f"Frame {frame_id} failed verification after "
            f"{retry_ctx.max_attempts} attempts."
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
            strip = await self.reference_provider.generate_row_strip(
                base_reference=base_reference,
                prompt=prompt,
                num_frames=animation.frames,
                frame_size=self.config.character.frame_size,
            )

            # Gate -1: Validate reference quality
            strip_bytes = frame_to_png_bytes(strip.convert("RGBA"))
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

    async def _run_gates_parallel(
        self,
        frame_rendered: bytes,
        anchor_rendered: bytes | None,
        reference_frame: bytes,
        prev_frame_rendered: bytes | None,
        frame_index: int,
        animation: AnimationDef,
        is_anchor: bool = False,
    ) -> list[GateVerdict]:
        """Run independent LLM gates in parallel using asyncio.gather().

        Gate 0 (reference fidelity) always runs.
        Gate 1 (anchor consistency) runs only when an anchor exists
        (i.e. not for the anchor frame itself).
        Gate 2 (temporal continuity) runs only when a previous frame exists.

        Returns:
            List of GateResult objects from all gates.
        """
        frame_desc = ""
        if animation.frame_descriptions and frame_index < len(
            animation.frame_descriptions
        ):
            frame_desc = animation.frame_descriptions[frame_index]

        gates: list[Any] = [
            self.gate_checker.gate_0(frame_rendered, reference_frame, frame_desc),
        ]

        if anchor_rendered is not None and not is_anchor:
            gates.append(
                self.gate_checker.gate_1(frame_rendered, anchor_rendered),
            )

        if prev_frame_rendered is not None:
            gates.append(
                self.gate_checker.gate_2(frame_rendered, prev_frame_rendered),
            )

        return list(await asyncio.gather(*gates))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_palette(self) -> PaletteConfig:
        """Get the primary palette from the config.

        Raises:
            ValueError: If no palette is configured.
        """
        if "P1" in self.config.palettes:
            return self.config.palettes["P1"]
        if self.config.palettes:
            return next(iter(self.config.palettes.values()))
        raise ValueError(
            "No palette configured. Provide a palette in the YAML config "
            "or enable auto_palette with a preprocessor."
        )
