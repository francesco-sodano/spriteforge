"""Row-level frame generation and row coherence checks."""

from __future__ import annotations

import asyncio
import io
import re
from typing import Any

from PIL import Image

from spriteforge.errors import GateError
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker
from spriteforge.logging import get_logger
from spriteforge.models import (
    AnimationDef,
    FrameContext,
    PaletteConfig,
    SpritesheetSpec,
)
from spriteforge.observability import RunMetricsCollector
from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.prompts.providers import build_reference_prompt
from spriteforge.renderer import frame_to_png_bytes, render_frame, render_row_strip

logger = get_logger("row_processor")


class RowProcessor:
    """Encapsulates row-level generation for anchor and non-anchor rows."""

    def __init__(
        self,
        config: SpritesheetSpec,
        frame_generator: FrameGenerator,
        gate_checker: LLMGateChecker,
        reference_provider: ReferenceProvider,
        call_tracker: Any | None = None,
        metrics_collector: RunMetricsCollector | None = None,
    ) -> None:
        self.config = config
        self.frame_generator = frame_generator
        self.gate_checker = gate_checker
        self.reference_provider = reference_provider
        self.call_tracker = call_tracker
        self.metrics_collector = metrics_collector
        self._closed = False

    async def close(self) -> None:
        """Close resources owned by row processing."""
        if self._closed:
            return
        self._closed = True
        await self.frame_generator.close()
        await self.gate_checker.close()
        if hasattr(self.reference_provider, "close"):
            await self.reference_provider.close()

    def _record_usage_from_verdict(self, verdict: GateVerdict) -> None:
        """Record token usage from gate verdict metadata when available."""
        if self.call_tracker is None:
            return
        usage = verdict.details.get("token_usage")
        if not isinstance(usage, dict):
            return
        self.call_tracker.record_tokens(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )

    def _record_gate_verdict(self, verdict: GateVerdict) -> None:
        """Record gate pass/fail metrics when collector exists."""
        if self.metrics_collector is None:
            return
        self.metrics_collector.record_gate_verdict(verdict)

    async def process_anchor_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        quantized_reference: bytes | None,
    ) -> tuple[list[str], bytes, list[list[str]]]:
        """Generate row 0 and return (anchor_grid, anchor_rendered, frame_grids)."""
        reference_strip = await self._generate_reference_strip(
            base_reference, animation
        )

        ref_frame = self._extract_reference_frame(
            reference_strip,
            0,
            self.config.character.frame_width,
            self.config.character.frame_height,
        )
        anchor_context = self._build_frame_context(
            palette=palette,
            palette_map=palette_map,
            animation=animation,
            anchor_grid=None,
            anchor_rendered=None,
            quantized_reference=quantized_reference,
        )
        generated_anchor_grid = await self.frame_generator.generate_verified_frame(
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
        frame_context = self._build_frame_context(
            palette=palette,
            palette_map=palette_map,
            animation=animation,
            anchor_grid=generated_anchor_grid,
            anchor_rendered=generated_anchor_rendered,
            quantized_reference=None,
        )

        prev_grid: list[str] | None = generated_anchor_grid
        prev_rendered: bytes | None = generated_anchor_rendered
        for fi in range(1, animation.frames):
            ref_frame = self._extract_reference_frame(
                reference_strip,
                fi,
                self.config.character.frame_width,
                self.config.character.frame_height,
            )
            grid = await self.frame_generator.generate_verified_frame(
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

        await self._run_gate_3a(
            base_reference=base_reference,
            reference_strip=reference_strip,
            animation=animation,
            frame_context=frame_context,
            frame_grids=frame_grids,
            is_anchor=True,
        )
        anchor_grid = frame_grids[0]
        anchor_rendered = frame_to_png_bytes(render_frame(anchor_grid, anchor_context))
        return anchor_grid, anchor_rendered, frame_grids

    async def process_row(
        self,
        base_reference: bytes,
        animation: AnimationDef,
        palette: PaletteConfig,
        palette_map: dict[str, tuple[int, int, int, int]],
        anchor_grid: list[str],
        anchor_rendered: bytes,
    ) -> list[list[str]]:
        """Generate a non-anchor row and return all frame grids."""
        reference_strip = await self._generate_reference_strip(
            base_reference, animation
        )
        frame_context = self._build_frame_context(
            palette=palette,
            palette_map=palette_map,
            animation=animation,
            anchor_grid=anchor_grid,
            anchor_rendered=anchor_rendered,
            quantized_reference=None,
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
            grid = await self.frame_generator.generate_verified_frame(
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

        await self._run_gate_3a(
            base_reference=base_reference,
            reference_strip=reference_strip,
            animation=animation,
            frame_context=frame_context,
            frame_grids=frame_grids,
            is_anchor=False,
        )
        return frame_grids

    async def _run_gate_3a(
        self,
        base_reference: bytes,
        reference_strip: Image.Image,
        animation: AnimationDef,
        frame_context: FrameContext,
        frame_grids: list[list[str]],
        is_anchor: bool,
    ) -> None:
        """Run Gate 3A row-coherence check with retries and frame regeneration."""
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
            self._record_usage_from_verdict(verdict)
            self._record_gate_verdict(verdict)
            if verdict.passed:
                return

            logger.warning(
                "Gate 3A failed for %s (attempt %d/%d): %s",
                animation.name,
                retry_attempt + 1,
                max_retries + 1,
                verdict.feedback,
            )
            if retry_attempt >= max_retries:
                raise GateError(
                    f"Gate 3A (row coherence) failed for '{animation.name}' "
                    f"after {max_retries + 1} attempts: {verdict.feedback}"
                )

            logger.info(
                "Retrying Gate 3A for %s: regenerating problematic frames",
                animation.name,
            )
            frames_to_regenerate = self._identify_problematic_frames(
                verdict.feedback, animation.frames
            )
            for fi in frames_to_regenerate:
                ref_frame = self._extract_reference_frame(
                    reference_strip,
                    fi,
                    self.config.character.frame_width,
                    self.config.character.frame_height,
                )
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
                frame_grids[fi] = await self.frame_generator.generate_verified_frame(
                    reference_frame=ref_frame,
                    context=frame_context,
                    frame_index=fi,
                    is_anchor=(is_anchor and fi == 0),
                    prev_frame_grid=prev_frame_grid,
                    prev_frame_rendered=prev_frame_rendered,
                    base_reference=base_reference,
                )

    async def _generate_reference_strip(
        self,
        base_reference: bytes,
        animation: AnimationDef,
    ) -> Image.Image:
        """Generate and verify a rough reference strip for an animation row."""
        max_ref_retries = 3
        prompt = build_reference_prompt(
            animation,
            self.config.character,
            self.config.character.description,
        )
        for attempt in range(max_ref_retries):
            if self.call_tracker:
                self.call_tracker.increment("reference_generation")
            timeout_s = self.config.generation.request_timeout_seconds
            try:
                strip = await asyncio.wait_for(
                    self.reference_provider.generate_row_strip(
                        base_reference=base_reference,
                        prompt=prompt,
                        num_frames=animation.frames,
                        frame_size=self.config.character.frame_size,
                    ),
                    timeout=timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "Reference strip generation timed out for %s "
                    "(attempt %d/%d, timeout=%.1fs)",
                    animation.name,
                    attempt + 1,
                    max_ref_retries,
                    timeout_s,
                )
                continue

            strip_bytes = frame_to_png_bytes(strip.convert("RGBA"))
            if self.call_tracker:
                self.call_tracker.increment("gate_minus_1")
            verdict = await self.gate_checker.gate_minus_1(
                strip_bytes, base_reference, animation
            )
            self._record_usage_from_verdict(verdict)
            self._record_gate_verdict(verdict)
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

    def _identify_problematic_frames(self, feedback: str, num_frames: int) -> list[int]:
        """Identify frame indices that may be problematic based on Gate 3A feedback."""
        frame_list_pattern = r"[Ff]rame[s]?\s+([0-9,\sand]+)"
        matches: list[str] = []
        for chunk in re.findall(frame_list_pattern, feedback):
            matches.extend(re.findall(r"\d+", chunk))

        if matches:
            indices = []
            for match in matches:
                idx = int(match)
                if idx >= num_frames:
                    logger.warning(
                        "Gate 3A feedback references out-of-range frame index %d "
                        "(num_frames=%d); clamping to last frame.",
                        idx,
                        num_frames,
                    )
                    idx = num_frames - 1
                if 0 <= idx < num_frames:
                    indices.append(idx)
            if indices:
                logger.info(
                    "Identified frames to regenerate from feedback: %s",
                    sorted(set(indices)),
                )
                return sorted(set(indices))

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
        """Build a FrameContext for frame generation in this row."""
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

    @staticmethod
    def _extract_reference_frame(
        reference_strip: Image.Image,
        frame_index: int,
        frame_width: int = 64,
        frame_height: int = 64,
    ) -> bytes:
        """Crop a single frame from a reference strip and return as PNG bytes."""
        max_frames = reference_strip.width // frame_width
        if frame_index < 0 or frame_index >= max_frames:
            raise ValueError(
                f"Frame index {frame_index} out of bounds for strip "
                f"with {max_frames} frames (width={reference_strip.width}, "
                f"frame_width={frame_width})."
            )

        left = frame_index * frame_width
        cropped = reference_strip.crop((left, 0, left + frame_width, frame_height))
        with io.BytesIO() as buf:
            cropped.save(buf, format="PNG")
            return buf.getvalue()
