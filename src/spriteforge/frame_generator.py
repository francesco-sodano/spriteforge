"""Frame-level generation with verification gates and retry logic.

Extracted from SpriteForgeWorkflow to isolate frame-level concerns:
generate → programmatic-check → render → LLM-gate → retry.
"""

from __future__ import annotations

import asyncio
from typing import Any

from spriteforge.errors import RetryExhaustedError
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GridGenerator
from spriteforge.models import AnimationDef, FrameContext, GenerationConfig
from spriteforge.observability import RunMetricsCollector
from spriteforge.renderer import frame_to_png_bytes, render_frame
from spriteforge.retry import RetryManager


class FrameGenerator:
    """Encapsulates frame-level generation with verification and retry logic.

    This class handles the full generate → verify → retry cycle for a single frame:
    1. Generate frame via GridGenerator
    2. Run programmatic checks (fast-fail)
    3. Render grid to PNG
    4. Run LLM gates (Gate 0, Gate 1, Gate 2) in parallel
    5. If any gate fails: record failure, escalate, retry
    6. After max failures: raise RetryExhaustedError
    """

    def __init__(
        self,
        grid_generator: GridGenerator,
        gate_checker: LLMGateChecker,
        programmatic_checker: ProgrammaticChecker,
        retry_manager: RetryManager,
        generation_config: GenerationConfig,
        call_tracker: Any | None = None,
        metrics_collector: RunMetricsCollector | None = None,
    ) -> None:
        """Initialize the frame generator.

        Args:
            grid_generator: Stage 2 grid generator.
            gate_checker: LLM-based verification gate checker.
            programmatic_checker: Fast deterministic grid checks.
            retry_manager: Retry and escalation engine.
            generation_config: Generation settings (for budget config).
            call_tracker: Optional CallTracker for budget enforcement.
        """
        self.grid_generator = grid_generator
        self.gate_checker = gate_checker
        self.programmatic_checker = programmatic_checker
        self.retry_manager = retry_manager
        self.generation_config = generation_config
        self.call_tracker = call_tracker
        self.metrics_collector = metrics_collector
        self._closed = False

    async def close(self) -> None:
        """Close resources owned by frame generation."""
        if self._closed:
            return
        self._closed = True
        await self.grid_generator.close()

    def _record_usage(self, usage: dict[str, int] | None) -> None:
        """Record token usage in the call tracker when available."""
        if self.call_tracker is None or not usage:
            return
        self.call_tracker.record_tokens(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )

    def _record_usage_from_grid_generator(self) -> None:
        """Record token usage from the last grid generation call."""
        usage_getter = getattr(self.grid_generator, "get_last_usage", None)
        if callable(usage_getter):
            self._record_usage(usage_getter())

    def _record_usage_from_gate_verdicts(self, verdicts: list[GateVerdict]) -> None:
        """Record token usage from gate verdict metadata."""
        for verdict in verdicts:
            usage = verdict.details.get("token_usage")
            if isinstance(usage, dict):
                self._record_usage(usage)

    def _record_gate_verdicts(self, verdicts: list[GateVerdict]) -> None:
        """Record gate verdict pass/fail metrics when collector exists."""
        if self.metrics_collector is None:
            return
        for verdict in verdicts:
            self.metrics_collector.record_gate_verdict(verdict)

    async def generate_verified_frame(
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

        1. Generate frame via GridGenerator
        2. Run programmatic checks (fast-fail)
        3. Render grid to PNG
        4. Run LLM gates (Gate 0, Gate 1, optionally Gate 2) in parallel
        5. If any gate fails: record failure, escalate, retry
        6. After max failures: raise RetryExhaustedError

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
        animation = context.animation
        frame_id = f"row{animation.row}_frame{frame_index}"

        # Use per-row retry budget if configured
        max_attempts = None
        if (
            self.generation_config.budget
            and self.generation_config.budget.max_retries_per_row > 0
        ):
            max_attempts = self.generation_config.budget.max_retries_per_row

        retry_ctx = self.retry_manager.create_context(
            frame_id, max_attempts=max_attempts
        )

        while self.retry_manager.should_retry(retry_ctx):
            attempt = retry_ctx.current_attempt + 1
            temperature = self.retry_manager.get_temperature(attempt)
            guidance = ""
            if retry_ctx.current_attempt > 0:
                guidance = self.retry_manager.build_escalated_guidance(retry_ctx)

            # Generate the grid
            if self.call_tracker:
                self.call_tracker.increment("grid_generation")
            grid = await self.grid_generator.generate_frame(
                reference_frame=reference_frame,
                context=context,
                frame_index=frame_index,
                is_anchor=is_anchor,
                base_reference=base_reference,
                prev_frame_grid=prev_frame_grid if not is_anchor else None,
                prev_frame_rendered=prev_frame_rendered if not is_anchor else None,
                temperature=temperature,
                additional_guidance=guidance,
            )
            self._record_usage_from_grid_generator()

            # Programmatic checks (fast-fail)
            prog_verdicts = self.programmatic_checker.run_all(grid, context)
            self._record_gate_verdicts(prog_verdicts)
            prog_failures = [v for v in prog_verdicts if not v.passed]
            if prog_failures:
                retry_ctx = self.retry_manager.record_failure(
                    retry_ctx, prog_failures, grid=grid
                )
                continue

            # Render grid to PNG
            frame_img = render_frame(grid, context)
            frame_bytes = frame_to_png_bytes(frame_img)

            # Run LLM gates in parallel
            llm_verdicts = await self._run_gates_parallel(
                frame_rendered=frame_bytes,
                anchor_rendered=context.anchor_rendered,
                reference_frame=reference_frame,
                prev_frame_rendered=prev_frame_rendered,
                frame_index=frame_index,
                animation=animation,
                is_anchor=is_anchor,
            )
            self._record_usage_from_gate_verdicts(llm_verdicts)
            self._record_gate_verdicts(llm_verdicts)

            llm_failures = [v for v in llm_verdicts if not v.passed]
            if llm_failures:
                retry_ctx = self.retry_manager.record_failure(
                    retry_ctx, llm_failures, grid=grid
                )
                continue

            # All checks passed
            return grid

        # Exhausted all retries
        # Get the tier for the last attempt to include in error message
        last_attempt = max(1, retry_ctx.current_attempt)
        try:
            last_tier = self.retry_manager.get_tier(last_attempt)
        except ValueError:
            last_tier = self.retry_manager.get_tier(
                self.retry_manager._config.constrained_range[0]
            )
        raise RetryExhaustedError(
            f"Frame {frame_id} failed verification after "
            f"{retry_ctx.max_attempts} attempts. "
            f"Last tier: {last_tier.value}, "
            f"failures: {len(retry_ctx.failure_history)}"
        )

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

        if self.call_tracker:
            self.call_tracker.increment("gate_check")
        gate_0_verdict = await self.gate_checker.gate_0(
            frame_rendered,
            reference_frame,
            frame_desc,
        )
        if not gate_0_verdict.passed:
            return [gate_0_verdict]

        gates: list[Any] = []
        gate_count = 0

        if anchor_rendered is not None and not is_anchor:
            gates.append(
                self.gate_checker.gate_1(frame_rendered, anchor_rendered),
            )
            gate_count += 1

        if prev_frame_rendered is not None:
            gates.append(
                self.gate_checker.gate_2(frame_rendered, prev_frame_rendered),
            )
            gate_count += 1

        # Track gate calls (Gate 0 was already counted above)
        if self.call_tracker:
            for _ in range(gate_count):
                self.call_tracker.increment("gate_check")

        if not gates:
            return [gate_0_verdict]

        remaining = list(await asyncio.gather(*gates))
        return [gate_0_verdict, *remaining]
