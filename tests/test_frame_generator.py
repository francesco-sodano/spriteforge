"""Tests for spriteforge.frame_generator — frame-level generation with verification."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from spriteforge.errors import RetryExhaustedError
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GridGenerator
from spriteforge.models import (
    AnimationDef,
    FrameContext,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
)
from spriteforge.palette import build_palette_map
from spriteforge.retry import RetryManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal 1×1 transparent PNG for use as image bytes.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_valid_grid(fill: str = ".", rows: int = 64, cols: int = 64) -> list[str]:
    """Create a uniform grid filled with a single symbol."""
    return [fill * cols for _ in range(rows)]


def _make_sprite_grid() -> list[str]:
    """Create a minimal valid sprite grid with outline + content."""
    grid = ["." * 64 for _ in range(64)]
    # Place some outline and skin pixels mid-grid
    grid[30] = "." * 20 + "O" * 10 + "s" * 10 + "." * 24
    grid[56] = "." * 20 + "O" * 10 + "." * 34  # Feet area
    return grid


def _passing_verdict(gate_name: str = "gate_0") -> GateVerdict:
    """Create a passing GateVerdict."""
    return GateVerdict(
        gate_name=gate_name,
        passed=True,
        confidence=0.9,
        feedback="Looks good.",
    )


def _failing_verdict(gate_name: str = "gate_0") -> GateVerdict:
    """Create a failing GateVerdict."""
    return GateVerdict(
        gate_name=gate_name,
        passed=False,
        confidence=0.3,
        feedback="Quality issue detected.",
    )


@pytest.fixture()
def sample_palette() -> PaletteConfig:
    """A minimal palette for testing."""
    return PaletteConfig(
        name="P1",
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
            PaletteColor(element="Hair", symbol="h", r=220, g=185, b=90),
        ],
    )


@pytest.fixture()
def sample_animation() -> AnimationDef:
    """A minimal animation definition for testing."""
    return AnimationDef(
        name="idle",
        row=0,
        frames=3,
        timing_ms=150,
        prompt_context="Standing idle",
    )


@pytest.fixture()
def sample_context(
    sample_palette: PaletteConfig,
    sample_animation: AnimationDef,
) -> FrameContext:
    """A minimal FrameContext for testing."""
    palette_map = build_palette_map(sample_palette)
    return FrameContext(
        palette=sample_palette,
        palette_map=palette_map,
        generation=GenerationConfig(),
        frame_width=64,
        frame_height=64,
        animation=sample_animation,
        spritesheet_columns=14,
        anchor_grid=None,
        anchor_rendered=None,
        quantized_reference=None,
    )


def _build_frame_generator(
    grid_generator: Any = None,
    gate_checker: Any = None,
    programmatic_checker: Any = None,
    retry_manager: Any = None,
    generation_config: GenerationConfig | None = None,
    call_tracker: Any = None,
) -> FrameGenerator:
    """Build a FrameGenerator with mock dependencies."""
    if grid_generator is None:
        grid_generator = AsyncMock(spec=GridGenerator)
        grid_generator.generate_frame = AsyncMock(return_value=_make_sprite_grid())

    if gate_checker is None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

    if programmatic_checker is None:
        programmatic_checker = MagicMock(spec=ProgrammaticChecker)
        programmatic_checker.run_all = MagicMock(
            return_value=[_passing_verdict("programmatic")]
        )

    if retry_manager is None:
        retry_manager = RetryManager()

    if generation_config is None:
        generation_config = GenerationConfig()

    return FrameGenerator(
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=retry_manager,
        generation_config=generation_config,
        call_tracker=call_tracker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateVerifiedFrameRetry:
    """Tests for frame retry behavior on gate failure."""

    @pytest.mark.asyncio
    async def test_run_frame_retry_on_gate_failure(
        self,
        sample_context: FrameContext,
    ) -> None:
        """Gate fails first attempt → retries with escalated params → succeeds."""
        # Make gate_0 fail on first call, pass on second
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_0 = AsyncMock(
            side_effect=[
                _failing_verdict("gate_0"),
                _passing_verdict("gate_0"),
            ]
        )
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        frame_gen = _build_frame_generator(gate_checker=gate_checker)

        # Should succeed after retry
        grid = await frame_gen.generate_verified_frame(
            reference_frame=_TINY_PNG,
            context=sample_context,
            frame_index=0,
            is_anchor=True,
            base_reference=_TINY_PNG,
        )

        assert len(grid) == 64
        assert len(grid[0]) == 64
        # Verify gate_0 was called twice (once failed, once succeeded)
        assert gate_checker.gate_0.call_count == 2


class TestRunFrameExhaustsRetries:
    """All attempts fail → raises RetryExhaustedError."""

    @pytest.mark.asyncio
    async def test_run_frame_exhausts_retries(
        self,
        sample_context: FrameContext,
    ) -> None:
        """All retry attempts fail → raises RetryExhaustedError."""
        # Make programmatic checker always fail
        prog_checker = MagicMock(spec=ProgrammaticChecker)
        prog_checker.run_all = MagicMock(
            return_value=[_failing_verdict("programmatic")]
        )

        frame_gen = _build_frame_generator(programmatic_checker=prog_checker)

        with pytest.raises(RetryExhaustedError, match="failed verification"):
            await frame_gen.generate_verified_frame(
                reference_frame=_TINY_PNG,
                context=sample_context,
                frame_index=0,
                is_anchor=True,
                base_reference=_TINY_PNG,
            )

    @pytest.mark.asyncio
    async def test_retry_exhausted_error_message(
        self,
        sample_context: FrameContext,
    ) -> None:
        """Verify RetryExhaustedError message includes frame ID, attempts, tier, and failure count."""
        # Make programmatic checker always fail
        prog_checker = MagicMock(spec=ProgrammaticChecker)
        prog_checker.run_all = MagicMock(
            return_value=[_failing_verdict("programmatic")]
        )

        frame_gen = _build_frame_generator(programmatic_checker=prog_checker)

        try:
            await frame_gen.generate_verified_frame(
                reference_frame=_TINY_PNG,
                context=sample_context,
                frame_index=0,
                is_anchor=True,
                base_reference=_TINY_PNG,
            )
            pytest.fail("Expected RetryExhaustedError to be raised")
        except RetryExhaustedError as e:
            error_message = str(e)
            # Verify all required components are in the message
            assert "row0_frame0" in error_message or "Frame" in error_message
            assert "10 attempts" in error_message  # max_attempts
            assert "tier" in error_message.lower()
            assert "constrained" in error_message  # Last tier
            assert "failures:" in error_message.lower()


class TestRunGatesParallel:
    """Tests for parallel gate execution."""

    @pytest.mark.asyncio
    async def test_run_gates_parallel_gate_0_always_runs(
        self,
        sample_context: FrameContext,
    ) -> None:
        """Gate 0 (reference fidelity) always runs."""
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        frame_gen = _build_frame_generator(gate_checker=gate_checker)

        # Anchor frame: only gate_0 should run
        grid = await frame_gen.generate_verified_frame(
            reference_frame=_TINY_PNG,
            context=sample_context,
            frame_index=0,
            is_anchor=True,
            base_reference=_TINY_PNG,
        )

        assert gate_checker.gate_0.call_count == 1
        # Gate 1 and 2 should not run for anchor frame
        assert gate_checker.gate_1.call_count == 0
        assert gate_checker.gate_2.call_count == 0

    @pytest.mark.asyncio
    async def test_run_gates_parallel_gate_1_runs_for_non_anchor(
        self,
        sample_context: FrameContext,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Gate 1 (anchor consistency) runs for non-anchor frames with anchor."""
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        frame_gen = _build_frame_generator(gate_checker=gate_checker)

        # Create a new context with anchor_rendered set
        palette_map = build_palette_map(sample_palette)
        context_with_anchor = FrameContext(
            palette=sample_palette,
            palette_map=palette_map,
            generation=GenerationConfig(),
            frame_width=64,
            frame_height=64,
            animation=sample_animation,
            spritesheet_columns=14,
            anchor_grid=_make_sprite_grid(),
            anchor_rendered=_TINY_PNG,
            quantized_reference=None,
        )

        # Non-anchor frame with anchor reference
        grid = await frame_gen.generate_verified_frame(
            reference_frame=_TINY_PNG,
            context=context_with_anchor,
            frame_index=1,
            is_anchor=False,
            prev_frame_grid=_make_sprite_grid(),
            prev_frame_rendered=_TINY_PNG,
        )

        assert gate_checker.gate_0.call_count == 1
        assert gate_checker.gate_1.call_count == 1  # Should run for non-anchor
        assert gate_checker.gate_2.call_count == 1  # Should run with prev frame

    @pytest.mark.asyncio
    async def test_run_gates_parallel_gate_2_runs_with_prev_frame(
        self,
        sample_context: FrameContext,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Gate 2 (temporal continuity) runs when previous frame exists."""
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        frame_gen = _build_frame_generator(gate_checker=gate_checker)

        # Create a new context with anchor_rendered set
        palette_map = build_palette_map(sample_palette)
        context_with_anchor = FrameContext(
            palette=sample_palette,
            palette_map=palette_map,
            generation=GenerationConfig(),
            frame_width=64,
            frame_height=64,
            animation=sample_animation,
            spritesheet_columns=14,
            anchor_grid=_make_sprite_grid(),
            anchor_rendered=_TINY_PNG,
            quantized_reference=None,
        )

        # Frame with previous frame
        grid = await frame_gen.generate_verified_frame(
            reference_frame=_TINY_PNG,
            context=context_with_anchor,
            frame_index=1,
            is_anchor=False,
            prev_frame_grid=_make_sprite_grid(),
            prev_frame_rendered=_TINY_PNG,
        )

        assert gate_checker.gate_0.call_count == 1
        assert gate_checker.gate_2.call_count == 1  # Should run with prev frame


class TestCallTracking:
    """Tests for call tracking integration."""

    @pytest.mark.asyncio
    async def test_call_tracker_increments_grid_generation(
        self,
        sample_context: FrameContext,
    ) -> None:
        """CallTracker increments grid_generation counter."""
        call_tracker = MagicMock()
        call_tracker.increment = MagicMock()

        frame_gen = _build_frame_generator(call_tracker=call_tracker)

        grid = await frame_gen.generate_verified_frame(
            reference_frame=_TINY_PNG,
            context=sample_context,
            frame_index=0,
            is_anchor=True,
            base_reference=_TINY_PNG,
        )

        # Should increment grid_generation once
        assert call_tracker.increment.call_count >= 1
        call_tracker.increment.assert_any_call("grid_generation")

    @pytest.mark.asyncio
    async def test_call_tracker_increments_gate_checks(
        self,
        sample_context: FrameContext,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """CallTracker increments gate_check counter for each gate."""
        call_tracker = MagicMock()
        call_tracker.increment = MagicMock()

        # Create a new context with anchor_rendered set for Gate 1
        palette_map = build_palette_map(sample_palette)
        context_with_anchor = FrameContext(
            palette=sample_palette,
            palette_map=palette_map,
            generation=GenerationConfig(),
            frame_width=64,
            frame_height=64,
            animation=sample_animation,
            spritesheet_columns=14,
            anchor_grid=_make_sprite_grid(),
            anchor_rendered=_TINY_PNG,
            quantized_reference=None,
        )

        frame_gen = _build_frame_generator(call_tracker=call_tracker)

        grid = await frame_gen.generate_verified_frame(
            reference_frame=_TINY_PNG,
            context=context_with_anchor,
            frame_index=1,
            is_anchor=False,
            prev_frame_grid=_make_sprite_grid(),
            prev_frame_rendered=_TINY_PNG,
        )

        # Should increment gate_check three times (Gate 0, Gate 1, Gate 2)
        gate_check_calls = [
            call
            for call in call_tracker.increment.call_args_list
            if call[0][0] == "gate_check"
        ]
        assert len(gate_check_calls) == 3
