"""Tests for spriteforge.retry — retry and escalation engine."""

from __future__ import annotations

import pytest

from spriteforge.gates import GateVerdict
from spriteforge.retry import (
    RetryConfig,
    RetryContext,
    RetryManager,
    RetryTier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verdict(
    gate_name: str = "gate_0",
    passed: bool = False,
    feedback: str = "Something went wrong",
    confidence: float = 0.5,
) -> GateVerdict:
    """Create a :class:`GateVerdict` for testing."""
    return GateVerdict(
        gate_name=gate_name,
        passed=passed,
        confidence=confidence,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# RetryTier enum
# ---------------------------------------------------------------------------


class TestRetryTier:
    """Tests for the RetryTier enum values."""

    def test_soft_value(self) -> None:
        assert RetryTier.SOFT.value == "soft"

    def test_guided_value(self) -> None:
        assert RetryTier.GUIDED.value == "guided"

    def test_constrained_value(self) -> None:
        assert RetryTier.CONSTRAINED.value == "constrained"

    def test_is_str_enum(self) -> None:
        assert isinstance(RetryTier.SOFT, str)


# ---------------------------------------------------------------------------
# get_tier
# ---------------------------------------------------------------------------


class TestGetTier:
    """Tests for RetryManager.get_tier()."""

    def test_get_tier_soft(self) -> None:
        mgr = RetryManager()
        for attempt in (1, 2, 3):
            assert mgr.get_tier(attempt) is RetryTier.SOFT

    def test_get_tier_guided(self) -> None:
        mgr = RetryManager()
        for attempt in (4, 5, 6):
            assert mgr.get_tier(attempt) is RetryTier.GUIDED

    def test_get_tier_constrained(self) -> None:
        mgr = RetryManager()
        for attempt in (7, 8, 9, 10):
            assert mgr.get_tier(attempt) is RetryTier.CONSTRAINED

    def test_get_tier_out_of_range_below(self) -> None:
        mgr = RetryManager()
        with pytest.raises(ValueError):
            mgr.get_tier(0)

    def test_get_tier_clamps_beyond_constrained(self) -> None:
        """Attempts above constrained_range clamp to CONSTRAINED."""
        mgr = RetryManager()
        assert mgr.get_tier(11) is RetryTier.CONSTRAINED
        assert mgr.get_tier(20) is RetryTier.CONSTRAINED


# ---------------------------------------------------------------------------
# get_temperature
# ---------------------------------------------------------------------------


class TestGetTemperature:
    """Tests for RetryManager.get_temperature()."""

    def test_get_temperature_soft(self) -> None:
        mgr = RetryManager()
        for attempt in (1, 2, 3):
            assert mgr.get_temperature(attempt) == 1.0

    def test_get_temperature_guided(self) -> None:
        mgr = RetryManager()
        for attempt in (4, 5, 6):
            assert mgr.get_temperature(attempt) == 0.7

    def test_get_temperature_constrained(self) -> None:
        mgr = RetryManager()
        for attempt in (7, 8, 9, 10):
            assert mgr.get_temperature(attempt) == 0.3

    def test_get_temperature_beyond_constrained_range(self) -> None:
        """Attempts above constrained_range still return constrained temperature."""
        mgr = RetryManager()
        assert mgr.get_temperature(11) == 0.3
        assert mgr.get_temperature(15) == 0.3


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    """Tests for RetryManager.should_retry()."""

    def test_should_retry_first_attempt(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        assert ctx.current_attempt == 0
        assert mgr.should_retry(ctx) is True

    def test_should_retry_ninth_attempt(self) -> None:
        mgr = RetryManager()
        ctx = RetryContext(frame_id="row0_frame0", current_attempt=9, max_attempts=10)
        assert mgr.should_retry(ctx) is True

    def test_should_retry_exhausted(self) -> None:
        mgr = RetryManager()
        ctx = RetryContext(frame_id="row0_frame0", current_attempt=10, max_attempts=10)
        assert mgr.should_retry(ctx) is False

    def test_should_retry_over_max(self) -> None:
        mgr = RetryManager()
        ctx = RetryContext(frame_id="f", current_attempt=15, max_attempts=10)
        assert mgr.should_retry(ctx) is False


# ---------------------------------------------------------------------------
# record_failure
# ---------------------------------------------------------------------------


class TestRecordFailure:
    """Tests for RetryManager.record_failure()."""

    def test_record_failure_increments_attempt(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        assert ctx.current_attempt == 0
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 1

    def test_record_failure_accumulates_feedback(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        v1 = _verdict(gate_name="gate_0", feedback="Bad anatomy")
        v2 = _verdict(gate_name="gate_1", feedback="Wrong color")
        ctx = mgr.record_failure(ctx, [v1, v2])
        assert "Bad anatomy" in ctx.accumulated_feedback
        assert "Wrong color" in ctx.accumulated_feedback
        assert len(ctx.accumulated_feedback) == 2

    def test_record_failure_stores_grid(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        grid = ["." * 64] * 64
        ctx = mgr.record_failure(ctx, [_verdict()], grid=grid)
        assert ctx.last_grid == grid

    def test_record_failure_preserves_grid_when_none(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        grid = ["." * 64] * 64
        ctx = mgr.record_failure(ctx, [_verdict()], grid=grid)
        # Second failure without grid — should keep the previous one.
        ctx = mgr.record_failure(ctx, [_verdict(feedback="another")])
        assert ctx.last_grid == grid

    def test_record_failure_appends_history(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        v1 = _verdict(gate_name="gate_0")
        v2 = _verdict(gate_name="gate_1")
        ctx = mgr.record_failure(ctx, [v1])
        ctx = mgr.record_failure(ctx, [v2])
        assert len(ctx.failure_history) == 2
        assert ctx.failure_history[0].gate_name == "gate_0"
        assert ctx.failure_history[1].gate_name == "gate_1"

    def test_record_failure_skips_empty_feedback(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        ctx = mgr.record_failure(ctx, [_verdict(feedback="")])
        assert ctx.accumulated_feedback == []

    def test_record_failure_last_valid_attempt(self) -> None:
        """current_attempt=8, max_attempts=10 → no crash, correct tier."""
        mgr = RetryManager()
        ctx = RetryContext(frame_id="row0_frame0", current_attempt=8, max_attempts=10)
        # Should not raise — next_attempt=9 is still < 10
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 9

    def test_record_failure_first_attempt(self) -> None:
        """current_attempt=0 → tier is 'soft', no crash."""
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 1

    def test_record_failure_exhausted(self) -> None:
        """current_attempt=9, max_attempts=10 → 'exhausted' path, no crash."""
        mgr = RetryManager()
        ctx = RetryContext(frame_id="row0_frame0", current_attempt=9, max_attempts=10)
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 10

    def test_record_failure_custom_max_attempts(self) -> None:
        """Non-default max_attempts boundary values work correctly."""
        config = RetryConfig(
            max_retries=3,
            soft_range=(1, 1),
            guided_range=(2, 2),
            constrained_range=(3, 3),
        )
        mgr = RetryManager(config=config)
        # current_attempt=0 → next=1 < 3 → get_tier(min(2,3)) works
        ctx = RetryContext(frame_id="f", current_attempt=0, max_attempts=3)
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 1

        # current_attempt=1 → next=2 < 3 → get_tier(min(3,3)) works
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 2

        # current_attempt=2 → next=3, 3 < 3 is False → exhausted
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 3

    def test_record_failure_no_value_error_at_boundary(self) -> None:
        """get_tier/get_temperature must not receive out-of-range values.

        With constrained_range=(4,4) and max_retries=5, current_attempt=3
        makes next_attempt=4 and the old code would call get_tier(5) which
        is outside all tier ranges → ValueError.
        """
        config = RetryConfig(
            max_retries=5,
            soft_range=(1, 2),
            guided_range=(3, 3),
            constrained_range=(4, 4),
        )
        mgr = RetryManager(config=config)
        ctx = RetryContext(frame_id="f", current_attempt=3, max_attempts=5)
        # Must not raise ValueError
        ctx = mgr.record_failure(ctx, [_verdict()])
        assert ctx.current_attempt == 4


# ---------------------------------------------------------------------------
# create_context
# ---------------------------------------------------------------------------


class TestCreateContext:
    """Tests for RetryManager.create_context()."""

    def test_create_context_fresh(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        assert ctx.frame_id == "row0_frame0"
        assert ctx.current_attempt == 0
        assert ctx.failure_history == []
        assert ctx.accumulated_feedback == []
        assert ctx.last_grid is None

    def test_create_context_uses_config_max(self) -> None:
        mgr = RetryManager(config=RetryConfig(max_retries=5))
        ctx = mgr.create_context("row1_frame3")
        assert ctx.max_attempts == 5


# ---------------------------------------------------------------------------
# build_escalated_guidance
# ---------------------------------------------------------------------------


class TestBuildEscalatedGuidance:
    """Tests for RetryManager.build_escalated_guidance()."""

    def test_build_guidance_soft_is_minimal(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        # current_attempt=0, next attempt=1 → SOFT
        guidance = mgr.build_escalated_guidance(ctx)
        assert "64 rows" in guidance
        assert "palette" in guidance.lower()
        # Soft guidance should NOT contain failure details.
        assert "failed" not in guidance.lower()

    def test_build_guidance_guided_includes_feedback(self) -> None:
        mgr = RetryManager()
        ctx = RetryContext(
            frame_id="row0_frame0",
            current_attempt=3,  # next attempt=4 → GUIDED
            max_attempts=10,
            failure_history=[
                _verdict(gate_name="gate_1", feedback="Hair color inconsistent"),
                _verdict(gate_name="gate_0", feedback="Arm position wrong"),
            ],
            accumulated_feedback=["Hair color inconsistent", "Arm position wrong"],
        )
        guidance = mgr.build_escalated_guidance(ctx)
        assert "gate_1" in guidance
        assert "Hair color inconsistent" in guidance
        assert "gate_0" in guidance
        assert "Arm position wrong" in guidance
        assert "specifically address" in guidance.lower()

    def test_build_guidance_constrained_is_prescriptive(self) -> None:
        mgr = RetryManager()
        ctx = RetryContext(
            frame_id="row0_frame0",
            current_attempt=6,  # next attempt=7 → CONSTRAINED
            max_attempts=10,
            failure_history=[
                _verdict(gate_name="gate_0", feedback="Bad proportions"),
            ],
            accumulated_feedback=["Bad proportions", "Missing outline"],
        )
        guidance = mgr.build_escalated_guidance(ctx)
        assert "CRITICAL" in guidance
        assert "6 times" in guidance
        assert "Bad proportions" in guidance
        assert "Missing outline" in guidance
        assert "reference image" in guidance.lower()

    def test_build_guidance_constrained_at_attempt_9(self) -> None:
        mgr = RetryManager()
        ctx = RetryContext(
            frame_id="f",
            current_attempt=9,  # next attempt=10 → CONSTRAINED
            max_attempts=10,
            failure_history=[],
            accumulated_feedback=["fb1"],
        )
        guidance = mgr.build_escalated_guidance(ctx)
        assert "CRITICAL" in guidance
        assert "fb1" in guidance

    def test_build_guidance_beyond_constrained_range(self) -> None:
        """Attempts beyond constrained_range still produce constrained guidance."""
        mgr = RetryManager()
        ctx = RetryContext(
            frame_id="f",
            current_attempt=11,  # next attempt=12 → beyond range
            max_attempts=15,
            failure_history=[],
            accumulated_feedback=["fb"],
        )
        # Must not raise ValueError
        guidance = mgr.build_escalated_guidance(ctx)
        assert "CRITICAL" in guidance


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Tests for custom RetryConfig being respected."""

    def test_custom_config_respected(self) -> None:
        config = RetryConfig(
            max_retries=6,
            soft_range=(1, 2),
            guided_range=(3, 4),
            constrained_range=(5, 6),
            soft_temperature=0.9,
            guided_temperature=0.6,
            constrained_temperature=0.2,
        )
        mgr = RetryManager(config=config)

        assert mgr.get_tier(1) is RetryTier.SOFT
        assert mgr.get_tier(2) is RetryTier.SOFT
        assert mgr.get_tier(3) is RetryTier.GUIDED
        assert mgr.get_tier(4) is RetryTier.GUIDED
        assert mgr.get_tier(5) is RetryTier.CONSTRAINED
        assert mgr.get_tier(6) is RetryTier.CONSTRAINED

        assert mgr.get_temperature(1) == 0.9
        assert mgr.get_temperature(3) == 0.6
        assert mgr.get_temperature(5) == 0.2

        ctx = mgr.create_context("frame")
        assert ctx.max_attempts == 6

    def test_default_config_values(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 10
        assert config.soft_range == (1, 3)
        assert config.guided_range == (4, 6)
        assert config.constrained_range == (7, 10)
        assert config.soft_temperature == 1.0
        assert config.guided_temperature == 0.7
        assert config.constrained_temperature == 0.3


# ---------------------------------------------------------------------------
# Dynamic retry guidance dimensions (Phase 5)
# ---------------------------------------------------------------------------


class TestDynamicRetryGuidance:
    """Tests for parameterized frame dimensions in retry guidance."""

    def test_soft_guidance_default_64x64(self) -> None:
        mgr = RetryManager()
        ctx = mgr.create_context("row0_frame0")
        guidance = mgr.build_escalated_guidance(ctx)
        assert "64 rows of 64 characters" in guidance

    def test_soft_guidance_32x32(self) -> None:
        mgr = RetryManager(frame_width=32, frame_height=32)
        ctx = mgr.create_context("row0_frame0")
        guidance = mgr.build_escalated_guidance(ctx)
        assert "32 rows of 32 characters" in guidance

    def test_soft_guidance_128x128(self) -> None:
        mgr = RetryManager(frame_width=128, frame_height=128)
        ctx = mgr.create_context("row0_frame0")
        guidance = mgr.build_escalated_guidance(ctx)
        assert "128 rows of 128 characters" in guidance

    def test_soft_guidance_no_hardcoded_64(self) -> None:
        mgr = RetryManager(frame_width=32, frame_height=32)
        ctx = mgr.create_context("row0_frame0")
        guidance = mgr.build_escalated_guidance(ctx)
        assert "64" not in guidance

    def test_constrained_guidance_feet_row_scales(self) -> None:
        # 64px → feet_row = int(64 * 0.875) = 56
        mgr64 = RetryManager(frame_width=64, frame_height=64)
        ctx64 = RetryContext(
            frame_id="f",
            current_attempt=6,
            max_attempts=10,
            failure_history=[],
            accumulated_feedback=["fb"],
        )
        guidance64 = mgr64.build_escalated_guidance(ctx64)
        assert "Row 56" in guidance64

        # 32px → feet_row = int(32 * 0.875) = 28
        mgr32 = RetryManager(frame_width=32, frame_height=32)
        ctx32 = RetryContext(
            frame_id="f",
            current_attempt=6,
            max_attempts=10,
            failure_history=[],
            accumulated_feedback=["fb"],
        )
        guidance32 = mgr32.build_escalated_guidance(ctx32)
        assert "Row 28" in guidance32
        assert "56" not in guidance32

    def test_guided_prompt_uses_dimensions(self) -> None:
        mgr = RetryManager(frame_width=48, frame_height=48)
        ctx = RetryContext(
            frame_id="f",
            current_attempt=3,
            max_attempts=10,
            failure_history=[
                _verdict(gate_name="gate_0", feedback="Bad pose"),
            ],
            accumulated_feedback=["Bad pose"],
        )
        guidance = mgr.build_escalated_guidance(ctx)
        assert "48 rows of 48 characters" in guidance

    def test_retry_manager_passes_dimensions(self) -> None:
        mgr = RetryManager(frame_width=48, frame_height=96)
        # Soft tier: check dimensions in soft guidance
        ctx_soft = mgr.create_context("f")
        guidance = mgr.build_escalated_guidance(ctx_soft)
        assert "96 rows of 48 characters" in guidance

    def test_proportional_top_rows_32(self) -> None:
        # 32px: top_rows = max(1, int(32 * 0.075)) = max(1, 2) = 2
        mgr = RetryManager(frame_width=32, frame_height=32)
        ctx = RetryContext(
            frame_id="f",
            current_attempt=6,
            max_attempts=10,
            failure_history=[],
            accumulated_feedback=["fb"],
        )
        guidance = mgr.build_escalated_guidance(ctx)
        assert "Rows 0-2" in guidance

    def test_proportional_top_rows_128(self) -> None:
        # 128px: top_rows = max(1, int(128 * 0.075)) = max(1, 9) = 9
        mgr = RetryManager(frame_width=128, frame_height=128)
        ctx = RetryContext(
            frame_id="f",
            current_attempt=6,
            max_attempts=10,
            failure_history=[],
            accumulated_feedback=["fb"],
        )
        guidance = mgr.build_escalated_guidance(ctx)
        assert "Rows 0-9" in guidance
