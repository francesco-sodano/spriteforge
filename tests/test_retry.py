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

    def test_get_tier_out_of_range(self) -> None:
        mgr = RetryManager()
        with pytest.raises(ValueError):
            mgr.get_tier(0)
        with pytest.raises(ValueError):
            mgr.get_tier(11)


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
