"""Tests for spriteforge.budget — call tracking and cost estimation."""

from __future__ import annotations

import threading

import pytest

from spriteforge.budget import CallEstimate, CallTracker, estimate_calls
from spriteforge.errors import BudgetExhaustedError
from spriteforge.models import (
    AnimationDef,
    BudgetConfig,
    CharacterConfig,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)

# ---------------------------------------------------------------------------
# BudgetConfig model tests
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    """Tests for the BudgetConfig model."""

    def test_valid_budget(self) -> None:
        budget = BudgetConfig(
            max_llm_calls=500, max_retries_per_row=30, warn_at_percentage=0.8
        )
        assert budget.max_llm_calls == 500
        assert budget.max_retries_per_row == 30
        assert budget.warn_at_percentage == 0.8

    def test_default_values(self) -> None:
        budget = BudgetConfig()
        assert budget.max_llm_calls == 0  # No limit
        assert budget.max_retries_per_row == 0  # Use RetryConfig default
        assert budget.warn_at_percentage == 0.8

    def test_negative_max_calls_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(max_llm_calls=-1)

    def test_negative_max_retries_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(max_retries_per_row=-1)

    def test_warn_percentage_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(warn_at_percentage=1.5)
        with pytest.raises(ValueError):
            BudgetConfig(warn_at_percentage=-0.1)

    def test_warn_percentage_edge_cases(self) -> None:
        # 0.0 and 1.0 should be valid
        budget1 = BudgetConfig(warn_at_percentage=0.0)
        assert budget1.warn_at_percentage == 0.0
        budget2 = BudgetConfig(warn_at_percentage=1.0)
        assert budget2.warn_at_percentage == 1.0


# ---------------------------------------------------------------------------
# CallTracker tests
# ---------------------------------------------------------------------------


class TestCallTracker:
    """Tests for the CallTracker class."""

    def test_tracker_without_budget(self) -> None:
        """Tracker without budget should not enforce limits."""
        tracker = CallTracker()
        for _ in range(1000):
            tracker.increment()
        assert tracker.count == 1000

    def test_tracker_with_budget_within_limit(self) -> None:
        """Tracker should allow calls within budget."""
        budget = BudgetConfig(max_llm_calls=10)
        tracker = CallTracker(budget)
        for _ in range(10):
            tracker.increment()
        assert tracker.count == 10

    def test_tracker_with_budget_exceeds_limit(self) -> None:
        """Tracker should raise BudgetExhaustedError when limit exceeded."""
        budget = BudgetConfig(max_llm_calls=5)
        tracker = CallTracker(budget)
        for _ in range(5):
            tracker.increment()

        with pytest.raises(BudgetExhaustedError, match="call budget exhausted"):
            tracker.increment()

    def test_tracker_warning_at_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tracker should log warning at configured threshold."""
        budget = BudgetConfig(max_llm_calls=10, warn_at_percentage=0.8)
        tracker = CallTracker(budget)

        # Should not warn before threshold
        for _ in range(7):
            tracker.increment()
        assert "Budget warning" not in caplog.text

        # Should warn at 80% (8/10)
        tracker.increment()
        assert "Budget warning" in caplog.text
        assert "8/10" in caplog.text

    def test_tracker_warning_only_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Tracker should only log warning once."""
        budget = BudgetConfig(max_llm_calls=10, warn_at_percentage=0.8)
        tracker = CallTracker(budget)

        for _ in range(10):
            tracker.increment()

        # Count occurrences of warning message
        warning_count = caplog.text.count("Budget warning")
        assert warning_count == 1

    def test_tracker_reset(self) -> None:
        """Tracker reset should clear count and warning flag."""
        budget = BudgetConfig(max_llm_calls=10, warn_at_percentage=0.8)
        tracker = CallTracker(budget)

        for _ in range(9):
            tracker.increment()

        tracker.reset()
        assert tracker.count == 0

        # Should be able to increment again without error
        for _ in range(10):
            tracker.increment()
        assert tracker.count == 10

    def test_tracker_thread_safety(self) -> None:
        """Tracker should be thread-safe."""
        budget = BudgetConfig(max_llm_calls=1000)
        tracker = CallTracker(budget)

        def worker() -> None:
            for _ in range(100):
                tracker.increment()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.count == 1000

    def test_tracker_call_type_param(self) -> None:
        """Tracker should accept call_type parameter (for logging)."""
        budget = BudgetConfig(max_llm_calls=2)
        tracker = CallTracker(budget)

        tracker.increment("grid_generation")
        tracker.increment("gate_check")
        assert tracker.count == 2

    def test_tracker_zero_budget_means_no_limit(self) -> None:
        """Budget with max_llm_calls=0 should not enforce limits."""
        budget = BudgetConfig(max_llm_calls=0)
        tracker = CallTracker(budget)
        for _ in range(1000):
            tracker.increment()
        assert tracker.count == 1000

    def test_tracker_best_effort_mode_does_not_raise(self) -> None:
        budget = BudgetConfig(max_llm_calls=2, enforcement_mode="best_effort")
        tracker = CallTracker(budget)
        tracker.increment()
        tracker.increment()
        tracker.increment()
        assert tracker.count == 3


# ---------------------------------------------------------------------------
# Token tracking tests
# ---------------------------------------------------------------------------


class TestCallTrackerTokenTracking:
    """Tests for token tracking in CallTracker."""

    def test_record_tokens_when_enabled(self) -> None:
        """Token counts accumulate when track_tokens is True."""
        budget = BudgetConfig(track_tokens=True)
        tracker = CallTracker(budget)
        tracker.record_tokens(100, 50)
        tracker.record_tokens(200, 80)
        usage = tracker.token_usage
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 130
        assert usage["total_tokens"] == 430

    def test_record_tokens_ignored_when_disabled(self) -> None:
        """Token counts are not recorded when track_tokens is False."""
        budget = BudgetConfig(track_tokens=False)
        tracker = CallTracker(budget)
        tracker.record_tokens(100, 50)
        usage = tracker.token_usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0

    def test_record_tokens_ignored_without_budget(self) -> None:
        """Token counts are not recorded when no budget is set."""
        tracker = CallTracker()
        tracker.record_tokens(100, 50)
        assert tracker.token_usage["total_tokens"] == 0

    def test_reset_clears_tokens(self) -> None:
        """Reset clears accumulated token counts."""
        budget = BudgetConfig(track_tokens=True)
        tracker = CallTracker(budget)
        tracker.record_tokens(100, 50)
        tracker.reset()
        assert tracker.token_usage["total_tokens"] == 0

    def test_token_tracking_thread_safety(self) -> None:
        """Token recording is thread-safe."""
        budget = BudgetConfig(track_tokens=True)
        tracker = CallTracker(budget)

        def worker() -> None:
            for _ in range(100):
                tracker.record_tokens(10, 5)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        usage = tracker.token_usage
        assert usage["prompt_tokens"] == 10000
        assert usage["completion_tokens"] == 5000


# ---------------------------------------------------------------------------
# estimate_calls tests
# ---------------------------------------------------------------------------


class TestEstimateCalls:
    """Tests for the estimate_calls function."""

    def test_estimate_simple_single_animation(self) -> None:
        """Estimate for a simple single-row animation."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
            ],
        )

        estimate = estimate_calls(config)

        # Min case: 1 row × (1 ref + 1 gate_-1 + 4 frames + gates + 1 gate_3a)
        # Per frame: 1 grid gen + gates (0, 1 for non-anchor, 2 for non-first)
        # Frame 0 (anchor): gate_0 only = 1 gate
        # Frame 1-3: gate_0 + gate_1 + gate_2 = 3 gates each
        # Total gates: 1 + 3 + 3 + 3 = 10
        assert estimate.min_calls > 0
        assert estimate.expected_calls >= estimate.min_calls
        assert estimate.max_calls >= estimate.expected_calls

        # Verify breakdown structure
        assert "min" in estimate.breakdown
        assert "expected" in estimate.breakdown
        assert "max" in estimate.breakdown

    def test_estimate_uses_budget_retry_rates(self) -> None:
        config = SpritesheetSpec(
            character=CharacterConfig(name="hero"),
            animations=[AnimationDef(name="idle", row=0, frames=4, timing_ms=100)],
            generation=GenerationConfig(
                budget=BudgetConfig(
                    expected_reference_retry_rate=0.0,
                    expected_frame_retry_rate=0.0,
                    expected_row_retry_rate=0.0,
                )
            ),
            palette=PaletteConfig(
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[PaletteColor(element="Skin", symbol="s", r=1, g=1, b=1)],
            ),
        )
        est = estimate_calls(config)
        # With 0 retry rates, expected calls should be close to min calls.
        assert est.expected_calls >= est.min_calls
        assert est.expected_calls - est.min_calls <= 2

    def test_estimate_multi_row_animation(self) -> None:
        """Estimate for multi-row spritesheet."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=6, timing_ms=150),
                AnimationDef(name="walk", row=1, frames=8, timing_ms=100),
                AnimationDef(name="attack", row=2, frames=5, timing_ms=80),
            ],
        )

        estimate = estimate_calls(config)

        # Should have more calls than single row
        assert estimate.min_calls > 0
        assert estimate.expected_calls >= estimate.min_calls
        assert estimate.max_calls >= estimate.expected_calls

        # Max should be significantly higher than min (due to retries)
        assert estimate.max_calls > estimate.min_calls * 5

    def test_estimate_with_budget_override(self) -> None:
        """Estimate respects budget.max_retries_per_row."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(
                budget=BudgetConfig(max_retries_per_row=5)  # Override default 10
            ),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
            ],
        )

        estimate = estimate_calls(config)

        # Max should use 5 retries per frame, not 10
        # With 4 frames: max_grid_gen = 4 * 5 = 20
        assert estimate.breakdown["max"]["grid_generation"] == 20

    def test_estimate_breakdown_fields(self) -> None:
        """Estimate breakdown should contain all expected fields."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
            ],
        )

        estimate = estimate_calls(config)

        # Check min breakdown
        assert "reference_generation" in estimate.breakdown["min"]
        assert "gate_minus_1" in estimate.breakdown["min"]
        assert "grid_generation" in estimate.breakdown["min"]
        assert "gate_0" in estimate.breakdown["min"]
        assert "gate_1" in estimate.breakdown["min"]
        assert "gate_2" in estimate.breakdown["min"]
        assert "gate_3a" in estimate.breakdown["min"]

        # Check expected breakdown
        assert "reference_generation" in estimate.breakdown["expected"]
        assert "gate_minus_1" in estimate.breakdown["expected"]
        assert "grid_generation" in estimate.breakdown["expected"]

        # Check max breakdown
        assert "reference_generation" in estimate.breakdown["max"]
        assert "gate_minus_1" in estimate.breakdown["max"]
        assert "grid_generation" in estimate.breakdown["max"]

    def test_estimate_empty_animations(self) -> None:
        """Estimate should handle empty animations gracefully."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=[],
        )

        estimate = estimate_calls(config)

        assert estimate.min_calls == 0
        assert estimate.expected_calls == 0
        assert estimate.max_calls == 0

    def test_estimate_large_spritesheet(self) -> None:
        """Estimate for a large spritesheet (like 16-row Theron)."""
        # Create 16 animations with 6 frames each
        animations = [
            AnimationDef(name=f"anim{i}", row=i, frames=6, timing_ms=100)
            for i in range(16)
        ]

        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=animations,
        )

        estimate = estimate_calls(config)

        # 16 rows × 6 frames = 96 frames total
        # Should be within reasonable bounds
        assert estimate.min_calls > 0
        assert estimate.min_calls < 1000  # Minimum should be reasonable
        assert estimate.max_calls > estimate.min_calls
        # With 10 retries per frame: could be very large
        assert estimate.max_calls < 10000  # But not absurd

    def test_estimate_returns_call_estimate_type(self) -> None:
        """estimate_calls should return CallEstimate dataclass."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="hero",
                frame_width=64,
                frame_height=64,
                description="Test character",
            ),
            generation=GenerationConfig(),
            palette=PaletteConfig(
                name="P1",
                transparent_symbol=".",
                outline=PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0),
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=255, g=200, b=150),
                ],
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
            ],
        )

        estimate = estimate_calls(config)

        assert isinstance(estimate, CallEstimate)
        assert hasattr(estimate, "min_calls")
        assert hasattr(estimate, "expected_calls")
        assert hasattr(estimate, "max_calls")
        assert hasattr(estimate, "breakdown")
