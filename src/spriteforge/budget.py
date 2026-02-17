"""Budget tracking and cost estimation for LLM calls.

Provides thread-safe call tracking and dry-run estimation of min/expected/max
LLM calls based on spritesheet configuration.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from spriteforge.errors import BudgetExhaustedError
from spriteforge.logging import get_logger
from spriteforge.models import BudgetConfig, SpritesheetSpec

logger = get_logger("budget")


# ---------------------------------------------------------------------------
# Call tracker
# ---------------------------------------------------------------------------


class CallTracker:
    """Thread-safe counter for tracking LLM calls against a budget.

    Tracks total calls across all providers (chat and reference generation).
    Raises BudgetExhaustedError when budget is exceeded. Logs warnings at
    configurable threshold.

    Optionally tracks prompt and completion token counts when
    ``budget.track_tokens`` is enabled.
    """

    def __init__(self, budget: BudgetConfig | None = None) -> None:
        """Initialize the tracker with optional budget constraints.

        Args:
            budget: Budget configuration. When None, no limits are enforced.
        """
        self._budget = budget
        self._count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._lock = threading.Lock()
        self._warned = False

    @property
    def count(self) -> int:
        """Get the current call count (thread-safe)."""
        with self._lock:
            return self._count

    @property
    def token_usage(self) -> dict[str, int]:
        """Get accumulated token counts (thread-safe)."""
        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            }

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from an LLM call.

        Only records when ``budget.track_tokens`` is enabled.

        Args:
            prompt_tokens: Number of prompt/input tokens.
            completion_tokens: Number of completion/output tokens.
        """
        if self._budget is None or not self._budget.track_tokens:
            return
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens

    def increment(self, call_type: str = "LLM") -> None:
        """Increment the call counter and check budget.

        Args:
            call_type: Human-readable description of the call (for logging).

        Raises:
            BudgetExhaustedError: If budget limit is exceeded.
        """
        with self._lock:
            self._count += 1
            current = self._count

        if self._budget is None or self._budget.max_llm_calls == 0:
            # No budget enforced
            return

        max_calls = self._budget.max_llm_calls
        warn_threshold = int(max_calls * self._budget.warn_at_percentage)

        # Check for warning threshold
        if current >= warn_threshold and not self._warned:
            logger.warning(
                "Budget warning: %d/%d calls used (%.0f%%)",
                current,
                max_calls,
                (current / max_calls) * 100,
            )
            self._warned = True

        # Check for hard limit
        if current > max_calls:
            raise BudgetExhaustedError(
                f"LLM call budget exhausted: {current} calls made, "
                f"limit is {max_calls}. Increase budget.max_llm_calls "
                f"or reduce spritesheet complexity."
            )

    def reset(self) -> None:
        """Reset the counter to zero (for testing)."""
        with self._lock:
            self._count = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._warned = False


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


@dataclass
class CallEstimate:
    """Estimated LLM call counts for a spritesheet generation.

    Attributes:
        min_calls: Minimum calls (all frames pass on first attempt).
        expected_calls: Expected calls (some retries, typical scenario).
        max_calls: Maximum calls (worst case with all retries).
        breakdown: Human-readable breakdown of call types.
    """

    min_calls: int
    expected_calls: int
    max_calls: int
    breakdown: dict[str, dict[str, int]]


def estimate_calls(config: SpritesheetSpec) -> CallEstimate:
    """Estimate min/expected/max LLM calls for a spritesheet generation.

    Args:
        config: The spritesheet specification.

    Returns:
        CallEstimate with min/expected/max call counts and breakdown.

    Notes:
        Assumptions for expected case:
        - 30% of reference strips fail Gate -1 once (retry)
        - 20% of frames fail gates once (1 retry per failed frame)
        - 5% of rows fail Gate 3A once (row regeneration)

        Per-frame call breakdown:
        - 1 grid generation call (Stage 2)
        - 2-4 gate calls (Gate 0 always; Gate 1 if non-anchor; Gate 2 if non-first-frame)

        Per-row call breakdown:
        - 1 reference strip generation (Stage 1)
        - 1 Gate -1 check (reference validation)
        - 1 Gate 3A check (row coherence)
    """
    # Get retry config limits
    gen_config = config.generation
    budget = gen_config.budget
    max_retries_per_frame = 10  # Default from RetryConfig
    if budget and budget.max_retries_per_row > 0:
        max_retries_per_frame = budget.max_retries_per_row

    # Count animations and total frames
    animations = config.animations
    total_rows = len(animations)
    total_frames = sum(anim.frames for anim in animations)

    # --- MIN CASE: All frames pass on first attempt ---
    # Per row: 1 ref gen + 1 gate_-1 + N frames + 1 gate_3a
    # Per frame: 1 grid gen + 2-4 gates (0, maybe 1, maybe 2)

    min_ref_calls = total_rows  # 1 per row
    min_gate_minus_1 = total_rows  # 1 per row
    min_gate_3a = total_rows  # 1 per row

    min_grid_gen = total_frames  # 1 per frame

    # Gate counts per frame:
    # - Gate 0: always runs (total_frames)
    # - Gate 1: runs for all non-anchor frames
    # - Gate 2: runs for all frames except first frame of each row
    gate_0_count = total_frames
    gate_1_count = total_frames - total_rows  # Skip anchor frame (F0) of each row
    gate_2_count = total_frames - total_rows  # Skip first frame of each row

    min_gates_per_frame = gate_0_count + gate_1_count + gate_2_count

    min_calls = (
        min_ref_calls
        + min_gate_minus_1
        + min_grid_gen
        + min_gates_per_frame
        + min_gate_3a
    )

    # --- EXPECTED CASE: Some retries ---
    # Assume 30% of ref strips retry once (2x ref gen, 2x gate_-1)
    # Assume 20% of frames retry once (2x grid gen, 2x gates)
    # Assume 5% of rows fail Gate 3A (causes row regeneration)

    ref_retry_rate = 0.30
    frame_retry_rate = 0.20
    row_retry_rate = 0.05

    expected_ref_calls = int(total_rows * (1 + ref_retry_rate * 1))
    expected_gate_minus_1 = int(total_rows * (1 + ref_retry_rate * 1))

    expected_grid_gen = int(total_frames * (1 + frame_retry_rate * 1))
    expected_gates_per_frame = int(
        (gate_0_count + gate_1_count + gate_2_count) * (1 + frame_retry_rate * 1)
    )

    # Row failures cause entire row regeneration (all frames + gates + gate_3a)
    expected_row_retries = int(total_rows * row_retry_rate)
    avg_frames_per_row = total_frames // total_rows if total_rows > 0 else 0
    expected_row_retry_calls = expected_row_retries * (
        1 + avg_frames_per_row * (1 + 3)  # grid gen + ~3 gates per frame
    )

    expected_gate_3a = total_rows + expected_row_retries

    expected_calls = (
        expected_ref_calls
        + expected_gate_minus_1
        + expected_grid_gen
        + expected_gates_per_frame
        + expected_gate_3a
        + expected_row_retry_calls
    )

    # --- MAX CASE: All frames hit max retries ---
    # Each reference strip retries max 3 times (Stage 1 hardcoded)
    max_ref_retries = 3
    max_ref_calls = total_rows * max_ref_retries
    max_gate_minus_1 = total_rows * max_ref_retries

    # Each frame retries up to max_retries_per_frame
    max_grid_gen = total_frames * max_retries_per_frame
    max_gates_per_frame = (
        gate_0_count + gate_1_count + gate_2_count
    ) * max_retries_per_frame

    # Assume Gate 3A fails for every row and triggers full row regeneration
    # (which itself can retry, multiplicative effect)
    max_gate_3a = total_rows * 2  # Initial + 1 retry per row

    max_calls = (
        max_ref_calls
        + max_gate_minus_1
        + max_grid_gen
        + max_gates_per_frame
        + max_gate_3a
    )

    # Build breakdown
    breakdown = {
        "min": {
            "reference_generation": min_ref_calls,
            "gate_minus_1": min_gate_minus_1,
            "grid_generation": min_grid_gen,
            "gate_0": gate_0_count,
            "gate_1": gate_1_count,
            "gate_2": gate_2_count,
            "gate_3a": min_gate_3a,
        },
        "expected": {
            "reference_generation": expected_ref_calls,
            "gate_minus_1": expected_gate_minus_1,
            "grid_generation": expected_grid_gen,
            "gates_per_frame": expected_gates_per_frame,
            "gate_3a": expected_gate_3a,
        },
        "max": {
            "reference_generation": max_ref_calls,
            "gate_minus_1": max_gate_minus_1,
            "grid_generation": max_grid_gen,
            "gates_per_frame": max_gates_per_frame,
            "gate_3a": max_gate_3a,
        },
    }

    return CallEstimate(
        min_calls=min_calls,
        expected_calls=expected_calls,
        max_calls=max_calls,
        breakdown=breakdown,
    )
