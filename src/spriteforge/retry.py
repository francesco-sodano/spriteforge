"""Retry and escalation engine for frame generation.

Manages up to 10 generation attempts per frame with a 3-tier escalation
strategy.  When a frame fails verification gates the retry engine determines
the next attempt's parameters — adjusting temperature, injecting failure
feedback into prompts, and progressively constraining the generation.

The retry engine is *stateless per call* — all mutable state lives in
:class:`RetryContext` objects that the caller passes in and receives back.
It does **not** invoke gates or generators; it only computes parameters for
the next attempt.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from spriteforge.gates import GateVerdict
from spriteforge.logging import get_logger
from spriteforge.prompts.retry import (
    build_constrained_guidance,
    build_guided_guidance,
    build_soft_guidance,
)

logger = get_logger("retry")

# ---------------------------------------------------------------------------
# Retry tier
# ---------------------------------------------------------------------------


class RetryTier(str, Enum):
    """Escalation tier for retry attempts.

    * **SOFT** (attempts 1–3): creative, light guidance.
    * **GUIDED** (attempts 4–6): moderate constraints with specific feedback.
    * **CONSTRAINED** (attempts 7–10): heavy constraints, prescriptive guidance.
    """

    SOFT = "soft"
    GUIDED = "guided"
    CONSTRAINED = "constrained"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Configuration for the retry engine.

    Attributes:
        max_retries: Maximum number of retry attempts per frame.
        soft_range: Inclusive 1-based attempt range for the SOFT tier.
        guided_range: Inclusive 1-based attempt range for the GUIDED tier.
        constrained_range: Inclusive 1-based attempt range for the CONSTRAINED tier.
        soft_temperature: LLM temperature for SOFT tier attempts.
        guided_temperature: LLM temperature for GUIDED tier attempts.
        constrained_temperature: LLM temperature for CONSTRAINED tier attempts.
    """

    max_retries: int = 10
    soft_range: tuple[int, int] = (1, 3)
    guided_range: tuple[int, int] = (4, 6)
    constrained_range: tuple[int, int] = (7, 10)
    soft_temperature: float = 1.0
    guided_temperature: float = 0.7
    constrained_temperature: float = 0.3


# ---------------------------------------------------------------------------
# Retry context
# ---------------------------------------------------------------------------


class RetryContext(BaseModel):
    """Tracks state across retry attempts for a single frame.

    Attributes:
        frame_id: Identifier for the frame (e.g. ``"row0_frame0"``).
        current_attempt: 0-based attempt counter (0 = no attempt yet).
        max_attempts: Hard cap on total attempts.
        failure_history: All :class:`GateVerdict` objects from failed attempts.
        accumulated_feedback: Extracted feedback strings across attempts.
        last_grid: Grid from the most recent failed attempt (may be used by
            constrained-tier guidance).
    """

    frame_id: str
    current_attempt: int = 0
    max_attempts: int = 10
    failure_history: list[GateVerdict] = []
    accumulated_feedback: list[str] = []
    last_grid: list[str] | None = None


# ---------------------------------------------------------------------------
# Retry manager
# ---------------------------------------------------------------------------


class RetryManager:
    """Manages retry attempts with escalating constraints.

    The manager is a pure-logic component: it inspects a :class:`RetryContext`
    and returns parameters (tier, temperature, guidance text) for the next
    generation attempt.  It never calls external services.
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        frame_width: int = 64,
        frame_height: int = 64,
    ) -> None:
        """Initialize with optional custom configuration.

        Args:
            config: Override default tier boundaries and temperatures.
                When *None*, the defaults from :class:`RetryConfig` are used.
            frame_width: Pixel width of each frame (used in guidance text).
            frame_height: Pixel height of each frame (used in guidance text).
        """
        self._config = config or RetryConfig()
        self._frame_width = frame_width
        self._frame_height = frame_height

    # -- tier & temperature -------------------------------------------------

    def get_tier(self, attempt: int) -> RetryTier:
        """Determine which escalation tier an attempt falls in.

        Args:
            attempt: 1-based attempt number.

        Returns:
            The :class:`RetryTier` for this attempt.

        Raises:
            ValueError: If *attempt* does not fall within any configured range.
        """
        cfg = self._config
        if cfg.soft_range[0] <= attempt <= cfg.soft_range[1]:
            return RetryTier.SOFT
        if cfg.guided_range[0] <= attempt <= cfg.guided_range[1]:
            return RetryTier.GUIDED
        if cfg.constrained_range[0] <= attempt <= cfg.constrained_range[1]:
            return RetryTier.CONSTRAINED
        raise ValueError(
            f"Attempt {attempt} does not fall within any configured tier range."
        )

    def get_temperature(self, attempt: int) -> float:
        """Get the LLM temperature for a given attempt number.

        Args:
            attempt: 1-based attempt number.

        Returns:
            Temperature value (1.0 for soft, 0.7 for guided, 0.3 for
            constrained).
        """
        tier = self.get_tier(attempt)
        cfg = self._config
        if tier is RetryTier.SOFT:
            return cfg.soft_temperature
        if tier is RetryTier.GUIDED:
            return cfg.guided_temperature
        return cfg.constrained_temperature

    # -- retry control ------------------------------------------------------

    def should_retry(self, context: RetryContext) -> bool:
        """Determine whether another retry attempt should be made.

        Args:
            context: The current retry context.

        Returns:
            *True* if more attempts remain, *False* if exhausted.
        """
        return context.current_attempt < context.max_attempts

    def record_failure(
        self,
        context: RetryContext,
        verdicts: list[GateVerdict],
        grid: list[str] | None = None,
    ) -> RetryContext:
        """Record a failed attempt and return an updated context.

        The returned :class:`RetryContext` has the attempt counter
        incremented, verdicts appended to the failure history, feedback
        strings extracted, and the optional *grid* stored.

        Args:
            context: The current retry context.
            verdicts: Gate verdicts from the failed attempt.
            grid: The grid that was generated (for constrained-tier
                reference).

        Returns:
            Updated :class:`RetryContext`.
        """
        new_feedback = [v.feedback for v in verdicts if v.feedback]
        next_attempt = context.current_attempt + 1
        if next_attempt < context.max_attempts:
            tier = self.get_tier(next_attempt + 1)
            temperature = self.get_temperature(next_attempt + 1)
            logger.info(
                "Retry %d/%d for %s — %s (tier %s, temp=%.1f)",
                next_attempt,
                context.max_attempts,
                context.frame_id,
                tier.value,
                tier.value,
                temperature,
            )
        else:
            logger.error(
                "All %d retries exhausted for %s",
                context.max_attempts,
                context.frame_id,
            )
        return context.model_copy(
            update={
                "current_attempt": next_attempt,
                "failure_history": context.failure_history + verdicts,
                "accumulated_feedback": context.accumulated_feedback + new_feedback,
                "last_grid": grid if grid is not None else context.last_grid,
            }
        )

    def create_context(self, frame_id: str) -> RetryContext:
        """Create a fresh retry context for a new frame.

        Args:
            frame_id: Identifier for the frame (e.g. ``"row0_frame0"``).

        Returns:
            A new :class:`RetryContext` with attempt count 0 and empty
            history.
        """
        return RetryContext(
            frame_id=frame_id,
            max_attempts=self._config.max_retries,
        )

    # -- guidance builder ---------------------------------------------------

    def build_escalated_guidance(self, context: RetryContext) -> str:
        """Build additional prompt guidance based on failure history.

        * **SOFT** tier: minimal — basic pixel-art rules only.
        * **GUIDED** tier: specific — includes gate feedback from failures.
        * **CONSTRAINED** tier: prescriptive — accumulated history with
          pixel-level correction instructions.

        Args:
            context: The current retry context with failure history.

        Returns:
            Additional guidance text to inject into the generation prompt.
        """
        attempt = context.current_attempt + 1  # next attempt (1-based)
        tier = self.get_tier(attempt)

        if tier is RetryTier.SOFT:
            return self._build_soft_guidance()
        if tier is RetryTier.GUIDED:
            return self._build_guided_guidance(context)
        return self._build_constrained_guidance(context)

    # -- private helpers ----------------------------------------------------

    def _build_soft_guidance(self) -> str:
        """Minimal guidance for the SOFT tier."""
        return build_soft_guidance(
            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )

    def _build_guided_guidance(self, context: RetryContext) -> str:
        """Guidance for the GUIDED tier — includes specific gate feedback."""
        return build_guided_guidance(
            context.failure_history,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )

    def _build_constrained_guidance(self, context: RetryContext) -> str:
        """Prescriptive guidance for the CONSTRAINED tier."""
        return build_constrained_guidance(
            context.current_attempt,
            context.accumulated_feedback,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )
