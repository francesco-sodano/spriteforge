"""Prompt templates for retry escalation guidance.

Contains the three guidance builders (soft, guided, constrained) — all
originally defined as private static methods in
:pymod:`spriteforge.retry.RetryManager`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spriteforge.gates import GateVerdict


def build_soft_guidance() -> str:
    """Build minimal guidance for the SOFT tier (attempts 1-3).

    Returns:
        A short reminder of fundamental pixel-art grid rules.
    """
    return (
        "Ensure the grid is exactly 64 rows of 64 characters. "
        "Use only the provided palette symbols. "
        "Place feet near row 56. Include a 1-pixel dark outline."
    )


def build_guided_guidance(
    failure_history: list[GateVerdict],
) -> str:
    """Build guidance for the GUIDED tier (attempts 4-6).

    Includes specific gate feedback from previous failures.

    Args:
        failure_history: All gate verdicts from previous failed attempts.

    Returns:
        Guidance text referencing specific failures to address.
    """
    lines: list[str] = ["Previous attempts failed for these reasons:"]
    for verdict in failure_history:
        if not verdict.passed and verdict.feedback:
            lines.append(f"- {verdict.gate_name}: '{verdict.feedback}'")
    lines.append("Please specifically address these issues in your next attempt.")
    return "\n".join(lines)


def build_constrained_guidance(
    current_attempt: int,
    accumulated_feedback: list[str],
) -> str:
    """Build prescriptive guidance for the CONSTRAINED tier (attempts 7-10).

    Args:
        current_attempt: 0-based attempt count so far.
        accumulated_feedback: All feedback strings from previous attempts.

    Returns:
        Highly prescriptive guidance with exact pixel-level constraints.
    """
    lines: list[str] = [
        f"CRITICAL: You have failed {current_attempt} times. "
        "Follow these exact constraints:",
        "- Rows 0-4: must be all '.' (transparent above head)",
        "- Rows 5-15: hair region, use 'h' symbol",
        "- Row 56 area: feet must be present (non-transparent)",
        "Previous specific failures:",
    ]
    for fb in accumulated_feedback:
        lines.append(f"- {fb}")
    lines.append("Output must exactly match the reference image.")
    return "\n".join(lines)
