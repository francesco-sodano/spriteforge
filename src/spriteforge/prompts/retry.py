"""Prompt templates for retry escalation guidance.

Contains the three guidance builders (soft, guided, constrained) — all
originally defined as private static methods in
:pymod:`spriteforge.retry.RetryManager`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spriteforge.gates import GateVerdict


def build_soft_guidance(
    frame_width: int = 64,
    frame_height: int = 64,
) -> str:
    """Build minimal guidance for the SOFT tier (attempts 1-3).

    Args:
        frame_width: Pixel width of each frame.
        frame_height: Pixel height of each frame.

    Returns:
        A short reminder of fundamental pixel-art grid rules.
    """
    feet_row = int(frame_height * 0.875)
    return (
        f"Ensure the grid is exactly {frame_height} rows of {frame_width} characters. "
        "Use only the provided palette symbols. "
        f"Place feet near row {feet_row}. Include a 1-pixel dark outline."
    )


def build_guided_guidance(
    failure_history: list[GateVerdict],
    frame_width: int = 64,
    frame_height: int = 64,
) -> str:
    """Build guidance for the GUIDED tier (attempts 4-6).

    Includes specific gate feedback from previous failures.

    Args:
        failure_history: All gate verdicts from previous failed attempts.
        frame_width: Pixel width of each frame.
        frame_height: Pixel height of each frame.

    Returns:
        Guidance text referencing specific failures to address.
    """
    lines: list[str] = [
        "Previous attempts failed for these reasons:",
        f"(Grid must be exactly {frame_height} rows of {frame_width} characters.)",
    ]
    for verdict in failure_history:
        if not verdict.passed and verdict.feedback:
            lines.append(f"- {verdict.gate_name}: '{verdict.feedback}'")
    lines.append("Please specifically address these issues in your next attempt.")
    return "\n".join(lines)


def build_constrained_guidance(
    current_attempt: int,
    accumulated_feedback: list[str],
    frame_width: int = 64,
    frame_height: int = 64,
) -> str:
    """Build prescriptive guidance for the CONSTRAINED tier (attempts 7-10).

    Args:
        current_attempt: 0-based attempt count so far.
        accumulated_feedback: All feedback strings from previous attempts.
        frame_width: Pixel width of each frame.
        frame_height: Pixel height of each frame.

    Returns:
        Highly prescriptive guidance with exact pixel-level constraints.
    """
    top_rows = max(1, int(frame_height * 0.075))
    hair_end = max(top_rows + 1, int(frame_height * 0.234))
    feet_row = int(frame_height * 0.875)
    lines: list[str] = [
        f"CRITICAL: You have failed {current_attempt} times. "
        "Follow these exact constraints:",
        f"- Rows 0-{top_rows}: must be all '.' (transparent above head)",
        f"- Rows {top_rows + 1}-{hair_end}: hair region, use 'h' symbol",
        f"- Row {feet_row} area: feet must be present (non-transparent)",
        "Previous specific failures:",
    ]
    for fb in accumulated_feedback:
        lines.append(f"- {fb}")
    lines.append("Output must exactly match the reference image.")
    return "\n".join(lines)
