"""Multi-gate verification system for generated frames.

Combines fast programmatic checks (grid dimensions, valid symbols, structural
rules) with LLM-based vision checks (reference fidelity, identity consistency,
temporal smoothness, row coherence) to validate each generated frame.

Gate verdicts carry structured feedback that feeds into the retry engine.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from spriteforge.models import AnimationDef, PaletteConfig
from spriteforge.providers.chat import ChatProvider
from spriteforge.utils import image_to_data_url

# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------


class GateVerdict(BaseModel):
    """Structured result from a verification gate.

    Attributes:
        gate_name: Identifier for the gate (e.g. ``"gate_0"``,
            ``"programmatic_dimensions"``).
        passed: Whether the check passed.
        confidence: Confidence score between 0.0 and 1.0.
        feedback: Human-readable explanation (injected into retry prompts).
        details: Optional structured details about the check.
    """

    gate_name: str
    passed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    details: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Programmatic checker
# ---------------------------------------------------------------------------


class ProgrammaticChecker:
    """Fast, deterministic structural checks on raw grids."""

    def check_dimensions(
        self,
        grid: list[str],
        expected_rows: int = 64,
        expected_cols: int = 64,
    ) -> GateVerdict:
        """Verify grid has exactly *expected_rows* rows, each *expected_cols* chars.

        Args:
            grid: The palette-indexed grid to check.
            expected_rows: Expected number of rows.
            expected_cols: Expected number of columns per row.

        Returns:
            A ``GateVerdict`` indicating pass/fail with feedback.
        """
        if len(grid) != expected_rows:
            return GateVerdict(
                gate_name="programmatic_dimensions",
                passed=False,
                confidence=1.0,
                feedback=(f"Grid has {len(grid)} rows, expected {expected_rows}."),
                details={
                    "actual_rows": len(grid),
                    "expected_rows": expected_rows,
                },
            )

        for i, row in enumerate(grid):
            if len(row) != expected_cols:
                return GateVerdict(
                    gate_name="programmatic_dimensions",
                    passed=False,
                    confidence=1.0,
                    feedback=(
                        f"Row {i} has {len(row)} characters, "
                        f"expected {expected_cols}."
                    ),
                    details={
                        "row_index": i,
                        "actual_cols": len(row),
                        "expected_cols": expected_cols,
                    },
                )

        return GateVerdict(
            gate_name="programmatic_dimensions",
            passed=True,
            confidence=1.0,
            feedback="Grid dimensions are correct.",
        )

    def check_valid_symbols(
        self,
        grid: list[str],
        palette: PaletteConfig,
    ) -> GateVerdict:
        """Verify all characters in *grid* are valid palette symbols.

        Args:
            grid: The palette-indexed grid to check.
            palette: Palette config defining valid symbols.

        Returns:
            A ``GateVerdict`` indicating pass/fail with feedback.
        """
        valid_symbols: set[str] = {palette.transparent_symbol, palette.outline.symbol}
        for color in palette.colors:
            valid_symbols.add(color.symbol)

        invalid: list[str] = []
        seen: set[str] = set()
        for row in grid:
            for ch in row:
                if ch not in valid_symbols and ch not in seen:
                    invalid.append(ch)
                    seen.add(ch)

        if invalid:
            return GateVerdict(
                gate_name="programmatic_symbols",
                passed=False,
                confidence=1.0,
                feedback=(
                    f"Grid contains invalid palette symbols: "
                    f"{', '.join(repr(s) for s in invalid)}. "
                    f"Only these symbols are allowed: "
                    f"{', '.join(sorted(valid_symbols))}."
                ),
                details={"invalid_symbols": invalid},
            )

        return GateVerdict(
            gate_name="programmatic_symbols",
            passed=True,
            confidence=1.0,
            feedback="All grid symbols are valid palette entries.",
        )

    def check_outline_presence(
        self,
        grid: list[str],
        outline_symbol: str = "O",
    ) -> GateVerdict:
        """Verify the grid contains outline pixels.

        Args:
            grid: The palette-indexed grid to check.
            outline_symbol: The outline palette symbol.

        Returns:
            A ``GateVerdict`` indicating pass/fail with feedback.
        """
        for row in grid:
            if outline_symbol in row:
                return GateVerdict(
                    gate_name="programmatic_outline",
                    passed=True,
                    confidence=1.0,
                    feedback="Outline pixels are present.",
                )

        return GateVerdict(
            gate_name="programmatic_outline",
            passed=False,
            confidence=1.0,
            feedback=(
                f"No outline pixels ('{outline_symbol}') found in grid. "
                f"Every non-transparent sprite pixel must have a 1-pixel "
                f"dark outline."
            ),
        )

    def check_not_empty(
        self,
        grid: list[str],
        transparent_symbol: str = ".",
    ) -> GateVerdict:
        """Verify the grid isn't entirely transparent.

        Args:
            grid: The palette-indexed grid to check.
            transparent_symbol: The transparent palette symbol.

        Returns:
            A ``GateVerdict`` indicating pass/fail with feedback.
        """
        for row in grid:
            for ch in row:
                if ch != transparent_symbol:
                    return GateVerdict(
                        gate_name="programmatic_not_empty",
                        passed=True,
                        confidence=1.0,
                        feedback="Grid contains non-transparent content.",
                    )

        return GateVerdict(
            gate_name="programmatic_not_empty",
            passed=False,
            confidence=1.0,
            feedback=(
                "Grid is entirely transparent — no sprite content found. "
                "The grid must contain non-transparent pixels forming the "
                "character sprite."
            ),
        )

    def check_feet_position(
        self,
        grid: list[str],
        transparent_symbol: str = ".",
        expected_foot_row: int = 55,
    ) -> GateVerdict:
        """Verify non-transparent pixels exist near the expected foot row.

        Heuristic: at least some non-transparent pixels in rows 50–58.

        Args:
            grid: The palette-indexed grid to check.
            transparent_symbol: The transparent palette symbol.
            expected_foot_row: The expected y-position for feet (0-indexed).

        Returns:
            A ``GateVerdict`` indicating pass/fail with feedback.
        """
        foot_zone_start = max(0, expected_foot_row - 5)
        foot_zone_end = min(len(grid), expected_foot_row + 4)

        for row_idx in range(foot_zone_start, foot_zone_end):
            if row_idx < len(grid):
                for ch in grid[row_idx]:
                    if ch != transparent_symbol:
                        return GateVerdict(
                            gate_name="programmatic_feet_position",
                            passed=True,
                            confidence=0.8,
                            feedback=(
                                f"Non-transparent pixels found near "
                                f"expected foot row ({expected_foot_row})."
                            ),
                        )

        return GateVerdict(
            gate_name="programmatic_feet_position",
            passed=False,
            confidence=0.8,
            feedback=(
                f"No non-transparent pixels found near expected foot row "
                f"({expected_foot_row}). The character's feet should be "
                f"positioned around rows {foot_zone_start}–{foot_zone_end - 1}."
            ),
        )

    def run_all(
        self,
        grid: list[str],
        palette: PaletteConfig,
    ) -> list[GateVerdict]:
        """Run all programmatic checks and return all verdicts.

        Args:
            grid: The palette-indexed grid to check.
            palette: Palette config defining valid symbols.

        Returns:
            A list of ``GateVerdict`` objects, one per check.
        """
        return [
            self.check_dimensions(grid),
            self.check_valid_symbols(grid, palette),
            self.check_outline_presence(grid, palette.outline.symbol),
            self.check_not_empty(grid, palette.transparent_symbol),
            self.check_feet_position(grid, palette.transparent_symbol),
        ]


# ---------------------------------------------------------------------------
# Gate prompt templates
# ---------------------------------------------------------------------------

GATE_VERDICT_SCHEMA = """
Return your analysis as JSON (no markdown fences):
{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "feedback": "Specific, actionable feedback for improvement"
}
"""

GATE_MINUS_1_PROMPT = """\
You are a pixel-art quality assurance expert. Analyze the provided reference \
animation strip and determine whether it is suitable as a visual guide for \
pixel-precise sprite generation.

## Your task
Compare the reference animation strip against the base character reference \
image and evaluate:

1. **Frame count**: Does the strip show the expected number of frames \
({expected_frames})?
2. **Character identity**: Does the character match the base reference in \
appearance (body shape, equipment, colors)?
3. **Pose correctness**: Do the poses match the animation "{animation_name}" \
({animation_context})?
4. **Proportional consistency**: Are proportions consistent across all frames?

## Verdict criteria
- **PASS** if the strip is a usable visual guide for pixel-precise translation
- **FAIL** if frames are missing, character is wrong, or poses are incorrect

{verdict_schema}
"""

GATE_0_PROMPT = """\
You are a pixel-art quality assurance expert. Compare a rendered pixel-art \
frame against its visual reference and evaluate fidelity.

## Your task
Evaluate whether the pixel-art frame faithfully represents the reference:

1. **Overall pose**: Does the pose match the reference?
2. **Proportions**: Are body proportions preserved?
3. **Key elements**: Are weapons, accessories, and equipment present?
4. **Color distribution**: Does the color usage roughly match?

{frame_description_section}

## Verdict criteria
- **PASS** if the pixel frame is a faithful representation of the reference
- **FAIL** if the pose, proportions, or key elements are significantly wrong

{verdict_schema}
"""

GATE_1_PROMPT = """\
You are a pixel-art identity verification expert. Compare a rendered frame \
against the anchor frame (IDLE Frame 0) to verify character identity.

## Your task
Determine if the rendered frame shows the SAME character as the anchor:

1. **Body proportions**: Same height, width, body shape?
2. **Colors**: Same hair, face, outfit colors?
3. **Equipment**: Same weapons/accessories visible?
4. **Pixel style**: Consistent pixel-art style and outline thickness?

## Verdict criteria
- **PASS** if this is clearly the same character as the anchor
- **FAIL** if proportions, colors, or equipment differ significantly

{verdict_schema}
"""

GATE_2_PROMPT = """\
You are a pixel-art animation continuity expert. Compare a rendered frame \
against the previous frame to evaluate temporal consistency.

## Your task
Evaluate the transition between the previous frame and the current frame:

1. **No teleportation**: Body parts haven't jumped to entirely new positions
2. **Incremental change**: Pose change is gradual, not a radical jump
3. **Outline consistency**: Outline thickness is maintained
4. **Color consistency**: Colors remain stable between frames

## Verdict criteria
- **PASS** if the transition is smooth and temporally coherent
- **FAIL** if there are jarring discontinuities or teleportation artifacts

{verdict_schema}
"""

GATE_3A_PROMPT = """\
You are a pixel-art animation quality expert. Evaluate a full animation row \
strip for overall coherence and game-readiness.

## Your task
Evaluate the assembled animation row for the "{animation_name}" animation \
({animation_context}):

1. **Character consistency**: All frames clearly show the same character
2. **Animation progression**: Logical start → peak → recovery flow
3. **Visual continuity**: No jarring visual discontinuities between frames
4. **Game quality**: Overall quality suitable for use in a 2D game

## Verdict criteria
- **PASS** if the row is a cohesive, game-ready animation sequence
- **FAIL** if there are identity inconsistencies or broken animation flow

{verdict_schema}
"""


# ---------------------------------------------------------------------------
# Verdict response parser
# ---------------------------------------------------------------------------


def parse_verdict_response(response_text: str, gate_name: str) -> GateVerdict:
    """Parse LLM response into a ``GateVerdict``.

    Handles JSON extraction from markdown code fences, missing fields with
    defaults. Defaults to ``passed=False`` if parsing fails (fail-safe).

    Args:
        response_text: Raw text response from the LLM.
        gate_name: Name of the gate for the verdict.

    Returns:
        A ``GateVerdict`` parsed from the response or a fail-safe default.
    """
    text = response_text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return GateVerdict(
            gate_name=gate_name,
            passed=False,
            confidence=0.0,
            feedback=f"Failed to parse gate response as JSON: {response_text[:200]}",
        )

    if not isinstance(data, dict):
        return GateVerdict(
            gate_name=gate_name,
            passed=False,
            confidence=0.0,
            feedback=f"Gate response is not a JSON object: {response_text[:200]}",
        )

    passed = bool(data.get("passed", False))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    feedback = str(data.get("feedback", "No feedback provided."))

    return GateVerdict(
        gate_name=gate_name,
        passed=passed,
        confidence=confidence,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# LLM gate checker
# ---------------------------------------------------------------------------


class LLMGateChecker:
    """Vision-based quality checks using Claude Opus 4.6.

    Each gate sends images as vision input and receives a structured
    JSON verdict. All gates use temperature 0.0 for deterministic judgment.
    """

    def __init__(
        self,
        chat_provider: ChatProvider,
    ) -> None:
        """Initialize the LLM gate checker.

        Args:
            chat_provider: Chat provider for LLM calls.
        """
        self._chat = chat_provider

    # ------------------------------------------------------------------
    # Gate methods
    # ------------------------------------------------------------------

    async def gate_minus_1(
        self,
        reference_strip: bytes,
        base_reference: bytes,
        animation: AnimationDef,
    ) -> GateVerdict:
        """Gate -1: Reference quality check.

        Validates that the rough reference strip (from Stage 1) is
        suitable for pixel-precise translation.

        Args:
            reference_strip: PNG bytes of the generated reference strip.
            base_reference: PNG bytes of the base character reference.
            animation: Animation definition for this row.

        Returns:
            A ``GateVerdict`` with the assessment.
        """
        prompt_text = GATE_MINUS_1_PROMPT.format(
            expected_frames=animation.frames,
            animation_name=animation.name,
            animation_context=animation.prompt_context or animation.name,
            verdict_schema=GATE_VERDICT_SCHEMA,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(reference_strip)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(base_reference)},
            },
        ]

        response_text = await self._chat.chat(
            [{"role": "user", "content": content}], temperature=0.0
        )
        return parse_verdict_response(response_text, "gate_minus_1")

    async def gate_0(
        self,
        rendered_frame: bytes,
        reference_frame: bytes,
        frame_description: str = "",
    ) -> GateVerdict:
        """Gate 0: Reference fidelity.

        Does the pixel grid faithfully represent the reference frame?

        Args:
            rendered_frame: PNG bytes of the rendered pixel-art frame.
            reference_frame: PNG bytes of the reference frame.
            frame_description: Optional description of the expected pose.

        Returns:
            A ``GateVerdict`` with the assessment.
        """
        desc_section = ""
        if frame_description:
            desc_section = f"Expected pose/action: {frame_description}"

        prompt_text = GATE_0_PROMPT.format(
            frame_description_section=desc_section,
            verdict_schema=GATE_VERDICT_SCHEMA,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(rendered_frame)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(reference_frame)},
            },
        ]

        response_text = await self._chat.chat(
            [{"role": "user", "content": content}], temperature=0.0
        )
        return parse_verdict_response(response_text, "gate_0")

    async def gate_1(
        self,
        rendered_frame: bytes,
        anchor_frame: bytes,
    ) -> GateVerdict:
        """Gate 1: Anchor consistency (identity).

        Is this the same character as the IDLE F0 anchor?

        Args:
            rendered_frame: PNG bytes of the rendered frame.
            anchor_frame: PNG bytes of the anchor frame (IDLE F0).

        Returns:
            A ``GateVerdict`` with the assessment.
        """
        prompt_text = GATE_1_PROMPT.format(
            verdict_schema=GATE_VERDICT_SCHEMA,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(rendered_frame)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(anchor_frame)},
            },
        ]

        response_text = await self._chat.chat(
            [{"role": "user", "content": content}], temperature=0.0
        )
        return parse_verdict_response(response_text, "gate_1")

    async def gate_2(
        self,
        rendered_frame: bytes,
        prev_frame: bytes,
    ) -> GateVerdict:
        """Gate 2: Temporal consistency.

        Is the transition from the previous frame smooth?

        Args:
            rendered_frame: PNG bytes of the current rendered frame.
            prev_frame: PNG bytes of the previous rendered frame.

        Returns:
            A ``GateVerdict`` with the assessment.
        """
        prompt_text = GATE_2_PROMPT.format(
            verdict_schema=GATE_VERDICT_SCHEMA,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(rendered_frame)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(prev_frame)},
            },
        ]

        response_text = await self._chat.chat(
            [{"role": "user", "content": content}], temperature=0.0
        )
        return parse_verdict_response(response_text, "gate_2")

    async def gate_3a(
        self,
        rendered_row_strip: bytes,
        reference_strip: bytes,
        animation: AnimationDef,
    ) -> GateVerdict:
        """Gate 3A: Row coherence.

        Does the full assembled animation row look like a cohesive animation?

        Args:
            rendered_row_strip: PNG bytes of the assembled row strip.
            reference_strip: PNG bytes of the Stage 1 reference strip.
            animation: Animation definition for this row.

        Returns:
            A ``GateVerdict`` with the assessment.
        """
        prompt_text = GATE_3A_PROMPT.format(
            animation_name=animation.name,
            animation_context=animation.prompt_context or animation.name,
            verdict_schema=GATE_VERDICT_SCHEMA,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(rendered_row_strip)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(reference_strip)},
            },
        ]

        response_text = await self._chat.chat(
            [{"role": "user", "content": content}], temperature=0.0
        )
        return parse_verdict_response(response_text, "gate_3a")
