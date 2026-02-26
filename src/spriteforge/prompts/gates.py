"""Prompt constants and builders for verification gates.

Contains prompt templates for all gate checks (Gate -1, 0, 1, 2, 3A) and
the shared verdict JSON schema — all originally defined inline in
:pymod:`spriteforge.gates`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Verdict schema (shared across all gates)
# ---------------------------------------------------------------------------

GATE_VERDICT_SCHEMA: str = """
Return your analysis as JSON (no markdown fences):
{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "feedback": "Specific, actionable feedback for improvement",
    "problematic_frame_indices": [0, 2, 5]
}

Include "problematic_frame_indices" only when relevant (primarily Gate 3A) and
only when "passed" is false.
"""

# ---------------------------------------------------------------------------
# Gate -1: Reference strip quality
# ---------------------------------------------------------------------------

GATE_MINUS_1_PROMPT: str = """\
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

# ---------------------------------------------------------------------------
# Gate 0: Reference fidelity
# ---------------------------------------------------------------------------

GATE_0_PROMPT: str = """\
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

# ---------------------------------------------------------------------------
# Gate 1: Anchor consistency (identity)
# ---------------------------------------------------------------------------

GATE_1_PROMPT: str = """\
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

# ---------------------------------------------------------------------------
# Gate 2: Temporal consistency
# ---------------------------------------------------------------------------

GATE_2_PROMPT: str = """\
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

# ---------------------------------------------------------------------------
# Gate 3A: Row coherence
# ---------------------------------------------------------------------------

GATE_3A_PROMPT: str = """\
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
