"""Shared constants for sprite anatomy positioning.

These ratios define where key body regions fall relative to frame height.
They are used by both the programmatic gate checker (gates.py) and the
retry guidance prompts (prompts/retry.py) to ensure consistency between
what the verifier rejects and what the retry prompt instructs.

All ratios are applied as: ``int(frame_height * ratio)``.
"""

# ---------------------------------------------------------------------------
# Sprite anatomy ratios
# ---------------------------------------------------------------------------

# ~87.5% — where the character's feet should sit (non-transparent row)
FEET_ROW_RATIO: float = 0.875

# ±8% window around FEET_ROW_RATIO used for foot-position detection
FEET_WINDOW_RATIO: float = 0.08

# ~7.5% — transparent padding rows above the head
TOP_PADDING_RATIO: float = 0.075

# ~23.4% — approximate end of the hair region
HAIR_END_RATIO: float = 0.234
