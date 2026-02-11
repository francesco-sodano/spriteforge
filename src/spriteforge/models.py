"""Pydantic data models for characters, animations, palettes, and spritesheets."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class PaletteColor(BaseModel):
    """A single named color entry used in a character palette.

    Attributes:
        element: Human-readable name (e.g. "Skin", "Hair").
        r: Red channel (0–255).
        g: Green channel (0–255).
        b: Blue channel (0–255).
    """

    element: str
    r: int = Field(..., ge=0, le=255)
    g: int = Field(..., ge=0, le=255)
    b: int = Field(..., ge=0, le=255)

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Return the color as an ``(R, G, B)`` tuple."""
        return (self.r, self.g, self.b)


class AnimationDef(BaseModel):
    """Definition of a single animation row in the spritesheet.

    Attributes:
        name: Animation identifier (e.g. "idle", "walk").
        row: Row index in the spritesheet (>= 0).
        frames: Number of frames in this animation (>= 1).
        loop: Whether the animation loops.
        timing_ms: Milliseconds per frame (> 0).
        hit_frame: Optional frame index that represents the hit point.
    """

    name: str
    row: int = Field(..., ge=0)
    frames: int = Field(..., ge=1)
    loop: bool = False
    timing_ms: int = Field(..., gt=0)
    hit_frame: int | None = None


class CharacterConfig(BaseModel):
    """Character metadata and physical properties.

    Attributes:
        name: Character display name.
        character_class: RPG class (e.g. "Warrior", "Ranger").
        frame_width: Pixel width of each frame.
        frame_height: Pixel height of each frame.
        spritesheet_columns: Maximum frames per row in the sheet.
    """

    name: str
    character_class: str = ""
    frame_width: int = Field(default=64, gt=0)
    frame_height: int = Field(default=64, gt=0)
    spritesheet_columns: int = Field(default=14, gt=0)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return ``(frame_width, frame_height)``."""
        return (self.frame_width, self.frame_height)


class SpritesheetSpec(BaseModel):
    """Top-level model combining character, animations, and layout.

    Attributes:
        character: The character this spritesheet belongs to.
        animations: Ordered list of animation definitions.
        base_image_path: Optional path to the base reference image.
        output_path: Optional path for the generated spritesheet.
    """

    character: CharacterConfig
    animations: list[AnimationDef] = []
    base_image_path: str = ""
    output_path: str = ""

    @property
    def total_rows(self) -> int:
        """Number of animation rows in the spritesheet."""
        return len(self.animations)

    @property
    def sheet_width(self) -> int:
        """Total pixel width of the spritesheet."""
        return self.character.spritesheet_columns * self.character.frame_width

    @property
    def sheet_height(self) -> int:
        """Total pixel height of the spritesheet."""
        return len(self.animations) * self.character.frame_height

    @property
    def total_frames(self) -> int:
        """Sum of frame counts across all animations."""
        return sum(a.frames for a in self.animations)
