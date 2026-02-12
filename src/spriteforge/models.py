"""Pydantic data models for characters, animations, palettes, and spritesheets."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class PaletteColor(BaseModel):
    """A single named color entry used in a character palette.

    Attributes:
        element: Human-readable name (e.g. "Skin", "Hair").
        symbol: Single-character palette symbol (e.g. "s", "h").
        r: Red channel (0–255).
        g: Green channel (0–255).
        b: Blue channel (0–255).
    """

    element: str
    symbol: str = ""
    r: int = Field(..., ge=0, le=255)
    g: int = Field(..., ge=0, le=255)
    b: int = Field(..., ge=0, le=255)

    @field_validator("symbol")
    @classmethod
    def _symbol_must_be_single_char(cls, v: str) -> str:
        if len(v) != 1:
            raise ValueError("symbol must be exactly one character")
        return v

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Return the color as an ``(R, G, B)`` tuple."""
        return (self.r, self.g, self.b)

    @property
    def rgba(self) -> tuple[int, int, int, int]:
        """Return the color as an ``(R, G, B, A)`` tuple (always fully opaque)."""
        return (self.r, self.g, self.b, 255)


class PaletteConfig(BaseModel):
    """Palette configuration mapping symbols to RGBA colors.

    Attributes:
        transparent_symbol: Symbol for fully transparent pixels.
        outline: The outline color entry (symbol + RGB).
        colors: List of named palette color entries.
    """

    transparent_symbol: str = "."
    outline: PaletteColor = PaletteColor(
        element="Outline", symbol="O", r=20, g=40, b=40
    )
    colors: list[PaletteColor] = []

    @model_validator(mode="after")
    def _no_duplicate_symbols(self) -> "PaletteConfig":
        symbols: list[str] = [self.transparent_symbol, self.outline.symbol]
        for color in self.colors:
            symbols.append(color.symbol)
        seen: set[str] = set()
        for s in symbols:
            if s in seen:
                raise ValueError(f"Duplicate palette symbol: {s!r}")
            seen.add(s)
        return self


class AnimationDef(BaseModel):
    """Definition of a single animation row in the spritesheet.

    Attributes:
        name: Animation identifier (e.g. "idle", "walk").
        row: Row index in the spritesheet (>= 0).
        frames: Number of frames in this animation (>= 1).
        loop: Whether the animation loops.
        timing_ms: Milliseconds per frame (> 0).
        hit_frame: Optional frame index that represents the hit point.
        frame_descriptions: Optional per-frame pose descriptions for prompt generation.
    """

    name: str
    row: int = Field(..., ge=0)
    frames: int = Field(..., ge=1)
    loop: bool = False
    timing_ms: int = Field(..., gt=0)
    hit_frame: int | None = None
    frame_descriptions: list[str] = []

    @model_validator(mode="after")
    def _frame_descriptions_length(self) -> "AnimationDef":
        if self.frame_descriptions and len(self.frame_descriptions) != self.frames:
            raise ValueError(
                f"frame_descriptions length ({len(self.frame_descriptions)}) "
                f"must equal frames ({self.frames})"
            )
        return self


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

    @model_validator(mode="after")
    def _validate_animations(self) -> "SpritesheetSpec":
        # Reject duplicate row indices
        seen_rows: set[int] = set()
        for anim in self.animations:
            if anim.row in seen_rows:
                raise ValueError(f"Duplicate row index: {anim.row}")
            seen_rows.add(anim.row)
        # Reject frames exceeding spritesheet columns
        cols = self.character.spritesheet_columns
        for anim in self.animations:
            if anim.frames > cols:
                raise ValueError(
                    f"Animation {anim.name!r} has {anim.frames} frames, "
                    f"exceeding spritesheet_columns ({cols})"
                )
        return self

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
