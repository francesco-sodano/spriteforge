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
    symbol: str
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
        name: Palette identifier (e.g. "P1", "P2").
        transparent_symbol: Symbol for fully transparent pixels.
        outline: The outline color entry (symbol + RGB).
        colors: List of named palette color entries.
    """

    name: str = ""
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
        prompt_context: Animation-specific context used in Stage 2 prompt generation.
    """

    name: str
    row: int = Field(..., ge=0)
    frames: int = Field(..., ge=1)
    loop: bool = False
    timing_ms: int = Field(..., gt=0)
    hit_frame: int | None = None
    frame_descriptions: list[str] = []
    prompt_context: str = ""

    @model_validator(mode="after")
    def _validate_animation_constraints(self) -> "AnimationDef":
        if self.frame_descriptions and len(self.frame_descriptions) != self.frames:
            raise ValueError(
                f"frame_descriptions length ({len(self.frame_descriptions)}) "
                f"must equal frames ({self.frames})"
            )
        if self.hit_frame is not None and self.hit_frame >= self.frames:
            raise ValueError(
                f"hit_frame ({self.hit_frame}) must be < frames ({self.frames})"
            )
        return self


class CharacterConfig(BaseModel):
    """Character metadata and physical properties.

    Attributes:
        name: Character display name.
        character_class: RPG class (e.g. "Warrior", "Ranger").
        description: Detailed visual description for AI prompt generation.
        frame_width: Pixel width of each frame.
        frame_height: Pixel height of each frame.
        spritesheet_columns: Maximum frames per row in the sheet.
    """

    name: str
    character_class: str = ""
    description: str = ""
    frame_width: int = Field(default=64, gt=0)
    frame_height: int = Field(default=64, gt=0)
    spritesheet_columns: int = Field(default=14, gt=0)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return ``(frame_width, frame_height)``."""
        return (self.frame_width, self.frame_height)


class BudgetConfig(BaseModel):
    """Budget constraints for LLM call tracking.

    Attributes:
        max_llm_calls: Hard cap on total LLM calls (across all providers).
            When 0 (default), no limit is enforced.
        max_retries_per_row: Per-row retry budget (overrides RetryConfig.max_retries
            when set). When 0 (default), RetryConfig.max_retries is used.
        warn_at_percentage: Emit warning when budget consumption reaches this
            percentage (0.0–1.0). Default 0.8 (80%).
    """

    max_llm_calls: int = Field(default=0, ge=0)
    max_retries_per_row: int = Field(default=0, ge=0)
    warn_at_percentage: float = Field(default=0.8, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    """Generation settings that control Stage 1 and Stage 2 AI behavior.

    Attributes:
        style: Art style description for prompts (e.g., "Modern HD pixel art").
        facing: Direction the character faces ("right" or "left").
        feet_row: Y-coordinate where feet should be placed (~56 for 64px frames).
        outline_width: Outline thickness in pixels.
        rules: Additional generation rules as free-form text for prompts.
        auto_palette: When True, palette is auto-extracted from the base
            reference image by the preprocessor instead of using the
            YAML-defined palette.
        max_palette_colors: Maximum number of palette colors (excluding
            transparent) to extract via quantization when *auto_palette*
            is enabled.  Default 16 (1 outline + 15 character colors).
            Limited to 23 (1 outline symbol + 22 available symbols in
            SYMBOL_POOL; '.' transparent symbol is implicit).
        grid_model: Azure AI Foundry model deployment name for Stage 2 grid
            generation (needs strong spatial reasoning).
        gate_model: Azure AI Foundry model deployment name for verification
            gates (simpler pass/fail checks).
        labeling_model: Azure AI Foundry model deployment name for semantic
            palette labeling (color naming).
        reference_model: Azure AI Foundry model deployment name for Stage 1
            reference image generation.
        budget: Optional budget constraints for LLM call tracking and limits.
    """

    style: str = "Modern HD pixel art (Dead Cells / Owlboy style)"
    facing: str = "right"
    feet_row: int = Field(default=56, ge=0)
    outline_width: int = Field(default=1, ge=0)
    rules: str = ""
    auto_palette: bool = False
    max_palette_colors: int = Field(default=16, ge=2, le=23)
    semantic_labels: bool = True
    grid_model: str = "gpt-5.2"
    gate_model: str = "gpt-5-mini"
    labeling_model: str = "gpt-5-nano"
    reference_model: str = "gpt-image-1.5"
    budget: BudgetConfig | None = None

    @field_validator("facing")
    @classmethod
    def _validate_facing(cls, v: str) -> str:
        if v.lower() not in ("right", "left"):
            raise ValueError(f"facing must be 'right' or 'left', got {v!r}")
        return v.lower()


class FrameContext(BaseModel):
    """Context bundle for frame generation and verification.

    Encapsulates all immutable frame-generation parameters to reduce
    parameter threading across the call chain.

    Attributes:
        palette: Palette config with symbol → RGBA mappings.
        palette_map: Pre-computed symbol → RGBA tuple mapping.
        generation: Generation config (style, facing, rules).
        frame_width: Width of each frame in pixels.
        frame_height: Height of each frame in pixels.
        animation: Animation definition for this frame's row.
        spritesheet_columns: Number of columns in the output spritesheet.
        anchor_grid: Optional anchor frame grid (from row 0, frame 0).
        anchor_rendered: Optional anchor frame PNG bytes.
        quantized_reference: Optional quantized reference PNG bytes.
    """

    palette: PaletteConfig
    palette_map: dict[str, tuple[int, int, int, int]]
    generation: GenerationConfig
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    animation: AnimationDef
    spritesheet_columns: int = Field(gt=0)
    anchor_grid: list[str] | None = None
    anchor_rendered: bytes | None = None
    quantized_reference: bytes | None = None

    model_config = {"frozen": True}


class SpritesheetSpec(BaseModel):
    """Top-level model combining character, animations, and layout.

    Attributes:
        character: The character this spritesheet belongs to.
        animations: Ordered list of animation definitions.
        palettes: Named palette configurations (e.g. "P1", "P2").
        generation: Generation settings for AI pipeline behavior.
        base_image_path: Optional path to the base reference image.
        output_path: Optional path for the generated spritesheet.
    """

    character: CharacterConfig
    animations: list[AnimationDef] = []
    palettes: dict[str, PaletteConfig] = {}
    generation: GenerationConfig = GenerationConfig()
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
