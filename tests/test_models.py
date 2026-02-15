"""Tests for spriteforge.models — data models for animations, characters, and spritesheets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)

# ---------------------------------------------------------------------------
# AnimationDef
# ---------------------------------------------------------------------------


class TestAnimationDef:
    """Tests for the AnimationDef model."""

    def test_valid_animation(self) -> None:
        anim = AnimationDef(name="idle", row=0, frames=6, loop=True, timing_ms=150)
        assert anim.name == "idle"
        assert anim.row == 0
        assert anim.frames == 6
        assert anim.loop is True
        assert anim.timing_ms == 150
        assert anim.hit_frame is None

    def test_animation_with_hit_frame(self) -> None:
        anim = AnimationDef(
            name="attack1", row=2, frames=5, loop=False, timing_ms=80, hit_frame=2
        )
        assert anim.hit_frame == 2

    def test_negative_row_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnimationDef(name="bad", row=-1, frames=1, timing_ms=100)

    def test_zero_frames_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnimationDef(name="bad", row=0, frames=0, timing_ms=100)

    def test_zero_timing_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnimationDef(name="bad", row=0, frames=1, timing_ms=0)

    def test_animation_def_frame_descriptions_valid(self) -> None:
        anim = AnimationDef(
            name="idle",
            row=0,
            frames=3,
            timing_ms=100,
            frame_descriptions=["stand", "breathe", "blink"],
        )
        assert len(anim.frame_descriptions) == 3

    def test_animation_def_frame_descriptions_empty_is_ok(self) -> None:
        anim = AnimationDef(name="idle", row=0, frames=3, timing_ms=100)
        assert anim.frame_descriptions == []

    def test_animation_def_frame_descriptions_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="frame_descriptions length"):
            AnimationDef(
                name="idle",
                row=0,
                frames=3,
                timing_ms=100,
                frame_descriptions=["a", "b", "c", "d", "e"],
            )

    def test_animation_def_hit_frame_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="hit_frame.*must be < frames"):
            AnimationDef(name="attack1", row=2, frames=5, timing_ms=80, hit_frame=5)

    def test_animation_def_hit_frame_way_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="hit_frame.*must be < frames"):
            AnimationDef(name="attack1", row=2, frames=5, timing_ms=80, hit_frame=10)

    def test_animation_def_hit_frame_at_last_valid_index(self) -> None:
        anim = AnimationDef(name="attack1", row=2, frames=5, timing_ms=80, hit_frame=4)
        assert anim.hit_frame == 4

    def test_animation_def_prompt_context_default(self) -> None:
        anim = AnimationDef(name="idle", row=0, frames=6, timing_ms=150)
        assert anim.prompt_context == ""

    def test_animation_def_prompt_context_custom(self) -> None:
        anim = AnimationDef(
            name="idle",
            row=0,
            frames=6,
            timing_ms=150,
            prompt_context="Relaxed ready stance with bow at side",
        )
        assert anim.prompt_context == "Relaxed ready stance with bow at side"


# ---------------------------------------------------------------------------
# PaletteColor
# ---------------------------------------------------------------------------


class TestPaletteColor:
    """Tests for the PaletteColor model."""

    def test_valid_color(self) -> None:
        color = PaletteColor(element="Skin", symbol="s", r=210, g=170, b=130)
        assert color.rgb == (210, 170, 130)
        assert color.symbol == "s"

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColor(element="Bad", symbol="x", r=256, g=0, b=0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColor(element="Bad", symbol="x", r=-1, g=0, b=0)

    def test_palette_color_rejects_multi_char_symbol(self) -> None:
        with pytest.raises(ValidationError, match="exactly one character"):
            PaletteColor(element="X", symbol="ab", r=0, g=0, b=0)

    def test_palette_color_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValidationError, match="exactly one character"):
            PaletteColor(element="X", symbol="", r=0, g=0, b=0)

    def test_palette_color_symbol_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColor(element="Skin", r=210, g=170, b=130)  # type: ignore[call-arg]

    def test_palette_color_rgba(self) -> None:
        color = PaletteColor(element="Skin", symbol="s", r=100, g=150, b=200)
        assert color.rgba == (100, 150, 200, 255)


# ---------------------------------------------------------------------------
# PaletteConfig
# ---------------------------------------------------------------------------


class TestPaletteConfig:
    """Tests for the PaletteConfig model validators."""

    def test_palette_config_rejects_duplicate_symbols(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate palette symbol"):
            PaletteConfig(
                colors=[
                    PaletteColor(element="Skin", symbol="s", r=0, g=0, b=0),
                    PaletteColor(element="Hair", symbol="s", r=0, g=0, b=0),
                ],
            )

    def test_palette_config_name_default(self) -> None:
        palette = PaletteConfig()
        assert palette.name == ""

    def test_palette_config_name_custom(self) -> None:
        palette = PaletteConfig(name="P1")
        assert palette.name == "P1"


# ---------------------------------------------------------------------------
# CharacterConfig
# ---------------------------------------------------------------------------


class TestCharacterConfig:
    """Tests for the CharacterConfig model."""

    def test_defaults(self) -> None:
        char = CharacterConfig(name="Test")
        assert char.frame_width == 64
        assert char.frame_height == 64
        assert char.spritesheet_columns == 14
        assert char.character_class == ""
        assert char.frame_size == (64, 64)

    def test_custom_frame_size(self) -> None:
        char = CharacterConfig(name="Big", frame_width=128, frame_height=128)
        assert char.frame_size == (128, 128)

    def test_zero_frame_width_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CharacterConfig(name="Bad", frame_width=0)

    def test_character_config_description_default(self) -> None:
        char = CharacterConfig(name="Test")
        assert char.description == ""

    def test_character_config_description_custom(self) -> None:
        char = CharacterConfig(
            name="Goblin",
            description="Small green goblin with tattered armor.",
        )
        assert char.description == "Small green goblin with tattered armor."


# ---------------------------------------------------------------------------
# SpritesheetSpec
# ---------------------------------------------------------------------------


class TestSpritesheetSpec:
    """Tests for the SpritesheetSpec model."""

    def test_empty_animations(self) -> None:
        spec = SpritesheetSpec(character=CharacterConfig(name="Empty"))
        assert spec.total_rows == 0
        assert spec.sheet_width == 14 * 64
        assert spec.sheet_height == 0

    def test_sheet_dimensions(self) -> None:
        animations = [
            AnimationDef(name="idle", row=0, frames=6, timing_ms=150),
            AnimationDef(name="walk", row=1, frames=8, timing_ms=100),
        ]
        spec = SpritesheetSpec(
            character=CharacterConfig(name="Hero", spritesheet_columns=14),
            animations=animations,
        )
        assert spec.total_rows == 2
        assert spec.sheet_width == 896
        assert spec.sheet_height == 128

    def test_base_image_and_output_paths(self) -> None:
        spec = SpritesheetSpec(
            character=CharacterConfig(name="Hero"),
            base_image_path="/tmp/ref.png",
            output_path="/tmp/out.png",
        )
        assert spec.base_image_path == "/tmp/ref.png"
        assert spec.output_path == "/tmp/out.png"

    def test_spritesheet_spec_rejects_duplicate_rows(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate row index"):
            SpritesheetSpec(
                character=CharacterConfig(name="Hero"),
                animations=[
                    AnimationDef(name="idle", row=0, frames=6, timing_ms=150),
                    AnimationDef(name="walk", row=0, frames=8, timing_ms=100),
                ],
            )

    def test_spritesheet_spec_rejects_excess_frames(self) -> None:
        with pytest.raises(ValidationError, match="exceeding spritesheet_columns"):
            SpritesheetSpec(
                character=CharacterConfig(name="Hero", spritesheet_columns=14),
                animations=[
                    AnimationDef(name="big", row=0, frames=15, timing_ms=100),
                ],
            )

    def test_spritesheet_spec_total_frames(self) -> None:
        spec = SpritesheetSpec(
            character=CharacterConfig(name="Hero"),
            animations=[
                AnimationDef(name="idle", row=0, frames=6, timing_ms=150),
                AnimationDef(name="walk", row=1, frames=8, timing_ms=100),
            ],
        )
        assert spec.total_frames == 14

    def test_spritesheet_spec_palettes_default_empty(self) -> None:
        spec = SpritesheetSpec(character=CharacterConfig(name="Hero"))
        assert spec.palettes == {}

    def test_spritesheet_spec_palettes_with_entries(self) -> None:
        p1 = PaletteConfig(
            name="P1",
            colors=[
                PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
            ],
        )
        spec = SpritesheetSpec(
            character=CharacterConfig(name="Hero"),
            palettes={"P1": p1},
        )
        assert "P1" in spec.palettes
        assert spec.palettes["P1"].name == "P1"

    def test_spritesheet_spec_generation_default(self) -> None:
        spec = SpritesheetSpec(character=CharacterConfig(name="Hero"))
        assert isinstance(spec.generation, GenerationConfig)
        assert spec.generation.facing == "right"
        assert spec.generation.feet_row == 56


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    """Tests for the GenerationConfig model."""

    def test_generation_config_defaults(self) -> None:
        gen = GenerationConfig()
        assert gen.style == "Modern HD pixel art (Dead Cells / Owlboy style)"
        assert gen.facing == "right"
        assert gen.feet_row == 56
        assert gen.outline_width == 1
        assert gen.rules == ""

    def test_generation_config_facing_right(self) -> None:
        gen = GenerationConfig(facing="right")
        assert gen.facing == "right"

    def test_generation_config_facing_left(self) -> None:
        gen = GenerationConfig(facing="left")
        assert gen.facing == "left"

    def test_generation_config_facing_invalid(self) -> None:
        with pytest.raises(ValidationError, match="facing must be 'right' or 'left'"):
            GenerationConfig(facing="up")

    def test_generation_config_facing_normalized(self) -> None:
        gen = GenerationConfig(facing="LEFT")
        assert gen.facing == "left"

    def test_generation_config_custom_values(self) -> None:
        gen = GenerationConfig(
            style="Retro 8-bit",
            facing="left",
            feet_row=48,
            outline_width=2,
            rules="No anti-aliasing.",
        )
        assert gen.style == "Retro 8-bit"
        assert gen.facing == "left"
        assert gen.feet_row == 48
        assert gen.outline_width == 2
        assert gen.rules == "No anti-aliasing."

    def test_generation_config_auto_palette_default_false(self) -> None:
        gen = GenerationConfig()
        assert gen.auto_palette is False

    def test_generation_config_auto_palette_true(self) -> None:
        gen = GenerationConfig(auto_palette=True)
        assert gen.auto_palette is True

    def test_generation_config_max_palette_colors_default(self) -> None:
        gen = GenerationConfig()
        assert gen.max_palette_colors == 16

    def test_generation_config_max_palette_colors_custom(self) -> None:
        gen = GenerationConfig(max_palette_colors=12)
        assert gen.max_palette_colors == 12

    def test_generation_config_max_palette_colors_min_boundary(self) -> None:
        gen = GenerationConfig(max_palette_colors=2)
        assert gen.max_palette_colors == 2

    def test_generation_config_max_palette_colors_max_boundary(self) -> None:
        gen = GenerationConfig(max_palette_colors=23)
        assert gen.max_palette_colors == 23

    def test_generation_config_max_palette_colors_below_min(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(max_palette_colors=1)

    def test_generation_config_max_palette_colors_above_max(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(max_palette_colors=24)

    def test_generation_config_max_palette_colors_zero(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(max_palette_colors=0)

    def test_generation_config_model_defaults(self) -> None:
        gen = GenerationConfig()
        assert gen.grid_model == "gpt-5.2"
        assert gen.gate_model == "gpt-5-mini"
        assert gen.labeling_model == "gpt-5-nano"
        assert gen.reference_model == "gpt-image-1.5"

    def test_generation_config_custom_models(self) -> None:
        gen = GenerationConfig(
            grid_model="custom-grid",
            gate_model="custom-gate",
            labeling_model="custom-label",
            reference_model="custom-ref",
        )
        assert gen.grid_model == "custom-grid"
        assert gen.gate_model == "custom-gate"
        assert gen.labeling_model == "custom-label"
        assert gen.reference_model == "custom-ref"

    def test_generation_config_partial_override(self) -> None:
        gen = GenerationConfig(grid_model="custom-only")
        assert gen.grid_model == "custom-only"
        assert gen.gate_model == "gpt-5-mini"
        assert gen.labeling_model == "gpt-5-nano"
        assert gen.reference_model == "gpt-image-1.5"

    def test_generation_config_serialization(self) -> None:
        gen = GenerationConfig(
            grid_model="test-grid",
            gate_model="test-gate",
        )
        data = gen.model_dump()
        assert data["grid_model"] == "test-grid"
        assert data["gate_model"] == "test-gate"
        assert data["labeling_model"] == "gpt-5-nano"
        assert data["reference_model"] == "gpt-image-1.5"
        # Test round-trip
        gen2 = GenerationConfig(**data)
        assert gen2.grid_model == gen.grid_model
        assert gen2.gate_model == gen.gate_model
        assert gen2.labeling_model == gen.labeling_model
        assert gen2.reference_model == gen.reference_model
