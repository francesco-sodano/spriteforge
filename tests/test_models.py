"""Tests for spriteforge.models — data models for animations, characters, and spritesheets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    PaletteColor,
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


# ---------------------------------------------------------------------------
# PaletteColor
# ---------------------------------------------------------------------------


class TestPaletteColor:
    """Tests for the PaletteColor model."""

    def test_valid_color(self) -> None:
        color = PaletteColor(element="Skin", r=210, g=170, b=130)
        assert color.rgb == (210, 170, 130)

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColor(element="Bad", r=256, g=0, b=0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColor(element="Bad", r=-1, g=0, b=0)


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
