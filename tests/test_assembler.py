"""Tests for spriteforge.assembler — sprite row assembly into a final spritesheet."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from spriteforge.assembler import _open_image, assemble_spritesheet
from spriteforge.models import AnimationDef, CharacterConfig, SpritesheetSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row_png(width: int, height: int) -> bytes:
    """Create a minimal RGBA PNG as bytes."""
    img = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _open_image
# ---------------------------------------------------------------------------


class TestOpenImage:
    """Tests for the _open_image helper."""

    def test_open_image_from_bytes(self) -> None:
        data = _make_row_png(64, 64)
        img = _open_image(data)
        assert img.mode == "RGBA"
        assert img.size == (64, 64)

    def test_open_image_from_path(self, tmp_path: Path) -> None:
        p = tmp_path / "row.png"
        p.write_bytes(_make_row_png(128, 64))
        img = _open_image(p)
        assert img.mode == "RGBA"
        assert img.size == (128, 64)

    def test_open_image_from_str_path(self, tmp_path: Path) -> None:
        p = tmp_path / "row.png"
        p.write_bytes(_make_row_png(128, 64))
        img = _open_image(str(p))
        assert img.mode == "RGBA"
        assert img.size == (128, 64)

    def test_open_image_missing_path(self) -> None:
        with pytest.raises(FileNotFoundError, match="Row image not found"):
            _open_image("/nonexistent/path.png")

    def test_open_image_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported image source type"):
            _open_image(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# assemble_spritesheet
# ---------------------------------------------------------------------------


class TestAssembleSpritesheet:
    """Tests for the assemble_spritesheet function."""

    @pytest.fixture()
    def two_row_spec(self) -> SpritesheetSpec:
        return SpritesheetSpec(
            character=CharacterConfig(
                name="Hero", frame_width=64, frame_height=64, spritesheet_columns=14
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=6, timing_ms=150),
                AnimationDef(name="walk", row=1, frames=8, timing_ms=100),
            ],
        )

    def test_assemble_spritesheet_happy_path(
        self, two_row_spec: SpritesheetSpec
    ) -> None:
        row_images = {
            0: _make_row_png(6 * 64, 64),
            1: _make_row_png(8 * 64, 64),
        }
        sheet = assemble_spritesheet(row_images, two_row_spec)
        assert sheet.mode == "RGBA"
        assert sheet.size == (896, 128)

    def test_assemble_spritesheet_missing_row(
        self, two_row_spec: SpritesheetSpec
    ) -> None:
        row_images = {0: _make_row_png(6 * 64, 64)}
        with pytest.raises(ValueError, match="Missing image for row 1"):
            assemble_spritesheet(row_images, two_row_spec)

    def test_assemble_spritesheet_height_mismatch(
        self, two_row_spec: SpritesheetSpec
    ) -> None:
        row_images = {
            0: _make_row_png(6 * 64, 64),
            1: _make_row_png(8 * 64, 32),  # wrong height
        }
        with pytest.raises(ValueError, match="does not match frame height"):
            assemble_spritesheet(row_images, two_row_spec)

    def test_assemble_spritesheet_width_exceeds_sheet(
        self, two_row_spec: SpritesheetSpec
    ) -> None:
        row_images = {
            0: _make_row_png(6 * 64, 64),
            1: _make_row_png(896 + 64, 64),  # wider than sheet
        }
        with pytest.raises(ValueError, match="exceeds sheet width"):
            assemble_spritesheet(row_images, two_row_spec)

    def test_assemble_spritesheet_saves_to_file(
        self, two_row_spec: SpritesheetSpec, tmp_path: Path
    ) -> None:
        row_images = {
            0: _make_row_png(6 * 64, 64),
            1: _make_row_png(8 * 64, 64),
        }
        out = tmp_path / "sheet.png"
        sheet = assemble_spritesheet(row_images, two_row_spec, output_path=out)
        assert out.exists()
        saved = Image.open(out)
        assert saved.size == sheet.size

    def test_assemble_spritesheet_creates_parent_dirs(
        self, two_row_spec: SpritesheetSpec, tmp_path: Path
    ) -> None:
        row_images = {
            0: _make_row_png(6 * 64, 64),
            1: _make_row_png(8 * 64, 64),
        }
        out = tmp_path / "nested" / "deep" / "sheet.png"
        assemble_spritesheet(row_images, two_row_spec, output_path=out)
        assert out.exists()
