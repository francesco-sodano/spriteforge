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


def _make_row_png(
    width: int, height: int, color: tuple[int, int, int, int] = (255, 0, 0, 255)
) -> bytes:
    """Create a minimal RGBA PNG as bytes."""
    img = Image.new("RGBA", (width, height), color)
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

    def test_assemble_non_contiguous_rows(self) -> None:
        """Animations with rows [0, 2, 5] → all rows visible, sheet height = 3 * frame_height."""
        spec = SpritesheetSpec(
            character=CharacterConfig(
                name="Hero", frame_width=64, frame_height=64, spritesheet_columns=14
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
                AnimationDef(name="walk", row=2, frames=4, timing_ms=100),
                AnimationDef(name="attack", row=5, frames=4, timing_ms=100),
            ],
        )
        colors = [
            (255, 0, 0, 255),  # red for row 0
            (0, 255, 0, 255),  # green for row 2
            (0, 0, 255, 255),  # blue for row 5
        ]
        row_images = {
            0: _make_row_png(4 * 64, 64, colors[0]),
            2: _make_row_png(4 * 64, 64, colors[1]),
            5: _make_row_png(4 * 64, 64, colors[2]),
        }
        sheet = assemble_spritesheet(row_images, spec)
        # Sheet should be compact: 3 rows × 64 = 192, not (5+1)*64 = 384
        assert sheet.size == (896, 192)
        # Each row should be visible at sequential positions
        for idx, color in enumerate(colors):
            pixel = sheet.getpixel((0, idx * 64))
            assert pixel == color, f"Row at sequential index {idx} has wrong color"

    def test_assemble_contiguous_rows_unchanged(self) -> None:
        """Contiguous row indices [0, 1, 2] still work correctly (regression)."""
        spec = SpritesheetSpec(
            character=CharacterConfig(
                name="Hero", frame_width=64, frame_height=64, spritesheet_columns=14
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
                AnimationDef(name="walk", row=1, frames=4, timing_ms=100),
                AnimationDef(name="attack", row=2, frames=4, timing_ms=100),
            ],
        )
        colors = [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
        ]
        row_images = {
            0: _make_row_png(4 * 64, 64, colors[0]),
            1: _make_row_png(4 * 64, 64, colors[1]),
            2: _make_row_png(4 * 64, 64, colors[2]),
        }
        sheet = assemble_spritesheet(row_images, spec)
        assert sheet.size == (896, 192)
        for idx, color in enumerate(colors):
            pixel = sheet.getpixel((0, idx * 64))
            assert pixel == color

    def test_assemble_single_row_sparse_index(self) -> None:
        """Single animation at row=3 → sheet height = 1 * frame_height, row visible."""
        spec = SpritesheetSpec(
            character=CharacterConfig(
                name="Hero", frame_width=64, frame_height=64, spritesheet_columns=14
            ),
            animations=[
                AnimationDef(name="idle", row=3, frames=4, timing_ms=150),
            ],
        )
        row_images = {3: _make_row_png(4 * 64, 64, (255, 0, 0, 255))}
        sheet = assemble_spritesheet(row_images, spec)
        # Sheet should be 1 row, not 4 rows
        assert sheet.size == (896, 64)
        pixel = sheet.getpixel((0, 0))
        assert pixel == (255, 0, 0, 255)

    def test_sheet_height_matches_animation_count(self) -> None:
        """Verify sheet dimensions match actual animation count, not max row index."""
        spec = SpritesheetSpec(
            character=CharacterConfig(
                name="Hero", frame_width=64, frame_height=64, spritesheet_columns=14
            ),
            animations=[
                AnimationDef(name="idle", row=0, frames=4, timing_ms=150),
                AnimationDef(name="walk", row=2, frames=4, timing_ms=100),
                AnimationDef(name="attack", row=5, frames=4, timing_ms=100),
            ],
        )
        # sheet_height should be based on animation count (3), not max row (5)
        assert spec.sheet_height == 3 * 64
        assert spec.sheet_height == 192
