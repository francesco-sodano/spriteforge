"""Tests for spriteforge.config — YAML configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from spriteforge.config import load_config


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test config files."""
    return tmp_path


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_valid_config(self, config_dir: Path) -> None:
        cfg = config_dir / "valid.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Theron Ashblade"
                  class: "Warrior"
                  frame_size: [64, 64]
                  spritesheet_columns: 14

                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    loop: true
                    timing_ms: 150

                  - name: walk
                    row: 1
                    frames: 8
                    loop: true
                    timing_ms: 100
            """))
        spec = load_config(cfg)
        assert spec.character.name == "Theron Ashblade"
        assert spec.character.character_class == "Warrior"
        assert spec.character.frame_size == (64, 64)
        assert len(spec.animations) == 2
        assert spec.animations[0].name == "idle"
        assert spec.animations[1].name == "walk"

    def test_config_with_hit_frame(self, config_dir: Path) -> None:
        cfg = config_dir / "hit.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Test"
                animations:
                  - name: attack1
                    row: 0
                    frames: 5
                    loop: false
                    timing_ms: 80
                    hit_frame: 2
            """))
        spec = load_config(cfg)
        assert spec.animations[0].hit_frame == 2

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path.yaml")

    def test_missing_character_section(self, config_dir: Path) -> None:
        cfg = config_dir / "no_char.yaml"
        cfg.write_text("animations: []\n")
        with pytest.raises(ValueError, match="'character' section"):
            load_config(cfg)

    def test_missing_animations_section(self, config_dir: Path) -> None:
        cfg = config_dir / "no_anim.yaml"
        cfg.write_text("character:\n  name: Test\n")
        with pytest.raises(ValueError, match="'animations' section"):
            load_config(cfg)

    def test_duplicate_row_indices(self, config_dir: Path) -> None:
        cfg = config_dir / "dupe.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                  - name: walk
                    row: 0
                    frames: 8
                    timing_ms: 100
            """))
        with pytest.raises(ValueError, match="Duplicate row"):
            load_config(cfg)

    def test_bad_frame_size(self, config_dir: Path) -> None:
        cfg = config_dir / "bad_fs.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                  frame_size: [64]
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
            """))
        with pytest.raises(ValueError, match="frame_size"):
            load_config(cfg)

    def test_non_mapping_top_level(self, config_dir: Path) -> None:
        cfg = config_dir / "list.yaml"
        cfg.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(cfg)

    def test_animations_sorted_by_row(self, config_dir: Path) -> None:
        cfg = config_dir / "unsorted.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: walk
                    row: 1
                    frames: 8
                    timing_ms: 100
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
            """))
        spec = load_config(cfg)
        assert spec.animations[0].row == 0
        assert spec.animations[1].row == 1

    def test_optional_paths(self, config_dir: Path) -> None:
        cfg = config_dir / "paths.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                base_image_path: /tmp/ref.png
                output_path: /tmp/out.png
            """))
        spec = load_config(cfg)
        assert spec.base_image_path == "/tmp/ref.png"
        assert spec.output_path == "/tmp/out.png"

    def test_malformed_yaml_syntax(self, config_dir: Path) -> None:
        cfg = config_dir / "broken.yaml"
        cfg.write_text("{ bad yaml [")
        with pytest.raises(ValueError, match="Malformed YAML"):
            load_config(cfg)
