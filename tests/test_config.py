"""Tests for spriteforge.config — YAML configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from spriteforge.config import load_config, validate_config


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

    def test_character_section_not_a_mapping(self, config_dir: Path) -> None:
        cfg = config_dir / "char_str.yaml"
        cfg.write_text(textwrap.dedent("""\
                character: "just a string"
                animations: []
            """))
        with pytest.raises(
            ValueError, match="'character' section must be a YAML mapping"
        ):
            load_config(cfg)

    def test_animations_section_not_a_sequence(self, config_dir: Path) -> None:
        cfg = config_dir / "anim_str.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations: "not a list"
            """))
        with pytest.raises(
            ValueError, match="'animations' section must be a YAML sequence"
        ):
            load_config(cfg)

    def test_load_config_with_palette(self, config_dir: Path) -> None:
        cfg = config_dir / "palette.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Goblin
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [80, 140, 60]
            """))
        spec = load_config(cfg)
        assert "P1" in spec.palettes
        assert spec.palettes["P1"].name == "P1"
        assert len(spec.palettes["P1"].colors) == 1

    def test_load_config_with_generation(self, config_dir: Path) -> None:
        cfg = config_dir / "gen.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Goblin
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                generation:
                  style: "Retro 8-bit"
                  facing: "left"
                  feet_row: 48
                  outline_width: 2
                  rules: "No anti-aliasing."
            """))
        spec = load_config(cfg)
        assert spec.generation.style == "Retro 8-bit"
        assert spec.generation.facing == "left"
        assert spec.generation.feet_row == 48
        assert spec.generation.outline_width == 2
        assert spec.generation.rules == "No anti-aliasing."

    def test_load_config_without_palette(self, config_dir: Path) -> None:
        cfg = config_dir / "no_palette.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
            """))
        spec = load_config(cfg)
        assert spec.palettes == {}

    def test_load_config_without_generation(self, config_dir: Path) -> None:
        cfg = config_dir / "no_gen.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
            """))
        spec = load_config(cfg)
        assert spec.generation.facing == "right"
        assert spec.generation.feet_row == 56

    def test_load_config_with_description(self, config_dir: Path) -> None:
        cfg = config_dir / "desc.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Goblin
                  description: "Small green goblin with tattered armor."
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
            """))
        spec = load_config(cfg)
        assert spec.character.description == "Small green goblin with tattered armor."

    def test_load_config_palette_outline(self, config_dir: Path) -> None:
        cfg = config_dir / "outline.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors: []
            """))
        spec = load_config(cfg)
        palette = spec.palettes["P1"]
        assert palette.outline.symbol == "O"
        assert palette.outline.rgb == (20, 15, 10)

    def test_load_config_palette_colors(self, config_dir: Path) -> None:
        cfg = config_dir / "colors.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [80, 140, 60]
                    - symbol: "e"
                      name: "Eyes"
                      rgb: [200, 30, 30]
            """))
        spec = load_config(cfg)
        palette = spec.palettes["P1"]
        assert len(palette.colors) == 2
        assert palette.colors[0].symbol == "s"
        assert palette.colors[0].element == "Skin"
        assert palette.colors[0].rgb == (80, 140, 60)
        assert palette.colors[1].symbol == "e"
        assert palette.colors[1].element == "Eyes"
        assert palette.colors[1].rgb == (200, 30, 30)

    def test_load_config_palette_duplicate_symbols(self, config_dir: Path) -> None:
        cfg = config_dir / "dup_sym.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: Test
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [80, 140, 60]
                    - symbol: "s"
                      name: "Other"
                      rgb: [100, 100, 100]
            """))
        with pytest.raises(ValidationError, match="[Dd]uplicate"):
            load_config(cfg)

    def test_load_config_full_generic_character(self, config_dir: Path) -> None:
        cfg = config_dir / "goblin.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Goblin Scout"
                  class: "Enemy"
                  description: |
                    Small, hunched goblin with bright green skin.
                  frame_width: 64
                  frame_height: 64
                  spritesheet_columns: 14
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [80, 140, 60]
                    - symbol: "e"
                      name: "Eyes"
                      rgb: [200, 30, 30]
                    - symbol: "a"
                      name: "Armor"
                      rgb: [110, 75, 40]
                    - symbol: "w"
                      name: "Weapon"
                      rgb: [160, 160, 170]
                    - symbol: "t"
                      name: "Teeth"
                      rgb: [230, 220, 190]
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    loop: true
                    timing_ms: 150
                    prompt_context: "Hunched standing pose."
                  - name: walk
                    row: 1
                    frames: 6
                    loop: true
                    timing_ms: 100
                    prompt_context: "Skulking walk."
                  - name: attack
                    row: 2
                    frames: 4
                    loop: false
                    timing_ms: 80
                    hit_frame: 2
                    prompt_context: "Quick overhead slash."
                  - name: hit
                    row: 3
                    frames: 3
                    loop: false
                    timing_ms: 100
                    prompt_context: "Recoil from being struck."
                  - name: death
                    row: 4
                    frames: 5
                    loop: false
                    timing_ms: 120
                    prompt_context: "Falls backward."
                generation:
                  style: "Modern HD pixel art (Dead Cells / Owlboy style)"
                  facing: "right"
                  feet_row: 56
                  rules: "64x64 pixel frames."
                base_image_path: "assets/goblin_scout_reference.png"
                output_path: "output/goblin_scout_spritesheet.png"
            """))
        spec = load_config(cfg)
        assert spec.character.name == "Goblin Scout"
        assert spec.character.character_class == "Enemy"
        assert "goblin" in spec.character.description.lower()
        assert len(spec.animations) == 5
        assert "P1" in spec.palettes
        assert len(spec.palettes["P1"].colors) == 5
        assert spec.generation.facing == "right"
        assert spec.base_image_path == "assets/goblin_scout_reference.png"
        assert spec.output_path == "output/goblin_scout_spritesheet.png"

    def test_load_config_minimal_enemy(self, config_dir: Path) -> None:
        cfg = config_dir / "minimal.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Bat"
                  class: "Enemy"
                animations:
                  - name: idle
                    row: 0
                    frames: 2
                    timing_ms: 200
                  - name: attack
                    row: 1
                    frames: 3
                    timing_ms: 100
                  - name: death
                    row: 2
                    frames: 3
                    timing_ms: 120
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [10, 10, 10]
                  colors:
                    - symbol: "b"
                      name: "Body"
                      rgb: [60, 40, 30]
                    - symbol: "w"
                      name: "Wings"
                      rgb: [80, 60, 50]
                    - symbol: "e"
                      name: "Eyes"
                      rgb: [255, 0, 0]
                    - symbol: "t"
                      name: "Teeth"
                      rgb: [240, 240, 230]
            """))
        spec = load_config(cfg)
        assert spec.character.name == "Bat"
        assert len(spec.animations) == 3
        assert len(spec.palettes["P1"].colors) == 4

    def test_load_config_with_auto_palette(self, config_dir: Path) -> None:
        cfg = config_dir / "auto_palette.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Auto"
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                generation:
                  auto_palette: true
                  max_palette_colors: 12
            """))
        spec = load_config(cfg)
        assert spec.generation.auto_palette is True
        assert spec.generation.max_palette_colors == 12

    def test_load_config_auto_palette_no_palette_section(
        self, config_dir: Path
    ) -> None:
        """When auto_palette is true and no palette section, config still loads."""
        cfg = config_dir / "auto_no_pal.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "NoPalette"
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                generation:
                  auto_palette: true
            """))
        spec = load_config(cfg)
        assert spec.generation.auto_palette is True
        assert spec.palettes == {}

    def test_load_config_auto_palette_with_palette_section(
        self, config_dir: Path
    ) -> None:
        """auto_palette + explicit palette section both load (palette usable as fallback)."""
        cfg = config_dir / "auto_with_pal.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Both"
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [80, 140, 60]
                generation:
                  auto_palette: true
                  max_palette_colors: 8
            """))
        spec = load_config(cfg)
        assert spec.generation.auto_palette is True
        assert spec.generation.max_palette_colors == 8
        assert "P1" in spec.palettes
        assert len(spec.palettes["P1"].colors) == 1

    def test_load_config_generation_max_palette_colors_default(
        self, config_dir: Path
    ) -> None:
        """max_palette_colors defaults to 16 when not specified."""
        cfg = config_dir / "gen_defaults.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "Defaults"
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                generation:
                  facing: "left"
            """))
        spec = load_config(cfg)
        assert spec.generation.auto_palette is False
        assert spec.generation.max_palette_colors == 16

    def test_load_config_with_model_fields(self, config_dir: Path) -> None:
        """Model fields can be specified in the generation section."""
        cfg = config_dir / "models.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
                generation:
                  grid_model: "custom-grid"
                  gate_model: "custom-gate"
                  labeling_model: "custom-label"
                  reference_model: "custom-ref"
            """))
        spec = load_config(cfg)
        assert spec.generation.grid_model == "custom-grid"
        assert spec.generation.gate_model == "custom-gate"
        assert spec.generation.labeling_model == "custom-label"
        assert spec.generation.reference_model == "custom-ref"

    def test_load_config_without_model_fields(self, config_dir: Path) -> None:
        """Model fields use defaults when not specified."""
        cfg = config_dir / "no_models.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 4
                    timing_ms: 150
            """))
        spec = load_config(cfg)
        assert spec.generation.grid_model == "gpt-5.2"
        assert spec.generation.gate_model == "gpt-5-mini"
        assert spec.generation.labeling_model == "gpt-5-nano"
        assert spec.generation.reference_model == "gpt-image-1.5"

    def test_load_config_existing_configs_still_valid(self, tmp_path: Path) -> None:
        """All existing configs in configs/ directory load without error."""
        from pathlib import Path

        configs_dir = Path(__file__).parent.parent / "configs"
        if not configs_dir.exists():
            pytest.skip("configs directory not found")

        # Test main character configs
        for config_file in ["theron.yaml", "sylara.yaml", "drunn.yaml"]:
            config_path = configs_dir / config_file
            if config_path.exists():
                spec = load_config(config_path)
                # Verify default model fields are applied
                assert spec.generation.grid_model == "gpt-5.2"
                assert spec.generation.gate_model == "gpt-5-mini"
                assert spec.generation.labeling_model == "gpt-5-nano"
                assert spec.generation.reference_model == "gpt-image-1.5"


class TestValidateConfig:
    """Tests for the validate_config function."""

    def test_valid_config_returns_empty_warnings(self, config_dir: Path) -> None:
        """A valid config should return an empty list of warnings."""
        cfg = config_dir / "valid.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
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
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [235, 210, 185]
                generation:
                  grid_model: "gpt-5.2"
                  gate_model: "gpt-5-mini"
                  labeling_model: "gpt-5-nano"
                  reference_model: "gpt-image-1.5"
            """))
        warnings = validate_config(cfg, check_base_image=False)
        assert warnings == []

    def test_duplicate_palette_symbol_raises_error(self, config_dir: Path) -> None:
        """Duplicate palette symbols should raise a ValidationError."""
        cfg = config_dir / "dup_sym.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [235, 210, 185]
                    - symbol: "s"
                      name: "Hair"
                      rgb: [100, 100, 100]
            """))
        # Duplicate symbols are caught by PaletteConfig validator during load_config
        with pytest.raises(ValidationError, match="[Dd]uplicate"):
            validate_config(cfg)

    def test_zero_frame_count_raises_error(self, config_dir: Path) -> None:
        """Animation with zero frames should raise a ValidationError."""
        cfg = config_dir / "zero_frames.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 0
                    timing_ms: 150
            """))
        # Zero frames is caught by AnimationDef field validator during load_config
        with pytest.raises(ValidationError):
            validate_config(cfg)

    def test_row_index_gap_produces_warning(self, config_dir: Path) -> None:
        """Non-contiguous row indices should produce a warning."""
        cfg = config_dir / "gap_rows.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                  - name: walk
                    row: 1
                    frames: 8
                    timing_ms: 100
                  - name: attack
                    row: 3
                    frames: 5
                    timing_ms: 80
            """))
        warnings = validate_config(cfg, check_base_image=False)
        assert len(warnings) == 1
        assert "gap" in warnings[0].lower()
        assert "[2]" in warnings[0]

    def test_missing_base_image_raises_error(self, config_dir: Path) -> None:
        """Config with non-existent base image should raise ValueError."""
        cfg = config_dir / "missing_img.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                base_image_path: "/nonexistent/path.png"
            """))
        with pytest.raises(ValueError, match="[Bb]ase image.*does not exist"):
            validate_config(cfg, check_base_image=True)

    def test_check_base_image_false_skips_path_check(self, config_dir: Path) -> None:
        """With check_base_image=False, missing base image should not raise error."""
        cfg = config_dir / "missing_img.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                base_image_path: "/nonexistent/path.png"
            """))
        # Should not raise error
        warnings = validate_config(cfg, check_base_image=False)
        # Should succeed without error
        assert isinstance(warnings, list)

    def test_empty_model_deployment_name_produces_warning(
        self, config_dir: Path
    ) -> None:
        """Empty model deployment name should produce a warning."""
        cfg = config_dir / "empty_model.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                generation:
                  grid_model: ""
                  gate_model: "gpt-5-mini"
            """))
        warnings = validate_config(cfg, check_base_image=False)
        assert len(warnings) >= 1
        assert any("grid_model" in w for w in warnings)

    def test_non_standard_outline_symbol_produces_warning(
        self, config_dir: Path
    ) -> None:
        """Non-'O' outline symbol should produce a warning."""
        cfg = config_dir / "custom_outline.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "X"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
                    - symbol: "s"
                      name: "Skin"
                      rgb: [235, 210, 185]
            """))
        warnings = validate_config(cfg, check_base_image=False)
        assert len(warnings) >= 1
        assert any("outline" in w.lower() and "'X'" in w for w in warnings)

    def test_excessive_palette_size_produces_warning(self, config_dir: Path) -> None:
        """Palette with >20 colors should produce a warning."""
        cfg = config_dir / "large_palette.yaml"
        # Create a config with 21 colors
        colors_yaml = "\n".join(
            [
                f'    - symbol: "{chr(ord("a") + i)}"\n'
                f'      name: "Color{i}"\n'
                f"      rgb: [{i*10}, {i*10}, {i*10}]"
                for i in range(21)
            ]
        )
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                palette:
                  outline:
                    symbol: "O"
                    name: "Outline"
                    rgb: [20, 15, 10]
                  colors:
            """) + colors_yaml)
        warnings = validate_config(cfg, check_base_image=False)
        assert len(warnings) >= 1
        assert any("21 colors" in w and ">20" in w for w in warnings)

    def test_all_example_configs_validate_cleanly(self) -> None:
        """All configs in configs/ directory should validate without errors."""
        from pathlib import Path

        configs_dir = Path(__file__).parent.parent / "configs"
        if not configs_dir.exists():
            pytest.skip("configs directory not found")

        # Test main character configs
        for config_file in ["theron.yaml", "sylara.yaml", "drunn.yaml"]:
            config_path = configs_dir / config_file
            if config_path.exists():
                # Validate with base image check disabled since paths may be relative
                warnings = validate_config(config_path, check_base_image=False)
                # Warnings are acceptable, but errors should not be raised
                assert isinstance(warnings, list)

    def test_relative_base_image_path_resolution(self, config_dir: Path) -> None:
        """Relative base image paths should be resolved relative to config file."""
        # Create a base image file in the same directory as the config
        img_path = config_dir / "ref.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header

        cfg = config_dir / "rel_path.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                animations:
                  - name: idle
                    row: 0
                    frames: 6
                    timing_ms: 150
                base_image_path: "ref.png"
            """))

        # Should not raise error since ref.png exists in the same directory
        warnings = validate_config(cfg, check_base_image=True)
        assert isinstance(warnings, list)

    def test_config_file_not_found_raises_error(self) -> None:
        """Non-existent config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            validate_config("/nonexistent/config.yaml")

    def test_frames_exceeding_max_columns_raises_error(self, config_dir: Path) -> None:
        """Animation with frames > spritesheet_columns should raise ValueError."""
        cfg = config_dir / "too_many_frames.yaml"
        cfg.write_text(textwrap.dedent("""\
                character:
                  name: "TestChar"
                  spritesheet_columns: 10
                animations:
                  - name: idle
                    row: 0
                    frames: 15
                    timing_ms: 150
            """))
        # This is caught by SpritesheetSpec validator during load_config
        with pytest.raises(ValueError, match="exceeding spritesheet_columns"):
            validate_config(cfg)
