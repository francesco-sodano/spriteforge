"""Tests for character YAML configuration files in configs/."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spriteforge.config import load_config
from spriteforge.models import SpritesheetSpec

# Resolve the configs directory relative to the repo root.
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture()
def sylara_spec() -> SpritesheetSpec:
    """Load Sylara's config and return a validated SpritesheetSpec."""
    return load_config(CONFIGS_DIR / "sylara.yaml")


@pytest.fixture()
def theron_spec() -> SpritesheetSpec:
    """Load Theron's config and return a validated SpritesheetSpec."""
    return load_config(CONFIGS_DIR / "theron.yaml")


@pytest.fixture()
def drunn_spec() -> SpritesheetSpec:
    """Load Drunn's config and return a validated SpritesheetSpec."""
    return load_config(CONFIGS_DIR / "drunn.yaml")


@pytest.fixture()
def all_specs(
    sylara_spec: SpritesheetSpec,
    theron_spec: SpritesheetSpec,
    drunn_spec: SpritesheetSpec,
) -> dict[str, SpritesheetSpec]:
    """Return all three character specs keyed by name."""
    return {
        "sylara": sylara_spec,
        "theron": theron_spec,
        "drunn": drunn_spec,
    }


# ── Config loading tests ──────────────────────────────────────────────


class TestConfigLoading:
    """Each YAML file must load into a valid SpritesheetSpec."""

    def test_load_sylara_config(self, sylara_spec: SpritesheetSpec) -> None:
        assert sylara_spec.character.name == "Sylara Windarrow"
        assert sylara_spec.character.character_class == "Ranger"

    def test_load_theron_config(self, theron_spec: SpritesheetSpec) -> None:
        assert theron_spec.character.name == "Theron Ashblade"
        assert theron_spec.character.character_class == "Warrior"

    def test_load_drunn_config(self, drunn_spec: SpritesheetSpec) -> None:
        assert drunn_spec.character.name == "Drunn Ironhelm"
        assert drunn_spec.character.character_class == "Berserker"


# ── Palette validation tests ──────────────────────────────────────────


class TestPaletteValidation:
    """Verify palette color counts and values match the instruction docs."""

    def test_sylara_palette_count(self, sylara_spec: SpritesheetSpec) -> None:
        assert sylara_spec.palette is not None
        palette = sylara_spec.palette
        # 10 colors + transparent + outline = 12 symbols
        assert len(palette.colors) == 10

    def test_theron_palette_count(self, theron_spec: SpritesheetSpec) -> None:
        assert theron_spec.palette is not None
        palette = theron_spec.palette
        # 8 base colors + edge glow = 9 colors + transparent + outline = 11 symbols
        assert len(palette.colors) == 9

    def test_drunn_palette_count(self, drunn_spec: SpritesheetSpec) -> None:
        assert drunn_spec.palette is not None
        palette = drunn_spec.palette
        # 10 colors + transparent + outline = 12 symbols
        assert len(palette.colors) == 10

    def test_sylara_outline_color(self, sylara_spec: SpritesheetSpec) -> None:
        assert sylara_spec.palette is not None
        outline = sylara_spec.palette.outline
        assert outline.symbol == "O"
        assert outline.rgb == (0, 80, 80)

    def test_theron_outline_color(self, theron_spec: SpritesheetSpec) -> None:
        assert theron_spec.palette is not None
        outline = theron_spec.palette.outline
        assert outline.symbol == "O"
        assert outline.rgb == (20, 15, 10)

    def test_drunn_outline_color(self, drunn_spec: SpritesheetSpec) -> None:
        assert drunn_spec.palette is not None
        outline = drunn_spec.palette.outline
        assert outline.symbol == "O"
        assert outline.rgb == (20, 15, 10)


# ── Animation validation tests ───────────────────────────────────────


EXPECTED_ANIMATION_NAMES = [
    "idle",
    "walk",
    "attack1",
    "attack2",
    "attack3",
    "jump",
    "jump_attack",
    "magic",
    "hit",
    "knockdown",
    "getup",
    "death",
    "mount_idle",
    "mount_attack",
    "run",
    "throw",
]

EXPECTED_FRAME_COUNTS = [6, 8, 5, 5, 7, 4, 4, 8, 3, 4, 4, 6, 4, 5, 6, 6]

LOOPING_ANIMATIONS = {"idle", "walk", "mount_idle", "run"}


class TestAnimationValidation:
    """Animation row definitions must match the spec."""

    def test_all_have_16_animations(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            assert len(spec.animations) == 16, f"{name} has {len(spec.animations)}"

    def test_animation_row_indices_sequential(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            rows = [a.row for a in spec.animations]
            assert rows == list(range(16)), f"{name} rows: {rows}"

    def test_frame_counts_match_spec(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            counts = [a.frames for a in spec.animations]
            assert counts == EXPECTED_FRAME_COUNTS, f"{name} frame counts: {counts}"

    def test_looping_flags_correct(self, all_specs: dict[str, SpritesheetSpec]) -> None:
        for name, spec in all_specs.items():
            for anim in spec.animations:
                if anim.name in LOOPING_ANIMATIONS:
                    assert anim.loop is True, f"{name}/{anim.name} should loop"
                else:
                    assert anim.loop is False, f"{name}/{anim.name} should not loop"

    def test_attack_active_frames_set(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        attack_names = {"attack1", "attack2", "attack3", "jump_attack", "mount_attack"}
        for name, spec in all_specs.items():
            for anim in spec.animations:
                if anim.name in attack_names:
                    assert (
                        anim.hit_frame is not None
                    ), f"{name}/{anim.name} missing hit_frame"

    def test_throw_active_frame_set(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            throw_anim = next(a for a in spec.animations if a.name == "throw")
            assert throw_anim.hit_frame is not None, f"{name}/throw missing hit_frame"


# ── Frame timing tests ───────────────────────────────────────────────

# Expected timings per character: {animation_name: timing_ms}
SYLARA_TIMINGS = {
    "idle": 150,
    "walk": 100,
    "attack1": 70,
    "attack2": 65,
    "attack3": 80,
    "jump": 90,
    "jump_attack": 80,
    "magic": 120,
    "hit": 100,
    "knockdown": 100,
    "getup": 100,
    "death": 130,
    "mount_idle": 150,
    "mount_attack": 80,
    "run": 70,
    "throw": 100,
}

THERON_TIMINGS = {
    "idle": 150,
    "walk": 100,
    "attack1": 80,
    "attack2": 70,
    "attack3": 90,
    "jump": 100,
    "jump_attack": 80,
    "magic": 120,
    "hit": 100,
    "knockdown": 100,
    "getup": 120,
    "death": 130,
    "mount_idle": 150,
    "mount_attack": 80,
    "run": 80,
    "throw": 100,
}

DRUNN_TIMINGS = {
    "idle": 160,
    "walk": 120,
    "attack1": 100,
    "attack2": 90,
    "attack3": 110,
    "jump": 110,
    "jump_attack": 90,
    "magic": 140,
    "hit": 100,
    "knockdown": 110,
    "getup": 130,
    "death": 140,
    "mount_idle": 160,
    "mount_attack": 90,
    "run": 100,
    "throw": 110,
}


class TestFrameTimings:
    """Frame timings must match per-character values from the instruction docs."""

    def test_sylara_timings(self, sylara_spec: SpritesheetSpec) -> None:
        for anim in sylara_spec.animations:
            expected = SYLARA_TIMINGS[anim.name]
            assert (
                anim.timing_ms == expected
            ), f"sylara/{anim.name}: {anim.timing_ms} != {expected}"

    def test_theron_timings(self, theron_spec: SpritesheetSpec) -> None:
        for anim in theron_spec.animations:
            expected = THERON_TIMINGS[anim.name]
            assert (
                anim.timing_ms == expected
            ), f"theron/{anim.name}: {anim.timing_ms} != {expected}"

    def test_drunn_timings(self, drunn_spec: SpritesheetSpec) -> None:
        for anim in drunn_spec.animations:
            expected = DRUNN_TIMINGS[anim.name]
            assert (
                anim.timing_ms == expected
            ), f"drunn/{anim.name}: {anim.timing_ms} != {expected}"

    def test_frame_timings_differ_per_character(
        self,
        sylara_spec: SpritesheetSpec,
        theron_spec: SpritesheetSpec,
        drunn_spec: SpritesheetSpec,
    ) -> None:
        """For speed-related animations, Drunn >= Theron >= Sylara."""
        speed_anims = ["attack1", "attack2", "attack3", "run", "walk"]
        for anim_name in speed_anims:
            s = next(a for a in sylara_spec.animations if a.name == anim_name)
            t = next(a for a in theron_spec.animations if a.name == anim_name)
            d = next(a for a in drunn_spec.animations if a.name == anim_name)
            assert d.timing_ms >= t.timing_ms >= s.timing_ms, (
                f"{anim_name}: Drunn({d.timing_ms}) >= Theron({t.timing_ms}) "
                f">= Sylara({s.timing_ms})"
            )


# ── Special animation flags ──────────────────────────────────────────


class TestSpecialAnimationFlags:
    """Verify hold_last_frame, upper_body_only, and other special flags."""

    def test_death_does_not_loop(self, all_specs: dict[str, SpritesheetSpec]) -> None:
        for name, spec in all_specs.items():
            death = next(a for a in spec.animations if a.name == "death")
            assert death.loop is False, f"{name}/death should not loop"

    def test_prompt_context_present(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        """Every animation must have a non-empty prompt_context."""
        for name, spec in all_specs.items():
            for anim in spec.animations:
                assert (
                    anim.prompt_context.strip()
                ), f"{name}/{anim.name} has empty prompt_context"


# ── Cross-character consistency tests ────────────────────────────────


class TestCrossCharacterConsistency:
    """All three characters must share the same animation structure."""

    def test_all_characters_same_frame_counts(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        specs = list(all_specs.values())
        base_counts = [a.frames for a in specs[0].animations]
        for spec in specs[1:]:
            counts = [a.frames for a in spec.animations]
            assert counts == base_counts

    def test_all_characters_same_animation_names(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        specs = list(all_specs.values())
        base_names = [a.name for a in specs[0].animations]
        assert base_names == EXPECTED_ANIMATION_NAMES
        for spec in specs[1:]:
            names = [a.name for a in spec.animations]
            assert names == base_names

    def test_spritesheet_dimensions_consistent(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            assert spec.sheet_width == 896, f"{name} width: {spec.sheet_width}"
            assert spec.sheet_height == 1024, f"{name} height: {spec.sheet_height}"

    def test_all_characters_have_descriptions(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            assert spec.character.description.strip(), f"{name} has empty description"

    def test_all_characters_have_base_image(
        self, all_specs: dict[str, SpritesheetSpec]
    ) -> None:
        for name, spec in all_specs.items():
            assert spec.base_image_path, f"{name} has no base_image_path"
            assert spec.base_image_path.startswith("docs_assets/")
            assert spec.base_image_path.endswith(".png")


# ── Template and example config tests ────────────────────────────────


EXAMPLES_DIR = CONFIGS_DIR / "examples"


class TestTemplateConfig:
    """Tests for the annotated template config file."""

    def test_template_is_valid_yaml(self) -> None:
        """configs/template.yaml must parse as valid YAML."""
        with open(CONFIGS_DIR / "template.yaml", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)
        assert "character" in data
        assert "animations" in data
        assert "palette" in data


class TestExampleConfigs:
    """Tests for the example config files in configs/examples/."""

    def test_simple_enemy_example_loads(self) -> None:
        """configs/examples/simple_enemy.yaml loads via load_config()."""
        spec = load_config(EXAMPLES_DIR / "simple_enemy.yaml")
        assert isinstance(spec, SpritesheetSpec)

    def test_hero_example_loads(self) -> None:
        """configs/examples/hero.yaml loads via load_config()."""
        spec = load_config(EXAMPLES_DIR / "hero.yaml")
        assert isinstance(spec, SpritesheetSpec)

    def test_minimal_generated_example_loads(self) -> None:
        """configs/examples/minimal_generated.yaml loads via load_config()."""
        spec = load_config(EXAMPLES_DIR / "minimal_generated.yaml")
        assert isinstance(spec, SpritesheetSpec)

    def test_simple_enemy_has_minimal_animations(self) -> None:
        """Simple enemy must have at least 3 animation rows."""
        spec = load_config(EXAMPLES_DIR / "simple_enemy.yaml")
        assert len(spec.animations) >= 3

    def test_hero_has_full_animations(self) -> None:
        """Hero example must have at least 10 animation rows."""
        spec = load_config(EXAMPLES_DIR / "hero.yaml")
        assert len(spec.animations) >= 10

    def test_minimal_generated_has_minimal_animations(self) -> None:
        """Minimal generated example keeps the smallest practical action set."""
        spec = load_config(EXAMPLES_DIR / "minimal_generated.yaml")
        assert len(spec.animations) == 3
        assert [animation.row for animation in spec.animations] == [0, 1, 2]

    def test_examples_have_palette(self) -> None:
        """Both examples must have a palette section."""
        for name in ("simple_enemy.yaml", "hero.yaml"):
            spec = load_config(EXAMPLES_DIR / name)
            assert spec.palette is not None, f"{name} has no palette"
            assert len(spec.palette.colors) > 0, f"{name} has no colors"

    def test_examples_have_description(self) -> None:
        """Both examples must have a character description."""
        for name in ("simple_enemy.yaml", "hero.yaml"):
            spec = load_config(EXAMPLES_DIR / name)
            assert spec.character.description.strip(), f"{name} has empty description"

    def test_minimal_generated_uses_auto_palette(self) -> None:
        """Generated minimal example relies on auto_palette instead of a palette block."""
        spec = load_config(EXAMPLES_DIR / "minimal_generated.yaml")
        assert spec.palette is None
        assert spec.generation.auto_palette is True
