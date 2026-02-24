"""Tests for spriteforge CLI entry point and programmatic API."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from spriteforge import __main__
from spriteforge.app import run_spriteforge
from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    SpritesheetSpec,
)


def _write_minimal_config(path: Path, *, include_base_image_path: bool = False) -> None:
    base_image_line = 'base_image_path: "base.png"\n' if include_base_image_path else ""
    path.write_text(
        "\n".join(
            [
                "character:",
                '  name: "Test Hero"',
                '  class: "Warrior"',
                '  description: "Test character"',
                "  frame_size: [64, 64]",
                "  spritesheet_columns: 4",
                "",
                "animations:",
                '  - name: "idle"',
                "    row: 0",
                "    frames: 1",
                "    timing_ms: 100",
                "",
                base_image_line,
            ]
        ),
        encoding="utf-8",
    )


def _make_spec(*, name: str = "Test Hero") -> SpritesheetSpec:
    return SpritesheetSpec(
        character=CharacterConfig(
            name=name,
            character_class="Warrior",
            description="Test",
            frame_width=64,
            frame_height=64,
            spritesheet_columns=4,
        ),
        animations=[
            AnimationDef(
                name="idle",
                row=0,
                frames=1,
                timing_ms=100,
                prompt_context="Idle",
            )
        ],
        generation=GenerationConfig(),
    )


def _write_test_png(path: Path) -> None:
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(path)


def test_help_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["spriteforge", "--help"])
    with pytest.raises(SystemExit) as exc:
        __main__.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--base-image" in out
    assert "--auto-palette" in out


def test_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["spriteforge"])
    with pytest.raises(SystemExit) as exc:
        __main__.main()
    assert exc.value.code == 2


def test_dry_run_validates_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "valid.yaml"
    _write_minimal_config(config_path)

    parser = __main__.build_parser()
    args = parser.parse_args(["--config", str(config_path), "--dry-run"])

    exit_code = asyncio.run(__main__.async_main(args))

    assert exit_code == 0
    assert "Config valid: Test Hero, 1 animation" in capsys.readouterr().out


def test_dry_run_invalid_config(capsys: pytest.CaptureFixture[str]) -> None:
    parser = __main__.build_parser()
    args = parser.parse_args(["--config", "missing.yaml", "--dry-run"])

    exit_code = asyncio.run(__main__.async_main(args))

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err


def test_missing_base_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "valid.yaml"
    _write_minimal_config(config_path)

    parser = __main__.build_parser()
    args = parser.parse_args(
        ["--config", str(config_path), "--base-image", str(tmp_path / "missing.png")]
    )

    exit_code = asyncio.run(__main__.async_main(args))

    assert exit_code == 1
    assert "Base image not found" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_full_run_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.png"
    _write_test_png(base)
    output = tmp_path / "sheet.png"

    spec = _make_spec()
    monkeypatch.setattr(__main__, "load_config", lambda _path: spec)

    class DummyWorkflow:
        async def __aenter__(self) -> "DummyWorkflow":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def run(self, **_kwargs: object) -> Path:
            return output

    async def fake_create_workflow(**_kwargs: object) -> DummyWorkflow:
        return DummyWorkflow()

    monkeypatch.setattr(__main__, "create_workflow", fake_create_workflow)

    args = Namespace(
        config=tmp_path / "config.yaml",
        base_image=base,
        output=output,
        auto_palette=False,
        max_colors=16,
        debug=False,
        verbose=False,
        dry_run=False,
    )

    assert await __main__.async_main(args) == 0


@pytest.mark.asyncio
async def test_output_defaults_to_character_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.png"
    _write_test_png(base)

    spec = _make_spec(name="Unit Hero")
    monkeypatch.setattr(__main__, "load_config", lambda _path: spec)

    run_kwargs: dict[str, object] = {}

    class DummyWorkflow:
        async def __aenter__(self) -> "DummyWorkflow":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def run(self, **kwargs: object) -> Path:
            run_kwargs.update(kwargs)
            return Path("output/unit_hero_spritesheet.png")

    async def fake_create_workflow(**_kwargs: object) -> DummyWorkflow:
        return DummyWorkflow()

    monkeypatch.setattr(__main__, "create_workflow", fake_create_workflow)

    args = Namespace(
        config=tmp_path / "config.yaml",
        base_image=base,
        output=None,
        auto_palette=False,
        max_colors=16,
        debug=False,
        verbose=False,
        dry_run=False,
    )

    assert await __main__.async_main(args) == 0
    assert run_kwargs["output_path"] == Path("output/unit_hero_spritesheet.png")


def test_verbose_enables_debug_logging() -> None:
    __main__.configure_logging(verbose=True)
    assert __main__.logging.getLogger("spriteforge").level == __main__.logging.DEBUG


@pytest.mark.asyncio
async def test_auto_palette_flag_sets_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.png"
    _write_test_png(base)

    spec = _make_spec()
    monkeypatch.setattr(__main__, "load_config", lambda _path: spec)

    class DummyWorkflow:
        async def __aenter__(self) -> "DummyWorkflow":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def run(self, **_kwargs: object) -> Path:
            return Path("output/test_hero_spritesheet.png")

    async def fake_create_workflow(**_kwargs: object) -> DummyWorkflow:
        return DummyWorkflow()

    monkeypatch.setattr(__main__, "create_workflow", fake_create_workflow)

    args = Namespace(
        config=tmp_path / "config.yaml",
        base_image=base,
        output=None,
        auto_palette=True,
        max_colors=16,
        debug=False,
        verbose=False,
        dry_run=False,
    )

    await __main__.async_main(args)
    assert spec.generation.auto_palette is True


@pytest.mark.asyncio
async def test_max_colors_flag_sets_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.png"
    _write_test_png(base)

    spec = _make_spec()
    monkeypatch.setattr(__main__, "load_config", lambda _path: spec)

    class DummyWorkflow:
        async def __aenter__(self) -> "DummyWorkflow":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def run(self, **_kwargs: object) -> Path:
            return Path("output/test_hero_spritesheet.png")

    async def fake_create_workflow(**_kwargs: object) -> DummyWorkflow:
        return DummyWorkflow()

    monkeypatch.setattr(__main__, "create_workflow", fake_create_workflow)

    args = Namespace(
        config=tmp_path / "config.yaml",
        base_image=base,
        output=None,
        auto_palette=False,
        max_colors=12,
        debug=False,
        verbose=False,
        dry_run=False,
    )

    await __main__.async_main(args)
    assert spec.generation.max_palette_colors == 12


@pytest.mark.asyncio
async def test_dry_run_auto_palette_shows_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _make_spec()
    monkeypatch.setattr(__main__, "load_config", lambda _path: spec)

    args = Namespace(
        config=tmp_path / "config.yaml",
        base_image=None,
        output=None,
        auto_palette=True,
        max_colors=16,
        debug=False,
        verbose=False,
        dry_run=True,
    )

    assert await __main__.async_main(args) == 0
    assert "palette: auto (from base image)" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_preprocessor_created_for_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.png"
    _write_test_png(base)

    spec = _make_spec()
    monkeypatch.setattr(__main__, "load_config", lambda _path: spec)

    workflow_kwargs: dict[str, object] = {}

    class DummyWorkflow:
        async def __aenter__(self) -> "DummyWorkflow":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def run(self, **_kwargs: object) -> Path:
            return Path("output/test_hero_spritesheet.png")

    async def fake_create_workflow(**kwargs: object) -> DummyWorkflow:
        workflow_kwargs.update(kwargs)
        return DummyWorkflow()

    monkeypatch.setattr(__main__, "create_workflow", fake_create_workflow)

    args = Namespace(
        config=tmp_path / "config.yaml",
        base_image=base,
        output=None,
        auto_palette=True,
        max_colors=16,
        debug=False,
        verbose=False,
        dry_run=False,
    )

    assert await __main__.async_main(args) == 0
    assert workflow_kwargs["preprocessor"] is __main__.preprocess_reference


@pytest.mark.asyncio
async def test_run_spriteforge_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    base = tmp_path / "base.png"
    _write_test_png(base)

    monkeypatch.setattr("spriteforge.app.load_config", lambda _path: _make_spec())

    class DummyWorkflow:
        async def __aenter__(self) -> "DummyWorkflow":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def run(self, **kwargs: object) -> Path:
            return kwargs["output_path"]  # type: ignore[return-value]

    async def fake_create_workflow(**_kwargs: object) -> DummyWorkflow:
        return DummyWorkflow()

    monkeypatch.setattr("spriteforge.app.create_workflow", fake_create_workflow)

    result = await run_spriteforge(config_path=config, base_image_path=base)
    assert result == Path("output/test_hero_spritesheet.png")


@pytest.mark.asyncio
async def test_run_spriteforge_missing_config(tmp_path: Path) -> None:
    base = tmp_path / "base.png"
    _write_test_png(base)

    with pytest.raises(FileNotFoundError):
        await run_spriteforge(
            config_path=tmp_path / "missing.yaml",
            base_image_path=base,
        )
