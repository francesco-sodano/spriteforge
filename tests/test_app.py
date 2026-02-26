"""Tests for __main__ entry point and programmatic app API."""

from __future__ import annotations

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


def test_main_help_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["spriteforge", "--help"])
    with pytest.raises(SystemExit) as exc:
        __main__.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "generate" in out
    assert "validate" in out
    assert "estimate" in out


def test_main_missing_generate_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["spriteforge", "generate"])
    with pytest.raises(SystemExit) as exc:
        __main__.main()
    assert exc.value.code == 2


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


@pytest.mark.asyncio
async def test_run_spriteforge_missing_base_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)

    monkeypatch.setattr("spriteforge.app.load_config", lambda _path: _make_spec())

    with pytest.raises(FileNotFoundError, match="Base image not found"):
        await run_spriteforge(
            config_path=config,
            base_image_path=tmp_path / "missing.png",
        )
