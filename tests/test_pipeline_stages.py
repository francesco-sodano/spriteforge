"""Stage-isolation tests for pipeline stage classes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spriteforge.errors import RowGenerationError
from spriteforge.models import AnimationDef, PaletteColor, PaletteConfig
from spriteforge.pipeline_stages import (
    AnchorRowStage,
    PreprocessingStage,
    RemainingRowsStage,
)


def _animation(name: str, row: int) -> AnimationDef:
    return AnimationDef(name=name, row=row, frames=2, timing_ms=100)


def _palette() -> PaletteConfig:
    return PaletteConfig(
        name="P1",
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185)],
    )


def _workflow(
    animations: list[AnimationDef],
    checkpoint_manager: MagicMock | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(animations=animations),
        max_concurrent_rows=0,
        checkpoint_manager=checkpoint_manager,
        row_processor=SimpleNamespace(
            process_row=AsyncMock(),
            _build_frame_context=MagicMock(return_value={}),
        ),
        _build_anchor_recovery_policy=MagicMock(),
        _build_anchor_recovery_state=MagicMock(),
        _regenerate_anchor_row=AsyncMock(),
        _reset_non_anchor_progress=MagicMock(),
    )


def _preprocess_workflow(
    *,
    preprocessor: MagicMock | None,
    auto_palette: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        preprocessor=preprocessor,
        config=SimpleNamespace(
            character=SimpleNamespace(frame_width=64, frame_height=64),
            generation=SimpleNamespace(
                max_palette_colors=16, semantic_labels=True, auto_palette=auto_palette
            ),
        ),
    )


def _anchor_workflow(
    *,
    animations: list[AnimationDef],
    checkpoint_manager: MagicMock | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(animations=animations),
        checkpoint_manager=checkpoint_manager,
        row_processor=SimpleNamespace(
            _build_frame_context=MagicMock(return_value={}),
            process_anchor_row=AsyncMock(),
        ),
    )


def _grid(fill: str = ".") -> list[str]:
    return [fill * 64 for _ in range(64)]


def _grids(count: int, fill: str = ".") -> list[list[str]]:
    return [_grid(fill) for _ in range(count)]


def test_preprocessing_stage_no_preprocessor_returns_inputs() -> None:
    workflow = _preprocess_workflow(preprocessor=None, auto_palette=False)
    stage = PreprocessingStage(
        workflow=workflow, base_reference_path="base.png", progress_callback=None
    )

    palette = _palette()
    palette_map = {"O": (20, 40, 40, 255)}

    out_palette, out_palette_map, quantized = stage.execute(palette, palette_map)

    assert out_palette == palette
    assert out_palette_map == palette_map
    assert quantized is None


def test_preprocessing_stage_updates_quantized_only_without_auto_palette() -> None:
    preprocessor = MagicMock(
        return_value=SimpleNamespace(
            quantized_png_bytes=b"quantized",
            palette=_palette(),
        )
    )
    workflow = _preprocess_workflow(preprocessor=preprocessor, auto_palette=False)
    progress = MagicMock()
    stage = PreprocessingStage(
        workflow=workflow, base_reference_path="base.png", progress_callback=progress
    )

    palette = _palette()
    palette_map = {"O": (20, 40, 40, 255)}
    out_palette, out_palette_map, quantized = stage.execute(palette, palette_map)

    assert out_palette == palette
    assert out_palette_map == palette_map
    assert quantized == b"quantized"
    progress.assert_any_call("preprocessing", 0, 1)
    progress.assert_any_call("preprocessing", 1, 1)


def test_preprocessing_stage_auto_palette_replaces_palette_and_map() -> None:
    extracted_palette = PaletteConfig(
        name="P2",
        outline=PaletteColor(element="Outline", symbol="O", r=1, g=2, b=3),
        colors=[PaletteColor(element="Armor", symbol="a", r=10, g=20, b=30)],
    )
    preprocessor = MagicMock(
        return_value=SimpleNamespace(
            quantized_png_bytes=b"quantized",
            palette=extracted_palette,
        )
    )
    workflow = _preprocess_workflow(preprocessor=preprocessor, auto_palette=True)
    stage = PreprocessingStage(
        workflow=workflow, base_reference_path="base.png", progress_callback=None
    )

    out_palette, out_palette_map, quantized = stage.execute(
        _palette(), {"O": (0, 0, 0, 255)}
    )

    assert out_palette == extracted_palette
    assert quantized == b"quantized"
    assert out_palette_map["a"] == (10, 20, 30, 255)


@pytest.mark.asyncio
async def test_anchor_stage_loads_from_checkpoint() -> None:
    animations = [_animation("idle", 0)]
    checkpoint_manager = MagicMock()
    row0_grids = _grids(2, ".")
    checkpoint_manager.load_row.return_value = (b"row0-strip", row0_grids)
    workflow = _anchor_workflow(
        animations=animations, checkpoint_manager=checkpoint_manager
    )

    stage = AnchorRowStage(
        workflow=workflow, base_reference=b"ref", total_rows=1, progress_callback=None
    )
    row_images: dict[int, bytes] = {}

    with (
        patch("spriteforge.pipeline_stages.render_frame", return_value=object()),
        patch(
            "spriteforge.pipeline_stages.frame_to_png_bytes",
            return_value=b"anchor-rendered",
        ),
    ):
        anchor_animation, anchor_grid, anchor_rendered = await stage.execute(
            palette=_palette(),
            palette_map={"O": (20, 40, 40, 255)},
            quantized_reference=None,
            completed_rows={0},
            row_images=row_images,
        )

    assert anchor_animation.name == "idle"
    assert anchor_grid == row0_grids[0]
    assert anchor_rendered == b"anchor-rendered"
    assert row_images[0] == b"row0-strip"
    workflow.row_processor.process_anchor_row.assert_not_awaited()


@pytest.mark.asyncio
async def test_anchor_stage_generates_and_saves_checkpoint() -> None:
    animations = [_animation("idle", 0)]
    checkpoint_manager = MagicMock()
    workflow = _anchor_workflow(
        animations=animations, checkpoint_manager=checkpoint_manager
    )

    anchor_grid = _grid("A")
    row0_grids = _grids(2, "A")
    workflow.row_processor.process_anchor_row = AsyncMock(
        return_value=(anchor_grid, b"anchor-rendered", row0_grids)
    )

    stage = AnchorRowStage(
        workflow=workflow, base_reference=b"ref", total_rows=1, progress_callback=None
    )
    row_images: dict[int, bytes] = {}

    with (
        patch("spriteforge.pipeline_stages.render_row_strip", return_value=object()),
        patch(
            "spriteforge.pipeline_stages.frame_to_png_bytes", return_value=b"row0-strip"
        ),
    ):
        anchor_animation, out_anchor_grid, out_anchor_rendered = await stage.execute(
            palette=_palette(),
            palette_map={"O": (20, 40, 40, 255)},
            quantized_reference=b"quantized",
            completed_rows=set(),
            row_images=row_images,
        )

    assert anchor_animation.name == "idle"
    assert out_anchor_grid == anchor_grid
    assert out_anchor_rendered == b"anchor-rendered"
    assert row_images[0] == b"row0-strip"
    workflow.row_processor.process_anchor_row.assert_awaited_once()
    checkpoint_manager.save_row.assert_called_once()


@pytest.mark.asyncio
async def test_execute_returns_when_no_non_anchor_rows() -> None:
    animations = [_animation("idle", 0)]
    workflow = _workflow(animations)
    stage = RemainingRowsStage(
        workflow=workflow, base_reference=b"ref", total_rows=1, progress_callback=None
    )

    await stage.execute(
        palette=_palette(),
        palette_map={"O": (20, 40, 40, 255)},
        quantized_reference=None,
        anchor_animation=animations[0],
        anchor_grid=["." * 64 for _ in range(64)],
        anchor_rendered=b"anchor",
        completed_rows={0},
        row_images={0: b"row0"},
    )

    workflow._build_anchor_recovery_policy.assert_not_called()
    workflow.row_processor.process_row.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_uses_checkpoint_for_completed_rows() -> None:
    animations = [_animation("idle", 0), _animation("walk", 1)]
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_row.return_value = (
        b"checkpoint-strip",
        [["." * 64 for _ in range(64)] for _ in range(2)],
    )
    workflow = _workflow(animations, checkpoint_manager=checkpoint_manager)
    workflow._build_anchor_recovery_policy.return_value = MagicMock()
    workflow._build_anchor_recovery_state.return_value = SimpleNamespace(attempts=0)

    stage = RemainingRowsStage(
        workflow=workflow, base_reference=b"ref", total_rows=2, progress_callback=None
    )
    row_images = {0: b"row0"}

    await stage.execute(
        palette=_palette(),
        palette_map={"O": (20, 40, 40, 255)},
        quantized_reference=None,
        anchor_animation=animations[0],
        anchor_grid=["." * 64 for _ in range(64)],
        anchor_rendered=b"anchor",
        completed_rows={0, 1},
        row_images=row_images,
    )

    assert row_images[1] == b"checkpoint-strip"
    workflow.row_processor.process_row.assert_not_awaited()
    checkpoint_manager.load_row.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_execute_raises_when_recovery_not_allowed() -> None:
    animations = [_animation("idle", 0), _animation("walk", 1)]
    workflow = _workflow(animations)
    workflow.row_processor.process_row = AsyncMock(side_effect=RuntimeError("boom"))

    policy = SimpleNamespace(
        max_anchor_regenerations=0,
        decide=lambda **_kwargs: SimpleNamespace(
            should_regenerate_anchor=False,
            failure_ratio=1.0,
        ),
    )
    workflow._build_anchor_recovery_policy.return_value = policy
    workflow._build_anchor_recovery_state.return_value = SimpleNamespace(attempts=0)

    stage = RemainingRowsStage(
        workflow=workflow, base_reference=b"ref", total_rows=2, progress_callback=None
    )

    with pytest.raises(RowGenerationError, match="Failed to generate 1 of 1"):
        await stage.execute(
            palette=_palette(),
            palette_map={"O": (20, 40, 40, 255)},
            quantized_reference=None,
            anchor_animation=animations[0],
            anchor_grid=["." * 64 for _ in range(64)],
            anchor_rendered=b"anchor",
            completed_rows={0},
            row_images={0: b"row0"},
        )


@pytest.mark.asyncio
async def test_execute_regenerates_anchor_then_succeeds() -> None:
    animations = [_animation("idle", 0), _animation("walk", 1)]
    workflow = _workflow(animations)
    row_grids = [["." * 64 for _ in range(64)] for _ in range(2)]
    workflow.row_processor.process_row = AsyncMock(
        side_effect=[RuntimeError("first failure"), row_grids]
    )

    policy = SimpleNamespace(
        max_anchor_regenerations=1,
        decide=lambda **_kwargs: SimpleNamespace(
            should_regenerate_anchor=True,
            failure_ratio=1.0,
        ),
    )
    state = SimpleNamespace(attempts=0)
    workflow._build_anchor_recovery_policy.return_value = policy
    workflow._build_anchor_recovery_state.return_value = state
    workflow._regenerate_anchor_row = AsyncMock(
        return_value=(["A" * 64 for _ in range(64)], b"anchor-new")
    )
    workflow._reset_non_anchor_progress = MagicMock(return_value={0})

    stage = RemainingRowsStage(
        workflow=workflow, base_reference=b"ref", total_rows=2, progress_callback=None
    )
    row_images = {0: b"row0"}

    with (
        patch("spriteforge.pipeline_stages.render_row_strip", return_value=object()),
        patch("spriteforge.pipeline_stages.frame_to_png_bytes", return_value=b"row1"),
    ):
        await stage.execute(
            palette=_palette(),
            palette_map={"O": (20, 40, 40, 255)},
            quantized_reference=None,
            anchor_animation=animations[0],
            anchor_grid=["." * 64 for _ in range(64)],
            anchor_rendered=b"anchor",
            completed_rows={0},
            row_images=row_images,
        )

    assert workflow.row_processor.process_row.await_count == 2
    workflow._regenerate_anchor_row.assert_awaited_once()
    workflow._reset_non_anchor_progress.assert_called_once()
    assert state.attempts == 1
    assert row_images[1] == b"row1"
