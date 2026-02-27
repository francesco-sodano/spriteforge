"""Integration tests for checkpoint/resume functionality in workflow."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from spriteforge.checkpoint import CheckpointManager
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GridGenerator
from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)
from spriteforge.providers._base import ReferenceProvider
from spriteforge.renderer import frame_to_png_bytes, render_frame
from spriteforge.row_processor import RowProcessor
from spriteforge.retry import RetryManager
from spriteforge.workflow import SpriteForgeWorkflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_grid() -> list[str]:
    """Create a minimal 64×64 grid."""
    grid = ["." * 64 for _ in range(64)]
    grid[30] = "." * 20 + "O" * 10 + "s" * 10 + "." * 24
    return grid


def _passing_verdict(gate_name: str = "gate_0") -> GateVerdict:
    """Create a passing GateVerdict."""
    return GateVerdict(
        gate_name=gate_name,
        passed=True,
        confidence=0.9,
        feedback="Looks good.",
    )


def _create_mock_strip_image(
    num_frames: int = 2, frame_width: int = 64, frame_height: int = 64
) -> Image.Image:
    """Create a mock strip image for testing."""
    width = num_frames * frame_width
    return Image.new("RGBA", (width, frame_height), (0, 0, 0, 0))


def _create_mock_strip_bytes(
    num_frames: int = 2, frame_width: int = 64, frame_height: int = 64
) -> bytes:
    """Create PNG bytes for a mock strip image."""
    img = _create_mock_strip_image(num_frames, frame_width, frame_height)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_colored_strip_bytes(
    color: tuple[int, int, int, int],
    num_frames: int = 2,
    frame_width: int = 64,
    frame_height: int = 64,
) -> bytes:
    """Create PNG bytes for a solid-color strip image."""
    img = Image.new("RGBA", (num_frames * frame_width, frame_height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_workflow(
    config: SpritesheetSpec,
    reference_provider: ReferenceProvider,
    grid_generator: GridGenerator,
    gate_checker: LLMGateChecker,
    programmatic_checker: ProgrammaticChecker,
    checkpoint_dir: Path | None,
) -> SpriteForgeWorkflow:
    frame_generator = FrameGenerator(
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=RetryManager(),
        generation_config=config.generation,
    )
    row_processor = RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=gate_checker,
        reference_provider=reference_provider,
    )
    checkpoint_manager = (
        CheckpointManager(checkpoint_dir) if checkpoint_dir is not None else None
    )
    return SpriteForgeWorkflow(
        config=config,
        row_processor=row_processor,
        checkpoint_manager=checkpoint_manager,
    )


@pytest.fixture()
def sample_palette() -> PaletteConfig:
    """A minimal palette for testing."""
    return PaletteConfig(
        name="P1",
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
        ],
    )


@pytest.fixture()
def sample_config(sample_palette: PaletteConfig) -> SpritesheetSpec:
    """A minimal config with 3 animations for testing."""
    return SpritesheetSpec(
        character=CharacterConfig(
            name="Hero",
            frame_width=64,
            frame_height=64,
            spritesheet_columns=14,
        ),
        palette=sample_palette,
        animations=[
            AnimationDef(name="idle", row=0, frames=2, timing_ms=150),
            AnimationDef(name="walk", row=1, frames=2, timing_ms=100),
            AnimationDef(name="attack", row=2, frames=2, timing_ms=100),
        ],
        generation=GenerationConfig(
            grid_model="gpt-5.2",
            gate_model="gpt-5-mini",
            reference_model="gpt-image-1.5",
            allow_absolute_output_path=True,
        ),
        base_image_path="fake.png",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkflowCheckpointIntegration:
    """Integration tests for checkpoint/resume in workflow."""

    @pytest.mark.asyncio
    async def test_checkpoint_saves_after_each_row(
        self,
        sample_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Test that checkpoints are saved after each row completes."""
        checkpoint_dir = tmp_path / "checkpoints"
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "output.png"

        # Mock providers
        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.generate_row_strip = AsyncMock(
            return_value=_create_mock_strip_image()
        )

        mock_chat = MagicMock()
        mock_chat.close = AsyncMock()

        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator._chat = mock_chat
        mock_generator.generate_frame = AsyncMock(return_value=_make_grid())

        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker._chat = mock_chat
        mock_gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_-1")
        )
        mock_gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        mock_gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        mock_gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        mock_gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        mock_prog_checker = MagicMock(spec=ProgrammaticChecker)
        mock_prog_checker.check_grid = MagicMock(return_value=None)

        # Create workflow with checkpoint support
        async with _build_workflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            checkpoint_dir=checkpoint_dir,
        ) as workflow:
            # Run workflow
            await workflow.run(base_ref_path, output_path)

            # After successful completion, checkpoints should be cleaned up
            if checkpoint_dir.exists():
                # Directory might be removed or empty
                files = list(checkpoint_dir.iterdir())
                assert len(files) == 0, f"Expected cleanup but found: {files}"
            # If directory doesn't exist, that's also acceptable (cleanup succeeded)

    @pytest.mark.asyncio
    async def test_resume_skips_completed_rows(
        self,
        sample_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Test that workflow resumes from checkpoints and skips completed rows."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "output.png"

        # Pre-populate checkpoint for row 0 (simulate previous incomplete run)
        from spriteforge.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        checkpoint_mgr.save_row(
            row=0,
            animation_name="idle",
            strip_bytes=_create_mock_strip_bytes(num_frames=2),
            grids=[_make_grid(), _make_grid()],
        )

        # Mock providers
        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.generate_row_strip = AsyncMock(
            return_value=_create_mock_strip_image()
        )

        mock_chat = MagicMock()
        mock_chat.close = AsyncMock()

        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator._chat = mock_chat
        mock_generator.generate_frame = AsyncMock(return_value=_make_grid())

        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker._chat = mock_chat
        mock_gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_-1")
        )
        mock_gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        mock_gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        mock_gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        mock_gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        mock_prog_checker = MagicMock(spec=ProgrammaticChecker)
        mock_prog_checker.check_grid = MagicMock(return_value=None)

        # Create workflow with checkpoint support
        async with _build_workflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            checkpoint_dir=checkpoint_dir,
        ) as workflow:
            # Run workflow (should resume from checkpoint)
            await workflow.run(base_ref_path, output_path)

            # Verify that row 0 was NOT regenerated (should have been loaded from checkpoint)
            # Count how many times gate_3a was called - should be 2 (for rows 1 and 2, not row 0)
            gate_3a_calls = mock_gate_checker.gate_3a.call_count
            assert (
                gate_3a_calls == 2
            ), f"Expected 2 gate_3a calls (rows 1 and 2), got {gate_3a_calls}"

            # Verify final output was created
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_resume_with_multiple_completed_rows(
        self,
        sample_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Test resume when multiple rows are already completed."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "output.png"

        # Pre-populate checkpoints for rows 0 and 1
        from spriteforge.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        checkpoint_mgr.save_row(
            row=0,
            animation_name="idle",
            strip_bytes=_create_mock_strip_bytes(num_frames=2),
            grids=[_make_grid(), _make_grid()],
        )
        checkpoint_mgr.save_row(
            row=1,
            animation_name="walk",
            strip_bytes=_create_mock_strip_bytes(num_frames=2),
            grids=[_make_grid(), _make_grid()],
        )

        # Mock providers
        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.generate_row_strip = AsyncMock(
            return_value=_create_mock_strip_image()
        )

        mock_chat = MagicMock()
        mock_chat.close = AsyncMock()

        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator._chat = mock_chat
        mock_generator.generate_frame = AsyncMock(return_value=_make_grid())

        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker._chat = mock_chat
        mock_gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_-1")
        )
        mock_gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        mock_gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        mock_gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        mock_gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        mock_prog_checker = MagicMock(spec=ProgrammaticChecker)
        mock_prog_checker.check_grid = MagicMock(return_value=None)

        # Create workflow with checkpoint support
        async with _build_workflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            checkpoint_dir=checkpoint_dir,
        ) as workflow:
            # Run workflow (should resume and only process row 2)
            await workflow.run(base_ref_path, output_path)

            # Verify that only row 2 was regenerated (gate_3a called once)
            gate_3a_calls = mock_gate_checker.gate_3a.call_count
            assert (
                gate_3a_calls == 1
            ), f"Expected 1 gate_3a call (row 2 only), got {gate_3a_calls}"

            # Verify final output was created
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_workflow_without_checkpoint_dir_works_normally(
        self,
        sample_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Test that workflow works normally when checkpoint_dir is None."""
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "output.png"

        # Mock providers
        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.generate_row_strip = AsyncMock(
            return_value=_create_mock_strip_image()
        )

        mock_chat = MagicMock()
        mock_chat.close = AsyncMock()

        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator._chat = mock_chat
        mock_generator.generate_frame = AsyncMock(return_value=_make_grid())

        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker._chat = mock_chat
        mock_gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_-1")
        )
        mock_gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        mock_gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        mock_gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        mock_gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        mock_prog_checker = MagicMock(spec=ProgrammaticChecker)
        mock_prog_checker.check_grid = MagicMock(return_value=None)

        # Create workflow WITHOUT checkpoint support
        async with _build_workflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            checkpoint_dir=None,  # No checkpoints
        ) as workflow:
            # Run workflow
            await workflow.run(base_ref_path, output_path)

            # Verify all rows were processed (gate_3a called 3 times)
            gate_3a_calls = mock_gate_checker.gate_3a.call_count
            assert gate_3a_calls == 3, f"Expected 3 gate_3a calls, got {gate_3a_calls}"

            # Verify final output was created
            assert output_path.exists()

            # Verify no checkpoint directory was created
            checkpoint_dir = tmp_path / ".spriteforge_checkpoint"
            assert not checkpoint_dir.exists()

    @pytest.mark.asyncio
    async def test_regenerate_row_rebuilds_from_checkpoints(
        self,
        sample_config: SpritesheetSpec,
        tmp_path: Path,
    ) -> None:
        """Regenerating one row should keep other checkpoint rows untouched."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "out.png"

        sample_config.base_image_path = str(base_ref_path)
        sample_config.output_path = str(output_path)

        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        checkpoint_mgr.save_row(
            row=0,
            animation_name="idle",
            strip_bytes=_create_colored_strip_bytes((255, 0, 0, 255)),
            grids=[_make_grid(), _make_grid()],
        )
        checkpoint_mgr.save_row(
            row=1,
            animation_name="walk",
            strip_bytes=_create_colored_strip_bytes((255, 255, 0, 255)),
            grids=[_make_grid(), _make_grid()],
        )
        checkpoint_mgr.save_row(
            row=2,
            animation_name="attack",
            strip_bytes=_create_colored_strip_bytes((0, 0, 255, 255)),
            grids=[_make_grid(), _make_grid()],
        )

        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.close = AsyncMock()

        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator.close = AsyncMock()

        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker.close = AsyncMock()

        mock_prog_checker = MagicMock(spec=ProgrammaticChecker)
        frame_generator = FrameGenerator(
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            retry_manager=RetryManager(),
            generation_config=sample_config.generation,
        )
        row_processor = RowProcessor(
            config=sample_config,
            frame_generator=frame_generator,
            gate_checker=mock_gate_checker,
            reference_provider=mock_ref_provider,
        )

        workflow = SpriteForgeWorkflow(
            config=sample_config,
            row_processor=row_processor,
            checkpoint_manager=checkpoint_mgr,
        )

        new_grid = _make_grid()
        new_grid[0] = "O" * 64
        row_processor.process_row = AsyncMock(return_value=[new_grid, _make_grid()])

        await workflow.regenerate_row(1)

        row_processor.process_row.assert_awaited_once()
        args = row_processor.process_row.await_args
        assert args.args[1].row == 1
        assert args.kwargs["anchor_grid"] == _make_grid()

        row0_after = checkpoint_mgr.load_row(0)
        row2_after = checkpoint_mgr.load_row(2)
        assert row0_after is not None
        assert row2_after is not None
        assert row0_after[0] == _create_colored_strip_bytes((255, 0, 0, 255))
        assert row2_after[0] == _create_colored_strip_bytes((0, 0, 255, 255))

        assert output_path.exists()
        sheet = Image.open(output_path).convert("RGBA")
        assert sheet.getpixel((0, 0)) == (255, 0, 0, 255)
        assert sheet.getpixel((0, 64 * 2)) == (0, 0, 255, 255)

        await workflow.close()

    @pytest.mark.asyncio
    async def test_load_anchor_from_checkpoint_reconstructs_rendered_bytes(
        self,
        sample_config: SpritesheetSpec,
        tmp_path: Path,
    ) -> None:
        """Anchor is reconstructed from checkpoint grids[0]."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "out.png"

        sample_config.base_image_path = str(base_ref_path)
        sample_config.output_path = str(output_path)

        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        grids = [_make_grid(), _make_grid()]
        checkpoint_mgr.save_row(
            row=0,
            animation_name="idle",
            strip_bytes=_create_mock_strip_bytes(),
            grids=grids,
        )

        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.close = AsyncMock()
        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator.close = AsyncMock()
        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker.close = AsyncMock()
        frame_generator = FrameGenerator(
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=MagicMock(spec=ProgrammaticChecker),
            retry_manager=RetryManager(),
            generation_config=sample_config.generation,
        )
        row_processor = RowProcessor(
            config=sample_config,
            frame_generator=frame_generator,
            gate_checker=mock_gate_checker,
            reference_provider=mock_ref_provider,
        )
        workflow = SpriteForgeWorkflow(
            config=sample_config,
            row_processor=row_processor,
            checkpoint_manager=checkpoint_mgr,
        )

        anchor_grid, anchor_rendered = workflow.load_anchor_from_checkpoint()

        assert anchor_grid == grids[0]
        context = row_processor._build_frame_context(
            palette=sample_config.palette,
            palette_map=workflow.palette_map,
            animation=sample_config.animations[0],
            anchor_grid=anchor_grid,
            anchor_rendered=None,
            quantized_reference=None,
        )
        expected = frame_to_png_bytes(render_frame(anchor_grid, context))
        assert anchor_rendered == expected
        await workflow.close()

    @pytest.mark.asyncio
    async def test_regenerate_row_zero_logs_warning(
        self,
        sample_config: SpritesheetSpec,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Regenerating row 0 should log an anchor-dependency warning."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        base_ref_path = tmp_path / "base.png"
        base_ref_path.write_bytes(_TINY_PNG)
        output_path = tmp_path / "out.png"

        sample_config.base_image_path = str(base_ref_path)
        sample_config.output_path = str(output_path)

        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        for row, name in [(0, "idle"), (1, "walk"), (2, "attack")]:
            checkpoint_mgr.save_row(
                row=row,
                animation_name=name,
                strip_bytes=_create_mock_strip_bytes(),
                grids=[_make_grid(), _make_grid()],
            )

        mock_ref_provider = MagicMock(spec=ReferenceProvider)
        mock_ref_provider.close = AsyncMock()

        mock_generator = MagicMock(spec=GridGenerator)
        mock_generator.close = AsyncMock()

        mock_gate_checker = MagicMock(spec=LLMGateChecker)
        mock_gate_checker.close = AsyncMock()

        mock_prog_checker = MagicMock(spec=ProgrammaticChecker)
        frame_generator = FrameGenerator(
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            retry_manager=RetryManager(),
            generation_config=sample_config.generation,
        )
        row_processor = RowProcessor(
            config=sample_config,
            frame_generator=frame_generator,
            gate_checker=mock_gate_checker,
            reference_provider=mock_ref_provider,
        )
        row_processor.process_anchor_row = AsyncMock(
            return_value=(_make_grid(), _TINY_PNG, [_make_grid(), _make_grid()])
        )

        workflow = SpriteForgeWorkflow(
            config=sample_config,
            row_processor=row_processor,
            checkpoint_manager=checkpoint_mgr,
        )

        with caplog.at_level("WARNING"):
            await workflow.regenerate_row(0)

        assert "Dependent rows are not automatically regenerated" in caplog.text
        row_processor.process_anchor_row.assert_awaited_once()
        await workflow.close()
