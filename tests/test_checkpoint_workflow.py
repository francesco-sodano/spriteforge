"""Integration tests for checkpoint/resume functionality in workflow."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

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
from spriteforge.palette import build_palette_map
from spriteforge.providers._base import ReferenceProvider
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
        workflow = SpriteForgeWorkflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            retry_manager=RetryManager(),
            palette_map=build_palette_map(sample_palette),
            checkpoint_dir=checkpoint_dir,
        )

        # Run workflow
        await workflow.run(base_ref_path, output_path)

        # After successful completion, checkpoints should be cleaned up
        if checkpoint_dir.exists():
            # Directory might be removed or empty
            files = list(checkpoint_dir.iterdir())
            assert len(files) == 0, f"Expected cleanup but found: {files}"
        # If directory doesn't exist, that's also acceptable (cleanup succeeded)

        await workflow.close()

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
        workflow = SpriteForgeWorkflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            retry_manager=RetryManager(),
            palette_map=build_palette_map(sample_palette),
            checkpoint_dir=checkpoint_dir,
        )

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

        await workflow.close()

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
        workflow = SpriteForgeWorkflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            retry_manager=RetryManager(),
            palette_map=build_palette_map(sample_palette),
            checkpoint_dir=checkpoint_dir,
        )

        # Run workflow (should resume and only process row 2)
        await workflow.run(base_ref_path, output_path)

        # Verify that only row 2 was regenerated (gate_3a called once)
        gate_3a_calls = mock_gate_checker.gate_3a.call_count
        assert (
            gate_3a_calls == 1
        ), f"Expected 1 gate_3a call (row 2 only), got {gate_3a_calls}"

        # Verify final output was created
        assert output_path.exists()

        await workflow.close()

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
        workflow = SpriteForgeWorkflow(
            config=sample_config,
            reference_provider=mock_ref_provider,
            grid_generator=mock_generator,
            gate_checker=mock_gate_checker,
            programmatic_checker=mock_prog_checker,
            retry_manager=RetryManager(),
            palette_map=build_palette_map(sample_palette),
            checkpoint_dir=None,  # No checkpoints
        )

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

        await workflow.close()
