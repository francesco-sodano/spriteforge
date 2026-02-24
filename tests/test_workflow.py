"""Tests for spriteforge.workflow — full pipeline orchestrator."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from spriteforge.errors import GateError, RetryExhaustedError
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import GateVerdict, LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GenerationError, GridGenerator
from spriteforge.models import (
    AnimationDef,
    CharacterConfig,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
    SpritesheetSpec,
)
from spriteforge.preprocessor import PreprocessResult
from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.row_processor import RowProcessor
from spriteforge.retry import RetryManager
from spriteforge.workflow import SpriteForgeWorkflow, assemble_final_spritesheet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal 1×1 transparent PNG for use as image bytes.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_valid_grid(fill: str = ".", rows: int = 64, cols: int = 64) -> list[str]:
    """Create a uniform grid filled with a single symbol."""
    return [fill * cols for _ in range(rows)]


def _make_sprite_grid() -> list[str]:
    """Create a minimal valid sprite grid with outline + content."""
    grid = ["." * 64 for _ in range(64)]
    # Place some outline and skin pixels mid-grid
    grid[30] = "." * 20 + "O" * 10 + "s" * 10 + "." * 24
    grid[56] = "." * 20 + "O" * 10 + "." * 34  # Feet area
    return grid


def _make_strip_image(
    num_frames: int = 6,
    frame_width: int = 64,
    frame_height: int = 64,
) -> Image.Image:
    """Create a minimal strip image."""
    width = frame_width * num_frames
    return Image.new("RGBA", (width, frame_height), (0, 0, 0, 0))


def _passing_verdict(gate_name: str = "gate_0") -> GateVerdict:
    """Create a passing GateVerdict."""
    return GateVerdict(
        gate_name=gate_name,
        passed=True,
        confidence=0.9,
        feedback="Looks good.",
    )


def _failing_verdict(gate_name: str = "gate_0") -> GateVerdict:
    """Create a failing GateVerdict."""
    return GateVerdict(
        gate_name=gate_name,
        passed=False,
        confidence=0.3,
        feedback="Quality issue detected.",
    )


@pytest.fixture()
def sample_palette() -> PaletteConfig:
    """A minimal palette for testing."""
    return PaletteConfig(
        name="P1",
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
            PaletteColor(element="Hair", symbol="h", r=220, g=185, b=90),
        ],
    )


@pytest.fixture()
def single_row_config(sample_palette: PaletteConfig) -> SpritesheetSpec:
    """A config with a single animation row."""
    return SpritesheetSpec(
        character=CharacterConfig(
            name="TestChar",
            character_class="Warrior",
            description="A test character",
            frame_width=64,
            frame_height=64,
            spritesheet_columns=14,
        ),
        animations=[
            AnimationDef(
                name="idle",
                row=0,
                frames=3,
                timing_ms=150,
                prompt_context="Standing idle",
            ),
        ],
        palette=sample_palette,
        generation=GenerationConfig(),
    )


@pytest.fixture()
def multi_row_config(sample_palette: PaletteConfig) -> SpritesheetSpec:
    """A config with multiple animation rows."""
    return SpritesheetSpec(
        character=CharacterConfig(
            name="TestChar",
            character_class="Warrior",
            description="A test character",
            frame_width=64,
            frame_height=64,
            spritesheet_columns=14,
        ),
        animations=[
            AnimationDef(
                name="idle",
                row=0,
                frames=3,
                timing_ms=150,
                prompt_context="Standing idle",
            ),
            AnimationDef(
                name="walk",
                row=1,
                frames=4,
                timing_ms=100,
                prompt_context="Walking forward",
            ),
        ],
        palette=sample_palette,
        generation=GenerationConfig(),
    )


def _build_workflow(
    config: SpritesheetSpec,
    palette: PaletteConfig,
    reference_provider: Any = None,
    grid_generator: Any = None,
    gate_checker: Any = None,
    programmatic_checker: Any = None,
    retry_manager: Any = None,
    preprocessor: Any = None,
) -> SpriteForgeWorkflow:
    """Build a SpriteForgeWorkflow with mock dependencies."""
    if reference_provider is None:
        reference_provider = AsyncMock(spec=ReferenceProvider)
        reference_provider.generate_row_strip = AsyncMock(
            return_value=_make_strip_image(6)
        )
        reference_provider.close = AsyncMock()

    if grid_generator is None:
        grid_generator = AsyncMock(spec=GridGenerator)
        grid_generator.generate_anchor_frame = AsyncMock(
            return_value=_make_sprite_grid()
        )
        grid_generator.generate_frame = AsyncMock(return_value=_make_sprite_grid())

    if gate_checker is None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

    if programmatic_checker is None:
        programmatic_checker = MagicMock(spec=ProgrammaticChecker)
        programmatic_checker.run_all = MagicMock(
            return_value=[_passing_verdict("programmatic")]
        )

    if retry_manager is None:
        retry_manager = RetryManager()

    frame_generator = FrameGenerator(
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=retry_manager,
        generation_config=config.generation,
    )
    row_processor = RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=gate_checker,
        reference_provider=reference_provider,
    )

    return SpriteForgeWorkflow(
        config=config,
        row_processor=row_processor,
        preprocessor=preprocessor,
    )


# ---------------------------------------------------------------------------
# Unit tests: _extract_reference_frame
# ---------------------------------------------------------------------------


class TestExtractReferenceFrame:
    """Tests for RowProcessor._extract_reference_frame()."""

    def test_extract_reference_frame_first(self) -> None:
        """Frame 0 from strip → correct crop coordinates."""
        strip = _make_strip_image(num_frames=6, frame_width=64, frame_height=64)
        result = RowProcessor._extract_reference_frame(
            strip, frame_index=0, frame_width=64, frame_height=64
        )
        # Should produce valid PNG bytes
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_extract_reference_frame_last(self) -> None:
        """Last frame from strip → correct crop."""
        strip = _make_strip_image(num_frames=6, frame_width=64, frame_height=64)
        result = RowProcessor._extract_reference_frame(
            strip, frame_index=5, frame_width=64, frame_height=64
        )
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_extract_reference_frame_out_of_bounds(self) -> None:
        """Index beyond strip width → ValueError."""
        strip = _make_strip_image(num_frames=3, frame_width=64, frame_height=64)
        with pytest.raises(ValueError, match="out of bounds"):
            RowProcessor._extract_reference_frame(
                strip, frame_index=3, frame_width=64, frame_height=64
            )

    def test_extract_reference_frame_negative_index(self) -> None:
        """Negative index → ValueError."""
        strip = _make_strip_image(num_frames=3, frame_width=64, frame_height=64)
        with pytest.raises(ValueError, match="out of bounds"):
            RowProcessor._extract_reference_frame(
                strip, frame_index=-1, frame_width=64, frame_height=64
            )


# ---------------------------------------------------------------------------
# Integration tests (mocked providers/generators)
# ---------------------------------------------------------------------------


class TestRunSingleRowHappyPath:
    """Single-row config, all gates pass → spritesheet generated."""

    @pytest.mark.asyncio
    async def test_run_single_row_happy_path(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        wf = _build_workflow(single_row_config, sample_palette)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "base_ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "output" / "spritesheet.png"

        result = await wf.run(ref_path, out_path)

        assert result == out_path
        assert out_path.exists()
        output_img = Image.open(str(out_path))
        # Single row: 14*64 x 1*64
        assert output_img.width == 14 * 64
        assert output_img.height == 64


class TestRunAnchorFirst:
    """Row 0 is always processed before other rows."""

    @pytest.mark.asyncio
    async def test_run_anchor_first(
        self,
        multi_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        call_order: list[str] = []

        gen = AsyncMock(spec=GridGenerator)

        async def mock_frame(*args: Any, **kwargs: Any) -> list[str]:
            is_anchor = kwargs.get("is_anchor", False)
            if is_anchor:
                call_order.append("anchor")
            else:
                call_order.append("frame")
            return _make_sprite_grid()

        gen.generate_frame = mock_frame

        wf = _build_workflow(multi_row_config, sample_palette, grid_generator=gen)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Anchor is generated first
        assert call_order[0] == "anchor"


class TestRunFrameRetryOnGateFailure:
    """Gate fails first attempt → retries with escalated params → succeeds."""

    @pytest.mark.asyncio
    async def test_run_frame_retry_on_gate_failure(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        # Make gate_0 fail on first call, pass on second
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(
            side_effect=[
                _failing_verdict("gate_0"),
                _passing_verdict("gate_0"),
                _passing_verdict("gate_0"),
                _passing_verdict("gate_0"),
                _passing_verdict("gate_0"),
                _passing_verdict("gate_0"),
            ]
        )
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        result = await wf.run(ref_path, out_path)
        assert out_path.exists()


class TestRunFrameExhaustsRetries:
    """All 10 attempts fail → raises RetryExhaustedError."""

    @pytest.mark.asyncio
    async def test_run_frame_exhausts_retries(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        # Make programmatic checker always fail
        prog_checker = MagicMock(spec=ProgrammaticChecker)
        prog_checker.run_all = MagicMock(
            return_value=[_failing_verdict("programmatic")]
        )

        wf = _build_workflow(
            single_row_config,
            sample_palette,
            programmatic_checker=prog_checker,
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        with pytest.raises(RetryExhaustedError, match="failed verification"):
            await wf.run(ref_path, out_path)

    @pytest.mark.asyncio
    async def test_retry_exhausted_error_message(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Verify RetryExhaustedError message includes frame ID, attempts, tier, and failure count."""
        # Make programmatic checker always fail
        prog_checker = MagicMock(spec=ProgrammaticChecker)
        prog_checker.run_all = MagicMock(
            return_value=[_failing_verdict("programmatic")]
        )

        wf = _build_workflow(
            single_row_config,
            sample_palette,
            programmatic_checker=prog_checker,
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        try:
            await wf.run(ref_path, out_path)
            pytest.fail("Expected RetryExhaustedError to be raised")
        except RetryExhaustedError as e:
            error_message = str(e)
            # Verify all required components are in the message
            assert "row0_frame0" in error_message or "Frame" in error_message
            assert "10 attempts" in error_message  # max_attempts
            assert "tier" in error_message.lower()
            assert "constrained" in error_message  # Last tier
            assert "failures:" in error_message.lower()


class TestRunReferenceRetry:
    """Gate -1 fails → reference regenerated."""

    @pytest.mark.asyncio
    async def test_run_reference_retry(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            side_effect=[
                _failing_verdict("gate_minus_1"),
                _passing_verdict("gate_minus_1"),
            ]
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        ref_provider = AsyncMock(spec=ReferenceProvider)
        ref_provider.generate_row_strip = AsyncMock(return_value=_make_strip_image(6))
        ref_provider.close = AsyncMock()

        wf = _build_workflow(
            single_row_config,
            sample_palette,
            gate_checker=gate_checker,
            reference_provider=ref_provider,
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Reference provider was called twice (first fail, second pass)
        assert ref_provider.generate_row_strip.call_count == 2


class TestRunGate3aValidatesRow:
    """After all frames pass, Gate 3A runs on assembled strip."""

    @pytest.mark.asyncio
    async def test_run_gate_3a_validates_row(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Gate 3A was called (once per row)
        assert gate_checker.gate_3a.call_count == 1


class TestRunProgressCallback:
    """Progress callback invoked for each stage."""

    @pytest.mark.asyncio
    async def test_run_progress_callback_called(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        wf = _build_workflow(single_row_config, sample_palette)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        callback_calls: list[tuple[str, int, int]] = []

        def progress(stage: str, current: int, total: int) -> None:
            callback_calls.append((stage, current, total))

        await wf.run(ref_path, out_path, progress_callback=progress)

        # Should have been called for: row(0, 1), row(1, 1), assembly(0, 1), assembly(1, 1)
        stage_names = [c[0] for c in callback_calls]
        assert "row" in stage_names
        assert "assembly" in stage_names
        assert len(callback_calls) >= 3


class TestFinalAssemblyStage:
    """Tests for final assembly stage extraction."""

    @pytest.mark.asyncio
    async def test_assemble_final_spritesheet_finalizes_pipeline(
        self,
        single_row_config: SpritesheetSpec,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "out.png"
        checkpoint_manager = MagicMock()
        callback_calls: list[tuple[str, int, int]] = []

        def progress(stage: str, current: int, total: int) -> None:
            callback_calls.append((stage, current, total))

        row_images = {0: b"row0"}

        with patch("spriteforge.workflow.assemble_spritesheet") as mock_assemble:
            result = await assemble_final_spritesheet(
                row_images=row_images,
                config=single_row_config,
                output_path=out_path,
                checkpoint_manager=checkpoint_manager,
                progress_callback=progress,
            )

        assert result == out_path
        mock_assemble.assert_called_once_with(
            row_images, single_row_config, output_path=out_path
        )
        checkpoint_manager.cleanup.assert_called_once()
        assert callback_calls == [("assembly", 0, 1), ("assembly", 1, 1)]

    @pytest.mark.asyncio
    async def test_run_delegates_to_assemble_final_spritesheet(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        wf = _build_workflow(single_row_config, sample_palette)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        mock_assemble = AsyncMock(return_value=out_path)
        wf.assembler = mock_assemble
        result = await wf.run(ref_path, out_path)

        assert result == out_path
        mock_assemble.assert_awaited_once()
        assert mock_assemble.await_args.kwargs["config"] is single_row_config
        assert mock_assemble.await_args.kwargs["output_path"] == out_path
        assert mock_assemble.await_args.kwargs["checkpoint_manager"] is None


class TestRunFullPipeline:
    """Multi-row config → complete spritesheet output."""

    @pytest.mark.asyncio
    async def test_run_full_pipeline(
        self,
        multi_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        wf = _build_workflow(multi_row_config, sample_palette)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        result = await wf.run(ref_path, out_path)

        assert result == out_path
        assert out_path.exists()
        output_img = Image.open(str(out_path))
        # 2 rows: 14*64 x 2*64
        assert output_img.width == 14 * 64
        assert output_img.height == 2 * 64


class TestPrevFramePassedToGenerator:
    """Frame N receives frame N-1 as context."""

    @pytest.mark.asyncio
    async def test_prev_frame_passed_to_generator(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gen = AsyncMock(spec=GridGenerator)
        gen.generate_frame = AsyncMock(return_value=_make_sprite_grid())

        wf = _build_workflow(single_row_config, sample_palette, grid_generator=gen)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # generate_frame is called for all 3 frames (anchor + frames 1 and 2)
        assert gen.generate_frame.call_count == 3

        # First call (frame 0, anchor) should have is_anchor=True
        first_call = gen.generate_frame.call_args_list[0]
        assert first_call.kwargs.get("is_anchor") is True

        # Second call (frame 1) should have prev_frame_grid set (from anchor)
        second_call = gen.generate_frame.call_args_list[1]
        assert second_call.kwargs.get("prev_frame_grid") is not None
        assert second_call.kwargs.get("prev_frame_rendered") is not None

        # Third call (frame 2) should also have prev_frame_grid and prev_frame_rendered set
        third_call = gen.generate_frame.call_args_list[2]
        assert third_call.kwargs.get("prev_frame_grid") is not None
        assert third_call.kwargs.get("prev_frame_rendered") is not None


class TestAnchorPassedToAllFrames:
    """Every frame generation includes the anchor."""

    @pytest.mark.asyncio
    async def test_anchor_passed_to_all_frames(
        self,
        multi_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gen = AsyncMock(spec=GridGenerator)
        anchor_grid = _make_sprite_grid()
        gen.generate_frame = AsyncMock(return_value=anchor_grid)

        wf = _build_workflow(multi_row_config, sample_palette, grid_generator=gen)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # All non-anchor generate_frame calls should have context with anchor_grid set
        for call in gen.generate_frame.call_args_list:
            is_anchor = call.kwargs.get("is_anchor", False)
            context = call.kwargs.get("context")
            assert context is not None, "context parameter must be present"
            # Non-anchor frames should have anchor_grid and anchor_rendered in context
            if not is_anchor:
                assert (
                    context.anchor_grid is not None
                ), "context.anchor_grid must be set for non-anchor frames"
                assert (
                    context.anchor_rendered is not None
                ), "context.anchor_rendered must be set for non-anchor frames"


# ---------------------------------------------------------------------------
# Preprocessor integration tests
# ---------------------------------------------------------------------------


class TestRunWithPreprocessorAutoPalette:
    """Preprocessor extracts palette → used for pipeline."""

    @pytest.mark.asyncio
    async def test_run_with_preprocessor_auto_palette(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        auto_config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=2,
                    timing_ms=150,
                    prompt_context="Idle",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(auto_palette=True),
        )

        extracted_palette = PaletteConfig(
            name="auto",
            outline=PaletteColor(element="Outline", symbol="O", r=10, g=10, b=10),
            colors=[
                PaletteColor(element="Color 1", symbol="s", r=200, g=200, b=200),
                PaletteColor(element="Color 2", symbol="h", r=100, g=100, b=100),
            ],
        )

        mock_preprocessor = MagicMock()
        mock_preprocessor.return_value = PreprocessResult(
            quantized_image=Image.new("RGBA", (64, 64)),
            palette=extracted_palette,
            quantized_png_bytes=_TINY_PNG,
            original_color_count=100,
            final_color_count=3,
        )

        wf = _build_workflow(
            auto_config, sample_palette, preprocessor=mock_preprocessor
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        mock_preprocessor.assert_called_once()
        # Config should NOT have been mutated — palette stays original
        assert wf.config.palette.name == "P1"


class TestRunWithPreprocessorManualPalette:
    """Preprocessor provides quantized image only → YAML palette used."""

    @pytest.mark.asyncio
    async def test_run_with_preprocessor_manual_palette(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        mock_preprocessor = MagicMock()
        mock_preprocessor.return_value = PreprocessResult(
            quantized_image=Image.new("RGBA", (64, 64)),
            palette=PaletteConfig(name="auto"),
            quantized_png_bytes=_TINY_PNG,
            original_color_count=100,
            final_color_count=10,
        )

        wf = _build_workflow(
            single_row_config, sample_palette, preprocessor=mock_preprocessor
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Original palette should remain (auto_palette=False by default)
        assert wf.config.palette.name == "P1"


class TestRunWithoutPreprocessor:
    """No preprocessor → works as before."""

    @pytest.mark.asyncio
    async def test_run_without_preprocessor(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        wf = _build_workflow(single_row_config, sample_palette)
        assert wf.preprocessor is None

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        result = await wf.run(ref_path, out_path)
        assert result == out_path
        assert out_path.exists()


class TestQuantizedReferencePassedToAnchor:
    """Quantized bytes reach the generator for the anchor frame."""

    @pytest.mark.asyncio
    async def test_quantized_reference_passed_to_anchor_generation(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gen = AsyncMock(spec=GridGenerator)
        gen.generate_frame = AsyncMock(return_value=_make_sprite_grid())

        mock_preprocessor = MagicMock()
        mock_preprocessor.return_value = PreprocessResult(
            quantized_image=Image.new("RGBA", (64, 64)),
            palette=PaletteConfig(name="auto"),
            quantized_png_bytes=b"quantized_bytes_here",
            original_color_count=100,
            final_color_count=10,
        )

        wf = _build_workflow(
            single_row_config,
            sample_palette,
            grid_generator=gen,
            preprocessor=mock_preprocessor,
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Check that the anchor frame generation (first call with is_anchor=True)
        # was called with quantized_reference in context
        anchor_call = None
        for call in gen.generate_frame.call_args_list:
            if call.kwargs.get("is_anchor"):
                anchor_call = call
                break

        assert anchor_call is not None, "Anchor frame call not found"
        call_kwargs = anchor_call.kwargs
        context = call_kwargs.get("context")
        assert context is not None, "context parameter must be present"
        assert context.quantized_reference == b"quantized_bytes_here"


class TestPreprocessorResultReplacesPalette:
    """When auto_palette, palette_map is rebuilt from extracted palette."""

    @pytest.mark.asyncio
    async def test_preprocessor_result_replaces_palette(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        auto_config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=1,
                    timing_ms=150,
                    prompt_context="Idle",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(auto_palette=True),
        )

        new_outline = PaletteColor(element="Outline", symbol="O", r=0, g=0, b=0)
        new_color = PaletteColor(element="Color 1", symbol="s", r=255, g=0, b=0)
        extracted = PaletteConfig(
            name="auto",
            outline=new_outline,
            colors=[new_color],
        )

        mock_preprocessor = MagicMock()
        mock_preprocessor.return_value = PreprocessResult(
            quantized_image=Image.new("RGBA", (64, 64)),
            palette=extracted,
            quantized_png_bytes=_TINY_PNG,
            original_color_count=50,
            final_color_count=2,
        )

        # Need the grid to use only the new palette symbols
        def make_grid_with_new_palette() -> list[str]:
            grid = ["." * 64 for _ in range(64)]
            grid[30] = "." * 20 + "O" * 10 + "s" * 10 + "." * 24
            grid[56] = "." * 20 + "O" * 10 + "." * 34
            return grid

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_anchor_frame = AsyncMock(return_value=make_grid_with_new_palette())
        gen.generate_frame = AsyncMock(return_value=make_grid_with_new_palette())

        wf = _build_workflow(
            auto_config,
            sample_palette,
            grid_generator=gen,
            preprocessor=mock_preprocessor,
        )

        original_map = wf.palette_map.copy()

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # palette_map should NOT have been mutated — instance state is preserved
        assert wf.palette_map == original_map


# ---------------------------------------------------------------------------
# Variable frame dimension tests
# ---------------------------------------------------------------------------


class TestFrameDimensionsPassedToGenerator:
    """Workflow passes frame_width/frame_height from config to generator."""

    @pytest.mark.asyncio
    async def test_custom_dimensions_passed_to_anchor(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Generate with 32×32 config → generator receives frame_width/height."""
        config_32 = SpritesheetSpec(
            character=CharacterConfig(
                name="SmallChar",
                frame_width=32,
                frame_height=32,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=1,
                    timing_ms=150,
                    prompt_context="Standing",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(),
        )

        small_grid = _make_valid_grid(".", rows=32, cols=32)

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_frame = AsyncMock(return_value=small_grid)

        wf = _build_workflow(config_32, sample_palette, grid_generator=gen)

        ref_img = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Find the anchor frame call (is_anchor=True)
        anchor_call = None
        for call in gen.generate_frame.call_args_list:
            if call.kwargs.get("is_anchor"):
                anchor_call = call
                break

        assert anchor_call is not None, "Anchor frame call not found"
        call_kwargs = anchor_call.kwargs
        context = call_kwargs["context"]
        assert context is not None
        assert context.frame_width == 32
        assert context.frame_height == 32

    @pytest.mark.asyncio
    async def test_custom_dimensions_passed_to_frame(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Generate with 32×32 config → generate_frame receives dimensions."""
        config_32 = SpritesheetSpec(
            character=CharacterConfig(
                name="SmallChar",
                frame_width=32,
                frame_height=32,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=3,
                    timing_ms=150,
                    prompt_context="Standing",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(),
        )

        small_grid = _make_valid_grid(".", rows=32, cols=32)

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_anchor_frame = AsyncMock(return_value=small_grid)
        gen.generate_frame = AsyncMock(return_value=small_grid)

        wf = _build_workflow(config_32, sample_palette, grid_generator=gen)

        ref_img = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # generate_frame called for frames 1 and 2
        for call in gen.generate_frame.call_args_list:
            context = call.kwargs["context"]
            assert context is not None
            assert context.frame_width == 32
            assert context.frame_height == 32


# ---------------------------------------------------------------------------
# Gate 3A verdict checking tests
# ---------------------------------------------------------------------------


class TestGate3aFailureIsDetected:
    """Gate 3A failure raises GateError."""

    @pytest.mark.asyncio
    async def test_gate_3a_failure_is_detected(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_failing_verdict("gate_3a"))

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        with pytest.raises(GateError, match="Gate 3A"):
            await wf.run(ref_path, out_path)


class TestGate3aPassContinuesNormally:
    """Gate 3A pass allows pipeline to complete successfully."""

    @pytest.mark.asyncio
    async def test_gate_3a_pass_continues_normally(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        result = await wf.run(ref_path, out_path)

        assert result == out_path
        assert out_path.exists()


class TestGate3aCheckedInAnchorRow:
    """_process_anchor_row inspects Gate 3A verdict."""

    @pytest.mark.asyncio
    async def test_gate_3a_checked_in_anchor_row(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Single-row config only uses _process_anchor_row; Gate 3A failure raises."""
        # Disable retries for this test to verify immediate failure
        single_row_config.generation.gate_3a_max_retries = 0

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_failing_verdict("gate_3a"))

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        with pytest.raises(GateError, match="Gate 3A"):
            await wf.run(ref_path, out_path)

        # Gate 3A was called exactly once (for anchor row, no retries)
        assert gate_checker.gate_3a.call_count == 1


class TestGate3aCheckedInProcessRow:
    """_process_row inspects Gate 3A verdict."""

    @pytest.mark.asyncio
    async def test_gate_3a_checked_in_process_row(
        self,
        multi_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Multi-row config: anchor row passes, second row Gate 3A fails."""
        # Disable retries for this test
        multi_row_config.generation.gate_3a_max_retries = 0

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        # First call (anchor row) passes, second call (_process_row) fails
        gate_checker.gate_3a = AsyncMock(
            side_effect=[
                _passing_verdict("gate_3a"),
                _failing_verdict("gate_3a"),
            ]
        )

        wf = _build_workflow(
            multi_row_config, sample_palette, gate_checker=gate_checker
        )
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        with pytest.raises(GateError, match="Failed to generate"):
            await wf.run(ref_path, out_path)

        # Gate 3A was called twice (anchor row + second row, no retries)
        assert gate_checker.gate_3a.call_count == 2


# ---------------------------------------------------------------------------
# Anchor retry escalation tests (bug fix)
# ---------------------------------------------------------------------------


class TestAnchorRetryEscalatesTemperature:
    """After failures, anchor frame is retried with lower temperature."""

    @pytest.mark.asyncio
    async def test_anchor_retry_escalates_temperature(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """After 3 failures, verify anchor is retried with lower temperature."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                character_class="Warrior",
                description="A test character",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=1,
                    timing_ms=150,
                    prompt_context="Standing idle",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(),
        )

        # Track calls to generate_frame to inspect temperature for anchor calls
        anchor_calls: list[dict[str, Any]] = []

        async def mock_frame(*args: Any, **kwargs: Any) -> list[str]:
            if kwargs.get("is_anchor"):
                anchor_calls.append(kwargs)
            return _make_sprite_grid()

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_frame = mock_frame

        # Programmatic checker fails first 4 times, then passes
        prog_checker = MagicMock(spec=ProgrammaticChecker)
        fail_count = 0

        def prog_side_effect(*args: Any, **kwargs: Any) -> list[GateVerdict]:
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 4:
                return [_failing_verdict("programmatic")]
            return [_passing_verdict("programmatic")]

        prog_checker.run_all = MagicMock(side_effect=prog_side_effect)

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        wf = _build_workflow(
            config,
            sample_palette,
            grid_generator=gen,
            programmatic_checker=prog_checker,
            gate_checker=gate_checker,
        )

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Should have been called 5 times (4 failures + 1 success)
        assert len(anchor_calls) == 5

        # First 3 attempts at soft tier (temp=1.0)
        assert anchor_calls[0].get("temperature") == 1.0
        assert anchor_calls[1].get("temperature") == 1.0
        assert anchor_calls[2].get("temperature") == 1.0

        # Attempt 4+ at guided tier (temp=0.7)
        assert anchor_calls[3].get("temperature") == 0.7
        assert anchor_calls[4].get("temperature") == 0.7


class TestAnchorRetryPassesGuidance:
    """Gate failure feedback is passed as additional_guidance to anchor."""

    @pytest.mark.asyncio
    async def test_anchor_retry_passes_guidance(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Verify gate failure feedback is passed as additional_guidance."""
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                character_class="Warrior",
                description="A test character",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=1,
                    timing_ms=150,
                    prompt_context="Standing idle",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(),
        )

        anchor_calls: list[dict[str, Any]] = []

        async def mock_frame(*args: Any, **kwargs: Any) -> list[str]:
            if kwargs.get("is_anchor"):
                anchor_calls.append(kwargs)
            return _make_sprite_grid()

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_frame = mock_frame

        # Programmatic checker fails first, then passes
        prog_checker = MagicMock(spec=ProgrammaticChecker)
        call_count = 0

        def prog_side_effect(*args: Any, **kwargs: Any) -> list[GateVerdict]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return [_failing_verdict("programmatic")]
            return [_passing_verdict("programmatic")]

        prog_checker.run_all = MagicMock(side_effect=prog_side_effect)

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        wf = _build_workflow(
            config,
            sample_palette,
            grid_generator=gen,
            programmatic_checker=prog_checker,
            gate_checker=gate_checker,
        )

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # First call: no guidance (first attempt)
        assert anchor_calls[0].get("additional_guidance") == ""

        # Second call: guidance should be non-empty (retry with feedback)
        assert anchor_calls[1].get("additional_guidance") != ""


# ---------------------------------------------------------------------------
# Re-entrancy / no-mutation tests (bug fix)
# ---------------------------------------------------------------------------


class TestRunDoesNotMutateConfig:
    """run() must not mutate self.config."""

    @pytest.mark.asyncio
    async def test_run_does_not_mutate_config(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """After run(), self.config should be identical to the original."""
        wf = _build_workflow(single_row_config, sample_palette)

        # Snapshot original config state
        original_palette = wf.config.palette
        original_palette_map = dict(wf.palette_map)

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Config must not have been mutated
        assert wf.config.palette == original_palette
        assert wf.palette_map == original_palette_map


class TestAutoPaletteUsesLocalState:
    """With auto_palette=True, extracted palette is used without mutating self.config."""

    @pytest.mark.asyncio
    async def test_auto_palette_uses_local_state(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        auto_config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=1,
                    timing_ms=150,
                    prompt_context="Idle",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(auto_palette=True),
        )

        extracted_palette = PaletteConfig(
            name="auto",
            outline=PaletteColor(element="Outline", symbol="O", r=10, g=10, b=10),
            colors=[
                PaletteColor(element="Color 1", symbol="s", r=200, g=200, b=200),
                PaletteColor(element="Color 2", symbol="h", r=100, g=100, b=100),
            ],
        )

        mock_preprocessor = MagicMock()
        mock_preprocessor.return_value = PreprocessResult(
            quantized_image=Image.new("RGBA", (64, 64)),
            palette=extracted_palette,
            quantized_png_bytes=_TINY_PNG,
            original_color_count=100,
            final_color_count=3,
        )

        wf = _build_workflow(
            auto_config, sample_palette, preprocessor=mock_preprocessor
        )

        # Snapshot original state
        original_palette_name = wf.config.palette.name
        original_palette_map = dict(wf.palette_map)

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # Config must NOT have been mutated
        assert wf.config.palette.name == original_palette_name
        assert wf.palette_map == original_palette_map


class TestMultipleRunsIndependent:
    """Calling run() multiple times should produce independent results."""

    @pytest.mark.asyncio
    async def test_multiple_runs_independent(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        auto_config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=1,
                    timing_ms=150,
                    prompt_context="Idle",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(auto_palette=True),
        )

        extracted_palette = PaletteConfig(
            name="auto",
            outline=PaletteColor(element="Outline", symbol="O", r=10, g=10, b=10),
            colors=[
                PaletteColor(element="Color 1", symbol="s", r=200, g=200, b=200),
                PaletteColor(element="Color 2", symbol="h", r=100, g=100, b=100),
            ],
        )

        mock_preprocessor = MagicMock()
        mock_preprocessor.return_value = PreprocessResult(
            quantized_image=Image.new("RGBA", (64, 64)),
            palette=extracted_palette,
            quantized_png_bytes=_TINY_PNG,
            original_color_count=100,
            final_color_count=3,
        )

        wf = _build_workflow(
            auto_config, sample_palette, preprocessor=mock_preprocessor
        )

        # Snapshot original config state
        original_palette = wf.config.palette
        original_palette_map = dict(wf.palette_map)

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))

        # Run twice
        out1 = tmp_path / "out1.png"
        await wf.run(ref_path, out1)

        out2 = tmp_path / "out2.png"
        await wf.run(ref_path, out2)

        # Both outputs should exist
        assert out1.exists()
        assert out2.exists()

        # Config and palette_map should still be identical to originals
        assert wf.config.palette == original_palette
        assert wf.palette_map == original_palette_map


# ---------------------------------------------------------------------------
# Gate 3A row-level retry tests
# ---------------------------------------------------------------------------


class TestGate3ARowLevelRetry:
    """Tests for Gate 3A retry mechanism at row level."""

    @pytest.mark.asyncio
    async def test_gate_3a_retry_succeeds_on_second_attempt(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Gate 3A fails once, then succeeds on retry after frame regeneration."""
        # Modify config to enable retries
        single_row_config.generation.gate_3a_max_retries = 2

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        # Gate 3A fails first time with specific feedback, then passes
        gate_checker.gate_3a = AsyncMock(
            side_effect=[
                _failing_verdict("gate_3a"),
                _passing_verdict("gate_3a"),
            ]
        )

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        # Should succeed after retry
        result = await wf.run(ref_path, out_path)
        assert result == out_path
        assert out_path.exists()

        # Gate 3A should have been called twice
        assert gate_checker.gate_3a.call_count == 2

    @pytest.mark.asyncio
    async def test_gate_3a_retry_exhaustion_raises_error(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Gate 3A fails all retries and raises GateError."""
        # Set max retries to 1 for faster test
        single_row_config.generation.gate_3a_max_retries = 1

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        # Gate 3A always fails
        gate_checker.gate_3a = AsyncMock(return_value=_failing_verdict("gate_3a"))

        wf = _build_workflow(
            single_row_config, sample_palette, gate_checker=gate_checker
        )

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        # Should raise GateError after exhausting retries
        with pytest.raises(GateError, match="Gate 3A.*after 2 attempts"):
            await wf.run(ref_path, out_path)

        # Gate 3A should have been called 2 times (initial + 1 retry)
        assert gate_checker.gate_3a.call_count == 2

    @pytest.mark.asyncio
    async def test_gate_3a_retry_identifies_problematic_frames(
        self,
        single_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Gate 3A retry identifies and regenerates problematic frames."""
        single_row_config.generation.gate_3a_max_retries = 2

        # Track which frames are generated
        generated_frames: list[int] = []

        async def track_generation(*args: Any, **kwargs: Any) -> list[str]:
            frame_index = kwargs.get("frame_index", -1)
            generated_frames.append(frame_index)
            return _make_sprite_grid()

        grid_gen = AsyncMock(spec=GridGenerator)
        grid_gen.generate_frame = track_generation

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        # Gate 3A fails first with feedback mentioning frame 2, then passes
        failing_verdict = GateVerdict(
            gate_name="gate_3a",
            passed=False,
            confidence=0.3,
            feedback="Frame 2 has inconsistent character design",
        )
        gate_checker.gate_3a = AsyncMock(
            side_effect=[
                failing_verdict,
                _passing_verdict("gate_3a"),
            ]
        )

        wf = _build_workflow(
            single_row_config,
            sample_palette,
            gate_checker=gate_checker,
            grid_generator=grid_gen,
        )

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        result = await wf.run(ref_path, out_path)
        assert result == out_path

        # Should have generated frames 0, 1, 2 initially, then regenerated frame 2
        # Note: single_row_config has 3 frames (0, 1, 2)
        # The anchor frame (0) is generated first, then 1 and 2
        # On retry, frame 2 should be regenerated
        assert 0 in generated_frames  # anchor
        assert 1 in generated_frames  # first pass
        assert 2 in generated_frames  # first pass
        # Frame 2 should appear at least twice (initial + retry)
        assert generated_frames.count(2) >= 2


class TestParallelRowGracefulFailure:
    """Tests for graceful handling of parallel row failures."""

    @pytest.mark.asyncio
    async def test_one_row_failure_does_not_crash_other_rows(
        self,
        multi_row_config: SpritesheetSpec,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """One row failure doesn't prevent other rows from completing."""
        # Disable retries to test immediate failure
        multi_row_config.generation.gate_3a_max_retries = 0

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        # Gate 3A passes for anchor row, fails for second row
        gate_checker.gate_3a = AsyncMock(
            side_effect=[
                _passing_verdict("gate_3a"),  # anchor row
                _failing_verdict("gate_3a"),  # second row
            ]
        )

        wf = _build_workflow(
            multi_row_config, sample_palette, gate_checker=gate_checker
        )

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        # Should fail overall, but report partial success
        with pytest.raises(GateError, match="Failed to generate 1 of 1.*rows"):
            await wf.run(ref_path, out_path)

        # Gate 3A should have been called for both rows
        assert gate_checker.gate_3a.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_results_preserved_on_failure(
        self,
        sample_palette: PaletteConfig,
        tmp_path: Path,
    ) -> None:
        """Partial results are preserved when some rows fail."""
        # Create a config with 3 rows
        config = SpritesheetSpec(
            character=CharacterConfig(
                name="TestChar",
                character_class="Warrior",
                description="A test character",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=2,
                    timing_ms=150,
                    prompt_context="Standing idle",
                ),
                AnimationDef(
                    name="walk",
                    row=1,
                    frames=2,
                    timing_ms=100,
                    prompt_context="Walking forward",
                ),
                AnimationDef(
                    name="run",
                    row=2,
                    frames=2,
                    timing_ms=80,
                    prompt_context="Running fast",
                ),
            ],
            palette=sample_palette,
            generation=GenerationConfig(gate_3a_max_retries=0),  # Disable retries
        )

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_0 = AsyncMock(return_value=_passing_verdict("gate_0"))
        gate_checker.gate_1 = AsyncMock(return_value=_passing_verdict("gate_1"))
        gate_checker.gate_2 = AsyncMock(return_value=_passing_verdict("gate_2"))

        # Anchor passes, row 1 passes, row 2 fails
        gate_checker.gate_3a = AsyncMock(
            side_effect=[
                _passing_verdict("gate_3a"),  # anchor (idle)
                _passing_verdict("gate_3a"),  # walk
                _failing_verdict("gate_3a"),  # run
            ]
        )

        wf = _build_workflow(config, sample_palette, gate_checker=gate_checker)

        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        # Should fail but report 1 successful non-anchor row
        with pytest.raises(GateError, match="Successfully generated rows: 1"):
            await wf.run(ref_path, out_path)

        # All three rows should have attempted Gate 3A
        assert gate_checker.gate_3a.call_count == 3


# ---------------------------------------------------------------------------
# Integration tests — real Azure AI calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_single_row_integration(
    azure_project_endpoint: str,
    tmp_path: Path,
) -> None:
    """Integration test: generate a single-row spritesheet with real Azure AI.

    Uses real Azure AI Foundry providers to verify end-to-end pipeline works.
    This test is expensive (API calls) and slow (~2-5 minutes), so it only
    generates a minimal spritesheet (1 animation row, 2 frames).

    Auto-skips when AZURE_AI_PROJECT_ENDPOINT is not set or credentials unavailable.
    """
    from spriteforge.providers import AzureChatProvider, GPTImageProvider

    # Create a minimal test palette
    test_palette = PaletteConfig(
        name="TestPalette",
        outline=PaletteColor(element="Outline", symbol="O", r=15, g=30, b=10),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=80, g=160, b=50),
            PaletteColor(element="Eyes", symbol="e", r=200, g=30, b=30),
            PaletteColor(element="Vest", symbol="v", r=110, g=75, b=40),
        ],
    )

    # Create a minimal test config with just 1 row, 2 frames
    config = SpritesheetSpec(
        character=CharacterConfig(
            name="TestEnemy",
            character_class="Enemy",
            description=(
                "Small green goblin with red eyes. Wiry build, 40 pixels tall. "
                "Wears brown vest, carries rusty sword."
            ),
            frame_width=64,
            frame_height=64,
            spritesheet_columns=14,
        ),
        animations=[
            AnimationDef(
                name="idle",
                row=0,
                frames=2,  # Minimal frame count for faster test
                loop=True,
                timing_ms=160,
                prompt_context=(
                    "Standing pose with knees bent, sword held low. "
                    "Slight weight shift side to side."
                ),
            ),
        ],
        palette=test_palette,
        generation=GenerationConfig(
            style="Modern HD pixel art",
            facing="right",
            feet_row=56,
            rules=(
                "64x64 pixel frames. Transparent background. "
                "1px dark outline. Character centered. Feet at y=56."
            ),
        ),
        base_image_path="docs_assets/theron_base_reference.png",
        output_path="output/test_spritesheet.png",
    )

    # Create real Azure providers
    grid_chat_provider = AzureChatProvider(
        project_endpoint=azure_project_endpoint,
        model_deployment_name=os.environ.get(
            "SPRITEFORGE_TEST_GRID_MODEL", config.generation.grid_model
        ),
    )
    gate_chat_provider = AzureChatProvider(
        project_endpoint=azure_project_endpoint,
        model_deployment_name=os.environ.get(
            "SPRITEFORGE_TEST_GATE_MODEL", config.generation.gate_model
        ),
    )
    # GPTImageProvider now reads AZURE_OPENAI_GPT_IMAGE_ENDPOINT from environment
    # and uses DefaultAzureCredential for bearer token auth
    ref_provider = GPTImageProvider(
        model_deployment=os.environ.get(
            "SPRITEFORGE_TEST_REFERENCE_MODEL", config.generation.reference_model
        ),
    )

    frame_generator = FrameGenerator(
        grid_generator=GridGenerator(grid_chat_provider),
        gate_checker=LLMGateChecker(gate_chat_provider),
        programmatic_checker=ProgrammaticChecker(),
        retry_manager=RetryManager(),
        generation_config=config.generation,
    )
    row_processor = RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=frame_generator.gate_checker,
        reference_provider=ref_provider,
    )

    # Create workflow with real components
    workflow = SpriteForgeWorkflow(
        config=config,
        row_processor=row_processor,
    )

    # Run workflow
    output = tmp_path / "test_spritesheet.png"
    try:
        result = await workflow.run(
            base_reference_path="docs_assets/theron_base_reference.png",
            output_path=output,
        )

        # Verify output exists
        assert output.exists(), "Spritesheet PNG should be created"
        assert result == output, "Result path should match output path"

        # Verify it's a valid PNG with expected dimensions
        img = Image.open(output)
        assert img.format == "PNG", "Output should be PNG format"
        assert img.mode == "RGBA", "Output should have alpha channel"

        # Expected dimensions: 1 row × 2 frames = 128×64 (with padding to 14 columns = 896×64)
        expected_width = 64 * 14  # spritesheet_columns
        expected_height = 64 * 1  # 1 row
        assert img.size == (expected_width, expected_height), (
            f"Expected dimensions {expected_width}×{expected_height}, "
            f"got {img.size[0]}×{img.size[1]}"
        )

    finally:
        # Clean up provider resources
        await grid_chat_provider.close()
        await gate_chat_provider.close()
        await ref_provider.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_integration_invalid_config(
    azure_project_endpoint: str,
    tmp_path: Path,
) -> None:
    """Integration test: verify proper error handling with invalid config.

    Tests that the workflow raises appropriate errors when given a config
    with invalid palette symbols or missing required fields.
    """
    from spriteforge.providers import AzureChatProvider, GPTImageProvider

    # Create a config with an invalid palette (symbol conflict)
    with pytest.raises(ValueError, match="Duplicate palette symbol"):
        bad_palette = PaletteConfig(
            name="BadPalette",
            outline=PaletteColor(element="Outline", symbol="O", r=15, g=30, b=10),
            colors=[
                PaletteColor(element="Color1", symbol="s", r=80, g=160, b=50),
                PaletteColor(
                    element="Color2", symbol="s", r=200, g=30, b=30
                ),  # Duplicate!
            ],
        )
        SpritesheetSpec(
            character=CharacterConfig(
                name="BadConfig",
                character_class="Enemy",
                description="Test",
                frame_width=64,
                frame_height=64,
                spritesheet_columns=14,
            ),
            animations=[
                AnimationDef(
                    name="idle",
                    row=0,
                    frames=2,
                    loop=True,
                    timing_ms=160,
                    prompt_context="Standing pose",
                ),
            ],
            palette=bad_palette,
            generation=GenerationConfig(
                style="Pixel art",
                facing="right",
                feet_row=56,
                rules="64x64 frames",
            ),
            base_image_path="docs_assets/theron_base_reference.png",
            output_path="output/bad.png",
        )


# ---------------------------------------------------------------------------
# Unit tests: create_workflow factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_workflow_uses_grid_model(
    single_row_config: SpritesheetSpec,
) -> None:
    """Verify GridGenerator receives a provider with the grid_model deployment."""
    from unittest.mock import AsyncMock, patch

    from spriteforge.workflow import create_workflow

    # Set custom model names in config
    single_row_config.generation.grid_model = "custom-grid-model"
    single_row_config.generation.gate_model = "custom-gate-model"
    single_row_config.generation.reference_model = "custom-ref-model"

    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    with (
        patch("spriteforge.providers.azure_chat.AzureChatProvider") as MockChatProvider,
        patch("spriteforge.providers.gpt_image.GPTImageProvider") as MockImageProvider,
    ):
        mock_grid_provider = AsyncMock()
        mock_gate_provider = AsyncMock()
        MockChatProvider.side_effect = [mock_grid_provider, mock_gate_provider]
        MockImageProvider.return_value = AsyncMock()

        async with await create_workflow(
            single_row_config,
            project_endpoint="https://test.azure.com",
            credential=mock_credential,
        ) as workflow:
            # Verify AzureChatProvider was called with grid_model for the first provider
            calls = MockChatProvider.call_args_list
            assert len(calls) == 2, "Should create two chat providers (grid + gate)"

            # First call should be for grid model
            grid_call = calls[0]
            assert grid_call.kwargs["model_deployment_name"] == "custom-grid-model"
            assert grid_call.kwargs["azure_endpoint"] == "https://test.azure.com"
            assert grid_call.kwargs["credential"] == mock_credential


@pytest.mark.asyncio
async def test_create_workflow_uses_gate_model(
    single_row_config: SpritesheetSpec,
) -> None:
    """Verify LLMGateChecker receives a provider with the gate_model deployment."""
    from unittest.mock import AsyncMock, patch

    from spriteforge.workflow import create_workflow

    single_row_config.generation.grid_model = "custom-grid-model"
    single_row_config.generation.gate_model = "custom-gate-model"
    single_row_config.generation.reference_model = "custom-ref-model"

    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    with (
        patch("spriteforge.providers.azure_chat.AzureChatProvider") as MockChatProvider,
        patch("spriteforge.providers.gpt_image.GPTImageProvider") as MockImageProvider,
    ):
        mock_grid_provider = AsyncMock()
        mock_gate_provider = AsyncMock()
        MockChatProvider.side_effect = [mock_grid_provider, mock_gate_provider]
        MockImageProvider.return_value = AsyncMock()

        async with await create_workflow(
            single_row_config,
            project_endpoint="https://test.azure.com",
            credential=mock_credential,
        ) as workflow:
            # Verify AzureChatProvider was called with gate_model for the second provider
            calls = MockChatProvider.call_args_list
            assert len(calls) == 2, "Should create two chat providers (grid + gate)"

            # Second call should be for gate model
            gate_call = calls[1]
            assert gate_call.kwargs["model_deployment_name"] == "custom-gate-model"
            assert gate_call.kwargs["azure_endpoint"] == "https://test.azure.com"
            assert gate_call.kwargs["credential"] == mock_credential


@pytest.mark.asyncio
async def test_create_workflow_uses_reference_model(
    single_row_config: SpritesheetSpec,
) -> None:
    """Verify GPTImageProvider uses the reference_model deployment."""
    from unittest.mock import AsyncMock, patch

    from spriteforge.workflow import create_workflow

    single_row_config.generation.grid_model = "custom-grid-model"
    single_row_config.generation.gate_model = "custom-gate-model"
    single_row_config.generation.reference_model = "custom-ref-model"

    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    with (
        patch("spriteforge.providers.azure_chat.AzureChatProvider") as MockChatProvider,
        patch("spriteforge.providers.gpt_image.GPTImageProvider") as MockImageProvider,
        patch.dict(os.environ, {"AZURE_OPENAI_GPT_IMAGE_ENDPOINT": ""}),
    ):
        mock_grid_provider = AsyncMock()
        mock_gate_provider = AsyncMock()
        MockChatProvider.side_effect = [mock_grid_provider, mock_gate_provider]
        mock_ref_provider = AsyncMock()
        MockImageProvider.return_value = mock_ref_provider

        async with await create_workflow(
            single_row_config,
            project_endpoint="https://test.azure.com",
            credential=mock_credential,
        ) as workflow:
            # Verify GPTImageProvider was called with reference_model and credential
            MockImageProvider.assert_called_once_with(
                azure_endpoint=None,
                credential=mock_credential,
                model_deployment="custom-ref-model",
            )


@pytest.mark.asyncio
async def test_create_workflow_shared_credential(
    single_row_config: SpritesheetSpec,
) -> None:
    """Verify all providers share the same credential object."""
    from unittest.mock import AsyncMock, patch

    from spriteforge.workflow import create_workflow

    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    with (
        patch("spriteforge.providers.azure_chat.AzureChatProvider") as MockChatProvider,
        patch("spriteforge.providers.gpt_image.GPTImageProvider") as MockImageProvider,
    ):
        mock_grid_provider = AsyncMock()
        mock_gate_provider = AsyncMock()
        MockChatProvider.side_effect = [mock_grid_provider, mock_gate_provider]
        MockImageProvider.return_value = AsyncMock()

        async with await create_workflow(
            single_row_config,
            project_endpoint="https://test.azure.com",
            credential=mock_credential,
        ) as workflow:
            # All chat providers should receive the same credential
            for call in MockChatProvider.call_args_list:
                assert call.kwargs["credential"] == mock_credential

            MockImageProvider.assert_called_once()
            # GPTImageProvider now also receives the shared credential
            assert MockImageProvider.call_args.kwargs["credential"] == mock_credential

        # When user-provided credential, it should not be closed
        mock_credential.close.assert_not_called()


@pytest.mark.asyncio
async def test_create_workflow_fallback_env(
    single_row_config: SpritesheetSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify factory reads AZURE_AI_PROJECT_ENDPOINT when endpoint not provided."""
    from unittest.mock import AsyncMock, patch

    from spriteforge.workflow import create_workflow

    # Set environment variable
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://env-endpoint.azure.com")

    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    with (
        patch("spriteforge.providers.azure_chat.AzureChatProvider") as MockChatProvider,
        patch("spriteforge.providers.gpt_image.GPTImageProvider") as MockImageProvider,
    ):
        mock_grid_provider = AsyncMock()
        mock_gate_provider = AsyncMock()
        MockChatProvider.side_effect = [mock_grid_provider, mock_gate_provider]
        MockImageProvider.return_value = AsyncMock()

        # Don't pass endpoint — should fall back to env var
        async with await create_workflow(
            single_row_config,
            credential=mock_credential,
        ) as workflow:
            # All chat providers should use the env var endpoint
            for call in MockChatProvider.call_args_list:
                assert call.kwargs["azure_endpoint"] == "https://env-endpoint.azure.com"

            MockImageProvider.assert_called_once()
            # GPTImageProvider receives credential and reads its own endpoint
            # from AZURE_OPENAI_GPT_IMAGE_ENDPOINT env var
            assert "credential" in MockImageProvider.call_args.kwargs


@pytest.mark.asyncio
async def test_workflow_close_cleans_all(single_row_config: SpritesheetSpec) -> None:
    """Verify close() calls close() on all providers."""
    from unittest.mock import AsyncMock

    # Create mock providers with close methods
    mock_grid_provider = AsyncMock()
    mock_grid_provider.close = AsyncMock()

    mock_gate_provider = AsyncMock()
    mock_gate_provider.close = AsyncMock()

    mock_ref_provider = AsyncMock()
    mock_ref_provider.close = AsyncMock()

    mock_grid_gen = AsyncMock()
    mock_grid_gen._chat = mock_grid_provider

    mock_gate_checker = AsyncMock()
    mock_gate_checker._chat = mock_gate_provider
    mock_prog_checker = ProgrammaticChecker()
    mock_retry_manager = RetryManager()

    frame_generator = FrameGenerator(
        grid_generator=mock_grid_gen,
        gate_checker=mock_gate_checker,
        programmatic_checker=mock_prog_checker,
        retry_manager=mock_retry_manager,
        generation_config=single_row_config.generation,
    )
    row_processor = RowProcessor(
        config=single_row_config,
        frame_generator=frame_generator,
        gate_checker=mock_gate_checker,
        reference_provider=mock_ref_provider,
    )

    workflow = SpriteForgeWorkflow(
        config=single_row_config,
        row_processor=row_processor,
    )

    # Close workflow
    await workflow.close()

    # Verify all providers were closed
    mock_grid_provider.close.assert_called_once()
    mock_gate_provider.close.assert_called_once()
    mock_ref_provider.close.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_close_with_owned_credential(
    single_row_config: SpritesheetSpec,
) -> None:
    """Verify close() closes owned credentials created by factory."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from spriteforge.workflow import create_workflow

    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    # Create mock azure modules
    mock_azure = SimpleNamespace()
    mock_identity = SimpleNamespace()
    mock_aio = SimpleNamespace()
    mock_aio.DefaultAzureCredential = lambda: mock_credential
    mock_identity.aio = mock_aio
    mock_azure.identity = mock_identity

    with (
        patch.dict(
            sys.modules,
            {
                "azure": mock_azure,
                "azure.identity": mock_identity,
                "azure.identity.aio": mock_aio,
            },
        ),
        patch("spriteforge.providers.azure_chat.AzureChatProvider") as MockChatProvider,
        patch("spriteforge.providers.gpt_image.GPTImageProvider") as MockImageProvider,
    ):
        mock_grid_provider = AsyncMock()
        mock_gate_provider = AsyncMock()
        MockChatProvider.side_effect = [mock_grid_provider, mock_gate_provider]
        MockImageProvider.return_value = AsyncMock()

        # Don't pass credential — factory creates one
        workflow = await create_workflow(
            single_row_config,
            project_endpoint="https://test.azure.com",
        )

        # Close workflow — should close owned credential
        await workflow.close()
        mock_credential.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration test: create_workflow with real Azure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_workflow_real_azure(
    azure_project_endpoint: str,
    azure_credential: Any,
    tmp_path: Path,
) -> None:
    """Integration test: create_workflow with real Azure providers.

    Verifies the factory correctly wires real providers with different
    model deployments for grid generation, gate verification, and
    reference generation.

    Auto-skips when AZURE_AI_PROJECT_ENDPOINT is not set or credentials unavailable.
    """
    from spriteforge.config import load_config
    from spriteforge.workflow import create_workflow

    # Create a minimal test config
    test_palette = PaletteConfig(
        name="TestPalette",
        outline=PaletteColor(element="Outline", symbol="O", r=15, g=30, b=10),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=80, g=160, b=50),
            PaletteColor(element="Eyes", symbol="e", r=200, g=30, b=30),
        ],
    )

    config = SpritesheetSpec(
        character=CharacterConfig(
            name="TestEnemy",
            character_class="Enemy",
            description="Small green goblin with red eyes.",
            frame_width=64,
            frame_height=64,
            spritesheet_columns=14,
        ),
        animations=[
            AnimationDef(
                name="idle",
                row=0,
                frames=2,
                loop=True,
                timing_ms=160,
                prompt_context="Standing pose with slight movement.",
            ),
        ],
        palette=test_palette,
        generation=GenerationConfig(
            style="Modern HD pixel art",
            facing="right",
            feet_row=56,
            rules="64x64 frames. Transparent background. 1px dark outline.",
            grid_model="gpt-5.2",
            gate_model="gpt-5-mini",
            reference_model="gpt-image-1.5",
        ),
        base_image_path="docs_assets/theron_base_reference.png",
        output_path="output/test_factory_spritesheet.png",
    )

    # Create workflow using factory
    async with await create_workflow(
        config=config,
        project_endpoint=azure_project_endpoint,
        credential=azure_credential,
    ) as workflow:
        # Verify workflow was created
        assert workflow is not None
        assert workflow.config == config
        assert workflow.grid_generator is not None
        assert workflow.gate_checker is not None
        assert workflow.reference_provider is not None

        # Verify providers are properly initialized
        assert hasattr(workflow.grid_generator, "_chat")
        assert hasattr(workflow.gate_checker, "_chat")
        assert hasattr(workflow.reference_provider, "close")

        # Could optionally run a minimal generation here, but that's expensive
        # and already covered by test_workflow_single_row_integration
