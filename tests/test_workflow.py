"""Tests for spriteforge.workflow — full pipeline orchestrator."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from spriteforge.errors import GateError
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
from spriteforge.palette import build_palette_map
from spriteforge.preprocessor import PreprocessResult
from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.retry import RetryManager
from spriteforge.workflow import SpriteForgeWorkflow

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
        palettes={"P1": sample_palette},
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
        palettes={"P1": sample_palette},
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
    palette_map = build_palette_map(palette)

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

    return SpriteForgeWorkflow(
        config=config,
        reference_provider=reference_provider,
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=retry_manager,
        palette_map=palette_map,
        preprocessor=preprocessor,
    )


# ---------------------------------------------------------------------------
# Unit tests: _extract_reference_frame
# ---------------------------------------------------------------------------


class TestExtractReferenceFrame:
    """Tests for SpriteForgeWorkflow._extract_reference_frame()."""

    def test_extract_reference_frame_first(self) -> None:
        """Frame 0 from strip → correct crop coordinates."""
        strip = _make_strip_image(num_frames=6, frame_width=64, frame_height=64)
        result = SpriteForgeWorkflow._extract_reference_frame(
            strip, frame_index=0, frame_width=64, frame_height=64
        )
        # Should produce valid PNG bytes
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_extract_reference_frame_last(self) -> None:
        """Last frame from strip → correct crop."""
        strip = _make_strip_image(num_frames=6, frame_width=64, frame_height=64)
        result = SpriteForgeWorkflow._extract_reference_frame(
            strip, frame_index=5, frame_width=64, frame_height=64
        )
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_extract_reference_frame_out_of_bounds(self) -> None:
        """Index beyond strip width → ValueError."""
        strip = _make_strip_image(num_frames=3, frame_width=64, frame_height=64)
        with pytest.raises(ValueError, match="out of bounds"):
            SpriteForgeWorkflow._extract_reference_frame(
                strip, frame_index=3, frame_width=64, frame_height=64
            )

    def test_extract_reference_frame_negative_index(self) -> None:
        """Negative index → ValueError."""
        strip = _make_strip_image(num_frames=3, frame_width=64, frame_height=64)
        with pytest.raises(ValueError, match="out of bounds"):
            SpriteForgeWorkflow._extract_reference_frame(
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

        async def mock_anchor(*args: Any, **kwargs: Any) -> list[str]:
            call_order.append("anchor")
            return _make_sprite_grid()

        async def mock_frame(*args: Any, **kwargs: Any) -> list[str]:
            call_order.append("frame")
            return _make_sprite_grid()

        gen.generate_anchor_frame = mock_anchor
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
    """All 10 attempts fail → raises GenerationError."""

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

        with pytest.raises(GenerationError, match="failed verification"):
            await wf.run(ref_path, out_path)


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
        gen.generate_anchor_frame = AsyncMock(return_value=_make_sprite_grid())
        gen.generate_frame = AsyncMock(return_value=_make_sprite_grid())

        wf = _build_workflow(single_row_config, sample_palette, grid_generator=gen)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # generate_frame is called for frames 1 and 2 (after anchor)
        # 3 frames total, anchor is frame 0 via generate_anchor_frame
        assert gen.generate_frame.call_count == 2

        # First call (frame 1) should have prev_frame_grid set (from anchor)
        first_call = gen.generate_frame.call_args_list[0]
        assert first_call.kwargs.get("prev_frame_grid") is not None
        assert first_call.kwargs.get("prev_frame_rendered") is not None

        # Second call (frame 2) should also have prev_frame_grid set
        second_call = gen.generate_frame.call_args_list[1]
        assert second_call.kwargs.get("prev_frame_grid") is not None


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
        gen.generate_anchor_frame = AsyncMock(return_value=anchor_grid)
        gen.generate_frame = AsyncMock(return_value=_make_sprite_grid())

        wf = _build_workflow(multi_row_config, sample_palette, grid_generator=gen)
        ref_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
        ref_path = tmp_path / "ref.png"
        ref_img.save(str(ref_path))
        out_path = tmp_path / "out.png"

        await wf.run(ref_path, out_path)

        # All generate_frame calls should have anchor_grid set
        for call in gen.generate_frame.call_args_list:
            assert call.kwargs.get("anchor_grid") is not None
            assert call.kwargs.get("anchor_rendered") is not None


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
            palettes={"P1": sample_palette},
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
        # Palette should have been replaced
        assert wf.config.palettes["P1"].name == "auto"


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
        assert wf.config.palettes["P1"].name == "P1"


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
        gen.generate_anchor_frame = AsyncMock(return_value=_make_sprite_grid())
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

        # Check that generate_anchor_frame was called with quantized_reference
        call_kwargs = gen.generate_anchor_frame.call_args.kwargs
        assert call_kwargs.get("quantized_reference") == b"quantized_bytes_here"


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
            palettes={"P1": sample_palette},
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

        # palette_map should have been rebuilt with new colors
        assert wf.palette_map != original_map
        assert wf.palette_map["O"] == (0, 0, 0, 255)
        assert wf.palette_map["s"] == (255, 0, 0, 255)


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
            palettes={"P1": sample_palette},
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

        call_kwargs = gen.generate_anchor_frame.call_args.kwargs
        assert call_kwargs["frame_width"] == 32
        assert call_kwargs["frame_height"] == 32

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
            palettes={"P1": sample_palette},
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
            assert call.kwargs["frame_width"] == 32
            assert call.kwargs["frame_height"] == 32


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

        # Gate 3A was called exactly once (for anchor row)
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

        with pytest.raises(GateError, match="Gate 3A"):
            await wf.run(ref_path, out_path)

        # Gate 3A was called twice (anchor row + second row)
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
            palettes={"P1": sample_palette},
            generation=GenerationConfig(),
        )

        # Track calls to generate_anchor_frame to inspect temperature
        anchor_calls: list[dict[str, Any]] = []

        async def mock_anchor(*args: Any, **kwargs: Any) -> list[str]:
            anchor_calls.append(kwargs)
            return _make_sprite_grid()

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_anchor_frame = mock_anchor
        gen.generate_frame = AsyncMock(return_value=_make_sprite_grid())

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
            palettes={"P1": sample_palette},
            generation=GenerationConfig(),
        )

        anchor_calls: list[dict[str, Any]] = []

        async def mock_anchor(*args: Any, **kwargs: Any) -> list[str]:
            anchor_calls.append(kwargs)
            return _make_sprite_grid()

        gen = AsyncMock(spec=GridGenerator)
        gen.generate_anchor_frame = mock_anchor
        gen.generate_frame = AsyncMock(return_value=_make_sprite_grid())

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
