"""Tests for spriteforge.row_processor."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from spriteforge.gates import GateVerdict, LLMGateChecker
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
from spriteforge.row_processor import RowProcessor

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_sprite_grid() -> list[str]:
    return ["." * 64 for _ in range(64)]


def _make_strip_image(num_frames: int = 3) -> Image.Image:
    return Image.new("RGBA", (64 * num_frames, 64), (0, 0, 0, 0))


def _passing_verdict(gate_name: str) -> GateVerdict:
    return GateVerdict(gate_name=gate_name, passed=True, confidence=0.9, feedback="ok")


@pytest.fixture()
def sample_palette() -> PaletteConfig:
    return PaletteConfig(
        name="P1",
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185)],
    )


@pytest.fixture()
def sample_config(sample_palette: PaletteConfig) -> SpritesheetSpec:
    return SpritesheetSpec(
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
                frames=3,
                timing_ms=150,
                prompt_context="Standing idle",
            ),
            AnimationDef(
                name="walk",
                row=1,
                frames=3,
                timing_ms=100,
                prompt_context="Walking forward",
            ),
        ],
        palette=sample_palette,
        generation=GenerationConfig(),
    )


def _build_row_processor(
    config: SpritesheetSpec,
    reference_provider: Any = None,
    gate_checker: Any = None,
    frame_generator: Any = None,
) -> RowProcessor:
    if reference_provider is None:
        reference_provider = AsyncMock(spec=ReferenceProvider)
        reference_provider.generate_row_strip = AsyncMock(
            return_value=_make_strip_image(3)
        )

    if gate_checker is None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

    if frame_generator is None:
        frame_generator = MagicMock()
        frame_generator.generate_verified_frame = AsyncMock(
            return_value=_make_sprite_grid()
        )

    return RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=gate_checker,
        reference_provider=reference_provider,
    )


class TestRowProcessorReferenceRetry:
    @pytest.mark.asyncio
    async def test_reference_strip_retry(
        self, sample_config: SpritesheetSpec, sample_palette: PaletteConfig
    ) -> None:
        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            side_effect=[
                GateVerdict(
                    gate_name="gate_minus_1",
                    passed=False,
                    confidence=0.3,
                    feedback="bad reference",
                ),
                _passing_verdict("gate_minus_1"),
            ]
        )
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        ref_provider = AsyncMock(spec=ReferenceProvider)
        ref_provider.generate_row_strip = AsyncMock(return_value=_make_strip_image(3))

        row_processor = _build_row_processor(
            sample_config, reference_provider=ref_provider, gate_checker=gate_checker
        )
        palette_map = build_palette_map(sample_palette)

        await row_processor.process_anchor_row(
            base_reference=_TINY_PNG,
            animation=sample_config.animations[0],
            palette=sample_palette,
            palette_map=palette_map,
            quantized_reference=None,
        )

        assert ref_provider.generate_row_strip.call_count == 2


class TestRowProcessorGate3ARetry:
    @pytest.mark.asyncio
    async def test_gate_3a_retry_regenerates_problematic_frame(
        self, sample_config: SpritesheetSpec, sample_palette: PaletteConfig
    ) -> None:
        generated_frames: list[int] = []

        async def track_generation(*args: Any, **kwargs: Any) -> list[str]:
            generated_frames.append(kwargs.get("frame_index", -1))
            return _make_sprite_grid()

        frame_generator = MagicMock()
        frame_generator.generate_verified_frame = AsyncMock(
            side_effect=track_generation
        )

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_3a = AsyncMock(
            side_effect=[
                GateVerdict(
                    gate_name="gate_3a",
                    passed=False,
                    confidence=0.3,
                    feedback="Frame 2 is inconsistent",
                ),
                _passing_verdict("gate_3a"),
            ]
        )

        row_processor = _build_row_processor(
            sample_config,
            gate_checker=gate_checker,
            frame_generator=frame_generator,
        )
        palette_map = build_palette_map(sample_palette)

        await row_processor.process_anchor_row(
            base_reference=_TINY_PNG,
            animation=sample_config.animations[0],
            palette=sample_palette,
            palette_map=palette_map,
            quantized_reference=None,
        )

        assert 0 in generated_frames
        assert 1 in generated_frames
        assert generated_frames.count(2) >= 2

    def test_identify_problematic_frames_handles_list_feedback(
        self, sample_config: SpritesheetSpec
    ) -> None:
        row_processor = _build_row_processor(sample_config)
        indices = row_processor._identify_problematic_frames(
            "frames 2, 5, and 7 have jitter",
            num_frames=8,
        )
        assert indices == [2, 5, 7]


class TestRowProcessorTimeouts:
    @pytest.mark.asyncio
    async def test_reference_strip_timeout_retries_and_recovers(
        self, sample_config: SpritesheetSpec
    ) -> None:
        sample_config.generation.request_timeout_seconds = 0.01

        gate_checker = AsyncMock(spec=LLMGateChecker)
        gate_checker.gate_minus_1 = AsyncMock(
            return_value=_passing_verdict("gate_minus_1")
        )
        gate_checker.gate_3a = AsyncMock(return_value=_passing_verdict("gate_3a"))

        call_count = 0

        async def flaky_generate(*args: Any, **kwargs: Any) -> Image.Image:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.Event().wait()
            return _make_strip_image(3)

        ref_provider = AsyncMock(spec=ReferenceProvider)
        ref_provider.generate_row_strip = AsyncMock(side_effect=flaky_generate)

        row_processor = _build_row_processor(
            sample_config,
            reference_provider=ref_provider,
            gate_checker=gate_checker,
        )

        strip = await row_processor._generate_reference_strip(
            _TINY_PNG,
            sample_config.animations[0],
        )

        assert strip.size == (64 * 3, 64)
        assert ref_provider.generate_row_strip.call_count == 2


class TestRowProcessorFrameSequencing:
    @pytest.mark.asyncio
    async def test_process_row_passes_prev_frame_context(
        self, sample_config: SpritesheetSpec, sample_palette: PaletteConfig
    ) -> None:
        frame_generator = MagicMock()
        frame_generator.generate_verified_frame = AsyncMock(
            return_value=_make_sprite_grid()
        )

        row_processor = _build_row_processor(
            sample_config,
            frame_generator=frame_generator,
        )
        palette_map = build_palette_map(sample_palette)

        await row_processor.process_row(
            base_reference=_TINY_PNG,
            animation=sample_config.animations[1],
            palette=sample_palette,
            palette_map=palette_map,
            anchor_grid=_make_sprite_grid(),
            anchor_rendered=_TINY_PNG,
        )

        first_call = frame_generator.generate_verified_frame.call_args_list[0]
        second_call = frame_generator.generate_verified_frame.call_args_list[1]
        third_call = frame_generator.generate_verified_frame.call_args_list[2]

        assert first_call.kwargs.get("prev_frame_grid") is None
        assert first_call.kwargs.get("prev_frame_rendered") is None
        assert second_call.kwargs.get("prev_frame_grid") is not None
        assert second_call.kwargs.get("prev_frame_rendered") is not None
        assert third_call.kwargs.get("prev_frame_grid") is not None
        assert third_call.kwargs.get("prev_frame_rendered") is not None
