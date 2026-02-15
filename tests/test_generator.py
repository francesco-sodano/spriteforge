"""Tests for spriteforge.generator — Stage 2 grid generation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from spriteforge.generator import (
    GenerationError,
    GridGenerator,
    _build_palette_map_text,
    _build_system_prompt,
    parse_grid_response,
)
from spriteforge.utils import image_to_data_url
from spriteforge.models import (
    AnimationDef,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
)

from mock_chat_provider import MockChatProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal 1×1 transparent PNG (89 bytes) for use as image bytes in tests.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_valid_grid(symbol: str = ".", rows: int = 64, cols: int = 64) -> list[str]:
    """Create a uniform grid filled with a single symbol."""
    return [symbol * cols for _ in range(rows)]


def _make_valid_json_response(symbol: str = ".") -> str:
    """Create a valid JSON response wrapping a 64×64 grid."""
    grid = _make_valid_grid(symbol)
    return json.dumps({"grid": grid})


@pytest.fixture()
def sample_palette() -> PaletteConfig:
    """A minimal palette for testing."""
    return PaletteConfig(
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
            PaletteColor(element="Hair", symbol="h", r=220, g=185, b=90),
        ],
    )


@pytest.fixture()
def sample_animation() -> AnimationDef:
    """A minimal animation definition for testing."""
    return AnimationDef(
        name="idle",
        row=0,
        frames=6,
        timing_ms=150,
        prompt_context="Character standing in relaxed idle pose",
        frame_descriptions=[
            "Standing neutral",
            "Slight breathing",
            "Blink",
            "Slight sway",
            "Return neutral",
            "Idle loop",
        ],
    )


# ---------------------------------------------------------------------------
# parse_grid_response
# ---------------------------------------------------------------------------


class TestParseGridResponse:
    """Tests for the parse_grid_response function."""

    def test_parse_grid_response_valid_json(self) -> None:
        """Valid JSON → 64×64 grid."""
        response = _make_valid_json_response(".")
        grid = parse_grid_response(response)
        assert len(grid) == 64
        assert all(len(row) == 64 for row in grid)

    def test_parse_grid_response_with_code_fences(self) -> None:
        """```json ... ``` wrapping → still parses."""
        inner = _make_valid_json_response(".")
        response = f"```json\n{inner}\n```"
        grid = parse_grid_response(response)
        assert len(grid) == 64
        assert all(len(row) == 64 for row in grid)

    def test_parse_grid_response_with_plain_fences(self) -> None:
        """``` ... ``` wrapping without json tag → still parses."""
        inner = _make_valid_json_response(".")
        response = f"```\n{inner}\n```"
        grid = parse_grid_response(response)
        assert len(grid) == 64

    def test_parse_grid_response_wrong_row_count(self) -> None:
        """63 rows → GenerationError."""
        grid = _make_valid_grid(".", rows=63)
        response = json.dumps({"grid": grid})
        with pytest.raises(GenerationError, match="64 rows"):
            parse_grid_response(response)

    def test_parse_grid_response_wrong_col_count(self) -> None:
        """Row with 63 chars → GenerationError."""
        grid = _make_valid_grid(".", rows=64, cols=64)
        grid[0] = "." * 63  # One row too short
        response = json.dumps({"grid": grid})
        with pytest.raises(GenerationError, match="64 characters"):
            parse_grid_response(response)

    def test_parse_grid_response_not_json(self) -> None:
        """Plain text → GenerationError."""
        with pytest.raises(GenerationError, match="Failed to parse"):
            parse_grid_response("This is not JSON at all")

    def test_parse_grid_response_missing_grid_key(self) -> None:
        """JSON without 'grid' key → GenerationError."""
        with pytest.raises(GenerationError, match="'grid' key"):
            parse_grid_response('{"data": []}')

    def test_parse_grid_response_non_string_row(self) -> None:
        """Grid row is not a string → GenerationError."""
        grid: list[Any] = _make_valid_grid(".")
        grid[5] = 12345  # Not a string
        response = json.dumps({"grid": grid})
        with pytest.raises(GenerationError, match="not a string"):
            parse_grid_response(response)

    def test_parse_grid_response_grid_not_list(self) -> None:
        """'grid' value is not a list → GenerationError."""
        response = json.dumps({"grid": "not a list"})
        with pytest.raises(GenerationError, match="list of strings"):
            parse_grid_response(response)

    def test_parse_grid_response_custom_dimensions(self) -> None:
        """32×32 grid → parses correctly with matching expected dimensions."""
        grid = _make_valid_grid(".", rows=32, cols=32)
        response = json.dumps({"grid": grid})
        result = parse_grid_response(response, expected_rows=32, expected_cols=32)
        assert len(result) == 32
        assert all(len(row) == 32 for row in result)

    def test_parse_grid_response_128x128(self) -> None:
        """128×128 grid → parses correctly."""
        grid = _make_valid_grid(".", rows=128, cols=128)
        response = json.dumps({"grid": grid})
        result = parse_grid_response(response, expected_rows=128, expected_cols=128)
        assert len(result) == 128
        assert all(len(row) == 128 for row in result)

    def test_parse_grid_response_non_square(self) -> None:
        """Non-square grid (48×32) → parses correctly."""
        grid = _make_valid_grid(".", rows=32, cols=48)
        response = json.dumps({"grid": grid})
        result = parse_grid_response(response, expected_rows=32, expected_cols=48)
        assert len(result) == 32
        assert all(len(row) == 48 for row in result)

    def test_parse_grid_response_wrong_custom_row_count(self) -> None:
        """Grid with 31 rows when expecting 32 → GenerationError."""
        grid = _make_valid_grid(".", rows=31, cols=32)
        response = json.dumps({"grid": grid})
        with pytest.raises(GenerationError, match="32 rows"):
            parse_grid_response(response, expected_rows=32, expected_cols=32)

    def test_parse_grid_response_wrong_custom_col_count(self) -> None:
        """Grid row with 31 chars when expecting 32 → GenerationError."""
        grid = _make_valid_grid(".", rows=32, cols=32)
        grid[0] = "." * 31
        response = json.dumps({"grid": grid})
        with pytest.raises(GenerationError, match="32 characters"):
            parse_grid_response(response, expected_rows=32, expected_cols=32)


# ---------------------------------------------------------------------------
# GridGenerator init
# ---------------------------------------------------------------------------


class TestGridGeneratorInit:
    """Tests for GridGenerator initialization."""

    def test_generator_init_accepts_chat_provider(self) -> None:
        """GridGenerator accepts a ChatProvider."""
        mock = MockChatProvider()
        gen = GridGenerator(chat_provider=mock)
        assert gen._chat is mock

    def test_generator_uses_chat_provider(self) -> None:
        """GridGenerator stores the provided chat provider."""
        mock = MockChatProvider(responses=["test"])
        gen = GridGenerator(chat_provider=mock)
        assert isinstance(gen._chat, MockChatProvider)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_build_palette_map_text(self, sample_palette: PaletteConfig) -> None:
        """Palette map text includes all symbols."""
        text = _build_palette_map_text(sample_palette)
        assert "`.`" in text
        assert "`O`" in text
        assert "`s`" in text
        assert "`h`" in text
        assert "Transparent" in text
        assert "Outline" in text
        assert "Skin" in text
        assert "Hair" in text

    def test_image_to_data_url(self) -> None:
        """Image bytes are base64-encoded in a data URL."""
        url = image_to_data_url(b"\x89PNG")
        assert url.startswith("data:image/png;base64,")
        assert "iVBO" in url  # base64 of \x89PNG starts with iVBO

    def test_build_system_prompt_contains_palette(
        self, sample_palette: PaletteConfig
    ) -> None:
        """System prompt includes palette symbols."""
        gen = GenerationConfig()
        prompt = _build_system_prompt(sample_palette, gen)
        assert "`s`" in prompt
        assert "`h`" in prompt
        assert "`O`" in prompt
        assert "right" in prompt
        assert "56" in prompt

    def test_build_system_prompt_custom_generation(
        self, sample_palette: PaletteConfig
    ) -> None:
        """System prompt uses custom generation config values."""
        gen = GenerationConfig(
            style="Retro 8-bit",
            facing="left",
            feet_row=60,
            rules="Keep arms at sides",
        )
        prompt = _build_system_prompt(sample_palette, gen)
        assert "Retro 8-bit" in prompt
        assert "left" in prompt
        assert "60" in prompt
        assert "Keep arms at sides" in prompt

    def test_build_system_prompt_custom_dimensions(
        self, sample_palette: PaletteConfig
    ) -> None:
        """System prompt uses custom width/height when provided."""
        gen = GenerationConfig()
        prompt = _build_system_prompt(sample_palette, gen, width=32, height=48)
        assert "32×48" in prompt
        assert "32 characters" in prompt or "32" in prompt
        assert "48" in prompt


# ---------------------------------------------------------------------------
# generate_anchor_frame (mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateAnchorFrame:
    """Tests for GridGenerator.generate_anchor_frame (mocked chat provider)."""

    @pytest.fixture()
    def mock_provider(self) -> MockChatProvider:
        return MockChatProvider(responses=[_make_valid_json_response(".")])

    @pytest.fixture()
    def generator(self, mock_provider: MockChatProvider) -> GridGenerator:
        return GridGenerator(chat_provider=mock_provider)

    @pytest.mark.asyncio
    async def test_generate_anchor_frame_sends_images(
        self,
        generator: GridGenerator,
        mock_provider: MockChatProvider,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Verify multimodal request includes base_ref and reference images."""
        grid = await generator.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
        )

        assert len(grid) == 64
        assert len(mock_provider._call_history) == 1

        # Check the messages sent to the LLM
        messages = mock_provider._call_history[0]["messages"]

        # Should have system + user message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # User message should have multimodal content
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # base_reference + reference_frame = 2 images
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_generate_anchor_frame_includes_palette_in_prompt(
        self,
        generator: GridGenerator,
        mock_provider: MockChatProvider,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """System prompt contains palette symbols."""
        await generator.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
        )

        messages = mock_provider._call_history[0]["messages"]
        system_content = messages[0]["content"]

        assert "`s`" in system_content
        assert "`h`" in system_content
        assert "`O`" in system_content
        assert "Skin" in system_content

    @pytest.mark.asyncio
    async def test_generate_anchor_with_quantized_reference(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Quantized image is included in the multimodal request."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)

        grid = await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            quantized_reference=_TINY_PNG,
        )

        assert len(grid) == 64

        messages = mock._call_history[0]["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # base_reference + reference_frame + quantized_reference = 3 images
        assert len(image_parts) == 3

    @pytest.mark.asyncio
    async def test_generate_anchor_without_quantized_reference(
        self,
        generator: GridGenerator,
        mock_provider: MockChatProvider,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Works as before when quantized_reference=None."""
        grid = await generator.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            quantized_reference=None,
        )

        assert len(grid) == 64

        messages = mock_provider._call_history[0]["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # No quantized ref → only 2 images
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_generate_anchor_prompt_includes_tracing_instructions(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Prompt contains 'trace'/'refine' language when quantized ref provided."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)

        await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            quantized_reference=_TINY_PNG,
        )

        messages = mock._call_history[0]["messages"]
        user_content = messages[1]["content"]
        text_parts = [p["text"] for p in user_content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert "trac" in full_text.lower()  # "tracing" or "trace"
        assert "refine" in full_text.lower()

    @pytest.mark.asyncio
    async def test_generate_anchor_frame_custom_dimensions(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Anchor frame with 32×32 dimensions → correct grid size and prompt."""
        grid_32 = _make_valid_grid(".", rows=32, cols=32)
        response = json.dumps({"grid": grid_32})
        mock = MockChatProvider(responses=[response])
        gen = GridGenerator(chat_provider=mock)

        grid = await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_width=32,
            frame_height=32,
        )

        assert len(grid) == 32
        assert all(len(row) == 32 for row in grid)

        # System prompt should reference 32×32 dimensions
        messages = mock._call_history[0]["messages"]
        system_content = messages[0]["content"]
        assert "32×32" in system_content

    @pytest.mark.asyncio
    async def test_generate_anchor_frame_quantized_section_uses_custom_dims(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Quantized section references actual frame dimensions, not hardcoded 64."""
        grid_48 = _make_valid_grid(".", rows=48, cols=48)
        response = json.dumps({"grid": grid_48})
        mock = MockChatProvider(responses=[response])
        gen = GridGenerator(chat_provider=mock)

        await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            quantized_reference=_TINY_PNG,
            frame_width=48,
            frame_height=48,
        )

        messages = mock._call_history[0]["messages"]
        user_content = messages[1]["content"]
        text_parts = [p["text"] for p in user_content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert "48×48" in full_text


# ---------------------------------------------------------------------------
# generate_frame (mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateFrame:
    """Tests for GridGenerator.generate_frame (mocked chat provider)."""

    @pytest.fixture()
    def mock_provider(self) -> MockChatProvider:
        return MockChatProvider(responses=[_make_valid_json_response(".")])

    @pytest.fixture()
    def generator(self, mock_provider: MockChatProvider) -> GridGenerator:
        return GridGenerator(chat_provider=mock_provider)

    @pytest.fixture()
    def anchor_grid(self) -> list[str]:
        return _make_valid_grid(".")

    @pytest.mark.asyncio
    async def test_generate_frame_includes_anchor_image(
        self,
        generator: GridGenerator,
        mock_provider: MockChatProvider,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Request includes rendered anchor image."""
        grid = await generator.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
        )

        assert len(grid) == 64

        messages = mock_provider._call_history[0]["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # reference_frame + anchor_rendered = 2 images
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_generate_frame_includes_prev_frame_when_provided(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Previous frame image is sent when provided."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)
        anchor_grid = _make_valid_grid(".")
        prev_grid = _make_valid_grid(".")

        grid = await gen.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=2,
            prev_frame_grid=prev_grid,
            prev_frame_rendered=_TINY_PNG,
        )

        assert len(grid) == 64

        messages = mock._call_history[0]["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # reference_frame + anchor_rendered + prev_frame = 3 images
        assert len(image_parts) == 3

        # Check prev frame grid is in text context
        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert "Previous frame" in full_text

    @pytest.mark.asyncio
    async def test_generate_frame_omits_prev_frame_when_none(
        self,
        generator: GridGenerator,
        mock_provider: MockChatProvider,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """No previous frame → not in request."""
        grid = await generator.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
            prev_frame_grid=None,
            prev_frame_rendered=None,
        )

        assert len(grid) == 64

        messages = mock_provider._call_history[0]["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # Only reference_frame + anchor_rendered = 2 images
        assert len(image_parts) == 2

        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert "Previous frame" not in full_text

    @pytest.mark.asyncio
    async def test_generate_frame_uses_temperature(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Temperature parameter passed to chat provider."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)
        anchor_grid = _make_valid_grid(".")

        await gen.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
            temperature=0.3,
        )

        assert mock._call_history[0]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_generate_frame_with_additional_guidance(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Extra guidance appears in prompt."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)
        anchor_grid = _make_valid_grid(".")

        guidance = "Focus on arm position and ensure sword is visible"

        await gen.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
            additional_guidance=guidance,
        )

        messages = mock._call_history[0]["messages"]
        content = messages[1]["content"]
        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert guidance in full_text

    @pytest.mark.asyncio
    async def test_generate_frame_no_quantized_reference(
        self,
        generator: GridGenerator,
        mock_provider: MockChatProvider,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Regular frames don't receive quantized reference (only anchor does)."""
        await generator.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
        )

        messages = mock_provider._call_history[0]["messages"]
        content = messages[1]["content"]
        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        # No "trace" or quantized reference language
        assert "quantized" not in full_text.lower()

    @pytest.mark.asyncio
    async def test_generate_frame_custom_dimensions(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Frame generation with 32×32 dimensions → correct grid size and prompt."""
        grid_32 = _make_valid_grid(".", rows=32, cols=32)
        response = json.dumps({"grid": grid_32})
        mock = MockChatProvider(responses=[response])
        gen = GridGenerator(chat_provider=mock)
        anchor_grid_32 = _make_valid_grid(".", rows=32, cols=32)

        grid = await gen.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid_32,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
            frame_width=32,
            frame_height=32,
        )

        assert len(grid) == 32
        assert all(len(row) == 32 for row in grid)

        # System prompt should reference 32×32
        messages = mock._call_history[0]["messages"]
        system_content = messages[0]["content"]
        assert "32×32" in system_content

    @pytest.mark.asyncio
    async def test_generate_frame_non_square_dimensions(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Frame generation with non-square 48×32 dimensions → correct grid."""
        grid_ns = _make_valid_grid(".", rows=32, cols=48)
        response = json.dumps({"grid": grid_ns})
        mock = MockChatProvider(responses=[response])
        gen = GridGenerator(chat_provider=mock)
        anchor_grid_ns = _make_valid_grid(".", rows=32, cols=48)

        grid = await gen.generate_frame(
            reference_frame=_TINY_PNG,
            anchor_grid=anchor_grid_ns,
            anchor_rendered=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            frame_index=1,
            frame_width=48,
            frame_height=32,
        )

        assert len(grid) == 32
        assert all(len(row) == 48 for row in grid)


# ---------------------------------------------------------------------------
# Anchor frame: temperature & additional_guidance (bug fix tests)
# ---------------------------------------------------------------------------


class TestAnchorFrameRetryParams:
    """Tests for temperature and additional_guidance in generate_anchor_frame."""

    @pytest.mark.asyncio
    async def test_anchor_frame_uses_custom_temperature(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Pass temperature=0.5 → verify chat is called with temperature=0.5."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)

        await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            temperature=0.5,
        )

        assert mock._call_history[0]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_anchor_frame_uses_additional_guidance(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Pass non-empty guidance → verify it appears in the prompt."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)

        guidance = "Focus on arm position and ensure sword is visible"

        await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
            additional_guidance=guidance,
        )

        messages = mock._call_history[0]["messages"]
        content = messages[1]["content"]
        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert guidance in full_text

    @pytest.mark.asyncio
    async def test_anchor_frame_defaults_to_temp_1(
        self,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """No temperature param → verify default 1.0 behavior preserved."""
        mock = MockChatProvider(responses=[_make_valid_json_response(".")])
        gen = GridGenerator(chat_provider=mock)

        await gen.generate_anchor_frame(
            base_reference=_TINY_PNG,
            reference_frame=_TINY_PNG,
            palette=sample_palette,
            animation=sample_animation,
        )

        assert mock._call_history[0]["temperature"] == 1.0

    def test_anchor_and_frame_signatures_match(self) -> None:
        """Both methods accept temperature and additional_guidance."""
        import inspect

        anchor_sig = inspect.signature(GridGenerator.generate_anchor_frame)
        frame_sig = inspect.signature(GridGenerator.generate_frame)

        anchor_params = anchor_sig.parameters
        frame_params = frame_sig.parameters

        assert "temperature" in anchor_params
        assert "temperature" in frame_params
        assert "additional_guidance" in anchor_params
        assert "additional_guidance" in frame_params

        # Both should default to the same values
        assert anchor_params["temperature"].default == 1.0
        assert frame_params["temperature"].default == 1.0
        assert anchor_params["additional_guidance"].default == ""
        assert frame_params["additional_guidance"].default == ""
