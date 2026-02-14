"""Tests for spriteforge.generator — Stage 2 grid generation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spriteforge.generator import (
    GenerationError,
    GridGenerator,
    _build_palette_map_text,
    _build_system_prompt,
    _image_to_data_url,
    parse_grid_response,
)
from spriteforge.models import (
    AnimationDef,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
)

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


# ---------------------------------------------------------------------------
# GridGenerator init
# ---------------------------------------------------------------------------


class TestGridGeneratorInit:
    """Tests for GridGenerator initialization."""

    def test_generator_init_from_env(self) -> None:
        """Reads endpoint from AZURE_AI_PROJECT_ENDPOINT."""
        with patch.dict(
            "os.environ", {"AZURE_AI_PROJECT_ENDPOINT": "https://test.endpoint"}
        ):
            gen = GridGenerator()
            assert gen._endpoint == "https://test.endpoint"

    def test_generator_init_explicit_endpoint(self) -> None:
        """Uses provided endpoint over env var."""
        gen = GridGenerator(project_endpoint="https://explicit.endpoint")
        assert gen._endpoint == "https://explicit.endpoint"

    def test_generator_init_default_model(self) -> None:
        """Default model deployment is claude-opus-4-6."""
        gen = GridGenerator(project_endpoint="https://test.endpoint")
        assert gen._model_deployment == "claude-opus-4-6"

    def test_generator_init_custom_model(self) -> None:
        """Custom model deployment name."""
        gen = GridGenerator(
            project_endpoint="https://test.endpoint",
            model_deployment="my-claude",
        )
        assert gen._model_deployment == "my-claude"


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
        url = _image_to_data_url(b"\x89PNG")
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


# ---------------------------------------------------------------------------
# generate_anchor_frame (mocked LLM)
# ---------------------------------------------------------------------------


def _mock_openai_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestGenerateAnchorFrame:
    """Tests for GridGenerator.generate_anchor_frame (mocked Azure API)."""

    @pytest.fixture()
    def generator(self) -> GridGenerator:
        return GridGenerator(project_endpoint="https://test.endpoint")

    @pytest.mark.asyncio
    async def test_generate_anchor_frame_sends_images(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Verify multimodal request includes base_ref and reference images."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            grid = await generator.generate_anchor_frame(
                base_reference=_TINY_PNG,
                reference_frame=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
            )

        assert len(grid) == 64

        # Check the messages sent to the LLM
        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]

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
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """System prompt contains palette symbols."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            await generator.generate_anchor_frame(
                base_reference=_TINY_PNG,
                reference_frame=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
            )

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_content = messages[0]["content"]

        assert "`s`" in system_content
        assert "`h`" in system_content
        assert "`O`" in system_content
        assert "Skin" in system_content

    @pytest.mark.asyncio
    async def test_generate_anchor_with_quantized_reference(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Quantized image is included in the multimodal request."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            grid = await generator.generate_anchor_frame(
                base_reference=_TINY_PNG,
                reference_frame=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                quantized_reference=_TINY_PNG,
            )

        assert len(grid) == 64

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # base_reference + reference_frame + quantized_reference = 3 images
        assert len(image_parts) == 3

    @pytest.mark.asyncio
    async def test_generate_anchor_without_quantized_reference(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Works as before when quantized_reference=None."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            grid = await generator.generate_anchor_frame(
                base_reference=_TINY_PNG,
                reference_frame=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                quantized_reference=None,
            )

        assert len(grid) == 64

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # No quantized ref → only 2 images
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_generate_anchor_prompt_includes_tracing_instructions(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Prompt contains 'trace'/'refine' language when quantized ref provided."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            await generator.generate_anchor_frame(
                base_reference=_TINY_PNG,
                reference_frame=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                quantized_reference=_TINY_PNG,
            )

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_content = messages[1]["content"]
        text_parts = [p["text"] for p in user_content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert "trac" in full_text.lower()  # "tracing" or "trace"
        assert "refine" in full_text.lower()


# ---------------------------------------------------------------------------
# generate_frame (mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateFrame:
    """Tests for GridGenerator.generate_frame (mocked Azure API)."""

    @pytest.fixture()
    def generator(self) -> GridGenerator:
        return GridGenerator(project_endpoint="https://test.endpoint")

    @pytest.fixture()
    def anchor_grid(self) -> list[str]:
        return _make_valid_grid(".")

    @pytest.mark.asyncio
    async def test_generate_frame_includes_anchor_image(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Request includes rendered anchor image."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            grid = await generator.generate_frame(
                reference_frame=_TINY_PNG,
                anchor_grid=anchor_grid,
                anchor_rendered=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                frame_index=1,
            )

        assert len(grid) == 64

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        # reference_frame + anchor_rendered = 2 images
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_generate_frame_includes_prev_frame_when_provided(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Previous frame image is sent when provided."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        prev_grid = _make_valid_grid(".")

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            grid = await generator.generate_frame(
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

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
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
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """No previous frame → not in request."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
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

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
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
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Temperature parameter passed to API."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            await generator.generate_frame(
                reference_frame=_TINY_PNG,
                anchor_grid=anchor_grid,
                anchor_rendered=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                frame_index=1,
                temperature=0.3,
            )

        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_generate_frame_with_additional_guidance(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Extra guidance appears in prompt."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        guidance = "Focus on arm position and ensure sword is visible"

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            await generator.generate_frame(
                reference_frame=_TINY_PNG,
                anchor_grid=anchor_grid,
                anchor_rendered=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                frame_index=1,
                additional_guidance=guidance,
            )

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        content = messages[1]["content"]
        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        assert guidance in full_text

    @pytest.mark.asyncio
    async def test_generate_frame_no_quantized_reference(
        self,
        generator: GridGenerator,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
        anchor_grid: list[str],
    ) -> None:
        """Regular frames don't receive quantized reference (only anchor does)."""
        valid_response = _make_valid_json_response(".")
        mock_response = _mock_openai_response(valid_response)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.close = AsyncMock()

        mock_project = AsyncMock()
        mock_project.get_openai_client = MagicMock(return_value=mock_openai)
        mock_project.close = AsyncMock()

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.ai.projects.aio.AIProjectClient",
                return_value=mock_project,
            ),
        ):
            await generator.generate_frame(
                reference_frame=_TINY_PNG,
                anchor_grid=anchor_grid,
                anchor_rendered=_TINY_PNG,
                palette=sample_palette,
                animation=sample_animation,
                frame_index=1,
            )

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        content = messages[1]["content"]
        text_parts = [p["text"] for p in content if p["type"] == "text"]
        full_text = " ".join(text_parts)
        # No "trace" or quantized reference language
        assert "quantized" not in full_text.lower()
