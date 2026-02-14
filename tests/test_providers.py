"""Tests for the reference image provider module."""

from __future__ import annotations

import io
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from spriteforge.models import AnimationDef, CharacterConfig
from spriteforge.providers import GPTImageProvider, ProviderError, ReferenceProvider
from spriteforge.providers.gpt_image import build_reference_prompt

# ---------------------------------------------------------------------------
# Helper: create a small dummy PNG as bytes
# ---------------------------------------------------------------------------


def _dummy_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal RGBA PNG as bytes."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def walk_animation() -> AnimationDef:
    """An 8-frame WALK animation."""
    return AnimationDef(
        name="WALK",
        row=1,
        frames=8,
        timing_ms=100,
        prompt_context="Character walking forward with a steady gait.",
    )


@pytest.fixture()
def character() -> CharacterConfig:
    """A simple character config."""
    return CharacterConfig(
        name="Theron Ashblade",
        character_class="Warrior",
        description="A battle-scarred warrior with flame-red hair.",
    )


# ---------------------------------------------------------------------------
# Tests: GPTImageProvider initialisation
# ---------------------------------------------------------------------------


class TestGPTImageProviderInit:
    """Tests for GPTImageProvider.__init__."""

    def test_init_explicit_endpoint(self) -> None:
        """Provider accepts an explicit endpoint."""
        provider = GPTImageProvider(project_endpoint="https://example.azure.com")
        assert provider._endpoint == "https://example.azure.com"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider reads AZURE_AI_PROJECT_ENDPOINT from environment."""
        monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://env.azure.com")
        provider = GPTImageProvider()
        assert provider._endpoint == "https://env.azure.com"

    def test_init_missing_endpoint_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider raises ProviderError when no endpoint is available."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        with pytest.raises(ProviderError):
            GPTImageProvider()

    def test_init_custom_model(self) -> None:
        """Provider stores a custom model deployment name."""
        provider = GPTImageProvider(
            project_endpoint="https://example.azure.com",
            model_deployment="my-custom-model",
        )
        assert provider._model == "my-custom-model"


# ---------------------------------------------------------------------------
# Tests: GPTImageProvider.generate_row_strip
# ---------------------------------------------------------------------------


class TestGPTImageProviderGenerate:
    """Tests for GPTImageProvider.generate_row_strip (mocked API)."""

    @pytest.mark.asyncio
    async def test_generate_returns_image(self) -> None:
        """Mocked API returns a valid PIL Image."""
        provider = GPTImageProvider(project_endpoint="https://example.azure.com")

        # Create a dummy generated image (strip of 8 frames)
        dummy_strip = Image.new("RGBA", (512, 64), (100, 100, 100, 255))
        buf = io.BytesIO()
        dummy_strip.save(buf, format="PNG")
        import base64

        b64_image = base64.b64encode(buf.getvalue()).decode("ascii")

        # Mock the response object
        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64_image

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        # Mock openai_client.images.generate (async)
        mock_images = MagicMock()
        mock_images.generate = AsyncMock(return_value=mock_response)

        mock_openai_client = MagicMock()
        mock_openai_client.images = mock_images

        # Mock AIProjectClient
        mock_ai_client = MagicMock()
        mock_ai_client.get_openai_client.return_value = mock_openai_client

        provider._client = mock_ai_client

        result = await provider.generate_row_strip(
            base_reference=_dummy_png_bytes(),
            prompt="Test prompt",
            num_frames=8,
        )

        assert isinstance(result, Image.Image)
        assert result.size == (512, 64)

    @pytest.mark.asyncio
    async def test_generate_api_error_raises_provider_error(self) -> None:
        """API failure raises ProviderError."""
        provider = GPTImageProvider(project_endpoint="https://example.azure.com")

        # Mock openai_client.images.generate to raise
        mock_images = MagicMock()
        mock_images.generate = AsyncMock(side_effect=RuntimeError("API unavailable"))
        mock_openai_client = MagicMock()
        mock_openai_client.images = mock_images

        mock_ai_client = MagicMock()
        mock_ai_client.get_openai_client.return_value = mock_openai_client

        provider._client = mock_ai_client

        with pytest.raises(ProviderError, match="Image generation failed"):
            await provider.generate_row_strip(
                base_reference=_dummy_png_bytes(),
                prompt="Test prompt",
                num_frames=8,
            )

    @pytest.mark.asyncio
    async def test_generate_invalid_base64_raises_provider_error(
        self,
    ) -> None:
        """Invalid base64 response raises ProviderError."""
        provider = GPTImageProvider(project_endpoint="https://example.azure.com")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = "not-valid-base64!!!"

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        mock_images = MagicMock()
        mock_images.generate = AsyncMock(return_value=mock_response)

        mock_openai_client = MagicMock()
        mock_openai_client.images = mock_images

        mock_ai_client = MagicMock()
        mock_ai_client.get_openai_client.return_value = mock_openai_client

        provider._client = mock_ai_client

        with pytest.raises(ProviderError, match="Failed to decode"):
            await provider.generate_row_strip(
                base_reference=_dummy_png_bytes(),
                prompt="Test prompt",
                num_frames=4,
            )


# ---------------------------------------------------------------------------
# Tests: GPTImageProvider.close
# ---------------------------------------------------------------------------


class TestGPTImageProviderClose:
    """Tests for GPTImageProvider.close."""

    @pytest.mark.asyncio
    async def test_close_no_client(self) -> None:
        """close() does not raise when no client was created."""
        provider = GPTImageProvider(project_endpoint="https://example.azure.com")
        await provider.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        """close() calls close on the underlying client."""
        provider = GPTImageProvider(project_endpoint="https://example.azure.com")
        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.close.assert_awaited_once()
        assert provider._client is None


# ---------------------------------------------------------------------------
# Tests: build_reference_prompt
# ---------------------------------------------------------------------------


class TestBuildReferencePrompt:
    """Tests for the prompt builder function."""

    def test_contains_animation_name(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt includes the animation name."""
        prompt = build_reference_prompt(walk_animation, character)
        assert "WALK" in prompt

    def test_contains_frame_count(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt includes '8 frames' for an 8-frame animation."""
        prompt = build_reference_prompt(walk_animation, character)
        assert "8 frames" in prompt

    def test_contains_character_name(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt includes the character name."""
        prompt = build_reference_prompt(walk_animation, character)
        assert "Theron Ashblade" in prompt

    def test_contains_character_class(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt includes the character class."""
        prompt = build_reference_prompt(walk_animation, character)
        assert "Warrior" in prompt

    def test_contains_character_description(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt includes the character description when provided."""
        prompt = build_reference_prompt(
            walk_animation, character, character_description="Tall and strong"
        )
        assert "Tall and strong" in prompt

    def test_no_description_when_empty(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt omits description section when empty."""
        prompt = build_reference_prompt(walk_animation, character)
        assert "Description:" not in prompt

    def test_contains_prompt_context(
        self, walk_animation: AnimationDef, character: CharacterConfig
    ) -> None:
        """Prompt includes the animation's prompt_context."""
        prompt = build_reference_prompt(walk_animation, character)
        assert "walking forward with a steady gait" in prompt

    def test_no_class_when_empty(self, walk_animation: AnimationDef) -> None:
        """Prompt omits class section when character has no class."""
        char = CharacterConfig(name="Generic NPC")
        prompt = build_reference_prompt(walk_animation, char)
        assert "Class:" not in prompt


# ---------------------------------------------------------------------------
# Tests: ReferenceProvider is abstract
# ---------------------------------------------------------------------------


class TestReferenceProviderAbstract:
    """Verify the ABC contract."""

    def test_cannot_instantiate_directly(self) -> None:
        """ReferenceProvider cannot be instantiated."""
        with pytest.raises(TypeError):
            ReferenceProvider()  # type: ignore[abstract]

    def test_gpt_image_provider_is_subclass(self) -> None:
        """GPTImageProvider is a subclass of ReferenceProvider."""
        assert issubclass(GPTImageProvider, ReferenceProvider)
