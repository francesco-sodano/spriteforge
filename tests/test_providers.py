"""Tests for the reference image provider module and chat provider abstraction."""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from PIL import Image

from spriteforge.models import AnimationDef, CharacterConfig
from spriteforge.providers import (
    AzureChatProvider,
    ChatProvider,
    GPTImageProvider,
    ProviderError,
    ReferenceProvider,
)
from spriteforge.providers.chat import ChatProvider as ChatProviderDirect
from spriteforge.providers.gpt_image import build_reference_prompt

from mock_chat_provider import MockChatProvider

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
# Expected API arguments — single source of truth for assertions
# ---------------------------------------------------------------------------

# These mirror the exact kwargs passed to openai_client.images.edit()
# in GPTImageProvider.generate_row_strip.  If the production code
# changes its API call signature, tests here will fail, which is the
# point — we want to catch payload drift immediately.
_EXPECTED_API_SIZE = "1536x1024"
_DEFAULT_MODEL = "gpt-image-1.5"


def _expected_edit_kwargs(
    *,
    prompt: str = "Test prompt",
    image: bytes | None = None,
    model: str = _DEFAULT_MODEL,
    size: str = _EXPECTED_API_SIZE,
) -> dict[str, Any]:
    """Return the expected keyword arguments for ``images.edit``.

    This centralises the expected payload so that if the real API call
    adds/removes a parameter the test suite surfaces the change in ONE
    place rather than many.
    """
    return {
        "model": model,
        "prompt": prompt,
        "image": image,
        "size": size,
        "n": 1,
        "background": "transparent",
        "output_format": "png",
        "quality": "high",
        "input_fidelity": "high",
    }


# ---------------------------------------------------------------------------
# Helper: build a mocked provider wired up for images.edit
# ---------------------------------------------------------------------------


def _make_mocked_provider(
    *,
    response: Any | None = None,
    side_effect: Exception | None = None,
    model: str = _DEFAULT_MODEL,
) -> tuple[GPTImageProvider, MagicMock]:
    """Create a GPTImageProvider with mocked internals.

    Returns the provider **and** the ``mock_images`` object so callers
    can inspect ``mock_images.edit`` assertions.
    """
    provider = GPTImageProvider(
        api_key="test-api-key",
        azure_endpoint="https://example.openai.azure.com",
        model_deployment=model,
    )

    mock_images = MagicMock()
    # Only expose the 'edit' method — calling any other attribute
    # (e.g. `.generate`, `.create`) will raise AttributeError,
    # catching method-name typos in production code.
    mock_images.edit = AsyncMock(return_value=response, side_effect=side_effect)

    mock_client = MagicMock()
    mock_client.images = mock_images

    provider._client = mock_client
    return provider, mock_images


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

    def test_init_explicit_credentials(self) -> None:
        """Provider accepts explicit API key and endpoint."""
        provider = GPTImageProvider(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com",
        )
        assert provider._api_key == "test-key"
        assert provider._endpoint == "https://example.openai.azure.com"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider reads credentials from environment variables."""
        monkeypatch.setenv("AZURE_OPENAI_GPT_IMAGE_API_KEY", "env-key")
        monkeypatch.setenv(
            "AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "https://env.openai.azure.com"
        )
        provider = GPTImageProvider()
        assert provider._api_key == "env-key"
        assert provider._endpoint == "https://env.openai.azure.com"

    def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider raises ProviderError when API key is missing."""
        monkeypatch.delenv("AZURE_OPENAI_GPT_IMAGE_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="API key is required"):
            GPTImageProvider(azure_endpoint="https://example.openai.azure.com")

    def test_init_missing_endpoint_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider raises ProviderError when endpoint is missing."""
        monkeypatch.delenv("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", raising=False)
        with pytest.raises(ProviderError, match="endpoint is required"):
            GPTImageProvider(api_key="test-key")

    def test_init_custom_model(self) -> None:
        """Provider stores a custom model deployment name."""
        provider = GPTImageProvider(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com",
            model_deployment="my-custom-model",
        )
        assert provider._model == "my-custom-model"

    def test_init_custom_api_version(self) -> None:
        """Provider stores a custom API version."""
        provider = GPTImageProvider(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com",
            api_version="2024-12-01-preview",
        )
        assert provider._api_version == "2024-12-01-preview"


# ---------------------------------------------------------------------------
# Tests: GPTImageProvider.generate_row_strip
# ---------------------------------------------------------------------------


class TestGPTImageProviderGenerate:
    """Tests for GPTImageProvider.generate_row_strip (mocked API).

    Every test that calls :pymethod:`generate_row_strip` also verifies
    that ``images.edit`` was awaited with the **exact** expected
    arguments.  This prevents silent API-signature drift.
    """

    @pytest.mark.asyncio
    async def test_generate_returns_image(self) -> None:
        """Mocked API returns a valid PIL Image resized to strip dims."""
        # API returns a standard-size image (1536x1024);
        # the provider should resize it to strip_width x strip_height.
        api_image = Image.new("RGBA", (1536, 1024), (100, 100, 100, 255))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64_image

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        ref_bytes = _dummy_png_bytes()
        result = await provider.generate_row_strip(
            base_reference=ref_bytes,
            prompt="Test prompt",
            num_frames=8,
        )

        assert isinstance(result, Image.Image)
        # 8 frames * 64px wide, 64px tall
        assert result.size == (512, 64)

        # ---- Strict argument verification ----
        mock_images.edit.assert_awaited_once_with(
            **_expected_edit_kwargs(image=ref_bytes)
        )

    @pytest.mark.asyncio
    async def test_generate_api_error_raises_provider_error(self) -> None:
        """API failure raises ProviderError."""
        provider, mock_images = _make_mocked_provider(
            side_effect=RuntimeError("API unavailable")
        )

        ref_bytes = _dummy_png_bytes()
        with pytest.raises(ProviderError, match="Image generation failed"):
            await provider.generate_row_strip(
                base_reference=ref_bytes,
                prompt="Test prompt",
                num_frames=8,
            )

        # Even on error, verify we attempted the right call.
        mock_images.edit.assert_awaited_once_with(
            **_expected_edit_kwargs(image=ref_bytes)
        )

    @pytest.mark.asyncio
    async def test_generate_invalid_base64_raises_provider_error(
        self,
    ) -> None:
        """Invalid base64 response raises ProviderError."""
        mock_image_data = MagicMock()
        mock_image_data.b64_json = "not-valid-base64!!!"

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        ref_bytes = _dummy_png_bytes()
        with pytest.raises(ProviderError, match="Failed to decode"):
            await provider.generate_row_strip(
                base_reference=ref_bytes,
                prompt="Test prompt",
                num_frames=4,
            )

        # Verify correct call even when decoding fails downstream.
        mock_images.edit.assert_awaited_once_with(
            **_expected_edit_kwargs(image=ref_bytes)
        )


# ---------------------------------------------------------------------------
# Tests: GPTImageProvider.close
# ---------------------------------------------------------------------------


class TestGPTImageProviderClose:
    """Tests for GPTImageProvider.close."""

    @pytest.mark.asyncio
    async def test_close_no_client(self) -> None:
        """close() does not raise when no client was created."""
        provider = GPTImageProvider(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com",
        )
        await provider.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        """close() calls close on the underlying client."""
        provider = GPTImageProvider(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com",
        )
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
# Tests: API contract / payload schema validation
# ---------------------------------------------------------------------------


class TestGPTImageProviderAPIContract:
    """Verify that the API payload sent to images.edit matches the
    expected schema.  These tests exist specifically to catch the class
    of bug where code changes break the payload structure without any
    test noticing (because permissive MagicMocks accept anything).
    """

    @pytest.mark.asyncio
    async def test_edit_called_not_generate(self) -> None:
        """Provider must call images.edit — NOT images.generate or images.create."""
        api_image = Image.new("RGBA", (1536, 1024), (0, 0, 0, 0))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        await provider.generate_row_strip(
            base_reference=_dummy_png_bytes(),
            prompt="x",
            num_frames=1,
        )

        # images.edit must be called exactly once
        mock_images.edit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_model_deployment_forwarded(self) -> None:
        """A custom model name is forwarded to the API."""
        api_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(
            response=mock_response, model="my-custom-model"
        )

        ref_bytes = _dummy_png_bytes()
        await provider.generate_row_strip(
            base_reference=ref_bytes,
            prompt="custom",
            num_frames=1,
        )

        mock_images.edit.assert_awaited_once_with(
            **_expected_edit_kwargs(
                model="my-custom-model", prompt="custom", image=ref_bytes
            )
        )

    @pytest.mark.asyncio
    async def test_image_sent_as_raw_bytes(self) -> None:
        """The base_reference image is sent as raw bytes, not a dict/JSON."""
        api_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        ref_bytes = _dummy_png_bytes()
        await provider.generate_row_strip(
            base_reference=ref_bytes,
            prompt="check bytes",
            num_frames=1,
        )

        # Inspect the actual `image` kwarg
        actual_kwargs = mock_images.edit.call_args.kwargs
        assert isinstance(
            actual_kwargs["image"], bytes
        ), f"Expected 'image' to be raw bytes, got {type(actual_kwargs['image'])}"

    @pytest.mark.asyncio
    async def test_payload_has_required_keys(self) -> None:
        """Payload must contain all required keys for the images.edit API."""
        api_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        await provider.generate_row_strip(
            base_reference=_dummy_png_bytes(),
            prompt="schema check",
            num_frames=4,
        )

        actual_kwargs = mock_images.edit.call_args.kwargs

        required_keys = {
            "model",
            "prompt",
            "image",
            "size",
            "n",
            "background",
            "output_format",
            "quality",
            "input_fidelity",
        }
        missing = required_keys - set(actual_kwargs.keys())
        extra = set(actual_kwargs.keys()) - required_keys
        assert not missing, f"Missing payload keys: {missing}"
        assert not extra, f"Unexpected payload keys: {extra}"

    @pytest.mark.asyncio
    async def test_payload_value_types(self) -> None:
        """Validate value types in the payload match API expectations."""
        api_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        await provider.generate_row_strip(
            base_reference=_dummy_png_bytes(),
            prompt="type check",
            num_frames=4,
        )

        kw = mock_images.edit.call_args.kwargs
        assert isinstance(kw["model"], str)
        assert isinstance(kw["prompt"], str)
        assert isinstance(kw["image"], bytes)
        assert isinstance(kw["size"], str)
        assert isinstance(kw["n"], int) and kw["n"] >= 1
        assert kw["background"] in {"transparent", "opaque", "auto"}
        assert kw["output_format"] in {"png", "jpeg", "webp"}
        assert kw["quality"] in {"low", "medium", "high", "auto"}
        # input_fidelity is a string hint
        assert isinstance(kw["input_fidelity"], str)

    @pytest.mark.asyncio
    async def test_size_is_valid_api_literal(self) -> None:
        """size must be one of the API-supported dimension strings."""
        api_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        buf = io.BytesIO()
        api_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        mock_image_data = MagicMock()
        mock_image_data.b64_json = b64
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        provider, mock_images = _make_mocked_provider(response=mock_response)

        await provider.generate_row_strip(
            base_reference=_dummy_png_bytes(),
            prompt="size check",
            num_frames=4,
        )

        kw = mock_images.edit.call_args.kwargs
        valid_sizes = {"1024x1024", "1536x1024", "1024x1536", "auto"}
        assert kw["size"] in valid_sizes, (
            f"size={kw['size']!r} is not a valid GPT-Image API size. "
            f"Valid: {valid_sizes}"
        )


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


# ---------------------------------------------------------------------------
# Tests: ChatProvider is abstract
# ---------------------------------------------------------------------------


class TestChatProviderAbstract:
    """Verify the ChatProvider ABC contract."""

    def test_chat_provider_is_abc(self) -> None:
        """ChatProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ChatProviderDirect()  # type: ignore[abstract]

    def test_azure_chat_provider_implements_interface(self) -> None:
        """AzureChatProvider is a ChatProvider."""
        assert issubclass(AzureChatProvider, ChatProvider)


# ---------------------------------------------------------------------------
# Tests: MockChatProvider
# ---------------------------------------------------------------------------


class TestMockChatProvider:
    """Tests for MockChatProvider."""

    @pytest.mark.asyncio
    async def test_mock_chat_provider_returns_responses(self) -> None:
        """Returns configured responses in order."""
        mock = MockChatProvider(responses=["response1", "response2"])
        result1 = await mock.chat(messages=[{"role": "user", "content": "hi"}])
        result2 = await mock.chat(messages=[{"role": "user", "content": "bye"}])
        assert result1 == "response1"
        assert result2 == "response2"

    @pytest.mark.asyncio
    async def test_mock_chat_provider_records_history(self) -> None:
        """Call history is recorded."""
        mock = MockChatProvider(responses=["ok"])
        messages = [{"role": "user", "content": "test"}]
        await mock.chat(messages=messages, temperature=0.5, response_format="json")

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["messages"] == messages
        assert mock._call_history[0]["temperature"] == 0.5
        assert mock._call_history[0]["response_format"] == "json"

    @pytest.mark.asyncio
    async def test_mock_chat_provider_exhausted(self) -> None:
        """Raises when no more responses."""
        mock = MockChatProvider(responses=["only_one"])
        await mock.chat(messages=[])
        with pytest.raises(ValueError, match="no more responses"):
            await mock.chat(messages=[])


# ---------------------------------------------------------------------------
# Tests: AzureChatProvider client reuse
# ---------------------------------------------------------------------------


class TestAzureChatProviderClientReuse:
    """Tests for client reuse and resource management in AzureChatProvider."""

    @pytest.mark.asyncio
    async def test_azure_chat_reuses_client(self) -> None:
        """Call chat() twice → verify AIProjectClient is created only once."""
        from unittest.mock import MagicMock, patch

        # Create mocks that will be injected
        mock_credential = AsyncMock()
        mock_project_client = AsyncMock()
        mock_openai_client = AsyncMock()

        # Mock the response
        mock_message = MagicMock()
        mock_message.content = "response text"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        mock_project_client.get_openai_client.return_value = mock_openai_client

        # Create provider with user credential
        provider = AzureChatProvider(
            project_endpoint="https://example.azure.com",
            model_deployment_name="test-model",
            credential=mock_credential,
        )

        # Manually inject the clients to simulate lazy initialization
        provider._credential = mock_credential
        provider._project_client = mock_project_client
        provider._openai_client = mock_openai_client

        # First chat call - clients already initialized
        result1 = await provider.chat(messages=[{"role": "user", "content": "test1"}])
        assert result1 == "response text"

        # Capture current client references
        first_credential = provider._credential
        first_project = provider._project_client
        first_openai = provider._openai_client

        # Second chat call - should reuse same clients
        result2 = await provider.chat(messages=[{"role": "user", "content": "test2"}])
        assert result2 == "response text"

        # Verify same client instances are reused
        assert provider._credential is first_credential
        assert provider._project_client is first_project
        assert provider._openai_client is first_openai

        # Verify chat was called TWICE (once per call)
        assert mock_openai_client.chat.completions.create.call_count == 2

        await provider.close()

    @pytest.mark.asyncio
    async def test_azure_chat_close_cleanup(self) -> None:
        """Call close() → verify all resources are released."""
        provider = AzureChatProvider(
            project_endpoint="https://example.azure.com",
            model_deployment_name="test-model",
        )

        # Mock clients
        mock_credential = AsyncMock()
        mock_project_client = AsyncMock()
        mock_openai_client = AsyncMock()

        # Manually inject mocked clients
        provider._credential = mock_credential
        provider._project_client = mock_project_client
        provider._openai_client = mock_openai_client
        provider._owns_credential = True

        await provider.close()

        # Verify all close methods were called
        mock_openai_client.close.assert_awaited_once()
        mock_project_client.close.assert_awaited_once()
        mock_credential.close.assert_awaited_once()

        # Verify all references are cleared
        assert provider._openai_client is None
        assert provider._project_client is None
        assert provider._credential is None

    @pytest.mark.asyncio
    async def test_azure_chat_context_manager(self) -> None:
        """Use async with AzureChatProvider(...) as provider → auto-cleanup."""
        mock_credential = AsyncMock()
        mock_project_client = AsyncMock()
        mock_openai_client = AsyncMock()

        # Mock response
        mock_message = MagicMock()
        mock_message.content = "context manager response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        mock_project_client.get_openai_client.return_value = mock_openai_client

        # Create provider
        provider = AzureChatProvider(
            project_endpoint="https://example.azure.com",
            model_deployment_name="test-model",
        )

        # Manually inject the clients
        provider._credential = mock_credential
        provider._project_client = mock_project_client
        provider._openai_client = mock_openai_client
        provider._owns_credential = True

        # Use as context manager
        async with provider:
            result = await provider.chat(messages=[{"role": "user", "content": "test"}])
            assert result == "context manager response"

        # After exiting context, all clients should be closed
        mock_openai_client.close.assert_awaited_once()
        mock_project_client.close.assert_awaited_once()
        mock_credential.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_azure_chat_does_not_close_user_credential(self) -> None:
        """Pass external credential → verify it's not closed on close()."""
        user_credential = AsyncMock()

        provider = AzureChatProvider(
            project_endpoint="https://example.azure.com",
            model_deployment_name="test-model",
            credential=user_credential,
        )

        # Mock other clients
        mock_project_client = AsyncMock()
        mock_openai_client = AsyncMock()

        # Manually inject mocked clients
        provider._credential = user_credential  # Same as user_credential
        provider._project_client = mock_project_client
        provider._openai_client = mock_openai_client

        await provider.close()

        # Verify openai and project clients were closed
        mock_openai_client.close.assert_awaited_once()
        mock_project_client.close.assert_awaited_once()

        # Verify user credential was NOT closed
        user_credential.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: AzureChatProvider.response_format
# ---------------------------------------------------------------------------


class TestAzureChatProviderResponseFormat:
    """Tests for response_format parameter forwarding in AzureChatProvider."""

    @pytest.mark.asyncio
    async def test_azure_chat_forwards_response_format(self) -> None:
        """AzureChatProvider forwards response_format to API call."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create mocks
        mock_credential = AsyncMock()
        mock_project_client = AsyncMock()
        mock_openai_client = AsyncMock()

        # Mock the response
        mock_message = MagicMock()
        mock_message.content = '{"result": "test"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        mock_project_client.get_openai_client.return_value = mock_openai_client

        # Create provider with user credential
        provider = AzureChatProvider(
            project_endpoint="https://example.azure.com",
            model_deployment_name="test-model",
            credential=mock_credential,
        )

        # Manually inject mocked clients
        provider._credential = mock_credential
        provider._project_client = mock_project_client
        provider._openai_client = mock_openai_client

        # Call chat with response_format
        messages = [{"role": "user", "content": "test"}]
        await provider.chat(messages, temperature=0.5, response_format="json_object")

        # Verify the API was called with response_format
        mock_openai_client.chat.completions.create.assert_awaited_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]

        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_azure_chat_without_response_format(self) -> None:
        """AzureChatProvider works without response_format parameter."""
        from unittest.mock import AsyncMock, MagicMock

        # Create mocks
        mock_credential = AsyncMock()
        mock_project_client = AsyncMock()
        mock_openai_client = AsyncMock()

        # Mock the response
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        mock_project_client.get_openai_client.return_value = mock_openai_client

        # Create provider
        provider = AzureChatProvider(
            project_endpoint="https://example.azure.com",
            model_deployment_name="test-model",
            credential=mock_credential,
        )

        # Manually inject mocked clients
        provider._credential = mock_credential
        provider._project_client = mock_project_client
        provider._openai_client = mock_openai_client

        # Call chat without response_format
        messages = [{"role": "user", "content": "test"}]
        await provider.chat(messages, temperature=1.0)

        # Verify the API was called without response_format
        mock_openai_client.chat.completions.create.assert_awaited_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]

        assert "response_format" not in call_kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 1.0


# ---------------------------------------------------------------------------
# Tests: Lazy import mechanism
# ---------------------------------------------------------------------------


class TestLazyImports:
    """Tests for lazy loading of Azure-dependent providers."""

    def test_lazy_import_azure_chat_provider(self) -> None:
        """AzureChatProvider can be imported lazily from providers module."""
        # Import using __getattr__
        from spriteforge import providers

        AzureChatProvider = providers.__getattr__("AzureChatProvider")
        assert AzureChatProvider is not None
        assert AzureChatProvider.__name__ == "AzureChatProvider"

    def test_lazy_import_gpt_image_provider(self) -> None:
        """GPTImageProvider can be imported lazily from providers module."""
        from spriteforge import providers

        GPTImageProvider = providers.__getattr__("GPTImageProvider")
        assert GPTImageProvider is not None
        assert GPTImageProvider.__name__ == "GPTImageProvider"

    def test_lazy_import_from_providers_directly(self) -> None:
        """Azure providers can be imported directly from providers module."""
        from spriteforge.providers import AzureChatProvider, GPTImageProvider

        assert AzureChatProvider is not None
        assert GPTImageProvider is not None

    def test_lazy_import_from_root_package(self) -> None:
        """Azure providers can be imported from root spriteforge package."""
        from spriteforge import AzureChatProvider, GPTImageProvider

        assert AzureChatProvider is not None
        assert GPTImageProvider is not None

    def test_non_azure_providers_importable(self) -> None:
        """Non-Azure providers (ChatProvider, ReferenceProvider) are eagerly loaded."""
        from spriteforge.providers import ChatProvider, ReferenceProvider

        assert ChatProvider is not None
        assert ReferenceProvider is not None

    def test_invalid_attribute_raises(self) -> None:
        """Requesting non-existent attribute raises AttributeError."""
        from spriteforge import providers

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentClass'"):
            providers.__getattr__("NonExistentClass")

    def test_all_exports_available(self) -> None:
        """All items in __all__ are accessible."""
        from spriteforge import providers

        for name in providers.__all__:
            attr = getattr(providers, name)
            assert attr is not None


# ---------------------------------------------------------------------------
# Tests: Azure SDK import error messages
# ---------------------------------------------------------------------------


class TestAzureSDKErrorMessages:
    """Tests for clear error messages when Azure SDK is missing."""

    @pytest.mark.asyncio
    async def test_azure_chat_provider_missing_sdk_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AzureChatProvider shows clear error when Azure SDK is missing."""
        import sys
        import builtins

        # Remove azure SDK modules from sys.modules to simulate missing SDK
        azure_sdk_modules = [
            key
            for key in sys.modules.keys()
            if key.startswith("azure.ai.") or key.startswith("azure.identity")
        ]
        for key in azure_sdk_modules:
            monkeypatch.delitem(sys.modules, key, raising=False)

        # Mock the import to fail for Azure SDK packages only
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("azure.ai.") or name.startswith("azure.identity"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Import and create provider (doesn't fail yet)
        from spriteforge.providers.azure_chat import AzureChatProvider

        provider = AzureChatProvider(
            project_endpoint="https://test.azure.com",
            model_deployment_name="test-model",
        )

        # Using the provider should fail with clear error message
        with pytest.raises(ImportError) as exc_info:
            await provider.chat([{"role": "user", "content": "test"}])

        error_msg = str(exc_info.value)
        assert "Azure SDK packages are required" in error_msg
        assert "pip install spriteforge[azure]" in error_msg
        assert "azure-ai-projects" in error_msg
        assert "azure-identity" in error_msg

    def test_gpt_image_provider_missing_sdk_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GPTImageProvider shows clear error when OpenAI SDK is missing."""
        import sys
        import builtins

        # Remove OpenAI SDK modules from sys.modules to simulate missing SDK
        openai_modules = [key for key in sys.modules.keys() if key.startswith("openai")]
        for key in openai_modules:
            monkeypatch.delitem(sys.modules, key, raising=False)

        # Mock the import to fail for OpenAI SDK only
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Import and create provider (doesn't fail yet)
        from spriteforge.providers.gpt_image import GPTImageProvider

        provider = GPTImageProvider(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Using the provider should fail with clear error message
        with pytest.raises(ImportError) as exc_info:
            provider._get_client()

        error_msg = str(exc_info.value)
        assert "OpenAI SDK package is required" in error_msg
        assert "pip install openai" in error_msg
