"""GPT-Image-1.5 reference image provider via Azure OpenAI.

Uses ``AsyncAzureOpenAI`` from the ``openai`` package to call
GPT-Image-1.5 with API key authentication.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from spriteforge.logging import get_logger
from spriteforge.prompts.providers import build_reference_prompt
from spriteforge.providers._base import ProviderError, ReferenceProvider

logger = get_logger("providers")


class GPTImageProvider(ReferenceProvider):
    """Reference generation using Azure-hosted GPT-Image-1.5 model.

    Uses ``AsyncAzureOpenAI`` from the ``openai`` package to call
    GPT-Image-1.5 with API key authentication.

    Implements the :class:`~spriteforge.providers.ReferenceProvider`
    interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        model_deployment: str = "gpt-image-1.5",
        api_version: str = "2025-04-01-preview",
    ) -> None:
        """Initialize GPT-Image provider.

        Args:
            api_key: Azure OpenAI API key. If ``None``, reads from
                ``AZURE_OPENAI_GPT_IMAGE_API_KEY`` environment variable.
            azure_endpoint: Azure OpenAI endpoint URL.
                If ``None``, reads from ``AZURE_OPENAI_GPT_IMAGE_ENDPOINT``
                environment variable.
            model_deployment: Model deployment name in Azure OpenAI.
            api_version: Azure OpenAI API version.

        Raises:
            ProviderError: If API key or endpoint is missing.
        """
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_GPT_IMAGE_API_KEY", "")
        if not self._api_key:
            raise ProviderError(
                "Azure OpenAI API key is required. "
                "Pass api_key or set AZURE_OPENAI_GPT_IMAGE_API_KEY."
            )

        self._endpoint = azure_endpoint or os.environ.get(
            "AZURE_OPENAI_GPT_IMAGE_ENDPOINT", ""
        )
        if not self._endpoint:
            raise ProviderError(
                "Azure OpenAI endpoint is required. "
                "Pass azure_endpoint or set AZURE_OPENAI_GPT_IMAGE_ENDPOINT."
            )

        self._model = model_deployment
        self._api_version = api_version
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily create and return the Azure OpenAI client.

        Returns:
            An ``AsyncAzureOpenAI`` instance.
        """
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "OpenAI SDK package is required for GPTImageProvider. "
                    "Install with: pip install openai"
                ) from exc

            self._client = AsyncAzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
            )
        return self._client

    async def generate_row_strip(
        self,
        base_reference: bytes,
        prompt: str,
        num_frames: int,
        frame_size: tuple[int, int] = (64, 64),
    ) -> Image.Image:
        """Generate a rough animation reference strip.

        Args:
            base_reference: PNG bytes of the base character reference image.
            prompt: Text prompt describing the animation row to generate.
            num_frames: Number of frames to generate in the strip.
            frame_size: (width, height) of each frame in pixels.

        Returns:
            A PIL Image containing *num_frames* side-by-side, each
            approximately *frame_size*.

        Raises:
            ProviderError: If the image generation fails.
        """
        strip_width = frame_size[0] * num_frames
        strip_height = frame_size[1]

        logger.info(
            "Generating reference strip via GPT-Image-1.5 (%d frames, %dx%d)",
            num_frames,
            strip_width,
            strip_height,
        )

        # Pick the best standard size for the strip.  GPT Image models only
        # support 1024x1024, 1536x1024, 1024x1536, or "auto".  Reference
        # strips are wider than tall, so landscape (1536x1024) is the best
        # fixed option.  We resize to the desired strip size afterward.
        # See: https://developers.openai.com/api/docs/guides/image-generation
        api_size: str = "1536x1024"

        try:
            client = self._get_client()

            # images.edit accepts `image` as raw file bytes (FileTypes),
            # NOT a dict/JSON structure.  The SDK sends it as multipart/form-data.
            response = await client.images.edit(
                model=self._model,
                prompt=prompt,
                image=base_reference,
                size=api_size,
                n=1,
                background="transparent",
                output_format="png",
                quality="high",
                input_fidelity="high",
            )
        except Exception as exc:
            logger.error("Reference generation failed: %s", exc)
            raise ProviderError(f"Image generation failed: {exc}") from exc

        try:
            # GPT Image models always return base64-encoded images
            # (response_format is a DALL-E-only parameter).
            image_b64: str = response.data[0].b64_json  # type: ignore[union-attr]
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))
            image.load()

            # Resize from the standard API size to the desired strip
            # dimensions.  This is a rough reference — not pixel-precise.
            if image.size != (strip_width, strip_height):
                image = image.resize(
                    (strip_width, strip_height), Image.Resampling.LANCZOS
                )

            logger.info("Reference strip generated (%dx%d)", image.width, image.height)
            return image
        except Exception as exc:
            logger.error("Failed to decode generated image: %s", exc)
            raise ProviderError(f"Failed to decode generated image: {exc}") from exc

    async def close(self) -> None:
        """Clean up provider resources.

        Closes the Azure OpenAI client.
        """
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
