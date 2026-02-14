"""GPT-Image-1.5 reference image provider via Azure AI Foundry.

Uses ``AIProjectClient`` from the ``azure-ai-projects`` package to call
GPT-Image-1.5 through the same Azure AI Foundry project that hosts
Claude Opus 4.6 for Stage 2.  Authentication uses
``DefaultAzureCredential`` — no separate API keys are needed.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from spriteforge.models import AnimationDef, CharacterConfig
from spriteforge.providers._base import ProviderError, ReferenceProvider


def build_reference_prompt(
    animation: AnimationDef,
    character: CharacterConfig,
    character_description: str = "",
) -> str:
    """Build an image generation prompt for a reference animation strip.

    Args:
        animation: The animation row spec.
        character: The character spec.
        character_description: Optional detailed character description.

    Returns:
        A text prompt suitable for GPT-Image-1.5.
    """
    parts: list[str] = []

    parts.append(
        f"Create a horizontal animation strip of {animation.frames} frames "
        f"for a {character.frame_width}x{character.frame_height} pixel-art character."
    )

    parts.append(f"Character: {character.name}.")

    if character.character_class:
        parts.append(f"Class: {character.character_class}.")

    if character_description:
        parts.append(f"Description: {character_description}")

    parts.append(f"Animation: {animation.name}.")

    if animation.prompt_context:
        parts.append(f"Animation context: {animation.prompt_context}")

    parts.append(
        f"Show {animation.frames} frames arranged side by side, left to right. "
        "Each frame should clearly show the character pose for this animation step. "
        "Transparent background."
    )

    return " ".join(parts)


class GPTImageProvider(ReferenceProvider):
    """Reference generation using Azure-hosted GPT-Image-1.5 model.

    Uses ``AIProjectClient`` from the ``azure-ai-projects`` package
    to call GPT-Image-1.5 via the same Azure AI Foundry project used
    for Stage 2 (Claude Opus 4.6).  Authentication uses
    ``DefaultAzureCredential``.

    Implements the :class:`~spriteforge.providers.ReferenceProvider`
    interface.
    """

    def __init__(
        self,
        project_endpoint: str | None = None,
        model_deployment: str = "gpt-image-1.5",
    ) -> None:
        """Initialize GPT-Image provider.

        Args:
            project_endpoint: Azure AI Foundry project endpoint.
                If ``None``, reads from ``AZURE_AI_PROJECT_ENDPOINT``
                environment variable.
            model_deployment: Model deployment name in Foundry.

        Raises:
            ProviderError: If no endpoint is available.
        """
        self._endpoint = project_endpoint or os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT", ""
        )
        if not self._endpoint:
            raise ProviderError(
                "Azure AI project endpoint is required. "
                "Pass project_endpoint or set AZURE_AI_PROJECT_ENDPOINT."
            )
        self._model = model_deployment
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily create and return the Azure AI project client.

        Returns:
            An ``AIProjectClient`` instance.
        """
        if self._client is None:
            from azure.ai.projects.aio import AIProjectClient  # type: ignore[import-not-found]
            from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-not-found]

            credential = DefaultAzureCredential()
            self._client = AIProjectClient(
                credential=credential,
                endpoint=self._endpoint,
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
        client = self._get_client()
        strip_width = frame_size[0] * num_frames
        strip_height = frame_size[1]

        try:
            openai_client = client.get_openai_client()

            ref_b64 = base64.b64encode(base_reference).decode("ascii")
            response = await openai_client.images.generate(
                model=self._model,
                prompt=prompt,
                image=[
                    {
                        "type": "input_image",
                        "input_image": {
                            "url": f"data:image/png;base64,{ref_b64}",
                        },
                    },
                ],
                size=f"{strip_width}x{strip_height}",
                n=1,
                response_format="b64_json",
            )
        except Exception as exc:
            raise ProviderError(f"Image generation failed: {exc}") from exc

        try:
            image_b64: str = response.data[0].b64_json  # type: ignore[union-attr]
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            return image
        except Exception as exc:
            raise ProviderError(f"Failed to decode generated image: {exc}") from exc

    async def close(self) -> None:
        """Clean up provider resources."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
