"""Tests for web vision description helpers."""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

from spriteforge.models import GenerationConfig
from spriteforge.providers.azure_chat import AzureChatProvider
from spriteforge.providers.chat import ChatProvider
from spriteforge.web.vision import describe_character_from_image

_TEST_BASE_COLOR = (120, 80, 60, 255)


def _test_png_bytes() -> bytes:
    image = Image.new("RGBA", (64, 64), _TEST_BASE_COLOR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_describe_character_from_image_builds_vision_message() -> None:
    class MockProvider(ChatProvider):
        captured_messages = None
        captured_temperature = None

        async def chat(self, messages, temperature):
            self.captured_messages = messages
            self.captured_temperature = temperature
            return " Detailed character description. "

    provider = MockProvider()
    result = await describe_character_from_image(_test_png_bytes(), provider)

    assert result == "Detailed character description."
    assert provider.captured_temperature == 0.5
    assert provider.captured_messages is not None
    content = provider.captured_messages[0]["content"]
    assert content[0]["type"] == "text"
    assert "at least 100 words" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_describe_character_from_image_real_azure(
    azure_project_endpoint: str,
) -> None:
    provider = AzureChatProvider(
        project_endpoint=azure_project_endpoint,
        model_deployment_name=os.environ.get(
            "SPRITEFORGE_TEST_LABELING_MODEL", GenerationConfig().labeling_model
        ),
    )
    try:
        result = await describe_character_from_image(_test_png_bytes(), provider)
        assert isinstance(result, str)
        assert len(result.split()) >= 100
    finally:
        await provider.close()
