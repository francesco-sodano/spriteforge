"""Best-effort character description drafting from a reference image."""

from __future__ import annotations

import io
from pathlib import Path

from spriteforge.preprocessing.image_io import (
    resize_reference,
    validate_reference_image,
)
from spriteforge.providers.chat import ChatProvider
from spriteforge.utils import image_to_data_url_limited


def deterministic_description_fallback(character_name: str) -> str:
    """Return deterministic fallback text used when drafting is unavailable."""
    return (
        f"Pixel-art character named {character_name}. "
        "Update this draft with detailed notes about body, face, clothing, colors, and gear."
    )


async def draft_character_description_from_image(
    image_path: str | Path,
    character_name: str,
    chat_provider: ChatProvider,
) -> str:
    """Generate a best-effort character description draft from an image."""
    fallback = deterministic_description_fallback(character_name)
    try:
        image = validate_reference_image(image_path)
        resized = resize_reference(image, 64, 64)
        with io.BytesIO() as buffer:
            resized.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        image_data_url = image_to_data_url_limited(image_bytes, max_bytes=4_000_000)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this pixel-art game character in 2-4 concise sentences. "
                            f"Character name: {character_name}. "
                            "Focus on visual traits: body shape, face/hair, outfit, colors, and held gear. "
                            "Return plain text only."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]
        drafted = (await chat_provider.chat(messages=messages, temperature=0.5)).strip()
        return drafted or fallback
    except Exception:
        return fallback
