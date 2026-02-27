"""Vision helpers for web character authoring."""

from __future__ import annotations

from spriteforge.providers.chat import ChatProvider
from spriteforge.utils import image_to_data_url_limited


async def describe_character_from_image(
    image_bytes: bytes, provider: ChatProvider
) -> str:
    """Generate a detailed character description from PNG image bytes.

    Args:
        image_bytes: Uploaded base image bytes (PNG) used for vision analysis.
        provider: Chat provider used to execute the vision completion.

    Returns:
        A plain-text character description suitable for pre-filling the GUI.
    """
    image_data_url = image_to_data_url_limited(image_bytes, max_bytes=4_000_000)
    prompt = (
        "Describe this character in at least 100 words for a pixel-art sprite workflow. "
        "Cover appearance, outfit and gear, colors, pose/body language, silhouette, and art style. "
        "Be specific enough for an artist to recreate the character consistently. "
        "Return plain text only."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
    return (await provider.chat(messages=messages, temperature=0.5)).strip()
