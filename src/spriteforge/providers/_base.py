"""Base class and exceptions for reference image providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class ProviderError(Exception):
    """Raised when a reference image provider fails."""


class ReferenceProvider(ABC):
    """Abstract base for rough reference image generation providers."""

    @abstractmethod
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
            approximately *frame_size*.  This is a rough reference —
            not pixel precise.

        Raises:
            ProviderError: If the image generation fails.
        """

    @abstractmethod
    async def close(self) -> None:
        """Clean up provider resources."""
