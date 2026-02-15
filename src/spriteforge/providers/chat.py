"""Chat provider abstraction for LLM text/vision calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ChatProvider(ABC):
    """Abstract base class for chat/vision LLM providers.

    Implementations handle the specifics of calling a particular LLM API.
    The interface is intentionally minimal — just text/vision in, text out.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> str:
        """Send a chat completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Content can be a string or a list of content parts
                      (text + image_url) for vision requests.
            temperature: Sampling temperature (0.0 = deterministic).
            response_format: Optional response format hint (e.g., "json_object").

        Returns:
            The LLM's response text content.

        Raises:
            GenerationError: If the API call fails.
        """

    async def close(self) -> None:
        """Clean up provider resources.

        Default implementation does nothing. Providers that need cleanup
        should override this method.
        """
        pass
