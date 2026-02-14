"""Mock chat provider for unit testing."""

from __future__ import annotations

from typing import Any

from spriteforge.providers.chat import ChatProvider


class MockChatProvider(ChatProvider):
    """A mock chat provider that returns pre-configured responses.

    Usage::

        mock = MockChatProvider(responses=["response1", "response2"])
        result = await mock.chat(messages=[...])
        assert result == "response1"
        result = await mock.chat(messages=[...])
        assert result == "response2"
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_history: list[dict[str, Any]] = []
        self._call_index = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> str:
        """Return the next pre-configured response.

        Args:
            messages: Message dicts (recorded in call history).
            temperature: Sampling temperature (recorded in call history).
            response_format: Response format hint (recorded in call history).

        Returns:
            The next response string from the pre-configured list.

        Raises:
            ValueError: If no more responses are configured.
        """
        self._call_history.append(
            {
                "messages": messages,
                "temperature": temperature,
                "response_format": response_format,
            }
        )
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response
        raise ValueError("MockChatProvider has no more responses configured")
