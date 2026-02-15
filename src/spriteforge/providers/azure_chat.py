"""Azure AI Foundry chat provider implementation."""

from __future__ import annotations

from typing import Any

from spriteforge.errors import GenerationError
from spriteforge.providers.chat import ChatProvider


class AzureChatProvider(ChatProvider):
    """Chat provider using Azure AI Foundry (Claude, GPT, etc.).

    Consolidates the duplicated ``_call_llm()`` logic from ``generator.py``
    and ``gates.py`` into a single implementation.

    Implements async context manager protocol for automatic cleanup::

        async with AzureChatProvider(...) as provider:
            response = await provider.chat(messages)
            # Clients are reused across calls
        # Automatic cleanup on exit
    """

    def __init__(
        self,
        project_endpoint: str,
        model_deployment_name: str,
        credential: Any | None = None,
    ) -> None:
        """Initialize the Azure chat provider.

        Args:
            project_endpoint: Azure AI Foundry project endpoint URL.
            model_deployment_name: Deployment name of the chat model.
            credential: Azure credential (defaults to ``DefaultAzureCredential``).
                        If provided, the caller is responsible for closing it.
        """
        self._endpoint = project_endpoint
        self._model_deployment_name = model_deployment_name
        self._user_credential = credential
        self._owns_credential = credential is None

        # Lazily initialized on first use
        self._credential: Any | None = None
        self._project_client: Any | None = None
        self._openai_client: Any | None = None

    async def _ensure_client(self) -> Any:
        """Ensure clients are initialized, creating them on first use.

        Returns:
            The OpenAI client for making chat completion requests.

        Raises:
            GenerationError: If endpoint is not configured.
        """
        if self._openai_client is not None:
            return self._openai_client

        from azure.ai.projects.aio import AIProjectClient  # type: ignore[import-untyped,import-not-found]
        from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-untyped,import-not-found]

        if not self._endpoint:
            raise GenerationError(
                "No Azure AI Foundry endpoint configured. "
                "Set AZURE_AI_PROJECT_ENDPOINT or pass project_endpoint."
            )

        # Create credential if not provided by user
        if self._user_credential is not None:
            self._credential = self._user_credential
        else:
            self._credential = DefaultAzureCredential()

        # Create Azure AI Project client
        self._project_client = AIProjectClient(
            credential=self._credential,
            endpoint=self._endpoint,
        )

        # Get OpenAI client (sync method that returns async client)
        self._openai_client = self._project_client.get_openai_client()

        return self._openai_client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> str:
        """Send a chat completion request via Azure AI Foundry.

        Clients are created on first use and reused across subsequent calls.

        Args:
            messages: OpenAI-style messages list.
            temperature: Sampling temperature.
            response_format: Optional response format hint (e.g., "json_object").

        Returns:
            The text content of the first choice.

        Raises:
            GenerationError: If the API call fails or returns no content.
        """
        client = await self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": self._model_deployment_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 16384,
        }
        if response_format:
            kwargs["response_format"] = {"type": response_format}
        response = await client.chat.completions.create(
            **kwargs,  # type: ignore[arg-type]
        )

        if (
            not response.choices
            or not response.choices[0].message
            or not response.choices[0].message.content
        ):
            raise GenerationError("LLM returned no content")

        return str(response.choices[0].message.content)

    async def close(self) -> None:
        """Clean up resources.

        Closes all clients and credentials owned by this provider.
        User-supplied credentials are NOT closed.
        """
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None

        if self._project_client is not None:
            await self._project_client.close()
            self._project_client = None

        if self._owns_credential and self._credential is not None:
            await self._credential.close()
            self._credential = None

    async def __aenter__(self) -> "AzureChatProvider":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager, cleaning up resources."""
        await self.close()
