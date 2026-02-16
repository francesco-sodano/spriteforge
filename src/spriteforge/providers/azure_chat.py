"""Azure OpenAI chat provider using Entra ID bearer token auth.

Uses ``AsyncAzureOpenAI`` directly with ``get_bearer_token_provider``
for authentication, matching the pattern used by ``GPTImageProvider``.
"""

from __future__ import annotations

import logging
from typing import Any

from spriteforge.errors import GenerationError
from spriteforge.providers.chat import ChatProvider

logger = logging.getLogger(__name__)

_AZURE_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
_DEFAULT_API_VERSION = "2024-12-01-preview"


class AzureChatProvider(ChatProvider):
    """Chat provider using Azure OpenAI with Entra ID bearer token auth.

    Uses ``AsyncAzureOpenAI`` directly (no ``AIProjectClient`` dependency)
    with ``DefaultAzureCredential`` + ``get_bearer_token_provider`` for
    authentication — the same pattern as ``GPTImageProvider``.

    Implements async context manager protocol for automatic cleanup::

        async with AzureChatProvider(...) as provider:
            response = await provider.chat(messages)
            # Clients are reused across calls
        # Automatic cleanup on exit
    """

    def __init__(
        self,
        azure_endpoint: str | None = None,
        model_deployment_name: str = "gpt-5-mini",
        credential: Any | None = None,
        api_version: str = _DEFAULT_API_VERSION,
        *,
        # Legacy alias — accepted but mapped to azure_endpoint
        project_endpoint: str | None = None,
    ) -> None:
        """Initialize the Azure chat provider.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
                            (e.g. ``https://<resource>.openai.azure.com``).
                            Falls back to ``AZURE_OPENAI_ENDPOINT`` or
                            ``AZURE_OPENAI_GPT_IMAGE_ENDPOINT`` env vars.
            model_deployment_name: Deployment name of the chat model.
            credential: Azure credential (defaults to ``DefaultAzureCredential``).
                        If provided, the caller is responsible for closing it.
            api_version: Azure OpenAI API version.
            project_endpoint: **Deprecated** — alias for *azure_endpoint*.
                              Accepted for backward compatibility with callers
                              that previously used the Foundry project endpoint.
        """
        import os

        # Resolve endpoint: explicit > legacy alias > env vars
        resolved = (
            azure_endpoint
            or project_endpoint
            or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            or os.environ.get("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "")
        )
        # Strip Foundry project path if present (e.g. /api/projects/foo)
        if resolved and "/api/projects/" in resolved:
            # Extract base: https://<resource>.services.ai.azure.com
            base = resolved.split("/api/projects/")[0]
            # Derive OpenAI endpoint from services endpoint
            resource_host = base.replace("https://", "").replace(
                ".services.ai.azure.com", ""
            )
            resolved = f"https://{resource_host}.openai.azure.com"
            logger.info(
                "Converted Foundry project endpoint to OpenAI endpoint: %s",
                resolved,
            )

        self._endpoint = resolved
        self._model_deployment_name = model_deployment_name
        self._user_credential = credential
        self._owns_credential = credential is None
        self._api_version = api_version

        # Lazily initialized on first use
        self._credential: Any | None = None
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily create and return the Azure OpenAI client.

        Uses ``get_bearer_token_provider`` from ``azure.identity.aio`` to
        obtain Entra ID tokens for the Azure Cognitive Services scope.

        Returns:
            An ``AsyncAzureOpenAI`` instance.
        """
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncAzureOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK package is required for AzureChatProvider. "
                "Install with: pip install openai"
            ) from exc

        try:
            from azure.identity.aio import (  # type: ignore[import-untyped,import-not-found]
                DefaultAzureCredential,
                get_bearer_token_provider,
            )
        except ImportError as exc:
            raise ImportError(
                "Azure Identity package is required for AzureChatProvider. "
                "Install with: pip install azure-identity"
            ) from exc

        if not self._endpoint:
            raise GenerationError(
                "No Azure OpenAI endpoint configured. "
                "Set AZURE_OPENAI_ENDPOINT or pass azure_endpoint."
            )

        # Create or reuse credential
        if self._user_credential is not None:
            self._credential = self._user_credential
        else:
            self._credential = DefaultAzureCredential()

        token_provider = get_bearer_token_provider(
            self._credential,
            _AZURE_COGNITIVE_SCOPE,
        )

        self._client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=str(self._endpoint),
            api_version=self._api_version,
        )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> str:
        """Send a chat completion request via Azure OpenAI.

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
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model_deployment_name,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": 16384,
        }
        if response_format:
            kwargs["response_format"] = {"type": response_format}

        try:
            response = await client.chat.completions.create(
                **kwargs,  # type: ignore[arg-type]
            )
        except Exception as exc:
            # Reasoning models (e.g. gpt-5-mini) reject temperature != 1.
            # Retry without temperature when the API tells us so.
            if "temperature" in str(exc) and "unsupported" in str(exc).lower():
                kwargs.pop("temperature", None)
                response = await client.chat.completions.create(
                    **kwargs,  # type: ignore[arg-type]
                )
            else:
                raise

        if (
            not response.choices
            or not response.choices[0].message
            or not response.choices[0].message.content
        ):
            raise GenerationError("LLM returned no content")

        return str(response.choices[0].message.content)

    async def close(self) -> None:
        """Clean up resources.

        Closes the OpenAI client and credentials owned by this provider.
        User-supplied credentials are NOT closed.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None

        if self._owns_credential and self._credential is not None:
            await self._credential.close()
            self._credential = None

    async def __aenter__(self) -> "AzureChatProvider":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager, cleaning up resources."""
        await self.close()
