"""Azure AI Foundry chat provider implementation."""

from __future__ import annotations

from typing import Any

from spriteforge.errors import GenerationError
from spriteforge.providers.chat import ChatProvider


class AzureChatProvider(ChatProvider):
    """Chat provider using Azure AI Foundry (Claude, GPT, etc.).

    Consolidates the duplicated ``_call_llm()`` logic from ``generator.py``
    and ``gates.py`` into a single implementation.
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
        """
        self._endpoint = project_endpoint
        self._model_deployment_name = model_deployment_name
        self._credential = credential

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> str:
        """Send a chat completion request via Azure AI Foundry.

        Args:
            messages: OpenAI-style messages list.
            temperature: Sampling temperature.
            response_format: Optional response format hint (e.g., "json_object").

        Returns:
            The text content of the first choice.

        Raises:
            GenerationError: If the API call fails or returns no content.
        """
        from azure.ai.projects.aio import AIProjectClient  # type: ignore[import-untyped,import-not-found]
        from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-untyped,import-not-found]

        if not self._endpoint:
            raise GenerationError(
                "No Azure AI Foundry endpoint configured. "
                "Set AZURE_AI_PROJECT_ENDPOINT or pass project_endpoint."
            )

        credential = self._credential or DefaultAzureCredential()
        try:
            project_client = AIProjectClient(
                credential=credential,
                endpoint=self._endpoint,
            )
            try:
                openai_client = project_client.get_openai_client()
                try:
                    kwargs: dict[str, Any] = {
                        "model": self._model_deployment_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 16384,
                    }
                    response = await openai_client.chat.completions.create(
                        **kwargs,  # type: ignore[arg-type]
                    )
                finally:
                    await openai_client.close()
            finally:
                await project_client.close()
        finally:
            if self._credential is None:
                await credential.close()

        if (
            not response.choices
            or not response.choices[0].message
            or not response.choices[0].message.content
        ):
            raise GenerationError("LLM returned no content")

        return str(response.choices[0].message.content)
