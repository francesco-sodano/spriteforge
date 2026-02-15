"""Reference image providers and chat/vision LLM providers.

This package contains abstract base classes and concrete implementations
for generating rough animation reference strips (Stage 1) and for
chat/vision LLM calls (Stage 2 grid generation + verification gates).
"""

from __future__ import annotations

from typing import Any

from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.providers.chat import ChatProvider


# Lazy imports for Azure-dependent providers
def __getattr__(name: str) -> Any:
    if name == "AzureChatProvider":
        from spriteforge.providers.azure_chat import AzureChatProvider

        return AzureChatProvider
    if name == "GPTImageProvider":
        from spriteforge.providers.gpt_image import GPTImageProvider

        return GPTImageProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AzureChatProvider",
    "ChatProvider",
    "GPTImageProvider",
    "ProviderError",
    "ReferenceProvider",
]
