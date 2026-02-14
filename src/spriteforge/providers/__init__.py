"""Reference image providers and chat/vision LLM providers.

This package contains abstract base classes and concrete implementations
for generating rough animation reference strips (Stage 1) and for
chat/vision LLM calls (Stage 2 grid generation + verification gates).
"""

from __future__ import annotations

from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.providers.azure_chat import AzureChatProvider
from spriteforge.providers.chat import ChatProvider
from spriteforge.providers.gpt_image import GPTImageProvider

__all__ = [
    "AzureChatProvider",
    "ChatProvider",
    "GPTImageProvider",
    "ProviderError",
    "ReferenceProvider",
]
