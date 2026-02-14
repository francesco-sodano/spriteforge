"""Reference image providers for Stage 1 rough reference generation.

This package contains the abstract base class and concrete implementations
for generating rough animation reference strips used as visual targets in
the Stage 2 pixel-precise grid generation.
"""

from __future__ import annotations

from spriteforge.providers._base import ProviderError, ReferenceProvider
from spriteforge.providers.gpt_image import GPTImageProvider

__all__ = [
    "GPTImageProvider",
    "ProviderError",
    "ReferenceProvider",
]
