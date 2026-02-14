"""SpriteForge error hierarchy.

All custom exceptions inherit from SpriteForgeError, enabling callers
to catch the base class for blanket error handling or specific
subclasses for targeted recovery.
"""


class SpriteForgeError(Exception):
    """Base exception for all SpriteForge errors."""


class ConfigError(SpriteForgeError):
    """Raised when configuration loading or validation fails."""


class PaletteError(SpriteForgeError):
    """Raised when palette operations fail (invalid symbols, swap mismatch)."""


class RenderError(SpriteForgeError):
    """Raised when grid rendering fails (bad dimensions, unknown symbols)."""


class GenerationError(SpriteForgeError):
    """Raised when LLM grid generation fails (parse error, API failure)."""


class GateError(SpriteForgeError):
    """Raised when a verification gate encounters an unrecoverable error."""


class RetryExhaustedError(SpriteForgeError):
    """Raised when all retry attempts are exhausted for a frame."""


class ProviderError(SpriteForgeError):
    """Raised when a reference image provider fails."""
