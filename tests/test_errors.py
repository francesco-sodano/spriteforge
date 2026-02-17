"""Tests for spriteforge.errors — error hierarchy."""

from __future__ import annotations

import pytest

from spriteforge.errors import (
    BudgetExhaustedError,
    ConfigError,
    GateError,
    GenerationError,
    PaletteError,
    ProviderError,
    RenderError,
    RetryExhaustedError,
    SpriteForgeError,
)


class TestErrorHierarchy:
    """Verify the SpriteForge error inheritance tree."""

    def test_spriteforge_error_is_base(self) -> None:
        """SpriteForgeError is a subclass of Exception."""
        assert issubclass(SpriteForgeError, Exception)

    def test_all_errors_inherit_from_base(self) -> None:
        """All custom errors inherit from SpriteForgeError."""
        for cls in (
            ConfigError,
            PaletteError,
            RenderError,
            GenerationError,
            GateError,
            RetryExhaustedError,
            ProviderError,
            BudgetExhaustedError,
        ):
            assert issubclass(cls, SpriteForgeError), f"{cls.__name__} missing base"

    def test_catch_base_catches_all(self) -> None:
        """try/except SpriteForgeError catches any subclass."""
        for cls in (
            ConfigError,
            PaletteError,
            RenderError,
            GenerationError,
            GateError,
            RetryExhaustedError,
            ProviderError,
            BudgetExhaustedError,
        ):
            with pytest.raises(SpriteForgeError):
                raise cls("test")

    def test_specific_errors_distinguishable(self) -> None:
        """ConfigError is not PaletteError and vice-versa."""
        assert not issubclass(ConfigError, PaletteError)
        assert not issubclass(PaletteError, ConfigError)
        assert not issubclass(GenerationError, ProviderError)
        assert not issubclass(ProviderError, GenerationError)

    def test_error_messages_preserved(self) -> None:
        """Error message string is accessible via str(e)."""
        msg = "something went wrong"
        err = GenerationError(msg)
        assert str(err) == msg

        err2 = ConfigError("bad config")
        assert str(err2) == "bad config"
