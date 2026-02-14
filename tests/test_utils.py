"""Tests for spriteforge.utils — shared utility functions."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from spriteforge.utils import (
    image_to_base64,
    image_to_data_url,
    parse_json_from_llm,
    strip_code_fences,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 4, height: int = 4) -> Image.Image:
    """Create a small RGBA PIL Image for testing."""
    return Image.new("RGBA", (width, height), (255, 0, 0, 255))


def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Create PNG bytes from a small image."""
    img = _make_pil_image(width, height)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# image_to_data_url
# ---------------------------------------------------------------------------


class TestImageToDataUrl:
    """Tests for image_to_data_url()."""

    def test_image_to_data_url_from_pil_image(self) -> None:
        """PIL Image → valid data URL string."""
        img = _make_pil_image()
        url = image_to_data_url(img)
        assert url.startswith("data:image/png;base64,")
        # Decode the base64 and verify it's valid PNG
        b64_part = url.split(",", 1)[1]
        raw = base64.b64decode(b64_part)
        assert raw[:4] == b"\x89PNG"

    def test_image_to_data_url_from_bytes(self) -> None:
        """PNG bytes → valid data URL string."""
        png_bytes = _make_png_bytes()
        url = image_to_data_url(png_bytes)
        assert url.startswith("data:image/png;base64,")
        b64_part = url.split(",", 1)[1]
        raw = base64.b64decode(b64_part)
        assert raw == png_bytes

    def test_image_to_data_url_custom_media_type(self) -> None:
        """Custom media type in URL."""
        url = image_to_data_url(b"\x89PNG", media_type="image/jpeg")
        assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# image_to_base64
# ---------------------------------------------------------------------------


class TestImageToBase64:
    """Tests for image_to_base64()."""

    def test_image_to_base64_from_pil_image(self) -> None:
        """PIL Image → base64 string."""
        img = _make_pil_image()
        b64 = image_to_base64(img)
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"

    def test_image_to_base64_from_bytes(self) -> None:
        """Raw bytes → base64 string."""
        data = b"hello world"
        b64 = image_to_base64(data)
        assert base64.b64decode(b64) == data


# ---------------------------------------------------------------------------
# strip_code_fences
# ---------------------------------------------------------------------------


class TestStripCodeFences:
    """Tests for strip_code_fences()."""

    def test_strip_code_fences_json(self) -> None:
        """```json\\n{...}\\n``` → {..}"""
        text = '```json\n{"key": "value"}\n```'
        assert strip_code_fences(text) == '{"key": "value"}'

    def test_strip_code_fences_plain(self) -> None:
        """```\\n{...}\\n``` → {..}"""
        text = '```\n{"key": "value"}\n```'
        assert strip_code_fences(text) == '{"key": "value"}'

    def test_strip_code_fences_no_fences(self) -> None:
        """Plain text → unchanged."""
        text = '{"key": "value"}'
        assert strip_code_fences(text) == text

    def test_strip_code_fences_nested(self) -> None:
        """Only outermost fences stripped."""
        text = '```json\n{"code": "```inner```"}\n```'
        result = strip_code_fences(text)
        # The inner content should remain (regex is non-greedy)
        assert "code" in result


# ---------------------------------------------------------------------------
# parse_json_from_llm
# ---------------------------------------------------------------------------


class TestParseJsonFromLlm:
    """Tests for parse_json_from_llm()."""

    def test_parse_json_from_llm_valid(self) -> None:
        """Valid JSON → parsed dict."""
        text = '{"passed": true, "confidence": 0.9}'
        result = parse_json_from_llm(text)
        assert result == {"passed": True, "confidence": 0.9}

    def test_parse_json_from_llm_with_fences(self) -> None:
        """Fenced JSON → parsed dict."""
        text = '```json\n{"passed": true}\n```'
        result = parse_json_from_llm(text)
        assert result == {"passed": True}

    def test_parse_json_from_llm_trailing_comma(self) -> None:
        """JSON with trailing comma → handled gracefully."""
        text = '{"key": "value",}'
        result = parse_json_from_llm(text)
        assert result == {"key": "value"}

    def test_parse_json_from_llm_invalid(self) -> None:
        """Non-JSON text → raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_json_from_llm("this is not json at all")
