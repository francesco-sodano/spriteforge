"""Tests for spriteforge.gates — multi-gate verification system."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from spriteforge.errors import GateError
from spriteforge.gates import (
    GateVerdict,
    LLMGateChecker,
    ProgrammaticChecker,
    parse_verdict_response,
)
from spriteforge.models import (
    AnimationDef,
    FrameContext,
    GenerationConfig,
    PaletteColor,
    PaletteConfig,
)
from spriteforge.palette import build_palette_map

from mock_chat_provider import MockChatProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal 1×1 transparent PNG (89 bytes) for use as image bytes in tests.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_valid_grid(symbol: str = ".", rows: int = 64, cols: int = 64) -> list[str]:
    """Create a uniform grid filled with a single symbol."""
    return [symbol * cols for _ in range(rows)]


@pytest.fixture()
def sample_palette() -> PaletteConfig:
    """A minimal palette for testing."""
    return PaletteConfig(
        outline=PaletteColor(element="Outline", symbol="O", r=20, g=40, b=40),
        colors=[
            PaletteColor(element="Skin", symbol="s", r=235, g=210, b=185),
            PaletteColor(element="Hair", symbol="h", r=220, g=185, b=90),
        ],
    )


@pytest.fixture()
def sample_animation() -> AnimationDef:
    """A minimal animation definition for testing."""
    return AnimationDef(
        name="idle",
        row=0,
        frames=6,
        timing_ms=150,
        prompt_context="Character standing in relaxed idle pose",
    )


@pytest.fixture()
def checker() -> ProgrammaticChecker:
    """A ProgrammaticChecker instance."""
    return ProgrammaticChecker()


# ---------------------------------------------------------------------------
# GateVerdict model
# ---------------------------------------------------------------------------


class TestGateVerdict:
    """Tests for the GateVerdict model."""

    def test_gate_verdict_creation(self) -> None:
        """Basic creation with all required fields."""
        verdict = GateVerdict(
            gate_name="test_gate",
            passed=True,
            confidence=0.95,
            feedback="All good",
        )
        assert verdict.gate_name == "test_gate"
        assert verdict.passed is True
        assert verdict.confidence == 0.95
        assert verdict.feedback == "All good"
        assert verdict.details == {}

    def test_gate_verdict_with_details(self) -> None:
        """Creation with details dict."""
        verdict = GateVerdict(
            gate_name="test_gate",
            passed=False,
            confidence=0.5,
            feedback="Issue found",
            details={"error": "bad pixel"},
        )
        assert verdict.details == {"error": "bad pixel"}


# ---------------------------------------------------------------------------
# ProgrammaticChecker — dimensions
# ---------------------------------------------------------------------------


class TestCheckDimensions:
    """Tests for ProgrammaticChecker.check_dimensions."""

    def test_check_dimensions_valid(self, checker: ProgrammaticChecker) -> None:
        """64×64 grid → passed."""
        grid = _make_valid_grid(".", 64, 64)
        verdict = checker.check_dimensions(grid)
        assert verdict.passed is True
        assert verdict.gate_name == "programmatic_dimensions"

    def test_check_dimensions_wrong_rows(self, checker: ProgrammaticChecker) -> None:
        """63 rows → failed with correct feedback."""
        grid = _make_valid_grid(".", 63, 64)
        verdict = checker.check_dimensions(grid)
        assert verdict.passed is False
        assert "63" in verdict.feedback
        assert "64" in verdict.feedback

    def test_check_dimensions_wrong_cols(self, checker: ProgrammaticChecker) -> None:
        """Row with 63 chars → failed."""
        grid = _make_valid_grid(".", 64, 64)
        grid[0] = "." * 63
        verdict = checker.check_dimensions(grid)
        assert verdict.passed is False
        assert "63" in verdict.feedback


# ---------------------------------------------------------------------------
# ProgrammaticChecker — valid symbols
# ---------------------------------------------------------------------------


class TestCheckValidSymbols:
    """Tests for ProgrammaticChecker.check_valid_symbols."""

    def test_check_valid_symbols_all_valid(
        self, checker: ProgrammaticChecker, sample_palette: PaletteConfig
    ) -> None:
        """All palette symbols → passed."""
        grid = _make_valid_grid(".", 64, 64)
        verdict = checker.check_valid_symbols(grid, sample_palette)
        assert verdict.passed is True

    def test_check_valid_symbols_invalid(
        self, checker: ProgrammaticChecker, sample_palette: PaletteConfig
    ) -> None:
        """Contains 'X' → failed, feedback mentions 'X'."""
        grid = _make_valid_grid(".", 64, 64)
        grid[0] = "X" + "." * 63
        verdict = checker.check_valid_symbols(grid, sample_palette)
        assert verdict.passed is False
        assert "X" in verdict.feedback


# ---------------------------------------------------------------------------
# ProgrammaticChecker — outline presence
# ---------------------------------------------------------------------------


class TestCheckOutlinePresence:
    """Tests for ProgrammaticChecker.check_outline_presence."""

    def test_check_outline_presence_has_outline(
        self, checker: ProgrammaticChecker
    ) -> None:
        """Grid with 'O' → passed."""
        grid = _make_valid_grid(".", 64, 64)
        grid[30] = "." * 30 + "O" + "." * 33
        verdict = checker.check_outline_presence(grid)
        assert verdict.passed is True

    def test_check_outline_presence_no_outline(
        self, checker: ProgrammaticChecker
    ) -> None:
        """Grid with only '.' and 's' → failed."""
        grid = _make_valid_grid(".", 64, 64)
        grid[30] = "s" * 64
        verdict = checker.check_outline_presence(grid)
        assert verdict.passed is False
        assert "outline" in verdict.feedback.lower() or "O" in verdict.feedback


# ---------------------------------------------------------------------------
# ProgrammaticChecker — not empty
# ---------------------------------------------------------------------------


class TestCheckNotEmpty:
    """Tests for ProgrammaticChecker.check_not_empty."""

    def test_check_not_empty_has_content(self, checker: ProgrammaticChecker) -> None:
        """Non-transparent pixels → passed."""
        grid = _make_valid_grid(".", 64, 64)
        grid[30] = "." * 30 + "s" + "." * 33
        verdict = checker.check_not_empty(grid)
        assert verdict.passed is True

    def test_check_not_empty_all_transparent(
        self, checker: ProgrammaticChecker
    ) -> None:
        """Only '.' → failed."""
        grid = _make_valid_grid(".", 64, 64)
        verdict = checker.check_not_empty(grid)
        assert verdict.passed is False
        assert "transparent" in verdict.feedback.lower()


# ---------------------------------------------------------------------------
# ProgrammaticChecker — feet position
# ---------------------------------------------------------------------------


class TestCheckFeetPosition:
    """Tests for ProgrammaticChecker.check_feet_position."""

    def test_check_feet_position_near_bottom(
        self, checker: ProgrammaticChecker
    ) -> None:
        """Non-transparent pixels near row 55 → passed."""
        grid = _make_valid_grid(".", 64, 64)
        grid[55] = "." * 30 + "s" + "." * 33
        verdict = checker.check_feet_position(grid)
        assert verdict.passed is True

    def test_check_feet_position_at_top(self, checker: ProgrammaticChecker) -> None:
        """Character only in rows 0–10 → failed."""
        grid = _make_valid_grid(".", 64, 64)
        for i in range(11):
            grid[i] = "s" * 64
        verdict = checker.check_feet_position(grid)
        assert verdict.passed is False
        assert "foot" in verdict.feedback.lower() or "55" in verdict.feedback

    def test_feet_position_default_64x64(self, checker: ProgrammaticChecker) -> None:
        """Default 64×64 frame → foot row at int(64 * 0.875) = 56."""
        grid = _make_valid_grid(".", 64, 64)
        # Place content at row 56 (int(64 * FEET_ROW_RATIO) = 56)
        grid[56] = "." * 30 + "s" + "." * 33
        verdict = checker.check_feet_position(grid)
        assert verdict.passed is True
        # Verify default foot row is 56 (int(64 * 0.875) = 56)
        assert "56" in verdict.feedback

    def test_feet_position_32x32_frame(self, checker: ProgrammaticChecker) -> None:
        """32×32 frame → foot zone is proportionally smaller."""
        grid = _make_valid_grid(".", 32, 32)
        # For 32px: expected_foot_row = int(32 * 0.875) = 28
        # window = max(3, int(32 * 0.08)) = max(3, 2) = 3
        # foot_zone = 28 ± 3 = rows 25–31
        grid[28] = "." * 15 + "s" + "." * 16
        verdict = checker.check_feet_position(grid, frame_height=32)
        assert verdict.passed is True
        assert "28" in verdict.feedback

    def test_feet_position_128x128_frame(self, checker: ProgrammaticChecker) -> None:
        """128×128 frame → foot zone is proportionally larger."""
        grid = _make_valid_grid(".", 128, 128)
        # For 128px: expected_foot_row = int(128 * 0.875) = 112
        # window = max(3, int(128 * 0.08)) = max(3, 10) = 10
        # foot_zone = 112 ± 10 = rows 102–122
        grid[112] = "." * 60 + "s" + "." * 67
        verdict = checker.check_feet_position(grid, frame_height=128)
        assert verdict.passed is True
        assert "112" in verdict.feedback

    def test_feet_position_custom_foot_row(self, checker: ProgrammaticChecker) -> None:
        """Override foot row → window scales correctly."""
        grid = _make_valid_grid(".", 64, 64)
        # Custom foot row at 40, frame_height 64
        # window = max(3, int(64 * 0.08)) = 5
        # foot_zone = 40 ± 5 = rows 35–45
        grid[40] = "." * 30 + "s" + "." * 33
        verdict = checker.check_feet_position(
            grid, expected_foot_row=40, frame_height=64
        )
        assert verdict.passed is True
        assert "40" in verdict.feedback


# ---------------------------------------------------------------------------
# ProgrammaticChecker — run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    """Tests for ProgrammaticChecker.run_all."""

    def test_run_all_returns_all_verdicts(
        self,
        checker: ProgrammaticChecker,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """Returns 5 verdicts."""
        grid = _make_valid_grid(".", 64, 64)
        context = FrameContext(
            palette=sample_palette,
            palette_map=build_palette_map(sample_palette),
            generation=GenerationConfig(),
            frame_width=64,
            frame_height=64,
            animation=sample_animation,
            spritesheet_columns=14,
        )
        verdicts = checker.run_all(grid, context)
        assert len(verdicts) == 5
        assert all(isinstance(v, GateVerdict) for v in verdicts)

    def test_run_all_uses_config_feet_row(
        self,
        checker: ProgrammaticChecker,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """run_all respects GenerationConfig.feet_row instead of hardcoding."""
        grid = _make_valid_grid(".", 64, 64)
        # Place content only at row 40 — default feet_row=56 would fail
        grid[40] = "." * 30 + "s" + "." * 33
        context = FrameContext(
            palette=sample_palette,
            palette_map=build_palette_map(sample_palette),
            generation=GenerationConfig(feet_row=40),
            frame_width=64,
            frame_height=64,
            animation=sample_animation,
            spritesheet_columns=14,
        )
        verdicts = checker.run_all(grid, context)
        feet_verdict = [
            v for v in verdicts if v.gate_name == "programmatic_feet_position"
        ][0]
        assert feet_verdict.passed is True
        assert "40" in feet_verdict.feedback


# ---------------------------------------------------------------------------
# ProgrammaticChecker — variable frame sizes
# ---------------------------------------------------------------------------


class TestProgrammaticCheckerVariableSize:
    """Tests for ProgrammaticChecker with non-default frame dimensions."""

    def test_programmatic_checker_default_64x64(
        self, checker: ProgrammaticChecker
    ) -> None:
        """Default params check against 64 (backward compat)."""
        grid = _make_valid_grid(".", 64, 64)
        verdict = checker.check_dimensions(grid)
        assert verdict.passed is True

    def test_programmatic_checker_32x32(self, checker: ProgrammaticChecker) -> None:
        """32×32 grid passes dimension check."""
        grid = _make_valid_grid(".", 32, 32)
        verdict = checker.check_dimensions(grid, expected_rows=32, expected_cols=32)
        assert verdict.passed is True

    def test_programmatic_checker_feet_row_scales(
        self,
        checker: ProgrammaticChecker,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """feet_row must be set in config for non-64px frames."""
        grid = _make_valid_grid(".", 32, 32)
        # Place non-transparent pixels near row 28 (expected feet for 32px)
        grid[27] = "." * 15 + "s" + "." * 16
        context = FrameContext(
            palette=sample_palette,
            palette_map=build_palette_map(sample_palette),
            generation=GenerationConfig(feet_row=28),
            frame_width=32,
            frame_height=32,
            animation=sample_animation,
            spritesheet_columns=14,
        )
        verdicts = checker.run_all(grid, context)
        feet_verdict = [
            v for v in verdicts if v.gate_name == "programmatic_feet_position"
        ][0]
        assert feet_verdict.passed is True

    def test_programmatic_checker_feet_row_scales_128(
        self,
        checker: ProgrammaticChecker,
        sample_palette: PaletteConfig,
        sample_animation: AnimationDef,
    ) -> None:
        """feet_row must be set in config for 128px frames."""
        grid = _make_valid_grid(".", 128, 128)
        # Place non-transparent pixels near row 112
        grid[112] = "." * 60 + "s" + "." * 67
        context = FrameContext(
            palette=sample_palette,
            palette_map=build_palette_map(sample_palette),
            generation=GenerationConfig(feet_row=112),
            frame_width=128,
            frame_height=128,
            animation=sample_animation,
            spritesheet_columns=14,
        )
        verdicts = checker.run_all(grid, context)
        feet_verdict = [
            v for v in verdicts if v.gate_name == "programmatic_feet_position"
        ][0]
        assert feet_verdict.passed is True

    def test_programmatic_checker_wrong_dimensions_with_custom_size(
        self, checker: ProgrammaticChecker
    ) -> None:
        """Wrong grid size fails with correct message for custom dimensions."""
        grid = _make_valid_grid(".", 30, 30)
        verdict = checker.check_dimensions(grid, expected_rows=32, expected_cols=32)
        assert verdict.passed is False
        assert "30" in verdict.feedback
        assert "32" in verdict.feedback


# ---------------------------------------------------------------------------
# parse_verdict_response
# ---------------------------------------------------------------------------


class TestParseVerdictResponse:
    """Tests for parse_verdict_response."""

    def test_parse_verdict_valid_json(self) -> None:
        """Valid JSON → correct GateVerdict."""
        response = json.dumps(
            {"passed": True, "confidence": 0.95, "feedback": "Looks great"}
        )
        verdict = parse_verdict_response(response, "gate_0")
        assert verdict.passed is True
        assert verdict.confidence == 0.95
        assert verdict.feedback == "Looks great"
        assert verdict.gate_name == "gate_0"

    def test_parse_verdict_with_fences(self) -> None:
        """JSON in markdown fences → still parses."""
        inner = json.dumps(
            {"passed": False, "confidence": 0.3, "feedback": "Too blurry"}
        )
        response = f"```json\n{inner}\n```"
        verdict = parse_verdict_response(response, "gate_1")
        assert verdict.passed is False
        assert verdict.confidence == 0.3
        assert verdict.feedback == "Too blurry"

    def test_parse_verdict_invalid_defaults_to_fail(self) -> None:
        """Garbage text → GateVerdict(passed=False)."""
        verdict = parse_verdict_response("This is not JSON at all", "gate_0")
        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert verdict.details.get("parser_fallback") == "json_decode_error"

    def test_parse_verdict_invalid_logs_gate_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fallback logging includes the gate identifier for diagnostics."""
        with caplog.at_level("WARNING"):
            parse_verdict_response("not-json", "gate_2")

        assert "gate_2" in caplog.text
        assert "parse fallback" in caplog.text.lower()

    def test_parse_verdict_non_object_defaults_to_fail(self) -> None:
        """JSON arrays are rejected and fail closed with fallback details."""
        verdict = parse_verdict_response('["not", "an", "object"]', "gate_1")
        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert verdict.details.get("parser_fallback") == "non_object_json"

    def test_parse_verdict_extracts_problematic_frame_indices(self) -> None:
        response = json.dumps(
            {
                "passed": False,
                "confidence": 0.4,
                "feedback": "Regenerate frame 1 and 3",
                "problematic_frame_indices": [3, 1, 3],
            }
        )
        verdict = parse_verdict_response(response, "gate_3a")
        assert verdict.details.get("problematic_frame_indices") == [1, 3]


# ---------------------------------------------------------------------------
# LLM gate checker — mocked API
# ---------------------------------------------------------------------------


class TestLLMGateChecker:
    """Tests for LLMGateChecker (mocked chat provider)."""

    @pytest.fixture()
    def valid_verdict_json(self) -> str:
        return json.dumps(
            {"passed": True, "confidence": 0.9, "feedback": "Good quality"}
        )

    @pytest.fixture()
    def mock_provider(self, valid_verdict_json: str) -> MockChatProvider:
        return MockChatProvider(responses=[valid_verdict_json])

    @pytest.fixture()
    def gate_checker(self, mock_provider: MockChatProvider) -> LLMGateChecker:
        return LLMGateChecker(chat_provider=mock_provider)

    @pytest.mark.asyncio
    async def test_gate_minus_1_sends_images(
        self,
        gate_checker: LLMGateChecker,
        mock_provider: MockChatProvider,
        sample_animation: AnimationDef,
    ) -> None:
        """Chat provider receives reference strip + base ref."""
        verdict = await gate_checker.gate_minus_1(
            reference_strip=_TINY_PNG,
            base_reference=_TINY_PNG,
            animation=sample_animation,
        )

        assert verdict.gate_name == "gate_minus_1"
        assert verdict.passed is True
        assert len(mock_provider._call_history) == 1

        messages = mock_provider._call_history[0]["messages"]
        content = messages[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_gate_0_sends_rendered_and_reference(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Both images in request."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        verdict = await checker.gate_0(
            rendered_frame=_TINY_PNG,
            reference_frame=_TINY_PNG,
        )

        assert verdict.gate_name == "gate_0"

        messages = mock._call_history[0]["messages"]
        content = messages[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_gate_1_sends_rendered_and_anchor(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Both images in request."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        verdict = await checker.gate_1(
            rendered_frame=_TINY_PNG,
            anchor_frame=_TINY_PNG,
        )

        assert verdict.gate_name == "gate_1"

        messages = mock._call_history[0]["messages"]
        content = messages[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_gate_2_sends_rendered_and_prev(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Both images in request."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        verdict = await checker.gate_2(
            rendered_frame=_TINY_PNG,
            prev_frame=_TINY_PNG,
        )

        assert verdict.gate_name == "gate_2"

        messages = mock._call_history[0]["messages"]
        content = messages[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_gate_3a_sends_row_strip_and_reference(
        self,
        sample_animation: AnimationDef,
        valid_verdict_json: str,
    ) -> None:
        """Both strip images in request."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        verdict = await checker.gate_3a(
            rendered_row_strip=_TINY_PNG,
            reference_strip=_TINY_PNG,
            animation=sample_animation,
        )

        assert verdict.gate_name == "gate_3a"

        messages = mock._call_history[0]["messages"]
        content = messages[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 2

    @pytest.mark.asyncio
    async def test_gate_uses_temperature_zero(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Chat provider called with temperature=0.0."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_0(
            rendered_frame=_TINY_PNG,
            reference_frame=_TINY_PNG,
        )

        assert mock._call_history[0]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_gate_checker_uses_chat_provider(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Verify calls go through ChatProvider."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_0(
            rendered_frame=_TINY_PNG,
            reference_frame=_TINY_PNG,
        )

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Tests: LLMGateChecker response_format parameter
# ---------------------------------------------------------------------------


class TestLLMGateCheckerResponseFormat:
    """Tests for response_format parameter usage in gate checks."""

    @pytest.fixture()
    def valid_verdict_json(self) -> str:
        """Valid gate verdict JSON for testing."""
        return json.dumps(
            {
                "passed": True,
                "confidence": 0.95,
                "feedback": "Test passed",
            }
        )

    @pytest.fixture()
    def sample_animation(self) -> AnimationDef:
        """A minimal animation definition for testing."""
        return AnimationDef(
            name="idle",
            row=0,
            frames=6,
            timing_ms=150,
        )

    @pytest.mark.asyncio
    async def test_gate_minus_1_passes_response_format(
        self,
        valid_verdict_json: str,
        sample_animation: AnimationDef,
    ) -> None:
        """Gate -1 passes response_format='json_object' to chat()."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_minus_1(
            reference_strip=_TINY_PNG,
            base_reference=_TINY_PNG,
            animation=sample_animation,
        )

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["response_format"] == "json_object"


class TestLLMGateCheckerTimeouts:
    @pytest.fixture()
    def valid_verdict_json(self) -> str:
        return json.dumps(
            {
                "passed": True,
                "confidence": 0.95,
                "feedback": "Test passed",
            }
        )

    @pytest.fixture()
    def sample_animation(self) -> AnimationDef:
        return AnimationDef(
            name="idle",
            row=0,
            frames=6,
            timing_ms=150,
        )

    @pytest.mark.asyncio
    async def test_gate_timeout_raises_gate_error(self) -> None:
        class HangingProvider(MockChatProvider):
            async def chat(
                self,
                messages: list[dict[str, Any]],
                temperature: float = 1.0,
                response_format: str | None = None,
            ) -> str:
                await asyncio.Event().wait()
                return ""

        checker = LLMGateChecker(
            chat_provider=HangingProvider(),
            request_timeout_seconds=0.01,
        )

        with pytest.raises(GateError, match="timed out"):
            await checker.gate_0(
                rendered_frame=_TINY_PNG,
                reference_frame=_TINY_PNG,
            )

    @pytest.mark.asyncio
    async def test_gate_0_passes_response_format(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Gate 0 passes response_format='json_object' to chat()."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_0(
            rendered_frame=_TINY_PNG,
            reference_frame=_TINY_PNG,
        )

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["response_format"] == "json_object"

    @pytest.mark.asyncio
    async def test_gate_1_passes_response_format(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Gate 1 passes response_format='json_object' to chat()."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_1(
            rendered_frame=_TINY_PNG,
            anchor_frame=_TINY_PNG,
        )

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["response_format"] == "json_object"

    @pytest.mark.asyncio
    async def test_gate_2_passes_response_format(
        self,
        valid_verdict_json: str,
    ) -> None:
        """Gate 2 passes response_format='json_object' to chat()."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_2(
            rendered_frame=_TINY_PNG,
            prev_frame=_TINY_PNG,
        )

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["response_format"] == "json_object"

    @pytest.mark.asyncio
    async def test_gate_3a_passes_response_format(
        self,
        valid_verdict_json: str,
        sample_animation: AnimationDef,
    ) -> None:
        """Gate 3A passes response_format='json_object' to chat()."""
        mock = MockChatProvider(responses=[valid_verdict_json])
        checker = LLMGateChecker(chat_provider=mock)

        await checker.gate_3a(
            rendered_row_strip=_TINY_PNG,
            reference_strip=_TINY_PNG,
            animation=sample_animation,
        )

        assert len(mock._call_history) == 1
        assert mock._call_history[0]["response_format"] == "json_object"
