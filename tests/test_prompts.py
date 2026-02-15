"""Tests for spriteforge.prompts — prompt organization package."""

from __future__ import annotations

from spriteforge.prompts.generator import (
    GRID_SYSTEM_PROMPT,
    build_anchor_frame_prompt,
    build_frame_prompt,
)
from spriteforge.prompts.gates import (
    GATE_0_PROMPT,
    GATE_1_PROMPT,
    GATE_2_PROMPT,
    GATE_3A_PROMPT,
    GATE_MINUS_1_PROMPT,
    GATE_VERDICT_SCHEMA,
)
from spriteforge.prompts.providers import build_reference_prompt
from spriteforge.prompts.retry import (
    build_constrained_guidance,
    build_guided_guidance,
    build_soft_guidance,
)

from spriteforge.gates import GateVerdict
from spriteforge.models import AnimationDef, CharacterConfig

# ---------------------------------------------------------------------------
# Generator prompts
# ---------------------------------------------------------------------------


class TestGeneratorPrompts:
    """Tests for prompt constants and builders in prompts.generator."""

    def test_grid_system_prompt_not_empty(self) -> None:
        """System prompt is a non-empty string."""
        assert isinstance(GRID_SYSTEM_PROMPT, str)
        assert len(GRID_SYSTEM_PROMPT) > 0

    def test_build_anchor_frame_prompt_contains_palette(self) -> None:
        """Palette spec appears in output when provided in quantized section."""
        prompt = build_anchor_frame_prompt(
            animation_name="IDLE",
            animation_context="Character standing still",
            frame_description="Standing neutral pose",
            quantized_section="palette info here",
        )
        assert "palette info here" in prompt

    def test_build_anchor_frame_prompt_contains_animation_name(self) -> None:
        """Animation name appears in output."""
        prompt = build_anchor_frame_prompt(
            animation_name="IDLE",
            animation_context="Standing still",
            frame_description="Neutral",
            quantized_section="",
        )
        assert "IDLE" in prompt

    def test_build_frame_prompt_contains_frame_index(self) -> None:
        """Frame index and total appear in output."""
        prompt = build_frame_prompt(
            frame_index=3,
            animation_name="WALK",
            animation_context="Walking forward",
            frame_description="Left foot forward",
            additional_guidance="",
        )
        assert "3" in prompt
        assert "WALK" in prompt

    def test_build_frame_prompt_includes_prev_frame(self) -> None:
        """Additional guidance included when provided."""
        prompt = build_frame_prompt(
            frame_index=2,
            animation_name="WALK",
            animation_context="Walking",
            frame_description="Step",
            additional_guidance="Focus on arm position",
        )
        assert "Focus on arm position" in prompt

    def test_build_frame_prompt_excludes_prev_frame_when_none(self) -> None:
        """No prev frame section when additional_guidance is empty."""
        prompt = build_frame_prompt(
            frame_index=1,
            animation_name="WALK",
            animation_context="Walking",
            frame_description="Step",
            additional_guidance="",
        )
        assert "Previous frame" not in prompt


# ---------------------------------------------------------------------------
# Gate prompts
# ---------------------------------------------------------------------------


class TestGatePrompts:
    """Tests for prompt constants and builders in prompts.gates."""

    def test_gate_minus1_prompt_contains_animation(self) -> None:
        """Animation name placeholder in gate -1 prompt."""
        prompt = GATE_MINUS_1_PROMPT.format(
            expected_frames=6,
            animation_name="IDLE",
            animation_context="Standing still",
            verdict_schema=GATE_VERDICT_SCHEMA,
        )
        assert "IDLE" in prompt

    def test_gate0_prompt_contains_frame_info(self) -> None:
        """Gate 0 prompt contains frame-related structure."""
        prompt = GATE_0_PROMPT.format(
            frame_description_section="Expected: sword raised",
            verdict_schema=GATE_VERDICT_SCHEMA,
        )
        assert "sword raised" in prompt

    def test_gate1_prompt_mentions_anchor(self) -> None:
        """Gate 1 prompt references 'anchor'."""
        prompt = GATE_1_PROMPT.format(verdict_schema=GATE_VERDICT_SCHEMA)
        assert "anchor" in prompt.lower()

    def test_gate2_prompt_mentions_continuity(self) -> None:
        """Gate 2 prompt references 'continuity' or 'previous'."""
        prompt = GATE_2_PROMPT.format(verdict_schema=GATE_VERDICT_SCHEMA)
        text_lower = prompt.lower()
        assert "continuity" in text_lower or "previous" in text_lower

    def test_gate3a_prompt_mentions_coherence(self) -> None:
        """Gate 3A prompt references 'coherence' or 'sequence'."""
        prompt = GATE_3A_PROMPT.format(
            animation_name="WALK",
            animation_context="Walking forward",
            verdict_schema=GATE_VERDICT_SCHEMA,
        )
        text_lower = prompt.lower()
        assert "coherence" in text_lower or "sequence" in text_lower

    def test_gate_system_prompt_not_empty(self) -> None:
        """Gate verdict schema is non-empty."""
        assert isinstance(GATE_VERDICT_SCHEMA, str)
        assert len(GATE_VERDICT_SCHEMA) > 0


# ---------------------------------------------------------------------------
# Retry prompts
# ---------------------------------------------------------------------------


class TestRetryPrompts:
    """Tests for prompt builders in prompts.retry."""

    def test_build_soft_guidance_includes_dimensions(self) -> None:
        """Soft guidance mentions grid dimensions."""
        guidance = build_soft_guidance()
        assert "64" in guidance

    def test_build_soft_guidance_uses_dimensions(self) -> None:
        """Soft guidance mentions palette and outline."""
        guidance = build_soft_guidance()
        assert "palette" in guidance.lower()
        assert "outline" in guidance.lower()

    def test_build_constrained_guidance_most_restrictive(self) -> None:
        """Constrained text is more directive than soft."""
        soft = build_soft_guidance()
        constrained = build_constrained_guidance(
            current_attempt=8,
            accumulated_feedback=["Bad anatomy", "Missing outline"],
        )
        # Constrained should be longer and more directive
        assert len(constrained) > len(soft)
        assert "CRITICAL" in constrained
        assert "Bad anatomy" in constrained
        assert "Missing outline" in constrained

    def test_build_guided_guidance_includes_gate_feedback(self) -> None:
        """Guided guidance includes gate names and feedback from failures."""
        verdicts = [
            GateVerdict(
                gate_name="gate_0",
                passed=False,
                confidence=0.3,
                feedback="Arm position wrong",
            ),
            GateVerdict(
                gate_name="gate_1",
                passed=False,
                confidence=0.4,
                feedback="Hair color inconsistent",
            ),
        ]
        guidance = build_guided_guidance(verdicts)
        assert "gate_0" in guidance
        assert "Arm position wrong" in guidance
        assert "gate_1" in guidance
        assert "Hair color inconsistent" in guidance


# ---------------------------------------------------------------------------
# Provider prompts
# ---------------------------------------------------------------------------


class TestProviderPrompts:
    """Tests for prompt builders in prompts.providers."""

    def test_build_reference_prompt_contains_character(self) -> None:
        """Character description in output."""
        animation = AnimationDef(
            name="WALK",
            row=1,
            frames=8,
            timing_ms=100,
            prompt_context="Walking forward",
        )
        character = CharacterConfig(
            name="Theron Ashblade",
            character_class="Warrior",
            description="A battle-scarred warrior",
        )
        prompt = build_reference_prompt(
            animation, character, character_description="Tall and strong"
        )
        assert "Theron Ashblade" in prompt
        assert "Tall and strong" in prompt

    def test_build_reference_prompt_contains_frame_count(self) -> None:
        """Frame count in output."""
        animation = AnimationDef(
            name="IDLE",
            row=0,
            frames=6,
            timing_ms=150,
        )
        character = CharacterConfig(name="Test NPC")
        prompt = build_reference_prompt(animation, character)
        assert "6 frames" in prompt


# ---------------------------------------------------------------------------
# Preprocessor prompts
# ---------------------------------------------------------------------------


class TestPreprocessorPrompts:
    """Tests for prompt constants in prompts.preprocessor."""

    def test_palette_labeling_prompt_contains_colors(self) -> None:
        """Prompt includes RGB values."""
        from spriteforge.prompts.preprocessor import PALETTE_LABELING_PROMPT

        color_list = "1. RGB(235, 210, 185)\n2. RGB(220, 185, 90)"
        prompt = PALETTE_LABELING_PROMPT.format(
            character_description="A warrior",
            color_list=color_list,
            color_count=2,
        )
        assert "235, 210, 185" in prompt
        assert "220, 185, 90" in prompt

    def test_palette_labeling_prompt_contains_description(self) -> None:
        """Prompt includes character description."""
        from spriteforge.prompts.preprocessor import PALETTE_LABELING_PROMPT

        color_list = "1. RGB(255, 0, 0)"
        prompt = PALETTE_LABELING_PROMPT.format(
            character_description="A goblin with green skin",
            color_list=color_list,
            color_count=1,
        )
        assert "goblin with green skin" in prompt

    def test_palette_labeling_prompt_contains_count(self) -> None:
        """Prompt includes expected label count."""
        from spriteforge.prompts.preprocessor import PALETTE_LABELING_PROMPT

        color_list = "1. RGB(255, 0, 0)\n2. RGB(0, 255, 0)\n3. RGB(0, 0, 255)"
        prompt = PALETTE_LABELING_PROMPT.format(
            character_description="A warrior",
            color_list=color_list,
            color_count=3,
        )
        assert "3 labels" in prompt
