"""Prompts for preprocessor palette labeling."""

from __future__ import annotations

PALETTE_LABELING_PROMPT = """You are analyzing a pixel-art character sprite to label its color palette.

Character description: {character_description}

The following colors were extracted from the character image (excluding outline and transparent):
{color_list}

Look at the attached character image and assign a semantic label to each color based on what body part, clothing piece, or element it corresponds to.

Return ONLY a JSON object with a "labels" key containing an array of strings, one label per color, in the same order:
{{"labels": ["Skin", "Hair", "Armor", ...]}}

Rules:
- Labels should be short (1-3 words): "Skin", "Dark Hair", "Steel Armor"
- Use character-appropriate terms from the description
- If a color's purpose is ambiguous, label it by its visual role: "Accent", "Shadow", "Highlight"
- Return exactly {color_count} labels
"""
