---
applyTo: '**'
---

# SpriteForge Project Memory

High-signal decisions + invariants for working on SpriteForge.

## Core Architecture

- Input: **one YAML config** (source of truth) + **one base reference PNG**.
- Output: spritesheet PNG assembled from **pixel-grids** rendered deterministically in Python.
- Stage 1 (reference strips): `generation.reference_model` (default `gpt-image-1.5`).
- Stage 2 (grid generation): `generation.grid_model` (default `gpt-5.2`) outputs JSON `{"grid": [64 strings of 64 chars]}`.
- Verification: `generation.gate_model` (default `gpt-5-mini`, temp 0.0) + programmatic checks.
- Auto-palette labeling: `generation.labeling_model` (default `gpt-5-nano`) when `generation.auto_palette=true` and `generation.semantic_labels=true`.

## Grid + Palette Invariants

- Grid is always **64×64**, represented as 64 strings of length 64.
- `.` is always transparent (implicit; not listed in YAML palette).
- `O` is outline; outline color comes from `palette.outline`.
- All other symbols must be unique single characters, defined in `palette.colors`.
- `generation.max_palette_colors` is limited to **≤23**.

## Azure Access

- Auth: `DefaultAzureCredential` for **all** providers — no API keys.
- Env vars: `AZURE_AI_PROJECT_ENDPOINT` (chat/vision via AI Foundry), `AZURE_OPENAI_GPT_IMAGE_ENDPOINT` (image generation base URL).
- GPT-Image-1.5 uses Entra ID bearer tokens via `get_bearer_token_provider` from `azure.identity.aio` (scope: `https://cognitiveservices.azure.com/.default`).
- Corporate policy (`MCAPSGov`) enforces `disableLocalAuth=true` on all Cognitive Services resources — API key auth is blocked at the tenant level.
- Model deployment names are configurable in YAML under `generation.*_model`.

## Errors (Semantics)

- Reference strip failures (Gate -1 after 3 attempts) → `ProviderError`.
- Frame failures (after 10 attempts) → `RetryExhaustedError`.

## Retry & Escalation

- Frame retries: max 10 attempts, 3 tiers:
	- Soft: 1–3 @ temp 1.0
	- Guided: 4–6 @ temp 0.7
	- Constrained: 7–10 @ temp 0.3
- Reference strip retries: max 3 attempts (no tier escalation).

## Workflow Factory

- `create_workflow()` wires separate providers for grid vs gates, and a reference provider.
- Always `await workflow.close()`; safe to call multiple times.

## Testing

- Integration tests use `@pytest.mark.integration` and make **real Azure calls**.
- They are opt-in via `SPRITEFORGE_RUN_INTEGRATION=1`, and auto-skip if `AZURE_AI_PROJECT_ENDPOINT` is unset or credentials unavailable.

## Response Format

- Chat calls enforce structured JSON with `response_format="json_object"`.
