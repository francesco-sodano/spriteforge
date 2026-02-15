---
applyTo: '**'
---

# SpriteForge Project Memory

This file contains key architectural decisions, conventions, and context for AI agents working on the SpriteForge project.

## Four-Model Tiered Architecture (Decision)

SpriteForge uses a four-model tiered architecture optimized for cost and quality:

| Role | Deployment Name | Model | Usage |
|------|----------------|-------|-------|
| Grid Generation (Stage 2) | `gpt-5.2` | GPT-5.2 | Anchor frame + creative frames (~25% of calls) |
| Gate Verification | `gpt-5-mini` | GPT-5-mini | Gates 0, 1, 2, 3A, -1 (~75% of calls) |
| Semantic Labeling | `gpt-5-nano` | GPT-5-nano | Auto-palette color naming (1 call per character) |
| Reference Generation (Stage 1) | `gpt-image-1.5` | GPT-Image-1.5 | Rough reference strips |

All four models are deployed in the same Azure AI Foundry project and accessed via `DefaultAzureCredential`. Model selection is configured per-character in the YAML config's `generation:` section.

**Historical note:** The project initially used a two-model architecture (GPT-Image-1.5 + Claude Opus 4.6). The shift to the tiered GPT-5 family was driven by cost optimization (verification gates are 75% of LLM calls) and unified Azure deployment.

## Semantic Palette Labeling (Decision)

When `generation.auto_palette` is enabled, SpriteForge uses **GPT-5-nano** to generate semantic color labels (e.g., "Skin", "Hair", "Armor") instead of generic names like "Color 1", "Color 2".

**Why LLM over heuristics:**
- HSL-based heuristics (e.g., "warm brown → skin") are unreliable across diverse character types (aliens, robots, fantasy creatures)
- LLM sees the full character reference image + receives character description as context
- GPT-5-nano is cheap enough (~$0.00001 per call) that the cost is negligible for one call per character
- Fallback: If LLM labeling fails, `_describe_color()` provides HSL-based descriptive names (e.g., "dark greenish-blue")

**Implementation:** `label_palette_colors_with_llm()` in `preprocessor.py` (lines 188-254).

## Azure Infrastructure

### Deployed Models

All models are deployed in a single Azure AI Foundry project:

| Deployment Name | Model | Purpose |
|----------------|-------|---------|
| `gpt-5.2` | GPT-5.2 | Grid generation (Stage 2) |
| `gpt-5-mini` | GPT-5-mini | Verification gates |
| `gpt-5-nano` | GPT-5-nano | Semantic palette labeling |
| `gpt-image-1.5` | GPT-Image-1.5 | Reference image generation (Stage 1) |

### Authentication

- All Azure access uses `DefaultAzureCredential` (no API keys)
- Single environment variable: `AZURE_AI_PROJECT_ENDPOINT`
- All providers (AzureChatProvider, GPTImageProvider) share the same credential instance

## Error Semantics

- **Reference strip generation failures** → Raise `ProviderError` (not `RetryExhaustedError`) when Gate -1 fails after 3 attempts
- **Frame generation failures** → Raise `RetryExhaustedError` (not `GenerationError`) when all 10 retry attempts are exhausted
- `RetryExhaustedError` includes: `frame_id`, `max_attempts`, `tier`, and failure count

## Retry & Escalation

Frame generation uses a **3-tier escalation** strategy when gates fail:

| Tier | Attempts | Strategy | Temperature |
|------|----------|----------|-------------|
| Soft | 1–3 | Re-prompt with gate feedback appended | 1.0 |
| Guided | 4–6 | Restructured prompt with explicit constraints | 0.7 |
| Constrained | 7–10 | Minimal creative freedom, line-by-line instructions | 0.3 |

**Reference strips** use simpler retry logic: max 3 attempts, no tier escalation, raise `ProviderError` on failure.

## Workflow Factory Pattern

`create_workflow()` in `workflow.py` is the factory function that:
1. Creates separate `AzureChatProvider` instances for grid generation and gate verification (using `config.generation.grid_model` and `config.generation.gate_model`)
2. Creates `GPTImageProvider` with `config.generation.reference_model`
3. Shares a single `DefaultAzureCredential` instance across all providers
4. Returns fully wired `SpriteForgeWorkflow` instance

**Cleanup:** Always call `workflow.close()` to clean up all provider connections. Safe to call multiple times.

## Configuration Validation

- `validate_config()` in `config.py` performs dry-run validation without Azure/network calls
- Returns `list[str]` of warnings, raises `ValueError`/`ValidationError` for errors
- Use `check_base_image=False` to skip base_image_path validation
- Relative `base_image_path` is resolved relative to config file location

## Palette Constraints

- Max palette size: **23 colors** (1 outline + 22 from `SYMBOL_POOL`, excluding implicit `.` transparent)
- Symbols are single characters from `SYMBOL_POOL` in `preprocessor.py`
- Outline symbol is always `O`, transparent is always `.`
- Duplicate row validation exists **only** in `SpritesheetSpec.model_validator` (not in `config.py`)

## Testing Conventions

- **Integration tests** use `@pytest.mark.integration` marker
- Integration tests auto-skip when `AZURE_AI_PROJECT_ENDPOINT` unset or credentials unavailable
- **Real Azure calls** in integration tests — no mocking of Azure APIs
- Use shared fixtures: `azure_project_endpoint` and `azure_credential` from `conftest.py`
- Keep integration tests minimal (1 row, 2 frames) to control costs

## Lazy Imports

- SpriteForge uses lazy imports via `__getattr__` (PEP 562) for Azure-dependent providers
- `AzureChatProvider` and `GPTImageProvider` are lazily imported in both `providers/__init__.py` and root `__init__.py`
- Allows non-Azure features to work without Azure SDK installed
- Azure provider methods wrap imports in try-except with helpful error messages: "Install with: `pip install spriteforge[azure]`"

## Response Format Enforcement

- `GridGenerator` and `LLMGateChecker` pass `response_format="json_object"` to all `chat()` calls
- Enforces JSON output at the token level via OpenAI structured output API
- `AzureChatProvider.chat()` forwards `response_format` as `{"type": response_format}` when provided
