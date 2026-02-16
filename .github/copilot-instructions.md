# Copilot Instructions (SpriteForge)

## What this repo is

SpriteForge generates game-ready spritesheets from:
- a **self-contained YAML config** (source of truth)
- a **base character reference PNG**

Pipeline:
- Stage 1: GPT-Image-1.5 generates rough row reference strips
- Stage 2: GPT-5.2 outputs a **pixel grid** (JSON) that we render deterministically to PNG
- Gates: GPT-5-mini (temp 0.0) + programmatic checks, with retries/escalation

## Non-negotiable invariants

- Grid format: `{"grid": [64 strings of 64 chars]}` (64×64)
- Palette:
  - `.` is transparent (implicit)
  - `O` is outline
  - all other symbols are unique single characters defined in YAML
- Row 0 / Frame 0 is the **anchor frame** for character identity
- YAML is the single source of truth for palette + animations + model deployment names

## Tech + repo conventions

- Python 3.12, `uv` (use `pyproject.toml` as source of truth)
- Pydantic v2 models in `src/spriteforge/models.py`
- Formatting/typecheck/tests: `black`, `mypy`, `pytest`
- Layout: package code in `src/spriteforge/`, tests in `tests/`

## Azure + testing rules

- Auth: `DefaultAzureCredential` for **all** providers (chat, gates, image generation) — no API keys
- Env vars: `AZURE_AI_PROJECT_ENDPOINT` (chat/vision), `AZURE_OPENAI_GPT_IMAGE_ENDPOINT` (image generation base URL)
- GPT-Image-1.5 uses Entra ID bearer tokens via `get_bearer_token_provider` (scope: `https://cognitiveservices.azure.com/.default`)
- Integration tests:
  - mark with `@pytest.mark.integration`
  - **real Azure calls** (no mocking of Azure APIs)
  - opt-in via `SPRITEFORGE_RUN_INTEGRATION=1` (and auto-skip when endpoint/creds unavailable; see `tests/conftest.py`)

## When you change code

- Read the relevant modules + tests first; keep changes minimal and consistent.
- If adding/changing a third-party dependency or API usage, verify current docs via `fetch_webpage`.
- Update docs when behavior changes.
- Run: `black . && mypy src/ && pytest`.
