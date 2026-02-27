# SpriteForge

SpriteForge generates game-ready 2D pixel-art spritesheets from a base character reference PNG plus a self-contained YAML config, using a two-stage AI pipeline on Azure AI Foundry.

## Features

- **Two-stage generation pipeline** (reference strip + pixel-grid generation)
- **YAML-driven character definitions** (palette, animations, generation rules)
- **Verification gates + retry escalation** for quality control
- **Per-request timeout protection** for external model calls
- **Deterministic grid-to-PNG rendering** for stable output
- **Automatic spritesheet assembly** with transparent PNG output
- **CLI commands for init, generate, validate, and estimate**
- **Optional structured JSON logs** for log aggregation
- **Run summary JSON export** with retries, gate outcomes, and token usage
- **Optional auto-palette extraction** from base references

## Requirements

- Python 3.12
- Azure authentication via `DefaultAzureCredential`
- Environment variables:
  - `AZURE_AI_PROJECT_ENDPOINT` **or** `AZURE_OPENAI_ENDPOINT` (chat/gate endpoint)
  - `AZURE_OPENAI_GPT_IMAGE_ENDPOINT`

## Installation

```bash
git clone https://github.com/francesco-sodano/spriteforge.git
cd spriteforge
uv sync --group dev
```

Create a local environment file before running generation:

```bash
cp .env.example .env
```

## Quick Start (Minimal Init Flow)

Create a config from just a base image + actions, then run:

```bash
spriteforge init configs/examples/my_character_minimal.yaml \
  --character-name "my character" \
  --base-image-path docs_assets/theron_base_reference.png \
  --action "idle|breathing in place|4|120" \
  --action "walk|steady forward walk|6|100" \
  --action "attack|quick forward slash|4|90" \
  --non-interactive

# Optional: best-effort character.description drafting from the image
spriteforge init configs/examples/my_character_minimal.yaml \
  --character-name "my character" \
  --base-image-path docs_assets/theron_base_reference.png \
  --action "idle|breathing in place|4|120" \
  --non-interactive \
  --draft-description

# Observability-focused run
spriteforge validate configs/examples/my_character_minimal.yaml
spriteforge estimate configs/examples/my_character_minimal.yaml
spriteforge generate configs/examples/my_character_minimal.yaml \
  --output output/my_character_spritesheet.png

spriteforge generate configs/examples/simple_enemy.yaml \
  --json-logs \
  --run-summary output/simple_enemy_run_summary.json
```

Reference generated minimal example: `configs/examples/minimal_generated.yaml`.

Prefer full manual YAML authoring? See the [Character Config Guide](docs/character-config-guide.md)
for the advanced YAML-first path.

### Reliability Defaults

- `generation.request_timeout_seconds` defaults to `120.0` seconds and is applied to
  Stage 1 reference generation, Stage 2 grid generation, and LLM gate checks.
- Budget strict mode now blocks calls at the configured limit (`max_llm_calls`),
  so over-limit provider calls are prevented instead of allowing one extra call.

## Documentation

- [Architecture: How Generation Works](docs/architecture.md)
- [Cost Estimation Guide](docs/cost-estimation.md)
- [Character Config Guide](docs/character-config-guide.md)
- [Checkpoint / Resume Example](docs/checkpoint_resume_example.md)

## Development

Run checks:

```bash
black . && mypy src/ && pytest
```

Run targeted minimal-config init regressions:

```bash
pytest tests/test_cli.py -k "init and (test_init_non_interactive or test_init_interactive or test_init_generated_config)"
pytest tests/test_config_builder.py -k "deterministic"
```

Init-flow tests use `tmp_path` fixtures with a placeholder `base.png` file to avoid requiring real image assets.

Integration tests are opt-in and make real Azure calls:

```bash
SPRITEFORGE_RUN_INTEGRATION=1 pytest -m integration
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
