# SpriteForge

SpriteForge generates game-ready 2D pixel-art spritesheets from a base character reference PNG plus a self-contained YAML config, using a two-stage AI pipeline on Azure AI Foundry.

## Features

- **Two-stage generation pipeline** (reference strip + pixel-grid generation)
- **YAML-driven character definitions** (palette, animations, generation rules)
- **Verification gates + retry escalation** for quality control
- **Deterministic grid-to-PNG rendering** for stable output
- **Automatic spritesheet assembly** with transparent PNG output
- **CLI commands for generate, validate, and estimate**
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

## Quick Start

Start from `configs/template.yaml` (or an example in `configs/examples/`), then run:

```bash
spriteforge validate configs/examples/simple_enemy.yaml
spriteforge estimate configs/examples/simple_enemy.yaml
spriteforge generate configs/examples/simple_enemy.yaml
```

For full YAML authoring details, see the [Character Config Guide](docs/character-config-guide.md).

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

Integration tests are opt-in and make real Azure calls:

```bash
SPRITEFORGE_RUN_INTEGRATION=1 pytest -m integration
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
