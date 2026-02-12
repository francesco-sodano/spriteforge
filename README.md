# SpriteForge

An AI-powered spritesheet generator for 2D games. Feed it a base character image and a YAML config file defining animations (frame count, size, timing), and it uses a **two-stage AI pipeline** — GPT-Image-1.5 for rough reference generation and Claude Opus 4.6 for pixel-precise grid generation — both hosted on **Azure AI Foundry**, to produce a game-ready spritesheet PNG with transparent backgrounds.

## Features

- **AI-Powered Generation** — Uses a two-stage AI pipeline: GPT-Image-1.5 generates rough animation references, then Claude Opus 4.6 (with vision) translates them into pixel-precise 64×64 grids. Both models are accessed via Azure AI Foundry.
- **YAML-Driven Configuration** — Define animations, frame counts, sizes, and timing in a simple YAML config file.
- **Automatic Spritesheet Assembly** — Generates individual animation rows and stitches them into a single, game-ready spritesheet PNG.
- **Transparent Backgrounds** — All output sprites use PNG-32 with full alpha transparency, ready for any game engine.
- **Palette Swap Support** — Config files support P1/P2 color palettes for easy character recoloring via pixel-level color replacement.
- **Extensible Animation System** — Supports standard 2D game animations: idle, walk, run, attack combos, jump, magic, hit, knockdown, death, mounted combat, and throw.

## Requirements

- Python 3.12
- An Azure AI Foundry project endpoint (hosts both GPT-Image-1.5 and Claude Opus 4.6)
- Azure authentication via `DefaultAzureCredential` (no API keys needed)
- A base character reference image (PNG)
- A YAML configuration file defining the spritesheet layout and animations

## Installation

```bash
git clone https://github.com/francesco-sodano/spriteforge.git
cd spriteforge
uv sync --group dev
```

## Configuration

### Environment Variables

| Variable | Description |
|---|---|
| `AZURE_AI_PROJECT_ENDPOINT` | The Azure AI Foundry project endpoint (hosts both GPT-Image-1.5 and Claude Opus 4.6) |

Authentication uses `DefaultAzureCredential` — no API keys needed.

### YAML Config File

Define your spritesheet layout in a YAML file. Example:

```yaml
character:
  name: "Theron Ashblade"
  class: "Warrior"
  frame_size: [64, 64]
  spritesheet_columns: 14

animations:
  - name: idle
    row: 0
    frames: 6
    loop: true
    timing_ms: 150

  - name: walk
    row: 1
    frames: 8
    loop: true
    timing_ms: 100

  - name: attack1
    row: 2
    frames: 5
    loop: false
    timing_ms: 80
    hit_frame: 2
```

## Usage

```bash
# Generate a spritesheet from a config and base image
python -m spriteforge --config config.yaml --base-image character_ref.png --output spritesheet.png
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) as the package manager and a [Dev Container](https://containers.dev/) for a consistent development environment.

### Quick Start

1. Open the repo in VS Code and select **"Reopen in Container"**.
2. The dev container installs Python 3.12, creates a `.venv`, and syncs all dependencies automatically.

### Project Structure

```
src/spriteforge/        # Package source code
├── __init__.py         # Package exports
├── __main__.py         # CLI entry point
├── config.py           # YAML config loading and validation (Pydantic models)
├── generator.py        # Claude Opus 4.6 grid generation (Stage 2)
├── assembler.py        # Sprite row assembly into final spritesheet
├── models.py           # Data models for animations, characters, and spritesheets
├── palette.py          # Palette symbol → RGBA mapping
├── renderer.py         # Grid → PNG rendering
├── gates.py            # Verification gates (programmatic + LLM)
├── retry.py            # Retry & escalation engine
├── workflow.py         # Pipeline orchestrator (async Python)
└── providers/          # Stage 1 reference image provider (GPT-Image-1.5)
tests/                  # Tests (pytest)
scripts/                # Helper scripts
docs_assets/            # Spritesheet generation instructions for characters
```

### Run Tests

```bash
pytest
```

### Format & Lint

```bash
black .
mypy src/
```

### Run All Checks

```bash
black . && mypy src/ && pytest
```

## Docker

```bash
docker build -t spriteforge .
docker run --rm \
  -e AZURE_AI_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com" \
  -v $(pwd)/assets:/app/assets \
  spriteforge --config config.yaml --base-image base.png --output output.png
```

## Character Reference Docs

Detailed spritesheet generation instructions (prompts, palettes, and animation breakdowns) for each character are available in `docs_assets/`:

- [Theron Ashblade (Warrior)](docs_assets/spritesheet_instructions_theron.md)
- [Sylara Windarrow (Ranger)](docs_assets/spritesheet_instructions_sylara.md)
- [Drunn Ironhelm (Berserker)](docs_assets/spritesheet_instructions_drunn.md)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.