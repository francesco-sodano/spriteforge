# SpriteForge

An AI-powered spritesheet generator for 2D games. Feed it a base character image and a YAML config file defining animations (frame count, size, timing), and it uses an Azure-hosted image-generation model to produce each animation row, then assembles them into a single, game-ready spritesheet PNG with transparent backgrounds.

## Features

- **AI-Powered Generation** — Uses Azure-hosted image-generation models to create pixel-art animation frames from a base character reference image.
- **YAML-Driven Configuration** — Define animations, frame counts, sizes, and timing in a simple YAML config file.
- **Automatic Spritesheet Assembly** — Generates individual animation rows and stitches them into a single, game-ready spritesheet PNG.
- **Transparent Backgrounds** — All output sprites use PNG-32 with full alpha transparency, ready for any game engine.
- **Palette Swap Support** — Config files support P1/P2 color palettes for easy character recoloring via pixel-level color replacement.
- **Extensible Animation System** — Supports standard 2D game animations: idle, walk, run, attack combos, jump, magic, hit, knockdown, death, mounted combat, and throw.

## Requirements

- Python 3.12
- An Azure endpoint for image generation (e.g., Azure OpenAI with DALL-E or a custom model)
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
| `AZURE_IMAGE_ENDPOINT` | The Azure image-generation model endpoint URL |
| `AZURE_IMAGE_API_KEY` | API key for the Azure endpoint (optional if using `DefaultAzureCredential`) |

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
├── generator.py        # Azure image-generation client
├── assembler.py        # Sprite row assembly into final spritesheet
└── models.py           # Data models for animations, characters, and spritesheets
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
  -e AZURE_IMAGE_ENDPOINT="https://your-endpoint.openai.azure.com/" \
  -e AZURE_IMAGE_API_KEY="your-key" \
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