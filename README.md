# SpriteForge

An AI-powered spritesheet generator for 2D pixel-art games. Feed it a **base character reference image** and a **self-contained YAML config file** defining the character (palette, animations, generation rules), and it uses a **two-stage AI pipeline** (hosted on **Azure AI Foundry**) to produce a game-ready spritesheet PNG with transparent backgrounds.

**One run = one character spritesheet.** Any character — heroes, enemies, bosses, NPCs — can be generated from the same input format.

## Features

- **Two-Stage Pipeline** — Stage 1 uses **GPT-Image-1.5** to generate rough animation reference strips; Stage 2 uses **GPT-5.2** (vision) to produce **pixel-precise 64×64 grids**.
- **Any Character, One YAML** — Each character is fully defined by a single self-contained YAML config file. Heroes get 16 animations, enemies might need 5, bosses 20 — the pipeline handles any layout.
- **YAML-Driven Configuration** — Define character description, color palette, animations, frame counts, timing, and generation rules in one YAML file. No code changes needed for new characters.
- **Verification Gates + Retry** — Uses a deterministic gate model (**GPT-5-mini**, temp 0.0) plus programmatic checks; failed frames retry with 3-tier escalation.
- **Deterministic Rendering** — The LLM outputs a structured grid (JSON); pure Python renders the grid to PNG using exact palette RGB.
- **Automatic Spritesheet Assembly** — Generates animation rows and stitches them into a final spritesheet PNG.
- **Transparent Backgrounds** — All output sprites use PNG-32 with full alpha transparency, ready for any game engine.
- **Palette Swap Support** — Config files support P1/P2 color palettes for easy character recoloring via pixel-level color replacement.
- **Flexible Animation System** — Define any number of animation rows per character with any frame count. Standard 2D game animations (idle, walk, attack, death, etc.) and custom animations are all supported.
- **Auto-Palette Extraction (Optional)** — Extract a palette from the base reference image via median-cut quantization, with optional **semantic color labels** via **GPT-5-nano**.

## Preprocessor (Auto-Palette)

SpriteForge includes an image preprocessor that can automatically extract a color palette from the base reference image, eliminating the need to manually define palette colors.

### How It Works

1. **Validate** — Checks the reference image for minimum size (32×32) and compatible aspect ratio.
2. **Resize** — Scales the image to the target frame dimensions (default 64×64) using nearest-neighbor interpolation to preserve pixel-art sharpness.
3. **Quantize** — Reduces colors to the target palette size using PIL's `MEDIANCUT` algorithm while preserving the alpha channel.
4. **Extract Palette** — Builds a `PaletteConfig` from the quantized colors:
   - The **darkest color** (by luminance) is automatically assigned as the outline (`O`).
   - Remaining colors are assigned symbols from the priority pool: `s`, `h`, `e`, `a`, `v`, `b`, `c`, `d`, ...
   - You can override outline selection by passing a specific `outline_color`.

### Enabling Auto-Palette

Set `auto_palette: true` in the `generation` section of your YAML config:

```yaml
generation:
  auto_palette: true
  max_palette_colors: 12  # optional, default: 16
```

When `auto_palette` is `true`, the `palette` section in the YAML becomes optional (it can still serve as a fallback). The quantized reference image is also used as a pixel-level visual guide for anchor frame generation.

## Requirements

- Python 3.12
- An Azure AI Foundry project endpoint (hosts your deployed models)
- Azure authentication via `DefaultAzureCredential` (no API keys needed)

**Per character:**
- A base character reference image (PNG)
- A self-contained YAML configuration file (character description, palette, animations, generation settings)

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
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint for chat/vision model deployments |
| `AZURE_OPENAI_GPT_IMAGE_ENDPOINT` | Azure OpenAI resource base URL for GPT-Image-1.5 (e.g. `https://myresource.openai.azure.com/`) |

Authentication uses `DefaultAzureCredential` everywhere — **no API keys needed**. Both the chat/vision providers (via Azure AI Foundry) and the image generation provider (via Azure OpenAI) use Entra ID bearer token authentication.

### YAML Config File

Each character is defined by a self-contained YAML config file. The config includes character metadata, color palette, animation definitions, and generation settings. Copy `configs/template.yaml` to get started, or see `docs/character-config-guide.md` for a full walkthrough.

**Simple enemy example (5 animations, 5 colors):**

```yaml
character:
  name: "goblin_scout"
  class: "Enemy"
  description: |
    Small, hunched goblin with bright green skin. Wears tattered brown
    leather armor. Carries a crude short sword. Beady red eyes, pointed
    ears, sharp teeth. ~36px tall in 64x64 frame.
  frame_width: 64
  frame_height: 64
  spritesheet_columns: 14

palette:
  outline:
    symbol: "O"
    name: "Outline"
    rgb: [20, 15, 10]
  colors:
    - symbol: "s"
      name: "Skin"
      rgb: [80, 140, 60]
    - symbol: "e"
      name: "Eyes"
      rgb: [200, 30, 30]
    - symbol: "a"
      name: "Armor"
      rgb: [110, 75, 40]
    - symbol: "w"
      name: "Weapon"
      rgb: [160, 160, 170]
    - symbol: "t"
      name: "Teeth"
      rgb: [230, 220, 190]

animations:
  - name: idle
    row: 0
    frames: 4
    loop: true
    timing_ms: 150
    prompt_context: |
      Hunched standing pose. Sword held loosely at side.

  - name: walk
    row: 1
    frames: 6
    loop: true
    timing_ms: 100
    prompt_context: |
      Skulking walk, hunched forward. Sword at ready.

  - name: attack
    row: 2
    frames: 4
    loop: false
    timing_ms: 80
    hit_frame: 2
    prompt_context: |
      Quick overhead slash with short sword.

  - name: hit
    row: 3
    frames: 3
    loop: false
    timing_ms: 100
    prompt_context: |
      Recoil from being struck. Head snaps back.

  - name: death
    row: 4
    frames: 5
    loop: false
    timing_ms: 120
    prompt_context: |
      Falls backward, sword drops. Final frame prone on ground.

generation:
  style: "Modern HD pixel art (Dead Cells / Owlboy style)"
  facing: "right"
  feet_row: 56
  rules: |
    64x64 pixel frames. Transparent PNG-32 background.
    No anti-aliasing to background. 1px dark outline.
    Character centered horizontally. Feet at y=56.

base_image_path: "assets/goblin_scout_reference.png"
output_path: "output/goblin_scout_spritesheet.png"
```

**Full hero example (16 animations, 10+ colors):** See `configs/` for the Theron, Sylara, and Drunn configs.

## Usage

There is no CLI entry point yet (see `tests/test_app.py`, pending issue #10). For now, use the programmatic workflow.

```python
import asyncio
from pathlib import Path

from spriteforge import create_workflow, load_config


async def main() -> None:
  config = load_config("configs/theron.yaml")
  workflow = await create_workflow(config=config)

  try:
    output_path = Path("output") / f"{config.character.name}_spritesheet.png"
    result_path = await workflow.run(
      base_reference_path=config.base_image_path,
      output_path=output_path,
    )
    print(f"Saved: {result_path}")
  finally:
    await workflow.close()


asyncio.run(main())
```

Also see `scripts/example_factory.py` for a minimal “create workflow + close” example.

## Development

This project uses [uv](https://docs.astral.sh/uv/) as the package manager and a [Dev Container](https://containers.dev/) for a consistent development environment.

### Quick Start

1. Open the repo in VS Code and select **"Reopen in Container"**.
2. The dev container installs Python 3.12, creates a `.venv`, and syncs all dependencies automatically.

### Project Structure

```
src/spriteforge/        # Package source code
├── __init__.py         # Package exports
├── config.py           # YAML config loading and validation (Pydantic models)
├── generator.py        # GPT-5.2 grid generation (Stage 2)
├── assembler.py        # Sprite row assembly into final spritesheet
├── models.py           # Data models for animations, characters, and spritesheets
├── palette.py          # Palette symbol → RGBA mapping (generic, config-driven)
├── preprocessor.py     # Image preprocessing (resize, quantize, auto-palette)
├── renderer.py         # Grid → PNG rendering
├── gates.py            # Verification gates (programmatic + LLM)
├── retry.py            # Retry & escalation engine
├── workflow.py         # Pipeline orchestrator (async Python)
└── providers/          # Stage 1 reference image provider (GPT-Image-1.5)
configs/                # Character YAML configs (one per character)
├── template.yaml       # Annotated template for creating new characters
├── examples/           # Example configs for different character types
├── sylara.yaml         # Sylara Windarrow (Ranger) — example hero
├── theron.yaml         # Theron Ashblade (Warrior) — example hero
└── drunn.yaml          # Drunn Ironhelm (Berserker) — example hero
tests/                  # Tests (pytest)
scripts/                # Helper scripts
docs/                   # User documentation
└── character-config-guide.md  # How to create character configs
docs_assets/            # Character instruction docs & base reference images
```

### Run Tests

```bash
pytest
```

Integration tests are opt-in and make real Azure calls:

```bash
SPRITEFORGE_RUN_INTEGRATION=1 pytest -m integration
```

You can override the Azure model deployment names used by integration tests via env vars:

| Variable | Purpose |
|---|---|
| `SPRITEFORGE_TEST_GRID_MODEL` | Grid generation model deployment (defaults to `generation.grid_model`) |
| `SPRITEFORGE_TEST_GATE_MODEL` | Gate verification model deployment (defaults to `generation.gate_model`) |
| `SPRITEFORGE_TEST_REFERENCE_MODEL` | Reference image model deployment (defaults to `generation.reference_model`) |
| `SPRITEFORGE_TEST_LABELING_MODEL` | Auto-palette labeling model deployment (defaults to `generation.labeling_model`) |

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
```

Note: the Docker image currently builds the library environment, but running it as a CLI is still pending (issue #10). The current `Dockerfile` uses `python -m spriteforge` as a placeholder.

## Creating New Characters

SpriteForge works with **any character** — you just need:

1. **A YAML config file** — Defines everything: character description, color palette, animations, and generation settings. Copy `configs/template.yaml` and fill in the values.
2. **A base reference image** (PNG) — A character reference sheet showing the design.

See [Character Config Guide](docs/character-config-guide.md) for a full walkthrough, and `configs/examples/` for complete examples ranging from simple enemies to complex heroes.

### Quick Start for a New Character

```bash
# 1. Copy the template
cp configs/template.yaml configs/my_enemy.yaml

# 2. Edit with your character's details (description, palette, animations)
# 3. Create or generate a base reference image (PNG)
```

Run SpriteForge using the programmatic workflow (CLI is pending issue #10):

```python
import asyncio
from pathlib import Path

from spriteforge import create_workflow, load_config


async def main() -> None:
  config = load_config("configs/my_enemy.yaml")
  workflow = await create_workflow(config=config)

  try:
    output_path = Path("output") / f"{config.character.name}_spritesheet.png"
    result_path = await workflow.run(
      base_reference_path=config.base_image_path,
      output_path=output_path,
    )
    print(f"Saved: {result_path}")
  finally:
    await workflow.close()


asyncio.run(main())
```

## Character Reference Docs (Original Heroes)

Detailed spritesheet generation instructions for the original three playable characters are in `docs_assets/`:

- [Theron Ashblade (Warrior)](docs_assets/spritesheet_instructions_theron.md)
- [Sylara Windarrow (Ranger)](docs_assets/spritesheet_instructions_sylara.md)
- [Drunn Ironhelm (Berserker)](docs_assets/spritesheet_instructions_drunn.md)

These serve as reference examples. Any new character follows the same YAML config format — no code changes needed.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.