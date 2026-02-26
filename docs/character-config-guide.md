# Character Configuration Guide

This guide explains how to create a YAML configuration file for a new SpriteForge character. Every character — hero, enemy, boss, or NPC — is defined by a single YAML config file and a base reference image.

## Quick Start

1. Copy the template:
   ```bash
   cp configs/template.yaml configs/my_character.yaml
   ```
2. Fill in the character metadata, palette, and animations.
3. Run SpriteForge (programmatic workflow):
   ```python
   import asyncio
   from pathlib import Path

   from spriteforge import create_workflow, load_config


   async def main() -> None:
     spec = load_config("configs/my_character.yaml")
     async with await create_workflow(config=spec) as workflow:
       output_path = Path("output") / f"{spec.character.name}_spritesheet.png"
       result = await workflow.run(
         base_reference_path=spec.base_image_path,
         output_path=output_path,
       )
       print(f"Saved: {result}")


   asyncio.run(main())
   ```

  Note: if you pass a custom `credential=...` to `create_workflow()`, you keep ownership of that credential; otherwise SpriteForge creates and closes `DefaultAzureCredential` for you.

CLI commands are available for `validate`, `estimate`, and `generate` via `spriteforge`.

For complete runnable examples, see:
- `configs/examples/simple_enemy.yaml` — A goblin with 5 animations and 5 colors.
- `configs/examples/hero.yaml` — A knight with 16 animations and 11 colors.

For builder/CLI onboarding with reduced authoring inputs, see
[`docs/minimal-input-contract.md`](./minimal-input-contract.md).

---

## Config File Structure

A character config has five top-level sections:

| Section | Required | Description |
|---------|----------|-------------|
| `character` | **Yes** | Name, class, description, frame dimensions |
| `animations` | **Yes** | List of animation rows (name, frames, timing, prompt) |
| `palette` | No* | Color symbols and RGB values |
| `generation` | No | Art style, facing direction, rules |
| Paths | No | `base_image_path` and `output_path` |

\* Required unless `generation.auto_palette` is `true`.

---

## Character Metadata

```yaml
character:
  name: "my_character"          # REQUIRED — used in filenames
  class: "Enemy"                # OPTIONAL — display / prompt context
  description: |                # REQUIRED for quality — detailed visual description
    ...
  frame_width: 64               # OPTIONAL (default: 64)
  frame_height: 64              # OPTIONAL (default: 64)
  spritesheet_columns: 14       # OPTIONAL (default: 14)
```

### Writing a Good Description

The `description` field is the **most important field for generation quality**. The AI reads it to understand what the character looks like. Aim for at least 100 words covering:

- **Body** — build, height, proportions, approximate pixel height in the frame
- **Face** — skin color, eye color, hair style and color, facial features
- **Clothing** — armor type, fabric, colors, accessories
- **Weapons** — type, size, which hand, distinctive features
- **Distinctive traits** — wings, horns, tail, glowing effects, scars

**Good example:**
> Tall, broad-shouldered human knight, approximately 48-52 pixels tall. Fair skin with short silver-white hair. Wears blue-steel plate armor with silver trim. Deep blue surcoat with lightning bolt emblem. Wields a longsword in the right hand. Left gauntlet crackles with storm energy. Short navy half-cape from the right shoulder.

**Too vague:**
> A knight with a sword.

### Frame Size

The default `64×64` pixels works for most characters. You can also use `frame_size: [64, 64]` as a shorthand. Only change this if your character needs a different canvas size.

Both formats are valid and equivalent:

```yaml
# Explicit width and height:
frame_width: 64
frame_height: 64

# OR shorthand:
frame_size: [64, 64]
```

The config loader maps `frame_size` to `frame_width` and `frame_height` internally.

---

## Palette Design

The palette maps single-character symbols to RGB colors. Every pixel in the generated sprite will be one of these colors.

```yaml
palette:
  outline:
    symbol: "O"
    name: "Outline"
    rgb: [20, 15, 10]
  colors:
    - symbol: "s"
      name: "Skin"
      rgb: [210, 170, 130]
    - symbol: "h"
      name: "Hair"
      rgb: [60, 40, 25]
```

### Rules

| Rule | Details |
|------|---------|
| `.` is always transparent | Do **not** list it in the palette — it is implicit |
| `O` is the outline symbol | Define its color in the `outline` section |
| Symbols are single characters | Letters, digits, or punctuation — must be unique |
| RGB format | Always `[R, G, B]` as a list of 3 integers (0–255) |
| Use mnemonic symbols | `s` = skin, `h` = hair, `a` = armor, etc. |

### How Many Colors?

| Character type | Recommended colors |
|----------------|-------------------|
| Simple enemy | 3–5 colors |
| Standard character | 6–8 colors |
| Detailed hero | 8–12 colors |

Keep the palette small — pixel art is about constraint. More than 12 colors rarely improves quality.

### Auto-Palette

If you prefer to have the palette extracted automatically from the base reference image, set:

```yaml
generation:
  auto_palette: true
  max_palette_colors: 12  # optional, valid range: 2-23 (default: 16)
```

When `auto_palette` is `true`, the `palette` section is optional (though it can still serve as a fallback).

#### How Quantization Works

The preprocessor uses PIL's **MEDIANCUT** color quantization algorithm to reduce the reference image to the target number of colors:

1. The image is resized to the frame dimensions (default 64×64) using **nearest-neighbor** interpolation.
2. Colors are quantized to `max_palette_colors` (default 16) distinct opaque colors. Transparent pixels are preserved — only the RGB channels are quantized, and the original alpha channel is re-applied.
3. The **darkest color by luminance** (`0.299R + 0.587G + 0.114B`) is automatically selected as the outline color (`O` symbol).
4. Remaining colors are assigned symbols from a priority pool: `s`, `h`, `e`, `a`, `v`, `b`, `c`, `d`, `g`, `i`, `k`, `l`, `m`, `n`, `p`, `r`, `t`, `u`, `w`, `x`, `y`, `z`.
5. You can override the outline selection by providing a specific `outline_color` programmatically.

The quantized reference image is also passed to the anchor frame generation (Stage 2) as a pixel-level visual guide, helping the AI match colors more accurately.

---

## Animation Planning

Animations are defined as rows in the spritesheet. Each row has a name, frame count, timing, and a prompt context that tells the AI what to draw.

```yaml
animations:
  - name: idle
    row: 0
    frames: 4
    loop: true
    timing_ms: 150
    prompt_context: |
      Relaxed standing pose, subtle breathing motion.
```

### Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | **Yes** | Animation identifier (lowercase, e.g. `idle`, `walk`) |
| `row` | **Yes** | Row index in spritesheet (0-based, must be unique) |
| `frames` | **Yes** | Number of frames (1 to `spritesheet_columns`) |
| `loop` | No | Whether the animation loops (default: `false`) |
| `timing_ms` | **Yes** | Milliseconds per frame (higher = slower) |
| `hit_frame` | No | Frame index where an attack connects |
| `prompt_context` | No* | Visual description for AI generation |
| `frame_descriptions` | No | Per-frame pose descriptions (length must be <= `frames`) |

\* Strongly recommended for quality. Empty `prompt_context` degrades output.

### Row 0 Is Special

**Row 0, Frame 0** is always the **anchor frame** — it establishes the character's look for all other frames and rows. Make Row 0 your most basic animation (typically `idle`).

### Common Animation Sets

**Minimal enemy (5 rows):**
`idle`, `walk`, `attack`, `hit`, `death`

**Standard character (8 rows):**
`idle`, `walk`, `attack1`, `attack2`, `jump`, `hit`, `death`, `run`

**Full hero (16 rows):**
`idle`, `walk`, `attack1`, `attack2`, `attack3`, `jump`, `jump_attack`, `magic`, `hit`, `knockdown`, `getup`, `death`, `mount_idle`, `mount_attack`, `run`, `throw`

### Timing Guidelines

| Speed | timing_ms | Good for |
|-------|-----------|----------|
| Fast | 60–80 ms | Quick attacks, run cycles |
| Medium | 100–120 ms | Walk, magic, recovery |
| Slow | 130–160 ms | Idle, death, heavy characters |

Heavier characters should use higher `timing_ms` values for the same animation type.

---

## Prompt Context Writing

The `prompt_context` field for each animation is critical — it tells the AI exactly what to draw in each animation row.

### Tips

1. **Describe motion, not story.** Say "sword swings from right to left" not "the warrior attacks his foe."
2. **Reference the weapon.** "Overhead chop with short sword" is better than "attack animation."
3. **Mention body parts.** "Head snaps back, body shifts backward" gives the AI clear pose cues.
4. **Note special frames.** If frame 2 is the impact frame, describe what peak extension looks like.
5. **Keep it concise.** 2–4 sentences per animation is usually enough.

**Good:**
> Quick overhead chop with short sword. Rears back slightly, then lunges forward with a wild downward slash. Stumbles slightly on follow-through.

**Too vague:**
> Attacks the enemy.

---

## Generation Settings

```yaml
generation:
  style: "Modern HD pixel art (Dead Cells / Owlboy style)"
  facing: "right"
  feet_row: 56
  rules: |
    64x64 pixel frames. Transparent PNG-32 background.
    No anti-aliasing to background. 1px dark outline.
    Character centered horizontally.
```

| Field | Default | Description |
|-------|---------|-------------|
| `style` | `"Modern HD pixel art (Dead Cells / Owlboy style)"` | Art style prompt |
| `facing` | `"right"` | Direction character faces (`right` or `left`) |
| `feet_row` | `56` | Y-coordinate of feet in the frame |
| `outline_width` | `1` | Outline thickness in pixels |
| `rules` | `""` | Free-form generation rules |
| `auto_palette` | `false` | Extract palette from reference image |
| `max_palette_colors` | `16` | Max colors when using auto_palette |
| `semantic_labels` | `true` | Use LLM to generate semantic palette color names when `auto_palette` is `true` |
| `grid_model` | `"gpt-5.2"` | Azure AI Foundry deployment name for Stage 2 grid generation |
| `gate_model` | `"gpt-5-mini"` | Azure AI Foundry deployment name for verification gates |
| `labeling_model` | `"gpt-5-nano"` | Azure AI Foundry deployment name for semantic palette labeling |
| `reference_model` | `"gpt-image-1.5"` | Azure AI Foundry deployment name for Stage 1 reference generation |
| `gate_3a_max_retries` | `2` | Retries for row-level Gate 3A coherence failures |
| `fallback_regen_frames` | `2` | Trailing frames to regenerate when Gate 3A feedback is non-specific |
| `compact_grid_context` | `false` | RLE-compress anchor/previous grid context to reduce token usage |
| `max_image_bytes` | `4000000` | Max image payload bytes for multimodal model requests |
| `request_timeout_seconds` | `120.0` | Per-request timeout (seconds) for external model calls |
| `max_anchor_regenerations` | `0` | Max anchor-row regeneration attempts for cascade recovery |
| `anchor_regen_failure_ratio` | `1.0` | Failed-row ratio threshold to trigger anchor regeneration |
| `allow_absolute_output_path` | `false` | Allow writing output outside workspace-relative paths |
| `budget` | `null` | Optional LLM call budget controls (`max_llm_calls`, retries, warnings, token tracking) |

### Budget Enforcement Modes

When `generation.budget.max_llm_calls` is set:

- `enforcement_mode: strict` blocks calls at the limit (no over-limit provider call).
- `enforcement_mode: best_effort` allows continuation and logs warnings after the limit.

For estimate planning, you can optionally tune:

- `expected_reference_retry_rate`
- `expected_frame_retry_rate`
- `expected_row_retry_rate`

---

## Examples by Character Type

### Simple Enemy

A basic goblin with 5 animations and 5 colors. See `configs/examples/simple_enemy.yaml`.

Key characteristics:
- 5 animation rows: idle, walk, attack, hit, death
- 5 palette colors + outline
- Shorter descriptions (the AI needs less detail for simpler characters)
- Single attack animation with `hit_frame`

### Full Hero

A detailed knight with 16 animations and 11 colors. See `configs/examples/hero.yaml`.

Key characteristics:
- 16 animation rows covering all gameplay actions
- 11 palette colors + outline for detailed armor and effects
- Rich `description` field (~150 words)
- Detailed `prompt_context` for each animation
- Multiple attack animations with combo flow descriptions
- Mount animations (upper body only)
- Magic/special ability animation

### Boss Character

A boss typically falls between an enemy and a hero:
- 8–12 animation rows (add rage, summon, or phase-change animations)
- 8–10 palette colors
- Larger character in the frame (may need adjusted `feet_row`)
- Multiple attack patterns with varied `hit_frame` values

---

## Validation

Verify your config loads correctly before running generation:

```python
from spriteforge.config import load_config

spec = load_config("configs/my_character.yaml")
print(f"Character: {spec.character.name}")
print(f"Animations: {len(spec.animations)}")
if spec.palette is not None:
    print(f"Palette colors: {len(spec.palette.colors) + 1}")  # + outline
else:
    print("Palette: auto (no explicit palette defined)")
print(f"Sheet size: {spec.sheet_width}x{spec.sheet_height}")
```

You can also run the config-loading tests:

```bash
pytest tests/test_configs.py -v
```

### Common Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Duplicate palette symbol` | Two colors share the same symbol | Use unique single characters |
| `Duplicate row index` | Two animations have the same `row` | Assign unique row numbers |
| `hit_frame must be < frames` | `hit_frame` ≥ `frames` count | Use 0-based index within frame range |
| `frames exceeding spritesheet_columns` | More frames than columns allow | Reduce frames or increase `spritesheet_columns` |
| `symbol must be exactly one character` | Multi-character palette symbol | Use a single character |

---

## Troubleshooting

**Q: My character looks different across animation rows.**
A: Make sure the `description` is detailed and consistent. Row 0 Frame 0 is the anchor — all other frames reference it for consistency.

**Q: Colors in the output don't match what I specified.**
A: Verify RGB values are in `[R, G, B]` format (list of 3 integers, not hex strings). Check that each color symbol is unique.

**Q: The AI generates poses that don't match my animation.**
A: Write more specific `prompt_context`. Describe the physical motion (limb positions, weapon trajectory) rather than the narrative action.

**Q: How do I add a completely new animation type?**
A: Just add another entry to the `animations` list with a unique `name` and `row` index. There are no restrictions on animation names.

**Q: Can I use different frame sizes for different characters?**
A: Yes — set `frame_width` and `frame_height` (or `frame_size`) per character. The default 64×64 works for most cases.
