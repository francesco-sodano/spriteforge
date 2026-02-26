# Minimal Input Contract and YAML Expansion Rules (v1)

This design note defines the canonical **minimal authoring contract** and how it is expanded into a full `SpritesheetSpec`-compatible YAML.

It is intended for config-builder and CLI onboarding flows where users provide only essential animation intent.

## 1) Required command inputs

The minimal contract requires:

- `config_output_path`: filesystem path where the expanded YAML is written.
- `base_image_path`: path to the base reference PNG.
- `character_name`: stable character identifier.
- `actions`: ordered list of action objects, each requiring:
  - `name`
  - `movement_description`
  - `frames`
  - `timing_ms`

Example minimal payload:

```yaml
config_output_path: "configs/my_character.generated.yaml"
base_image_path: "docs_assets/my_character_base_reference.png"
character_name: "my_character"
actions:
  - name: "idle"
    movement_description: "Breathing in place with subtle torso sway."
    frames: 6
    timing_ms: 140
  - name: "attack_slash"
    movement_description: "Quick forward slash with right arm and torso twist."
    frames: 8
    timing_ms: 90
```

## 2) Deterministic expansion rules

Expansion is deterministic and order-preserving.

### 2.1 Action-to-animation expansion

For each `actions[i]`:

- `animations[i].name = actions[i].name`
- `animations[i].row = i` (0..N-1 in the provided order)
- `animations[i].frames = actions[i].frames`
- `animations[i].timing_ms = actions[i].timing_ms`
- `animations[i].loop` defaults by normalized action name:
  - `true` for `idle`, `walk`, `run`
  - `false` for all other names
- `animations[i].prompt_context` uses this template:
  - `"{action_name}: {movement_description}"`
  - where `action_name` is the exact input `name` and spacing is normalized to single spaces.
- `animations[i].frame_descriptions = []` (not inferred in v1)
- `animations[i].hit_frame = null` (not inferred in v1)

### 2.2 Top-level field expansion

| Minimal input | Generated YAML / `SpritesheetSpec` field |
|---|---|
| `character_name` | `character.name` |
| `base_image_path` | `base_image_path` |
| `actions[*]` | `animations[*]` using rules above |
| `config_output_path` | Output file location only (not a `SpritesheetSpec` field) |

### 2.3 Generated defaults (v1)

Defaults below are set explicitly in generated YAML to keep behavior stable and readable:

- `generation.auto_palette: true`
- `generation.style: "Modern HD pixel art (Dead Cells / Owlboy style)"`
- `generation.facing: "right"`
- `generation.feet_row: 56`
- `generation.outline_width: 1`
- `generation.rules: ""`
- `generation.max_palette_colors: 16`
- `generation.semantic_labels: true`
- `generation.grid_model: "gpt-5.2"`
- `generation.gate_model: "gpt-5-mini"`
- `generation.labeling_model: "gpt-5-nano"`
- `generation.reference_model: "gpt-image-1.5"`
- `generation.gate_3a_max_retries: 2`
- `generation.fallback_regen_frames: 2`
- `generation.compact_grid_context: false`
- `generation.max_image_bytes: 4000000`
- `generation.request_timeout_seconds: 120.0`
- `generation.max_anchor_regenerations: 0`
- `generation.anchor_regen_failure_ratio: 1.0`
- `generation.allow_absolute_output_path: false`

Character/frame defaults:

- `character.class: ""`
- `character.description: ""`
- `character.frame_width: 64`
- `character.frame_height: 64`
- `character.spritesheet_columns: 14`
- `output_path: ""` (CLI/runtime may override)

## 3) Explicit invariants (checkpoint safety)

These values must remain explicit and immutable after expansion for a run:

- Animation order and `row` indices (`row = list index`) must not be re-sorted.
- `name`, `frames`, and `timing_ms` are exact user inputs.
- `Row 0 / Frame 0` remains the anchor identity frame by construction.
- `generation` model deployment names remain explicit strings in YAML.
- `generation.auto_palette` remains explicitly `true` unless a future version introduces an explicit override input.

## 4) Excluded inferences (v1)

The expander must **not** infer or synthesize:

- Palette entries (`palette`, color symbols, RGB values) when `auto_palette=true`.
- `hit_frame`.
- Per-frame `frame_descriptions`.
- Character biography/details beyond `character_name`.
- Action ordering changes, row compaction, or action merging/splitting.
- Timing/frame-count edits from semantic interpretation of `movement_description`.
- Any model selection beyond the fixed defaults listed above.
