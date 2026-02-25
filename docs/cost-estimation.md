# Cost Estimation Guide

Use `spriteforge estimate <config.yaml>` to preview LLM call volume before running generation.

> `estimate` reports **call counts**, not currency. To convert to money, multiply calls (and optionally token usage) by your Azure model pricing.

## What `spriteforge estimate` Outputs

The command prints:

1. Character summary (rows/animations, total frames)
2. Budget settings (if configured)
3. Three scenarios: **Minimum**, **Expected**, **Maximum**
4. Per-scenario breakdown by call type
5. Optional budget check (`generation.budget.max_llm_calls`)

Example command:

```bash
spriteforge estimate configs/examples/simple_enemy.yaml
```

## What Drives Cost

The biggest drivers are:

- **Rows (animations)**: each row has Stage 1 calls plus Gate 3A checks.
- **Frames per row**: each frame needs Stage 2 generation and gate checks.
- **Retry rates**: failed checks cause additional Stage 1/2 and gate calls.

A useful mental model:

- More rows/frames = more baseline calls.
- More retries = multiplicative overhead.
- Complex characters/animations usually increase retry probability.

## Scenario Assumptions

`estimate_calls()` computes three scenarios.

### Minimum

Assumes every check passes on first attempt:

- 1 Stage 1 reference generation per row
- 1 Gate -1 per row
- 1 Stage 2 grid generation per frame
- Gate 0 for every frame
- Gate 1 and Gate 2 for non-first-frame continuity/identity checks
- 1 Gate 3A per row

### Expected

Uses retry-rate assumptions from `generation.budget` when configured,
otherwise defaults from `BudgetConfig`:

- **30%** of rows retry Stage 1 once (reference + Gate -1)
- **20%** of frames retry once
- **5%** of rows fail Gate 3A once and trigger row-level regeneration

These map to:

- `generation.budget.expected_reference_retry_rate`
- `generation.budget.expected_frame_retry_rate`
- `generation.budget.expected_row_retry_rate`

They affect estimation only (planning), not runtime gate/retry execution behavior.

### Maximum

Worst-case approximation with retry limits:

- Stage 1 reference generation retried up to 3 times per row
- Each frame retried up to per-row retry budget (`budget.max_retries_per_row`) or
  default frame retry cap (**10 attempts**, `RetryConfig.max_retries`)
- Gate volumes scale with those retries

Use this to understand upper-bound risk, not typical spend.

## Reading the Breakdown

Breakdown keys map to call categories:

- `reference_generation`: Stage 1 row strip creation
- `gate_minus_1`: Stage 1 strip quality check
- `grid_generation`: Stage 2 frame generation calls
- `gate_0`, `gate_1`, `gate_2` (or `gates_per_frame` in aggregated scenarios)
- `gate_3a`: row-level coherence checks

If budget is configured, the CLI also tells you whether the **expected** scenario fits under your budget cap.

## Azure Pricing Context

Because SpriteForge can use different deployments for reference, grid, and gate models, pricing varies per model and region.

Use official Azure pricing pages/calculator for current rates:

- Azure OpenAI pricing: <https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/>

A practical workflow:

1. Run `spriteforge estimate`.
2. Map call types to deployed models in your YAML.
3. Multiply projected usage by model-specific pricing.
4. Keep margin for variance between expected and maximum scenarios.

## Example Walkthrough: Simple Enemy vs. Full Hero

Sample output from repository configs:

| Config | Rows | Frames | Minimum | Expected | Maximum |
|---|---:|---:|---:|---:|---:|
| `configs/examples/simple_enemy.yaml` (Grukk) | 5 | 22 | 93 | 110 | 1600 |
| `configs/theron.yaml` (Theron Ashblade) | 16 | 85 | 356 | 425 | 3208 |

Interpretation:

- The hero config has ~3.4x more frames and rows, so baseline calls are much higher.
- Maximum scenarios rise sharply because retries multiply per-frame and per-row gate work.
- Budget caps should be set against at least the expected scenario, with operational headroom.

## Tips to Reduce Cost

- Start with fewer animation rows, then scale up.
- Reduce frame counts on non-critical animations.
- Keep prompts and design constraints clear to reduce retries.
- Use `spriteforge validate` before generation to catch config issues early.
- Set `generation.budget.max_llm_calls` to prevent runaway runs.
