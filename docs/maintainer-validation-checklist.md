# Maintainer Rollout and Validation Checklist

This document describes the validation procedure that must be executed before merging or releasing changes to the minimal-input onboarding flow (`spriteforge init`).

It covers formatting/lint/type/test gates, targeted CLI smoke checks, end-to-end smoke with a generated config, regression checks for existing full-YAML users, known limitations, and troubleshooting notes for common init-flow failures.

---

## 1) Pre-merge gate checks

Run all of these from the repository root. Every check must pass (exit 0) before merging.

```bash
# Formatting
black --check .

# Type checking
mypy src/

# Unit + integration-skipped test suite
pytest

# Run only the init-flow focused regressions
pytest tests/test_cli.py -k "init"
pytest tests/test_config_builder.py
```

Expected outcome: `pytest` reports 0 failures and 0 errors (skips are allowed for integration tests).

---

## 2) Init-flow CLI smoke checks (non-interactive)

These commands exercise the full `init → validate → estimate` path without requiring Azure credentials or a real base image.

> **Note:** Use a real PNG file for `--base-image-path`. The placeholder `docs_assets/theron_base_reference.png` is suitable for local smoke testing.

### 2.1 Minimal non-interactive init

```bash
spriteforge init /tmp/smoke_test_minimal.yaml \
  --character-name "smoke character" \
  --base-image-path docs_assets/theron_base_reference.png \
  --action "idle|breathing in place|4|120" \
  --action "walk|steady forward walk|6|100" \
  --action "attack|quick forward slash|4|90" \
  --non-interactive
```

**Expected:** exits 0, prints `Config written to /tmp/smoke_test_minimal.yaml`.

### 2.2 Validate the generated config

```bash
spriteforge validate /tmp/smoke_test_minimal.yaml --no-check-base-image
```

**Expected:** exits 0, prints a validation summary with 3 animations, no errors.

### 2.3 Validate with base image check (requires real PNG to exist at path in config)

```bash
spriteforge validate /tmp/smoke_test_minimal.yaml
```

**Expected:** exits 0 (the base image was provided in step 2.1 and exists on disk).

### 2.4 Estimate

```bash
spriteforge estimate /tmp/smoke_test_minimal.yaml
```

**Expected:** exits 0, prints estimated LLM call counts for 3 rows.

### 2.5 Force overwrite

```bash
spriteforge init /tmp/smoke_test_minimal.yaml \
  --character-name "overwrite test" \
  --base-image-path docs_assets/theron_base_reference.png \
  --action "idle|breathing in place|4|120" \
  --non-interactive \
  --force
```

**Expected:** exits 0, overwrites the existing file without prompting.

### 2.6 Missing required field (should fail)

```bash
spriteforge init /tmp/smoke_missing_field.yaml \
  --base-image-path docs_assets/theron_base_reference.png \
  --action "idle|breathing in place|4|120" \
  --non-interactive
```

**Expected:** exits non-zero, error message references missing `--character-name`.

### 2.7 Malformed action input (should fail)

```bash
spriteforge init /tmp/smoke_bad_action.yaml \
  --character-name "bad action" \
  --base-image-path docs_assets/theron_base_reference.png \
  --action "idle|breathing in place|not_a_number|120" \
  --non-interactive
```

**Expected:** exits non-zero, error message references invalid frame count.

---

## 3) End-to-end smoke with the committed minimal example

Run against `configs/examples/minimal_generated.yaml` (the committed reference config):

```bash
spriteforge validate configs/examples/minimal_generated.yaml

spriteforge estimate configs/examples/minimal_generated.yaml
```

**Expected:** both commands exit 0.

If Azure credentials are available, optionally run a full generation:

```bash
spriteforge generate configs/examples/minimal_generated.yaml \
  --output /tmp/smoke_minimal_spritesheet.png
```

---

## 4) Regression checks for full-YAML users

Confirm existing configs continue to work without changes:

```bash
# Load and validate the reference full-YAML configs
spriteforge validate configs/examples/simple_enemy.yaml --no-check-base-image
spriteforge validate configs/examples/hero.yaml --no-check-base-image

# Estimate for both
spriteforge estimate configs/examples/simple_enemy.yaml
spriteforge estimate configs/examples/hero.yaml

# Config-loading unit tests
pytest tests/test_configs.py -v
```

**Expected:** all commands exit 0, all test cases pass.

---

## 5) Known limitations (v1)

- **No per-frame descriptions.** `frame_descriptions` is always set to `[]` by the minimal builder. Manual YAML editing is required to add per-frame prompts after init.
- **No hit frame inference.** `hit_frame` is always `null` from init output. Set it manually after generation if needed.
- **No palette definition.** The init flow always uses `auto_palette: true` with default settings. To use a fixed palette, edit the generated YAML and add a `palette:` section.
- **character.description is empty by default.** Use `--draft-description` to populate it via LLM (best-effort, adds latency), or fill it in by hand.
- **Model deployment names are hardcoded defaults.** Generated configs embed model names from the defaults table in [`docs/minimal-input-contract.md`](./minimal-input-contract.md). Override them by editing the `generation:` section of the generated YAML.
- **`--draft-description` is best-effort.** If the vision/chat endpoint is unavailable, init still succeeds and writes deterministic fallback text to `character.description`.

---

## 6) Troubleshooting: common init-flow failures

See also: [Character Config Guide — Troubleshooting](./character-config-guide.md#troubleshooting-init-flow).

### 6.1 `--character-name` missing in non-interactive mode

**Symptom:** `Error: Missing option '--character-name'` with non-zero exit.

**Fix:** Always supply `--character-name "..."` when running with `--non-interactive`. Omitting it is intentionally an error.

### 6.2 `--action` pipe format wrong

**Symptom:** `Error: Invalid action format` or similar parsing error.

**Fix:** The format is `"name|movement_description|frames|timing_ms"` — exactly four pipe-separated fields. `frames` and `timing_ms` must be positive integers. Example: `"idle|breathing in place|4|120"`.

### 6.3 Output file already exists

**Symptom:** Init exits non-zero with a message about the config file already existing.

**Fix:** Pass `--force` to overwrite, or choose a different output path.

### 6.4 `--draft-description` fails or adds unexpected delay

**Symptom:** Init takes unexpectedly long or prints a warning about description drafting.

**Cause:** `--draft-description` makes a live vision/chat model call. If endpoint credentials are missing, the endpoint is unreachable, or the call fails, init falls back to deterministic placeholder text and **still succeeds**.

**Fix:** Check `AZURE_AI_PROJECT_ENDPOINT` / `AZURE_OPENAI_ENDPOINT` environment variables. If drafting is not needed, omit `--draft-description`.

### 6.5 `validate` fails with "base image not found"

**Symptom:** `spriteforge validate` exits non-zero with a "base image not found" message.

**Cause:** The `base_image_path` written into the config at init time refers to a path that does not exist on the current system (e.g., running validate on a different machine or after moving files).

**Fix:** Either pass `--no-check-base-image` to skip the file-existence check, or ensure the file exists at the path written in the config.

### 6.6 `validate` fails with "Missing required section"

**Symptom:** `validate` exits non-zero reporting a missing section.

**Cause:** The YAML file may be empty, truncated, or not a valid SpriteForge config (e.g., a hand-edited file with an accidental deletion).

**Fix:** Re-run `spriteforge init` to regenerate the config, or compare against `configs/examples/minimal_generated.yaml`.

### 6.7 `estimate` shows 0 LLM calls

**Symptom:** `estimate` output shows 0 calls for some or all rows.

**Cause:** This can happen if `frames: 0` was written into an animation (validation would also catch this).

**Fix:** Re-run `spriteforge init` with corrected `--action` entries ensuring `frames` is at least 1.

---

## 7) Attaching results to a PR / release note

When attaching checklist results to a pull request:

1. Copy the checklist items from sections 1–4 above.
2. Mark each item ✅ (passed) or ❌ (failed with notes).
3. Record the `pytest` summary line (e.g. `607 passed, 1 skipped`).
4. Record the `spriteforge --version` output.
5. Note any known limitations from section 5 that apply to this release.

Minimal changelog snippet for PRs that update the init flow:

```md
- Docs: refreshed maintainer rollout checklist (docs/maintainer-validation-checklist.md).
- Docs: added init-flow troubleshooting notes to character-config-guide.md.
```
