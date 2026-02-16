# SpriteForge Architectural Review

**Date**: 2026-02-16
**Reviewer**: Claude Sonnet 4.5
**Codebase Version**: 0.1.0

---

## Executive Summary

SpriteForge is an ambitious AI-powered spritesheet generator with a well-designed two-stage pipeline (GPT-Image-1.5 for reference strips → GPT-5.2 for pixel grids). The project demonstrates strong architectural fundamentals with clean separation of concerns, comprehensive type safety via Pydantic, and good test coverage (~14.7k LOC total, roughly 50% tests).

**Overall Assessment**: **7.5/10**

**Strengths**:
- Clear separation of concerns (config, models, generator, gates, workflow)
- Type-safe with Pydantic v2 models
- Good retry/escalation strategy
- Comprehensive YAML-driven configuration
- Well-documented invariants and constraints

**Critical Risks**:
- **Authentication inconsistency** (GPT-Image uses different auth pattern than chat providers)
- **Configuration complexity** (too many knobs for users)
- **Missing error recovery** (no partial save/resume capability)
- **Tight Azure coupling** (no provider abstraction for multi-cloud)
- **Prompt engineering scattered** across multiple modules
- **No CLI** (only programmatic API despite documentation suggesting CLI)

---

## 1. Architecture Overview

### 1.1 High-Level Flow

```
YAML Config + Base PNG
  ↓
[Preprocessor] (optional: resize, quantize, auto-palette)
  ↓
[Stage 1: Reference Generation]
  GPT-Image-1.5 → Row reference strips
  Gate -1 (QC check, max 3 retries)
  ↓
[Stage 2: Grid Generation]
  For each frame:
    GPT-5.2 → 64×64 pixel grid (JSON)
    Programmatic checks (dimensions, symbols)
    Gate 0 (reference fidelity)
    Gate 1 (anchor identity)
    Gate 2 (temporal smoothness)
    Retry with 3-tier escalation (max 10 attempts)
  ↓
[Row Assembly]
  Gate 3A (row coherence)
  ↓
[Final Assembly]
  Stitch rows → PNG spritesheet
```

### 1.2 Key Modules

| Module | LOC | Purpose | Complexity |
|--------|-----|---------|------------|
| `workflow.py` | 955 | Pipeline orchestration | **HIGH** |
| `gates.py` | 652 | Verification gates (LLM + programmatic) | **MEDIUM** |
| `preprocessor.py` | 616 | Image resize, quantize, auto-palette | **MEDIUM** |
| `generator.py` | 384 | Stage 2 grid generation | **MEDIUM** |
| `config.py` | 365 | YAML loading & validation | **MEDIUM** |
| `retry.py` | 306 | Retry escalation engine | **LOW** |
| `models.py` | 238 | Pydantic data models | **LOW** |

**Tests**: 2077 LOC for workflow, 1108 for providers → good coverage

---

## 2. Critical Architectural Issues

### 2.1 🔴 CRITICAL: Authentication Inconsistency

**Problem**: Two different auth patterns in the same codebase

**Evidence**:
```python
# Chat providers (AzureChatProvider) use DefaultAzureCredential
# File: src/spriteforge/providers/azure_chat.py
credential = DefaultAzureCredential()

# BUT GPTImageProvider historically used API key auth
# File: .github/copilot-instructions.md:33-35
# "Auth: DefaultAzureCredential for **all** providers — no API keys"
# "GPT-Image-1.5 uses Entra ID bearer tokens"
```

**Memory context shows migration in progress**:
> "GPTImageProvider uses AsyncAzureOpenAI with API key auth (AZURE_OPENAI_GPT_IMAGE_API_KEY and AZURE_OPENAI_GPT_IMAGE_ENDPOINT), not AIProjectClient + DefaultAzureCredential"

**Risk**:
- Configuration confusion for users
- Security audit failures (API keys vs. managed identity)
- Corporate policy violation (docs say "disableLocalAuth=true" enforced)
- Different credential lifecycle management

**Impact**: HIGH
**Likelihood**: CERTAIN (already happening based on memories)

**Recommendation**: Consolidate to single auth pattern (DefaultAzureCredential) across all providers.

---

### 2.2 🔴 CRITICAL: No Partial Save/Resume Capability

**Problem**: Pipeline failures lose all progress

**Scenario**:
- User generates 12 of 16 animation rows
- Row 13 fails exhaustively (RetryExhaustedError after 10 attempts)
- Entire run fails → **all 12 completed rows discarded**

**Evidence**:
```python
# File: src/spriteforge/workflow.py:643-648
if ctx.current_attempt >= ctx.max_attempts:
    raise RetryExhaustedError(...)  # No partial save
```

**Impact**:
- Wasted API costs (re-running completed rows)
- Poor UX for large characters (16+ animations)
- No debugging capability (can't inspect partial output)

**Risk**: HIGH (especially for production use with complex characters)

**Recommendation**:
1. Add checkpoint system (save completed rows to temp directory)
2. Support `--resume` flag to skip completed rows
3. Emit partial spritesheet on failure

---

### 2.3 🟡 MEDIUM: Configuration Complexity

**Problem**: Too many configuration knobs for average users

**Evidence** (GenerationConfig alone):
```python
class GenerationConfig(BaseModel):
    style: str
    facing: str
    feet_row: int
    outline_width: int
    rules: str
    auto_palette: bool
    max_palette_colors: int  # Limited to 23
    semantic_labels: bool
    grid_model: str          # "gpt-5.2"
    gate_model: str          # "gpt-5-mini"
    labeling_model: str      # "gpt-5-nano"
    reference_model: str     # "gpt-image-1.5"
```

**Issues**:
1. Model deployment names require Azure AI Foundry knowledge
2. `feet_row: 56` is pixel-precise → hard for non-technical users
3. `max_palette_colors: 23` limit is arcane (why 23? symbol pool size)
4. Four separate model deployment names (grid, gate, labeling, reference)

**User perspective**:
> "I just want to generate a goblin sprite. Why do I need to know about GPT-5.2 deployment names?"

**Recommendation**:
1. Create **presets**: `simple`, `standard`, `advanced`
2. Hide model deployment names behind preset system
3. Auto-calculate `feet_row` from `frame_height` (e.g., 87.5% of height)
4. Bundle `auto_palette + semantic_labels` into single boolean

---

### 2.4 🟡 MEDIUM: Prompt Engineering Scattered

**Problem**: Prompt construction logic spread across 5+ files

**Evidence**:
```
src/spriteforge/prompts/
  ├── generator.py        (grid generation prompts)
  ├── gates.py            (gate verification prompts)
  ├── retry.py            (retry guidance prompts)
  └── providers.py        (reference strip prompts)

src/spriteforge/generator.py
  └── _build_palette_map_text()  (inline prompt fragment)

src/spriteforge/workflow.py
  └── Direct string formatting for various stages
```

**Issues**:
1. **DRY violation**: Palette map text duplicated in multiple places
2. **Hard to iterate**: Prompt improvements require changes across files
3. **No versioning**: Can't A/B test prompt variants
4. **Difficult prompt optimization**: No central place to tune prompts

**Recommendation**:
1. Centralize all prompts in `src/spriteforge/prompts/templates.py`
2. Use Jinja2 or similar for template rendering
3. Add prompt versioning system for A/B testing
4. Extract common fragments (palette map, style description) as reusable components

---

### 2.5 🟡 MEDIUM: Azure Vendor Lock-in

**Problem**: No abstraction for cloud providers

**Evidence**:
```python
# File: src/spriteforge/workflow.py:872-889
from spriteforge.providers.azure_chat import AzureChatProvider
from spriteforge.providers.gpt_image import GPTImageProvider

endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
# Hardcoded Azure-specific environment variables
```

**No provider abstraction**:
```python
# No ChatProvider interface with multiple implementations:
#   - AzureChatProvider
#   - OpenAIChatProvider
#   - AnthropicChatProvider
#   - LocalLLMProvider
```

**Risk**:
- Can't migrate to AWS Bedrock or GCP Vertex AI
- Can't use OpenAI directly (cheaper for testing)
- Can't add local model support (Ollama, vLLM)

**Impact**: MEDIUM (limits adoption for non-Azure users)

**Recommendation**:
1. Define `ChatProvider` protocol/interface
2. Create factory pattern: `create_chat_provider(provider_type, config)`
3. Support provider configs in YAML:
   ```yaml
   providers:
     grid: {type: "azure", endpoint: "...", model: "gpt-5.2"}
     gate: {type: "openai", api_key_env: "OPENAI_KEY", model: "gpt-4o-mini"}
   ```

---

### 2.6 🟡 MEDIUM: Missing CLI Despite Documentation

**Problem**: README and docs reference CLI that doesn't exist

**Evidence**:
```markdown
# File: README.md:174
"There is no CLI entry point yet (see tests/test_app.py, pending issue #10)"

# But configs/template.yaml:7-9 shows:
# Usage:
#   python -m spriteforge --config configs/my_character.yaml \
#                          --base-image assets/my_character_ref.png
```

**User confusion**:
1. Template suggests CLI exists
2. README says it doesn't
3. Users must write async Python code

**Recommendation**:
1. Implement basic CLI in `src/spriteforge/__main__.py`
2. Support: `python -m spriteforge generate --config path/to/config.yaml`
3. Update template.yaml to match actual usage

---

## 3. Code Quality Issues (DRY Violations)

### 3.1 Palette Map Text Generation (Duplicated)

**Locations**:
1. `src/spriteforge/generator.py:97-115` - `_build_palette_map_text()`
2. Similar logic in `src/spriteforge/prompts/generator.py`
3. Inline palette descriptions in gate prompts

**Impact**: Changing palette format requires 3+ file edits

**Fix**: Extract to shared utility in `src/spriteforge/utils.py`

---

### 3.2 Grid Dimension Checks (Duplicated)

**Locations**:
1. `src/spriteforge/generator.py:78-87` - `parse_grid_response()`
2. `src/spriteforge/gates.py:65-100` - `ProgrammaticChecker.check_dimensions()`

**Both validate**: 64 rows, 64 cols per row

**Fix**: Single `validate_grid_dimensions()` function

---

### 3.3 Image-to-Data-URL Conversion

**Locations**:
1. `src/spriteforge/utils.py` - `image_to_data_url()`
2. Used in 5+ places across generator, gates, preprocessor

**Good**: Already centralized in utils ✅

---

### 3.4 Gate Verdict Parsing (JSON → Pydantic)

**Pattern repeated** in `gates.py` for each gate:
```python
try:
    data = json.loads(response_text)
    return GateVerdict(**data)
except (json.JSONDecodeError, ValidationError):
    # fallback logic
```

**Fix**: Extract `parse_gate_response(text) -> GateVerdict` helper

---

## 4. Usability Issues

### 4.1 Error Messages Lack Actionable Guidance

**Example**:
```python
# File: src/spriteforge/workflow.py:643-648
raise RetryExhaustedError(
    f"Failed to generate {frame_id} after {max_attempts} attempts"
)
```

**Missing**:
- Which gates failed most often?
- What was the most common failure reason?
- Suggested config changes to fix it

**Better**:
```python
raise RetryExhaustedError(
    f"Failed to generate {frame_id} after {max_attempts} attempts.\n"
    f"Most common failure: Gate 1 (identity mismatch) - 7 failures\n"
    f"Suggestion: Improve character.description or try semantic_labels=true"
)
```

---

### 4.2 No Progress Visualization

**Current**: Optional callback `progress_callback(stage, current, total)`

**Issues**:
1. Callback is primitive (stage name + counts)
2. No ETA calculation
3. No visual progress bar
4. Can't see which animation is being processed

**Better**:
```
Generating Theron Ashblade spritesheet...
[████████████████░░░░] 12/16 rows (75%) - ETA: 4m 23s
Current: walk (row 11, frame 4/6) - Retry tier: SOFT (attempt 2/10)
```

---

### 4.3 Character Description Quality Unknown Until Failure

**Problem**: Users write descriptions, run pipeline, fail 30 minutes later

**Better**: Pre-generation validation
```python
def validate_description(desc: str, character_class: str) -> list[str]:
    """Return warnings if description likely to cause failures."""
    warnings = []
    if len(desc.split()) < 50:
        warnings.append("Description too short (< 50 words)")
    if character_class == "Hero" and "armor" not in desc.lower():
        warnings.append("Hero class but no armor mentioned")
    # Use LLM to score description quality (fast gpt-5-nano call)
    return warnings
```

---

### 4.4 YAML Validation Errors Are Cryptic

**Example**:
```python
# User typo in YAML:
animations:
  - name: idle
    row: 0
    frames: "four"  # Should be int

# Error:
ValidationError: 1 validation error for AnimationDef
frames
  Input should be a valid integer [...]
```

**Better**:
```
Error in configs/my_character.yaml at line 12:
  animations[0].frames: Expected integer, got string "four"

  Hint: Change "four" to 4 (no quotes)
```

---

## 5. Workflow & Process Issues

### 5.1 No Iterative Improvement Loop

**Current workflow**:
1. User writes full config
2. Run full pipeline (30-60 minutes)
3. See final output
4. Tweak config
5. Re-run entire pipeline

**Better workflow** (fast iteration):
```bash
# Step 1: Preview mode (generate 1 row only)
spriteforge preview --config my_char.yaml --row 0

# Step 2: Validate anchor frame quality
# If good, proceed. If bad, adjust description.

# Step 3: Generate full spritesheet
spriteforge generate --config my_char.yaml
```

---

### 5.2 No Quality Metrics

**Current**: Binary pass/fail gates

**Better**: Aggregate quality scores
```python
class GenerationReport:
    total_frames: int
    retries_per_frame: dict[str, int]
    average_gate_confidence: float
    failed_gates_histogram: dict[str, int]
    quality_score: float  # 0.0 - 1.0

# Output at end:
"""
Generation complete: 48 frames
Quality score: 0.87 / 1.00
Average retries per frame: 1.4
Problematic animations: attack (row 7) - 4 retries avg
"""
```

---

### 5.3 Character Instruction Definition is Verbose

**Current**: `docs_assets/spritesheet_instructions_theron.md` is 300+ lines

**Issues**:
1. **Duplication** with YAML config (description repeated)
2. **Maintenance burden** (keep .md + .yaml in sync)
3. **Not machine-readable** (can't auto-generate prompts from .md)

**Better**: Single source of truth (YAML config)
```yaml
character:
  description: |
    # This becomes the source for both:
    # 1. Grid generation prompts
    # 2. Reference strip prompts
    # 3. Documentation (auto-generated)
```

**Auto-generate docs**:
```bash
spriteforge docs --config theron.yaml > docs/theron.md
```

---

## 6. Testing & Quality Assurance Gaps

### 6.1 Integration Test Complexity

**Issue**: Integration tests require full Azure setup
```python
# File: tests/conftest.py:27-60
SPRITEFORGE_RUN_INTEGRATION=1
AZURE_AI_PROJECT_ENDPOINT=...
AZURE_OPENAI_GPT_IMAGE_ENDPOINT=...
# Plus DefaultAzureCredential with valid token
```

**Problem**: High barrier for contributors

**Better**:
1. Add **mock mode** (record/replay Azure responses)
2. Provide **Docker Compose** with local LLM (Ollama + vLLM)
3. Split tests: `unit`, `integration_local`, `integration_azure`

---

### 6.2 No Performance Benchmarks

**Missing**:
- Time per frame (by tier)
- Cost per character (Azure API calls)
- Memory usage (peak during row assembly)

**Better**: Track and display
```
Performance Report:
  Total time: 34m 12s
  Time per frame: 42.5s avg (min: 18s, max: 3m 4s)
  API calls: 156 total (cost: ~$2.34 estimated)
  Peak memory: 2.4 GB (during row assembly)
```

---

### 6.3 No Smoke Tests for Common Configs

**Missing**: Pre-commit tests that validate example configs still work

**Add**:
```python
@pytest.mark.smoke
def test_all_example_configs_load():
    for config_path in Path("configs/examples").glob("*.yaml"):
        config = load_config(config_path)
        warnings = validate_config(config_path, check_base_image=False)
        assert len(warnings) == 0
```

---

## 7. Security & Compliance Issues

### 7.1 🔴 Secrets in Error Messages

**Risk**: API keys/tokens leak in logs

**Check needed**:
```python
# Are Azure credentials sanitized in error messages?
logger.error(f"Failed to authenticate: {credential}")  # ⚠️ Could leak
```

**Fix**: Add credential sanitization in logging config

---

### 7.2 🟡 No Input Validation on Base Image

**Risk**: Malicious PNG could exploit PIL vulnerabilities

**Current**:
```python
base_img = Image.open(base_reference_path)  # No validation
```

**Better**:
```python
def validate_base_image(path: Path) -> None:
    # Check file size (max 10 MB)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Base image too large (>10MB)")

    # Verify it's actually a PNG
    with Image.open(path) as img:
        if img.format != "PNG":
            raise ValueError(f"Expected PNG, got {img.format}")

        # Check dimensions
        if img.width > 2048 or img.height > 2048:
            raise ValueError("Base image too large (max 2048x2048)")
```

---

### 7.3 🟡 YAML Bomb Protection Missing

**Risk**: Malicious YAML could cause DoS

**Example**:
```yaml
# yaml_bomb.yaml
x: &x ["lol", "lol", "lol", ...]  # 1 million nested lists
y: *x
```

**Fix**: Already using `yaml.safe_load()` ✅ (good!)

---

## 8. Performance & Scalability Concerns

### 8.1 No Caching of Reference Strips

**Issue**: Re-running same config re-generates identical reference strips

**Optimization**:
```python
# Cache key: hash(base_image + animation.prompt_context + style)
cache_key = hashlib.sha256(
    base_img_bytes + animation.prompt_context.encode() + style.encode()
).hexdigest()

cached_strip = cache.get(f"refstrip_{cache_key}")
if cached_strip:
    return cached_strip
```

---

### 8.2 Row Assembly Memory Spike

**Issue**: All frames loaded into memory during assembly

**Evidence**:
```python
# File: src/spriteforge/assembler.py
# Loads all frames as PIL Images before stitching
```

**Problem**: 64×64 RGBA × 48 frames × 16 rows = ~3 MB (manageable)
But could spike with larger frames or more animations

**Recommendation**: Stream rows to disk incrementally

---

### 8.3 No Rate Limiting

**Issue**: Parallel row generation could hit Azure rate limits

**Current**:
```python
max_concurrent_rows: int = 0  # Unlimited!
```

**Better**:
```python
max_concurrent_rows: int = 4  # Default to 4
# Add exponential backoff on 429 Too Many Requests
```

---

## 9. Documentation Gaps

### 9.1 No Architecture Diagram

**Missing**: Visual diagram of pipeline stages

**Add**: Mermaid diagram in README.md

---

### 9.2 No Troubleshooting Guide

**Missing**: "Generation failed, now what?"

**Add**: `docs/troubleshooting.md` with:
- Common errors (RetryExhaustedError, GateError, ProviderError)
- Diagnostic commands
- How to adjust config to fix failures

---

### 9.3 No Cost Estimation Guide

**Missing**: "How much will this cost on Azure?"

**Add**: Cost calculator
```python
def estimate_cost(config: SpritesheetSpec) -> float:
    total_frames = config.total_frames
    avg_retries = 1.5

    # Stage 1: Reference strips
    reference_cost = len(config.animations) * 0.04  # $0.04 per image

    # Stage 2: Grid generation
    grid_cost = total_frames * avg_retries * 0.002  # $0.002 per grid

    # Gates
    gate_cost = total_frames * avg_retries * 3 * 0.0001  # 3 gates per frame

    return reference_cost + grid_cost + gate_cost

# Example:
# 16 animations × 6 frames avg = 96 frames
# Cost: ~$0.64 + $0.29 + $0.03 = ~$0.96 per character
```

---

## 10. Improvement Plan (Prioritized)

### Phase 1: Critical Fixes (1-2 weeks)

**Priority**: 🔴 HIGH

1. **Consolidate authentication** (Issue 2.1)
   - Migrate GPTImageProvider to DefaultAzureCredential
   - Remove API key environment variables
   - Update tests to use unified auth

2. **Add partial save/resume** (Issue 2.2)
   - Checkpoint completed rows to `.spriteforge/checkpoints/`
   - Support `--resume` flag
   - Emit partial spritesheet on failure

3. **Implement basic CLI** (Issue 2.6)
   - `src/spriteforge/__main__.py`
   - Commands: `generate`, `validate`, `preview`

4. **Add secrets sanitization** (Issue 7.1)
   - Audit all error messages
   - Add credential redaction filter

### Phase 2: Usability Improvements (2-3 weeks)

**Priority**: 🟡 MEDIUM

5. **Configuration presets** (Issue 2.3)
   ```yaml
   generation:
     preset: "standard"  # simple | standard | advanced
     # Other fields optional, override preset defaults
   ```

6. **Centralize prompts** (Issue 2.4)
   - Move all prompts to `prompts/templates/`
   - Use Jinja2 for rendering
   - Add prompt versioning

7. **Improve error messages** (Issue 4.1)
   - Add actionable guidance to all errors
   - Include common failure reasons in RetryExhaustedError

8. **Add progress visualization** (Issue 4.2)
   - Integrate `rich` library for progress bars
   - Show ETA, current row/frame, retry tier

9. **Pre-generation validation** (Issue 4.3)
   - Validate character description quality
   - Use LLM to score descriptions (fast check)

10. **Quality metrics report** (Issue 5.2)
    - Generate report at end of run
    - Include quality score, retry histogram, problematic animations

### Phase 3: Architectural Improvements (3-4 weeks)

**Priority**: 🟢 LOW (but important for long-term)

11. **Provider abstraction** (Issue 2.5)
    - Define `ChatProvider` protocol
    - Support multiple cloud providers (Azure, OpenAI, Anthropic)

12. **Iterative workflow** (Issue 5.1)
    - Add `preview` command (1 row only)
    - Add `generate-row` command (specific row)

13. **Auto-generate documentation** (Issue 5.3)
    - Single source of truth (YAML)
    - `spriteforge docs` command

14. **Reference strip caching** (Issue 8.1)
    - Cache based on content hash
    - Configurable cache directory

15. **Mock integration tests** (Issue 6.1)
    - Add record/replay mode for Azure responses
    - Support local LLM testing

### Phase 4: Polish & Documentation (1-2 weeks)

**Priority**: 🟢 LOW

16. **Architecture diagram** (Issue 9.1)
17. **Troubleshooting guide** (Issue 9.2)
18. **Cost estimation guide** (Issue 9.3)
19. **Performance benchmarks** (Issue 6.2)
20. **Smoke tests** (Issue 6.3)

---

## 11. Specific Code Simplifications (DRY)

### 11.1 Extract Common Validators

**Create**: `src/spriteforge/validators.py`
```python
def validate_grid_dimensions(
    grid: list[str],
    expected_rows: int = 64,
    expected_cols: int = 64
) -> None:
    """Raises ValueError if grid dimensions invalid."""
    if len(grid) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(grid)}")
    for i, row in enumerate(grid):
        if len(row) != expected_cols:
            raise ValueError(f"Row {i}: expected {expected_cols} chars, got {len(row)}")

def validate_palette_symbols(grid: list[str], palette: PaletteConfig) -> None:
    """Raises ValueError if grid contains undefined symbols."""
    allowed = {palette.transparent_symbol, palette.outline.symbol}
    allowed.update(c.symbol for c in palette.colors)

    for i, row in enumerate(grid):
        for j, char in enumerate(row):
            if char not in allowed:
                raise ValueError(f"Invalid symbol '{char}' at ({i}, {j})")
```

**Use in**:
- `generator.py:parse_grid_response()`
- `gates.py:ProgrammaticChecker.check_*()`

---

### 11.2 Consolidate Palette Map Rendering

**Create**: `src/spriteforge/prompts/utils.py`
```python
def render_palette_table(palette: PaletteConfig) -> str:
    """Generate human-readable palette symbol table."""
    lines = [f"- `{palette.transparent_symbol}` → Transparent (background)"]
    lines.append(f"- `{palette.outline.symbol}` → {palette.outline.element} {palette.outline.rgb}")
    for color in palette.colors:
        lines.append(f"- `{color.symbol}` → {color.element} {color.rgb}")
    return "\n".join(lines)
```

**Use in**:
- `generator.py:_build_palette_map_text()` (delete, use this)
- `prompts/generator.py` (import this)
- `prompts/gates.py` (import this)

---

### 11.3 Unify Image Data URL Conversion

**Already done** ✅ in `utils.py:image_to_data_url()`

Good example of DRY principle applied correctly.

---

### 11.4 Extract Gate Response Parser

**Create**: `src/spriteforge/gates.py` (add helper)
```python
def parse_gate_verdict(
    response_text: str,
    gate_name: str,
    fallback_passed: bool = False
) -> GateVerdict:
    """Parse LLM response into GateVerdict, with fallback."""
    try:
        # Strip markdown fences
        text = response_text.strip()
        fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
        match = fence_pattern.search(text)
        if match:
            text = match.group(1).strip()

        data = json.loads(text)
        return GateVerdict(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"{gate_name} response parse failed: {e}")
        return GateVerdict(
            gate_name=gate_name,
            passed=fallback_passed,
            confidence=0.0,
            feedback=f"Failed to parse {gate_name} response: {e}"
        )
```

**Use in**:
- `LLMGateChecker.check_reference_fidelity()` (Gate 0)
- `LLMGateChecker.check_identity_consistency()` (Gate 1)
- `LLMGateChecker.check_temporal_smoothness()` (Gate 2)
- `LLMGateChecker.check_row_coherence()` (Gate 3A)
- Reference strip QC (Gate -1)

---

## 12. Character Instruction Definition Improvements

### 12.1 Current State

**Files**:
- `docs_assets/spritesheet_instructions_theron.md` (300+ lines)
- `docs_assets/spritesheet_instructions_sylara.md`
- `docs_assets/spritesheet_instructions_drunn.md`

**Issues**:
1. **Duplication** with YAML configs
2. **Not machine-readable** (can't use in prompts directly)
3. **Maintenance burden** (keep .md + .yaml in sync)

### 12.2 Proposed Structure

**Single source of truth**: YAML config

```yaml
character:
  name: "theron_ashblade"
  class: "Warrior"

  # Enhanced description with structured sections
  description:
    summary: |
      Theron Ashblade is a battle-hardened warrior hero...

    physical:
      build: "Muscular, broad-shouldered, ~45px tall in 64×64 frame"
      skin: "Fair, weathered from battle"
      hair: "Short dark brown, slightly messy"
      face: "Strong jawline, determined eyes"
      height_px: 45

    equipment:
      armor: "Steel plate armor with gold trim"
      weapon: "Two-handed longsword (silver blade, gold hilt)"
      shield: null
      accessories: ["Red cape", "Leather belt with pouches"]

    distinctive_features:
      - "Scar across left eyebrow"
      - "Gold pauldrons on shoulders"
      - "Red cape flows behind in movement"

    visual_style: "Modern HD pixel art (Dead Cells style)"

  # Animation-specific notes
  animation_notes:
    idle: "Sword rests on right shoulder, cape billows slightly"
    walk: "Strong, confident stride, cape flows"
    attack: "Overhead slash, full body rotation"
    block: "Raises sword horizontally, defensive posture"
```

**Benefits**:
1. Machine-readable (can extract structured data for prompts)
2. Single source of truth (no .md files to maintain)
3. Auto-generate documentation: `spriteforge docs theron.yaml > theron.md`
4. Validation: Pydantic can enforce required fields

### 12.3 Auto-Generate Documentation

**Command**:
```bash
spriteforge docs --config configs/theron.yaml --output docs/theron.md
```

**Generated Markdown**:
```markdown
# Theron Ashblade (Warrior)

## Physical Appearance

- **Build**: Muscular, broad-shouldered, ~45px tall in 64×64 frame
- **Skin**: Fair, weathered from battle
- **Hair**: Short dark brown, slightly messy
...

## Animations

### idle (4 frames, 150ms)
Sword rests on right shoulder, cape billows slightly

[Auto-generated animation preview thumbnails]

### walk (6 frames, 100ms)
Strong, confident stride, cape flows
...
```

### 12.4 Simplify for Users

**Instead of** asking users to write detailed instructions in separate .md files:

**Provide wizard**:
```bash
spriteforge init --interactive

> Character name: Theron Ashblade
> Class: Warrior
> Describe build: Muscular, broad-shouldered
> Describe armor: Steel plate with gold trim
> Weapon type: Two-handed sword
> Weapon in which hand: Right
> Does character have a cape? (y/n): y
> Cape color: Red
...
[Generates configs/theron_ashblade.yaml with structured description]
```

---

## 13. Summary Recommendations

### Immediate Actions (This Sprint)

1. ✅ **Create this architectural review document** (you're reading it!)
2. 🔴 **Fix authentication inconsistency** (consolidate to DefaultAzureCredential)
3. 🔴 **Add partial save capability** (checkpoint system)
4. 🟡 **Implement basic CLI** (`generate`, `validate` commands)

### Short-Term (Next 2-4 Weeks)

5. 🟡 **Add configuration presets** (simple/standard/advanced)
6. 🟡 **Centralize prompt templates** (move to prompts/templates/)
7. 🟡 **Improve error messages** (actionable guidance)
8. 🟡 **Add progress visualization** (use `rich` library)

### Medium-Term (Next 1-2 Months)

9. 🟢 **Provider abstraction** (support OpenAI, Anthropic, local models)
10. 🟢 **Iterative workflow** (`preview`, `generate-row` commands)
11. 🟢 **Auto-generate docs from YAML** (single source of truth)
12. 🟢 **Reference strip caching** (avoid redundant API calls)

### Long-Term (Next Quarter)

13. 🟢 **Mock integration tests** (record/replay, local LLM support)
14. 🟢 **Performance benchmarks** (cost tracking, time profiling)
15. 🟢 **Quality metrics dashboard** (visualize gate pass rates, retry patterns)

---

## 14. Final Thoughts

SpriteForge is a **solid foundation** with **clear architectural vision**. The two-stage pipeline is well-designed, and the separation of concerns is excellent. The main risks are:

1. **Authentication inconsistency** (highest priority fix)
2. **Lack of partial save** (critical UX issue for production)
3. **Configuration complexity** (barrier to adoption)

With the recommended improvements, SpriteForge could become a **best-in-class AI spritesheet generator** suitable for indie game developers and studios.

**Key Insight**: The project has all the right pieces—it just needs **simplification** (presets, CLI), **resilience** (checkpoints, better errors), and **flexibility** (provider abstraction, caching).

**Estimated effort for Phase 1-2**: 4-6 weeks of focused development
**ROI**: Significantly improved user experience, production readiness, broader adoption

---

**Next Steps**:
1. Review this document with the team
2. Prioritize issues based on user feedback
3. Create GitHub issues for each recommendation
4. Start with Phase 1 (critical fixes)
5. Set up weekly architectural review sessions

---

**End of Review**
