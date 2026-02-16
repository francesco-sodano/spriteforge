# SpriteForge Improvement Plan

**Date**: 2026-02-16
**Status**: Proposed
**Owner**: Development Team

> **TL;DR**: This plan addresses critical architectural issues, improves usability, and reduces code duplication. Focus on **authentication unification**, **partial save capability**, and **configuration simplification** first.

---

## Quick Reference: Priority Matrix

| Priority | Issue | Impact | Effort | Timeline |
|----------|-------|--------|--------|----------|
| 🔴 P0 | Authentication inconsistency | HIGH | 1 week | Sprint 1 |
| 🔴 P0 | No partial save/resume | HIGH | 1 week | Sprint 1 |
| 🔴 P0 | Missing CLI | MEDIUM | 3 days | Sprint 1 |
| 🟡 P1 | Configuration complexity | MEDIUM | 1 week | Sprint 2 |
| 🟡 P1 | Prompt engineering scattered | MEDIUM | 1 week | Sprint 2 |
| 🟡 P1 | Error messages unclear | LOW | 3 days | Sprint 2 |
| 🟢 P2 | Provider abstraction | LOW | 2 weeks | Sprint 3-4 |
| 🟢 P2 | Iterative workflow | LOW | 1 week | Sprint 3 |

---

## Phase 1: Critical Fixes (Sprint 1 - Week 1-2)

### 1.1 Unify Authentication Pattern

**Problem**: GPTImageProvider uses different auth than chat providers

**Current State**:
```python
# Chat providers: DefaultAzureCredential ✅
# GPT Image: API key (AZURE_OPENAI_GPT_IMAGE_API_KEY) ❌
```

**Solution**:
```python
# File: src/spriteforge/providers/gpt_image.py
class GPTImageProvider:
    def __init__(self, credential: DefaultAzureCredential, ...):
        self._credential = credential
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )
        self._client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider,
            ...
        )
```

**Changes Required**:
- [ ] Update `GPTImageProvider.__init__()` to accept credential
- [ ] Remove `AZURE_OPENAI_GPT_IMAGE_API_KEY` from environment variables
- [ ] Update `create_workflow()` to pass credential to GPTImageProvider
- [ ] Update tests (`tests/test_providers.py`, `tests/conftest.py`)
- [ ] Update documentation (`.github/copilot-instructions.md`, `README.md`)

**Validation**:
- [ ] All integration tests pass with DefaultAzureCredential
- [ ] No API key references in code

**Effort**: 1 week (includes testing and documentation)

---

### 1.2 Add Partial Save & Resume Capability

**Problem**: Pipeline failure loses all progress

**Current State**:
```python
# workflow.run() completes all or nothing
# 12 hours of generation lost if row 13 fails
```

**Solution**: Checkpoint system

**Architecture**:
```
.spriteforge/
  checkpoints/
    {character_name}_{timestamp}/
      metadata.json          # Config, palette, progress
      row_0_strip.png        # Reference strips
      row_0_frame_*.png      # Individual frames
      row_0_assembled.png    # Assembled row
      ...
```

**Implementation**:
```python
# File: src/spriteforge/checkpoint.py (NEW)
class CheckpointManager:
    def save_row(self, row_index: int, frames: list[Image.Image]) -> None
    def load_row(self, row_index: int) -> list[Image.Image] | None
    def resume_from_checkpoint(self, checkpoint_dir: Path) -> dict

# File: src/spriteforge/workflow.py
async def run(self, ..., resume_from: Path | None = None):
    if resume_from:
        state = checkpoint_mgr.resume_from_checkpoint(resume_from)
        # Skip completed rows
```

**Changes Required**:
- [ ] Create `src/spriteforge/checkpoint.py`
- [ ] Add `CheckpointManager` class
- [ ] Update `SpriteForgeWorkflow.run()` to save after each row
- [ ] Add `--resume` flag to CLI
- [ ] Update `README.md` with resume instructions

**Validation**:
- [ ] Can resume from checkpoint after failure
- [ ] Skips completed rows (no redundant API calls)
- [ ] Partial spritesheet saved on failure

**Effort**: 1 week

---

### 1.3 Implement Basic CLI

**Problem**: No CLI despite documentation references

**Current State**:
```bash
# This doesn't work (but template.yaml suggests it):
python -m spriteforge --config configs/my_char.yaml
```

**Solution**:
```python
# File: src/spriteforge/__main__.py (NEW)
import click

@click.group()
def cli():
    """SpriteForge - AI-powered spritesheet generator"""

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--resume", type=click.Path(), help="Resume from checkpoint")
def generate(config: str, resume: str | None):
    """Generate a spritesheet from YAML config"""

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def validate(config: str):
    """Validate a config file without generating"""

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--row", type=int, default=0)
def preview(config: str, row: int):
    """Generate a single row preview"""
```

**Changes Required**:
- [ ] Create `src/spriteforge/__main__.py`
- [ ] Add `click` dependency to `pyproject.toml`
- [ ] Implement `generate`, `validate`, `preview` commands
- [ ] Update `configs/template.yaml` with correct usage
- [ ] Update `README.md` with CLI examples

**Validation**:
- [ ] `python -m spriteforge generate --config ...` works
- [ ] `python -m spriteforge validate --config ...` works
- [ ] Error messages are user-friendly

**Effort**: 3 days

---

## Phase 2: Usability Improvements (Sprint 2 - Week 3-4)

### 2.1 Configuration Presets

**Problem**: Too many knobs for average users

**Current State**:
```yaml
generation:
  style: "..."
  facing: "right"
  feet_row: 56
  outline_width: 1
  rules: "..."
  auto_palette: false
  max_palette_colors: 16
  semantic_labels: true
  grid_model: "gpt-5.2"
  gate_model: "gpt-5-mini"
  labeling_model: "gpt-5-nano"
  reference_model: "gpt-image-1.5"
```

**Solution**: Presets

```yaml
# Simple config (preset handles everything)
generation:
  preset: "standard"  # simple | standard | advanced

# Advanced config (override preset)
generation:
  preset: "standard"
  max_palette_colors: 20  # Override just this one
```

**Preset Definitions**:
```python
# File: src/spriteforge/presets.py (NEW)
PRESETS = {
    "simple": {
        "auto_palette": True,
        "semantic_labels": False,
        "max_palette_colors": 8,
        # Model names hidden from user
    },
    "standard": {
        "auto_palette": True,
        "semantic_labels": True,
        "max_palette_colors": 16,
    },
    "advanced": {
        # All options exposed
    }
}
```

**Changes Required**:
- [ ] Create `src/spriteforge/presets.py`
- [ ] Update `GenerationConfig` to support `preset` field
- [ ] Merge preset defaults with user overrides
- [ ] Update `configs/template.yaml` to use presets
- [ ] Update `docs/character-config-guide.md`

**Effort**: 1 week

---

### 2.2 Centralize Prompt Templates

**Problem**: Prompts scattered across 5+ files (DRY violation)

**Current State**:
```
src/spriteforge/prompts/
  ├── generator.py        (grid prompts)
  ├── gates.py            (gate prompts)
  ├── retry.py            (retry guidance)
  └── providers.py        (reference prompts)

src/spriteforge/generator.py
  └── _build_palette_map_text()  (inline fragment)
```

**Solution**: Centralized templates with Jinja2

```python
# File: src/spriteforge/prompts/templates.py (NEW)
from jinja2 import Environment, PackageLoader

env = Environment(loader=PackageLoader("spriteforge", "prompts/templates"))

def render_grid_prompt(**kwargs) -> str:
    return env.get_template("grid_generation.j2").render(**kwargs)

def render_gate_prompt(gate_name: str, **kwargs) -> str:
    return env.get_template(f"gate_{gate_name}.j2").render(**kwargs)
```

**Template Structure**:
```
src/spriteforge/prompts/templates/
  ├── _palette_table.j2       (reusable fragment)
  ├── _style_guide.j2         (reusable fragment)
  ├── grid_generation.j2
  ├── gate_0_reference.j2
  ├── gate_1_identity.j2
  ├── gate_2_temporal.j2
  ├── gate_3a_coherence.j2
  ├── retry_soft.j2
  ├── retry_guided.j2
  └── retry_constrained.j2
```

**Changes Required**:
- [ ] Add `jinja2` dependency
- [ ] Create `prompts/templates/` directory
- [ ] Convert all prompts to Jinja2 templates
- [ ] Update all callers to use `render_*_prompt()`
- [ ] Delete old inline prompt code

**Effort**: 1 week

---

### 2.3 Improve Error Messages

**Problem**: Errors lack actionable guidance

**Current State**:
```python
raise RetryExhaustedError(
    f"Failed to generate {frame_id} after {max_attempts} attempts"
)
```

**Solution**: Rich error context

```python
raise RetryExhaustedError(
    frame_id=frame_id,
    total_attempts=max_attempts,
    failure_summary={
        "gate_0": 2,  # 2 failures
        "gate_1": 7,  # 7 failures (identity mismatch)
        "gate_2": 1,
    },
    suggestions=[
        "Gate 1 failed most often (identity mismatch)",
        "Try improving character.description with more details",
        "Consider enabling semantic_labels=true for better color matching"
    ]
)
```

**Changes Required**:
- [ ] Update `RetryExhaustedError` to accept failure stats
- [ ] Update `RetryContext` to track per-gate failures
- [ ] Add suggestion generator based on failure patterns
- [ ] Update error handler to display suggestions
- [ ] Add to `docs/troubleshooting.md`

**Effort**: 3 days

---

### 2.4 Add Progress Visualization

**Problem**: No visual feedback during long runs

**Current State**:
```python
# Optional callback, but primitive
progress_callback(stage="row_generation", current=3, total=16)
```

**Solution**: Rich progress bars

```python
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("ETA: {task.fields[eta]}"),
) as progress:
    task = progress.add_task(
        f"Generating {character_name}",
        total=total_frames,
        eta="calculating..."
    )
```

**Display**:
```
Generating Theron Ashblade spritesheet...
⠋ [████████████████░░░░] 75% - ETA: 4m 23s
  Current: walk (row 11/16, frame 4/6)
  Retry tier: SOFT (attempt 2/10)
```

**Changes Required**:
- [ ] Add `rich` dependency
- [ ] Create `src/spriteforge/display.py`
- [ ] Update `workflow.run()` to use rich progress
- [ ] Add ETA calculation based on historical frame times
- [ ] Make it work in both CLI and programmatic mode

**Effort**: 3 days

---

## Phase 3: Architectural Improvements (Sprint 3-4 - Week 5-8)

### 3.1 Provider Abstraction

**Problem**: Tight coupling to Azure

**Current State**:
```python
from spriteforge.providers.azure_chat import AzureChatProvider
# No way to use OpenAI, Anthropic, or local models
```

**Solution**: Provider protocol

```python
# File: src/spriteforge/providers/chat.py
class ChatProvider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> str:
        ...

# File: src/spriteforge/providers/factory.py (NEW)
def create_chat_provider(config: dict) -> ChatProvider:
    provider_type = config["type"]
    if provider_type == "azure":
        return AzureChatProvider(...)
    elif provider_type == "openai":
        return OpenAIChatProvider(...)
    elif provider_type == "anthropic":
        return AnthropicChatProvider(...)
```

**YAML Config**:
```yaml
providers:
  grid:
    type: "azure"
    endpoint: "${AZURE_AI_PROJECT_ENDPOINT}"
    model: "gpt-5.2"

  gate:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
```

**Changes Required**:
- [ ] Define `ChatProvider` protocol
- [ ] Create provider factory
- [ ] Implement `OpenAIChatProvider`
- [ ] Implement `AnthropicChatProvider`
- [ ] Update `create_workflow()` to use factory
- [ ] Update config schema to support provider configs

**Effort**: 2 weeks

---

### 3.2 Iterative Workflow (Preview Mode)

**Problem**: Can't iterate quickly on config changes

**Solution**: Preview command

```bash
# Generate only row 0 (anchor frame)
spriteforge preview --config my_char.yaml --row 0

# Output:
# output/my_char_preview_row0.png
# Shows: idle animation (4 frames)
```

**Implementation**:
```python
# File: src/spriteforge/workflow.py
async def run_preview(self, row_index: int = 0) -> Path:
    """Generate a single row for preview."""
    animation = self.config.animations[row_index]
    # Run only this row (skip others)
    return preview_path
```

**Changes Required**:
- [ ] Add `SpriteForgeWorkflow.run_preview()` method
- [ ] Add `preview` CLI command
- [ ] Update `README.md` with preview workflow
- [ ] Add to `docs/character-config-guide.md`

**Effort**: 1 week

---

### 3.3 Reference Strip Caching

**Problem**: Re-running same config regenerates identical reference strips

**Solution**: Content-based caching

```python
# File: src/spriteforge/cache.py (NEW)
class ReferenceCache:
    def __init__(self, cache_dir: Path = Path.home() / ".spriteforge/cache"):
        self._cache_dir = cache_dir

    def get_cache_key(
        self,
        base_image: bytes,
        prompt: str,
        style: str,
        num_frames: int
    ) -> str:
        content = base_image + prompt.encode() + style.encode()
        return hashlib.sha256(content).hexdigest()

    def get(self, key: str) -> Image.Image | None:
        path = self._cache_dir / f"{key}.png"
        return Image.open(path) if path.exists() else None

    def set(self, key: str, image: Image.Image) -> None:
        path = self._cache_dir / f"{key}.png"
        image.save(path)
```

**Changes Required**:
- [ ] Create `src/spriteforge/cache.py`
- [ ] Add cache to `GPTImageProvider`
- [ ] Add `--no-cache` flag to CLI
- [ ] Add cache cleanup command (`spriteforge cache clear`)

**Effort**: 3 days

---

## Phase 4: Code Simplifications (DRY - Ongoing)

### 4.1 Extract Grid Validators

**Create**: `src/spriteforge/validators.py`

**Consolidate**:
- `generator.py:parse_grid_response()` dimension checks
- `gates.py:ProgrammaticChecker.check_dimensions()`
- `gates.py:ProgrammaticChecker.check_valid_symbols()`

**New API**:
```python
from spriteforge.validators import validate_grid_dimensions, validate_grid_symbols

# Usage:
validate_grid_dimensions(grid, expected_rows=64, expected_cols=64)
validate_grid_symbols(grid, palette)
```

**Effort**: 1 day

---

### 4.2 Extract Gate Response Parser

**Create**: `src/spriteforge/gates.py` helper

**Consolidate**:
- Gate 0, 1, 2, 3A all have identical JSON parsing logic

**New API**:
```python
def parse_gate_verdict(
    response_text: str,
    gate_name: str,
    fallback_passed: bool = False
) -> GateVerdict:
    # Unified parsing + fallback
```

**Effort**: 1 day

---

### 4.3 Extract Palette Rendering

**Create**: `src/spriteforge/prompts/utils.py`

**Consolidate**:
- `generator.py:_build_palette_map_text()`
- Similar code in `prompts/generator.py`

**New API**:
```python
from spriteforge.prompts.utils import render_palette_table

palette_text = render_palette_table(palette)
```

**Effort**: 1 day

---

## Character Instruction Definition Improvements

### 5.1 Structured YAML Descriptions

**Problem**: Freeform text descriptions are hard to parse

**Current State**:
```yaml
character:
  description: |
    Long freeform paragraph mixing all details...
```

**Solution**: Structured sections

```yaml
character:
  description:
    summary: "One-sentence overview"

    physical:
      build: "Muscular, 45px tall"
      skin: "Fair"
      hair: "Dark brown, short"
      face: "Strong jawline"

    equipment:
      armor: "Steel plate with gold trim"
      weapon: "Two-handed longsword"
      accessories: ["Red cape", "Belt pouches"]

    distinctive_features:
      - "Scar on left eyebrow"
      - "Gold pauldrons"

  animation_notes:
    idle: "Sword on shoulder, cape billows"
    walk: "Confident stride"
```

**Benefits**:
1. Machine-readable (can extract specific fields for prompts)
2. Validation (Pydantic enforces structure)
3. Auto-completion (IDE can suggest fields)

**Changes Required**:
- [ ] Update `CharacterConfig` model to support nested description
- [ ] Update config loader to parse structured description
- [ ] Add backward compatibility (still accept string description)
- [ ] Update `configs/template.yaml`
- [ ] Update `docs/character-config-guide.md`

**Effort**: 1 week

---

### 5.2 Auto-Generate Documentation

**Problem**: `.md` instruction files duplicate YAML content

**Solution**: Generate docs from YAML

```bash
spriteforge docs --config theron.yaml --output docs/theron.md
```

**Generated Output**:
```markdown
# Theron Ashblade (Warrior)

## Physical Appearance
- Build: Muscular, 45px tall
- Skin: Fair
...

## Animations
### idle (4 frames, 150ms/frame)
Sword on shoulder, cape billows

### walk (6 frames, 100ms/frame)
Confident stride
...

## Palette
| Symbol | Color Name | RGB |
|--------|------------|-----|
| O | Outline | [20, 40, 40] |
| s | Skin | [235, 210, 185] |
...
```

**Changes Required**:
- [ ] Create `src/spriteforge/commands/docs.py`
- [ ] Add Markdown template for character docs
- [ ] Add `docs` CLI command
- [ ] Delete static `.md` files (theron, sylara, drunn)
- [ ] Add to `.github/workflows/` (auto-generate on commit)

**Effort**: 3 days

---

### 5.3 Interactive Config Wizard

**Problem**: Users struggle to write good configs

**Solution**: Interactive wizard

```bash
spriteforge init --interactive

> Character name: Goblin Scout
> Class (Hero/Enemy/Boss/NPC): Enemy
> Character build: Small, hunched
> Approximate height in 64px frame: 36
> Skin color: Bright green
> Armor description: Tattered brown leather
> Weapon type: Short sword
> Weapon in which hand: Right
> Number of animations: 5
...
[Generates configs/goblin_scout.yaml]
```

**Implementation**:
```python
# File: src/spriteforge/commands/init.py
import questionary  # Interactive prompts library

def interactive_wizard() -> SpritesheetSpec:
    answers = questionary.form(
        name=questionary.text("Character name:"),
        char_class=questionary.select("Class:", choices=["Hero", "Enemy", "Boss", "NPC"]),
        ...
    ).ask()
    return build_spec_from_answers(answers)
```

**Changes Required**:
- [ ] Add `questionary` dependency
- [ ] Create `src/spriteforge/commands/init.py`
- [ ] Add `init` CLI command
- [ ] Create character templates for each class

**Effort**: 1 week

---

## Testing & Quality Improvements

### 6.1 Mock Integration Tests

**Problem**: Integration tests require full Azure setup

**Solution**: Record/replay mode

```python
# File: tests/mocks/recorder.py (NEW)
class ResponseRecorder:
    def record(self, request, response):
        # Save to tests/fixtures/responses/
        pass

    def replay(self, request):
        # Load from fixtures
        pass

# Usage in tests:
@pytest.mark.integration
@use_recorder("theron_row0")  # Uses fixture if exists, records if not
async def test_generate_row_0():
    ...
```

**Changes Required**:
- [ ] Create response recorder/replayer
- [ ] Add fixtures for common scenarios
- [ ] Update integration tests to use recorder
- [ ] Add `--record` flag to pytest

**Effort**: 1 week

---

### 6.2 Performance Benchmarks

**Problem**: No visibility into performance/cost

**Solution**: Benchmark suite

```python
# File: tests/benchmarks/test_performance.py (NEW)
@pytest.mark.benchmark
def test_frame_generation_time():
    times = []
    for i in range(10):
        start = time.time()
        frame = await generate_frame(...)
        times.append(time.time() - start)

    assert mean(times) < 60  # < 1 minute per frame
    print(f"Avg time per frame: {mean(times):.1f}s")
```

**Metrics to Track**:
- Time per frame (by retry tier)
- API calls per frame
- Cost per character (estimated)
- Memory usage (peak during assembly)

**Effort**: 3 days

---

### 6.3 Smoke Tests

**Problem**: Example configs could break unnoticed

**Solution**: Smoke test suite

```python
# File: tests/test_smoke.py (NEW)
@pytest.mark.smoke
@pytest.mark.parametrize("config_path", list(Path("configs/examples").glob("*.yaml")))
def test_example_config_loads(config_path):
    config = load_config(config_path)
    warnings = validate_config(config_path, check_base_image=False)
    assert len(warnings) == 0
```

**Run on**:
- Every commit (CI/CD)
- Pre-release checks

**Effort**: 1 day

---

## Security Improvements

### 7.1 Credential Sanitization

**Problem**: Credentials might leak in error messages

**Solution**: Logging filter

```python
# File: src/spriteforge/logging.py
class CredentialFilter(logging.Filter):
    def filter(self, record):
        # Redact any credential-like strings
        record.msg = redact_credentials(record.msg)
        return True

def redact_credentials(text: str) -> str:
    # Patterns to redact:
    # - Bearer tokens
    # - API keys
    # - SAS tokens
    return re.sub(r'Bearer [A-Za-z0-9\-_]+', 'Bearer [REDACTED]', text)
```

**Effort**: 1 day

---

### 7.2 Input Validation

**Problem**: No size limits on base images

**Solution**: Validation layer

```python
# File: src/spriteforge/validators.py
def validate_base_image(path: Path) -> None:
    # File size check (max 10 MB)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Base image too large (>10MB)")

    # Format check
    with Image.open(path) as img:
        if img.format != "PNG":
            raise ValueError(f"Expected PNG, got {img.format}")

        # Dimension check
        if img.width > 2048 or img.height > 2048:
            raise ValueError("Base image dimensions too large (max 2048×2048)")
```

**Effort**: 1 day

---

## Implementation Timeline

### Sprint 1 (Weeks 1-2): Critical Fixes
- Week 1: Authentication unification + partial save
- Week 2: CLI implementation + testing

### Sprint 2 (Weeks 3-4): Usability
- Week 3: Configuration presets + prompt centralization
- Week 4: Error messages + progress visualization

### Sprint 3-4 (Weeks 5-8): Architecture
- Week 5-6: Provider abstraction
- Week 7: Iterative workflow + caching
- Week 8: Character config improvements

### Ongoing: Code Quality
- DRY refactoring (1-2 days per sprint)
- Test improvements (1-2 days per sprint)
- Documentation updates (as needed)

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] All providers use DefaultAzureCredential
- [ ] Can resume from checkpoint after failure
- [ ] CLI works: `python -m spriteforge generate --config ...`

### Phase 2 Success Criteria
- [ ] Users can generate characters using `preset: "simple"`
- [ ] All prompts in `prompts/templates/`
- [ ] Error messages include actionable suggestions
- [ ] Progress bar shows ETA

### Phase 3 Success Criteria
- [ ] Can use OpenAI provider (not just Azure)
- [ ] Preview mode saves 95% of time during iteration
- [ ] Cache reduces redundant API calls by 50%+

### Code Quality Metrics
- [ ] DRY violations reduced by 70% (measured by duplicate code detection)
- [ ] Test coverage maintained above 80%
- [ ] Integration test setup time reduced from 30min to 5min (mock mode)

---

## Risk Mitigation

### Risk: Breaking Changes

**Mitigation**:
- Maintain backward compatibility for configs
- Add deprecation warnings (1 version ahead)
- Provide migration guide

### Risk: Performance Regression

**Mitigation**:
- Run benchmarks on every PR
- Set performance budgets (max time per frame)

### Risk: Scope Creep

**Mitigation**:
- Stick to phased plan
- Defer nice-to-haves to Phase 4
- Review priorities weekly

---

## Next Steps

1. **Review this plan** with the team (1 hour meeting)
2. **Create GitHub issues** for each Phase 1 item (1 hour)
3. **Set up project board** (Kanban with columns: Backlog, In Progress, Review, Done)
4. **Assign owners** for each Phase 1 task
5. **Start Sprint 1** on Monday

**First PR**: Authentication unification (estimated: 3-5 days)

---

**Questions? Concerns? Feedback?**

Open a discussion in GitHub or reach out to the architecture team.

---

**End of Plan**
