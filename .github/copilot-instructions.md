# Copilot Custom Instructions

Guidance for Copilot behavior for this repository.

## Persona
You are an expert, world-class software engineering assistant. Your goal is to help write clean, efficient, and correct code while following the project's standards and instructions.

## Project Overview
The project overview is the README.md file. It provides a high-level description of the project and its purpose. Always keep that in consideration and remember it.

**SpriteForge** is an AI-powered spritesheet generator for 2D pixel-art games. It takes a **self-contained YAML config file** and a **base character reference image**, then uses a **two-stage AI pipeline** to produce a game-ready spritesheet PNG with transparent backgrounds.

**One run = one character = one spritesheet.** SpriteForge works with **any character** — heroes, enemies, bosses, NPCs — as long as the character is defined in a self-contained YAML config. The YAML config is the single source of truth: it contains character metadata, palette definition, animation layout, and generation settings.

The original target game is **Blades of the Fallen Realm** — a Golden Axe-style beat 'em up. The three original characters (Theron Ashblade, Sylara Windarrow, Drunn Ironhelm) serve as reference examples in `configs/`.

## Architecture

### Two-Stage Pipeline

The core innovation is a **two-stage generation approach** that mimics how human pixel artists work: sketch first, then pixel.

```
Stage 1 (Image Model) → Gate -1 → Stage 2 (GPT Vision → Grid) → Gates → Retry → Render → Assemble
```

- **Stage 1 — Reference Generation:** GPT-Image-1.5 (via Azure AI Foundry) produces a rough, non-pixel-precise animation strip for each row. This is a visual reference, NOT the final output.
- **Stage 2 — Grid Generation:** GPT-5.2 (via Azure AI Foundry) receives the rough reference + anchor frame + palette spec, and outputs a **pixel-precise 64×64 JSON grid** (64 strings of 64 single-character palette symbols). This grid IS the final output — rendered to PNG by pure Python code.

### Key Design Principle

> "Don't ask an AI to generate an image — ask it to generate a **data structure** that you render into an image."

The LLM never outputs pixels directly. It outputs a structured grid of palette symbols. A deterministic renderer (`renderer.py`) converts this grid to PNG using exact RGB values. This eliminates color drift, anti-aliasing artifacts, and resolution inconsistencies.

### Pipeline Flow (Per Character)

```
1. Load config (YAML → SpritesheetSpec) — palette, animations, generation settings all from YAML
2. For Row 0 (first animation, typically IDLE) — anchor row:
   a. Stage 1: Generate rough reference strip
   b. Gate -1: Validate reference quality (LLM)
   c. Stage 2: Generate anchor frame (Frame 0 of first animation)
   d. Programmatic checks + Gate 0 + Gate 1 (LLM)
   e. Retry loop (up to 10 attempts with 3-tier escalation)
   f. Generate remaining frames (with anchor + prev frame context)
   g. Gate 3A: Validate assembled row coherence (LLM)
3. For remaining animation rows (count determined by YAML config):
   a. Stage 1: Generate rough reference strip for this animation
   b. Gate -1: Validate reference quality
   c. For each frame:
      - Stage 2: Generate frame (with anchor + reference + prev frame)
      - Programmatic checks → Gate 0 → Gate 1 (→ Gate 2 if not first frame)
      - Retry loop if any gate fails
   d. Gate 3A: Validate assembled row coherence
4. Assemble all rows into final spritesheet (dimensions determined by config)
5. Save PNG output
```

### Row Independence

Rows are **independent** of each other. Only the **first animation's Frame 0** (typically IDLE) serves as the cross-row anchor. There is no dependency graph between rows — they are processed sequentially but each row only needs the anchor frame as shared context.

### Verification Gates

A multi-gate verification system ensures quality:

| Gate | Type | What It Checks |
|------|------|----------------|
| Gate -1 | LLM (vision) | Reference strip quality — correct pose, frame count, character identity |
| Programmatic | Code | Grid dimensions (64×64), valid palette symbols, non-empty, transparent background % |
| Gate 0 | LLM (vision) | Single-frame quality — anatomy, proportions, pose matches animation |
| Gate 1 | LLM (vision) | Anchor consistency — same character as IDLE F0 (body proportions, colors, equipment, style) |
| Gate 2 | LLM (vision) | Frame-to-frame continuity — smooth transition from previous frame |
| Gate 3A | LLM (vision) | Row coherence — assembled strip looks like a valid animation sequence |

All LLM gates use GPT-5-mini at **temperature 0.0** for deterministic verification.

### Retry & Escalation

When a gate fails, the frame enters a **retry loop** with 3-tier escalation:

| Tier | Attempts | Strategy | Temperature |
|------|----------|----------|-------------|
| Soft | 1–3 | Re-prompt with gate feedback appended | 1.0 |
| Guided | 4–6 | Restructured prompt with explicit constraints from failures | 0.7 |
| Constrained | 7–10 | Minimal creative freedom, line-by-line instructions | 0.3 |

**The spritesheet MUST be generated** — the system must converge. Cost is not the primary driver; accuracy and simplicity are.

### Grid Output Format

Stage 2 outputs JSON:
```json
{
  "grid": [
    "................................................................",
    "...............OOO..............................................",
    "..............OsssO.............................................",
    // ... 64 rows total, each 64 characters
  ]
}
```

Each character in the grid maps to a palette entry (e.g., `.` = transparent, `O` = outline, `s` = skin, `h` = hair). Parsed via `json.loads()`.

### Palette System

Each character has a **palette defined in its YAML config** — typically 10–15 symbols: `.` (transparent), `O` (outline/dark border), plus 8–12 character-specific color symbols. Symbols are single characters, kept mnemonic (`s`=skin, `h`=hair, `v`=vest, etc.).

**The YAML config is the single source of truth for palette data.** Palette symbols, RGB values, and outline color are all defined per-character in their YAML config file. There are no hardcoded palette constants in the Python code.

#### Semantic Palette Labeling

When `generation.auto_palette` is enabled, SpriteForge can automatically generate semantic color labels (e.g., "Skin", "Hair", "Armor") using GPT-5-nano instead of generic names like "Color 1", "Color 2". This is controlled by the `generation.semantic_labels` boolean field (default: `true`).

The LLM sees the character reference image and description to provide contextually appropriate labels. If LLM labeling fails, the system falls back to HSL-based descriptive names via `_describe_color()` (e.g., "dark greenish-blue").

Example palette section (from a YAML config):
```yaml
palette:
  outline:
    symbol: "O"
    name: "Outline"
    rgb: [20, 15, 10]
  colors:
    - symbol: "s"
      name: "Skin"
      rgb: [235, 210, 185]
    - symbol: "h"
      name: "Hair"
      rgb: [220, 185, 90]
    - symbol: "e"
      name: "Eyes"
      rgb: [50, 180, 140]
    # ... more colors as needed
```

**Key rules:**
- `.` is always transparent `(0, 0, 0, 0)` — implicitly added, NOT listed in YAML
- `O` is the outline symbol — its color is defined per-character in the `outline` section
- All other symbols are defined in the `colors` list
- Palette size is variable (different characters can have different numbers of colors)

### Spritesheet Dimensions

- **Frame size:** Configurable per character (default: 64×64 pixels)
- **Spritesheet width:** `frame_width × max_columns` pixels (default: 14 columns → 896px)
- **Spritesheet height:** `frame_height × number_of_animation_rows` pixels (varies per character)
- **Animation rows:** Variable per character — defined in the YAML config (heroes may have 16 rows, enemies may have 5)
- **Frame counts per row:** Variable (defined per animation in YAML config)
- Unused cells are fully transparent; rows are padded right to spritesheet width

### Characters

Characters are **not hardcoded** in SpriteForge. Any character can be generated by providing:
1. A self-contained YAML config file (defines everything: metadata, palette, animations, generation rules)
2. A base character reference image (PNG)

The three original reference characters for Blades of the Fallen Realm are:

| Character | Class | Config | Notes |
|-----------|-------|--------|-------|
| Theron Ashblade | Warrior | `configs/theron.yaml` | 16 animation rows, medium speed |
| Sylara Windarrow | Ranger | `configs/sylara.yaml` | 16 animation rows, fast |
| Drunn Ironhelm | Berserker | `configs/drunn.yaml` | 16 animation rows, slow |

These configs serve as **reference examples** for creating new character configs. New characters (enemies, bosses, NPCs) can have completely different animation sets, palette sizes, and frame counts.

### Animation Layout

Animation rows are **defined per character** in the YAML config. Different characters can have completely different animation sets. Example:

**Hero character** (16 rows): IDLE, WALK, ATTACK1, ATTACK2, ATTACK3, JUMP, JUMP_ATTACK, MAGIC, HIT, KNOCKDOWN, GETUP, DEATH, MOUNT_IDLE, MOUNT_ATTACK, RUN, THROW

**Enemy character** (5 rows): IDLE, WALK, ATTACK, HIT, DEATH

**Boss character** (12 rows): IDLE, WALK, ATTACK1, ATTACK2, ATTACK3, SPECIAL, HIT, KNOCKDOWN, GETUP, DEATH, RAGE, SUMMON

The only constraint is that **Row 0 Frame 0** is always the anchor frame used for cross-row consistency.

## Module Architecture

```
src/spriteforge/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point (argparse → asyncio.run) (planned)
├── app.py               # Programmatic API (run_spriteforge()) (planned)
├── preprocessor.py      # Image preprocessing (resize, quantize, auto-palette)
├── models.py            # Pydantic data models (SpritesheetSpec, CharacterConfig, AnimationDef, PaletteConfig, GenerationConfig, etc.)
├── config.py            # YAML config loading and validation (palette, generation, character sections)
├── palette.py           # Palette symbol → RGBA mapping, validation
├── renderer.py          # Grid (list[str]) → PIL Image rendering
├── assembler.py         # Row images → final spritesheet assembly (EXISTING)
├── generator.py         # Grid generation (Stage 2) using configurable chat model
├── gates.py             # Verification gates (programmatic + LLM)
├── retry.py             # Retry & escalation engine (3-tier)
├── workflow.py          # Full pipeline orchestrator (async Python + optional Agent Framework)
└── providers/           # Stage 1 reference image provider
    ├── __init__.py      # ReferenceProvider base + factory
    └── gpt_image.py     # GPT-Image-1.5 via Azure Foundry

configs/                 # Character YAML configuration files
├── template.yaml          # Annotated template for creating new characters
├── sylara.yaml            # Example: Ranger (16 rows, full hero set)
├── theron.yaml            # Example: Warrior (16 rows, full hero set)
└── drunn.yaml             # Example: Berserker (16 rows, full hero set)

tests/                   # pytest tests
├── conftest.py
├── test_models.py       # EXISTING — tests for models.py
├── test_config.py       # EXISTING — tests for config.py
├── test_app.py          # EXISTING (empty) — tests for CLI + app.py
├── test_palette.py
├── test_renderer.py
├── test_generator.py
├── test_gates.py
├── test_retry.py
├── test_workflow.py
└── test_configs.py

docs_assets/             # Reference documentation for original 3 characters
├── spritesheet_instructions_sylara.md   # Source of truth for Sylara's design
├── spritesheet_instructions_theron.md   # Source of truth for Theron's design
├── spritesheet_instructions_drunn.md    # Source of truth for Drunn's design
├── sylara_base_reference.png
├── theron_base_reference.png
└── drunn_base_reference.png
```

**Note:** For new characters, the YAML config IS the source of truth. The `docs_assets/` files only serve as reference for the three original Blades of the Fallen Realm characters.

### Module Dependency Graph

```
models.py ← config.py ← workflow.py ← app.py (planned) ← __main__.py (planned)
    ↑          ↑
    |          |
palette.py  configs/*.yaml
    ↑
    |
renderer.py ← workflow.py
    |
assembler.py (existing)

preprocessor.py ← workflow.py (optional, for auto-palette)
providers/ ← workflow.py
generator.py ← workflow.py
gates.py ← workflow.py  (depends on renderer.py, palette.py)
retry.py ← workflow.py  (depends on gates.py)
```

### Existing Code

- **`assembler.py`** — Complete and working. Imports `SpritesheetSpec` from `spriteforge.models`. Functions: `assemble_spritesheet(row_images, spec, output_path)`, `_open_image(source)`.
- **`tests/test_models.py`** — Has content (tests for models.py). Implementation must pass these.
- **`tests/test_config.py`** — Has content (tests for config.py). Implementation must pass these.
- **`tests/test_palette.py`** — Has content (tests for palette.py). Implementation must pass these.

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.12.12+ | `>=3.12.12,<3.13` |
| Package Manager | uv | `pyproject.toml` as single source of truth |
| Build Backend | hatchling | |
| Stage 1 (Reference) | GPT-Image-1.5 | Via Azure AI Foundry |
| Stage 2 (Grid Gen) | Configurable (default: `gpt-5.2`) | Via Azure AI Foundry |
| Verification Gates | Configurable (default: `gpt-5-mini`) | Temperature 0.0 for deterministic checks |
| Semantic Labeling | Configurable (default: `gpt-5-nano`) | For auto-palette color naming |
| Azure Client | `azure-ai-projects` | `AzureAIClient` with `project_endpoint` + `model_deployment_name` |
| Azure Auth | `azure-identity` | `DefaultAzureCredential` (async) |
| Image Processing | Pillow (PIL) | Grid → PNG rendering, image assembly |
| Data Models | Pydantic v2 | Config validation, model serialization |
| Config Format | YAML | `pyyaml` for loading |
| Orchestration | Plain async Python | `asyncio` loops + `asyncio.gather()` for parallel gates. Agent Framework (`agent-framework` v1.0.0b*) is an optional enhancement for observability. |
| Formatting | black 26.1.0 | |
| Type Checking | mypy 1.19.1 | |
| Testing | pytest 9.0.2 | |

### Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure Foundry | Endpoint for all deployed models (GPT-5.2, GPT-5-mini, GPT-5-nano, GPT-Image-1.5) |

Authentication to Azure uses `DefaultAzureCredential` — no API keys needed. All models are accessed through the same Azure AI Foundry project. Model deployment names are configurable per-character in the YAML config's `generation:` section.

## Project Plan (GitHub Issues)

The full implementation is tracked as GitHub issues in dependency order. Issues #1–#11 are the core implementation; issues #19–#21 are the generalization work to support any character.

### Core Implementation

| # | Issue | Module(s) | Complexity | Dependencies |
|---|-------|-----------|------------|--------------|
| [#1](https://github.com/francesco-sodano/spriteforge/issues/1) | Data Models | `models.py` | Medium | None |
| [#2](https://github.com/francesco-sodano/spriteforge/issues/2) | YAML Config Loader | `config.py` | Small | #1 |
| [#3](https://github.com/francesco-sodano/spriteforge/issues/3) | Palette System | `palette.py` | Small | #1 |
| [#4](https://github.com/francesco-sodano/spriteforge/issues/4) | Grid Renderer | `renderer.py` | Medium | #1, #3 |
| [#5](https://github.com/francesco-sodano/spriteforge/issues/5) | Reference Image Provider | `providers/` | Medium | #1 |
| [#6](https://github.com/francesco-sodano/spriteforge/issues/6) | Grid Generator | `generator.py` | Large | #1, #3 |
| [#7](https://github.com/francesco-sodano/spriteforge/issues/7) | Verification Gates | `gates.py` | Large | #1, #3, #4 |
| [#8](https://github.com/francesco-sodano/spriteforge/issues/8) | Retry & Escalation Engine | `retry.py` | Medium | #7 |
| [#9](https://github.com/francesco-sodano/spriteforge/issues/9) | Workflow Orchestrator | `workflow.py` | Large | #1–#8 |
| [#10](https://github.com/francesco-sodano/spriteforge/issues/10) | CLI Entry Point | `__main__.py`, `app.py` | Small | #2, #3, #9 |
| [#11](https://github.com/francesco-sodano/spriteforge/issues/11) | Character YAML Configs | `configs/` | Small | #1, #2, #19 |

### Generalization (Any Character Support)

| # | Issue | Module(s) | Complexity | Dependencies |
|---|-------|-----------|------------|--------------|
| [#19](https://github.com/francesco-sodano/spriteforge/issues/19) | Generalize Input Format — Self-Contained YAML | `models.py`, `config.py` | Medium | #1, #2 |
| [#20](https://github.com/francesco-sodano/spriteforge/issues/20) | Remove Hardcoded Palette Constants | `palette.py`, `__init__.py` | Small | #3, #19 |
| [#21](https://github.com/francesco-sodano/spriteforge/issues/21) | Character Config Template & Documentation | `configs/`, docs | Small | #19 |

**Suggested execution order:** #1 → #2 + #3 (parallel) → #19 → #20 + #4 + #5 + #6 (parallel) → #7 → #8 → #11 + #21 (parallel) → #9 → #10

Each issue contains: proposed solution with code signatures, acceptance criteria checklist, test plan, dependency cross-references, technical notes, and out-of-scope boundaries.

### When Implementing an Issue

1. Read the full issue — every section is implementation-critical
2. Check that all dependency issues are resolved first
3. Follow the **Proposed Solution** code signatures exactly
4. Satisfy **all** Acceptance Criteria (mandatory checklist)
5. Implement the **Test Plan** tests exactly as described
6. Respect **Out of Scope** — do NOT implement anything listed there
7. Run `black . && mypy src/ && pytest` — zero errors before submitting
8. Ensure existing tests (`test_models.py`, `test_config.py`) continue to pass

## Project Specifics
The project in this repository has the following specifications:
- Language: Python 3.12.12+.
- Package manager: uv; `pyproject.toml` is the single source of truth for all project metadata, dependencies, and tool configurations.
- Dependencies: Runtime dependencies are in `[project.dependencies]`; dev tools (e.g., black, mypy, pytest) are in the `dev` group under `[dependency-groups]`. No `requirements.txt` files are used.
- Post-create: `postCreate.sh` installs the latest Python 3.12 via `uv`, creates a `.venv`, and runs `uv sync --group dev`.
- Layout: `src/spriteforge/` for package code, `tests/` for pytest, `scripts/` for helpers, `configs/` for character YAML files, `docs_assets/` for spritesheet instruction docs and base reference images, `Dockerfile` and `.dockerignore` at the repo root.
- Builds: hatchling backend.
- Docker: Respect `.dockerignore`; keep image builds using `uv` for installs.
- All async code uses `asyncio` — the pipeline is fully async.

### Key Runtime Dependencies (to be added)

```toml
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "pillow>=10.0",
    "azure-ai-projects>=1.0.0b*",
    "azure-identity>=1.0",
    # Optional: "agent-framework>=1.0.0b*",  # for observability/streaming
]
```

## General Approach
- Your knowledge on everything is out of date because your training date is in the past. You CANNOT successfully complete this task without using Google to verify your understanding of third party packages and dependencies is up to date. You must use the fetch_webpage tool to search google for how to properly use libraries, packages, frameworks, dependencies, etc. every single time you install or implement one. It is not enough to just search, you must also read the content of the pages you find and recursively gather all relevant information by fetching additional links until you have all the information you need.
- Before writing any code, you must thoroughly investigate the codebase to understand the current state of the project. Use your available tools to explore relevant files and directories, search for key functions or variables, and read through existing code to understand its purpose and interactions. Continuously validate and update your understanding as you gather more context.
- Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Use the sequential thinking tool if available. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

## Workflow
1. Fetch any URLs provided by the user using the `fetch_webpage` tool.
2. Understand the problem deeply. Carefully read the request and think critically about what is required. Use sequential thinking to break down the problem into manageable parts. Consider the following:
   - What is the expected behavior?
   - What are the edge cases?
   - What are the potential pitfalls?
   - How does this fit into the larger context of the codebase?
   - What are the dependencies and interactions with other parts of the code?
3. Investigate the codebase. Explore relevant files, search for key functions, and gather context to understand the existing implementation.
4. Research external information. Use web searches to find up-to-date documentation for APIs, libraries, or frameworks, as your own knowledge may be outdated.
5. Develop a clear, step-by-step plan. Break down the solution into manageable, incremental steps. Display this plan as a checklist using emojis to show the status.
6. Implement the plan incrementally. Make small, testable code changes.
7. Test frequently. After each incremental change, run tests to verify correctness.
8. Debug effectively. If a test fails, do not just retry. Analyze the full error output, form a hypothesis about the root cause, and modify your code before trying again.
9. Iterate on the implementation loop (steps 6-8) until the root cause is fixed and all local tests pass.
10. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.

## Code style
- Follow PEP 8 guidelines and use `snake_case` for variables and functions.
- Format all Python code with `black`.
- Use explicit type hints for all function signatures.
- Keep functions small, focused, and pure where practical. Avoid side effects and hidden global state.
- Write clear docstrings for all public modules, classes, and functions.
- Default to absolute imports relative to the `src` directory (e.g., `from spriteforge.module import ...`).
- Use Pydantic v2 models for all data structures (configs, specs, results).
- Use `async def` for all I/O-bound operations (API calls, file operations in the pipeline).
- State machines use string enums, not integers.

## Testing
- Use `pytest` as the test runner.
- Write clear, focused unit tests for all new logic.
- Place tests under the `tests/` directory, mirroring the `src/` package structure.
- Ensure tests are self-contained and do not depend on the execution order or state of other tests.
- Cover edge cases, happy paths, and expected failure scenarios in your tests.
- Use `pytest` fixtures for reusable setup and teardown logic.
- When fixing a bug, first write a failing test that reproduces the bug, then write the code to make it pass.
- **IMPORTANT:** `tests/test_models.py` and `tests/test_config.py` already exist with content — all implementations must pass these existing tests.
- **Real Azure integration tests:** Do NOT mock Azure API calls. Tests for Azure-dependent modules (providers, generator, gates, retry, workflow) must use **real Azure AI Foundry calls** via `DefaultAzureCredential`. The Foundry project and models are live.
- **Integration test marker:** Mark tests that require Azure with `@pytest.mark.integration`. These auto-skip when `AZURE_AI_PROJECT_ENDPOINT` is unset or credentials are unavailable (handled by `conftest.py`).
- **Running tests:** `pytest` runs all tests (integration tests skip if Azure unavailable); `pytest -m "not integration"` runs only unit tests; `pytest -m integration` runs only integration tests.
- **No secrets in tests:** Tests use `DefaultAzureCredential` + `AZURE_AI_PROJECT_ENDPOINT` env var. Never hardcode endpoints, keys, or tokens in test files.
- Use shared fixtures from `conftest.py`: `azure_project_endpoint` and `azure_credential` for integration tests.
- Use `pytest-asyncio` for testing async functions.

## Security
- Always be mindful of security best practices.
- Never hardcode secrets, API keys, or other sensitive credentials in the source code. Use environment variables or a secrets management system.
- Azure authentication uses `DefaultAzureCredential` — never store Azure keys in code.
- All models are accessed through Azure AI Foundry — no separate API keys needed.
- Sanitize all external inputs to prevent injection attacks (e.g., SQL injection, command injection).
- Be cautious when adding new dependencies and prefer packages with a good reputation and active maintenance.
- Write code that is resilient to denial-of-service attacks (e.g., by avoiding unbounded resource allocation).

## Performance Considerations
- Write efficient and performant code, but avoid premature optimization.
- Consider the time and space complexity (Big O notation) of your algorithms.
- Be mindful of I/O operations; batch them when possible and consider asynchronous methods for non-blocking tasks.
- When dealing with large datasets, prefer memory-efficient approaches like generators over materializing large lists.
- Process rows sequentially but run parallel LLM gates via `asyncio.gather()` where gates are independent.
- Don't hold all row images in memory simultaneously — process row by row and save to disk.

## Memory
- Use the memory file at `.github/instructions/memory.instruction.md` to persist key information across sessions. Before starting a new task, review it for context.
- If the file does not exist, create it and add the required front matter:
  ```yaml
  ---
  applyTo: '**'
  ---
  ```
- Store the following types of information in memory:
  - **User Preferences:** Explicit requests regarding code style, communication, or workflow.
  - **Architectural Decisions:** Key design choices made during development (e.g., "Decided to use a singleton pattern for the database connection").
  - **Project Context:** Important constraints or goals not captured in the code (e.g., "The `process_data` function must be optimized for memory usage over speed").

## Documentation
- Keep all documentation in sync with code changes.
- **Docstrings:** Update docstrings for any function or class whose signature, behavior, or return value changes.
- **README.md:** Update the `README.md` file if you make changes to the project's setup, installation, or high-level usage.
- **Citations:** When implementing logic based on external APIs or libraries, cite the source documentation URL in a code comment.
- **Commit Messages:** Write clear and descriptive commit messages that explain the "what" and "why" of your changes.
- **Spritesheet Instructions:** The files in `docs_assets/spritesheet_instructions_*.md` are the **source of truth** for character designs, palettes, and animation specifications. The YAML configs in `configs/` are derived from these docs.

## Issue Templates & Autonomous Workflow
This repository uses structured GitHub Issue Templates designed for AI agents to work autonomously:

- **Feature Request** (`.github/ISSUE_TEMPLATE/feature_request.yml`): Used when proposing new features or enhancements. Contains structured sections for priority, estimated complexity, summary, motivation, proposed solution, acceptance criteria, test plan, related files & code, breaking changes flag, technical notes, out-of-scope items, dependencies on other issues, and references.
- **Bug Report** (`.github/ISSUE_TEMPLATE/bug_report.yml`): Used when reporting bugs. Contains structured sections for priority, estimated complexity, bug summary, steps to reproduce, expected/actual behavior, affected code location, current workaround, root cause analysis, proposed fix, acceptance criteria, test plan, dependencies on other issues, environment, and references.

Both templates embed the project's coding standards directly in the header so the AI agent sees them immediately.

### Working from an Issue
When assigned a GitHub Issue created from one of these templates, follow this workflow:
1. **Read the full issue** — every section contains implementation-critical information.
2. **Check dependencies** — if the issue lists dependencies on other issues, verify they are resolved before starting.
3. **Assess priority and complexity** — use these to gauge effort and plan accordingly.
4. **Follow the project standards** embedded in the issue template header (code style, testing, type checking, etc.).
5. **Review related files** (features) or **affected code** (bugs) — read these files first to understand existing patterns.
6. **Check for breaking changes** (features) — if flagged, ensure backward compatibility is handled as described.
7. **Check for workarounds** (bugs) — be aware of any temporary workarounds to avoid breaking them, or remove them as part of the fix.
8. **Follow the Proposed Solution / Proposed Fix** — the human author has specified the approach; implement it as described.
9. **Satisfy all Acceptance Criteria** — treat these as a mandatory checklist. Every box must be checked.
10. **Implement the Test Plan** — write the exact tests described in the issue.
11. **Respect Out of Scope** (features) — do NOT implement anything listed there.
12. **Run validation** — execute `black . && mypy src/ && pytest` and ensure zero errors before submitting.
13. **For bug fixes** — always write a failing test FIRST that reproduces the bug, then implement the fix.
14. **Ask for missing information** — if any section of the issue is incomplete, ambiguous, or lacks the detail needed to implement the solution, ask the issue author for clarification before proceeding. Do NOT guess or make assumptions about unclear requirements.

## SpriteForge-Specific Conventions

### Grid Generation Rules
- Grids are always 64 rows × 64 columns (64 strings of 64 characters)
- Every character in a grid must be a valid palette symbol for that character (as defined in the YAML config)
- Transparent pixels use `.` — the background of every frame MUST be transparent
- Outline pixels use `O` — every character sprite has a 1px dark outline
- Grid is parsed from JSON: `{"grid": ["...", "...", ...]}`

### Frame Generation Context
When generating a frame, the LLM receives:
1. The **anchor frame** (Row 0, Frame 0) — always included for character consistency
2. The **rough reference frame** — cropped from Stage 1's reference strip
3. The **previous frame's grid** (if not first frame) — for animation continuity
4. The **palette specification** — exact symbols and RGB values (from YAML config)
5. The **animation description** — from the YAML config's `prompt_context` field
6. The **character description** — from the YAML config's `character.description` field
7. The **generation rules** — from the YAML config's `generation.rules` field

### Provider Architecture
- `ReferenceProvider` is the base class for Stage 1 reference generation
- `GPTImageProvider` is the sole provider — uses `AzureAIClient` from `azure-ai-projects` (through Azure Foundry)
- All models (Stage 1 reference, Stage 2 grid generation, verification gates, semantic labeling) use the **same Azure AI Foundry project** and `DefaultAzureCredential`
- Model deployment names are configurable per-character via the `generation:` section of the YAML config
- Provider interface: `async generate_row_strip(base_ref, animation, char_desc) -> Image`

### Retry Conventions
- Max 10 retries per frame (hard limit)
- Gate feedback is accumulated across retries — each retry sees ALL previous gate failures
- Temperature decreases as retries escalate: 1.0 → 0.7 → 0.3
- Reference strip retries are simpler: max 3 retries, no escalation
- If a frame exhausts all retries: raise `RetryExhaustedError` (pipeline halts for that character)
- If a reference strip exhausts retries: raise `ProviderError` (not `RetryExhaustedError`)

### File Naming
- Row strips: `{character}_row{NN:02d}_{animation}.png`
- Final spritesheet: `{character}_spritesheet.png`
- Debug intermediates: `debug/{character}/row{NN:02d}/frame{FF:02d}_{attempt}.png`

## Communication
- **Tone:** Always communicate in a casual, friendly, yet professional tone.
- **Narrate Your Plan:** Announce your actions and thought process before you execute them. Use the examples below as a guide.
  <examples>
  "Let me fetch the URL you provided to gather more information."
  "Ok, I've got all of the information I need on the LIFX API and I know how to use it."
  "Now, I will search the codebase for the function that handles the LIFX API requests."
  "I need to update several files here - stand by"
  "OK! Now let's run the tests to make sure everything is working correctly."
  "Whelp - I see we have some problems. Let's fix those up."
  </examples>
- **Be Concise:** Respond with clear, direct answers. Avoid unnecessary explanations, repetition, and filler.
- **Ask for Clarity:** If a request is ambiguous, ask clarifying questions before proceeding.
- **Structure Responses:** Use bullet points and code blocks to structure information clearly.
- **Write to Files:** Always write code directly to the correct files. Do not display code in the chat unless specifically asked.
- **Elaborate Sparingly:** Only provide detailed explanations when it is essential for accuracy or user understanding.
