# Copilot Custom Instructions

Guidance for Copilot behavior for this repository.

## Persona
You are an expert, world-class software engineering assistant. Your goal is to help write clean, efficient, and correct code while following the project's standards and instructions.

## Project Overview
The project overview is the README.md file. It provides a high-level description of the project and its purpose. Always keep that in consideration and remember it.

**SpriteForge** is an AI-powered spritesheet generator for 2D pixel-art games. It takes a base character reference image and a YAML config file, then uses a **two-stage AI pipeline** to produce a game-ready 896×1024 spritesheet PNG with transparent backgrounds.

The target game is **Blades of the Fallen Realm** — a Golden Axe-style beat 'em up. SpriteForge generates spritesheets for three playable characters: Theron Ashblade (Warrior), Sylara Windarrow (Ranger), and Drunn Ironhelm (Berserker).

## Architecture

### Two-Stage Pipeline

The core innovation is a **two-stage generation approach** that mimics how human pixel artists work: sketch first, then pixel.

```
Stage 1 (Image Model) → Gate -1 → Stage 2 (Claude Vision → Grid) → Gates → Retry → Render → Assemble
```

- **Stage 1 — Reference Generation:** An image-generation model (Gemini 2.5/3 Image or GPT-Image-1.5) produces a rough, non-pixel-precise animation strip for each row. This is a visual reference, NOT the final output.
- **Stage 2 — Grid Generation:** Claude Opus 4.6 (with vision) receives the rough reference + anchor frame + palette spec, and outputs a **pixel-precise 64×64 JSON grid** (64 strings of 64 single-character palette symbols). This grid IS the final output — rendered to PNG by pure Python code.

### Key Design Principle

> "Don't ask an AI to generate an image — ask it to generate a **data structure** that you render into an image."

The LLM never outputs pixels directly. It outputs a structured grid of palette symbols. A deterministic renderer (`renderer.py`) converts this grid to PNG using exact RGB values. This eliminates color drift, anti-aliasing artifacts, and resolution inconsistencies.

### Pipeline Flow (Per Character)

```
1. Load config (YAML → SpritesheetSpec)
2. For Row 0 (IDLE) — anchor row:
   a. Stage 1: Generate rough reference strip
   b. Gate -1: Validate reference quality (LLM)
   c. Stage 2: Generate IDLE Frame 0 (the anchor frame)
   d. Programmatic checks + Gate 0 + Gate 1 (LLM)
   e. Retry loop (up to 10 attempts with 3-tier escalation)
   f. Generate remaining IDLE frames (with anchor + prev frame context)
   g. Gate 3A: Validate assembled IDLE row coherence (LLM)
3. For Rows 1–15:
   a. Stage 1: Generate rough reference strip for this animation
   b. Gate -1: Validate reference quality
   c. For each frame:
      - Stage 2: Generate frame (with anchor + reference + prev frame)
      - Programmatic checks → Gate 0 → Gate 1 (→ Gate 2 if not first frame)
      - Retry loop if any gate fails
   d. Gate 3A: Validate assembled row coherence
4. Assemble all 16 rows into final 896×1024 spritesheet
5. Save PNG output
```

### Row Independence

Rows are **independent** of each other. Only **IDLE Frame 0** serves as the cross-row anchor. There is no dependency graph between rows — they are processed sequentially (Row 0 → Row 15) but each row only needs the anchor frame as shared context.

### Verification Gates

A multi-gate verification system ensures quality:

| Gate | Type | What It Checks |
|------|------|----------------|
| Gate -1 | LLM (vision) | Reference strip quality — correct pose, frame count, character identity |
| Programmatic | Code | Grid dimensions (64×64), valid palette symbols, non-empty, transparent background % |
| Gate 0 | LLM (vision) | Single-frame quality — anatomy, proportions, pose matches animation |
| Gate 1 | LLM (vision) | Palette compliance — colors match spec, no off-palette pixels |
| Gate 2 | LLM (vision) | Frame-to-frame continuity — smooth transition from previous frame |
| Gate 3A | LLM (vision) | Row coherence — assembled strip looks like a valid animation sequence |

All LLM gates use Claude Opus 4.6 at **temperature 0.0** for deterministic verification.

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

Each character has **12 symbols**: `.` (transparent), `O` (outline/dark border), plus 10 character-specific color symbols. Symbols are single characters, kept mnemonic (`s`=skin, `h`=hair, `v`=vest, etc.).

Example (Sylara):
```
. = transparent (0, 0, 0, 0)
O = outline (0, 80, 80, 255)
s = skin (235, 210, 185, 255)
h = hair (220, 185, 90, 255)
e = eyes (50, 180, 140, 255)
v = vest (50, 100, 45, 255)
p = pants (40, 75, 35, 255)
b = bracers (110, 75, 40, 255)
t = boots (65, 45, 30, 255)
w = bow wood (180, 150, 90, 255)
d = blade steel (190, 200, 210, 255)
f = fletching (255, 255, 240, 255)
```

### Spritesheet Dimensions

- **Frame size:** 64×64 pixels
- **Spritesheet:** 896×1024 pixels (14 columns × 64px, 16 rows × 64px)
- **16 animation rows** per character, each with varying frame counts (3–8 frames)
- **3 characters:** Sylara Windarrow, Theron Ashblade, Drunn Ironhelm
- Unused cells are fully transparent; rows are padded right to 896px

### Characters

| Character | Class | Key Visual Features | Speed |
|-----------|-------|---------------------|-------|
| Theron Ashblade | Warrior | Crimson cape, longsword (Emberfang), dark steel breastplate | Medium |
| Sylara Windarrow | Ranger | Elven ears, recurve bow, long golden hair, forest-green leather, no cape | Fast (10/10) |
| Drunn Ironhelm | Berserker | Horned helm, braided red beard, twin axes, stocky/wide dwarf | Slow (4/10) |

### Animation Rows (Identical Layout for All 3 Characters)

| Row | Animation | Frames | Looping | Notes |
|-----|-----------|--------|---------|-------|
| 0 | IDLE | 6 | Yes | Anchor row — Frame 0 is the cross-row anchor |
| 1 | WALK | 8 | Yes | |
| 2 | ATTACK1 | 5 | No | First hit of 3-hit combo |
| 3 | ATTACK2 | 5 | No | Second hit |
| 4 | ATTACK3 | 7 | No | Combo finisher (knockdown) |
| 5 | JUMP | 4 | No | |
| 6 | JUMP_ATTACK | 4 | No | |
| 7 | MAGIC | 8 | No | 3 tiers (VFX overlay handled by game) |
| 8 | HIT | 3 | No | |
| 9 | KNOCKDOWN | 4 | No | |
| 10 | GETUP | 4 | No | |
| 11 | DEATH | 6 | No | Holds on last frame |
| 12 | MOUNT_IDLE | 4 | Yes | Upper body only (waist up) |
| 13 | MOUNT_ATTACK | 5 | No | Upper body only |
| 14 | RUN | 6 | Yes | |
| 15 | THROW | 6 | No | Enemy NOT shown |

## Module Architecture

```
src/spriteforge/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point (argparse → asyncio.run)
├── app.py               # Programmatic API (run_spriteforge())
├── models.py            # Pydantic data models (SpritesheetSpec, CharacterSpec, AnimationSpec, PaletteConfig, etc.)
├── config.py            # YAML config loading and validation
├── palette.py           # Palette symbol → RGBA mapping, validation
├── renderer.py          # Grid (list[str]) → PIL Image rendering
├── assembler.py         # Row images → final spritesheet assembly (EXISTING)
├── generator.py         # Claude Opus 4.6 grid generation (Stage 2)
├── gates.py             # Verification gates (programmatic + LLM)
├── retry.py             # Retry & escalation engine (3-tier)
├── workflow.py          # Full pipeline orchestrator (async)
└── providers/           # Stage 1 reference image providers
    ├── __init__.py      # ReferenceProvider ABC
    ├── gemini.py        # Google Gemini Image (google-genai SDK)
    └── gpt_image.py     # GPT-Image-1.5 via Azure Foundry

configs/                 # Character YAML configuration files
├── sylara.yaml
├── theron.yaml
└── drunn.yaml

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

docs_assets/             # Character spritesheet instructions (source of truth)
├── spritesheet_instructions_sylara.md
├── spritesheet_instructions_theron.md
├── spritesheet_instructions_drunn.md
├── sylara_base_reference.png
├── theron_base_reference.png
└── drunn_base_reference.png
```

### Module Dependency Graph

```
models.py ← config.py ← workflow.py ← app.py ← __main__.py
    ↑          ↑
    |          |
palette.py  configs/*.yaml
    ↑
    |
renderer.py ← workflow.py
    |
assembler.py (existing)

providers/ ← workflow.py
generator.py ← workflow.py
gates.py ← workflow.py  (depends on renderer.py, palette.py)
retry.py ← workflow.py  (depends on gates.py)
```

### Existing Code

- **`assembler.py`** — Complete and working. Imports `SpritesheetSpec` from `spriteforge.models`. Functions: `assemble_spritesheet(row_images, spec, output_path)`, `_open_image(source)`.
- **`tests/test_models.py`** — Has content (tests for models.py). Implementation must pass these.
- **`tests/test_config.py`** — Has content (tests for config.py). Implementation must pass these.

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.12.12+ | `>=3.12.12,<3.13` |
| Package Manager | uv | `pyproject.toml` as single source of truth |
| Build Backend | hatchling | |
| Stage 1 (Reference) | Gemini 2.5/3 Image OR GPT-Image-1.5 | Provider-agnostic interface, config-driven selection |
| Stage 2 (Grid Gen) | Claude Opus 4.6 (vision) | Via Azure AI Foundry |
| Verification Gates | Claude Opus 4.6 | Temperature 0.0 for deterministic checks |
| Azure Client | `azure-ai-projects` | `AzureAIClient` with `project_endpoint` + `model_deployment_name` |
| Azure Auth | `azure-identity` | `DefaultAzureCredential` (async) |
| Gemini Client | `google-genai` | Direct Google API (NOT through Azure Foundry) |
| Image Processing | Pillow (PIL) | Grid → PNG rendering, image assembly |
| Data Models | Pydantic v2 | Config validation, model serialization |
| Config Format | YAML | `pyyaml` for loading |
| Orchestration | Microsoft Agent Framework (optional) | `agent-framework` v1.0.0b* — WorkflowBuilder, Executor |
| Formatting | black 26.1.0 | |
| Type Checking | mypy 1.19.1 | |
| Testing | pytest 9.0.2 | |

### Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure Foundry | Endpoint for Claude + GPT-Image |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google | API key for Gemini image generation |

Authentication to Azure uses `DefaultAzureCredential` — no API keys needed. Gemini requires a Google API key.

## Project Plan (GitHub Issues)

The full implementation is tracked as 11 GitHub issues in dependency order:

| # | Issue | Module(s) | Complexity | Dependencies |
|---|-------|-----------|------------|--------------|
| [#1](https://github.com/francesco-sodano/spriteforge/issues/1) | Data Models | `models.py` | Medium | None |
| [#2](https://github.com/francesco-sodano/spriteforge/issues/2) | YAML Config Loader | `config.py` | Small | #1 |
| [#3](https://github.com/francesco-sodano/spriteforge/issues/3) | Palette System | `palette.py` | Small | #1 |
| [#4](https://github.com/francesco-sodano/spriteforge/issues/4) | Grid Renderer | `renderer.py` | Medium | #1, #3 |
| [#5](https://github.com/francesco-sodano/spriteforge/issues/5) | Reference Image Providers | `providers/` | Large | #1 |
| [#6](https://github.com/francesco-sodano/spriteforge/issues/6) | Grid Generator | `generator.py` | Large | #1, #3 |
| [#7](https://github.com/francesco-sodano/spriteforge/issues/7) | Verification Gates | `gates.py` | Large | #1, #3, #4 |
| [#8](https://github.com/francesco-sodano/spriteforge/issues/8) | Retry & Escalation Engine | `retry.py` | Medium | #7 |
| [#9](https://github.com/francesco-sodano/spriteforge/issues/9) | Workflow Orchestrator | `workflow.py` | Large | #1–#8 |
| [#10](https://github.com/francesco-sodano/spriteforge/issues/10) | CLI Entry Point | `__main__.py`, `app.py` | Small | #2, #3, #5, #9 |
| [#11](https://github.com/francesco-sodano/spriteforge/issues/11) | Character YAML Configs | `configs/` | Small | #1, #2 |

**Suggested execution order:** #1 → #2 + #3 (parallel) → #4 + #5 + #6 (parallel) → #7 → #8 → #11 → #9 → #10

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
    "google-genai>=1.0",
    # "agent-framework>=1.0.0b*",  # optional — for workflow orchestration
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
- Mock all external API calls (Azure, Google, Claude) in tests — no real API calls in the test suite.
- Use `pytest-asyncio` for testing async functions.

## Security
- Always be mindful of security best practices.
- Never hardcode secrets, API keys, or other sensitive credentials in the source code. Use environment variables or a secrets management system.
- Azure authentication uses `DefaultAzureCredential` — never store Azure keys in code.
- Google/Gemini API key comes from `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable.
- Sanitize all external inputs to prevent injection attacks (e.g., SQL injection, command injection).
- Be cautious when adding new dependencies and prefer packages with a good reputation and active maintenance.
- Write code that is resilient to denial-of-service attacks (e.g., by avoiding unbounded resource allocation).

## Performance Considerations
- Write efficient and performant code, but avoid premature optimization.
- Consider the time and space complexity (Big O notation) of your algorithms.
- Be mindful of I/O operations; batch them when possible and consider asynchronous methods for non-blocking tasks.
- When dealing with large datasets, prefer memory-efficient approaches like generators over materializing large lists.
- Process rows sequentially but run parallel LLM gates via `asyncio.gather()` where gates are independent.
- Don't hold all 16 row images in memory simultaneously — process row by row and save to disk.

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
- Every character in a grid must be a valid palette symbol for that character
- Transparent pixels use `.` — the background of every frame MUST be transparent
- Outline pixels use `O` — every character sprite has a 1px dark outline
- Grid is parsed from JSON: `{"grid": ["...", "...", ...]}`

### Frame Generation Context
When generating a frame, the LLM receives:
1. The **anchor frame** (IDLE F0) — always included for character consistency
2. The **rough reference frame** — cropped from Stage 1's reference strip
3. The **previous frame's grid** (if not first frame) — for animation continuity
4. The **palette specification** — exact symbols and RGB values
5. The **animation description** — from the YAML config's `prompt_context` field

### Provider Architecture
- `ReferenceProvider` is an abstract base class
- `GeminiImageProvider` uses the `google-genai` SDK (direct Google API, NOT through Azure)
- `GPTImageProvider` uses `AzureAIClient` from `azure-ai-projects` (through Azure Foundry)
- Provider is selected via config or `--provider` CLI flag
- Both providers implement the same interface: `async generate_row_strip(base_ref, animation, char_desc) -> Image`

### Retry Conventions
- Max 10 retries per frame (hard limit)
- Gate feedback is accumulated across retries — each retry sees ALL previous gate failures
- Temperature decreases as retries escalate: 1.0 → 0.7 → 0.3
- Reference strip retries are simpler: max 3 retries, no escalation
- If a frame exhausts all retries: raise `GenerationError` (pipeline halts for that character)

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
