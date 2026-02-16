# SpriteForge: Code Review & Improvement Plan

## 1. Executive Summary & Honest View

**Verdict:** SpriteForge is a sophisticated, well-structured tool that solves a hard problem (pixel-perfect sprite generation) with a clever two-stage architecture. The codebase is clean, modern (Python 3.12, Pydantic, `uv`), and demonstrates a strong understanding of LLM orchestration (retries, verification gates).

**The Good:**
- **Architecture**: The "Anchor Frame" + "Reference Strip" approach is a smart way to maintain character identity across animations.
- **Robustness**: The 3-tier retry strategy (Soft -> Guided -> Constrained) is excellent for handling LLM stochasticity.
- **Code Quality**: Strong typing, clean separation of concerns (Generator vs Gates vs Retry), and proper use of `asyncio` for parallelism.

**The Bad:**
- **Usability**: The lack of a CLI is the biggest barrier. Requiring users to write a Python script to run a build is friction.
- **Complexity**: `workflow.py` is becoming a "god object", managing orchestration, concurrency, file I/O, and business logic.
- **Cost/Latency Risk**: The pipeline is heavy. A single frame might trigger 3-5 LLM calls (Generation + Gate 0 + Gate 1 + Gate 2). For a 50-frame character, that's hundreds of calls.

## 2. Architectural Review

### Strengths
1.  **Verification Gates**: The concept of distinct gates (Fidelity, Identity, Temporal, Row Coherence) is the "secret sauce" that makes this viable for production assets.
2.  **Config-Driven**: The self-contained YAML approach is perfect. It treats characters as code/configuration, enabling version control.
3.  **Stateless Retry Logic**: `RetryManager` is a highlight—pure logic that decides *what* to do next without performing the side effects itself.

### Weaknesses & Risks
1.  **Code Duplication (DRY)**:
    - `workflow.py`: `_process_anchor_row` and `_process_row` share ~80% of their logic (generation loops, rendering, saving).
    - `gates.py`: The `gate_X` methods are nearly identical boilerplate (construct prompt, call chat, parse response).
    - `generator.py`: Similar repetition in `generate_anchor_frame` vs `generate_frame`.
2.  **Observability**:
    - When a frame fails 10 times, the "last tier" error is raised, but debugging *why* (viewing the 10 failed grids/images) requires digging into logs or implementing custom artifacts saving.
    - Intermediate artifacts (rough reference strips) are ephemeral unless manually saved.
3.  **Prompt Management**:
    - Prompts are strings in `prompts/`. As the system grows, managing prompt versions and A/B testing will become hard.
4.  **Dependency on Specific Models**:
    - The code heavily assumes "GPT-4o/Claude 3.5 Sonnet" level capability. If models change or degrade, the rigid "JSON Grid" format might break.

## 3. Usage & Workflow Review

**Current State:**
- **Dev-centric**: "Clone repo, install `uv`, write `run.py`, execute."
- **Feedback Loop**: Slow. Users wait for the whole row/sheet to finish. `progress_callback` exists but is minimal.

**Desired State:**
- **User-centric**: `pip install spriteforge`, then `spriteforge build hero.yaml`.
- **Interactive**: CLI spinner, incremental saving (so a crash doesn't lose the whole sheet), and a `preview` command.

## 4. Improvement Plan

### Phase 1: Simplification (DRY & Refactoring)

**Goal**: Reduce code volume and cognitive load in core files.

1.  **Refactor `workflow.py`**:
    - Extract the "Single Frame Generation Loop" (Generate -> Check -> Retry) into a dedicated `FramePipeline` class.
    - Unify `_process_anchor_row` and `_process_row` to use this pipeline.
2.  **Generic Gate Runner**:
    - In `LLMGateChecker`, create a `_run_gate(prompt, images, gate_name)` method to handle the API call and JSON parsing. The specific gate methods (0, 1, 2) should just build the prompt/inputs and call this runner.
3.  **Prompt Builders**:
    - Standardize prompt construction. Ensure all prompts use a shared base context (palette, rules) injected systematically.

### Phase 2: Usability (CLI)

**Goal**: Make the tool usable by non-programmers/designers.

1.  **Implement `src/spriteforge/cli.py`**:
    - Use `typer` or `click` for a robust CLI.
    - **Commands**:
        - `build [config_path]`: Runs the pipeline.
        - `init [name]`: Scaffolds a new YAML config from the template.
        - `verify [config_path]`: Runs static analysis on the YAML (using `validate_config`).
2.  **Better Progress Reporting**:
    - Use `rich` for beautiful progress bars and live logs during the build.
    - Show "Frame 3/6: Retrying (Attempt 2/10)..." updates.

### Phase 3: Developer Experience & Observability

**Goal**: Make debugging and iterating easier.

1.  **Artifacts Mode**:
    - Add a `--save-artifacts` flag to save failed grids/images and intermediate reference strips to a `debug/` folder.
2.  **Dry Run**:
    - `spriteforge build --dry-run`: Validates config, checks image paths, and estimates cost (token count) without calling LLMs.

### Phase 4: Character Definition (Schema)

**Goal**: Harden the configuration format.

1.  **JSON Schema**: Generate a JSON Schema from the Pydantic models. This allows VS Code (and other editors) to provide autocompletion and validation for the YAML files.
2.  **Documentation**: Move the "Character Reference Docs" (markdown) into the codebase or a proper documentation site (MkDocs), auto-generated from the Pydantic models.

## Conclusion

SpriteForge is production-ready in its logic but prototype-level in its interface. The priority should be **Phase 1 (Refactoring)** to make the code maintainable, followed immediately by **Phase 2 (CLI)** to unlock usage. The architectural risks (cost) are inherent to the "LLM-for-pixels" approach but can be mitigated with caching and better observability.
