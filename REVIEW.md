# Code Review for SpriteForge

## Overview

This document provides a comprehensive review of the SpriteForge codebase, focusing on architecture, code quality, logic, performance, and feature suggestions. The review aims to improve the robustness, maintainability, and usability of the project.

## 1. Architecture & Dependency Management

### Provider Abstraction (Critical)
**Issue:** The project is tightly coupled to Azure AI Foundry (`azure-ai-projects`, `azure-identity`) through the provider implementations (`src/spriteforge/providers/azure_chat.py` and `gpt_image.py`). This hardlocks the tool to Azure users only.

**Recommendation:**
- Refactor `ChatProvider` and `ReferenceProvider` to be truly abstract base classes.
- Create separate provider implementations for direct API access (OpenAI, Anthropic, Google) using standard libraries like `openai` or `anthropic` (or a unified library like `litellm`).
- Use dependency injection or a factory pattern based on configuration (e.g., `provider: azure` vs `provider: openai`) to instantiate the correct provider at runtime.

### Prompt Management
**Issue:** Prompts are hardcoded as Python strings in `src/spriteforge/prompts/`. This makes it difficult for non-engineers to tweak prompts and version them separately from code logic.

**Recommendation:**
- Move prompt templates to external text files (e.g., Jinja2 templates) in a `templates/` directory.
- This allows for easier experimentation and A/B testing of prompts without modifying source code.

## 2. Code Quality & Logic

### Redundant Quantization Logic
**File:** `src/spriteforge/preprocessor.py`
**Observation:** `extract_palette_from_image` contains logic to check for quantization (`needs_quantize`) even though `preprocess_reference` already quantizes the image before calling it.
**Recommendation:** While safe, this redundancy adds complexity. Simplifying `extract_palette_from_image` to assume valid input (or just assert it) would make the code cleaner.

### Retry Logic Clarity
**File:** `src/spriteforge/retry.py`
**Observation:** The `record_failure` method logs an error ("All retries exhausted") internally when `next_attempt >= max_attempts`, but then returns the context. The actual loop termination happens in `workflow.py` via `should_retry`.
**Recommendation:**
- The logic is correct, but the split responsibility can be confusing. Consider having `record_failure` raise a specific `MaxRetriesExceededError` that the workflow catches, or returning a status object indicating whether to continue.
- Ensure 0-based indexing (`current_attempt`) and 1-based logging/user-facing numbers are consistently handled to avoid off-by-one errors (currently handled correctly but complex).

### Input Validation
**File:** `src/spriteforge/preprocessor.py`
**Observation:** `Image.open` is called on the input file without checking file size or dimensions first.
**Recommendation:**
- Add a check for file size (e.g., < 10MB) before opening to prevent DoS attacks (decompression bombs) if this code is ever exposed as a service.
- Validate dimensions more strictly before processing.

## 3. Performance & Cost Optimization

### Caching Strategy (High Impact)
**Issue:** The pipeline re-runs expensive generation steps (Stage 1 Reference, Stage 2 Grids) every time, even if inputs haven't changed or if a previous run partially succeeded.
**Recommendation:**
- Implement a caching layer (e.g., file-based hash of inputs + config) to store intermediate results.
- If `output_path` exists or a `.cache` directory has the reference strip for a given config hash, reuse it. This saves significant API costs and time.

### Interactive "Check" Mode
**Issue:** The tool runs the full expensive pipeline autonomously. If the Stage 1 reference strip is bad, the user pays for the expensive Stage 2 generation of a bad character.
**Recommendation:**
- Add an interactive mode (`--interactive`) that pauses after Stage 1 (Reference Generation) and asks the user to approve the reference strip (viewing it via CLI or opening the file) before proceeding to Stage 2.

### Cost Estimation
**Issue:** Users have no visibility into the estimated cost of a run (which can be high with Claude Opus).
**Recommendation:**
- Add a "dry-run" or "estimate" command that calculates the number of frames, estimated tokens (input + output), and projected cost based on current model pricing.

## 4. Feature Suggestions

### Semantic Palette Mapping
**Issue:** The auto-palette feature assigns generic names like "Color 1", "Color 2" to palette symbols. This deprives the AI of semantic context (e.g., "Skin", "Armor").
**Recommendation:**
- Use a small, cheap LLM call (e.g., GPT-4o-mini) or heuristics to label the extracted colors based on the reference image description.
- Example: "The reddish-brown color covering 30% of the image is likely the 'Armor'."
- This would improve the AI's ability to apply colors correctly in the grid generation phase.

### Progress Persistence
**Issue:** `workflow.py` stores generated row images in memory (`row_images` dict). If the process crashes (e.g., network error, OOM) on the last row, all progress is lost.
**Recommendation:**
- Save each completed row (or even frame) to a temporary directory or the output folder as it is generated.
- Allow resuming a run from these saved artifacts.

## 5. Security

### Azure Credential Handling
**File:** `src/spriteforge/providers/azure_chat.py`
**Observation:** `DefaultAzureCredential` is used, which is good practice. However, ensuring that credentials are not inadvertently logged or exposed in error messages is crucial.
**Recommendation:**
- Audit logging statements to ensure no sensitive information (like partial keys or tokens) is ever logged. The current logging seems safe, but explicit sanitization is a good safeguard.

## Summary

SpriteForge is a well-structured project with a clear separation of concerns. The main areas for improvement are:
1.  **Decoupling from Azure** to support a wider user base.
2.  **Cost optimization** via caching and interactive checks.
3.  **Usability improvements** like cost estimation and semantic palettes.
