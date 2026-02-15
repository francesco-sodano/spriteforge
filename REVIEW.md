# Code Review and Optimization Plan

## 1. Critical Logic Flaws: Hardcoded Frame Dimensions

**Severity**: High
**Impact**: Prevents variable frame sizes (e.g., 32x32, 128x128), violating requirement #3.

The codebase supports `frame_width` and `frame_height` configuration in `CharacterConfig` and correctly passes them to the `Preprocessor` and `Renderer`. However, the **Stage 2 Grid Generator** completely ignores these settings and enforces 64x64 dimensions.

### Evidence
- `src/spriteforge/generator.py`: `_build_system_prompt` defaults to `width=64, height=64`.
- `src/spriteforge/generator.py`: `parse_grid_response` defaults to `expected_rows=64, expected_cols=64`.
- `src/spriteforge/generator.py`: `QUANTIZED_REFERENCE_SECTION` is formatted with hardcoded `width=64, height=64`.
- `src/spriteforge/workflow.py`: Calls `grid_generator.generate_anchor_frame` and `generate_frame` without passing dimensions.

### Optimization Plan
1.  **Refactor `GridGenerator`**: Update `generate_anchor_frame` and `generate_frame` to accept `frame_width` and `frame_height` arguments.
2.  **Update Prompt Builders**: Pass these dimensions to `_build_system_prompt`, `build_anchor_frame_prompt`, and `build_frame_prompt`.
3.  **Update Parser**: Pass `frame_height` (rows) and `frame_width` (cols) to `parse_grid_response`.
4.  **Update Workflow**: In `SpriteForgeWorkflow`, extract dimensions from `self.config.character` and pass them to the generator calls.

---

## 2. API Integration: Azure `gpt-image-1.5`

**Severity**: Medium
**Impact**: Potential runtime errors if the API contract is misunderstood.

The code uses `openai_client.images.edit` for reference generation. This is correct for `gpt-image-1.5` on Azure (which supports editing), but it deviates from the standard DALL-E 3 API (which only supports `generate`).

### Evidence
- `src/spriteforge/providers/gpt_image.py` calls `images.edit`.
- **Verified**: Azure documentation confirms `gpt-image-1.5` supports editing and the `input_fidelity` parameter.
- **Risk**: The memory mentioned "Responses API" as an alternative. This might refer to a specific Azure API version nuance.

### Optimization Plan
1.  **Explicit Versioning**: Ensure the `api_version` used in `AzureOpenAI` client init (implicitly handled by `AIProjectClient`) matches the requirements for `gpt-image-1.5`.
2.  **Error Handling**: Add specific error handling for `400 Bad Request` that checks if the error message relates to "operation not supported" to give a clearer error to the user (e.g., "Model does not support editing").

---

## 3. Data Model: Palette Transparency

**Severity**: Low (Feature Limitation)
**Impact**: Limits artistic style (no semi-transparent effects like smoke/glass).

The `PaletteColor` model enforces fully opaque colors (`a=255`). While this satisfies the "transparent background" requirement, it prevents advanced pixel art techniques.

### Evidence
- `src/spriteforge/models.py`: `PaletteColor.rgba` property returns `(r, g, b, 255)`.
- `src/spriteforge/preprocessor.py`: Discards alpha channel for non-transparent pixels during palette extraction.

### Optimization Plan
1.  **Update `PaletteColor`**: Add an optional `a: int = 255` field.
2.  **Update Preprocessor**: Modify `extract_palette_from_image` to preserve alpha values. Use 4-channel quantization (RGBA) instead of 3-channel (RGB) + separate Alpha mask.

---

## 4. Code Quality & Preprocessing

**Severity**: Medium
**Impact**: Image quality issues and potential artifacts.

### Preprocessing Algorithms
- **Downscaling**: `resize_reference` uses `Image.Resampling.NEAREST`. This is excellent for **upscaling** pixel art, but for **downscaling** a high-res reference image (e.g., 1024x1024 -> 64x64), it causes severe aliasing and loss of detail.
    - **Fix**: Use `Image.Resampling.LANCZOS` or `BOX` when downscaling, and `NEAREST` only when upscaling.

### Quantization Logic
- The quantization logic in `preprocessor.py` converts RGBA -> RGB -> Quantize -> RGBA. This risks color bleeding between the background color (often black/white in RGB conversion) and the sprite edges.
    - **Fix**: Quantize on the premultiplied alpha or strictly on masked pixels.

### Concurrency
- `src/spriteforge/workflow.py` uses a manual semaphore and lock for progress reporting.
    - **Optimization**: Use `asyncio.as_completed` for cleaner iteration and progress tracking, removing the need for a manual lock inside the coroutine.

---

## 5. Testing Strategy

**Severity**: Medium
**Impact**: Tests pass but code might fail in production.

Existing tests use `MagicMock` for Azure clients without validating the schema.
- **Optimization**: Introduce integration tests (marked with `@pytest.mark.integration`) that hit the real Azure endpoint (or a high-fidelity mock) to verify the `images.edit` payload structure.
