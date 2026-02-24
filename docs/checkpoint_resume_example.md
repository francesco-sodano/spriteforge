# Checkpoint/Resume Support

SpriteForge supports checkpoint/resume for long-running generation pipelines. If the pipeline crashes partway through, you can resume from where it stopped without losing completed work.

## How It Works

1. **Automatic Checkpointing**: After each row completes Gate 3A verification, the workflow saves:
   - Row strip PNG to disk
   - Frame grids as JSON
   - Row metadata (animation name, row index)

2. **Resume on Restart**: When you restart with the same checkpoint directory:
   - The workflow detects which rows are already complete
   - Completed rows are loaded from disk (skipped)
   - Only remaining rows are processed

3. **Automatic Cleanup**: After successful final assembly, checkpoints are automatically deleted.

## Usage

### CLI

```bash
# Use default checkpoint location: .spriteforge/checkpoints/<character_name>
spriteforge generate configs/theron.yaml --resume

# Or provide an explicit checkpoint directory
spriteforge generate configs/theron.yaml --resume \
    --checkpoint-dir output/.spriteforge_checkpoint
```

### Programmatic API

```python
import asyncio
from pathlib import Path
from spriteforge import create_workflow, load_config

async def main() -> None:
    config = load_config("configs/theron.yaml")
    
    # Enable checkpointing by providing a checkpoint directory
    workflow = await create_workflow(
        config=config,
        checkpoint_dir="output/.spriteforge_checkpoint"  # Enable checkpoints
    )
    
    try:
        output_path = Path("output") / f"{config.character.name}_spritesheet.png"
        result_path = await workflow.run(
            base_reference_path=config.base_image_path,
            output_path=output_path,
        )
        print(f"Saved: {result_path}")
    finally:
        await workflow.close()

asyncio.run(main())
```

### Checkpoint Directory Structure

```
output/.spriteforge_checkpoint/
    row_000.png       # Row 0 strip PNG
    row_000.json      # Row 0 frame grids + metadata
    row_001.png       # Row 1 strip PNG
    row_001.json      # Row 1 frame grids + metadata
    ...
```

### Resume After Crash

Simply run the same script again with the same `checkpoint_dir`. The workflow will:
1. Detect existing checkpoints
2. Log: `"Found N completed checkpoint(s): [0, 1, 2, ...]"`
3. Skip completed rows
4. Process only remaining rows

## Example Scenario

```python
# First run - crashes after row 5
workflow = await create_workflow(
    config=config,
    checkpoint_dir="output/.spriteforge_checkpoint"
)
result = await workflow.run(base_ref, output_path)
# Crash! Rows 0-5 are saved as checkpoints

# Second run - resumes from row 6
workflow = await create_workflow(
    config=config,
    checkpoint_dir="output/.spriteforge_checkpoint"  # Same directory
)
result = await workflow.run(base_ref, output_path)
# Logs: "Found 6 completed checkpoint(s): [0, 1, 2, 3, 4, 5]"
# Loads rows 0-5 from disk, processes rows 6-15
# On success, cleans up all checkpoints
```

## Benefits

- **Crash Recovery**: Resume from crashes without losing hours of work
- **Cost Savings**: Don't re-run expensive AI model calls for completed rows
- **Flexible**: Enable/disable by simply passing or omitting `checkpoint_dir`
- **Zero Maintenance**: Checkpoints are automatically cleaned up on success

## Notes

- Checkpoints are **per-spritesheet** — use unique directories for different characters
- CLI default checkpoint directory: `.spriteforge/checkpoints/<character_name>` when `--resume` is used
- Programmatic suggested location: `{output_dir}/.spriteforge_checkpoint/`
- Each checkpoint is ~50-100KB (PNG + JSON metadata)
- Checkpoints are **not** portable between different configs or base images
- If you modify the config or base image, delete the checkpoint directory and start fresh
