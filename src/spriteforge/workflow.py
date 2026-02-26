"""Compatibility facade for workflow orchestration and factory wiring."""

from collections.abc import Callable
from pathlib import Path

from spriteforge.assembler import assemble_spritesheet
from spriteforge.checkpoint import CheckpointManager
from spriteforge.models import SpritesheetSpec
from spriteforge.pipeline import (
    AnchorRecoveryDecision,
    AnchorRecoveryPolicy,
    AnchorRecoveryState,
    CredentialHandle,
    SpriteForgeWorkflow,
)
from spriteforge.workflow_factory import create_workflow


async def assemble_final_spritesheet(
    row_images: dict[int, bytes],
    config: SpritesheetSpec,
    output_path: Path,
    checkpoint_manager: CheckpointManager | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Compatibility wrapper around final assembly stage for legacy patch points."""
    if progress_callback:
        progress_callback("assembly", 0, 1)

    assemble_spritesheet(row_images, config, output_path=output_path)

    if checkpoint_manager is not None:
        checkpoint_manager.cleanup()

    if progress_callback:
        progress_callback("assembly", 1, 1)

    return output_path


__all__ = [
    "AnchorRecoveryDecision",
    "AnchorRecoveryPolicy",
    "AnchorRecoveryState",
    "assemble_spritesheet",
    "CredentialHandle",
    "SpriteForgeWorkflow",
    "assemble_final_spritesheet",
    "create_workflow",
]
