"""Checkpoint management for long-running spritesheet generation.

Provides save/resume functionality to recover from crashes during pipeline
execution. After each row completes Gate 3A verification, the checkpoint
manager saves:
- Row strip PNG bytes
- Frame grids as JSON
- Row metadata (animation name, row index)

On resume, the workflow detects existing checkpoints and skips already-
completed rows, loading their saved outputs directly.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from spriteforge.logging import get_logger

logger = get_logger("checkpoint")


class CheckpointManager:
    """Manages checkpoints for resumable spritesheet generation.

    Creates a checkpoint directory structure:
    ```
    {checkpoint_dir}/
        row_000.png       # Row strip PNG
        row_000.json      # Frame grids + metadata
        row_001.png
        row_001.json
        ...
    ```

    Each checkpoint includes:
    - PNG bytes for the verified row strip
    - Frame grids as JSON array
    - Metadata (animation name, row index)
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
                Will be created if it doesn't exist.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._corrupt_rows: set[int] = set()
        self._lock = threading.RLock()
        logger.info("Checkpoint directory: %s", self.checkpoint_dir)

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        """Atomically write bytes to a path using same-directory temp file."""
        tmp_path = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        tmp_path.write_bytes(data)
        tmp_path.replace(path)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        """Atomically write text to a path using same-directory temp file."""
        self._atomic_write_bytes(path, text.encode("utf-8"))

    def save_row(
        self,
        row: int,
        animation_name: str,
        strip_bytes: bytes,
        grids: list[list[str]],
    ) -> None:
        """Save a completed row to disk.

        Args:
            row: Row index (0-based).
            animation_name: Name of the animation for this row.
            strip_bytes: PNG bytes of the rendered row strip.
            grids: List of frame grids (each grid is a list of 64 strings).
        """
        with self._lock:
            # Save PNG
            png_path = self.checkpoint_dir / f"row_{row:03d}.png"
            self._atomic_write_bytes(png_path, strip_bytes)

            # Save metadata + grids as JSON
            json_path = self.checkpoint_dir / f"row_{row:03d}.json"
            data: dict[str, Any] = {
                "version": self.CHECKPOINT_VERSION,
                "row": row,
                "animation_name": animation_name,
                "grids": grids,
            }
            self._atomic_write_text(json_path, json.dumps(data, indent=2))

        logger.debug(
            "Saved checkpoint for row %d (%s): %d frames",
            row,
            animation_name,
            len(grids),
        )

    def load_row(self, row: int) -> tuple[bytes, list[list[str]]] | None:
        """Load a saved row checkpoint.

        Args:
            row: Row index (0-based).

        Returns:
            Tuple of (strip_bytes, grids) if checkpoint exists, else None.
        """
        with self._lock:
            png_path = self.checkpoint_dir / f"row_{row:03d}.png"
            json_path = self.checkpoint_dir / f"row_{row:03d}.json"

            if not png_path.exists() or not json_path.exists():
                return None

            strip_bytes = png_path.read_bytes()

            try:
                raw = json.loads(json_path.read_text())
            except json.JSONDecodeError as exc:
                logger.error("Checkpoint JSON for row %d is corrupt: %s", row, exc)
                self._corrupt_rows.add(row)
                return None

            if not isinstance(raw, dict):
                logger.error(
                    "Checkpoint JSON for row %d has unexpected type %s; expected object",
                    row,
                    type(raw).__name__,
                )
                self._corrupt_rows.add(row)
                return None

            version = int(raw.get("version", 0))
            if version > self.CHECKPOINT_VERSION:
                logger.error(
                    "Checkpoint for row %d has unsupported version %d (max supported %d)",
                    row,
                    version,
                    self.CHECKPOINT_VERSION,
                )
                self._corrupt_rows.add(row)
                return None

            if "grids" not in raw:
                logger.error("Checkpoint for row %d is missing 'grids' key", row)
                self._corrupt_rows.add(row)
                return None

            grids = raw["grids"]
            if not isinstance(grids, list) or not all(
                isinstance(g, list) and all(isinstance(r, str) for r in g)
                for g in grids
            ):
                logger.error("Checkpoint for row %d has malformed 'grids' value", row)
                self._corrupt_rows.add(row)
                return None

            logger.debug(
                "Loaded checkpoint for row %d (%s): %d frames",
                row,
                raw.get("animation_name", "unknown"),
                len(grids),
            )
            self._corrupt_rows.discard(row)

            return strip_bytes, grids

    def completed_rows(self) -> set[int]:
        """Get the set of row indices that have completed checkpoints.

        Returns:
            Set of row indices (0-based) that have both PNG and JSON files.
        """
        with self._lock:
            completed: set[int] = set()

            # Look for row_NNN.json files
            for json_path in self.checkpoint_dir.iterdir():
                if not json_path.is_file():
                    continue
                if not json_path.name.lower().startswith("row_"):
                    continue
                if json_path.suffix.lower() != ".json":
                    continue
                # Extract row number from filename
                try:
                    row_num = int(json_path.stem.split("_")[1])
                    # Verify PNG exists too
                    png_matches = list(self.checkpoint_dir.glob(f"row_{row_num:03d}.*"))
                    has_png = any(
                        p.is_file() and p.suffix.lower() == ".png" for p in png_matches
                    )
                    if has_png:
                        completed.add(row_num)
                except (ValueError, IndexError):
                    logger.warning("Skipping invalid checkpoint file: %s", json_path)

            return completed

    @property
    def corrupt_rows(self) -> set[int]:
        """Rows for which checkpoint files were detected as corrupted."""
        return set(self._corrupt_rows)

    def cleanup(self) -> None:
        """Remove all checkpoint files after successful completion.

        This should be called after the final spritesheet is successfully
        assembled and saved.
        """
        with self._lock:
            if not self.checkpoint_dir.exists():
                return

            # Remove all checkpoint files
            file_count = 0
            for file_path in self.checkpoint_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    file_count += 1

            logger.info(
                "Cleaned up %d checkpoint files from %s",
                file_count,
                self.checkpoint_dir,
            )

            # Remove the checkpoint directory if empty
            try:
                self.checkpoint_dir.rmdir()
                logger.debug("Removed checkpoint directory: %s", self.checkpoint_dir)
            except OSError:
                # Directory not empty (might have subdirs or other files)
                logger.debug(
                    "Checkpoint directory not empty, skipping removal: %s",
                    self.checkpoint_dir,
                )
