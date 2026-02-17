"""Tests for spriteforge.checkpoint — checkpoint/resume functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from spriteforge.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal 1×1 transparent PNG for use as image bytes.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_grid() -> list[str]:
    """Create a minimal 64×64 grid."""
    return ["." * 64 for _ in range(64)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that initializing CheckpointManager creates the directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        assert not checkpoint_dir.exists()

        manager = CheckpointManager(checkpoint_dir)

        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()
        assert manager.checkpoint_dir == checkpoint_dir

    def test_init_with_existing_directory(self, tmp_path: Path) -> None:
        """Test that initializing with existing directory works."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        manager = CheckpointManager(checkpoint_dir)

        assert checkpoint_dir.exists()
        assert manager.checkpoint_dir == checkpoint_dir

    def test_save_row_creates_files(self, tmp_path: Path) -> None:
        """Test that save_row creates PNG and JSON files."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        grids = [_make_grid(), _make_grid()]

        manager.save_row(
            row=0,
            animation_name="idle",
            strip_bytes=_TINY_PNG,
            grids=grids,
        )

        png_path = manager.checkpoint_dir / "row_000.png"
        json_path = manager.checkpoint_dir / "row_000.json"

        assert png_path.exists()
        assert json_path.exists()
        assert png_path.read_bytes() == _TINY_PNG

    def test_save_row_json_content(self, tmp_path: Path) -> None:
        """Test that save_row JSON contains expected data."""
        import json

        manager = CheckpointManager(tmp_path / "checkpoints")
        grids = [_make_grid(), _make_grid()]

        manager.save_row(
            row=5,
            animation_name="walk",
            strip_bytes=_TINY_PNG,
            grids=grids,
        )

        json_path = manager.checkpoint_dir / "row_005.json"
        data = json.loads(json_path.read_text())

        assert data["row"] == 5
        assert data["animation_name"] == "walk"
        assert len(data["grids"]) == 2
        assert data["grids"][0] == grids[0]

    def test_load_row_success(self, tmp_path: Path) -> None:
        """Test that load_row returns saved data."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        grids = [_make_grid(), _make_grid(), _make_grid()]

        manager.save_row(
            row=3,
            animation_name="attack",
            strip_bytes=_TINY_PNG,
            grids=grids,
        )

        result = manager.load_row(3)

        assert result is not None
        strip_bytes, loaded_grids = result
        assert strip_bytes == _TINY_PNG
        assert loaded_grids == grids

    def test_load_row_missing(self, tmp_path: Path) -> None:
        """Test that load_row returns None for missing checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        result = manager.load_row(99)

        assert result is None

    def test_load_row_incomplete_checkpoint(self, tmp_path: Path) -> None:
        """Test that load_row returns None if only PNG or JSON exists."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        # Save only PNG
        png_path = manager.checkpoint_dir / "row_010.png"
        png_path.write_bytes(_TINY_PNG)

        result = manager.load_row(10)
        assert result is None

        # Save only JSON
        import json

        json_path = manager.checkpoint_dir / "row_011.json"
        json_path.write_text(json.dumps({"row": 11, "grids": []}))

        result = manager.load_row(11)
        assert result is None

    def test_completed_rows_empty(self, tmp_path: Path) -> None:
        """Test that completed_rows returns empty set for new manager."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        completed = manager.completed_rows()

        assert completed == set()

    def test_completed_rows_single(self, tmp_path: Path) -> None:
        """Test that completed_rows detects single checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.save_row(0, "idle", _TINY_PNG, [_make_grid()])

        completed = manager.completed_rows()

        assert completed == {0}

    def test_completed_rows_multiple(self, tmp_path: Path) -> None:
        """Test that completed_rows detects multiple checkpoints."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.save_row(0, "idle", _TINY_PNG, [_make_grid()])
        manager.save_row(2, "walk", _TINY_PNG, [_make_grid()])
        manager.save_row(5, "attack", _TINY_PNG, [_make_grid()])

        completed = manager.completed_rows()

        assert completed == {0, 2, 5}

    def test_completed_rows_ignores_incomplete(self, tmp_path: Path) -> None:
        """Test that completed_rows ignores incomplete checkpoints."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.save_row(0, "idle", _TINY_PNG, [_make_grid()])

        # Create incomplete checkpoint (only JSON)
        import json

        json_path = manager.checkpoint_dir / "row_001.json"
        json_path.write_text(json.dumps({"row": 1, "grids": []}))

        completed = manager.completed_rows()

        assert completed == {0}
        assert 1 not in completed

    def test_completed_rows_ignores_invalid_filenames(self, tmp_path: Path) -> None:
        """Test that completed_rows ignores invalid filenames."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.save_row(0, "idle", _TINY_PNG, [_make_grid()])

        # Create files with invalid names
        (manager.checkpoint_dir / "invalid.json").write_text("{}")
        (manager.checkpoint_dir / "row_abc.json").write_text("{}")
        (manager.checkpoint_dir / "row_.json").write_text("{}")

        completed = manager.completed_rows()

        assert completed == {0}

    def test_cleanup_removes_files(self, tmp_path: Path) -> None:
        """Test that cleanup removes all checkpoint files."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.save_row(0, "idle", _TINY_PNG, [_make_grid()])
        manager.save_row(1, "walk", _TINY_PNG, [_make_grid()])

        assert len(list(manager.checkpoint_dir.iterdir())) == 4  # 2 PNG + 2 JSON

        manager.cleanup()

        # Directory might be removed if empty
        if manager.checkpoint_dir.exists():
            assert len(list(manager.checkpoint_dir.iterdir())) == 0
        else:
            # Directory was removed (which is also valid)
            assert not manager.checkpoint_dir.exists()

    def test_cleanup_empty_directory(self, tmp_path: Path) -> None:
        """Test that cleanup works on empty directory."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        # Should not raise
        manager.cleanup()

    def test_cleanup_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test that cleanup works when directory doesn't exist."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)
        manager.checkpoint_dir.rmdir()  # Remove directory

        # Should not raise
        manager.cleanup()

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Test complete save/load cycle preserves data."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        grids = [
            _make_grid(),
            _make_grid(),
            _make_grid(),
            _make_grid(),
        ]

        manager.save_row(7, "jump", _TINY_PNG, grids)
        result = manager.load_row(7)

        assert result is not None
        strip_bytes, loaded_grids = result
        assert strip_bytes == _TINY_PNG
        assert loaded_grids == grids
        assert len(loaded_grids) == 4

    def test_multiple_rows_independent(self, tmp_path: Path) -> None:
        """Test that multiple rows can be saved/loaded independently."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        grid1 = [_make_grid()]
        grid2 = [_make_grid(), _make_grid()]
        grid3 = [_make_grid(), _make_grid(), _make_grid()]

        manager.save_row(0, "idle", _TINY_PNG, grid1)
        manager.save_row(1, "walk", _TINY_PNG, grid2)
        manager.save_row(2, "attack", _TINY_PNG, grid3)

        # Load in different order
        result2 = manager.load_row(1)
        result0 = manager.load_row(0)
        result2_again = manager.load_row(1)

        assert result0 is not None
        assert result2 is not None
        assert result2_again is not None

        _, grids0 = result0
        _, grids2 = result2
        _, grids2_again = result2_again

        assert len(grids0) == 1
        assert len(grids2) == 2
        assert len(grids2_again) == 2
        assert grids2 == grids2_again
