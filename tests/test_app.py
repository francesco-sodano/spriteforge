"""Tests for spriteforge CLI and programmatic API.

This module contains tests for:
- CLI entry point (__main__.py)
- Programmatic API (app.py run_spriteforge function)

NOTE: These tests are currently stubs/TODOs pending implementation
of issue #10 (CLI Entry Point). Once __main__.py and app.py are
implemented, these tests should be filled in according to the
test plan in that issue.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# CLI Tests (TODO: dependent on issue #10)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Pending issue #10: CLI entry point not yet implemented")
def test_cli_help_flag() -> None:
    """Test that --help flag displays usage information.

    TODO (issue #10): Once __main__.py is implemented, test that:
    - Running `spriteforge --help` displays help text
    - Help text includes all expected arguments (config, output, verbose, etc.)
    - Exit code is 0
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: CLI entry point not yet implemented")
def test_cli_version_flag() -> None:
    """Test that --version flag displays version information.

    TODO (issue #10): Once __main__.py is implemented, test that:
    - Running `spriteforge --version` displays version string
    - Version matches package version from pyproject.toml
    - Exit code is 0
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: CLI entry point not yet implemented")
def test_cli_missing_required_args() -> None:
    """Test that CLI fails gracefully when required args are missing.

    TODO (issue #10): Once __main__.py is implemented, test that:
    - Running without required args shows error message
    - Error message indicates which args are required
    - Exit code is non-zero
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: CLI entry point not yet implemented")
def test_cli_invalid_config_path() -> None:
    """Test that CLI handles invalid config file path gracefully.

    TODO (issue #10): Once __main__.py is implemented, test that:
    - Running with non-existent config file shows clear error
    - Error indicates which file was not found
    - Exit code is non-zero
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: CLI entry point not yet implemented")
def test_cli_verbose_flag() -> None:
    """Test that --verbose flag enables detailed logging.

    TODO (issue #10): Once __main__.py is implemented, test that:
    - Running with --verbose produces more log output
    - Debug-level messages are visible
    - Workflow progress is reported
    """
    pass


# ---------------------------------------------------------------------------
# Programmatic API Tests (TODO: dependent on issue #10)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Pending issue #10: Programmatic API not yet implemented")
def test_run_spriteforge_basic() -> None:
    """Test run_spriteforge() with minimal valid arguments.

    TODO (issue #10): Once app.py run_spriteforge() is implemented, test that:
    - Function accepts config path and output path
    - Returns Path to generated spritesheet
    - Generated file exists and is valid PNG
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: Programmatic API not yet implemented")
def test_run_spriteforge_with_options() -> None:
    """Test run_spriteforge() with all optional parameters.

    TODO (issue #10): Once app.py run_spriteforge() is implemented, test that:
    - Function accepts verbose, log_file, progress_callback params
    - Progress callback is invoked during generation
    - Log file is created when log_file path provided
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: Programmatic API not yet implemented")
def test_run_spriteforge_invalid_config() -> None:
    """Test run_spriteforge() error handling with invalid config.

    TODO (issue #10): Once app.py run_spriteforge() is implemented, test that:
    - Function raises ConfigError for invalid YAML
    - Error message is descriptive
    - No partial output files are created
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: Programmatic API not yet implemented")
def test_run_spriteforge_missing_reference_image() -> None:
    """Test run_spriteforge() error handling with missing reference image.

    TODO (issue #10): Once app.py run_spriteforge() is implemented, test that:
    - Function raises appropriate error when reference image not found
    - Error message indicates which file is missing
    - No partial output files are created
    """
    pass


@pytest.mark.skip(reason="Pending issue #10: Programmatic API not yet implemented")
def test_run_spriteforge_progress_callback() -> None:
    """Test that progress_callback receives expected updates.

    TODO (issue #10): Once app.py run_spriteforge() is implemented, test that:
    - Callback is invoked for each pipeline stage
    - Callback receives (stage_name, current, total) args
    - Final callback indicates completion
    """
    pass
