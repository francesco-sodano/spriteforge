"""Tests for spriteforge CLI commands.

This module tests the CLI commands (generate, validate, estimate)
implemented in src/spriteforge/cli.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from spriteforge.cli import main


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def test_config_path() -> Path:
    """Path to a test configuration file."""
    return Path("configs/theron.yaml")


@pytest.fixture
def test_config_simple() -> Path:
    """Path to a simple test configuration file."""
    # Use a minimal config if available
    simple_path = Path("configs/examples/simple_enemy.yaml")
    if simple_path.exists():
        return simple_path
    return Path("configs/theron.yaml")


# ---------------------------------------------------------------------------
# CLI Entry Point Tests
# ---------------------------------------------------------------------------


def test_cli_help_flag(cli_runner: CliRunner) -> None:
    """Test that --help flag displays usage information."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "SpriteForge" in result.output
    assert "generate" in result.output
    assert "init" in result.output
    assert "validate" in result.output
    assert "estimate" in result.output


def test_cli_version_flag(cli_runner: CliRunner) -> None:
    """Test that --version flag displays version information."""
    result = cli_runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "0.1.0" in result.output


# ---------------------------------------------------------------------------
# Init Command Tests
# ---------------------------------------------------------------------------


def test_init_help(cli_runner: CliRunner) -> None:
    """Test that init --help displays usage information."""
    result = cli_runner.invoke(main, ["init", "--help"])
    assert result.exit_code == 0
    assert "Create a minimal character config" in result.output
    assert "--character-name" in result.output
    assert "--base-image-path" in result.output
    assert "--action" in result.output
    assert "--non-interactive" in result.output
    assert "--draft-description" in result.output


def test_init_non_interactive_happy_path(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test non-interactive init with repeatable --action inputs."""
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")
    config_path = tmp_path / "generated.yaml"

    result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "test hero",
            "--base-image-path",
            str(base_image),
            "--action",
            "idle|breathing in place|4|120",
            "--action",
            "walk|steady forward walk|6|100",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0
    assert config_path.exists()
    assert "Config created" in result.output

    validate_result = cli_runner.invoke(main, ["validate", str(config_path)])
    assert validate_result.exit_code == 0

    estimate_result = cli_runner.invoke(main, ["estimate", str(config_path)])
    assert estimate_result.exit_code == 0
    assert "Minimum calls:" in estimate_result.output


def test_init_reprompts_invalid_prompt_values(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test interactive init rejects invalid values and eventually succeeds."""
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")
    config_path = tmp_path / "interactive.yaml"

    result = cli_runner.invoke(
        main,
        ["init", str(config_path)],
        input=(
            "\n"  # empty string (invalid character name)
            "hero\n"
            "missing.png\n"  # invalid base image path
            f"{base_image}\n"
            "\n"  # empty string (invalid action name)
            "idle\n"
            "\n"  # empty string (invalid movement description)
            "breathing in place\n"
            "0\n"  # invalid frames
            "4\n"
            "0\n"  # invalid timing
            "120\n"
            "n\n"
        ),
    )
    assert result.exit_code == 0
    assert result.output.count("Character name:") == 2
    assert "Path must point to an existing file" in result.output
    assert config_path.exists()


def test_init_non_interactive_requires_action(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test --non-interactive requires at least one --action."""
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")
    config_path = tmp_path / "generated.yaml"
    result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "hero",
            "--base-image-path",
            str(base_image),
            "--non-interactive",
        ],
    )
    assert result.exit_code != 0
    assert "At least one --action is required with --non-interactive" in result.output


def test_init_force_overwrites_existing_config(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test that --force allows overwriting an existing config file."""
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")
    config_path = tmp_path / "generated.yaml"

    first_result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "hero",
            "--base-image-path",
            str(base_image),
            "--action",
            "idle|breathing in place|4|120",
            "--non-interactive",
        ],
    )
    assert first_result.exit_code == 0
    assert "name: idle" in config_path.read_text()

    second_result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "hero",
            "--base-image-path",
            str(base_image),
            "--action",
            "walk|steady forward walk|6|100",
            "--non-interactive",
        ],
    )
    assert second_result.exit_code != 0
    assert "already exists" in second_result.output
    assert "name: idle" in config_path.read_text()

    third_result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "hero",
            "--base-image-path",
            str(base_image),
            "--action",
            "walk|steady forward walk|6|100",
            "--non-interactive",
            "--force",
        ],
    )
    assert third_result.exit_code == 0
    assert "name: walk" in config_path.read_text()


def test_init_interactive_prompts_for_output_path(cli_runner: CliRunner) -> None:
    """Test interactive init prompts for output path with character-based default."""
    with cli_runner.isolated_filesystem():
        base_image = Path("base.png")
        base_image.write_bytes(b"placeholder")

        result = cli_runner.invoke(
            main,
            ["init"],
            input=(
                "hero knight\n"
                "base.png\n"
                "idle\n"
                "breathing in place\n"
                "4\n"
                "120\n"
                "n\n"
                "\n"
            ),
        )

        assert result.exit_code == 0
        assert "[configs/hero_knight.yaml]" in result.output
        config_path = Path("configs/hero_knight.yaml")
        assert config_path.exists()
        config_text = config_path.read_text()
        assert "name: hero knight" in config_text
        assert "name: idle" in config_text


def test_init_non_interactive_draft_description_success(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test --draft-description writes generated draft text when available."""
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")
    config_path = tmp_path / "generated.yaml"

    async def _mock_draft(base_image_path: Path, character_name: str) -> str:
        return f"{character_name} has a bright cloak and a steel sword."

    monkeypatch.setattr(
        "spriteforge.cli._generate_character_description_draft", _mock_draft
    )

    result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "test hero",
            "--base-image-path",
            str(base_image),
            "--action",
            "idle|breathing in place|4|120",
            "--non-interactive",
            "--draft-description",
        ],
    )
    assert result.exit_code == 0
    assert (
        "description: test hero has a bright cloak and a steel sword."
        in config_path.read_text()
    )


def test_init_non_interactive_draft_description_fallback(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test --draft-description falls back deterministically on failures."""
    base_image = tmp_path / "base.png"
    base_image.write_bytes(b"placeholder")
    config_path = tmp_path / "generated.yaml"

    async def _mock_draft(base_image_path: Path, character_name: str) -> str:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "spriteforge.cli._generate_character_description_draft", _mock_draft
    )

    result = cli_runner.invoke(
        main,
        [
            "init",
            str(config_path),
            "--character-name",
            "test hero",
            "--base-image-path",
            str(base_image),
            "--action",
            "idle|breathing in place|4|120",
            "--non-interactive",
            "--draft-description",
        ],
    )
    assert result.exit_code == 0
    config_text = config_path.read_text()
    assert "Pixel-art character named test hero." in config_text


# ---------------------------------------------------------------------------
# Generate Command Tests
# ---------------------------------------------------------------------------


def test_generate_help(cli_runner: CliRunner) -> None:
    """Test that generate --help displays usage information."""
    result = cli_runner.invoke(main, ["generate", "--help"])
    assert result.exit_code == 0
    assert "Generate a spritesheet" in result.output
    assert "--output" in result.output
    assert "--base-image" in result.output
    assert "--verbose" in result.output
    assert "--json-logs" in result.output
    assert "--run-summary" in result.output


def test_generate_missing_config(cli_runner: CliRunner) -> None:
    """Test that generate fails gracefully when config file doesn't exist."""
    result = cli_runner.invoke(main, ["generate", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert (
        "nonexistent.yaml" in result.output.lower()
        or "not found" in result.output.lower()
    )


def test_generate_requires_base_image(
    cli_runner: CliRunner, test_config_path: Path, tmp_path: Path
) -> None:
    """Test that generate requires base image to be specified."""
    # Create a minimal config without base_image_path
    test_config = tmp_path / "test_config.yaml"
    test_config.write_text("""
character:
  name: "Test"
  class: "Warrior"
  description: "Test character"
  frame_size: [64, 64]
  spritesheet_columns: 10

animations:
  - name: "idle"
    row: 0
    frames: 1
    timing_ms: 100
""")
    result = cli_runner.invoke(main, ["generate", str(test_config)])
    assert result.exit_code != 0
    assert "No base image specified" in result.output


# ---------------------------------------------------------------------------
# Validate Command Tests
# ---------------------------------------------------------------------------


def test_validate_help(cli_runner: CliRunner) -> None:
    """Test that validate --help displays usage information."""
    result = cli_runner.invoke(main, ["validate", "--help"])
    assert result.exit_code == 0
    assert "Validate a character configuration" in result.output
    assert "--no-check-base-image" in result.output


def test_validate_missing_config(cli_runner: CliRunner) -> None:
    """Test that validate fails gracefully when config file doesn't exist."""
    result = cli_runner.invoke(main, ["validate", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert (
        "nonexistent.yaml" in result.output.lower()
        or "not found" in result.output.lower()
    )


def test_validate_valid_config_no_check_base(
    cli_runner: CliRunner, test_config_path: Path
) -> None:
    """Test that validate succeeds with valid config when skipping base image check."""
    if not test_config_path.exists():
        pytest.skip(f"Test config not found: {test_config_path}")

    result = cli_runner.invoke(
        main, ["validate", str(test_config_path), "--no-check-base-image"]
    )
    assert result.exit_code == 0
    assert "✓" in result.output or "valid" in result.output.lower()


def test_validate_valid_config_with_base_check(
    cli_runner: CliRunner, test_config_path: Path
) -> None:
    """Test that validate checks base image existence by default."""
    if not test_config_path.exists():
        pytest.skip(f"Test config not found: {test_config_path}")

    result = cli_runner.invoke(main, ["validate", str(test_config_path)])
    # Should succeed if base image exists in docs_assets/
    base_image = Path("docs_assets/theron_base_reference.png")
    if base_image.exists():
        assert result.exit_code == 0
        assert "✓" in result.output or "valid" in result.output.lower()
    else:
        # If base image doesn't exist, should fail with appropriate error
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()


def test_validate_invalid_yaml(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test that validate fails gracefully with malformed YAML."""
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("invalid: yaml: content:\n  - broken")

    result = cli_runner.invoke(main, ["validate", str(invalid_config)])
    assert result.exit_code != 0


def test_validate_missing_required_section(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test that validate fails when required sections are missing."""
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("character:\n  name: Test\n")

    result = cli_runner.invoke(main, ["validate", str(incomplete_config)])
    assert result.exit_code != 0
    assert "Missing" in result.output or "required" in result.output.lower()


# ---------------------------------------------------------------------------
# Estimate Command Tests
# ---------------------------------------------------------------------------


def test_estimate_help(cli_runner: CliRunner) -> None:
    """Test that estimate --help displays usage information."""
    result = cli_runner.invoke(main, ["estimate", "--help"])
    assert result.exit_code == 0
    assert "Estimate LLM call costs" in result.output


def test_estimate_missing_config(cli_runner: CliRunner) -> None:
    """Test that estimate fails gracefully when config file doesn't exist."""
    result = cli_runner.invoke(main, ["estimate", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert (
        "nonexistent.yaml" in result.output.lower()
        or "not found" in result.output.lower()
    )


def test_estimate_valid_config(cli_runner: CliRunner, test_config_path: Path) -> None:
    """Test that estimate produces call count estimates."""
    if not test_config_path.exists():
        pytest.skip(f"Test config not found: {test_config_path}")

    result = cli_runner.invoke(main, ["estimate", str(test_config_path)])
    assert result.exit_code == 0
    assert "Minimum calls:" in result.output
    assert "Expected calls:" in result.output
    assert "Maximum calls:" in result.output


def test_estimate_shows_breakdown(
    cli_runner: CliRunner, test_config_path: Path
) -> None:
    """Test that estimate shows detailed breakdown of call types."""
    if not test_config_path.exists():
        pytest.skip(f"Test config not found: {test_config_path}")

    result = cli_runner.invoke(main, ["estimate", str(test_config_path)])
    assert result.exit_code == 0
    # Check for breakdown sections
    assert "MINIMUM" in result.output
    assert "EXPECTED" in result.output
    assert "MAXIMUM" in result.output
    # Check for specific call types
    assert "reference_generation" in result.output
    assert "grid_generation" in result.output


def test_estimate_with_budget_config(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test that estimate shows budget check when budget is configured."""
    config_with_budget = tmp_path / "budget_config.yaml"
    config_with_budget.write_text("""
character:
  name: "Test"
  class: "Warrior"
  description: "Test character"
  frame_size: [64, 64]
  spritesheet_columns: 10

animations:
  - name: "idle"
    row: 0
    frames: 3
    timing_ms: 100

generation:
  budget:
    max_llm_calls: 100
    max_retries_per_row: 2
""")
    result = cli_runner.invoke(main, ["estimate", str(config_with_budget)])
    assert result.exit_code == 0
    assert "Budget" in result.output or "budget" in result.output.lower()
    assert "max_llm_calls: 100" in result.output


# ---------------------------------------------------------------------------
# Verbose Flag Tests
# ---------------------------------------------------------------------------


def test_verbose_flag_validate(cli_runner: CliRunner, test_config_path: Path) -> None:
    """Test that --verbose flag is accepted by validate command."""
    if not test_config_path.exists():
        pytest.skip(f"Test config not found: {test_config_path}")

    result = cli_runner.invoke(
        main, ["validate", str(test_config_path), "--no-check-base-image", "--verbose"]
    )
    # Should not error on --verbose flag
    assert result.exit_code == 0


def test_verbose_flag_estimate(cli_runner: CliRunner, test_config_path: Path) -> None:
    """Test that --verbose flag is accepted by estimate command."""
    if not test_config_path.exists():
        pytest.skip(f"Test config not found: {test_config_path}")

    result = cli_runner.invoke(main, ["estimate", str(test_config_path), "--verbose"])
    # Should not error on --verbose flag
    assert result.exit_code == 0
