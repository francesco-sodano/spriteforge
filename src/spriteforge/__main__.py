"""SpriteForge command-line entry point."""

from __future__ import annotations


def main() -> None:
    """CLI entry point delegating to the canonical Click-based CLI."""
    from spriteforge.cli import main as click_main

    click_main()


if __name__ == "__main__":
    main()
