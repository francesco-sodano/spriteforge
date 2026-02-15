"""Logging configuration for SpriteForge.

Provides a simple setup function and module-level logger factory.
Uses Python's built-in logging module — no external dependencies.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_FORMAT = "%(levelname)-5s | %(name)-12s | %(message)s"
VERBOSE_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)-12s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure logging for the spriteforge package.

    Calling this function sets up handlers on the ``spriteforge`` root
    logger.  Repeated calls are safe — existing handlers are cleared
    first to prevent duplicate output.

    Args:
        level: Logging level (default: INFO).
        verbose: If True, include timestamps in output.
        log_file: Optional file path to write logs to (in addition to stderr).
    """
    logger = logging.getLogger("spriteforge")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on repeated calls
    logger.handlers.clear()

    # Console handler
    fmt = VERBOSE_FORMAT if verbose else DEFAULT_FORMAT
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(VERBOSE_FORMAT))
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a SpriteForge module.

    Args:
        name: Module name (e.g., ``"generator"``, ``"gates"``).

    Returns:
        A logger instance under the ``spriteforge`` namespace.
    """
    return logging.getLogger(f"spriteforge.{name}")
