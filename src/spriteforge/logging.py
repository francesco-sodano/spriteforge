"""Logging configuration for SpriteForge.

Provides a simple setup function and module-level logger factory.
Uses Python's built-in logging module — no external dependencies.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

DEFAULT_FORMAT = "%(levelname)-5s | %(name)-12s | %(message)s"
VERBOSE_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)-12s | %(message)s"
_SETUP_LOCK = threading.Lock()


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
    with _SETUP_LOCK:
        logger = logging.getLogger("spriteforge")
        logger.setLevel(level)

        # Console handler
        fmt = VERBOSE_FORMAT if verbose else DEFAULT_FORMAT
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and getattr(h, "stream", None) is sys.stderr
        ]
        if stream_handlers:
            stream_handler = stream_handlers[0]
            for extra in stream_handlers[1:]:
                logger.removeHandler(extra)
        else:
            stream_handler = logging.StreamHandler(sys.stderr)
            logger.addHandler(stream_handler)
        stream_handler.setFormatter(logging.Formatter(fmt))

        # Optional file handler
        if log_file:
            target = os.path.abspath(str(log_file))
            file_handlers = [
                h
                for h in logger.handlers
                if isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", None) == target
            ]
            if file_handlers:
                file_handler = file_handlers[0]
            else:
                file_handler = logging.FileHandler(log_file)
                logger.addHandler(file_handler)
            file_handler.setFormatter(logging.Formatter(VERBOSE_FORMAT))


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a SpriteForge module.

    Args:
        name: Module name (e.g., ``"generator"``, ``"gates"``).

    Returns:
        A logger instance under the ``spriteforge`` namespace.
    """
    return logging.getLogger(f"spriteforge.{name}")
