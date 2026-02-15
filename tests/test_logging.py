"""Tests for spriteforge.logging module."""

from __future__ import annotations

import logging
import os
import tempfile

from spriteforge.logging import (
    DEFAULT_FORMAT,
    VERBOSE_FORMAT,
    get_logger,
    setup_logging,
)


class TestGetLogger:
    """Tests for the get_logger factory function."""

    def test_returns_correct_namespace(self) -> None:
        """get_logger('gates') returns a logger named 'spriteforge.gates'."""
        lg = get_logger("gates")
        assert lg.name == "spriteforge.gates"

    def test_returns_correct_namespace_generator(self) -> None:
        """get_logger('generator') returns 'spriteforge.generator'."""
        lg = get_logger("generator")
        assert lg.name == "spriteforge.generator"

    def test_returns_logger_instance(self) -> None:
        """get_logger returns a logging.Logger instance."""
        lg = get_logger("test")
        assert isinstance(lg, logging.Logger)

    def test_child_of_spriteforge(self) -> None:
        """Returned logger is a child of the 'spriteforge' root logger."""
        # Ensure the parent logger exists
        _parent = logging.getLogger("spriteforge")
        lg = get_logger("workflow")
        assert lg.parent is not None
        assert lg.parent.name == "spriteforge"


class TestSetupLogging:
    """Tests for the setup_logging configuration function."""

    def _cleanup_spriteforge_logger(self) -> None:
        """Remove all handlers from the spriteforge logger."""
        logger = logging.getLogger("spriteforge")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def setup_method(self) -> None:
        """Clean up before each test."""
        self._cleanup_spriteforge_logger()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self._cleanup_spriteforge_logger()

    def test_sets_level_debug(self) -> None:
        """After setup_logging(level=DEBUG), root logger level is DEBUG."""
        setup_logging(level=logging.DEBUG)
        logger = logging.getLogger("spriteforge")
        assert logger.level == logging.DEBUG

    def test_default_info_level(self) -> None:
        """Default level is INFO."""
        setup_logging()
        logger = logging.getLogger("spriteforge")
        assert logger.level == logging.INFO

    def test_sets_custom_level(self) -> None:
        """setup_logging respects a custom level like WARNING."""
        setup_logging(level=logging.WARNING)
        logger = logging.getLogger("spriteforge")
        assert logger.level == logging.WARNING

    def test_verbose_format_includes_timestamp(self) -> None:
        """Verbose mode uses a format that includes asctime."""
        setup_logging(verbose=True)
        logger = logging.getLogger("spriteforge")
        handler = logger.handlers[0]
        fmt = handler.formatter
        assert fmt is not None
        assert "asctime" in fmt._fmt

    def test_default_format_no_timestamp(self) -> None:
        """Default (non-verbose) format does not include asctime."""
        setup_logging(verbose=False)
        logger = logging.getLogger("spriteforge")
        handler = logger.handlers[0]
        fmt = handler.formatter
        assert fmt is not None
        assert "asctime" not in fmt._fmt

    def test_file_handler_added(self) -> None:
        """log_file parameter adds a FileHandler."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            setup_logging(log_file=log_path)
            logger = logging.getLogger("spriteforge")
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1
            assert file_handlers[0].baseFilename == os.path.abspath(log_path)
        finally:
            # Close handlers before deleting file
            for h in logging.getLogger("spriteforge").handlers:
                if isinstance(h, logging.FileHandler):
                    h.close()
            os.unlink(log_path)

    def test_file_handler_uses_verbose_format(self) -> None:
        """File handler always uses verbose format with timestamps."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            setup_logging(verbose=False, log_file=log_path)
            logger = logging.getLogger("spriteforge")
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1
            fmt = file_handlers[0].formatter
            assert fmt is not None
            assert "asctime" in fmt._fmt
        finally:
            for h in logging.getLogger("spriteforge").handlers:
                if isinstance(h, logging.FileHandler):
                    h.close()
            os.unlink(log_path)

    def test_no_setup_means_silent(self) -> None:
        """Without calling setup_logging(), no handlers are configured."""
        # Start fresh
        logger = logging.getLogger("spriteforge")
        logger.handlers.clear()
        # Don't call setup_logging
        assert len(logger.handlers) == 0

    def test_multiple_setup_calls_idempotent(self) -> None:
        """Calling setup_logging() twice doesn't duplicate handlers."""
        setup_logging()
        setup_logging()
        logger = logging.getLogger("spriteforge")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_multiple_setup_updates_level(self) -> None:
        """Calling setup_logging() again updates the level."""
        setup_logging(level=logging.DEBUG)
        setup_logging(level=logging.WARNING)
        logger = logging.getLogger("spriteforge")
        assert logger.level == logging.WARNING

    def test_console_handler_writes_to_stderr(self) -> None:
        """Console handler targets stderr."""
        import sys

        setup_logging()
        logger = logging.getLogger("spriteforge")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].stream is sys.stderr


class TestFormatStrings:
    """Tests for the format string constants."""

    def test_default_format_has_level_and_name(self) -> None:
        """DEFAULT_FORMAT includes levelname and name fields."""
        assert "%(levelname)" in DEFAULT_FORMAT
        assert "%(name)" in DEFAULT_FORMAT
        assert "%(message)" in DEFAULT_FORMAT

    def test_verbose_format_has_timestamp(self) -> None:
        """VERBOSE_FORMAT includes asctime field."""
        assert "%(asctime)" in VERBOSE_FORMAT
        assert "%(levelname)" in VERBOSE_FORMAT
        assert "%(name)" in VERBOSE_FORMAT
        assert "%(message)" in VERBOSE_FORMAT
