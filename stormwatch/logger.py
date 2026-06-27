"""
StormWatch AI - Logging Module
Structured logging with rich console output and file rotation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_tracebacks

from stormwatch.config import get_config

# Only install rich tracebacks when not in CI (they can hang in non-TTY envs)
_IN_CI = "CI" in __import__("os").environ

if not _IN_CI:
    install_rich_tracebacks(show_locals=True)

_CONSOLE = Console() if not _IN_CI else None
_LOGGERS: dict[str, logging.Logger] = {}


def _make_ci_handler(level: int) -> logging.Handler:
    """Return a plain stderr handler for CI environments."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%m/%d/%y %H:%M:%S",
        )
    )
    return handler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger with the given name.

    If ``name`` is ``None``, the caller's module name is used.
    """
    if name is None:
        import inspect

        frame = inspect.currentframe()
        name = frame.f_back.f_globals["__name__"] if frame else "stormwatch"

    if name in _LOGGERS:
        return _LOGGERS[name]

    config = get_config()
    level = getattr(logging, config.logging.level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    if _IN_CI:
        # Plain stderr logging in CI — Rich markup is invisible there
        logger.addHandler(_make_ci_handler(level))
    else:
        # Rich console handler for local development
        rich_handler = RichHandler(
            console=_CONSOLE,
            show_time=True,
            show_path=True,
            show_level=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(level)
        logger.addHandler(rich_handler)

        # File handler (optional — logs to stormwatch.log)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "stormwatch.log", mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(config.logging.format))
        logger.addHandler(file_handler)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def console() -> Console:
    """Return the shared Rich console instance."""
    return _CONSOLE
