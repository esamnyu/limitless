#!/usr/bin/env python3
"""
LOG SETUP — Centralized logging configuration for Weather Edge.

Usage in any module:
    from log_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Scanning NYC...")
    logger.warning("NWS timeout")
    logger.error("Order failed", exc_info=True)

Outputs:
  - Console: INFO+ (colored, concise) — for interactive use
  - File: DEBUG+ to logs/weather_edge.log — for cron/debugging
  - Rotates at 5 MB, keeps 5 backups
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "weather_edge.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5

__all__ = ["setup_logging", "get_logger"]

_initialized = False


def setup_logging(level: int = logging.DEBUG):
    """
    Configure root logger with console + file handlers.

    Safe to call multiple times — only initializes once.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler — INFO+ (user-facing output)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(console)

    # File handler — DEBUG+ (full detail for debugging)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with auto-initialization.

    First call triggers setup_logging(). Subsequent calls
    just return the named logger.
    """
    setup_logging()
    return logging.getLogger(name)
