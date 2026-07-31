"""
Logging configuration for the Indian Immigration Legal Assistant.

Uses structlog for structured logging with two output modes:
- CONSOLE: Human-readable rich output for development
- JSON: Machine-parseable structured logs for production log aggregators

Usage:
    from core.logging import get_logger

    log = get_logger(__name__)
    log.info("event_name", key="value", count=42)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

__all__ = ["configure_logging", "get_logger"]


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------


def _add_app_context(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject application-level context into every log record."""
    event_dict.setdefault("app", "indian-immigration-legal-assistant")
    return event_dict


def _drop_color_message_key(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Remove uvicorn's 'color_message' key which clutters JSON output.
    Uvicorn injects this when logging access logs.
    """
    event_dict.pop("color_message", None)
    return event_dict


# ---------------------------------------------------------------------------
# Main configuration function
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    fmt: str = "console",
    log_file: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure structlog + stdlib logging for the entire application.

    Args:
        level:        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt:          Output format – 'console' (rich dev output) or 'json'.
        log_file:     Optional path to a rotating log file. Logs to stdout only
                      if None.
        max_bytes:    Maximum bytes per log file before rotation.
        backup_count: Number of rotated log files to keep.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # ------------------------------------------------------------------
    # 1. Shared processors applied to every log event
    # ------------------------------------------------------------------
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_app_context,
        _drop_color_message_key,
    ]

    # ------------------------------------------------------------------
    # 2. Configure structlog
    # ------------------------------------------------------------------
    if fmt == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
        # In JSON mode prepend ExceptionRenderer so tracebacks are inlined
        final_processors: list[Processor] = [
            *shared_processors,
            structlog.processors.ExceptionRenderer(),
            structlog.processors.dict_tracebacks,
            renderer,
        ]
    else:
        # Console: pretty colours + indented tracebacks
        final_processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=final_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ------------------------------------------------------------------
    # 3. Configure stdlib logging so third-party libraries (uvicorn,
    #    httpx, langchain, etc.) also flow through structlog
    # ------------------------------------------------------------------
    stdlib_processors: list[Processor] = [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    if fmt == "json":
        stdlib_renderer: Processor = structlog.processors.JSONRenderer()
    else:
        stdlib_renderer = structlog.dev.ConsoleRenderer(colors=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            stdlib_renderer,
        ],
    )

    # ------------------------------------------------------------------
    # 4. Set up handlers
    # ------------------------------------------------------------------
    handlers: list[logging.Handler] = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    handlers.append(stdout_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # ------------------------------------------------------------------
    # 5. Apply to root logger
    # ------------------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress noisy third-party loggers
    _noisy_loggers = [
        "httpx",
        "httpcore",
        "openai._base_client",
        "urllib3.connectionpool",
        "qdrant_client",
        "cohere",
        "filelock",
    ]
    for logger_name in _noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to the given name.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A bound structlog logger.

    Example:
        log = get_logger(__name__)
        log.info("document_ingested", doc_id="abc123", chunks=42)
    """
    return structlog.get_logger(name)
