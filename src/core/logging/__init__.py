"""
Logging configuration entrypoint.

Imports:
    from core.logging import configure_logging, get_logger
"""

from core.logging.logger import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
