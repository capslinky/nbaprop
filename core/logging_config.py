"""Centralized logging configuration for NBA prop analysis."""
import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        datefmt='%H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
