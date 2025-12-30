"""Logging setup."""

import logging
import os
from typing import Optional


def configure_logging(run_id: Optional[str] = None) -> None:
    """Configure structured logging for runs."""
    level_name = os.environ.get("NBAPROP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    run_prefix = f"[run_id={run_id}] " if run_id else ""
    fmt = "%(asctime)s %(levelname)s " + run_prefix + "%(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)
