"""CLI entry points for the rebuild."""

from typing import Optional


def run_daily(config_path: Optional[str] = None) -> int:
    """Run the daily pipeline end-to-end."""
    raise NotImplementedError("Daily pipeline not implemented yet.")


def run_backtest(config_path: Optional[str] = None) -> int:
    """Run the backtest pipeline end-to-end."""
    raise NotImplementedError("Backtest pipeline not implemented yet.")
