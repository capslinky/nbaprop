"""
Analysis module - Orchestration layer for prop analysis.

This module provides:
- LivePropAnalyzer: Main analysis engine combining models, odds, and data
- Backtester: Historical backtesting with bankroll simulation

Usage:
    from analysis import LivePropAnalyzer

    analyzer = LivePropAnalyzer()

    # Analyze a single prop
    result = analyzer.analyze_prop("Luka Doncic", "points", 32.5)

    # Find all value props
    picks = analyzer.find_value_props(min_edge=0.05, min_confidence=0.4)

    # Analyze multiple props
    props = [
        {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5},
        {'player': 'Jayson Tatum', 'prop_type': 'rebounds', 'line': 8.5},
    ]
    results = analyzer.analyze_multiple_props(props)
"""

# Re-export from nba_integrations.py
from nba_integrations import LivePropAnalyzer

# Re-export Backtester from models
from models import Backtester, BetResult

# Import core utilities for convenience
from core.config import CONFIG
from core.odds_utils import kelly_criterion, calculate_ev

__all__ = [
    # Main analysis classes
    'LivePropAnalyzer',
    'Backtester',
    'BetResult',

    # Utilities
    'CONFIG',
    'kelly_criterion',
    'calculate_ev',
]
