# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

NBA player prop betting analysis system that fetches real NBA stats and betting odds, runs prediction models, and identifies value plays with positive expected value.

## Commands

```bash
# Install dependencies
pip install nba_api pandas numpy scipy openpyxl requests

# Run daily analysis (recommended CLI)
python -m nbaprop.cli run-daily

# Run with custom config
python -m nbaprop.cli run-daily --config config.json

# Legacy entry points (still functional)
python nba_quickstart.py --daily
python daily_runner.py --pre-game
streamlit run app.py  # Web dashboard
```

## Architecture

The codebase has two parallel architectures. **Use `nbaprop/` for new development.**

### Primary Architecture: `nbaprop/`

```
nbaprop/
├── cli.py                    # Main CLI entry point
├── config.py                 # Centralized Config dataclass
├── constants.py              # Team maps, star players, stat normalization
├── exceptions.py             # Custom error types
├── utils/
│   ├── __init__.py
│   └── odds.py               # Odds math, Kelly, EV calculations
├── ingestion/                # Data fetching layer
│   ├── nba_stats.py          # Player logs, team stats (bridges to legacy)
│   ├── odds.py               # Betting odds (bridges to legacy)
│   ├── injuries.py           # Injury tracking (bridges to legacy)
│   └── rosters.py            # Team rosters
├── features/
│   └── pipeline.py           # Feature engineering (13 adjustments)
├── models/
│   ├── baseline.py           # Baseline model
│   └── scoring.py            # Prop scoring with adjustments
├── normalization/
│   ├── schema.py             # Data schemas
│   └── normalize.py          # Data normalization
├── reporting/
│   ├── csv_output.py         # CSV export
│   └── pdf_output.py         # PDF report generation
├── backtest/
│   └── runner.py             # Backtesting framework
├── ops/
│   ├── metrics.py            # Performance metrics
│   └── __init__.py           # Rate limiting utilities
└── storage/
    └── cache.py              # Caching layer
```

### Legacy Architecture (for reference)

```
core/                         # Legacy shared foundations
├── config.py                 # Legacy Config (UPPERCASE fields)
├── constants.py              # Team maps (now in nbaprop/constants.py)
├── odds_utils.py             # Odds math (now in nbaprop/utils/odds.py)
├── exceptions.py             # Errors (now in nbaprop/exceptions.py)
└── ...

models/                       # Legacy prediction models
├── unified.py                # UnifiedPropModel (legacy)
└── ...

data/                         # Legacy data layer
└── fetchers/                 # Data fetcher implementations
    ├── nba_fetcher.py        # NBADataFetcher
    ├── odds_fetcher.py       # OddsAPIClient
    └── injury_tracker.py     # InjuryTracker
```

## Recommended Imports

```python
# New architecture (preferred)
from nbaprop.config import Config
from nbaprop.constants import TEAM_ABBREVIATIONS, normalize_team_abbrev
from nbaprop.exceptions import NBAPropError, PlayerNotFoundError
from nbaprop.utils.odds import kelly_criterion, calculate_ev, american_to_decimal

# Legacy imports (still work, but prefer nbaprop)
from core.config import CONFIG
from core.constants import normalize_team_abbrev
from data import NBADataFetcher, OddsAPIClient
from models import UnifiedPropModel
```

## Configuration

Configuration is in `nbaprop/config.py`. Load with environment variables or JSON:

```python
from nbaprop.config import Config

# Config loads from environment variables by default
# Key fields (lowercase in nbaprop, UPPERCASE in legacy):
config = Config(...)  # Loaded via nbaprop.cli

# Adjustment factors (13 total):
config.home_boost          # 1.025 (+2.5%)
config.away_penalty        # 0.975 (-2.5%)
config.b2b_penalty         # 0.92 (-8%)
config.min_edge_threshold  # 0.03 (3%)
config.min_confidence      # 0.40 (40%)
```

### Adjustment Factors Applied (13 total):

1. **Home/Away** - ±2.5% adjustment
2. **Back-to-Back** - 8% penalty for B2B games
3. **Rest Days** - Factor based on days since last game
4. **Opponent Defense** - Factor from team defense rankings
5. **Pace** - Combined pace factor from both teams
6. **Game Total** - Vegas O/U impact on scoring props
7. **Blowout Risk** - Minutes reduction from large spreads
8. **vs Team History** - Historical performance vs opponent (±10%)
9. **Minutes Trend** - Recent minutes trends (±10%)
10. **Injury Boost** - Usage increase when teammates out (+15%)
11. **True Shooting Regression** - TS% regression toward mean
12. **Usage Rate** - Shot volume adjustments
13. **Trend** - Hot/cold streak adjustments

## Data Flow

```
User Request (CLI)
         ↓
    nbaprop.cli.run_daily()
         ↓
┌────────────────────────────────────────────┐
│  1. Fetch odds from The Odds API           │
│  2. Build player list from props           │
│  3. Fetch player game logs (last 15)       │
│  4. Build features via pipeline.py:        │
│     - Calculate averages, std dev          │
│     - Compute true shooting, vs team       │
│     - Apply injury context                 │
│  5. Score props via scoring.py:            │
│     - Apply 13 adjustment factors          │
│     - Calculate edge and confidence        │
│  6. Filter by thresholds                   │
│  7. Generate CSV/PDF reports               │
└────────────────────────────────────────────┘
         ↓
    Daily picks output
```

## API Rate Limits

- **NBA Stats API**: 1 req/sec (1.0s base delay with jitter)
- **The Odds API**: 20,000 requests/month (paid subscription)

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_scoring.py -v

# Run with coverage
python -m pytest tests/ --cov=nbaprop
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `Config` | `nbaprop.config` | Centralized configuration |
| `score_prop()` | `nbaprop.models.scoring` | Prop scoring with 13 adjustments |
| `build_features()` | `nbaprop.features.pipeline` | Feature engineering |
| `CacheStore` | `nbaprop.storage` | Caching layer |
| `UnifiedPropModel` | `models.unified` | Legacy production model |
| `NBADataFetcher` | `data.fetchers` | NBA stats fetching |
| `OddsAPIClient` | `data.fetchers` | Betting odds fetching |

## Migration Notes

The codebase is being migrated from the legacy architecture (`core/`, `models/`, `data/`) to the new `nbaprop/` package. Key differences:

- **Config**: Legacy uses `core.config.CONFIG` (UPPERCASE), new uses `nbaprop.config.Config` (lowercase)
- **Constants**: Migrated to `nbaprop.constants` with same functions
- **Odds Utils**: Migrated to `nbaprop.utils.odds` with same functions
- **Exceptions**: Migrated to `nbaprop.exceptions` with same classes
- **Ingestion**: `nbaprop.ingestion.*` modules bridge to legacy fetchers

New code should use `nbaprop.*` imports. Legacy imports remain functional for backward compatibility.
