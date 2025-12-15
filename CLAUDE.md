# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

NBA player prop betting analysis system that fetches real NBA stats and betting odds, runs prediction models, and identifies value plays with positive expected value.

## Commands

```bash
# Install dependencies
pip install nba_api pandas numpy scipy openpyxl requests streamlit

# Run daily analysis (CLI)
python nba_quickstart.py --daily

# Analyze specific prop
python nba_quickstart.py --player "Luka Doncic" --prop points --line 32.5

# Run backtest simulation
python nba_quickstart.py --backtest

# Launch web dashboard
streamlit run app.py

# Run daily automation
python daily_runner.py --pre-game
python daily_runner.py --post-game
```

## Architecture

### Module Structure (Refactored)

```
nbaprop/
├── core/                      # Shared foundations
│   ├── config.py             # Centralized Config dataclass
│   ├── constants.py          # Team maps, star players
│   ├── odds_utils.py         # Odds math, Kelly, EV calculations
│   └── exceptions.py         # Custom error types
│
├── data/                      # Data layer
│   └── __init__.py           # Re-exports: NBADataFetcher, OddsAPIClient, InjuryTracker
│
├── models/                    # Prediction models
│   ├── __init__.py           # Re-exports all models
│   └── base.py               # BaseModel interface, StandardPrediction
│
├── analysis/                  # Analysis orchestration
│   └── __init__.py           # Re-exports: LivePropAnalyzer, Backtester
│
├── app.py                    # Streamlit dashboard
├── nba_quickstart.py         # CLI entry point
├── daily_runner.py           # Daily automation
├── nba_prop_model.py         # Model implementations (legacy location)
└── nba_integrations.py       # Data fetchers (legacy location)
```

### Recommended Imports

```python
# New modular imports (preferred)
from core.config import CONFIG
from core.constants import TEAM_ABBREVIATIONS, normalize_team_abbrev
from core.odds_utils import kelly_criterion, calculate_ev

from data import NBADataFetcher, OddsAPIClient, InjuryTracker
from models import UnifiedPropModel, PropAnalysis
from analysis import LivePropAnalyzer, Backtester

# Legacy imports still work (for backward compatibility)
from nba_prop_model import UnifiedPropModel
from nba_integrations import NBADataFetcher
from nba_quickstart import CONFIG  # Returns dict for backward compat
```

### Entry Points
| File | Purpose |
|------|---------|
| `nba_quickstart.py` | CLI entry point with config |
| `app.py` | Streamlit web dashboard with 3 tabs |
| `daily_runner.py` | Automated pre-game/post-game workflow |

### Core Module (`core/`)

**`core/config.py`** - Centralized configuration:
```python
from core.config import CONFIG

CONFIG.MIN_EDGE_THRESHOLD  # 0.03 (3%)
CONFIG.MIN_CONFIDENCE      # 0.40
CONFIG.KELLY_FRACTION      # 0.25
CONFIG.PREFERRED_BOOKS     # ['pinnacle', 'draftkings', ...]
```

**`core/constants.py`** - Single source of truth for:
- `TEAM_ABBREVIATIONS` - All team name variants
- `STAR_PLAYERS` - Star players by team
- `normalize_team_abbrev()` - Normalize any team reference

**`core/odds_utils.py`** - Betting math:
- `american_to_decimal()`, `american_to_implied_prob()`
- `calculate_ev()`, `calculate_edge()`
- `kelly_criterion()`, `calculate_confidence()`

**`core/exceptions.py`** - Custom exceptions:
- `NBAPropError`, `DataFetchError`, `PlayerNotFoundError`
- `InsufficientDataError`, `OddsAPIError`, `RateLimitError`

### Model Layer (`models/`)

**Primary Model: `UnifiedPropModel`**

```python
from models import UnifiedPropModel

model = UnifiedPropModel()
analysis = model.analyze("Luka Doncic", "points", 32.5)

print(f"Projection: {analysis.projection}")
print(f"Edge: {analysis.edge:.1%}")
print(f"Pick: {analysis.pick}")
print(f"Flags: {analysis.flags}")
```

**9 Adjustment Factors Applied:**
1. **Opponent Defense** - Factor from team defense rankings (SMASH/GOOD/NEUTRAL/HARD/TOUGH)
2. **Home/Away** - ±2.5% adjustment
3. **Back-to-Back** - 8% penalty for B2B games
4. **Pace** - Combined pace factor from both teams
5. **Game Total** - Vegas over/under impact on scoring props
6. **Blowout Risk** - Minutes reduction in blowouts (from spread)
7. **vs Team History** - Historical performance against opponent (capped ±10%)
8. **Minutes Trend** - Recent minutes trends (capped ±10%)
9. **Injury Boost** - Usage increase when teammates are out (capped +15%)

**Available Models:**
- `UnifiedPropModel` - Production model with 9 contextual adjustments
- `WeightedAverageModel` - Recency-weighted last 10 games
- `MedianModel` - Outlier-resistant median
- `SituationalModel` - Basic home/away and B2B adjustments
- `EnsembleModel` - Combines weighted average + median

### Data Layer (`data/`)

**Data Fetchers:**
- `NBADataFetcher` - Player game logs, team defense/pace ratings via `nba_api`
- `OddsAPIClient` - Live betting lines from The Odds API
- `InjuryTracker` - Player injuries, teammate usage boosts when stars are out

**Usage:**
```python
from data import NBADataFetcher, OddsAPIClient, InjuryTracker

fetcher = NBADataFetcher()
logs = fetcher.get_player_game_logs("Luka Doncic", last_n_games=15)

odds = OddsAPIClient(api_key="YOUR_KEY")
props = odds.get_player_props(event_id, markets=["player_points"])

injuries = InjuryTracker()
status = injuries.get_player_status("LeBron James")
```

### Analysis Layer (`analysis/`)

```python
from analysis import LivePropAnalyzer, Backtester

analyzer = LivePropAnalyzer()
picks = analyzer.find_value_props(min_edge=0.05, min_confidence=0.4)

backtester = Backtester(initial_bankroll=1000)
results = backtester.run_backtest(props_df, game_logs, model)
```

### Data Flow

```
User Request (CLI/Streamlit/API)
         ↓
    UnifiedPropModel.analyze()
         ↓
┌────────────────────────────────────────────┐
│  1. Fetch player game logs (last 15)       │
│  2. Auto-detect context:                   │
│     - Player team from matchup string      │
│     - B2B status from game dates           │
│  3. Load cached context (1hr TTL):         │
│     - Team defense vs position ratings     │
│     - Team pace factors                    │
│  4. Check injuries:                        │
│     - Player status (GTD/OUT/etc)          │
│     - Teammate boosts (stars out)          │
│  5. Calculate base projection:             │
│     - 60% recent (L5) + 40% older (6-15)   │
│     - Apply trend adjustment (±10%)        │
│  6. Apply 9 multiplicative adjustments     │
│  7. Calculate multi-factor confidence      │
│  8. Determine edge and pick                │
└────────────────────────────────────────────┘
         ↓
    PropAnalysis Result
```

## API Rate Limits

- **NBA Stats API**: 1 req/sec (1.0s base delay with jitter built in)
- **The Odds API**: 20,000 requests/month (paid subscription)
- **BallDontLie API**: Requires paid tier for stats endpoints

## Configuration

Configuration is centralized in `core/config.py`. Access via:

```python
from core.config import CONFIG

# Analysis settings
CONFIG.MIN_EDGE_THRESHOLD  # 0.03 (3% minimum edge)
CONFIG.MIN_CONFIDENCE      # 0.40 (40% minimum confidence)
CONFIG.LOOKBACK_GAMES      # 15 games to analyze

# Bankroll settings
CONFIG.KELLY_FRACTION      # 0.25 (quarter Kelly)
CONFIG.MAX_BET_PERCENT     # 0.03 (3% max bet)

# API settings
CONFIG.ODDS_API_KEY        # Set via environment variable
CONFIG.NBA_API_DELAY       # 1.0 second between calls
```

For backward compatibility, `nba_quickstart.py` exports a dict-style CONFIG.

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `UnifiedPropModel` | `models` | Primary prediction model with all context |
| `PropAnalysis` | `models` | Dataclass with full analysis results |
| `LivePropAnalyzer` | `analysis` | Batch analysis with odds integration |
| `NBADataFetcher` | `data` | NBA stats data fetching |
| `OddsAPIClient` | `data` | Betting odds fetching |
| `InjuryTracker` | `data` | Injury tracking and usage boosts |
| `Backtester` | `analysis` | Historical betting simulation |
| `Config` | `core.config` | Centralized configuration |
