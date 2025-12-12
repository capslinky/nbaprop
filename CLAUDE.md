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

# Run v2 analysis (Bayesian models, Kelly sizing)
python nba_props_v2.py --daily
python nba_props_v2.py --validate
```

## Architecture

### Entry Points
| File | Purpose |
|------|---------|
| `nba_quickstart.py` | CLI entry point with config (API keys, thresholds) |
| `app.py` | Streamlit web dashboard with 3 tabs: Find Best Plays, Single Game, Player Lookup |
| `nba_props_v2.py` | V2 analysis with Bayesian models, Kelly sizing, CLV tracking (separate research tool) |

### Model Layer (`nba_prop_model.py`)

**Primary Model: `UnifiedPropModel`**

The unified model combines all contextual factors into a single, consistent analysis:

```python
from nba_prop_model import UnifiedPropModel

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

**Legacy Models (for backtesting):**
- `WeightedAverageModel` - Recency-weighted last 10 games
- `MedianModel` - Outlier-resistant median
- `SituationalModel` - Basic home/away and B2B adjustments
- `EnsembleModel` - Combines weighted average + median

### Data Layer (`nba_integrations.py`)

**Data Fetchers:**
- `NBADataFetcher` - Player game logs, team defense/pace ratings via `nba_api`
- `BallDontLieFetcher` - Fallback data source when nba_api fails
- `HybridFetcher` - Orchestrates sources with automatic failover
- `ResilientFetcher` - Retry logic with exponential backoff

**Odds & Context:**
- `OddsAPIClient` - Live betting lines from The Odds API
- `InjuryTracker` - Player injuries, teammate usage boosts when stars are out

**Integration:**
- `LivePropAnalyzer` - Combines all data sources with `UnifiedPropModel` for live analysis

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

Edit `CONFIG` dict in `nba_quickstart.py`:
- `ODDS_API_KEY`: Your The Odds API key
- `MIN_EDGE_THRESHOLD`: Minimum edge to bet (default 0.05 = 5%)
- `MIN_CONFIDENCE`: Minimum confidence (default 0.40)
- `LOOKBACK_GAMES`: Games to analyze (default 15)

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `UnifiedPropModel` | `nba_prop_model.py` | Primary prediction model with all context |
| `PropAnalysis` | `nba_prop_model.py` | Dataclass with full analysis results |
| `LivePropAnalyzer` | `nba_integrations.py` | Batch analysis with odds integration |
| `NBADataFetcher` | `nba_integrations.py` | NBA stats data fetching |
| `OddsAPIClient` | `nba_integrations.py` | Betting odds fetching |
| `InjuryTracker` | `nba_integrations.py` | Injury tracking and usage boosts |
| `Backtester` | `nba_prop_model.py` | Historical betting simulation |
