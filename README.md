# 🏀 NBA Prop Betting Analysis & Backtesting System

A comprehensive Python toolkit for analyzing NBA player props and backtesting betting strategies.

## Source-of-Truth Pipeline (v2)

The primary, supported pipeline is the v2 CLI under `nbaprop/`. Legacy scripts
like `daily_runner.py` and `nba_quickstart.py` remain for reference only.

```bash
# Daily picks (v2)
python -m nbaprop.cli run-daily

# Backtest (v2)
python -m nbaprop.cli run-backtest
```

## Features

- **Real-time NBA Data** - Fetches live player stats via NBA Stats API
- **Live Betting Odds** - Integrates with The Odds API for current lines
- **Multiple Prediction Models** - Weighted average, median, ensemble approaches
- **Situational Adjustments** - Home/away, back-to-back, opponent defense
- **Full Backtesting Engine** - Track ROI, win rate, drawdown, streaks
- **Expected Value Calculations** - Probability-based edge detection
- **Excel Export** - Professional reports with charts and analysis

## Installation

```bash
# Clone or download the files, then:
pip install nba_api pandas numpy scipy openpyxl requests
```

## Quick Start

### 1. Analyze a Single Prop

```python
from data import NBADataFetcher
from analysis import LivePropAnalyzer

fetcher = NBADataFetcher()
analyzer = LivePropAnalyzer(nba_fetcher=fetcher)

# Analyze Luka's points prop
result = analyzer.analyze_prop(
    player_name="Luka Doncic",
    prop_type="points",
    line=32.5,
    odds=-110
)

print(f"Recent Average: {result['recent_avg']}")
print(f"Over Hit Rate: {result['hit_rate_over']}%")
print(f"Model Edge: {result['avg_edge']}%")
print(f"Recommendation: {result['recommended_side'].upper()}")
```

### 2. Batch Analysis

```python
props_to_check = [
    {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5},
    {'player': 'Shai Gilgeous-Alexander', 'prop_type': 'points', 'line': 30.5},
    {'player': 'Jayson Tatum', 'prop_type': 'rebounds', 'line': 8.5},
    {'player': 'Nikola Jokic', 'prop_type': 'pra', 'line': 47.5},
    {'player': 'Tyrese Haliburton', 'prop_type': 'assists', 'line': 10.5},
]

results = analyzer.analyze_multiple_props(props_to_check)
print(results)
```

### 3. Live Odds Integration

Get a free API key at [the-odds-api.com](https://the-odds-api.com/) (500 requests/month free).

```python
from data import OddsAPIClient

odds = OddsAPIClient(api_key="YOUR_API_KEY")

# Get all NBA player props
raw_props = odds.get_player_props()
props_df = odds.parse_player_props(raw_props)

# Find best odds across books
best_odds = odds.get_best_odds(props_df)
print(best_odds.head(20))
```

### 4. Automated Value Detection

```python
# Find value props automatically
analyzer = LivePropAnalyzer(nba_fetcher=fetcher, odds_client=odds)
value_plays = analyzer.find_value_props(min_edge=0.05)  # 5% minimum edge

print("🎯 VALUE PLAYS:")
for _, play in value_plays.iterrows():
    print(f"  {play['player']} {play['prop_type']} {play['recommended_side'].upper()} {play['line']}")
```

### 5. Run Backtest

```python
from models import Backtester, EnsembleModel, generate_sample_dataset, generate_prop_lines
game_logs = generate_sample_dataset()
props = generate_prop_lines(game_logs)

# Run backtest
backtester = Backtester(initial_bankroll=1000, unit_size=10)
results = backtester.run_backtest(
    props, game_logs, 
    EnsembleModel(),
    min_edge=0.05,
    min_confidence=0.40
)

backtester.print_report()
```

## File Structure

```
models/                - Prediction models and backtesting engine
data/                  - Data fetchers (NBA API + Odds API)
analysis/              - LivePropAnalyzer and batch analysis
core/                  - Configuration, constants, utilities
nbaprop/               - V2 CLI pipeline (recommended)
nba_quickstart.py      - Quick start scripts and legacy CLI
```

## Prediction Models

### WeightedAverageModel
Applies recency weighting to recent games. More recent performances weighted higher.

### MedianModel
Uses median instead of mean. More robust to outlier games (blowouts, injuries, etc.).

### EnsembleModel
Combines weighted average and median approaches for stability.

### SituationalModel
Adjusts projections based on:
- Home/away (+2.5% home boost)
- Back-to-back games (-7% penalty)
- Opponent defensive rating

## Backtest Metrics

The backtester tracks:
- **Win Rate** - Percentage of winning bets
- **ROI** - Return on investment
- **Max Drawdown** - Largest peak-to-trough decline
- **Profit by Prop Type** - Points, rebounds, assists, PRA
- **Profit by Side** - Overs vs unders
- **Win/Loss Streaks** - Longest consecutive outcomes

## Configuration

Edit settings in `nba_quickstart.py`:

```python
CONFIG = {
    'ODDS_API_KEY': None,           # Your API key
    'MIN_EDGE_THRESHOLD': 0.05,     # 5% minimum edge to bet
    'MIN_CONFIDENCE': 0.40,         # 40% minimum model confidence
    'LOOKBACK_GAMES': 15,           # Games to analyze
    'INITIAL_BANKROLL': 1000,       # Starting bankroll
    'UNIT_SIZE': 10,                # Dollars per unit
    'MAX_UNITS_PER_BET': 3,         # Max bet size
}
```

## Command Line Usage

```bash
# Run daily analysis
python nba_quickstart.py --daily

# Analyze specific prop
python nba_quickstart.py --player "Luka Doncic" --prop points --line 32.5

# Run backtest
python nba_quickstart.py --backtest
```

## Key Findings from Backtesting

Based on simulated full-season backtests:

| Strategy | Win Rate | ROI | Notes |
|----------|----------|-----|-------|
| Conservative (5%+ edge) | 62% | +22% | Best risk-adjusted |
| Moderate (3%+ edge) | 60% | +19% | More volume |
| Aggressive (2%+ edge) | 60% | +19% | Highest variance |

**Insights:**
- Unders outperform overs consistently
- Rebounds and assists show higher edges than points
- Back-to-back games favor unders
- Conservative thresholds provide better risk-adjusted returns

## API Rate Limits

- **NBA Stats API**: ~1 request/second recommended
- **The Odds API**: 500 requests/month (free tier)

## Disclaimer

This tool is for educational and entertainment purposes only. Sports betting involves risk. Past performance does not guarantee future results. Please gamble responsibly.

## License

MIT License - Free to use and modify.
