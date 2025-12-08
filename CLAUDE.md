# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA player prop betting analysis system that fetches real NBA stats and betting odds, runs prediction models, and backtests betting strategies.

## Commands

```bash
# Install dependencies
pip install nba_api pandas numpy scipy openpyxl requests

# Run daily analysis
python nba_quickstart.py --daily

# Analyze specific prop
python nba_quickstart.py --player "Luka Doncic" --prop points --line 32.5

# Run backtest simulation
python nba_quickstart.py --backtest
```

## Architecture

### Data Layer (`nba_integrations.py`)
- **NBADataFetcher**: Fetches player game logs, team defense ratings via `nba_api` (rate-limited 0.6s)
- **OddsAPIClient**: Retrieves live betting odds from The Odds API (requires API key, 500 req/month free)
- **LivePropAnalyzer**: Combines fetchers with models to analyze props, calculate edge/confidence, recommend sides

### Model Layer (`nba_prop_model.py`)
Four prediction models with common interface (`predict(history, line) -> Prediction`):
- **WeightedAverageModel**: Recency-weighted last 10 games
- **MedianModel**: Outlier-resistant median calculation
- **SituationalModel**: Adjusts for home/away (±2.5%), back-to-back (-7%), opponent defense
- **EnsembleModel**: Combines weighted average + median (50/50)

**Backtester class**: Runs betting simulations tracking win rate, ROI, drawdown, profit by prop type/side, streaks.

### Entry Point (`nba_quickstart.py`)
CLI interface and configuration. CONFIG dict contains API keys, thresholds (MIN_EDGE_THRESHOLD=0.05, MIN_CONFIDENCE=0.40), bankroll settings.

## Key Data Flow

1. Fetch player game logs (NBADataFetcher) and/or live odds (OddsAPIClient)
2. Run predictions through models → get projection, edge, confidence
3. Edge = (projection - line) / line; recommend over/under/pass based on thresholds
4. Backtest: simulate bets using generate_sample_dataset() for synthetic game logs

## API Rate Limits

- NBA Stats API: 1 req/sec (0.6s delay built in)
- The Odds API: 500 requests/month (free tier)
