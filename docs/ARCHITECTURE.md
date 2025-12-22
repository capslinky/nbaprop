# Architecture: NBA Prop Intelligence v2

## Intent
Build a reliable, reproducible, odds-aware NBA prop pipeline with clear separation between ingestion, normalization, features, modeling, scoring, and reporting.

## High-level flow
1. Ingest raw data from NBA stats, odds, and injury sources
2. Normalize into canonical schemas with stable IDs
3. Build feature matrices for each prop
4. Score props with probabilistic outputs, edge, and confidence
5. Report daily picks and backtest results

## System diagram
[Sources] -> [Ingestion] -> [Raw Storage] -> [Normalization] -> [Normalized Storage]
    -> [Feature Pipeline] -> [Model + Scoring] -> [Picks] -> [Reporting]

## Package layout (proposed)
- nbaprop/cli.py: CLI entry points
- nbaprop/config.py: config and environment loading
- nbaprop/ingestion/: source-specific fetchers
- nbaprop/storage/: cache and persistence
- nbaprop/normalization/: canonical IDs and schemas
- nbaprop/features/: feature pipeline
- nbaprop/models/: baseline models
- nbaprop/models/scoring.py: odds-aware scoring
- nbaprop/backtest/: leakage-safe backtest runner
- nbaprop/reporting/: CSV output and report helpers
- nbaprop/ops/: logging and metrics
- nbaprop/runtime/: run manifest and execution metadata

## Data contracts
Raw tables:
- raw_player_logs
- raw_odds_snapshots
- raw_injury_reports

Normalized tables:
- players
- teams
- games
- props
- prop_features
- picks

Outputs:
- picks_daily.csv
- backtest_report.csv

## Run manifest
Required fields:
- run_id
- started_at
- git_sha
- config_hash
- source_versions
- outputs

## Caching and rate limiting
- Shared rate limiter per source across all entry points
- Multi-layer cache: memory, disk, database
- Snapshot storage for odds and injury reports

## Error handling
- Fail fast on stale inputs or missing core sources
- Degrade gracefully when optional sources fail
- Log all fetch attempts with source, latency, and status

## Entry points
- nbaprop.cli.run_daily
- nbaprop.cli.run_backtest
