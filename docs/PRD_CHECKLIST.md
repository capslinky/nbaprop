# NBA Prop Rebuild Checklist

This checklist breaks the PRD into concrete steps to reach a running, working model with a daily pipeline.

## Phase 0: Scope and Setup
[ ] Confirm MVP scope: daily picks pipeline, CSV output, baseline model, odds-aware scoring
[ ] Choose mandatory data sources (NBA stats, odds, injury report) and define fallbacks
[ ] Define target runtime and environment (local vs server)
[ ] Create project layout and entry point (single CLI command)
[ ] Add config loader with dotenv support and validation

## Phase 1: Ingestion and Storage
[ ] Implement shared rate limiter and retry policy per data source
[ ] Build NBA stats ingestion with caching and schema validation
[ ] Build odds ingestion with snapshot storage and request budgeting
[ ] Build injury ingestion with official report first, fallbacks second
[ ] Store raw payloads and normalized tables with version tags

## Phase 2: Normalization
[ ] Create canonical IDs for players, teams, games, and props
[ ] Normalize timestamps into a single timezone
[ ] Enforce strict schemas for all normalized tables
[ ] Add data freshness checks and fail fast on stale inputs

## Phase 3: Feature Pipeline
[ ] Implement rolling windows (5/10/15) for each prop
[ ] Add matchup, pace, minutes trend, and injury features
[ ] Add usage and efficiency features with safe defaults
[ ] Add a debug mode to output feature vectors per prop

## Phase 4: Modeling and Scoring
[ ] Implement baseline probabilistic model and scoring outputs
[ ] Compute over probability, edge, EV, and confidence
[ ] Define pick rules (min edge and min confidence thresholds)
[ ] Add explainability output for adjustment breakdowns

## Phase 5: Daily Pipeline
[ ] Build the daily pipeline command to run end-to-end
[ ] Output picks to CSV with all context fields
[ ] Add run manifest (timestamp, git hash, config snapshot)
[ ] Add a dry-run mode that uses cached data only

## Phase 6: Backtesting
[ ] Implement leakage-free backtest harness with time alignment
[ ] Use historical odds snapshots, not current lines
[ ] Compute metrics: win rate, ROI, CLV, drawdown
[ ] Save backtest reports with config and data versions

## Phase 7: Testing and Ops
[ ] Add unit tests for ingestion, normalization, features, and scoring
[ ] Add integration tests with recorded fixtures
[ ] Add structured logs and per-source latency metrics
[ ] Add alerts for ingestion failures and stale data

## Definition of Done
[ ] Daily pipeline runs without manual intervention
[ ] Output CSV includes picks, odds-aware edge, confidence, and explanations
[ ] Backtest outputs are reproducible with versioned inputs
[ ] Cache and rate limits prevent API overload
