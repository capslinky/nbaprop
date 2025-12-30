# PRD: NBA Prop Intelligence v2 (Rebuild)

## Status
- Draft
- Owner: TBD
- Last updated: 2025-12-20

## Overview
Rebuild the NBA prop analysis system from scratch using best practices in data ingestion, modeling, and delivery. The new system must be reliable, reproducible, and odds-aware, while remaining explainable and fast enough for daily and live usage.

## Problem Statement
The current workflow is brittle, hard to reproduce, and too dependent on per-run state. Data ingestion is not consistently rate-limited or cached, context is inconsistently derived, and odds-awareness is not applied end to end. This causes instability, inconsistent outputs, and unnecessary API failures.

## Goals
- Deliver stable daily picks and optional live scanning with high run success rate.
- Make every run reproducible with consistent inputs, configs, and outputs.
- Make predictions odds-aware (true probability, edge, and EV).
- Provide clear explanations for each pick and a traceable adjustment breakdown.
- Support backtesting with leakage-free historical inputs.

## Success Metrics
- Daily run success rate >= 99 percent.
- NBA API error rate <= 1 percent per run (after retries).
- Cache hit rate >= 70 percent for repeat queries.
- Measurable CLV and ROI tracking in backtests.
- End-to-end daily run time <= 10 minutes on a single machine.

## Non-goals
- Automated wagering or sportsbook execution.
- Social features, sharing, or community rankings.
- Multi-sport expansion in the initial rebuild.

## Users and Use Cases
- Analyst: Run daily picks, review explanations, and evaluate performance.
- Bettor: Consume picks with confidence scores and context flags.
- Operator: Monitor system health, data freshness, and failures.

Primary use cases:
- Daily picks generation with CSV and PDF output.
- Live scan for updated lines and injury changes.
- Backtest on historical games and props with metrics.

## Scope
- Data ingestion, normalization, caching, and storage.
- Feature engineering and model scoring.
- Backtesting and evaluation.
- Reporting and a lightweight UI or CLI interface.
- Observability, logging, and operational safety.

## Requirements

### Functional Requirements
- Ingest NBA player logs, team stats, and schedules with strict rate limits.
- Ingest odds and lines from a sportsbook API with retries and backoff.
- Ingest official NBA injury report and fallbacks with source priority.
- Normalize all entities to canonical player and team identifiers.
- Generate features for matchup, pace, minutes trend, usage, and injuries.
- Score props with probabilistic outputs, edge, EV, and confidence.
- Produce explainability output with adjustment breakdowns.
- Backtest using historical odds and game logs with time alignment.
- Output picks as CSV and PDF, and support a minimal UI or CLI.

### Non-Functional Requirements
- Idempotent runs: rerunning a job with the same inputs produces the same output.
- Configurable and validated environment settings with dotenv support.
- All external calls must be cached, rate-limited, and retried.
- Clear separation of concerns between ingestion, features, modeling, and reporting.
- Extensible to new data sources with minimal changes.

## Data Sources and Contracts
- NBA Stats API: player logs, team stats, pace, advanced stats.
- Odds API: upcoming games, props, lines, and odds.
- Official NBA injury report: nba.com PDF.
- Optional news API for late-breaking context.

Data contracts:
- All ingestion outputs must conform to a strict schema with types and required fields.
- Each record must include a timestamp, source, and version tag.
- Errors and missing fields must be recorded, not silently dropped.

## Architecture and Components
- Ingestion layer: rate-limited fetchers with shared sessions and retries.
- Normalization layer: canonical player, team, and game identifiers.
- Feature pipeline: stat windows and context features.
- Model layer: probabilistic predictor and calibration.
- Scoring layer: edges, EV, and confidence.
- Reporting layer: CSV, PDF, and UI output.
- Ops layer: logging, metrics, and error handling.

## Caching and Rate Limiting
- Shared rate limiter per source across all workers and entry points.
- Multi-layer caching (memory, disk, and database).
- TTL-based cache invalidation with optional manual refresh.
- Negative caching for repeated failures.

## Storage and Data Retention
- Raw data stored in a durable format (Parquet or SQLite).
- Processed features stored with versioned schemas.
- Retention policy for raw and derived datasets.

## Modeling and Calibration
- Baseline models for projections (recent, median, weighted).
- Ensemble or probabilistic model for final scoring.
- Calibrate probabilities using historical results.
- Track model drift and performance over time.

## Backtesting and Evaluation
- Strict leakage prevention with time-aligned data.
- Use historical odds snapshots, not current lines.
- Report metrics: win rate, ROI, CLV, max drawdown, and confidence calibration.

## Reporting and Interfaces
- Daily picks output: CSV and PDF with context fields.
- Optional live scan with refreshed data.
- Minimal UI (web or CLI) for on-demand analysis.

## Observability and Ops
- Structured logs with request IDs and run IDs.
- Metrics for API latency, cache hit rate, and error rates.
- Alerts for data freshness, ingestion failures, and model drift.

## Security and Compliance
- Secrets in env vars only, not in code or logs.
- Optional encryption for stored data if running on shared hosts.
- Clear data provenance and audit trails for outputs.

## Milestones
- M1: Ingestion, normalization, shared caching, storage.
- M2: Feature pipeline and baseline model outputs.
- M3: Probabilistic model and calibration.
- M4: Backtesting and reporting outputs.
- M5: UI or CLI polish, observability, and hardening.

## Risks
- API rate limits and changes in upstream schemas.
- Injury data accuracy and timeliness.
- Historical odds availability and quality.
- Model overfitting without a robust validation plan.

## Open Questions
- Which entry points are required at launch: CLI only, UI, or API?
- Should we prioritize accuracy or explainability if they conflict?
- What is the minimum acceptable runtime and hardware target?
- Which data sources are mandatory versus optional?
- What level of automation is desired for daily runs?

## Acceptance Criteria
- A full daily run completes without manual intervention.
- Outputs include picks, edges, confidence, and explanations.
- Backtests are reproducible with versioned inputs and configs.
- Caches and rate limits prevent API overload.
