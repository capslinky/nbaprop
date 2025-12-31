# Existing Data Assets

Use these assets to seed ingestion, tests, backtests, and report formats. Treat them as snapshots, not live sources.

## Canonical references
- data/rosters_2025_26.json: roster + team mappings for the current season.
- data/nba_rosters_2025_26.md: roster notes and structure reference.
- data/nba_schedule_cache.json: cached schedule data for season alignment.

## Cached stats and results
- nba_gamelog_cache.db: cached NBA game logs for offline development.
- nba_historical_cache.db: historical NBA stats cache for backtesting.
- clv_tracking.db: CLV tracking database from prior runs.
- pick_results_tracking.csv: historical pick outcomes and tracking data.
- nba_prop_backtest_results.xlsx: aggregated backtest results.
- data/learned_weights.json: calibrated adjustment factors from prior runs.

## Injury and news samples
- nba_injury_report_2025-12-20.csv: parsed injury report snapshot.
- nba_injury_report_2025-12-20.pdf: raw official injury report sample.

## Output examples (format references)
- daily_reports/12-06-2025.csv: detailed daily report output.
- daily_reports/12-06-2025.xlsx: spreadsheet format for reports.
- daily_reports/12-06-2025_LEGEND.txt: column definitions for reports.
- nba_daily_picks_*.csv: daily picks CSV output samples.
- nba_picks_*.pdf: generated pick report PDFs.

## Test fixtures
- tests/fixtures/sample_api_responses.py: canned API responses for tests.
