"""CLI entry points for the rebuild."""

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import logging
import os
import subprocess
from zoneinfo import ZoneInfo

from nbaprop.config import Config
from nbaprop.ops.logging import configure_logging
from nbaprop.ops import get_metrics_recorder
from nbaprop.runtime.manifest import RunManifest
from nbaprop.storage import FileCache, JsonStorage
from nbaprop.ingestion.odds import fetch_odds_snapshot
from nbaprop.ingestion.nba_stats import fetch_player_logs, fetch_team_stats
from nbaprop.ingestion.injuries import fetch_injury_report
from nbaprop.normalization import normalize_raw_data
from nbaprop.features.pipeline import build_features
from nbaprop.models.baseline import BaselineModel
from nbaprop.models.calibration import (
    load_calibration,
    resolve_calibration_path,
    fit_linear_calibration,
    save_calibration,
)
from nbaprop.reporting.csv_output import write_picks_csv
from nbaprop.reporting.pdf_output import write_picks_pdf
from nbaprop.reporting.calibration_output import write_calibration_report
from nbaprop.ingestion.rosters import refresh_rosters
from nbaprop.backtest.runner import (
    run_recent_backtest,
    run_historical_backtest,
    load_historical_props_from_db,
    run_historical_model_backtest,
)
from nbaprop.review import (
    find_latest_picks_for_date,
    load_picks,
    load_picks_from_dir,
    load_picks_from_paths,
    evaluate_picks,
    evaluate_picks_range,
    calibrate_market_blend,
    update_env_value,
)
from nbaprop.ingestion.legacy_picks import (
    discover_legacy_pick_files,
    normalize_legacy_pick_files,
    load_pick_tracker_db,
    dedupe_rows,
)

logger = logging.getLogger(__name__)


def _get_git_sha(repo_root: Path) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return None


def _hash_config(config: Config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(manifest: RunManifest, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{manifest.run_id}.json"
    manifest.outputs["manifest"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _try_load_dotenv(repo_root: Path) -> Optional[str]:
    candidate_paths = [repo_root / ".env", repo_root.parent / ".env"]
    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        try:
            config = Config.load(str(env_path))
            os.environ.setdefault("ODDS_API_KEY", config.odds_api_key)
            os.environ.setdefault("ODDS_MAX_EVENTS", str(config.odds_max_events))
            os.environ.setdefault("ODDS_MAX_PLAYERS", str(config.odds_max_players))
            os.environ.setdefault("ODDS_PROP_MARKETS", ",".join(config.odds_prop_markets))
            if config.odds_min is not None:
                os.environ.setdefault("ODDS_MIN", str(config.odds_min))
            if config.odds_max is not None:
                os.environ.setdefault("ODDS_MAX", str(config.odds_max))
            if config.odds_confidence_min is not None:
                os.environ.setdefault("ODDS_CONFIDENCE_MIN", str(config.odds_confidence_min))
            if config.odds_confidence_max is not None:
                os.environ.setdefault("ODDS_CONFIDENCE_MAX", str(config.odds_confidence_max))
            os.environ.setdefault(
                "CONFIDENCE_CAP_OUTSIDE_BAND",
                str(config.confidence_cap_outside_band),
            )
            os.environ.setdefault("MIN_EDGE_THRESHOLD", str(config.min_edge_threshold))
            os.environ.setdefault("MIN_CONFIDENCE", str(config.min_confidence))
            os.environ.setdefault("NBA_API_DELAY", str(config.nba_api_delay))
            os.environ.setdefault("NBAPROP_CACHE_DIR", config.cache_dir)
            os.environ.setdefault("PERPLEXITY_API_KEY", config.perplexity_api_key)
            os.environ.setdefault("MARKET_BLEND", str(config.market_blend))
            os.environ.setdefault("MAX_PICKS", str(config.max_picks))
            os.environ.setdefault("HISTORICAL_PROPS_PATH", config.historical_props_path)
            os.environ.setdefault("RUN_DATE", config.run_date)
            os.environ.setdefault("AUTO_REVIEW", str(int(config.auto_review)))
            os.environ.setdefault("REVIEW_STEP", str(config.review_step))
            os.environ.setdefault("REVIEW_UPDATE_ENV", str(int(config.review_update_env)))
            os.environ.setdefault("NBAPROP_CALIBRATION_PATH", config.calibration_path)
            os.environ.setdefault("CALIBRATION_MIN_SAMPLES", str(config.calibration_min_samples))
            os.environ.setdefault("CLV_ENABLED", str(int(config.clv_enabled)))
            os.environ.setdefault("CLV_SNAPSHOT_TTL", str(config.clv_snapshot_ttl))
            return str(env_path)
        except Exception:
            continue
    return None


def _load_historical_props(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prop_type = row.get("prop_type") or row.get("prop")
            if prop_type:
                row["prop_type"] = prop_type.lower()
            side = row.get("side") or row.get("pick")
            if side:
                row["side"] = str(side).lower()
            rows.append(row)
    return rows


def _resolve_run_date(config: Config) -> str:
    if config.run_date:
        return config.run_date
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.strftime("%Y-%m-%d")


def _resolve_review_date(config: Config, target_date: Optional[str]) -> str:
    if target_date:
        return target_date
    if config.run_date:
        return config.run_date
    now = datetime.now(ZoneInfo("America/New_York"))
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")


def _resolve_auto_review_date(target_date: Optional[str]) -> str:
    if target_date:
        return target_date
    now = datetime.now(ZoneInfo("America/New_York"))
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")


def _resolve_optional_flag(flag_value: Optional[bool], config_value: bool) -> bool:
    return config_value if flag_value is None else flag_value


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _resolve_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    days: Optional[int],
) -> Tuple[Optional[str], Optional[str]]:
    tz = ZoneInfo("America/New_York")
    end = _parse_date(end_date)
    if end is None:
        end = (datetime.now(tz) - timedelta(days=1)).date()
    if days:
        start = end - timedelta(days=max(days, 1) - 1)
    else:
        start = _parse_date(start_date)
    start_str = start.strftime("%Y-%m-%d") if start else None
    end_str = end.strftime("%Y-%m-%d") if end else None
    return start_str, end_str


def run_daily(
    config_path: Optional[str] = None,
    auto_review: Optional[bool] = None,
    review_date: Optional[str] = None,
    review_step: Optional[float] = None,
    review_update_env: Optional[bool] = None,
    review_output_dir: Optional[str] = None,
    review_picks_path: Optional[str] = None,
    review_base_delay: Optional[float] = None,
) -> int:
    """Run the daily pipeline end-to-end (no-op scaffold)."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    config = Config.load(config_path=config_path)

    manifest = RunManifest()
    manifest.git_sha = _get_git_sha(repo_root)
    manifest.config_hash = _hash_config(config)

    configure_logging(run_id=manifest.run_id)
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    refresh_rosters(
        output_path=None,
        season=None,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
        force=False,
        max_age_days=1,
    )

    cache_dir = Path(config.cache_dir)
    runs_dir = cache_dir / "runs"
    cache = FileCache(cache_dir / "snapshots")
    storage = JsonStorage(str(cache_dir / "storage"))
    manifest_path = _write_manifest(manifest, runs_dir)
    calibration_path = resolve_calibration_path(config.calibration_path, config.cache_dir, repo_root)
    calibration_data = load_calibration(calibration_path)
    run_date = _resolve_run_date(config)
    snapshot = fetch_odds_snapshot(
        cache,
        ttl_seconds=60,
        api_key=config.odds_api_key,
        max_events=0,
        markets=config.odds_prop_markets,
        target_date=run_date,
        bookmakers=config.odds_bookmakers,
    )
    manifest.outputs["odds_snapshot"] = f"{snapshot.get('source')}:{snapshot.get('fetched_at')}"
    manifest.outputs["raw_odds_path"] = storage.write_table("raw_odds_snapshot", [snapshot])

    players = snapshot.get("players") or []
    if config.odds_max_players and config.odds_max_players > 0 and players:
        if len(players) > config.odds_max_players:
            logger.warning(
                "Limiting player logs to %d of %d players. Set ODDS_MAX_PLAYERS=0 to fetch all.",
                config.odds_max_players,
                len(players),
            )
        players = players[:config.odds_max_players]
    if not players:
        players = ["Example Player"]
    player_logs = fetch_player_logs(
        players,
        cache,
        ttl_seconds=300,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
    )
    manifest.outputs["raw_player_logs_path"] = storage.write_table("raw_player_logs", player_logs)

    team_stats = fetch_team_stats(
        cache,
        ttl_seconds=600,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
    )
    manifest.outputs["raw_team_stats_path"] = storage.write_table("raw_team_stats", team_stats)

    cache.invalidate("injury_report")
    perplexity_key = config.perplexity_api_key if config.perplexity_enabled else None
    injury_report = fetch_injury_report(
        cache,
        ttl_seconds=300,
        perplexity_api_key=perplexity_key,
    )
    manifest.outputs["raw_injury_report_path"] = storage.write_table("raw_injury_report", [injury_report])

    normalized = normalize_raw_data(snapshot, player_logs, injury_report)
    normalized["team_stats"] = team_stats
    normalized["prop_features"] = build_features(
        normalized.get("props", []),
        raw_player_logs=player_logs,
        injuries=normalized.get("injuries", []),
        players=normalized.get("players", []),
        team_stats=team_stats,
        excluded_prop_types=config.excluded_prop_types,
    )
    model = BaselineModel(
        min_edge=config.min_edge_threshold,
        min_confidence=config.min_confidence,
        prop_type_min_edge=config.prop_type_min_edge,
        prop_type_min_confidence=config.prop_type_min_confidence,
        excluded_prop_types=config.excluded_prop_types,
        injury_risk_edge_multiplier=config.injury_risk_edge_multiplier,
        injury_risk_confidence_multiplier=config.injury_risk_confidence_multiplier,
        market_blend=config.market_blend,
        calibration=calibration_data,
        odds_min=config.odds_min,
        odds_max=config.odds_max,
        odds_confidence_min=config.odds_confidence_min,
        odds_confidence_max=config.odds_confidence_max,
        confidence_cap_outside_band=config.confidence_cap_outside_band,
    )
    if calibration_data:
        manifest.outputs["calibration_params"] = str(calibration_path)
    normalized["picks"] = model.predict(normalized["prop_features"])
    for name, rows in normalized.items():
        manifest.outputs[f"normalized_{name}_path"] = storage.write_table(f"normalized_{name}", rows)

    picks_path = runs_dir / f"picks_{manifest.run_id}.csv"
    picks = normalized.get("picks", [])
    odds_fetched_at = snapshot.get("fetched_at")
    if odds_fetched_at:
        for row in picks:
            row["odds_fetched_at"] = odds_fetched_at
            row["odds_source"] = snapshot.get("source")
    write_picks_csv(picks, str(picks_path))
    manifest.outputs["picks_csv"] = str(picks_path)

    def _game_key(row: Dict) -> str:
        home = row.get("home_team")
        away = row.get("away_team")
        if home and away:
            return f"{away} @ {home}"
        return row.get("event_id") or "UNKNOWN"

    def _rank_score(row: Dict) -> float:
        edge = row.get("edge") or 0.0
        confidence = row.get("confidence") or 0.0
        if config.pick_ranking_mode == "edge":
            return edge
        return abs(edge) * confidence

    filtered_picks = [row for row in picks if row.get("pick") not in (None, "PASS")]
    for row in filtered_picks:
        row["game"] = _game_key(row)

    if config.top_picks_per_game and config.top_picks_per_game > 0:
        grouped: Dict[str, List[Dict]] = {}
        for row in filtered_picks:
            grouped.setdefault(row["game"], []).append(row)
        filtered_picks = []
        for rows in grouped.values():
            rows.sort(key=_rank_score, reverse=True)
            filtered_picks.extend(rows[: config.top_picks_per_game])

    filtered_picks.sort(key=_rank_score, reverse=True)
    if config.max_picks and config.max_picks > 0:
        filtered_picks = filtered_picks[:config.max_picks]
    filtered_path = runs_dir / f"picks_filtered_{manifest.run_id}.csv"
    write_picks_csv(filtered_picks, str(filtered_path))
    manifest.outputs["picks_filtered_csv"] = str(filtered_path)

    pdf_path = runs_dir / f"picks_{manifest.run_id}.pdf"
    pdf_result = write_picks_pdf(
        filtered_picks,
        str(pdf_path),
        title="NBA Prop Picks",
        all_games=snapshot.get("events") if isinstance(snapshot.get("events"), list) else None,
    )
    if pdf_result:
        manifest.outputs["picks_pdf"] = pdf_result

    metrics_path = runs_dir / f"metrics_{manifest.run_id}.json"
    metrics_path.write_text(
        json.dumps(get_metrics_recorder().snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["metrics"] = str(metrics_path)

    auto_review_enabled = _resolve_optional_flag(auto_review, config.auto_review)
    if auto_review_enabled:
        resolved_review_date = _resolve_auto_review_date(review_date)
        resolved_review_step = review_step if review_step is not None else config.review_step
        resolved_update_env = _resolve_optional_flag(review_update_env, config.review_update_env)
        review_dir = Path(review_output_dir) if review_output_dir else cache_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)

        try:
            if review_picks_path:
                review_picks_file = Path(review_picks_path)
            else:
                review_picks_file = find_latest_picks_for_date(runs_dir, resolved_review_date)
            review_picks = load_picks(review_picks_file)
            clv_snapshot = None
            if config.clv_enabled:
                clv_snapshot = fetch_odds_snapshot(
                    cache,
                    ttl_seconds=config.clv_snapshot_ttl,
                    api_key=config.odds_api_key,
                    max_events=0,
                    markets=config.odds_prop_markets,
                    target_date=resolved_review_date,
                    bookmakers=config.odds_bookmakers,
                )
                manifest.outputs["review_odds_snapshot"] = (
                    f"{clv_snapshot.get('source')}:{clv_snapshot.get('fetched_at')}"
                )
            review_results, review_summary = evaluate_picks(
                review_picks,
                resolved_review_date,
                cache_dir=config.cache_dir,
                base_delay=review_base_delay if review_base_delay is not None else config.nba_api_delay,
                odds_snapshot=clv_snapshot,
            )
            calibration = calibrate_market_blend(review_results, step=resolved_review_step)
            prob_key = "raw_model_probability"
            if not any(row.get(prob_key) not in (None, "") for row in review_results):
                prob_key = "model_probability"
            model_calibration = fit_linear_calibration(
                review_results,
                prob_key=prob_key,
                min_samples=config.calibration_min_samples,
            )

            review_csv = review_dir / f"review_{manifest.run_id}.csv"
            write_picks_csv(review_results, str(review_csv))
            manifest.outputs["review_csv"] = str(review_csv)

            summary_payload = {
                "target_date": resolved_review_date,
                "source_picks": str(review_picks_file),
                "reviewed_at": datetime.utcnow().isoformat(),
                "summary": review_summary,
                "calibration": calibration,
                "model_calibration_path": str(review_dir / f"model_calibration_{manifest.run_id}.json"),
            }
            summary_path = review_dir / f"review_summary_{manifest.run_id}.json"
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            manifest.outputs["review_summary"] = str(summary_path)

            model_calibration_path = review_dir / f"model_calibration_{manifest.run_id}.json"
            model_calibration_path.write_text(
                json.dumps(model_calibration, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            manifest.outputs["model_calibration"] = str(model_calibration_path)
            save_calibration(calibration_path, model_calibration)
            manifest.outputs["model_calibration_params"] = str(calibration_path)

            if resolved_update_env and calibration.get("best_blend") is not None:
                resolved_env = Path(dotenv_path or (repo_root / ".env"))
                update_env_value(resolved_env, "MARKET_BLEND", str(calibration["best_blend"]))
                manifest.outputs["updated_env"] = str(resolved_env)
        except Exception as exc:
            logger.warning("Auto-review failed: %s", exc)

    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("No-op daily run completed.")
    logger.info("Run manifest written to %s", manifest_path)
    return 0


def run_backtest(config_path: Optional[str] = None) -> int:
    """Run the backtest pipeline end-to-end."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    config = Config.load(config_path=config_path)

    manifest = RunManifest()
    manifest.git_sha = _get_git_sha(repo_root)
    manifest.config_hash = _hash_config(config)

    configure_logging(run_id=manifest.run_id)
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    refresh_rosters(
        output_path=None,
        season=None,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
        force=False,
        max_age_days=1,
    )

    cache_dir = Path(config.cache_dir)
    backtests_dir = cache_dir / "backtests"
    cache = FileCache(cache_dir / "snapshots")
    storage = JsonStorage(str(cache_dir / "storage"))
    calibration_path = resolve_calibration_path(config.calibration_path, config.cache_dir, repo_root)
    calibration_data = load_calibration(calibration_path)

    run_date = _resolve_run_date(config)
    snapshot = fetch_odds_snapshot(
        cache,
        ttl_seconds=60,
        api_key=config.odds_api_key,
        max_events=0,
        markets=config.odds_prop_markets,
        target_date=run_date,
        bookmakers=config.odds_bookmakers,
    )
    manifest.outputs["odds_snapshot"] = f"{snapshot.get('source')}:{snapshot.get('fetched_at')}"
    manifest.outputs["raw_odds_path"] = storage.write_table("raw_odds_snapshot", [snapshot])

    players = snapshot.get("players") or []
    if config.odds_max_players and config.odds_max_players > 0 and players:
        if len(players) > config.odds_max_players:
            logger.warning(
                "Limiting player logs to %d of %d players. Set ODDS_MAX_PLAYERS=0 to fetch all.",
                config.odds_max_players,
                len(players),
            )
        players = players[:config.odds_max_players]
    if not players:
        players = ["Example Player"]
    player_logs = fetch_player_logs(
        players,
        cache,
        ttl_seconds=300,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
    )
    manifest.outputs["raw_player_logs_path"] = storage.write_table("raw_player_logs", player_logs)

    team_stats = fetch_team_stats(
        cache,
        ttl_seconds=600,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
    )
    manifest.outputs["raw_team_stats_path"] = storage.write_table("raw_team_stats", team_stats)

    cache.invalidate("injury_report")
    perplexity_key = config.perplexity_api_key if config.perplexity_enabled else None
    injury_report = fetch_injury_report(
        cache,
        ttl_seconds=300,
        perplexity_api_key=perplexity_key,
    )
    manifest.outputs["raw_injury_report_path"] = storage.write_table("raw_injury_report", [injury_report])

    normalized = normalize_raw_data(snapshot, player_logs, injury_report)
    normalized["team_stats"] = team_stats
    for name, rows in normalized.items():
        manifest.outputs[f"normalized_{name}_path"] = storage.write_table(f"normalized_{name}", rows)

    backtest_payload = run_recent_backtest(
        normalized.get("props", []),
        raw_player_logs=player_logs,
        injuries=normalized.get("injuries", []),
        players=normalized.get("players", []),
        team_stats=team_stats,
        min_edge=config.min_edge_threshold,
        min_confidence=config.min_confidence,
        prop_type_min_edge=config.prop_type_min_edge,
        prop_type_min_confidence=config.prop_type_min_confidence,
        excluded_prop_types=config.excluded_prop_types,
        injury_risk_edge_multiplier=config.injury_risk_edge_multiplier,
        injury_risk_confidence_multiplier=config.injury_risk_confidence_multiplier,
        market_blend=config.market_blend,
        calibration=calibration_data,
        odds_min=config.odds_min,
        odds_max=config.odds_max,
        odds_confidence_min=config.odds_confidence_min,
        odds_confidence_max=config.odds_confidence_max,
        confidence_cap_outside_band=config.confidence_cap_outside_band,
        windows=(10, 15),
    )
    if calibration_data:
        manifest.outputs["calibration_params"] = str(calibration_path)

    backtests_dir.mkdir(parents=True, exist_ok=True)
    backtest_csv = backtests_dir / f"backtest_{manifest.run_id}.csv"
    write_picks_csv(backtest_payload["results"], str(backtest_csv))
    manifest.outputs["backtest_csv"] = str(backtest_csv)

    summary_path = backtests_dir / f"backtest_summary_{manifest.run_id}.json"
    summary_path.write_text(
        json.dumps(backtest_payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["backtest_summary"] = str(summary_path)

    backtest_json_path = storage.write_table(
        f"backtest_results_{manifest.run_id}",
        backtest_payload["results"],
    )
    manifest.outputs["backtest_results_path"] = backtest_json_path

    historical_path = Path(config.historical_props_path) if config.historical_props_path else None
    if historical_path and historical_path.exists():
        historical_props = _load_historical_props(historical_path)
        historical_payload = run_historical_backtest(
            historical_props,
            raw_player_logs=player_logs,
            injuries=normalized.get("injuries", []),
            players=normalized.get("players", []),
            team_stats=team_stats,
            min_edge=config.min_edge_threshold,
            min_confidence=config.min_confidence,
            market_blend=config.market_blend,
            calibration=calibration_data,
            odds_min=config.odds_min,
            odds_max=config.odds_max,
            odds_confidence_min=config.odds_confidence_min,
            odds_confidence_max=config.odds_confidence_max,
            confidence_cap_outside_band=config.confidence_cap_outside_band,
            excluded_prop_types=config.excluded_prop_types,
        )
        historical_csv = backtests_dir / f"historical_backtest_{manifest.run_id}.csv"
        write_picks_csv(historical_payload["results"], str(historical_csv))
        manifest.outputs["historical_backtest_csv"] = str(historical_csv)

        historical_summary_path = backtests_dir / f"historical_backtest_summary_{manifest.run_id}.json"
        historical_summary_path.write_text(
            json.dumps(historical_payload["summary"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest.outputs["historical_backtest_summary"] = str(historical_summary_path)
    elif config.historical_props_path:
        logger.warning("Historical props path not found: %s", config.historical_props_path)

    metrics_path = backtests_dir / f"metrics_{manifest.run_id}.json"
    metrics_path.write_text(
        json.dumps(get_metrics_recorder().snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["metrics"] = str(metrics_path)

    manifest_path = _write_manifest(manifest, backtests_dir)

    logger.info("Backtest run completed.")
    logger.info("Backtest manifest written to %s", manifest_path)
    return 0


def run_review(
    config_path: Optional[str] = None,
    target_date: Optional[str] = None,
    picks_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    update_env: bool = False,
    env_path: Optional[str] = None,
    step: Optional[float] = None,
    base_delay: Optional[float] = None,
) -> int:
    """Review prior picks and calibrate market blend."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    config = Config.load(config_path=config_path)

    manifest = RunManifest()
    manifest.git_sha = _get_git_sha(repo_root)
    manifest.config_hash = _hash_config(config)

    configure_logging(run_id=manifest.run_id)
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    review_date = _resolve_review_date(config, target_date)
    resolved_step = step if step is not None else config.review_step
    cache_dir = Path(config.cache_dir)
    runs_dir = cache_dir / "runs"
    review_dir = Path(output_dir) if output_dir else cache_dir / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = resolve_calibration_path(config.calibration_path, config.cache_dir, repo_root)

    if picks_path:
        picks_file = Path(picks_path)
    else:
        picks_file = find_latest_picks_for_date(runs_dir, review_date)
    picks = load_picks(picks_file)
    manifest.outputs["source_picks"] = str(picks_file)

    clv_snapshot = None
    if config.clv_enabled:
        clv_snapshot = fetch_odds_snapshot(
            FileCache(Path(config.cache_dir) / "snapshots"),
            ttl_seconds=config.clv_snapshot_ttl,
            api_key=config.odds_api_key,
            max_events=0,
            markets=config.odds_prop_markets,
            target_date=review_date,
            bookmakers=config.odds_bookmakers,
        )
        manifest.outputs["review_odds_snapshot"] = (
            f"{clv_snapshot.get('source')}:{clv_snapshot.get('fetched_at')}"
        )
    results, summary = evaluate_picks(
        picks,
        review_date,
        cache_dir=config.cache_dir,
        base_delay=base_delay if base_delay is not None else config.nba_api_delay,
        odds_snapshot=clv_snapshot,
    )
    calibration = calibrate_market_blend(results, step=resolved_step)
    prob_key = "raw_model_probability"
    if not any(row.get(prob_key) not in (None, "") for row in results):
        prob_key = "model_probability"
    model_calibration = fit_linear_calibration(
        results,
        prob_key=prob_key,
        min_samples=config.calibration_min_samples,
    )

    review_csv = review_dir / f"review_{manifest.run_id}.csv"
    write_picks_csv(results, str(review_csv))
    manifest.outputs["review_csv"] = str(review_csv)

    summary_payload = {
        "target_date": review_date,
        "source_picks": str(picks_file),
        "reviewed_at": datetime.utcnow().isoformat(),
        "summary": summary,
        "calibration": calibration,
        "model_calibration_path": str(review_dir / f"model_calibration_{manifest.run_id}.json"),
    }
    summary_path = review_dir / f"review_summary_{manifest.run_id}.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["review_summary"] = str(summary_path)

    model_calibration_path = review_dir / f"model_calibration_{manifest.run_id}.json"
    model_calibration_path.write_text(
        json.dumps(model_calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["model_calibration"] = str(model_calibration_path)
    save_calibration(calibration_path, model_calibration)
    manifest.outputs["model_calibration_params"] = str(calibration_path)

    if update_env and calibration.get("best_blend") is not None:
        resolved_env = Path(env_path or dotenv_path or (repo_root / ".env"))
        update_env_value(resolved_env, "MARKET_BLEND", str(calibration["best_blend"]))
        manifest.outputs["updated_env"] = str(resolved_env)

    manifest_path = _write_manifest(manifest, review_dir)
    logger.info("Review completed for %s", review_date)
    logger.info("Review manifest written to %s", manifest_path)
    return 0


def run_backtest_picks(
    config_path: Optional[str] = None,
    picks_dir: Optional[str] = None,
    picks_pattern: str = "picks_filtered_*.csv",
    picks_paths: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    output_dir: Optional[str] = None,
    bins: int = 10,
    prob_keys: Optional[str] = None,
    include_pass: bool = False,
    base_delay: Optional[float] = None,
    cache_only: bool = False,
) -> int:
    """Backtest historical picks and generate calibration artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    config = Config.load(config_path=config_path)

    manifest = RunManifest()
    manifest.git_sha = _get_git_sha(repo_root)
    manifest.config_hash = _hash_config(config)

    configure_logging(run_id=manifest.run_id)
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    resolved_start, resolved_end = _resolve_date_range(start_date, end_date, days)
    pick_paths: List[Path] = []
    base_dir = Path(picks_dir) if picks_dir else Path(config.cache_dir) / "runs"
    pick_paths.extend(sorted(base_dir.glob(picks_pattern)))
    if picks_paths:
        pick_paths.extend([Path(path) for path in picks_paths])
    # De-duplicate paths while preserving order
    seen_paths = set()
    deduped_paths: List[Path] = []
    for path in pick_paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        deduped_paths.append(path)
    pick_paths = deduped_paths

    if not pick_paths:
        logger.warning("No pick files found for backtest.")
        return 1

    picks = load_picks_from_paths(pick_paths)
    if not include_pass:
        picks = [
            row for row in picks
            if (row.get("pick") or row.get("side") or "").upper() in ("OVER", "UNDER")
        ]

    results, summary = evaluate_picks_range(
        picks,
        start_date=resolved_start,
        end_date=resolved_end,
        cache_dir=config.cache_dir,
        base_delay=base_delay if base_delay is not None else config.nba_api_delay,
        cache_only=cache_only,
    )

    output_root = Path(output_dir) if output_dir else Path(config.cache_dir) / "backtests" / "historical"
    output_root.mkdir(parents=True, exist_ok=True)

    results_path = output_root / f"historical_backtest_{manifest.run_id}.csv"
    write_picks_csv(results, str(results_path))
    manifest.outputs["historical_backtest_csv"] = str(results_path)

    summary_payload = {
        "date_range": {"start": resolved_start, "end": resolved_end},
        "source_files": [str(path) for path in pick_paths],
        "summary": summary,
    }
    summary_path = output_root / f"historical_backtest_summary_{manifest.run_id}.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["historical_backtest_summary"] = str(summary_path)

    prob_key_list = [
        key.strip()
        for key in (prob_keys or "probability,model_probability,raw_model_probability").split(",")
        if key.strip()
    ]
    available_prob_keys = []
    for key in prob_key_list:
        if any(row.get(key) not in (None, "") for row in results):
            available_prob_keys.append(key)

    if available_prob_keys:
        calibration_dir = output_root / f"calibration_{manifest.run_id}"
        calibration_report = write_calibration_report(
            results,
            calibration_dir,
            available_prob_keys,
            bins=bins,
            title_prefix="Calibration",
        )
        manifest.outputs["calibration_summary"] = str(calibration_dir / "calibration_summary.json")
        manifest.outputs["calibration_dir"] = str(calibration_dir)
        if calibration_report:
            manifest.outputs["calibration_keys"] = ",".join(available_prob_keys)

    manifest_path = _write_manifest(manifest, output_root)
    logger.info("Historical picks backtest completed.")
    logger.info("Backtest manifest written to %s", manifest_path)
    return 0


def run_historical_model(
    config_path: Optional[str] = None,
    db_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    output_dir: Optional[str] = None,
    bins: int = 10,
    base_delay: Optional[float] = None,
) -> int:
    """Run current model against historical props database."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    config = Config.load(config_path=config_path)

    manifest = RunManifest()
    manifest.git_sha = _get_git_sha(repo_root)
    manifest.config_hash = _hash_config(config)

    configure_logging(run_id=manifest.run_id)
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    resolved_start, resolved_end = _resolve_date_range(start_date, end_date, days)
    historical_db = Path(db_path) if db_path else repo_root / "nba_historical_cache.db"
    historical_props = load_historical_props_from_db(historical_db, resolved_start, resolved_end)
    if not historical_props:
        logger.warning("No historical props found in %s", historical_db)
        return 1

    calibration_path = resolve_calibration_path(config.calibration_path, config.cache_dir, repo_root)
    calibration_data = load_calibration(calibration_path)
    payload = run_historical_model_backtest(
        historical_props,
        cache_dir=config.cache_dir,
        min_edge=config.min_edge_threshold,
        min_confidence=config.min_confidence,
        prop_type_min_edge=config.prop_type_min_edge,
        prop_type_min_confidence=config.prop_type_min_confidence,
        excluded_prop_types=config.excluded_prop_types,
        injury_risk_edge_multiplier=config.injury_risk_edge_multiplier,
        injury_risk_confidence_multiplier=config.injury_risk_confidence_multiplier,
        market_blend=config.market_blend,
        calibration=calibration_data,
        odds_min=config.odds_min,
        odds_max=config.odds_max,
        odds_confidence_min=config.odds_confidence_min,
        odds_confidence_max=config.odds_confidence_max,
        confidence_cap_outside_band=config.confidence_cap_outside_band,
        base_delay=base_delay if base_delay is not None else config.nba_api_delay,
    )
    if calibration_data:
        manifest.outputs["calibration_params"] = str(calibration_path)

    output_root = Path(output_dir) if output_dir else Path(config.cache_dir) / "backtests" / "historical_model"
    output_root.mkdir(parents=True, exist_ok=True)

    results_path = output_root / f"historical_model_{manifest.run_id}.csv"
    write_picks_csv(payload["results"], str(results_path))
    manifest.outputs["historical_model_csv"] = str(results_path)

    summary_path = output_root / f"historical_model_summary_{manifest.run_id}.json"
    summary_path.write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.outputs["historical_model_summary"] = str(summary_path)

    calibration_dir = output_root / f"calibration_{manifest.run_id}"
    calibration_report = write_calibration_report(
        payload["results"],
        calibration_dir,
        ["probability", "model_probability"],
        bins=bins,
        title_prefix="Historical Model Calibration",
    )
    manifest.outputs["calibration_summary"] = str(calibration_dir / "calibration_summary.json")
    manifest.outputs["calibration_dir"] = str(calibration_dir)
    if calibration_report:
        manifest.outputs["calibration_keys"] = "probability,model_probability"

    manifest_path = _write_manifest(manifest, output_root)
    logger.info("Historical model backtest completed.")
    logger.info("Backtest manifest written to %s", manifest_path)
    return 0


def run_import_legacy_picks(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    paths: Optional[Sequence[str]] = None,
) -> int:
    """Normalize legacy pick CSVs into the new format."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    _ = Config.load(config_path=config_path)

    configure_logging(run_id="legacy-import")
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    if paths:
        pick_paths = [Path(path) for path in paths]
    else:
        pick_paths = discover_legacy_pick_files(repo_root)

    if not pick_paths:
        logger.warning("No legacy pick files found for import.")
        return 1

    normalized_rows, summary = normalize_legacy_pick_files(pick_paths)
    pick_tracker_db = repo_root / "clv_tracking.db"
    db_rows = load_pick_tracker_db(pick_tracker_db)
    if db_rows:
        normalized_rows.extend(db_rows)
        summary["included_pick_tracker_db"] = str(pick_tracker_db)
        summary["pick_tracker_rows"] = len(db_rows)
    normalized_rows = dedupe_rows(normalized_rows)
    summary["deduped_rows"] = len(normalized_rows)
    if not normalized_rows:
        logger.warning("No rows produced from legacy pick import.")
        return 1

    output_root = Path(output_dir) if output_dir else repo_root / ".cache" / "legacy_picks"
    output_root.mkdir(parents=True, exist_ok=True)
    output_name = output_file or "legacy_picks_combined.csv"
    output_path = output_root / output_name
    write_picks_csv(normalized_rows, str(output_path))

    summary_path = output_root / "legacy_picks_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Legacy picks imported to %s", output_path)
    logger.info("Summary written to %s", summary_path)
    return 0


def run_update_rosters(
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    force: bool = False,
    season: Optional[str] = None,
) -> int:
    """Refresh roster and depth chart source-of-truth file."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    config = Config.load(config_path=config_path)

    configure_logging(run_id="update-rosters")
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)

    path = Path(output_path) if output_path else None
    refreshed = refresh_rosters(
        output_path=path,
        season=season,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
        force=force,
        max_age_days=1,
    )
    logger.info("Roster source-of-truth updated: %s", refreshed)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NBA Prop Intelligence v2 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("run-daily", help="Run the daily pipeline")
    daily.add_argument("--config", dest="config_path", help="Path to config file")
    daily.add_argument(
        "--auto-review",
        dest="auto_review",
        action=argparse.BooleanOptionalAction,
        help="Review yesterday's picks after the daily run",
    )
    daily.add_argument("--review-date", dest="review_date", help="Override review date (YYYY-MM-DD)")
    daily.add_argument("--review-step", dest="review_step", type=float, help="Blend grid step")
    daily.add_argument(
        "--review-update-env",
        dest="review_update_env",
        action=argparse.BooleanOptionalAction,
        help="Update MARKET_BLEND during auto-review",
    )
    daily.add_argument("--review-output-dir", dest="review_output_dir", help="Output directory for review artifacts")
    daily.add_argument("--review-picks-path", dest="review_picks_path", help="Explicit picks CSV for review")
    daily.add_argument("--review-base-delay", dest="review_base_delay", type=float, help="NBA API delay override")

    backtest = subparsers.add_parser("run-backtest", help="Run the backtest pipeline")
    backtest.add_argument("--config", dest="config_path", help="Path to config file")

    review = subparsers.add_parser("review-picks", help="Review picks and calibrate market blend")
    review.add_argument("--config", dest="config_path", help="Path to config file")
    review.add_argument("--date", dest="target_date", help="Target game date (YYYY-MM-DD)")
    review.add_argument("--picks-path", dest="picks_path", help="Path to picks CSV")
    review.add_argument("--output-dir", dest="output_dir", help="Output directory for review artifacts")
    review.add_argument("--update-env", action="store_true", help="Update MARKET_BLEND in .env")
    review.add_argument("--env-path", dest="env_path", help="Explicit .env path")
    review.add_argument("--step", dest="step", type=float, help="Blend grid step")
    review.add_argument("--base-delay", dest="base_delay", type=float, help="NBA API delay override")

    backtest_picks = subparsers.add_parser(
        "backtest-picks",
        help="Backtest historical picks and generate calibration charts",
    )
    backtest_picks.add_argument("--config", dest="config_path", help="Path to config file")
    backtest_picks.add_argument("--picks-dir", dest="picks_dir", help="Directory containing picks CSVs")
    backtest_picks.add_argument("--pattern", dest="picks_pattern", default="picks_filtered_*.csv")
    backtest_picks.add_argument(
        "--picks-path",
        dest="picks_paths",
        action="append",
        help="Explicit picks CSV path (can be repeated)",
    )
    backtest_picks.add_argument("--start-date", dest="start_date", help="Start date (YYYY-MM-DD)")
    backtest_picks.add_argument("--end-date", dest="end_date", help="End date (YYYY-MM-DD)")
    backtest_picks.add_argument("--days", dest="days", type=int, help="Lookback window in days")
    backtest_picks.add_argument("--output-dir", dest="output_dir", help="Output directory for artifacts")
    backtest_picks.add_argument("--bins", dest="bins", type=int, default=10, help="Calibration bins")
    backtest_picks.add_argument(
        "--prob-keys",
        dest="prob_keys",
        help="Comma-separated probability columns (default: probability,model_probability)",
    )
    backtest_picks.add_argument(
        "--include-pass",
        dest="include_pass",
        action="store_true",
        help="Include PASS picks in evaluation output",
    )
    backtest_picks.add_argument("--base-delay", dest="base_delay", type=float, help="NBA API delay override")
    backtest_picks.add_argument(
        "--cache-only",
        dest="cache_only",
        action="store_true",
        help="Use local game log cache only (skip NBA API fetches).",
    )

    historical_model = subparsers.add_parser(
        "backtest-historical-model",
        help="Run current model against historical props database",
    )
    historical_model.add_argument("--config", dest="config_path", help="Path to config file")
    historical_model.add_argument("--db-path", dest="db_path", help="Path to historical props db")
    historical_model.add_argument("--start-date", dest="start_date", help="Start date (YYYY-MM-DD)")
    historical_model.add_argument("--end-date", dest="end_date", help="End date (YYYY-MM-DD)")
    historical_model.add_argument("--days", dest="days", type=int, help="Lookback window in days")
    historical_model.add_argument("--output-dir", dest="output_dir", help="Output directory for artifacts")
    historical_model.add_argument("--bins", dest="bins", type=int, default=10, help="Calibration bins")
    historical_model.add_argument("--base-delay", dest="base_delay", type=float, help="NBA API delay override")

    legacy = subparsers.add_parser(
        "import-legacy-picks",
        help="Normalize legacy pick CSVs into the new format",
    )
    legacy.add_argument("--config", dest="config_path", help="Path to config file")
    legacy.add_argument("--output-dir", dest="output_dir", help="Output directory for normalized picks")
    legacy.add_argument("--output-file", dest="output_file", help="Output CSV filename")
    legacy.add_argument(
        "--path",
        dest="paths",
        action="append",
        help="Explicit legacy pick CSV path (can be repeated)",
    )

    rosters = subparsers.add_parser("update-rosters", help="Refresh roster/depth chart source-of-truth")
    rosters.add_argument("--config", dest="config_path", help="Path to config file")
    rosters.add_argument("--output", dest="output_path", help="Override output path")
    rosters.add_argument("--season", dest="season", help="Override NBA season (e.g., 2025-26)")
    rosters.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Force refresh even if roster file is recent",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-daily":
        return run_daily(
            config_path=args.config_path,
            auto_review=getattr(args, "auto_review", None),
            review_date=getattr(args, "review_date", None),
            review_step=getattr(args, "review_step", None),
            review_update_env=getattr(args, "review_update_env", None),
            review_output_dir=getattr(args, "review_output_dir", None),
            review_picks_path=getattr(args, "review_picks_path", None),
            review_base_delay=getattr(args, "review_base_delay", None),
        )
    if args.command == "run-backtest":
        return run_backtest(config_path=args.config_path)
    if args.command == "review-picks":
        return run_review(
            config_path=args.config_path,
            target_date=getattr(args, "target_date", None),
            picks_path=getattr(args, "picks_path", None),
            output_dir=getattr(args, "output_dir", None),
            update_env=getattr(args, "update_env", False),
            env_path=getattr(args, "env_path", None),
            step=getattr(args, "step", None),
            base_delay=getattr(args, "base_delay", None),
        )
    if args.command == "backtest-picks":
        return run_backtest_picks(
            config_path=getattr(args, "config_path", None),
            picks_dir=getattr(args, "picks_dir", None),
            picks_pattern=getattr(args, "picks_pattern", "picks_filtered_*.csv"),
            picks_paths=getattr(args, "picks_paths", None),
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
            days=getattr(args, "days", None),
            output_dir=getattr(args, "output_dir", None),
            bins=getattr(args, "bins", 10),
            prob_keys=getattr(args, "prob_keys", None),
            include_pass=getattr(args, "include_pass", False),
            base_delay=getattr(args, "base_delay", None),
            cache_only=getattr(args, "cache_only", False),
        )
    if args.command == "backtest-historical-model":
        return run_historical_model(
            config_path=getattr(args, "config_path", None),
            db_path=getattr(args, "db_path", None),
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
            days=getattr(args, "days", None),
            output_dir=getattr(args, "output_dir", None),
            bins=getattr(args, "bins", 10),
            base_delay=getattr(args, "base_delay", None),
        )
    if args.command == "import-legacy-picks":
        return run_import_legacy_picks(
            config_path=getattr(args, "config_path", None),
            output_dir=getattr(args, "output_dir", None),
            output_file=getattr(args, "output_file", None),
            paths=getattr(args, "paths", None),
        )
    if args.command == "update-rosters":
        return run_update_rosters(
            config_path=getattr(args, "config_path", None),
            output_path=getattr(args, "output_path", None),
            force=getattr(args, "force", False),
            season=getattr(args, "season", None),
        )

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
