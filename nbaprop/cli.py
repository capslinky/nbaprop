"""CLI entry points for the rebuild."""

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence
import argparse
import hashlib
import json
import logging
import os
import subprocess

from nbaprop.config import Config
from nbaprop.ops.logging import configure_logging
from nbaprop.runtime.manifest import RunManifest
from nbaprop.storage import FileCache, JsonStorage
from nbaprop.ingestion.odds import fetch_odds_snapshot
from nbaprop.ingestion.nba_stats import fetch_player_logs
from nbaprop.ingestion.injuries import fetch_injury_report
from nbaprop.normalization import normalize_raw_data
from nbaprop.features.pipeline import build_features
from nbaprop.models.baseline import BaselineModel
from nbaprop.reporting.csv_output import write_picks_csv

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
            os.environ.setdefault("NBA_API_DELAY", str(config.nba_api_delay))
            os.environ.setdefault("NBAPROP_CACHE_DIR", config.cache_dir)
            return str(env_path)
        except Exception:
            continue
    return None


def run_daily(config_path: Optional[str] = None) -> int:
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

    cache_dir = Path(config.cache_dir)
    runs_dir = cache_dir / "runs"
    cache = FileCache(cache_dir / "snapshots")
    storage = JsonStorage(str(cache_dir / "storage"))
    manifest_path = _write_manifest(manifest, runs_dir)
    snapshot = fetch_odds_snapshot(
        cache,
        ttl_seconds=60,
        api_key=config.odds_api_key,
        max_events=config.odds_max_events,
    )
    manifest.outputs["odds_snapshot"] = f"{snapshot.get('source')}:{snapshot.get('fetched_at')}"
    manifest.outputs["raw_odds_path"] = storage.write_table("raw_odds_snapshot", [snapshot])

    players = ["Example Player"]
    player_logs = fetch_player_logs(
        players,
        cache,
        ttl_seconds=300,
        base_delay=config.nba_api_delay,
        cache_dir=config.cache_dir,
    )
    manifest.outputs["raw_player_logs_path"] = storage.write_table("raw_player_logs", player_logs)

    injury_report = fetch_injury_report(cache, ttl_seconds=300)
    manifest.outputs["raw_injury_report_path"] = storage.write_table("raw_injury_report", [injury_report])

    normalized = normalize_raw_data(snapshot, player_logs, injury_report)
    normalized["prop_features"] = build_features(normalized.get("props", []))
    model = BaselineModel()
    normalized["picks"] = model.predict(normalized["prop_features"])
    for name, rows in normalized.items():
        manifest.outputs[f"normalized_{name}_path"] = storage.write_table(f"normalized_{name}", rows)

    picks_path = runs_dir / f"picks_{manifest.run_id}.csv"
    write_picks_csv([], str(picks_path))
    manifest.outputs["picks_csv"] = str(picks_path)

    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("No-op daily run completed.")
    logger.info("Run manifest written to %s", manifest_path)
    return 0


def run_backtest(config_path: Optional[str] = None) -> int:
    """Run the backtest pipeline end-to-end (no-op scaffold)."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = _try_load_dotenv(repo_root)
    configure_logging()
    logger.error("Backtest pipeline is not implemented yet.")
    if config_path:
        logger.info("Loaded config from %s", config_path)
    elif dotenv_path:
        logger.info("Loaded config from %s", dotenv_path)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NBA Prop Intelligence v2 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("run-daily", help="Run the daily pipeline")
    daily.add_argument("--config", dest="config_path", help="Path to config file")

    backtest = subparsers.add_parser("run-backtest", help="Run the backtest pipeline")
    backtest.add_argument("--config", dest="config_path", help="Path to config file")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-daily":
        return run_daily(config_path=args.config_path)
    if args.command == "run-backtest":
        return run_backtest(config_path=args.config_path)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
