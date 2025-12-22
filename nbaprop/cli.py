"""CLI entry points for the rebuild."""

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence
import argparse
import hashlib
import json
import logging
import subprocess

from nbaprop.config import Config
from nbaprop.ops.logging import configure_logging
from nbaprop.runtime.manifest import RunManifest

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


def run_daily(config_path: Optional[str] = None) -> int:
    """Run the daily pipeline end-to-end (no-op scaffold)."""
    repo_root = Path(__file__).resolve().parents[1]
    config = Config.load(config_path=config_path)

    manifest = RunManifest()
    manifest.git_sha = _get_git_sha(repo_root)
    manifest.config_hash = _hash_config(config)

    configure_logging(run_id=manifest.run_id)
    if config_path:
        logger.info("Loaded config from %s", config_path)

    runs_dir = Path(config.cache_dir) / "runs"
    manifest_path = _write_manifest(manifest, runs_dir)

    logger.info("No-op daily run completed.")
    logger.info("Run manifest written to %s", manifest_path)
    return 0


def run_backtest(config_path: Optional[str] = None) -> int:
    """Run the backtest pipeline end-to-end (no-op scaffold)."""
    configure_logging()
    logger.error("Backtest pipeline is not implemented yet.")
    if config_path:
        logger.info("Loaded config from %s", config_path)
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
