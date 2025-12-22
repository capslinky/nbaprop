"""Smoke test for the no-op daily run output."""

import json
from pathlib import Path

from nbaprop.cli import run_daily
from nbaprop.config import Config


def test_run_daily_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAPROP_CACHE_DIR", str(tmp_path))

    exit_code = run_daily()
    assert exit_code == 0

    runs_dir = tmp_path / "runs"
    manifest_paths = list(runs_dir.glob("manifest_*.json"))
    assert manifest_paths, "manifest file should be created"

    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert "run_id" in manifest
    assert "outputs" in manifest

    outputs = manifest["outputs"]
    assert "picks_csv" in outputs
    assert "raw_odds_path" in outputs
    assert "raw_player_logs_path" in outputs
    assert "raw_injury_report_path" in outputs

    picks_path = Path(outputs["picks_csv"])
    assert picks_path.exists()
