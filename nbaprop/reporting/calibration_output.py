"""Calibration report helpers."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import logging
import math

logger = logging.getLogger(__name__)


def _outcome(row: Dict) -> Optional[int]:
    result = row.get("result")
    if result == "WIN":
        return 1
    if result == "LOSS":
        return 0
    return None


def _probability(row: Dict, key: str) -> Optional[float]:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 1:
        return None
    return value


def compute_calibration_bins(
    rows: Iterable[Dict],
    prob_key: str,
    bins: int = 10,
) -> Tuple[List[Dict], Dict]:
    if bins <= 0:
        bins = 10
    bin_edges = [i / bins for i in range(bins + 1)]
    bucket_rows = [
        {
            "lower": bin_edges[i],
            "upper": bin_edges[i + 1],
            "count": 0,
            "sum_prob": 0.0,
            "sum_outcome": 0.0,
        }
        for i in range(bins)
    ]

    for row in rows:
        outcome = _outcome(row)
        prob = _probability(row, prob_key)
        if outcome is None or prob is None:
            continue
        prob = min(1.0, max(0.0, prob))
        idx = min(int(prob * bins), bins - 1)
        bucket_rows[idx]["count"] += 1
        bucket_rows[idx]["sum_prob"] += prob
        bucket_rows[idx]["sum_outcome"] += outcome

    total = sum(bucket["count"] for bucket in bucket_rows)
    ece = 0.0
    mce = 0.0
    output = []
    for bucket in bucket_rows:
        count = bucket["count"]
        if count:
            avg_prob = bucket["sum_prob"] / count
            win_rate = bucket["sum_outcome"] / count
            gap = abs(avg_prob - win_rate)
            if total:
                ece += (count / total) * gap
            mce = max(mce, gap)
        else:
            avg_prob = None
            win_rate = None
        output.append({
            "bin_lower": bucket["lower"],
            "bin_upper": bucket["upper"],
            "count": count,
            "avg_prob": avg_prob,
            "win_rate": win_rate,
        })

    return output, {
        "samples": total,
        "ece": round(ece, 6),
        "mce": round(mce, 6),
    }


def compute_calibration_metrics(rows: Iterable[Dict], prob_key: str, bins: int = 10) -> Dict:
    outcomes = []
    probs = []
    for row in rows:
        outcome = _outcome(row)
        prob = _probability(row, prob_key)
        if outcome is None or prob is None:
            continue
        outcomes.append(outcome)
        probs.append(min(1.0, max(0.0, prob)))

    if not outcomes:
        return {
            "samples": 0,
            "brier": None,
            "log_loss": None,
            "accuracy": None,
            "ece": None,
            "mce": None,
        }

    brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(outcomes)
    eps = 1e-6
    log_loss = -sum(
        y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))
        for p, y in zip(probs, outcomes)
    ) / len(outcomes)
    accuracy = sum(1 for p, y in zip(probs, outcomes) if (p >= 0.5) == bool(y)) / len(outcomes)

    _, bin_metrics = compute_calibration_bins(rows, prob_key, bins=bins)

    return {
        "samples": len(outcomes),
        "brier": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
        "ece": bin_metrics.get("ece"),
        "mce": bin_metrics.get("mce"),
    }


def _write_bins_csv(bins: List[Dict], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0].keys()) if bins else [])
        if bins:
            writer.writeheader()
            writer.writerows(bins)


def _plot_calibration(
    bins: List[Dict],
    output_path: Path,
    title: str,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable for calibration plot: %s", exc)
        return None

    xs = [row["avg_prob"] for row in bins if row["count"]]
    ys = [row["win_rate"] for row in bins if row["count"]]
    counts = [row["count"] for row in bins]
    widths = [(row["bin_upper"] - row["bin_lower"]) for row in bins]
    lefts = [row["bin_lower"] for row in bins]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    if xs and ys:
        ax1.plot(xs, ys, marker="o", color="#1a1a2e")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Actual Win Rate")
    ax1.set_title(title)

    ax2.bar(lefts, counts, width=widths, align="edge", color="#4e79a7")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def write_calibration_report(
    rows: Iterable[Dict],
    output_dir: Path,
    prob_keys: Iterable[str],
    bins: int = 10,
    title_prefix: str = "Calibration",
) -> Dict[str, Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Dict] = {}

    for prob_key in prob_keys:
        bins_data, bin_metrics = compute_calibration_bins(rows, prob_key, bins=bins)
        metrics = compute_calibration_metrics(rows, prob_key, bins=bins)
        metrics.update({
            "prob_key": prob_key,
            "bins": bins,
        })

        key_slug = prob_key.replace(" ", "_")
        bins_path = output_dir / f"calibration_bins_{key_slug}.csv"
        if bins_data:
            _write_bins_csv(bins_data, bins_path)
        chart_path = output_dir / f"calibration_curve_{key_slug}.png"
        chart_output = _plot_calibration(
            bins_data,
            chart_path,
            f"{title_prefix} ({prob_key})",
        )

        report[prob_key] = {
            "metrics": metrics,
            "bin_metrics": bin_metrics,
            "bins_csv": str(bins_path) if bins_data else None,
            "chart": chart_output,
        }

    summary_path = output_dir / "calibration_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
