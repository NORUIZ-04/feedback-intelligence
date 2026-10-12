import json
from datetime import UTC, datetime
from pathlib import Path


def save_metrics(experiment: str, metrics: dict, results_dir: str = "results") -> Path:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": experiment,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    out = Path(results_dir) / f"{experiment}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out
