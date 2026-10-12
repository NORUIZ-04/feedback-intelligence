import json

from src.utils.config import load_config
from src.utils.results import save_metrics


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert "dataset" in cfg


def test_save_metrics(tmp_path):
    out = save_metrics("demo", {"accuracy": 0.9}, results_dir=str(tmp_path))
    data = json.loads(out.read_text())
    assert data["metrics"]["accuracy"] == 0.9