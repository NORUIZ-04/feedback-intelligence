"""Prepare the dataset: load, clean, profile, plot and save.

Run from the repository root:  python -m src.data.prepare
"""

from pathlib import Path

from src.data.analysis import profile_splits, render_markdown
from src.data.cleaning import build_clean_splits
from src.data.loader import get_label_names, load_raw
from src.data.plots import plot_class_distribution, plot_length_distribution
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.results import save_metrics
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    dataset_name = cfg["dataset"]["name"]
    results_dir = Path(cfg["paths"]["results_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s", dataset_name)
    ds = load_raw(dataset_name)
    labels = get_label_names(ds)
    raw = {split: ds[split].to_pandas() for split in ds}

    raw_profile = profile_splits(raw, labels)
    clean, cleaning_stats = build_clean_splits(raw)
    clean_profile = profile_splits(clean, labels)

    for split, df in clean.items():
        df.to_parquet(processed_dir / f"{split}.parquet", index=False)
        logger.info("Saved %s rows to %s", len(df), processed_dir / f"{split}.parquet")

    plot_dir = results_dir / "plots"
    plot_class_distribution(clean_profile["train"], plot_dir / "class_distribution.png")
    plot_length_distribution(clean["train"], plot_dir / "text_length_distribution.png")

    metrics = {
        "dataset": dataset_name,
        "labels": labels,
        "raw": raw_profile,
        "cleaning": cleaning_stats,
        "clean": clean_profile,
    }
    out = save_metrics("data_profile", metrics, results_dir=str(results_dir))
    logger.info("Saved metrics to %s", out)

    markdown = render_markdown(raw_profile, clean_profile, cleaning_stats, labels, dataset_name)
    (results_dir / "data_profile.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)


if __name__ == "__main__":
    main()