# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffold with `src/` package layout.
- YAML configuration loader, logging helper and seed helper.
- Metrics saver that stores experiment results as JSON in `results/`.
- Unit tests (pytest) and linting (ruff).
- Development log in `docs/development-log/`.
- Data pipeline: dataset loading, cleaning (label conflicts, duplicates, train/eval leakage), profiling and plots.
- Cleaned splits written to `data/processed/`; measured profile saved to `results/data_profile.json`.
- Unit tests for cleaning and profiling.

- Classic ML baselines on TF-IDF features: majority-class floor, Logistic Regression, Linear SVM, Random Forest.
- Class-weighting experiment and Logistic Regression C sweep (selected on validation).
- Baseline comparison table and chart generated from saved results; tests for evaluation helpers.
- Restored `.env.example`; added `scikit-learn` and `pyarrow` to requirements.

- DistilBERT fine-tuning (standard and class-weighted) on the cleaned emotion splits, with validation-based epoch selection.
- Token length analysis and `max_length=64` decision; dynamic-padding dataset and collate function.
- Final model comparison, confusion matrices, per-class F1 chart and error sample.
- Tests for the transformer data code and analysis helpers; pinned `transformers==5.19.0`.

- Hugging Face Hub release of the fine-tuned DistilBERT with a generated model card, load-back verification and tests.
- Corrected the published model license to reflect the dataset's educational and research use restriction; added dataset citation.
