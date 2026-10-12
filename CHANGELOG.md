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
