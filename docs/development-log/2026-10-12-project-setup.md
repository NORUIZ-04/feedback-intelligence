# Project setup and tooling

**Date:** 2026-10-12
**Status:** Done

## Goal
Create a production-style skeleton: folders, config, logging, a seed helper, a results saver, tests and linting. The results saver (`save_metrics`) is what stores every experiment's numbers in `results/` for the final README.

## What we did
1. Checked Python and Git on Windows.
2. Created the GitHub repo `feedback-intelligence` and cloned it to `D:\feedback-intelligence`.
3. Created and activated a virtual environment (`.venv`).
4. Created the folder structure and all starter files with one PowerShell script.
5. Installed `pyyaml`, `python-dotenv`, `pytest` and `ruff`.
6. Ran tests and linting, fixed one lint warning.
7. Committed and pushed to GitHub.

## Files created
- `configs/train.yaml`: seed, dataset, model and training settings
- `src/utils/config.py`: loads the YAML config
- `src/utils/logging_utils.py`: consistent logger
- `src/utils/seed.py`: sets random seeds for reproducibility
- `src/utils/results.py`: saves experiment metrics as JSON in `results/`
- `tests/test_utils.py`: tests for config loading and metric saving
- `pytest.ini`: lets tests import the `src` package
- `requirements.txt`, `.gitignore`, `.env.example`, `README.md`

## Commands run
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ruff check .
pytest -q
git add .
git commit -m "chore: scaffold project structure, tooling and config"
git push
```

## Output (real, captured from terminal)
```
> ruff check .
All checks passed!

> pytest -q
..                                                                       [100%]
2 passed in 0.05s
```

## Numbers produced
None for this step. It is infrastructure.

## Problems faced and fixes
| Problem | Cause | Fix |
|---|---|---|
| `python3` not found, `source` not recognized | Mac/Linux commands used on Windows PowerShell | Use `python` and `.venv\Scripts\Activate.ps1` |
| `ruff` warning UP017 in `results.py` | Newer Python prefers `datetime.UTC` | Replaced `timezone.utc` with `UTC` |
| Possible "No module named src" in pytest | `src` not on the import path | Added `pytest.ini` with `pythonpath = .` |

## What I learned
- Windows and Mac/Linux use different venv and activation commands.
- A saved-metrics helper from day one keeps every result reproducible.
- Linting early catches style issues before they pile up.

## Git commit
`chore: scaffold project structure, tooling and config`