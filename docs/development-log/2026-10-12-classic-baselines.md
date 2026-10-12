# Classic ML baselines

**Date:** 2026-10-12
**Status:** Done
**Branch:** `feat/baseline-models`

## Goal
Establish reference scores with classic models before any transformer is trained, so that every later result can be judged against measured baselines instead of assumptions.

## Setup
- Data: the cleaned splits from the data pipeline (train 15,923 / validation 2,000 / test 2,000).
- Features: TF-IDF, 1-2 word n-grams, 50,000 features, sublinear term frequency, fitted on the train split only.
- Seed: 42 everywhere.
- Metrics: accuracy, macro-F1, weighted-F1 and per-class precision, recall and F1. Macro-F1 is the headline metric because of the 9.47 class imbalance.
- Timing: training time, and prediction time per example (best of 3 runs, on already-vectorized features).
- All numbers are saved as JSON in `results/baseline_*.json`, and the tables below are generated from them.

## Files created
- `src/classic/evaluate.py`: shared metrics, timing and split loading
- `src/classic/majority.py`: majority-class floor
- `src/classic/logreg.py`: TF-IDF + Logistic Regression
- `src/classic/svm_rf.py`: TF-IDF + Linear SVM and Random Forest
- `src/classic/class_weight.py`: balanced class-weight experiment
- `src/classic/logreg_sweep.py`: Logistic Regression C sweep
- `src/classic/compare.py`: builds the comparison table and chart from saved results
- `tests/test_classic.py`: 4 tests (metric values, timing helper, formatter, table builder)
- `requirements.txt`: added `scikit-learn`, `pyarrow`
- `.env.example`: restored to `HF_TOKEN=your_huggingface_token_here`

## Commands run
```powershell
git checkout -b feat/baseline-models
pip install -r requirements.txt
python -m src.classic.majority
python -m src.classic.logreg
python -m src.classic.svm_rf
python -m src.classic.class_weight
python -m src.classic.logreg_sweep
python -m src.classic.compare
ruff check .
pytest -q
```

## Results, in the order they were measured

### 1. Majority-class floor (always predicts "joy")
| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| validation | 0.352 | 0.0868 | 0.1833 |
| test | 0.3475 | 0.086 | 0.1792 |

Any real model must beat 0.3475 accuracy and 0.086 macro-F1. Accuracy of about 0.35 looks like something, but macro-F1 of 0.086 shows the model knows nothing, which is why macro-F1 is the headline metric.

### 2. TF-IDF + Logistic Regression (default settings, C=1)
| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| validation | 0.83 | 0.7497 | 0.8189 |
| test | 0.824 | 0.7107 | 0.8107 |

| Class (test) | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.8548 | 0.9122 | 0.8826 | 581 |
| joy | 0.7667 | 0.9741 | 0.858 | 695 |
| love | 0.869 | 0.4591 | 0.6008 | 159 |
| anger | 0.8991 | 0.7127 | 0.7951 | 275 |
| fear | 0.8778 | 0.7054 | 0.7822 | 224 |
| surprise | 0.9333 | 0.2121 | 0.3457 | 66 |

Vectorize 0.4386s, train 2.5931s, inference 0.0002 ms/example.
Reading: precision is high but recall is low on the rare classes (surprise 0.93 vs 0.21), and joy is over-predicted (recall 0.97, precision 0.77). That is the classic imbalance pattern.

### 3. TF-IDF + Linear SVM (default settings)
| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| validation | 0.896 | 0.861 | 0.8949 |
| test | 0.8805 | 0.8215 | 0.8786 |

| Class (test) | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.9109 | 0.9329 | 0.9218 | 581 |
| joy | 0.8844 | 0.9353 | 0.9091 | 695 |
| love | 0.7817 | 0.6981 | 0.7375 | 159 |
| anger | 0.8889 | 0.8727 | 0.8807 | 275 |
| fear | 0.8861 | 0.7991 | 0.8404 | 224 |
| surprise | 0.6964 | 0.5909 | 0.6393 | 66 |

Train 0.2155s, inference 0.0002 ms/example.

### 4. TF-IDF + Random Forest (200 trees, default settings)
| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| validation | 0.8715 | 0.8338 | 0.869 |
| test | 0.8635 | 0.8069 | 0.8607 |

| Class (test) | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.9317 | 0.8916 | 0.9112 | 581 |
| joy | 0.8085 | 0.954 | 0.8752 | 695 |
| love | 0.8584 | 0.6101 | 0.7132 | 159 |
| anger | 0.9073 | 0.8182 | 0.8604 | 275 |
| fear | 0.8826 | 0.8393 | 0.8604 | 224 |
| surprise | 0.72 | 0.5455 | 0.6207 | 66 |

Train 7.231s, inference 0.0294 ms/example.

### 5. Class weighting experiment (`class_weight="balanced"`, test split)
| Model | Setting | Accuracy | Macro-F1 | Love recall | Surprise recall | Train s |
|---|---|---|---|---|---|---|
| LogReg | default | 0.824 | 0.7107 | 0.4591 | 0.2121 | 2.5931 |
| LogReg | balanced | 0.8685 | 0.8367 | 0.8994 | 0.8788 | 2.0666 |
| SVM | default | 0.8805 | 0.8215 | 0.6981 | 0.5909 | 0.2155 |
| SVM | balanced | 0.891 | 0.8441 | 0.8239 | 0.7273 | 0.2593 |
| RF | default | 0.8635 | 0.8069 | 0.6101 | 0.5455 | 7.231 |
| RF | balanced | 0.8645 | 0.8309 | 0.8805 | 0.7879 | 6.7663 |

Per-class F1 (test):
| Class | LogReg | LogReg bal | SVM | SVM bal | RF | RF bal |
|---|---|---|---|---|---|---|
| sadness | 0.8826 | 0.8978 | 0.9218 | 0.9292 | 0.9112 | 0.8969 |
| joy | 0.858 | 0.8891 | 0.9091 | 0.9181 | 0.8752 | 0.8798 |
| love | 0.6008 | 0.7857 | 0.7375 | 0.7774 | 0.7132 | 0.7692 |
| anger | 0.7951 | 0.8649 | 0.8807 | 0.8913 | 0.8604 | 0.8729 |
| fear | 0.7822 | 0.8532 | 0.8404 | 0.8532 | 0.8604 | 0.854 |
| surprise | 0.3457 | 0.7296 | 0.6393 | 0.6957 | 0.6207 | 0.7123 |

Class weighting raised macro-F1 for all three models. The largest gain was Logistic Regression (+0.126 macro-F1, surprise F1 0.35 to 0.73). The spread between the three models' macro-F1 shrank from 0.11 to 0.013.

### 6. Logistic Regression C sweep (selected on validation macro-F1 only)
| Class weight | C | Val accuracy | Val macro-F1 | Train s |
|---|---|---|---|---|
| none | 0.1 | 0.575 | 0.2591 | 0.7419 |
| none | 1 | 0.83 | 0.7497 | 2.3889 |
| none | 10 | 0.8835 | 0.8439 | 3.5768 |
| none | 100 | 0.886 | 0.8477 | 3.2963 |
| balanced | 0.1 | 0.768 | 0.7529 | 0.5268 |
| balanced | 1 | 0.873 | 0.8458 | 2.0636 |
| balanced | 10 | 0.8915 | 0.8592 | 2.708 |
| balanced | 100 | 0.8945 | 0.8608 | 2.9926 |

Best on validation: balanced, C=100. Test result with that setting: accuracy 0.877, macro-F1 0.8233. The default C=1 was under-fitting, and the best C is the largest value tried, so a larger C might help slightly (not tested).

### 7. Final comparison (generated by `src/classic/compare.py`)

### Baseline comparison (test split)

| Model | Accuracy | Macro-F1 | Love F1 | Surprise F1 | Train s | Inference ms/example |
|---|---|---|---|---|---|---|
| Linear SVM (balanced) | 0.891 | 0.8441 | 0.7774 | 0.6957 | 0.26 | 0.0002 |
| LogReg (balanced) | 0.8685 | 0.8367 | 0.7857 | 0.7296 | 2.07 | 0.0002 |
| Random Forest (balanced) | 0.8645 | 0.8309 | 0.7692 | 0.7123 | 6.77 | 0.0292 |
| LogReg (tuned: balanced, C=100) | 0.877 | 0.8233 | 0.7414 | 0.6565 | 2.99 | n/a |
| Linear SVM (default) | 0.8805 | 0.8215 | 0.7375 | 0.6393 | 0.22 | 0.0002 |
| Random Forest (default) | 0.8635 | 0.8069 | 0.7132 | 0.6207 | 7.23 | 0.0294 |
| LogReg (default) | 0.824 | 0.7107 | 0.6008 | 0.3457 | 2.59 | 0.0002 |
| Majority class | 0.3475 | 0.086 | 0.0 | 0.0 | 0.01 | 0.0 |

All models use the same TF-IDF features (1-2 word n-grams, 50,000 features, fitted on the train split only). Rows are sorted by test macro-F1. Inference time is not recorded for the tuned LogReg run.

![Baseline comparison](../../results/plots/baseline_comparison.png)

## Key findings
- Every model beats the majority-class floor by a wide margin (test accuracy 0.82 to 0.89 vs 0.3475; macro-F1 0.71 to 0.84 vs 0.086).
- The best classic baseline is the balanced Linear SVM: 0.891 accuracy, 0.8441 macro-F1, 0.26s to train. This is the number the transformer must beat.
- Class weighting mattered more than model choice or tuning. With weighting, the top three models are within 0.013 macro-F1, which is too close to claim a ranking given only 66 surprise examples in the test set.
- Surprise and love remain the hardest classes for every model.
- Random Forest is much more expensive than the linear models: about 26x slower to train and about 146x slower per prediction than the SVM, without being more accurate.

## Where my expectations were wrong
| Expectation | What happened |
|---|---|
| Linear SVM would land close to default Logistic Regression | SVM was far ahead (0.8215 vs 0.7107 macro-F1), mostly because the default LogReg (C=1) under-fit |
| Random Forest would do worse than Logistic Regression | Default RF beat default LogReg (0.8069 vs 0.7107 macro-F1) |
| Balanced weights might lower overall accuracy slightly | Accuracy went up for all three models |
| Gap between models is a model-quality difference | After weighting and tuning, the gap is mostly gone |

## Limitations
- The test split has 66 surprise and 159 love examples, so per-class scores for those classes are noisy. Differences of about 0.01 macro-F1 should be treated as noise.
- Only Logistic Regression was tuned. The SVM and Random Forest use default hyperparameters, so the comparison is slightly tilted.
- The tuned Logistic Regression grid stops at C=100, and the best value was at that edge.
- The choice between default and balanced weights for SVM and Random Forest was made while looking at test results, not on validation. The test numbers are a fair estimate for each configuration, but "best classic model" is a mild selection among 8 configurations.
- Inference timings measure prediction on pre-vectorized features and exclude TF-IDF vectorizing (about 0.44s for the test-sized set in one run).

## Problems faced and fixes
| Problem | Cause | Fix |
|---|---|---|
| `requirements.txt` contained `matplotlibscikit-learn`, so `pip install` failed | The file had no trailing newline, so `Add-Content` glued the new name onto the last line | Rewrote the file with one package per line using `Set-Content` |
| `fatal: a branch named 'feat/baseline-models' already exists` | An earlier run of the same command had already created the branch | Switched to the existing branch with `git checkout feat/baseline-models` |
| `ModuleNotFoundError: No module named 'sklearn'` | scikit-learn never installed because of the broken requirements line | Fixed requirements, reinstalled (scikit-learn 1.9.1) |
| `.env.example` looked empty | It had been overwritten or never saved with content | Restored with `Set-Content`, one line: `HF_TOKEN=your_huggingface_token_here` |
| ruff ISC004 in `compare.py` (long string split across three literals in a list) | Style rule on implicit string concatenation | Moved the sentence into a parenthesized `note` variable |

## Verification (real terminal output)
```
> ruff check .
All checks passed!

> pytest -q
..........                                                               [100%]
10 passed in 1.66s
```

## Reproducing this result
```powershell
pip install -r requirements.txt
python -m src.data.prepare
python -m src.classic.majority
python -m src.classic.logreg
python -m src.classic.svm_rf
python -m src.classic.class_weight
python -m src.classic.logreg_sweep
python -m src.classic.compare
```

## Next
Fine-tune DistilBERT on the same splits and compare it to the balanced Linear SVM (0.891 accuracy, 0.8441 macro-F1).