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

![Baseline comparison](results/plots/baseline_comparison.png)
