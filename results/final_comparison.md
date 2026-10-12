### Final model comparison (test split)

| Model | Accuracy | Macro-F1 | Love F1 | Surprise F1 | Train s | Inference ms/example | Inference setup |
|---|---|---|---|---|---|---|---|
| DistilBERT (class-weighted) | 0.9225 | 0.8903 | 0.8415 | 0.795 | 102.56 | 0.3223 | GPU, batch 64, full model |
| DistilBERT (unweighted) | 0.9265 | 0.8873 | 0.8358 | 0.7642 | 103.65 | 0.325 | GPU, batch 64, full model |
| Linear SVM (balanced) | 0.891 | 0.8441 | 0.7774 | 0.6957 | 0.26 | 0.0002 | CPU, pre-vectorized features |
| LogReg (balanced) | 0.8685 | 0.8367 | 0.7857 | 0.7296 | 2.07 | 0.0002 | CPU, pre-vectorized features |
| Random Forest (balanced) | 0.8645 | 0.8309 | 0.7692 | 0.7123 | 6.77 | 0.0292 | CPU, pre-vectorized features |
| LogReg (tuned: balanced, C=100) | 0.877 | 0.8233 | 0.7414 | 0.6565 | 2.99 | n/a | n/a |
| Linear SVM (default) | 0.8805 | 0.8215 | 0.7375 | 0.6393 | 0.22 | 0.0002 | CPU, pre-vectorized features |
| Random Forest (default) | 0.8635 | 0.8069 | 0.7132 | 0.6207 | 7.23 | 0.0294 | CPU, pre-vectorized features |
| LogReg (default) | 0.824 | 0.7107 | 0.6008 | 0.3457 | 2.59 | 0.0002 | CPU, pre-vectorized features |
| Majority class | 0.3475 | 0.086 | 0.0 | 0.0 | 0.01 | 0.0 | CPU, pre-vectorized features |

Rows are sorted by test macro-F1. Classic models train on CPU and use the same TF-IDF features; DistilBERT trains on an RTX 4050 laptop GPU with fp16. Inference times are not comparable across the two groups: classic timings exclude TF-IDF vectorizing, transformer timings are GPU forward passes.

![Confusion matrices](results/plots/confusion_matrices.png)

![Per-class F1](results/plots/per_class_f1.png)
