### Dataset: `dair-ai/emotion`

| Split | Raw rows | Clean rows |
|---|---|---|
| train | 16000 | 15923 |
| validation | 2000 | 2000 |
| test | 2000 | 2000 |

**Class distribution (clean train split)**

| Label | Count | Share |
|---|---|---|
| sadness | 4661 | 29.27% |
| joy | 5340 | 33.54% |
| love | 1283 | 8.06% |
| anger | 2152 | 13.52% |
| fear | 1923 | 12.08% |
| surprise | 564 | 3.54% |

Imbalance ratio (largest / smallest class): 9.47

**Text length in words (clean splits)**

| Split | Min | Mean | Median | P95 | Max |
|---|---|---|---|---|---|
| train | 2 | 19.18 | 17.0 | 41.0 | 66 |
| validation | 2 | 18.87 | 17.0 | 40.0 | 61 |
| test | 3 | 19.15 | 17.0 | 41.0 | 61 |

**Cleaning report**

| Check | Rows |
|---|---|
| Empty texts removed (all splits) | 0 |
| Train rows with conflicting labels removed | 60 |
| Train duplicate rows removed | 1 |
| Train rows also in validation/test removed | 16 |
| Texts shared by validation and test (kept) | 3 |

![Class distribution](results/plots/class_distribution.png)

![Text length distribution](results/plots/text_length_distribution.png)
