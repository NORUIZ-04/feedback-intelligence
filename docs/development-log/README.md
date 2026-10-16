# Development Log

Engineering notes written as the project is built: what was done, why, the commands, the measured results and the problems solved. Entries are named `YYYY-MM-DD-topic.md`. Use `_template.md` for new entries.

| Date | Topic | Outcome |
|---|---|---|
| 2026-10-12 | [Project setup and tooling](2026-10-12-project-setup.md) | Repo scaffold, config, logging, metrics saver, tests and linting in place |
| 2026-10-12 | [Data pipeline and dataset analysis](2026-10-12-data-pipeline.md) | Cleaned splits, dataset profile and plots saved to results |

| 2026-10-12 | [Classic ML baselines](2026-10-12-classic-baselines.md) | Majority floor, LogReg, SVM, Random Forest, class weighting and C sweep; best baseline balanced Linear SVM (0.891 accuracy, 0.8441 macro-F1) |

| 2026-10-12 | [Transformer fine-tuning](2026-10-12-transformer-finetuning.md) | DistilBERT test accuracy 0.9265, macro-F1 0.8873 (unweighted) and 0.8903 (weighted), versus 0.891 and 0.8441 for the best classic model |

| 2026-10-13 | [Hub release](2026-10-13-hub-release.md) | Model published; Hub copy scores 0.927 vs 0.9265 local; license corrected after publishing |

| 2026-10-13 | [RAG retrieval](2026-10-13-rag-retrieval.md) | 10-document corpus, 56 questions; best overall MRR: TF-IDF at 1000 characters (0.6787) |

| 2026-10-14 | [RAG question set v2](2026-10-14-rag-question-set-v2.md) | 72 answerable (48 original, 24 paraphrase), 32 unanswerable; 9 drafted questions removed after review |

| 2026-10-16 | [RAG answer generation](2026-10-16-rag-generation.md) | Phi-3.5-mini 4-bit 36/72 correct (lower bound), 18/32 refused; Qwen 1.5B 24/72, 5/32; reranker 38 to 48 of 56; score gate rejected |
