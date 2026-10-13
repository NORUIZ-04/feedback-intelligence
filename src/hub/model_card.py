"""Build the Hugging Face model card from saved result files.

Run from the repository root:  python -m src.hub.model_card
"""

import json
from pathlib import Path

REPO_ID = "BHARATHKUMAR0409/emotion-distilbert"
GITHUB = "https://github.com/NORUIZ-04/feedback-intelligence"


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]


def build_card() -> str:
    m = read("results/transformer_distilbert.json")
    w = read("results/transformer_distilbert_balanced.json")
    svm = read("results/baseline_linear_svm_balanced.json")
    t, wt, st = m["test"], w["test"], svm["test"]

    rows = []
    for lab, c in t["per_class"].items():
        rows.append(
            f"| {lab} | {c['precision']} | {c['recall']} | {c['f1']} | {c['support']} |"
        )
    per_class = "\n".join(rows)

    return f"""---
language: en
license: other
license_name: educational-and-research-use-only
library_name: transformers
base_model: distilbert-base-uncased
datasets:
- dair-ai/emotion
metrics:
- accuracy
- f1
pipeline_tag: text-classification
tags:
- emotion
- text-classification
- distilbert
model-index:
- name: emotion-distilbert
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: dair-ai/emotion
      type: dair-ai/emotion
      split: test
    metrics:
    - type: accuracy
      value: {t['accuracy']}
    - type: f1
      value: {t['macro_f1']}
      name: macro-F1
---

# emotion-distilbert

`distilbert-base-uncased` fine-tuned to classify short English text into six emotions:
sadness, joy, love, anger, fear, surprise.

Part of the project [feedback-intelligence]({GITHUB}), where every number below was
measured by code in that repository (see its development log for the full record).

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO_ID}", top_k=None)
print(clf("i feel so helpless right now"))
```

## Training data

[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), English tweets with
six emotion labels. Train rows with conflicting labels, exact duplicates, or texts also
present in validation/test were removed (77 of 16,000 rows), leaving 15,923 for training.
Validation (2,000) and test (2,000) were left unfiltered.

## Training procedure

| Setting | Value |
|---|---|
| Base model | distilbert-base-uncased (66,958,086 parameters) |
| Epochs | {m['epochs']} (best epoch by validation macro-F1: {m['best_epoch']}) |
| Batch size / learning rate | 32 / 2e-5 |
| Weight decay / warmup | 0.01 / 10% linear warmup, linear decay |
| Max length | {m['max_length']} tokens (truncates 0.157% of train) |
| Precision | fp16 mixed precision |
| Seed | 42 |
| Hardware | one RTX 4050 Laptop GPU (6 GB), {m['train_seconds']} s |
| Loss | standard cross-entropy (no class weights) |

## Evaluation (test split, 2,000 examples)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **This model** | **{t['accuracy']}** | **{t['macro_f1']}** |
| Class-weighted variant (not published) | {wt['accuracy']} | {wt['macro_f1']} |
| TF-IDF + Linear SVM, balanced (best classic baseline) | {st['accuracy']} | {st['macro_f1']} |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{per_class}

Validation: accuracy {m['validation']['accuracy']}, macro-F1 {m['validation']['macro_f1']}.

## License and intended use

This model was fine-tuned on dair-ai/emotion, whose card states: "The dataset should be used
for educational and research purposes only." Please treat this model the same way: use it for
educational and research purposes. It is not licensed for commercial use. I am not a lawyer;
if you need certainty, check the dataset terms yourself. The base model, distilbert-base-uncased,
is released under Apache-2.0.

## Citation

If you use this model, please also cite the dataset: Saravia, E., Liu, H.-C. T., Huang, Y.-H.,
Wu, J., Chen, Y.-S. (2018). CARER: Contextualized Affect Representations for Emotion Recognition.
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing,
pages 3687-3697. https://www.aclweb.org/anthology/D18-1404

## Limitations

- **Rare classes are weaker.** Surprise (3.5% of training data) has F1 {t['per_class']['surprise']['f1']}
  and recall {t['per_class']['surprise']['recall']}; love has F1 {t['per_class']['love']['f1']}.
  The most common error is joy predicted as love.
- **Label noise.** Many misclassified examples are ambiguous or arguably mislabeled.
  I believe the dataset labels come from automatic keyword-based labeling, but I did not verify this.
- **Single run.** One seed per configuration; differences of about 0.01 macro-F1 between
  configurations are within noise. The validation score was still rising at epoch 3.
- **Narrow domain.** English tweets in a specific first-person style ("i feel ..."). Performance
  on other text types (reviews, emails, other languages) was not measured.
- **Not for sensitive decisions.** Do not use it to infer a person's mental state or for
  anything with real consequences for individuals.

## Alternative

A class-weighted variant reached surprise recall {wt['per_class']['surprise']['recall']} but surprise
precision {wt['per_class']['surprise']['precision']}. It is not published here.
"""


def main() -> None:
    out = Path("models/transformer_distilbert/README.md")
    out.write_text(build_card(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()