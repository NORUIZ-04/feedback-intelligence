# RAG question set v2

**Date:** 2026-10-14
**Status:** Done

## Why
The first set (56 answerable, 12 unanswerable) was too small and written in one style. The gate experiment showed 12 unanswerable questions give 6 per half, so one question moved results by 17 points.

## What changed
| Set | v1 | v2 |
|---|---|---|
| Answerable originals | 44 | 48 |
| Paraphrases | 12 | 24 |
| Answerable total | 56 | 72 |
| Unanswerable | 12 | 32 |

Unanswerable groups in v2: first batch 12, on_topic 11, control 2, adjacent 7.

## Review of the 29 new unanswerable questions
I wrote u13 to u41 without reading the documents, then checked their key terms with a whole-word search. 11 had hits. Reading the snippets:

| Question | Finding | Decision |
|---|---|---|
| u14 learning rate | `learning_rate=2e-5` in a training example | Moved to answerable (q57) |
| u15 gradient checkpointing | `gradient_checkpointing` explained | Moved to answerable (q58) |
| u17 GPT-2 vocabulary | "vocabulary size of 50,257" stated | Moved to answerable (q59) |
| u30 keep best checkpoint | `load_best_model_at_end` explained | Moved to answerable (q60) |
| u23 webhook | A `hf webhooks` section includes creating one | Removed (answered) |
| u32 capital of France | A code example's output says "Paris" | Removed (accidentally answered) |
| u18, u19, u26 | Borderline mentions (BitsAndBytesConfig example, streaming on reload, prices via a command) | Removed to keep the set strict |
| u39 GGUF, u40 Whisper | Passing mentions | Kept |

Of 41 drafted, 9 were removed, leaving 32 unanswerable (the 12 first-batch questions plus 20 new ones). q57 to q60 have gold phrases taken from the snippets and verified against the files by script. q61 to q72 are casual, lowercase, typo-style paraphrases of 12 originals.

## Limitations
- Questions with no term hit are probably, not certainly, unanswerable. The check only finds listed terms, and an answer could use other words.
- Removing borderline questions makes the unanswerable set easier than real traffic.
- All questions are still written by the assistant. The casual paraphrases are a different style, not a different author.
- Earlier results use v1 and are not directly comparable with v2 runs.
- 32 unanswerable questions give about 3 points per question, still a small sample.

## Problems faced and fixes
| Problem | Cause | Fix |
|---|---|---|
| Guessed unanswerable questions were answerable | Written without reading the documents | Whole-word term check plus snippet review |
| `unanswerable.yaml` became invalid YAML | My rewrite script used `yaml.safe_dump` inline, which appends a document-end marker | Rebuilt the file with `json.dumps` values (JSON strings are valid YAML) |
| The unanswerable file was never committed | It was created after the last commit | Rebuilt from the questions in the conversation; now committed with the set |
