# RAG retrieval experiments: corpus, questions, chunking and retrievers

**Date:** 2026-10-13
**Status:** Done
**Branch:** `feat/rag-corpus`

## Goal
Build a documentation corpus and a labelled question set, then measure which chunk size and which retriever (TF-IDF, MiniLM, BGE-small, hybrid) finds the right passage, using Recall@k and MRR. This is the retrieval half of the question-answering system; LLM answer generation comes next.

## 1. Corpus
Ten markdown pages from three Hugging Face open-source repositories, downloaded from raw GitHub files by `src/rag/download.py`.

| Document | Raw words | Characters |
|---|---|---|
| transformers_pipeline_tutorial | 2146 | 17059 |
| transformers_fine_tuning | 899 | 7830 |
| transformers_tokenizers_summary | 1270 | 8577 |
| transformers_quantization | 1024 | 10484 |
| datasets_quickstart | 2274 | 18724 |
| datasets_process | 5279 | 39531 |
| hub_upload_models | 4094 | 28774 |
| hub_model_cards | 1266 | 10343 |
| hub_cli | 14974 | 105811 |
| hub_inference | 4076 | 32249 |

Total 37,302 raw words; 35,635 after cleaning. All 10 downloads succeeded on the first try, although the paths were written from memory.

**Licenses.** The LICENSE file of each repository was downloaded to `data/docs/licenses/` and checked: datasets and huggingface_hub start with "Apache License"; the transformers file starts with a copyright header, and a search confirmed "Apache License" and "Version 2.0, January 2004" further down. All three are Apache-2.0, which requires keeping the license text, so it is committed with the documents. (Lesson from the earlier model-card mistake: licenses were checked before use.)

## 2. Cleaning
`src/rag/clean.py` removes front matter, HTML comments, MDX import lines, HTML and JSX tags, code-fence lines and link targets, and keeps headings and code text. Cleaning removed 2% to 18% of words per document.

| Finding | Evidence |
|---|---|
| `hub_cli` is long, not auto-generated | Its most repeated line appears 3 times and the opening reads as prose, so my earlier guess was wrong. It still holds about 41% of all words and 354 of 491 headings |
| First cleaner missed multi-line tags | `datasets_quickstart` kept raw `class="..."` markup; fixed with a multi-line tag regex, after which no `class="` remained |
| Known leftovers | Some words are glued where a tag was removed (for example "Thisavoids"), markdown links containing backticks survive, and code lines such as `from transformers import pipeline` repeat across documents |

## 3. Question set
56 questions in `data/eval/questions.yaml`. Each has a gold answer made of a document plus a phrase that a retrieved chunk from that document must contain. `src/rag/verify_questions.py` checks every phrase exists in its document and in no other; all 56 passed, with no duplicate IDs. q09 and q47 have a phrase that occurs twice in its document, which makes them slightly easier.

| Document | Original | Paraphrase |
|---|---|---|
| transformers_quantization | 3 | 1 |
| transformers_fine_tuning | 5 | 1 |
| transformers_pipeline_tutorial | 4 | 1 |
| transformers_tokenizers_summary | 5 | 1 |
| datasets_quickstart | 3 | 0 |
| datasets_process | 6 | 2 |
| hub_model_cards | 4 | 1 |
| hub_upload_models | 5 | 1 |
| hub_inference | 5 | 2 |
| hub_cli | 4 | 2 |
| **Total** | **44** | **12** |

The 12 paraphrases (q45 to q56) reuse the gold phrase of an original but are worded with little word overlap, as a user who had not read the docs might ask. Results are reported separately for both groups, because the gap shows how much wording overlap inflates scores.

**Limitations of the question set**
- All 56 questions were written by one author (the assistant), so the numbers measure retrieval on that question style.
- Questions were written from the first 160 to 220 characters under each heading, so almost all point at section openings, which may favour chunkers that cut at headings.
- The originals are phrased close to the source text, which inflates word-overlap retrievers.
- Gold phrases are long (up to about 100 characters), which penalises small chunks by construction.
- There are no unanswerable questions yet; they will be needed to test faithfulness of generated answers.
- Only 12 paraphrases: one question moves a paraphrase Recall by 0.083.

## 4. Chunking and gold coverage
Character chunks with 15% overlap, cut at spaces (`src/rag/chunking.py`). Gold coverage is the share of questions whose gold phrase lies wholly inside one chunk, which caps Recall for any retriever.

| Chunk size | Overlap | Chunks | Mean chars | Gold covered | Uncovered |
|---|---|---|---|---|---|
| 200 | 30 | 1423 | 192.5 | 50/56 | q18, q20, q29, q32, q40, q49 |
| 500 | 75 | 567 | 490.6 | 56/56 | none |
| 1000 | 150 | 285 | 981.1 | 56/56 | none |

At 200 characters, 6 of 56 questions (10.7%) can never be answered correctly because the gold phrase is cut across a chunk boundary.

## 5. First baseline: TF-IDF only (`src/rag/evaluate_retrieval.py`)
| Chunk | Group | n | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| 200 | original | 44 | 0.4545 | 0.6364 | 0.6591 | 0.5473 |
| 200 | paraphrase | 12 | 0.0833 | 0.1667 | 0.1667 | 0.125 |
| 200 | all | 56 | 0.375 | 0.5357 | 0.5536 | 0.4568 |
| 500 | original | 44 | 0.5455 | 0.7727 | 0.8636 | 0.6769 |
| 500 | paraphrase | 12 | 0.1667 | 0.25 | 0.25 | 0.1944 |
| 500 | all | 56 | 0.4643 | 0.6607 | 0.7321 | 0.5735 |
| 1000 | original | 44 | 0.6818 | 0.9091 | 0.9318 | 0.7936 |
| 1000 | paraphrase | 12 | 0.1667 | 0.25 | 0.3333 | 0.2111 |
| 1000 | all | 56 | 0.5714 | 0.7679 | 0.8036 | 0.6688 |

With TF-IDF, originals scored far above paraphrases (Recall@5 0.9318 vs 0.3333 at 1000 characters), so the original scores are inflated by wording overlap. Larger chunks also win partly because top-5 at 1000 characters returns up to 5,000 characters of text; the equal-budget table below corrects for that. The MRR values here differ slightly from section 6 because this first script scored only the top 5 and the later one scores the top 15 (Recall@5 is identical in both).

## 6. Embeddings, hybrid and equal-budget comparison (`src/rag/compare_retrievers.py`)
Models: `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` (BGE queries use its recommended search prefix). Hybrid is reciprocal rank fusion (constant 60) of TF-IDF and BGE-small. Chunks and queries are embedded with normalised vectors and scored by dot product.

### Retrieval comparison (56 questions: 44 original, 12 paraphrase)

| Chunk | Retriever | R@1 | R@3 | R@5 | MRR | Orig R@5 | Para R@5 | Para MRR | ms/query | Embed s |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | TF-IDF | 0.375 | 0.5357 | 0.5536 | 0.4658 | 0.6591 | 0.1667 | 0.125 | 0.03 | n/a |
| 200 | MiniLM | 0.2857 | 0.4107 | 0.5357 | 0.3865 | 0.6591 | 0.0833 | 0.0962 | 0.3 | 1.33 |
| 200 | BGE-small | 0.4286 | 0.6071 | 0.6429 | 0.5208 | 0.7955 | 0.0833 | 0.0536 | 0.72 | 1.63 |
| 200 | Hybrid (TF-IDF + BGE-small) | 0.4821 | 0.5714 | 0.6071 | 0.5341 | 0.7045 | 0.25 | 0.1833 | 0.75 | 1.63 |
| 500 | TF-IDF | 0.4643 | 0.6607 | 0.7321 | 0.5843 | 0.8636 | 0.25 | 0.213 | 0.02 | n/a |
| 500 | MiniLM | 0.4107 | 0.5179 | 0.5536 | 0.4848 | 0.6818 | 0.0833 | 0.1111 | 0.32 | 0.82 |
| 500 | BGE-small | 0.5 | 0.6607 | 0.7143 | 0.5905 | 0.7955 | 0.4167 | 0.1833 | 0.43 | 1.83 |
| 500 | Hybrid (TF-IDF + BGE-small) | 0.4821 | 0.6429 | 0.6964 | 0.5919 | 0.7955 | 0.3333 | 0.1878 | 0.45 | 1.83 |
| 1000 | TF-IDF | 0.5714 | 0.7679 | 0.8036 | 0.6787 | 0.9318 | 0.3333 | 0.2394 | 0.03 | n/a |
| 1000 | MiniLM | 0.4107 | 0.5 | 0.5714 | 0.4971 | 0.6818 | 0.1667 | 0.0865 | 0.7 | 0.45 |
| 1000 | BGE-small | 0.5 | 0.625 | 0.6964 | 0.5866 | 0.7955 | 0.3333 | 0.1925 | 0.55 | 1.86 |
| 1000 | Hybrid (TF-IDF + BGE-small) | 0.4821 | 0.6964 | 0.8214 | 0.6189 | 0.9091 | 0.5 | 0.2162 | 0.58 | 1.86 |

**Equal context budget** (Recall over all 56 questions when the retriever may return at most this many characters, so chunk sizes compare fairly):

| Retriever | Budget | 200-char chunks | 500-char chunks | 1000-char chunks |
|---|---|---|---|---|
| TF-IDF | 1000 chars | 0.5536 | 0.625 | 0.5714 |
| TF-IDF | 2500 chars | 0.625 | 0.7321 | 0.7679 |
| MiniLM | 1000 chars | 0.5357 | 0.4821 | 0.4107 |
| MiniLM | 2500 chars | 0.6607 | 0.5536 | 0.5 |
| BGE-small | 1000 chars | 0.6429 | 0.6071 | 0.5 |
| BGE-small | 2500 chars | 0.6607 | 0.7143 | 0.625 |
| Hybrid (TF-IDF + BGE-small) | 1000 chars | 0.6071 | 0.625 | 0.4821 |
| Hybrid (TF-IDF + BGE-small) | 2500 chars | 0.6607 | 0.6964 | 0.6964 |

Search is exact (NumPy dot product, or TF-IDF cosine). Hybrid is reciprocal rank fusion of TF-IDF and BGE-small. Hybrid ms/query is the sum of its two parts.

![Retrieval comparison](../../results/plots/rag_retrieval_mrr.png)

### Findings (computed from the tables, not written by hand)

- Best overall MRR: TF-IDF at 1000 characters (MRR 0.6787, Recall@5 0.8036).
- Best paraphrase MRR: TF-IDF at 1000 characters (MRR 0.2394, Recall@5 0.3333).
- Paraphrase Recall@5, TF-IDF vs the better embedding model at the same chunk size (n = 12, so one question is 0.083):
  - 200 chars: TF-IDF 0.1667, MiniLM 0.0833 (not higher)
  - 500 chars: TF-IDF 0.25, BGE-small 0.4167 (higher)
  - 1000 chars: TF-IDF 0.3333, BGE-small 0.3333 (not higher)
- Overall MRR, hybrid vs BGE-small alone:
  - 200 chars: hybrid 0.5341, BGE-small 0.5208
  - 500 chars: hybrid 0.5919, BGE-small 0.5905
  - 1000 chars: hybrid 0.6189, BGE-small 0.5866
- FAISS (IndexFlatIP, BGE-small, 500-char chunks, 567 vectors): 0.084 ms/query, top-5 identical to NumPy search for 1.0 of queries.

## Where my expectations were right or wrong
| Expectation | What happened |
|---|---|
| About 1,300 chunks at 200 characters, a few hundred at 1000 | 1,423 and 285 |
| Coverage below 100% at 200 characters | 50/56, six phrases cut |
| `hub_cli` is an auto-generated reference with many repeated lines | Wrong: repeated lines at most 3 times, real prose; it is simply long |
| Originals score above paraphrases | Right, by a large margin |
| 200 characters scores lowest | Right |
| 500 vs 1000 characters was an open question | 1000 won with TF-IDF |
| Embeddings improve paraphrase recall over TF-IDF | Only partly: BGE-small beat TF-IDF at 500 characters (0.4167 vs 0.25, 5 vs 3 of 12 questions), not at 200 or 1000. Overall MRR was best for TF-IDF at 1000 characters |
| (My own error) I counted 28 questions after batch 2 | There were 30 |
| (My own error) A unit test expected MRR 0.3333 for ranks [1, None, 3, 6] | Correct value is 0.375; the test was wrong, not the code |

## Limitations
- One question author, section-opening bias, wording overlap, long gold phrases (see section 3).
- Retrieval is judged by whether a chunk contains the gold phrase, not by whether the chunk would let an LLM answer correctly. Another chunk may answer the question and still count as a miss.
- Only three chunk sizes with a fixed 15% overlap, character-based, no heading-aware splitting.
- Only two small embedding models. Their licenses were not checked; they were run locally for evaluation and nothing from them is redistributed.
- Timings include one warm-up call but are single runs on a laptop, so small differences are noise. Hybrid timing is the sum of its parts.
- Document counts per question are uneven, and `hub_cli` is 41% of the text but has only 4 originals and 2 paraphrases.
- Exact search is used throughout; the FAISS line in the findings says whether it was available.

## Problems faced and fixes
| Problem | Cause | Fix |
|---|---|---|
| Cleaner left raw `class="..."` markup | Tag regex only matched single-line tags | Multi-line tag regex plus a stray-attribute-line rule |
| Output appeared twice after batch 3 | Pasted twice; 56 unique IDs confirmed no double append | Added a unique-ID check |
| Gold phrases touching glued words | Tag removal joins words | Gold phrases chosen from clean parts of sentences and verified by script |
| Question count miscounted by me | Arithmetic slip | Corrected to 30 at that point; script reports counts |
| Ruff ISC004 risk in drafted code | Adjacent string literals inside list literals | Caught while drafting, parenthesised before the run; ruff passed on the first run |
| Wrong expected MRR in a drafted unit test (0.3333 vs 0.375) | The rank-6 hit contributes 1/6 | Caught while drafting, corrected to 0.375 before the run; the test never failed on your machine |

## Verification (real terminal output)
```
> ruff check .
All checks passed!

> pytest -q
........................                                                 [100%]
24 passed in 7.82s
```

## Reproducing this result
```powershell
pip install -r requirements.txt
python -m src.rag.download
python -m src.rag.clean
python -m src.rag.verify_questions
python -m src.rag.coverage
python -m src.rag.evaluate_retrieval
python -m src.rag.compare_retrievers
```

## Next
Generate answers with an LLM from the retrieved chunks, with sources, and measure faithfulness and hallucination on answerable and unanswerable questions.