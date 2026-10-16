# RAG answer generation: local LLMs, reranking, refusal and scoring

**Date:** 2026-10-16
**Status:** Done, with the limitations listed below
**Branch:** `feat/rag-generation`
**Previous entries:** corpus and retrieval `2026-10-13-rag-retrieval.md`, question set `2026-10-14-rag-question-set-v2.md`

## 1. Goal
Generate answers from retrieved documentation passages with a model that runs on an RTX 4050 laptop GPU (6141 MiB), then measure correctness, citation behaviour and refusal on questions the documents cannot answer. Try to improve answer quality with prompts, a reranker and a larger model, and record what did not work.

## 2. What was built
Pipeline: hybrid retrieval (TF-IDF plus BGE-small, reciprocal rank fusion, 500-character chunks), the top 15 candidates reranked by a cross-encoder, the top 4 passed to a local instruct model with a grounded prompt, then automatic scoring.

| Component | Choice | Notes |
|---|---|---|
| Retriever | Hybrid TF-IDF + BAAI/bge-small-en-v1.5 | 500-character chunks, 75 overlap |
| Reranker | BAAI/bge-reranker-base (MIT, 278,044,931 parameters) | Top 15 candidates in, top 4 out. Download 1.05 GB (1.11 GB reconstructed) |
| Generator A | Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0, 1,543,714,304 parameters) | fp16 on GPU |
| Generator B | microsoft/Phi-3.5-mini-instruct (MIT, 3,821,079,552 parameters) | 4-bit NF4, bitsandbytes 0.50.2, fp16 compute |
| Decoding | Greedy, up to 200 new tokens | No sampling, so reruns give the same answers |

## 3. Licenses checked before use (Hub, 2026-10-13)
| Model | License | Parameters | Used? |
|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | apache-2.0 | 1,543,714,304 | Yes |
| Qwen/Qwen2.5-3B-Instruct | other | 3,085,938,688 | No (license not reviewed) |
| Qwen/Qwen2.5-7B-Instruct | apache-2.0 | 7,615,616,512 | No (4-bit on 6 GB judged too tight, never tested) |
| microsoft/Phi-3.5-mini-instruct | mit | 3,821,079,552 | Yes |
| BAAI/bge-reranker-base | mit | 278,044,931 | Yes |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | apache-2.0 | 22,714,113 | No |

GPU memory in use before any run: 283 MiB of 6141 MiB. The licenses of the two embedding models used for retrieval were not checked (see the retrieval log).

## 4. Scoring rules (all computed by code, not by a model)
| Measure | Rule |
|---|---|
| Correct | At least 60% of the gold phrase's content words appear in the answer, and the answer is not a refusal |
| Number check | Only when the gold phrase has digits: all its numbers appear in the answer |
| Citation valid | The answer has `[n]` markers and every n points to a passage that was provided |
| Cited the gold passage | A cited passage is from the right document and contains the gold phrase |
| Retrieval hit | The gold phrase is inside one of the passages given to the model |
| Refusal (first version) | The exact sentence "I could not find this in the documents" |
| Refusal (final, tolerant) | The exact sentence anywhere, or a close variant ("could not find", "does not contain" and similar, together with a word like documents or passages) in the first sentence only |
| Hallucinated | An unanswerable question that was not refused (this does not check whether the answer is false) |

The correctness rule is a proxy. It undercounts paraphrased answers and can accept wrong answers that reuse the gold's words.

## 5. Experiments, in the order they were run

### 5.1 Smoke test (7.2)
One easy question: "What is the default batch size when using map with batched processing?" Retrieved `datasets_process-37, -60, -61, -24`. Qwen 1.5B answered "The default batch size when using map with batched processing is 1000." in 1.5 s, peak GPU 3.08 GB. It gave no citation. Not a result, only a pipeline check.

### 5.2 Unanswerable set, first review (7.3)
12 questions u01 to u12 written without reading the documents. A whole-word search of their key terms found 5 with hits (u03, u06, u07, u08, u10). Reading the snippets, all five were passing mentions:

| Question | Term found | What the corpus says |
|---|---|---|
| u03 GPU for Llama 3 70B | A100 | One list of GPU names (T4, A10G, A100) |
| u06 LoRA with PEFT | PEFT, LoRA | PEFT in one sentence, "lora" only as a search string |
| u07 export to ONNX | ONNX, optimum | ONNX in file listings, `optimum-quanto` in a table |
| u08 Rust tokenizers build | cargo, Rust | `hf_xet` is written in Rust; "cargo" is a news sentence ("the cargo for sale") |
| u10 Qwen2.5 context window | Qwen2.5 | The model name in a code example |

All 12 were kept. Substring matching also produced false positives, so the checker was changed to whole-word matching.

### 5.3 Baseline vs grounded prompt (7.4b, question set v1: 56 answerable, 12 unanswerable)
Hybrid retriever, 500 characters, top 4, Qwen 1.5B, peak GPU 3.12 GB.

| Metric | Baseline prompt | Grounded (strict) prompt |
|---|---|---|
| Retrieval hit (gold in top 4) | 0.6786 | 0.6786 |
| Correct | 0.4821 | 0.375 |
| Correct, when retrieval hit | 0.5789 | 0.5 |
| Correct, originals | 0.5455 | 0.4318 |
| Correct, paraphrases | 0.25 | 0.1667 |
| Number check passed | 0.6667 | 0.3333 |
| Abstained on answerable (wrongly) | 0.0 | 0.1964 |
| Has citation | 0.0 | 0.0893 |
| Citation valid | 0.0 | 0.0893 |
| Cited a chunk holding the gold | 0.0 | 0.0714 |
| Abstained on unanswerable (good) | 0.0 | 0.6667 |
| Hallucinated on unanswerable | 1.0 | 0.3333 |
| Mean seconds per answer | 3.0 | 1.7 |

The strict prompt cut hallucination from 100% to 33% but wrongly refused 11 of 56 answerable questions and lowered correctness. In a 10-answer eyeball sample, 5 of 7 non-refused answers looked correct, which showed the keyword scorer undercounts. Question u06 got an invented LoRA/PEFT recipe.

### 5.4 Soft prompt, citation example, top 6 (7.5a, v1 set)
The soft prompt (v2) lets the model answer whenever the passages support it, asks for a `[n]` citation after each sentence, and adds a one-shot example. Peak GPU 3.24 GB.

| Metric | baseline | strict | soft v2 | soft v2, top 6 |
|---|---|---|---|---|
| Retrieval hit | 0.6786 | 0.6786 | 0.6786 | 0.7143 |
| Correct | 0.4821 | 0.375 | 0.3214 | 0.3393 |
| Correct, when retrieval hit | 0.5789 | 0.5 | 0.4474 | 0.425 |
| Correct, originals | 0.5455 | 0.4318 | 0.3864 | 0.4091 |
| Correct, paraphrases | 0.25 | 0.1667 | 0.0833 | 0.0833 |
| Number check passed | 0.6667 | 0.3333 | 0.6667 | 0.6667 |
| Wrong abstentions (of 56) | 0 | 11 | 0 | 0 |
| ...of those, retrieval had hit | 0 | 3 | 0 | 0 |
| Has citation | 0.0 | 0.0893 | 0.875 | 0.8929 |
| Citation valid | 0.0 | 0.0893 | 0.875 | 0.8929 |
| Cited a chunk holding the gold | 0.0 | 0.0714 | 0.3929 | 0.3393 |
| Abstained on unanswerable | 0.0 | 0.6667 | 0.25 | 0.0833 |
| Hallucinated on unanswerable | 1.0 | 0.3333 | 0.75 | 0.9167 |
| Mean seconds per answer | 3.0 | 1.7 | 1.06 | 1.11 |

The soft prompt fixed wrong abstentions (11 to 0) and citations (0.0893 to 0.875), but 9 of 12 unanswerable questions were answered (u01, u03, u04, u05, u06, u07, u08, u09, u11) and correctness fell. Top 6 added two retrieval hits (0.6786 to 0.7143) without improving correctness. A 1.5B model could not both answer and refuse reliably through the prompt alone.

### 5.5 Reranker and score gate (7.5b, v1 set)
The gate refuses without calling the model when the best reranker score is below a threshold. The threshold (0.6285) was chosen on a dev half only. Paraphrases were kept in the same half as their original. Each half had 6 unanswerable questions, so one question moves a result by 17 points.

| Half | Answerable | Unanswerable | Gate keeps / refuses | Gold in hybrid top 4 | Gold in top 15 candidates | Gold in reranked top 4 |
|---|---|---|---|---|---|---|
| dev | 27 | 6 | 23 of 27 / 5 of 6 | 17/27 | 23/27 | 23/27 |
| test | 29 | 6 | 16 of 29 / 4 of 6 | 21/29 | 26/29 | 25/29 |

| Metric (soft prompt) | dev: hybrid top 4 | dev: reranked | dev: reranked + gate | test: hybrid top 4 | test: reranked | test: reranked + gate |
|---|---|---|---|---|---|---|
| Correct | 6/27 | 8/27 | 8/27 | 12/29 | 10/29 | 6/29 |
| Wrong abstentions | 0/27 | 1/27 | 5/27 | 0/29 | 0/29 | 13/29 |
| Cited the gold passage | 10/27 | 15/27 | 14/27 | 12/29 | 18/29 | 13/29 |
| Unanswerable refused | 1/6 | 1/6 | 5/6 | 2/6 | 1/6 | 4/6 |
| Unanswerable hallucinated | 5/6 | 5/6 | 1/6 | 4/6 | 5/6 | 2/6 |

Both halves combined (56 answerable, 12 unanswerable):

| Metric | Hybrid top 4 | Reranked top 4 | Reranked + gate |
|---|---|---|---|
| Gold in the 4 passages given | 38 (0.679) | 48 (0.857) | 48 |
| Gold in the top 15 candidates | 49 of 56 | | |
| Correct | 18 | 18 | 14 |
| Cited a passage with the gold | 22 | 33 | 27 |
| Wrong abstentions | 0 | 1 | 18 |
| Unanswerable refused | 3 | 2 | 9 |
| Unanswerable hallucinated | 9 | 10 | 3 |

Mean top reranker score: answerable with gold in the reranked top 4, 0.777 (n=48); answerable with gold missed, 0.129 (n=8); unanswerable, 0.281 (n=12). After gating, three unanswerable questions still got answers: u04 (dev, score 0.974), u06 (test, 0.990), u12 (test, 0.976). The gate wrongly refused 18 answerable questions: q06, q08, q15, q29, q33, q37, q41, q45, q46, q47, q48, q49, q50, q51, q53, q54, q55, q56. 11 of the 12 paraphrases were among them, and 5 of those had the gold passage in the top 4. q29 (score 0.993) was refused by the model itself, not by the gate.

### 5.6 Stability of the gate (7.5c, 200 random splits, no model run)
| Measurement | Result |
|---|---|
| Threshold chosen on dev | mean 0.249, 5th to 95th percentile 0.029 to 0.991 |
| Answerable kept on test | mean 0.743, 5th to 95th percentile 0.321 to 0.893 |
| Unanswerable refused on test | mean 0.617, 5th to 95th percentile 0.333 to 1.000 |
| AUC, originals vs unanswerable | 0.839 |
| AUC, paraphrases vs unanswerable | 0.431 (below chance) |
| AUC, all answerable vs unanswerable | 0.751 |
| Median top score | original 0.975 (n=44), paraphrase 0.033 (n=12), unanswerable 0.027 (n=12) |

The reranker measures relevance, and it scores a reworded question about as low as an unanswerable one. The gate was dropped and the reranker kept. The AUCs have no confidence intervals (n = 12 per group).

### 5.7 Question set v2 (7.5d)
48 original, 12 formal paraphrase (q45 to q56) and 12 casual paraphrase (q61 to q72) answerable questions plus q57 to q60 (originals), 72 answerable in total, and 32 unanswerable (12 first batch, 11 on-topic, 2 control, 7 adjacent). Of 41 drafted unanswerable questions, 9 were removed because the corpus answers or partly answers them (u14, u15, u17, u18, u19, u23, u26, u30, u32), and 4 of them (u14, u15, u17, u30) became answerable questions q57 to q60. Details are in `2026-10-14-rag-question-set-v2.md`. q57 to q60 are easier than the rest, because their gold phrases came from snippets read just before. Remaining flagged questions (u03, u06, u07, u08, u10, u39, u40) are passing mentions.

### 5.8 Qwen 1.5B on question set v2 (7.6a)
Reranked top 4 of 15, soft prompt, peak GPU 4.22 GB.

| Group | n | Gold in top 4 | Correct | Number check | Has citation | Cited the gold passage | Wrongly abstained |
|---|---|---|---|---|---|---|---|
| original | 48 | 45 | 19 | 3/4 | 43 | 32 | 1 |
| paraphrase_formal | 12 | 5 | 1 | 0/1 | 10 | 3 | 0 |
| paraphrase_casual | 12 | 8 | 4 | 1/1 | 11 | 5 | 0 |
| all | 72 | 58 | 24 | 4/6 | 64 | 40 | 1 |

| Unanswerable group | n | Refused | Answered anyway |
|---|---|---|---|
| adjacent | 7 | 0 | 7 |
| control | 2 | 2 | 0 |
| first_batch | 12 | 2 | 10 |
| on_topic | 11 | 0 | 11 |
| all | 32 | 4 | 28 |

Per question: retrieval 10.614 ms, reranking 98.898 ms, generation 0.928 s.

### 5.9 Phi-3.5-mini 4-bit load test (7.6b)
| Check | Value |
|---|---|
| Download | 6.59 GB transferred, 7.64 GB reconstructed, 4 min 09 s fetch |
| Load time (including download) | 279.6 s |
| Memory after load | 2.28 GB (peak 2.64 GB) |
| Generation | 3.5 s for 53 tokens (15.2 tokens/s) |
| Answer | "...is 1000, as mentioned in Passage [1]" with a quote |

Warnings ignored: `triton not found` and a `rope_parameters` notice. Loading, quantizing and generating all worked.

### 5.10 Phi-3.5-mini on question set v2 (7.6c)
Same retrieval, same prompt, peak GPU 4.01 GB.

| Group | n | Gold in top 4 | Correct | Number check | Has citation | Cited the gold passage | Wrongly abstained |
|---|---|---|---|---|---|---|---|
| original | 48 | 45 | 30 | 2/4 | 43 | 29 | 0 |
| paraphrase_formal | 12 | 5 | 2 | 0/1 | 11 | 4 | 0 |
| paraphrase_casual | 12 | 8 | 4 | 1/1 | 10 | 7 | 0 |
| all | 72 | 58 | 36 | 3/6 | 64 | 40 | 0 |

| Unanswerable group | n | Refused (exact sentence) | Answered anyway |
|---|---|---|---|
| adjacent | 7 | 5 | 2 |
| control | 2 | 1 | 1 |
| first_batch | 12 | 7 | 5 |
| on_topic | 11 | 2 | 9 |
| all | 32 | 15 | 17 |

Per question: retrieval 11.34 ms, reranking 101.117 ms, generation 3.86 s.

### 5.11 Qwen 1.5B vs Phi-3.5-mini, side by side (question set v2)
| Metric | Qwen 1.5B | Phi-3.5-mini 4-bit |
|---|---|---|
| Gold in top 4 | 58/72 | 58/72 |
| Correct (keyword scorer) | 24/72 | 36/72 |
| Correct, originals | 19/48 | 30/48 |
| Cited the gold passage | 40 | 40 |
| Wrong abstentions | 1 | 0 |
| Unanswerable refused (exact sentence) | 4/32 | 15/32 |
| Unanswerable refused (tolerant rule) | 5/32 | 18/32 |
| Seconds per answer | 0.928 | 3.86 |
| Peak GPU memory | 4.22 GB | 4.01 GB |

Retrieval is identical, so the gain is the model's. Phi is about 4x slower per answer. The paraphrase gains are small (5 versus 6 correct of 24) and within noise.

### 5.12 Refusal scorer fixes (7.6d, 7.6e, re-scoring saved answers, no model run)
The exact sentence undercounted refusals: Phi u34 said "I could not find this information in the documents provided" and was scored as answered. Re-scoring both saved runs:

| Rule | Qwen refused | Phi refused | Phi wrong abstentions |
|---|---|---|---|
| Exact sentence | 4/32 | 15/32 | 0 |
| Loose (phrase anywhere in the answer) | 5/32 | 21/32 | 1 |
| Tight (first sentence only) | 5/32 | 18/32 | 0 |

The loose rule wrongly counted three Phi answers (u09, u29 and the answerable q26, whose refusal-like phrase came later in an answer that attempted a reply). The tight rule changed only five answers, all real refusals when read in full: Qwen u13; Phi u13, u16, u34, and the second control. Refusals by group, tight rule:

| Group | n | Qwen | Phi |
|---|---|---|---|
| adjacent | 7 | 0 -> 0 | 5 -> 5 |
| control | 2 | 2 -> 2 | 1 -> 2 |
| first_batch | 12 | 2 -> 2 | 7 -> 7 |
| on_topic | 11 | 0 -> 1 | 2 -> 4 |

Correct counts did not change (Qwen 24/72, Phi 36/72). The sanity check (stored scores against the old rule) showed 0 mismatches for both runs. Phi u16 is a mixed case: it says the passages lack the answer, suggests `add_special_tokens` from its own knowledge, then says it could not find the answer. The tight rule counts it as a refusal.

### 5.13 Human check (7.7, 20 Phi answers labelled by the project owner)
The sample was stratified: 8 originals, 2 formal and 2 casual paraphrases, 4 on-topic, 2 first-batch and 2 adjacent unanswerable questions. The scorer's verdict was hidden from the labeller.

| Answerable | Human: correct | partial | wrong |
|---|---|---|---|
| Scorer says correct (4) | 4 | 0 | 0 |
| Scorer says not correct (8) | 4 | 1 | 3 |
| Scorer says abstained (0) | 0 | 0 | 0 |

| Unanswerable | Human: refused | mixed | answered |
|---|---|---|---|
| Scorer says refused (4) | 3 | 1 | 0 |
| Scorer says answered (4) | 0 | 1 | 3 |

Agreement (strict, partial counted as not correct, mixed as not refused): answerable 8/12, unanswerable 7/8. Human labels: 8 correct, 1 partial, 3 wrong. Disagreements: q18 (overlap 0.556), q62 (0.5), q16 (0.167) and q41 (0.4) were correct answers the scorer called not correct; u16 was refuse-then-guess. All four answers the scorer called correct were confirmed. The scorer's errors were one-directional: it undercounts. Phi's 36 of 72 is therefore a lower bound. Because the sample is stratified and small, it cannot be turned into a corrected accuracy.

### 5.14 Prompt example leak
The one-shot example's fake passage ("(demo) The tool exposes a --verbose flag...") appeared in answer q41 as a cited source. A search of the saved results found 0 lines mentioning "(demo)" for Qwen and 1 for Phi, so probably one answer in 104 (other fields were not checked). It was left unfixed: a rerun would change every number.

## 6. Key findings
- A reranker over the hybrid retriever raised gold-passage retrieval from 38 to 48 of 56 and cited-gold answers from 22 to 33 (question set v1).
- Retrieval caps the result: on question set v2 the gold phrase reaches the model for 58 of 72 answerable questions (45 of 48 originals, 5 of 12 formal paraphrases, 8 of 12 casual).
- The prompt cannot make a 1.5B model both answer and refuse: strict refused 11 of 56 answerable questions, soft refused 0 but answered 9 of 12 unanswerable ones.
- Phi-3.5-mini 4-bit answers more correctly (36 against 24 of 72) and refuses more (18 against 5 of 32), with 0 wrong abstentions, at about 4x the latency and similar memory.
- A reranker-score threshold gate is unstable (threshold 0.029 to 0.991 across splits) and confuses reworded questions with unanswerable ones (paraphrase AUC 0.431).
- On-topic unanswerable questions are the hard case: Phi refused 4 of 11 and Qwen 1 of 11 under the tolerant rule.
- Citation of the gold passage did not improve with the larger model (40 and 40), so it is limited by retrieval and the prompt.

## 7. Where my expectations were right or wrong
| Expectation | What happened |
|---|---|
| Baseline prompt would produce no citations and hallucinate on unanswerable questions | Right: 0.0 citations, hallucination 1.0 |
| Grounded prompt would refuse more | Right (0.6667), but it also lowered correctness (0.4821 to 0.375), which I did not expect |
| Soft prompt would raise correctness | Wrong: 0.375 to 0.3214 |
| Soft prompt would raise hallucination | Right, but steeper than expected (0.3333 to 0.75, and 0.9167 with top 6) |
| Top 6 would help retrieval a little | Right: 0.6786 to 0.7143 |
| A tuned gate would cut hallucination at a small cost | Wrong: it wrongly refused 18 of 56 answerable questions |
| Casual paraphrases would retrieve worse than formal ones | Wrong: 8 of 12 against 5 of 12 |
| Phi would load in 4-bit with 2.5 to 3 GB | Right: 2.28 GB at load, 2.64 GB peak |
| Phi would refuse more than Qwen without hurting answerable questions | Right: 15 against 4 (exact rule), 0 wrong abstentions |
| A "no citation" condition would identify real attempts | Wrong: real refusals also carry citations |
| Several guessed unanswerable questions were truly unanswerable | Wrong: 9 of 41 were answered or partly answered by the corpus |

## 8. Limitations
- One run per configuration with greedy decoding. 32 unanswerable questions is about 3 points per question; the paraphrase groups have 12 questions each, so differences of 2 to 3 questions are noise.
- All 104 questions were written by the assistant, including the casual paraphrases. The human check covers only 20 answers and was stratified.
- The keyword correctness rule is a lower bound; no corrected accuracy exists.
- Unanswerable questions without a term hit are probably, not certainly, unanswerable. Removing borderline ones made the set easier than real traffic.
- The one-shot example leaks into one answer (q41).
- Some Phi answers run to the 200-token limit and end mid-sentence, with glued words such as "and[3]". Not investigated.
- v1 results (56 plus 12 questions) and v2 results (72 plus 32) are not directly comparable.
- The 4-bit path was verified on one GPU; the `triton not found` warning was ignored.
- The index is rebuilt at every start; fine for evaluation, not for a service.
- Reranker latency was measured, but a larger candidate pool (30) was not tested, and 7 of 56 gold passages were missing from the top 15 on the v1 set.

## 9. Problems faced and fixes
| Problem | Cause | Fix |
|---|---|---|
| First generation run looked frozen | `2>$null` hid the model download progress | Stopped using it on first runs |
| Inline Python with escaped quotes failed | PowerShell quoting inside an f-string | Wrote scripts to temporary files |
| `unanswerable.yaml` became invalid YAML | `yaml.safe_dump` inline appended a document-end marker | Rebuilt the file with `json.dumps` values |
| Loose refusal rule over-counted | The phrase matched anywhere in an answer | First-sentence rule plus tests |
| Ruff PLC0206 and SIM401 style warnings | Loop and `if` style rules | `list(summaries)` and `.get()` (the second was auto-fixed) |
| `ruff` not recognized in a new terminal | Virtual environment not activated | Activated `.venv` |
| One patch reported NOT FOUND | The line had already been changed by an auto-fix | Verified with a search; no change needed |
| `bitsandbytes` missing from requirements | Installed by hand, never pinned | Pinned in this wrap-up |

## 10. Files
- `src/rag/retriever.py`, `generate.py`, `smoke.py`, `smoke_4bit.py`: hybrid retriever, generator (fp16 or 4-bit), smoke tests
- `src/rag/scoring.py`, `rescore.py`, `make_label_sheet.py`, `compare_labels.py`: scoring and human-check tools
- `src/rag/evaluate_generation.py`, `evaluate_prompts.py`, `evaluate_rerank.py`, `analyze_gate.py`, `evaluate_v2.py`: experiments
- `src/rag/verify_unanswerable.py`: whole-word check of the unanswerable set
- `data/eval/unanswerable.yaml`, `data/eval/questions.yaml` (72 answerable)
- `tests/test_scoring.py`, `tests/test_rag.py`
- Results: `results/rag_generation_1p5b.json`, `rag_prompt_variants_1p5b.json`, `rag_rerank_gate.json`, `rag_gate_stability.json`, `rag_v2_1p5b.json`, `rag_v2_phi35_4bit.json`, `rag_v2_rescored.json`, `human_labels.csv`, `human_label_comparison.json`

## 11. Commands run
```powershell
python -m src.rag.smoke
python -m src.rag.verify_unanswerable
python -m src.rag.evaluate_generation
python -m src.rag.evaluate_prompts
python -m src.rag.evaluate_rerank
python -m src.rag.analyze_gate
python -m src.rag.evaluate_v2 --tag 1p5b
python -m src.rag.smoke_4bit
python -m src.rag.evaluate_v2 --tag phi35_4bit --model microsoft/Phi-3.5-mini-instruct --4bit
python -m src.rag.rescore
python -m src.rag.make_label_sheet
python -m src.rag.compare_labels
```

## 12. Tests
The suite grew from 24 (after retrieval) to 30 (scoring), 32 (question set), 34 (tolerant refusal), 36 (first-sentence rule) and then by 3 in this wrap-up (rerank split and threshold helpers, first-sentence split). The final count is in the verification output below.

## 13. Verification (real terminal output)
```
> ruff check .
All checks passed!

> pytest -q
.......................................                                  [100%]
39 passed in 8.00s
```

## 14. Next
Build the API and the interface: serve the retriever, reranker and generator behind FastAPI with the index saved to disk, then a Gradio UI, Docker and the Hugging Face Space deployment.