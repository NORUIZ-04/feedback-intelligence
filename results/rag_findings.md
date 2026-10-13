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
