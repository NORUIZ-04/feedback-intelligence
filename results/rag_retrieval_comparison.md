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

![Retrieval comparison](results/plots/rag_retrieval_mrr.png)
