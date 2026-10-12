# Feedback Intelligence

An NLP system for customer feedback, built on Hugging Face. It combines a text classifier (classic ML baselines vs a fine-tuned transformer) with a retrieval-augmented question answering (RAG) service over company documents.

> Status: under active development. All reported metrics are measured by the code in this repository and saved to `results/`.

## Features

- **Feedback classifier**: TF-IDF baselines (Logistic Regression, SVM, Random Forest) compared against a fine-tuned DistilBERT.
- **Document Q&A (RAG)**: document chunking, embeddings, FAISS vector search and an LLM that answers with cited sources.
- **Serving**: FastAPI backend and a Gradio interface.
- **Deployment**: Docker image deployed to Hugging Face Spaces, with CI/CD on GitHub Actions.

## Tech stack

Python, PyTorch, Hugging Face Transformers / Datasets / Hub, scikit-learn, sentence-transformers, FAISS, FastAPI, Gradio, Docker, GitHub Actions, pytest, ruff.

## Quickstart (Windows PowerShell)

```powershell
git clone https://github.com/YOUR-USERNAME/feedback-intelligence.git
cd feedback-intelligence
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## Project structure

```
configs/       YAML configuration
src/           data, classic, transformer, rag, api, utils
app/           Gradio interface
tests/         unit tests
docs/          development log and design notes
results/       saved metrics from every experiment
```

## Results

Results tables (model comparison, retrieval metrics, RAG quality, API latency) will be published here as each component is completed.

## Roadmap

- [x] Project scaffold, config, logging, metrics saver, tests, linting
- [ ] Data pipeline and dataset analysis
- [ ] Classic ML baselines
- [ ] Transformer fine-tuning
- [ ] Evaluation and error analysis
- [ ] Model release on the Hugging Face Hub
- [ ] Chunking and retrieval experiments
- [ ] Embedding comparison and FAISS index
- [ ] LLM answer generation with faithfulness evaluation
- [ ] FastAPI service
- [ ] Gradio interface
- [ ] Docker and Spaces deployment
- [ ] CI/CD pipeline
- [ ] Monitoring and performance benchmarks

## Documentation

Engineering notes are in [`docs/development-log`](docs/development-log/README.md).

## License

MIT. See [LICENSE](LICENSE).