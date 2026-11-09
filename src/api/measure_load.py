"""Load every component once and report time and GPU memory. Not an HTTP test.

Run from the repository root:  python -m src.api.measure_load
"""

import torch

from src.api.service import Services
from src.utils.results import save_metrics

SAMPLES = [
    "i feel so helpless right now",
    "i am so happy that i got the job",
    "i am terrified of what comes next",
]
QUESTIONS = [
    "What is the default batch size when using map with batched processing?",
    "How do I upload a whole local folder to a repo?",
    "What is the monthly price of a Hugging Face Pro subscription?",
]


def main() -> None:
    torch.cuda.reset_peak_memory_stats()
    svc = Services()
    svc.load()
    status = svc.status()
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    print("\nLoad report (seconds, GPU GB added):")
    for name, info in status["load_report"].items():
        print(f"  {name}: {info['seconds']} s, +{info['gpu_gb_added']} GB")
    print(f"GPU allocated after loading: {status['gpu_gb_now']} GB | peak so far: {peak} GB")
    print(f"Generator {status['generator']} (4-bit: {status['four_bit']}), classifier {status['classifier']}")

    print("\nClassify:")
    for text, r in zip(SAMPLES, svc.classify(SAMPLES)):
        print(f"  {text} -> {r['label']} {r['score']}")

    print("\nAsk:")
    asks = []
    for q in QUESTIONS:
        r = svc.ask(q)
        asks.append({"question": q, "refused": r["refused"], "cited": r["cited"], **r["timings"]})
        print(f"  Q: {q}")
        print(f"  A: {r['answer'][:260]!r}")
        print(f"  refused={r['refused']} cited={r['cited']} top source={r['sources'][0]['id']} "
              f"(rerank {r['sources'][0]['rerank_score']}) timings={r['timings']}")

    save_metrics("api_service_load", {
        "load_report": status["load_report"], "gpu_after_load_gb": status["gpu_gb_now"],
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "asks": asks})


if __name__ == "__main__":
    main()