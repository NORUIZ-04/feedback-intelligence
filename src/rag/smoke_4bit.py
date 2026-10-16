"""Check that a model loads in 4-bit and answers one grounded question. Not a result.

Run from the repository root:  python -m src.rag.smoke_4bit [--model <hub name>]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.rag.generate import build_messages
from src.rag.retriever import HybridRetriever

QUESTION = "What is the default batch size when using map with batched processing?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    args = parser.parse_args()

    chunks = HybridRetriever(500).search(QUESTION, 4)
    torch.cuda.reset_peak_memory_stats()
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    start = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=config, device_map={"": 0}
    )
    load_seconds = time.perf_counter() - start
    after_load = torch.cuda.memory_allocated() / 1024**3

    prompt = tok.apply_chat_template(
        build_messages(QUESTION, chunks), tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    seconds = time.perf_counter() - start
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print()
    print("Model:", args.model)
    print("Retrieved:", [c["id"] for c in chunks])
    print("Answer:", answer)
    print(f"Load {load_seconds:.1f}s | memory after load {after_load:.2f} GB | "
          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Generation {seconds:.1f}s for {new_tokens} new tokens ({new_tokens / seconds:.1f} tokens/s)")


if __name__ == "__main__":
    main()