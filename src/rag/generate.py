"""Grounded answer generation with a local instruct model."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = (
    "You answer questions about Hugging Face documentation using ONLY the numbered "
    "passages provided. Cite the passages you used like [1] or [2]. If the passages do "
    "not contain the answer, reply exactly: I could not find this in the documents."
)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    passages = "\n\n".join(f"[{i}] ({c['doc']}) {c['text']}" for i, c in enumerate(chunks, 1))
    user = f"Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class Generator:
    def __init__(self, name: str = "Qwen/Qwen2.5-1.5B-Instruct", four_bit: bool = False):
        self.tok = AutoTokenizer.from_pretrained(name)
        if four_bit:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=config, device_map={"": 0}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")
        self.model.eval()

    def answer(self, messages: list[dict], max_new_tokens: int = 200) -> tuple[str, float]:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to("cuda")
        start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        seconds = time.perf_counter() - start
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip(), seconds