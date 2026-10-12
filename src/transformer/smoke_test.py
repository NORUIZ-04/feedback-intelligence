"""Short training run to confirm the pipeline works. Not a result.

Run from the repository root:  python -m src.transformer.smoke_test
"""

import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.loader import get_label_names, load_raw
from src.transformer.data import TextDataset, collate_fn, load_frame
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = get_label_names(load_raw(cfg["dataset"]["name"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={lab: i for i, lab in enumerate(labels)},
    ).to(device)

    train_df = load_frame("train", limit=1024)
    val_df = load_frame("validation", limit=256)
    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, t["max_length"])
    val_ds = TextDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, t["max_length"])
    collate = collate_fn(tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=64, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["fp16"] and device.type == "cuda")

    model.train()
    losses = []
    start = time.perf_counter()
    for epoch in range(2):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=t["fp16"] and device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
    seconds = time.perf_counter() - start

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    print()
    print(f"Device: {device}")
    print(f"Steps: {len(losses)} | first loss {losses[0]:.4f} | last loss {losses[-1]:.4f}")
    print(f"Time: {seconds:.1f}s | peak GPU memory: {peak_gb:.2f} GB")
    print(f"Accuracy on 256 validation examples: {correct / total:.4f} (smoke test only)")


if __name__ == "__main__":
    main()