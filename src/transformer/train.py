"""Fine-tune DistilBERT on the cleaned emotion splits.

Run from the repository root:
  python -m src.transformer.train             (standard cross-entropy)
  python -m src.transformer.train --weighted  (class-weighted cross-entropy)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.classic.evaluate import evaluate_predictions
from src.data.loader import get_label_names, load_raw
from src.transformer.data import TextDataset, collate_fn, load_frame
from src.utils.config import load_config
from src.utils.results import save_metrics
from src.utils.seed import set_seed


def predict(model, loader, device, use_amp):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def time_inference(model, loader, device, use_amp, n_examples, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(model, loader, device, use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return {
        "predict_total_ms": round(best * 1000, 2),
        "predict_ms_per_example": round(best * 1000 / n_examples, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(t["fp16"]) and device.type == "cuda"
    labels = get_label_names(load_raw(cfg["dataset"]["name"]))
    name = "transformer_distilbert_balanced" if args.weighted else "transformer_distilbert"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={lab: i for i, lab in enumerate(labels)},
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    frames = {s: load_frame(s) for s in ("train", "validation", "test")}
    datasets = {
        s: TextDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, t["max_length"])
        for s, df in frames.items()
    }
    collate = collate_fn(tokenizer.pad_token_id)
    train_dl = DataLoader(
        datasets["train"], batch_size=t["batch_size"], shuffle=True, collate_fn=collate
    )
    val_dl = DataLoader(datasets["validation"], batch_size=64, collate_fn=collate)
    test_dl = DataLoader(datasets["test"], batch_size=64, collate_fn=collate)
    y_val = np.asarray(frames["validation"]["label"])
    y_test = np.asarray(frames["test"]["label"])

    if args.weighted:
        counts = np.bincount(frames["train"]["label"], minlength=len(labels))
        weights = len(frames["train"]) / (len(labels) * counts)
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    total_steps = t["epochs"] * len(train_dl)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(t["warmup_ratio"] * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Run: {name} | device {device} | parameters {n_params:,} | steps {total_steps}")
    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["labels"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        val = evaluate_predictions(y_val, predict(model, val_dl, device, use_amp), labels)
        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train loss {row['train_loss']} | "
            f"val accuracy {row['val_accuracy']} | val macro-F1 {row['val_macro_f1']}"
        )
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = round(time.perf_counter() - start, 2)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device.type == "cuda" else 0.0

    model.load_state_dict(best_state)
    validation = evaluate_predictions(y_val, predict(model, val_dl, device, use_amp), labels)
    test = evaluate_predictions(y_test, predict(model, test_dl, device, use_amp), labels)
    inference = time_inference(model, test_dl, device, use_amp, len(y_test))

    save_dir = Path(cfg["paths"]["models_dir"]) / name
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    metrics = {
        "model": name,
        "class_weighted": args.weighted,
        "max_length": t["max_length"],
        "epochs": t["epochs"],
        "parameters": n_params,
        "best_epoch": best_epoch,
        "history": history,
        "train_seconds": train_seconds,
        "peak_gpu_gb": peak_gb,
        "validation": validation,
        "test": test,
        "inference": inference,
    }
    save_metrics(name, metrics, results_dir=cfg["paths"]["results_dir"])

    print()
    print(f"Best epoch (by validation macro-F1): {best_epoch}")
    print("| Split | Accuracy | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|")
    for split, m in (("validation", validation), ("test", test)):
        print(f"| {split} | {m['accuracy']} | {m['macro_f1']} | {m['weighted_f1']} |")
    print()
    print("Per-class (test)")
    print("| Class | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for lab, c in test["per_class"].items():
        print(f"| {lab} | {c['precision']} | {c['recall']} | {c['f1']} | {c['support']} |")
    print()
    print(
        f"Train: {train_seconds}s | peak GPU {peak_gb} GB | "
        f"inference {inference['predict_ms_per_example']} ms/example (batch 64, GPU)"
    )


if __name__ == "__main__":
    main()