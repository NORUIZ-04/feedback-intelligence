import pandas as pd
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Tokenizes lazily-free: tokenizes everything once, pads per batch later."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_frame(split: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(f"data/processed/{split}.parquet")
    return df.sample(n=limit, random_state=42) if limit else df


def collate_fn(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[i, :n] = torch.tensor(item["input_ids"])
            mask[i, :n] = 1
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate