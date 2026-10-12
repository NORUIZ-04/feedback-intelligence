from datasets import DatasetDict, load_dataset


def load_raw(name: str) -> DatasetDict:
    """Download (or load from the local cache) a Hugging Face dataset."""
    return load_dataset(name)


def get_label_names(ds: DatasetDict) -> list[str]:
    """Return the class names in label-id order."""
    return list(ds["train"].features["label"].names)