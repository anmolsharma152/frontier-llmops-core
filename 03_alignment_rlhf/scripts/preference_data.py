"""Preference dataset loading and formatting.

Loads from HuggingFace preference datasets (Anthropic HH-RLHF, UltraFeedback)
and formats them as chosen/rejected pairs for reward modelling or DPO.
"""

from datasets import load_dataset, DatasetDict


def load_preference_dataset(config: dict, dry_run: bool = False) -> DatasetDict:
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset", "Anthropic/hh-rlhf")
    max_samples = 10 if dry_run else None

    dataset = load_dataset(dataset_name, split="train")
    if max_samples:
        dataset = dataset.select(range(max_samples))

    def format_pair(example):
        if "chosen" in example and "rejected" in example:
            return {
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
        elif "chosen_response" in example:
            return {
                "chosen": example["chosen_response"],
                "rejected": example["rejected_response"],
            }
        raise KeyError(f"Unknown preference format: {list(example.keys())}")

    dataset = dataset.map(format_pair, remove_columns=[c for c in dataset.column_names if c not in ["chosen", "rejected"]])

    split = dataset.train_test_split(test_size=data_cfg.get("validation_split", 0.05), seed=42)
    return DatasetDict(train=split["train"], validation=split["test"])
