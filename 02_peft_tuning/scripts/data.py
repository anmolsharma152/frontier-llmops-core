"""Data preparation for QLoRA fine-tuning.

Loads from HuggingFace datasets or Phase-1 JSONL output,
tokenizes and formats for causal language modelling.
"""

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer


def prepare_dataset(config: dict, tokenizer: PreTrainedTokenizer, dry_run: bool = False) -> DatasetDict:
    data_cfg = config["data"]
    max_length = config["training"]["max_seq_length"]

    if data_cfg.get("dataset_path"):
        import json
        records = []
        with open(data_cfg["dataset_path"]) as f:
            for line in f:
                records.append(json.loads(line))
        dataset = DatasetDict.from_dict({
            "train": records[:10] if dry_run else records,
        })
    else:
        hf_name = data_cfg.get("hf_dataset", "databricks/databricks-dolly-15k")
        split = f"train[:{10 if dry_run else '100%'}]"
        dataset = load_dataset(hf_name, split=split)
        if isinstance(dataset, list):
            dataset = DatasetDict({"train": dataset})

    def format_prompt(example):
        text = example.get("response", example.get("completion", example.get("text", "")))
        prompt = example.get("prompt", example.get("instruction", ""))
        full_text = f"### Instruction\n{prompt}\n\n### Response\n{text}" if prompt else text
        return {"text": full_text}

    dataset = dataset.map(format_prompt)

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, batched=False, remove_columns=["text"])

    if dry_run:
        dataset = dataset.select(range(min(10, len(dataset))))

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=data_cfg.get("validation_split", 0.05), seed=42)
        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })

    return dataset
