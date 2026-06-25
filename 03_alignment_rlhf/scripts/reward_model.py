"""Reward model training.

Trains a binary classifier on chosen/rejected preference pairs
using a pretrained LM with a sequence classification head.
"""

import torch
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import DatasetDict

from .preference_data import load_preference_dataset


class RewardModelTrainer:
    def __init__(self, config: dict, device: torch.device, dry_run: bool = False):
        self.config = config
        self.device = device
        self.dry_run = dry_run
        self.output_dir = Path("output/reward_model")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _tokenize(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        max_length = self.config["reward"]["max_length"]

        def tokenize_pair(examples):
            chosen = tokenizer(
                examples["chosen"], truncation=True, max_length=max_length, padding="max_length"
            )
            rejected = tokenizer(
                examples["rejected"], truncation=True, max_length=max_length, padding="max_length"
            )
            chosen["labels"] = [1] * len(examples["chosen"])
            rejected["labels"] = [0] * len(examples["rejected"])

            combined = {k: chosen[k] + rejected[k] for k in chosen}
            return combined

        tokenized = dataset.map(tokenize_pair, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    def train(self):
        model_name = (
            self.config["model"]["dry_run_model"]
            if self.dry_run
            else self.config["model"]["reward_model"]
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_preference_dataset(self.config, self.dry_run)
        dataset = self._tokenize(dataset, tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.float32 if self.dry_run else torch.bfloat16,
            device_map=None if self.dry_run else "auto",
        )

        rm_cfg = self.config["reward"]
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1 if self.dry_run else rm_cfg.get("num_train_epochs", 2),
            per_device_train_batch_size=2 if self.dry_run else rm_cfg.get("per_device_train_batch_size", 4),
            learning_rate=rm_cfg.get("learning_rate", 1e-5),
            logging_steps=1 if self.dry_run else 10,
            save_steps=0 if self.dry_run else 200,
            evaluation_strategy="steps" if not self.dry_run else "no",
            eval_steps=200,
            save_total_limit=2,
            report_to="none" if self.dry_run else ["wandb"],
            fp16=not self.dry_run and torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
        )

        trainer.train()

        if not self.dry_run:
            model.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            print(f"Reward model saved to {self.output_dir}")
        else:
            print("[DRY-RUN] Reward model pipeline validated.")
