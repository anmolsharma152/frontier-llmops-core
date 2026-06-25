"""DPO (Direct Preference Optimization) alignment using trl.DPOTrainer.

Simpler alternative to PPO — directly optimises the policy on
preference pairs without a separate reward model.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer as TRLDPOTrainer

from .preference_data import load_preference_dataset


class DPOTrainer:
    def __init__(self, config: dict, device: torch.device, dry_run: bool = False):
        self.config = config
        self.device = device
        self.dry_run = dry_run
        self.output_dir = Path("output/dpo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
            import os as _os
            _os.environ["WANDB_MODE"] = "offline"

    def train(self):
        model_name = (
            self.config["model"]["dry_run_model"]
            if self.dry_run
            else self.config["model"]["sft_model"]
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.dry_run else torch.bfloat16,
            device_map=None if self.dry_run else "auto",
        )
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.dry_run else torch.bfloat16,
            device_map=None if self.dry_run else "auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        dataset = load_preference_dataset(self.config, self.dry_run)
        max_length = self.config["dpo"]["max_length"]

        def tokenize_dpo(examples):
            return tokenizer(
                examples["chosen"],
                examples["rejected"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_dpo, batched=True)

        dpo_cfg = self.config["dpo"]
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1 if self.dry_run else dpo_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=2 if self.dry_run else dpo_cfg.get("per_device_train_batch_size", 4),
            learning_rate=dpo_cfg.get("learning_rate", 5e-6),
            logging_steps=1 if self.dry_run else 10,
            save_steps=0 if self.dry_run else 200,
            evaluation_strategy="steps" if not self.dry_run else "no",
            eval_steps=200,
            save_total_limit=2,
            report_to="none" if self.dry_run else ["wandb"],
            fp16=not self.dry_run and torch.cuda.is_available(),
        )

        trainer = TRLDPOTrainer(
            model=model,
            ref_model=model_ref,
            args=args,
            beta=dpo_cfg.get("beta", 0.1),
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation"),
            tokenizer=tokenizer,
            max_length=max_length,
        )

        trainer.train()

        if not self.dry_run:
            trainer.save_model(self.output_dir)
            print(f"DPO model saved to {self.output_dir}")
        else:
            print("[DRY-RUN] DPO pipeline validated.")
