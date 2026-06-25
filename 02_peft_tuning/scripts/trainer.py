"""QLoRA fine-tuning trainer.

Supports two modes:
- Dry-run: GPT-2, 1-2 batches, CPU — validates the pipeline
- Full: Mistral 7B, 4-bit NF4, LoRA adapters, GPU
"""

import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .data import prepare_dataset


class QLoRATrainer:
    def __init__(self, config: dict, device: torch.device, dry_run: bool = False):
        self.config = config
        self.device = device
        self.dry_run = dry_run
        self.output_dir = Path(config["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        model_cfg = self.config["model"]
        model_name = model_cfg["dry_run_model"] if self.dry_run else model_cfg["base_model"]

        quant_config = None
        if not self.dry_run and model_cfg.get("load_in_4bit", True):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto" if not self.dry_run else None,
            torch_dtype=torch.bfloat16 if not self.dry_run else torch.float32,
            trust_remote_code=True,
        )

        if not self.dry_run:
            model = prepare_model_for_kbit_training(model)

        return model

    def _load_tokenizer(self):
        model_name = (
            self.config["model"]["dry_run_model"]
            if self.dry_run
            else self.config["model"]["base_model"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _apply_lora(self, model):
        lora_cfg = self.config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def train(self):
        tokenizer = self._load_tokenizer()
        model = self._load_model()
        model = self._apply_lora(model)

        train_cfg = self.config["training"]
        dataset = prepare_dataset(self.config, tokenizer, dry_run=self.dry_run)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1 if self.dry_run else train_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=2 if self.dry_run else train_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=1 if self.dry_run else train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
            logging_steps=1 if self.dry_run else train_cfg.get("logging_steps", 10),
            save_steps=0 if self.dry_run else train_cfg.get("save_steps", 200),
            evaluation_strategy="no" if self.dry_run else "steps",
            eval_steps=0 if self.dry_run else train_cfg.get("save_steps", 200),
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=1 if self.dry_run else 4,
            report_to="none" if self.dry_run else ["wandb"],
            fp16=not self.dry_run and torch.cuda.is_available(),
            bf16=False,
            gradient_checkpointing=not self.dry_run,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
        )

        trainer.train()

        if not self.dry_run:
            model.save_pretrained(self.output_dir / "adapter")
            tokenizer.save_pretrained(self.output_dir / "adapter")
            print(f"Adapter saved to {self.output_dir / 'adapter'}")
        else:
            print("[DRY-RUN] Pipeline validated successfully. No adapter saved.")
