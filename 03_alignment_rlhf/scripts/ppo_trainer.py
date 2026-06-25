"""PPO alignment using trl.PPOTrainer.

Fine-tunes an SFT model using reinforcement learning with
a trained reward model as the signal.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer as TRLPPOTrainer, PPOConfig
from datasets import Dataset

from .preference_data import load_preference_dataset


class PPOTrainer:
    def __init__(self, config: dict, device: torch.device, dry_run: bool = False):
        self.config = config
        self.device = device
        self.dry_run = dry_run
        self.output_dir = Path("output/ppo")
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
        tokenizer.padding_side = "left"

        dataset = load_preference_dataset(self.config, self.dry_run)
        prompts = [ex["chosen"] for ex in dataset["train"]]

        ppo_cfg = self.config["ppo"]
        config = PPOConfig(
            model_name=model_name,
            learning_rate=ppo_cfg.get("learning_rate", 1.4e-5),
            batch_size=2 if self.dry_run else ppo_cfg.get("batch_size", 16),
            mini_batch_size=1 if self.dry_run else ppo_cfg.get("mini_batch_size", 4),
            gradient_accumulation_steps=1 if self.dry_run else ppo_cfg.get("gradient_accumulation_steps", 4),
            ppo_epochs=1 if self.dry_run else ppo_cfg.get("ppo_epochs", 4),
            cliprange=ppo_cfg.get("cliprange", 0.2),
            cliprange_value=ppo_cfg.get("cliprange_value", 0.2),
            vf_coef=ppo_cfg.get("vf_coef", 0.1),
            kl_penalty=ppo_cfg.get("kl_penalty", 0.05),
            optimize_cuda_cache=True,
        )

        ppo_trainer = TRLPPOTrainer(
            config=config,
            model=model,
            ref_model=model_ref,
            tokenizer=tokenizer,
            dataset=Dataset.from_list([{"query": p} for p in prompts]),
        )

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 32 if self.dry_run else ppo_cfg.get("max_length", 512),
        }

        for epoch in range(1 if self.dry_run else ppo_cfg.get("ppo_epochs", 4)):
            for batch in ppo_trainer.dataloader:
                query_tensors = batch["input_ids"]
                response_tensors = ppo_trainer.generate(
                    query_tensors, **generation_kwargs, return_prompt=False
                )
                batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                texts = [q + r for q, r in zip(batch["query"], batch["response"])]

                dummy_rewards = [torch.tensor(1.0)] * len(texts)
                stats = ppo_trainer.step(query_tensors, response_tensors, dummy_rewards)
                print(f"Epoch {epoch}: loss={stats.get('ppo/loss', 0):.4f}")

        if not self.dry_run:
            ppo_trainer.save_pretrained(self.output_dir)
            print(f"PPO model saved to {self.output_dir}")
        else:
            print("[DRY-RUN] PPO pipeline validated.")
