"""Merge LoRA adapter weights back into the base model for deployment.

Produces a single set of merged weights compatible with standard
inference engines (vLLM, TGI, llama.cpp).
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(config: dict, adapter_path: str, output_path: str) -> None:
    base_model_name = config["model"]["base_model"]
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"Merged model saved to {output_dir}")
