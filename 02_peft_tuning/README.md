# Phase 2: Adaptation (PEFT / QLoRA)

## Goal

Fine-tune a pretrained base model (Mistral 7B) on domain-specific data using Parameter-Efficient Fine-Tuning (PEFT). We use QLoRA — 4-bit quantisation + Low-Rank Adapters — to fit training on consumer GPUs.

## Concepts

### LoRA (Low-Rank Adaptation)

Instead of updating all 7B parameters, LoRA injects small rank-decomposition matrices into attention layers. Only these adapters are trained, reducing memory from ~56 GB to ~16 GB for Mistral 7B.

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `r` (rank) | 8–64 | Higher = more expressivity, more memory |
| `lora_alpha` | 16–32 | Scaling factor; higher = stronger adaptation |
| `target_modules` | `q_proj`, `v_proj` | Which layers to adapt; full attention = better |

### QLoRA: 4-bit NormalFloat (NF4)

QLoRA adds double-quantization and paged optimizers to further reduce memory:
- Base model stored in 4-bit NF4 → ~4 GB for Mistral 7B
- Adapters remain in float32 for precise updates
- Enables fine-tuning on a single RTX 3090/4090

### Dry-Run Mode

When no GPU is available, `--dry-run` loads GPT-2 and runs 2 batches to validate the pipeline end-to-end.

## Running

```bash
# Dry-run on CPU (validates pipeline)
python scripts/main.py --dry-run train

# Full training (requires GPU)
python scripts/main.py train

# Inference with trained adapter
python scripts/main.py infer --checkpoint ./output/adapter --prompt "What is diabetes?"

# Merge adapter into base model
python scripts/main.py merge --adapter-path ./output/adapter --output-path ./merged-model
```

## Colab

Open `run_on_colab.ipynb` and follow the steps. It will:
1. Install dependencies
2. Download Mistral 7B in 4-bit
3. Train with LoRA
4. Save the adapter to Google Drive

## Files

| Path | Purpose |
|------|---------|
| `scripts/trainer.py` | QLoRA training loop (dry-run with GPT-2, full with Mistral 7B) |
| `scripts/data.py` | Dataset loading, tokenization, and formatting |
| `scripts/inference.py` | Generation with temperature/top-p sampling |
| `scripts/merge.py` | Merge LoRA adapter back into base model |
| `scripts/main.py` | Argparse CLI with `train`, `infer`, `merge` commands |
| `configs/defaults.yaml` | Model, LoRA, training, and data configuration |
| `run_on_colab.ipynb` | Step-by-step Colab notebook for T4 GPU training |
| `pyproject.toml` | Python dependencies (torch, transformers, peft, bitsandbytes, etc.) |

## Dependencies

```bash
pip install .
```

GPU strongly recommended for full training. Dry-run mode (`--dry-run`) works on CPU.

## References

- Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023) — "QLoRA: Efficient Finetuning of Quantized Language Models"
- PEFT docs — https://huggingface.co/docs/peft
