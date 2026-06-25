# Phase 3: RLHF Alignment

## Goal

Align a supervised fine-tuned (SFT) model with human preferences using Reinforcement Learning from Human Feedback (RLHF). Provides three approaches: reward modelling + PPO, and the simpler DPO alternative.

## Concepts

### RLHF Pipeline (InstructGPT style)

```
SFT Model → Sample responses → Reward model scores → PPO update policy
```

1. **Reward model**: A binary classifier trained on chosen/rejected pairs. Learns to predict which response humans prefer.
2. **PPO (Proximal Policy Optimisation)**: Uses the reward model's scores as a reward signal to update the policy. A KL penalty prevents the model from diverging too far from the SFT model.
3. **DPO (Direct Preference Optimization)**: Reformulates RLHF as a simple classification loss on preference pairs — no separate reward model or PPO loop needed. Often more stable.

### Key Challenges

| Challenge | Mitigation |
|-----------|------------|
| Reward hacking | KL penalty, reward normalisation |
| Preference noise | Crowd-sourced datasets, majority voting |
| Distribution collapse | Keeping a reference model for KL divergence |

## Running

```bash
# Dry-run on CPU (validates pipeline with GPT-2)
python scripts/main.py --dry-run reward
python scripts/main.py --dry-run ppo
python scripts/main.py --dry-run dpo

# Full training (requires GPU)
python scripts/main.py reward
python scripts/main.py ppo
python scripts/main.py dpo
```

## Colab

Open `run_on_colab.ipynb` for a step-by-step Colab setup.

## Files

| Path | Purpose |
|------|---------|
| `scripts/reward_model.py` | Binary classifier trained on chosen/rejected pairs |
| `scripts/ppo_trainer.py` | PPO alignment loop via `trl.PPOTrainer` |
| `scripts/dpo_trainer.py` | DPO alignment via `trl.DPOTrainer` (simpler alternative) |
| `scripts/preference_data.py` | Load and format Anthropic HH-RLHF / UltraFeedback datasets |
| `scripts/main.py` | Argparse CLI with `reward`, `ppo`, `dpo` commands |
| `configs/defaults.yaml` | Reward, PPO, DPO, and data configuration |
| `run_on_colab.ipynb` | Step-by-step Colab notebook for T4 GPU training |
| `pyproject.toml` | Python dependencies (torch, transformers, trl, datasets, etc.) |

## Dependencies

```bash
pip install .
```

GPU strongly recommended for full training. Dry-run mode (`--dry-run`) works on CPU.

## References

- Ouyang et al. (2022) — "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
- Rafailov et al. (2023) — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- TRL docs — https://huggingface.co/docs/trl
