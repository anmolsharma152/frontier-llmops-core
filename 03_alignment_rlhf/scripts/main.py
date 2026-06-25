import argparse
import yaml
import torch

from .reward_model import RewardModelTrainer
from .ppo_trainer import PPOTrainer
from .dpo_trainer import DPOTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def detect_device(dry_run: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if dry_run:
        print("[INFO] No GPU — dry-run mode enabled")
        return torch.device("cpu")
    raise RuntimeError("GPU required for training. Use --dry-run or run on Colab.")


def cmd_reward(config: dict, dry_run: bool) -> None:
    device = detect_device(dry_run)
    trainer = RewardModelTrainer(config, device, dry_run)
    trainer.train()


def cmd_ppo(config: dict, dry_run: bool) -> None:
    device = detect_device(dry_run)
    trainer = PPOTrainer(config, device, dry_run)
    trainer.train()


def cmd_dpo(config: dict, dry_run: bool) -> None:
    device = detect_device(dry_run)
    trainer = DPOTrainer(config, device, dry_run)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="RLHF Alignment Pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--dry-run", action="store_true", help="CPU dry-run with GPT-2")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("reward", help="Train reward model")
    subparsers.add_parser("ppo", help="Run PPO alignment")
    subparsers.add_parser("dpo", help="Run DPO alignment")

    args = parser.parse_args()
    config = load_config(args.config)

    dispatch = {
        "reward": lambda: cmd_reward(config, args.dry_run),
        "ppo": lambda: cmd_ppo(config, args.dry_run),
        "dpo": lambda: cmd_dpo(config, args.dry_run),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
