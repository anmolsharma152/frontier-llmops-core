import argparse
import yaml
import torch

from .trainer import QLoRATrainer
from .inference import ModelInference


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


def cmd_train(config: dict, dry_run: bool) -> None:
    device = detect_device(dry_run)
    trainer = QLoRATrainer(config, device, dry_run)
    trainer.train()


def cmd_infer(config: dict, checkpoint: str, prompt: str) -> None:
    infer = ModelInference(config, checkpoint)
    output = infer.generate(prompt)
    print(f"\nPrompt: {prompt}\nResponse: {output}\n")


def cmd_merge(config: dict, adapter_path: str, output_path: str) -> None:
    from .merge import merge_adapter
    merge_adapter(config, adapter_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml", help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline on CPU with tiny model")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Run QLoRA fine-tuning")
    infer_parser = subparsers.add_parser("infer", help="Run inference with trained adapter")
    infer_parser.add_argument("--checkpoint", required=True, help="Path to adapter checkpoint")
    infer_parser.add_argument("--prompt", required=True, help="Input prompt")
    merge_parser = subparsers.add_parser("merge", help="Merge adapter into base model")
    merge_parser.add_argument("--adapter-path", required=True)
    merge_parser.add_argument("--output-path", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    dispatch = {
        "train": lambda: cmd_train(config, args.dry_run),
        "infer": lambda: cmd_infer(config, args.checkpoint, args.prompt),
        "merge": lambda: cmd_merge(config, args.adapter_path, args.output_path),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
