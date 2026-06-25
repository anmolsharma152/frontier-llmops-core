import argparse
import yaml
import json
from pathlib import Path

from .llm_judge import LLMJudge
from .automated_metrics import AutomatedMetrics
from .eval_dataset import load_eval_dataset
from .report import generate_report


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_judge(config: dict, input_path: str, output_dir: Path) -> None:
    dataset = load_eval_dataset(config, input_path)
    judge = LLMJudge(config["judge"])
    results = judge.evaluate(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "judge_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Judged {len(results)} samples → {output_dir / 'judge_results.jsonl'}")


def cmd_metrics(config: dict, input_path: str, output_dir: Path) -> None:
    dataset = load_eval_dataset(config, input_path)
    metrics = AutomatedMetrics(config["metrics"])
    scores = metrics.compute_all(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Metrics → {output_dir / 'metrics.json'}")
    for name, value in scores.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")


def cmd_report(config: dict, judge_path: str, metrics_path: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(judge_path, metrics_path, output_dir)
    print(f"Report generated in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation & Metrics Pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--input", default=None, help="Input dataset path")
    parser.add_argument("--output-dir", default="./output")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("judge", help="LLM-as-a-judge evaluation")
    subparsers.add_parser("metrics", help="Automated metrics (BLEU, ROUGE, BERTScore)")
    report_parser = subparsers.add_parser("report", help="Generate evaluation report")
    report_parser.add_argument("--judge-results", required=True)
    report_parser.add_argument("--metrics-results", required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    dispatch = {
        "judge": lambda: cmd_judge(config, args.input, output_dir / "judge"),
        "metrics": lambda: cmd_metrics(config, args.input, output_dir / "metrics"),
        "report": lambda: cmd_report(config, args.judge_results, args.metrics_results, output_dir / "report"),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
