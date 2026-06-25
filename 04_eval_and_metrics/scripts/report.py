"""Generate evaluation report (Markdown + charts).

Combines LLM judge scores and automated metrics into a
human-readable report.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_report(judge_path: str, metrics_path: str, output_dir: Path) -> None:
    with open(judge_path) as f:
        judge_results = [json.loads(line) for line in f if line.strip()]

    with open(metrics_path) as f:
        metrics = json.load(f)

    report_lines = ["# Evaluation Report\n"]

    report_lines.append("## Automated Metrics\n")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    for name, value in metrics.items():
        if isinstance(value, float):
            report_lines.append(f"| {name} | {value:.4f} |")
        else:
            report_lines.append(f"| {name} | {value} |")

    report_lines.append("\n## LLM Judge Scores\n")
    criteria = [k for k in judge_results[0].keys() if k not in ("_sample_id", "justification", "error", "raw")]
    if criteria:
        report_lines.append("| Sample ID | " + " | ".join(c.capitalize() for c in criteria) + " |")
        report_lines.append("|-----------|" + "|".join("---" for _ in criteria) + "|")
        for r in judge_results:
            row = [str(r.get("_sample_id", "?"))]
            for c in criteria:
                row.append(str(r.get(c, "")))
            report_lines.append("| " + " | ".join(row) + " |")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if criteria:
        scores = {c: [] for c in criteria}
        for r in judge_results:
            for c in criteria:
                try:
                    scores[c].append(float(r.get(c, 0)))
                except (ValueError, TypeError):
                    pass
        x = np.arange(len(criteria))
        means = [np.mean(scores[c]) for c in criteria]
        axes[0].bar(x, means)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(criteria)
        axes[0].set_ylabel("Mean Score")
        axes[0].set_title("Judge Scores by Criterion")

    metric_names = [k for k in metrics if isinstance(metrics[k], float)]
    metric_values = [metrics[k] for k in metric_names]
    if metric_names:
        x2 = np.arange(len(metric_names))
        axes[1].bar(x2, metric_values)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(metric_names, rotation=45, ha="right")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Automated Metrics")

    plt.tight_layout()
    chart_path = output_dir / "eval_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    report_lines.append(f"\n![Evaluation Chart]({chart_path.name})\n")

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to {report_path}")
    print(f"Chart saved to {chart_path}")
