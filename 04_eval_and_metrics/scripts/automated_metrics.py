"""Automated reference-based metrics.

Computes BLEU, ROUGE-L, BERTScore, and METEOR between
model outputs and reference answers.
"""

import evaluate
import numpy as np


class AutomatedMetrics:
    def __init__(self, metric_names: list[str]):
        self.metric_names = metric_names if metric_names else ["bleu", "rouge", "bertscore", "meteor"]
        self._metrics = {}
        for name in self.metric_names:
            try:
                self._metrics[name] = evaluate.load(name)
            except Exception as e:
                print(f"Warning: could not load metric '{name}': {e}")

    def compute_one(self, predictions: list[str], references: list[str]) -> dict:
        scores = {}
        for name, metric in self._metrics.items():
            try:
                result = metric.compute(predictions=predictions, references=references)
                if name == "rouge" and isinstance(result, dict):
                    scores.update({f"rouge-{k}": v for k, v in result.items()})
                elif name == "bertscore" and isinstance(result, dict):
                    scores["bertscore_f1"] = float(np.mean(result["f1"]))
                elif name == "bleu" and isinstance(result, dict):
                    scores["bleu"] = result.get("bleu", result.get("score", 0))
                elif name == "meteor" and isinstance(result, dict):
                    scores["meteor"] = result.get("meteor", 0)
                else:
                    scores[name] = result
            except Exception as e:
                print(f"  Error computing {name}: {e}")
        return scores

    def compute_all(self, dataset: list[dict]) -> dict:
        predictions = [s.get("response", "") for s in dataset]
        references = [s.get("reference", "") for s in dataset]
        return self.compute_one(predictions, references)
