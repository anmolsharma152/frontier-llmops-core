"""Load evaluation datasets.

Supports Phase-1 JSONL output or built-in example samples.
"""

import json
from pathlib import Path


def load_eval_dataset(config: dict, input_path: str | None = None) -> list[dict]:
    if input_path:
        path = Path(input_path)
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append({
                    "id": record.get("id", str(len(samples))),
                    "prompt": record.get("prompt", ""),
                    "response": record.get("completion", record.get("response", record.get("text", ""))),
                    "reference": record.get("reference", record.get("completion", "")),
                })
        return samples

    num = config.get("eval_dataset", {}).get("num_samples", 3)
    return [
        {"id": "ex1", "prompt": "What is the capital of France?", "response": "Paris is the capital of France.", "reference": "The capital of France is Paris."},
        {"id": "ex2", "prompt": "Explain gravity in one sentence.", "response": "Gravity is a force attracting masses.", "reference": "Gravity is the attractive force between objects with mass."},
        {"id": "ex3", "prompt": "What is 2+2?", "response": "4", "reference": "4"},
    ][:num]
