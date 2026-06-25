"""Quality filters for generated/curated data.

Heuristic-based filtering to remove low-quality samples:
too short, too long, repetitive, or perplexity-outlier.
"""

import json
from collections import Counter
from pathlib import Path


class QualityFilter:
    def __init__(self, config: dict):
        self.min_length = config.get("min_length", 50)
        self.max_length = config.get("max_length", 4096)
        self.max_repetition_ratio = config.get("max_repetition_ratio", 0.3)
        self.perplexity_threshold = config.get("perplexity_threshold")

    @staticmethod
    def repetition_ratio(text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        counts = Counter(words)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(words)

    def is_valid(self, record: dict) -> tuple[bool, str]:
        text = record.get("completion", record.get("text", ""))

        if len(text) < self.min_length:
            return False, f"too_short ({len(text)} < {self.min_length})"
        if len(text) > self.max_length:
            return False, f"too_long ({len(text)} > {self.max_length})"

        ratio = self.repetition_ratio(text)
        if ratio > self.max_repetition_ratio:
            return False, f"repetitive (ratio={ratio:.2f} > {self.max_repetition_ratio})"

        return True, "pass"

    def filter_file(self, path: Path | str) -> list[dict]:
        path = Path(path)
        kept = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                valid, reason = self.is_valid(record)
                record["_filter_reason"] = reason
                if valid:
                    kept.append(record)
        return kept

    @staticmethod
    def save(records: list[dict], path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
