"""Synthetic data generation via OpenAI API.

Generates prompt-completion pairs for a given domain using
a configurable teacher model and system prompt template.
"""

import json
from pathlib import Path

from openai import OpenAI


class SyntheticDataGenerator:
    def __init__(self, config: dict):
        self.client = OpenAI()
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.8)
        self.n_pairs = config.get("n_pairs", 100)
        self.system_template = config.get(
            "system_prompt_template",
            "You are a domain expert. Generate {n} prompt-completion pairs about {domain}.",
        )

    def generate(self, domain: str, description: str = "") -> list[dict]:
        system_prompt = self.system_template.format(n=self.n_pairs, domain=domain)
        user_prompt = (
            f"Domain: {domain}\n"
            f"Description: {description}\n\n"
            f"Generate {self.n_pairs} diverse, high-quality prompt-completion pairs. "
            "Return as a JSON array of objects with keys 'prompt' and 'completion'. "
            "Each prompt should be self-contained and realistic. "
            "Each completion should be detailed and correct."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        pairs = data.get("pairs", data.get("data", []))
        if isinstance(pairs, dict):
            pairs = list(pairs.values())

        for i, pair in enumerate(pairs):
            pair.setdefault("id", f"syn-{i:05d}")
            pair.setdefault("domain", domain)

        return pairs

    @staticmethod
    def save(pairs: list[dict], path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
