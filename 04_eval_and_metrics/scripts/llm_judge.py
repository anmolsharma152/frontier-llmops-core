"""LLM-as-a-Judge evaluation.

Uses a judge model (e.g. GPT-4o) to score model outputs against
reference answers using a configurable rubric.
"""

import json
from openai import OpenAI


class LLMJudge:
    def __init__(self, config: dict):
        self.client = OpenAI()
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.0)
        self.rubric_template = config.get(
            "rubric_template",
            "Score the following response on a scale of 1-5 for accuracy, relevance, and clarity.\n\nPrompt: {prompt}\nResponse: {response}\nReference: {reference}\n\nReturn JSON with scores and justification.",
        )

    def _build_prompt(self, sample: dict) -> str:
        return self.rubric_template.format(
            prompt=sample.get("prompt", ""),
            response=sample.get("response", ""),
            reference=sample.get("reference", ""),
        )

    def _judge_one(self, sample: dict) -> dict:
        prompt = self._build_prompt(sample)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a strict but fair evaluator. Score responses objectively."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"raw": content, "error": "failed to parse"}
        result["_sample_id"] = sample.get("id", "")
        return result

    def evaluate(self, dataset: list[dict]) -> list[dict]:
        results = []
        for sample in dataset:
            result = self._judge_one(sample)
            results.append(result)
            print(f"  Judged {sample.get('id', '?')}: {result.get('accuracy', '?')}")
        return results
