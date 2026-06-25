"""Output guardrails for LLM responses.

Filters for toxicity, blocklisted topics, and length constraints.
"""

from dataclasses import dataclass, field


@dataclass
class GuardResult:
    allowed: bool
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


class OutputGuardrails:
    def __init__(self, config: dict):
        self.max_length = config.get("max_response_length", 8192)
        self.topics = config.get("blocklisted_topics", [])
        self.toxicity_threshold = config.get("toxicity_threshold", 0.8)

    def _check_length(self, text: str) -> GuardResult | None:
        if len(text) > self.max_length:
            return GuardResult(
                allowed=False,
                risk_score=0.7,
                reasons=[f"Response too long ({len(text)} > {self.max_length})"],
            )
        return None

    def _check_topics(self, text: str) -> GuardResult | None:
        topic_keywords = {
            "violence": ["kill", "hurt", "attack", "weapon", "bomb", "destroy", "fight"],
            "hate_speech": ["hate", "stupid", "idiot", "racial slur"],
            "self_harm": ["suicide", "self-harm", "hurt myself", "kill myself"],
        }

        for topic in self.topics:
            keywords = topic_keywords.get(topic, [])
            for kw in keywords:
                if kw.lower() in text.lower():
                    return GuardResult(
                        allowed=False,
                        risk_score=0.9,
                        reasons=[f"Blocklisted topic: {topic} (keyword: {kw})"],
                    )
        return None

    def check(self, text: str) -> GuardResult:
        checks = [self._check_length(text), self._check_topics(text)]
        for result in checks:
            if result is not None and not result.allowed:
                return result
        return GuardResult(allowed=True, risk_score=0.0)
