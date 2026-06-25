"""Input guardrails for LLM prompts.

Detects prompt injection attempts, PII leakage, and
enforces length and blocklist constraints.
"""

import re
from dataclasses import dataclass, field


@dataclass
class GuardResult:
    allowed: bool
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


class InputGuardrails:
    def __init__(self, config: dict):
        self.max_length = config.get("max_prompt_length", 4096)
        self.blocklist = config.get("blocklisted_keywords", [])
        self.injection_threshold = config.get("prompt_injection_threshold", 0.7)
        self.pii_config = config.get("pii", {"enabled": False})

    def _check_length(self, prompt: str) -> GuardResult | None:
        if len(prompt) > self.max_length:
            return GuardResult(
                allowed=False,
                risk_score=1.0,
                reasons=[f"Prompt too long ({len(prompt)} > {self.max_length})"],
            )
        return None

    def _check_blocklist(self, prompt: str) -> GuardResult | None:
        for keyword in self.blocklist:
            if keyword.lower() in prompt.lower():
                return GuardResult(
                    allowed=False,
                    risk_score=0.9,
                    reasons=[f"Blocklisted keyword: {keyword}"],
                )
        return None

    def _check_prompt_injection(self, prompt: str) -> GuardResult | None:
        injection_patterns = [
            r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior)",
            r"(?i)(you are|you're)\s+(now|free|released|unleashed)",
            r"(?i)(do\s+not\s+)?(output|respond|answer)\s+(with|in|as)",
            r"(?i)system\s+(prompt|message|instruction).*(override|change|new)",
            r"(?i)from\s+now\s+on\s*[,:]\s*",
            r"(?i)pretend\s+(you\s+are|to\s+be)",
            r"(?i)act\s+as\s+(if\s+)?",
            r"(?i)dan\s*(=|:)\s*",
        ]

        risk_score = 0.0
        reasons = []
        for pattern in injection_patterns:
            if re.search(pattern, prompt):
                risk_score = max(risk_score, 0.7)
                reasons.append("Prompt injection pattern detected")

        if risk_score >= self.injection_threshold:
            return GuardResult(allowed=False, risk_score=risk_score, reasons=reasons)

        return None

    def _check_pii(self, prompt: str) -> GuardResult | None:
        if not self.pii_config.get("enabled", False):
            return None

        pii_patterns = {
            "EMAIL_ADDRESS": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE_NUMBER": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        }

        entities = self.pii_config.get("entities", list(pii_patterns.keys()))
        found = []
        for entity in entities:
            pattern = pii_patterns.get(entity)
            if pattern and re.search(pattern, prompt):
                found.append(entity)

        if found:
            return GuardResult(
                allowed=False,
                risk_score=0.8,
                reasons=[f"PII detected: {', '.join(found)}"],
            )
        return None

    def check(self, prompt: str) -> GuardResult:
        checks = [
            self._check_length(prompt),
            self._check_blocklist(prompt),
            self._check_prompt_injection(prompt),
            self._check_pii(prompt),
        ]

        for result in checks:
            if result is not None and not result.allowed:
                return result

        return GuardResult(allowed=True, risk_score=0.0)
