"""End-to-end guardrails + cache pipeline.

Orchestrates: input guardrails → cache lookup → LLM call → output guardrails → cache store
"""

from dataclasses import dataclass, field
from openai import OpenAI

from .input_guardrails import InputGuardrails
from .output_guardrails import OutputGuardrails
from .semantic_cache import SemanticCache


@dataclass
class PipelineResult:
    allowed: bool
    response: str = ""
    cached: bool = False
    similarity: float = 0.0
    guardrail_reasons: list[str] = field(default_factory=list)


class GuardrailsPipeline:
    def __init__(self, config: dict):
        self.input_guard = InputGuardrails(config["guardrails"]["input"])
        self.output_guard = OutputGuardrails(config["guardrails"]["output"])
        self.cache = SemanticCache(config["cache"]) if config["cache"].get("enabled") else None

        llm_cfg = config["llm"]
        self.client = OpenAI(
            api_base=llm_cfg.get("api_base", "https://api.openai.com/v1"),
        )
        self.model = llm_cfg.get("model", "gpt-4o-mini")
        self.temperature = llm_cfg.get("temperature", 0.7)

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def run(self, prompt: str) -> PipelineResult:
        input_result = self.input_guard.check(prompt)
        if not input_result.allowed:
            return PipelineResult(allowed=False, guardrail_reasons=input_result.reasons)

        if self.cache:
            is_hit, cached_response = self.cache.get(prompt)
            if is_hit:
                output_result = self.output_guard.check(cached_response)
                if output_result.allowed:
                    return PipelineResult(
                        allowed=True,
                        response=cached_response,
                        cached=True,
                        similarity=self.cache.threshold,
                    )

        response = self._call_llm(prompt)

        output_result = self.output_guard.check(response)
        if not output_result.allowed:
            return PipelineResult(
                allowed=False,
                response=response,
                guardrail_reasons=output_result.reasons,
            )

        if self.cache:
            self.cache.set(prompt, response)

        return PipelineResult(allowed=True, response=response)
