"""Async OpenAI-compatible client for the vLLM server.
"""

from openai import OpenAI


class InferenceClient:
    def __init__(self, config: dict):
        self.client = OpenAI(
            api_key=config.get("api_key", "token-abc123"),
            base_url=config.get("base_url", "http://localhost:8000/v1"),
        )
        self.model = config.get("model")  # None → server default

    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str | list[str]:
        response = self.client.chat.completions.create(
            model=self.model or "default",
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            **kwargs,
        )

        if stream:
            return [chunk.choices[0].delta.content or "" for chunk in response]

        return response.choices[0].message.content
