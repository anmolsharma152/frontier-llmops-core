import argparse
import yaml

from .input_guardrails import InputGuardrails
from .output_guardrails import OutputGuardrails
from .semantic_cache import SemanticCache
from .pipeline import GuardrailsPipeline


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_guard_input(config: dict, prompt: str) -> None:
    guard = InputGuardrails(config["guardrails"]["input"])
    result = guard.check(prompt)
    print(f"Allowed: {result.allowed}")
    print(f"Risk score: {result.risk_score:.2f}")
    if result.reasons:
        print(f"Reasons: {', '.join(result.reasons)}")


def cmd_guard_output(config: dict, text: str) -> None:
    guard = OutputGuardrails(config["guardrails"]["output"])
    result = guard.check(text)
    print(f"Allowed: {result.allowed}")
    print(f"Risk score: {result.risk_score:.2f}")
    if result.reasons:
        print(f"Reasons: {', '.join(result.reasons)}")


def cmd_cache_stats(config: dict) -> None:
    cache = SemanticCache(config["cache"])
    stats = cache.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


def cmd_pipeline(config: dict, prompt: str) -> None:
    pipeline = GuardrailsPipeline(config)
    result = pipeline.run(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Allowed: {result.allowed}")
    if result.cached:
        print(f"Cached: yes (similarity={result.similarity:.3f})")
    print(f"Response: {result.response}")
    if result.guardrail_reasons:
        print(f"Guardrail blocks: {', '.join(result.guardrail_reasons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrails & Caching Pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    guard_input = subparsers.add_parser("guard-input", help="Check input prompt")
    guard_input.add_argument("--prompt", required=True)

    guard_output = subparsers.add_parser("guard-output", help="Check output text")
    guard_output.add_argument("--text", required=True)

    subparsers.add_parser("cache-stats", help="Show cache statistics")

    pipeline = subparsers.add_parser("pipeline", help="Run full guardrails + cache pipeline")
    pipeline.add_argument("--prompt", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    dispatch = {
        "guard-input": lambda: cmd_guard_input(config, args.prompt),
        "guard-output": lambda: cmd_guard_output(config, args.text),
        "cache-stats": lambda: cmd_cache_stats(config),
        "pipeline": lambda: cmd_pipeline(config, args.prompt),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
