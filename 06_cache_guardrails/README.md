# Phase 6: Guardrails & Caching

## Goal

Add a safety and efficiency layer around the deployed model. Input guardrails filter harmful prompts, output guardrails sanitise responses, and a semantic cache reduces redundant inference calls.

## Concepts

### Input Guardrails

| Threat | Detection |
|--------|-----------|
| Prompt injection | Pattern matching for "ignore previous instructions", DAN-style attacks |
| PII leakage | Regex patterns for emails, phones, SSNs, credit cards (optionally Presidio) |
| Blocklisted keywords | Configurable word/pattern blocklist |
| Length limits | Hard cap on prompt length |

### Output Guardrails

| Risk | Mitigation |
|------|------------|
| Toxic content | Keyword-based topic filtering (violence, hate speech, self-harm) |
| Blocklisted topics | Configurable topic list |
| Response length | Hard cap to prevent runaway generation |

### Semantic Cache

Instead of an exact key-value cache, **semantic caching** embeds the prompt and finds similar past queries. A hit above the similarity threshold returns the cached response — critical for production:

- **Reduces latency**: Cache hit → ~5 ms vs ~2 s LLM call
- **Reduces cost**: Fewer API calls / less GPU time
- **Backends**: Redis (persistent, shared) or in-memory dict (simple, no infra)

## Running

```bash
# Check input prompt
python scripts/main.py guard-input --prompt "What is the capital of France?"

# Check output text
python scripts/main.py guard-output --text "I think the answer is Paris."

# Show cache statistics
python scripts/main.py cache-stats

# Run full pipeline (guardrails → cache → LLM → guardrails)
python scripts/main.py pipeline --prompt "What is machine learning?"
```

## Files

| Path | Purpose |
|------|---------|
| `scripts/input_guardrails.py` | Prompt injection detection, PII scanning, blocklist enforcement |
| `scripts/output_guardrails.py` | Toxicity scoring, topic filtering, response length limits |
| `scripts/semantic_cache.py` | Embedding-based cache (Redis or in-memory) with similarity threshold |
| `scripts/pipeline.py` | Composable end-to-end pipeline: guard → cache → LLM → guard |
| `scripts/main.py` | Argparse CLI with `guard-input`, `guard-output`, `cache-stats`, `pipeline` |
| `configs/defaults.yaml` | Guardrail rules, cache backend/TTL/threshold, LLM provider |
| `pyproject.toml` | Python dependencies (openai, redis, sentence-transformers, presidio, etc.) |

## Dependencies

```bash
pip install .
```

Requires `OPENAI_API_KEY` for LLM calls. Redis is optional (falls back to in-memory cache).

## References

- Perez & Ribeiro (2022) — "Ignore Previous Prompt: Attack Techniques For Language Models"
- Rao et al. (2023) — "Semantic Caching for LLM-based Applications"
- Presidio — https://microsoft.github.io/presidio
