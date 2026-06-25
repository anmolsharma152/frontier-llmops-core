# Phase 1: Data & Curation

## Goal

Generate high-quality synthetic training data and prepare it for fine-tuning. This phase covers the full pipeline: synthetic generation → text chunking → embeddings → quality filtering.

## Concepts

### Synthetic Data Generation

When real domain-specific data is scarce or expensive to collect, a **teacher model** (GPT-4o, Claude, etc.) can generate prompt-completion pairs. Key considerations:

- **Prompt engineering**: A well-crafted system prompt + few-shot examples dramatically improves output quality.
- **Diversity**: Temperature > 0, varied phrasing, and domain sub-topics prevent mode collapse.
- **Format structure**: JSON output with `prompt` / `completion` keys makes downstream ingestion trivial.

### Chunking Strategies

| Strategy | Best for | Trade-off |
|----------|----------|-----------|
| Character | Simple fixed-length splits | May break sentences/words |
| Token | Aligns with model context windows | Needs tokenizer; language-dependent |
| Recursive | Respects natural boundaries (paragraphs → sentences) | Slower; separator tuning needed |
| Semantic | Groups by topical coherence | Requires embeddings; most expensive |

### Embeddings

Dense vector representations enable semantic search and clustering. `bge-small-en-v1.5` gives a good speed/quality trade-off for the pipeline.

### Quality Filtering

Heuristic guardrails remove obvious garbage before training:
- **Length bounds**: Reject too-short or too-long samples
- **Repetition ratio**: N-gram repetition → degenerate outputs
- **Perplexity**: Optional LM-based outlier detection

## Pipeline

```bash
# Full pipeline
python scripts/main.py --config configs/defaults.yaml all

# Individual steps
python scripts/main.py --config configs/defaults.yaml generate
python scripts/main.py --config configs/defaults.yaml --input output/01_generated/synthetic_pairs.jsonl chunk
python scripts/main.py --config configs/defaults.yaml --input output/02_chunked/chunks.jsonl embed
python scripts/main.py --config configs/defaults.yaml --input output/02_chunked/chunks.jsonl filter
```

## Dependencies

```bash
pip install .
```

Requires `OPENAI_API_KEY` in environment for generation.

## References

- Smith et al. (2023) — "Data Quality for LLM Training"
- Dao et al. (2022) — "FlashAttention: Fast and Memory-Efficient Exact Attention"
- BGE Embedding Models — https://huggingface.co/BAAI/bge-small-en-v1.5
