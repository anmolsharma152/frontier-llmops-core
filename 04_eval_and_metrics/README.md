# Phase 4: Evaluation & Metrics

## Goal

Quantitatively and qualitatively assess model quality using both automated reference-based metrics and LLM-as-a-judge scoring.

## Concepts

### LLM-as-a-Judge

A strong LLM (GPT-4o, Claude 3.5) scores outputs on criteria like accuracy, relevance, and clarity. This correlates well with human judgement and is far more scalable.

**Pitfalls to watch for:**
- **Position bias**: The judge prefers first or second response in pairwise comparisons
- **Self-enhancement bias**: The judge prefers its own family of models
- **Verbosity bias**: Longer responses score higher regardless of quality

### Automated Metrics

| Metric | What it measures | Limitations |
|--------|-----------------|-------------|
| BLEU | N-gram precision | Favours exact lexical matches |
| ROUGE-L | Longest common subsequence recall | Ignores semantics |
| BERTScore | Semantic similarity via embeddings | Slower, needs GPU |
| METEOR | Precision/recall with stemming and synonyms | Better correlation than BLEU |

## Running

```bash
# LLM-as-a-judge (requires OPENAI_API_KEY)
python scripts/main.py judge --input path/to/eval_data.jsonl

# Automated metrics
python scripts/main.py metrics --input path/to/eval_data.jsonl

# Generate report combining both
python scripts/main.py report --judge-results output/judge/judge_results.jsonl \
                               --metrics-results output/metrics/metrics.json
```

## References

- Zheng et al. (2023) — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- Chiang & Lee (2023) — "Can Large Language Models Be an Alternative to Human Evaluations?"
- Evaluate library — https://huggingface.co/docs/evaluate
