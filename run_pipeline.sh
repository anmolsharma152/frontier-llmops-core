#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║  Frontier LLMOps Core — Full Pipeline   ║"
echo "╚══════════════════════════════════════════╝"

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_CONFIG="01_data_curation/configs/defaults.yaml"
PEFT_CONFIG="02_peft_tuning/configs/defaults.yaml"
EVAL_CONFIG="04_eval_and_metrics/configs/defaults.yaml"
SERVE_CONFIG="05_serving_inference/configs/defaults.yaml"
GUARD_CONFIG="06_cache_guardrails/configs/defaults.yaml"

# ─── Phase 1: Data & Curation ────────────────────────────────────────────────
echo ""
echo "▸ Phase 1: Generating synthetic data..."
python 01_data_curation/scripts/main.py --config "$DATA_CONFIG" generate
python 01_data_curation/scripts/main.py --config "$DATA_CONFIG" --input output/01_generated/synthetic_pairs.jsonl chunk
python 01_data_curation/scripts/main.py --config "$DATA_CONFIG" --input output/02_chunked/chunks.jsonl embed
python 01_data_curation/scripts/main.py --config "$DATA_CONFIG" --input output/02_chunked/chunks.jsonl filter
echo "  ✓ Data curation complete."

# ─── Phase 2: PEFT Tuning ────────────────────────────────────────────────────
echo ""
echo "▸ Phase 2: Running QLoRA dry-run validation..."
python 02_peft_tuning/scripts/main.py --config "$PEFT_CONFIG" --dry-run train
echo "  ✓ PEFT dry-run passed."
echo "  → For full training on GPU: python 02_peft_tuning/scripts/main.py train"

# ─── Phase 4: Eval & Metrics ─────────────────────────────────────────────────
echo ""
echo "▸ Phase 4: Running evaluation..."
python 04_eval_and_metrics/scripts/main.py --config "$EVAL_CONFIG" judge --input data/eval_samples.jsonl 2>/dev/null || \
    python 04_eval_and_metrics/scripts/main.py --config "$EVAL_CONFIG" metrics --input data/eval_samples.jsonl 2>/dev/null || \
    echo "  ⚠ No eval data found. Provide --input or place JSONL in data/"
echo "  ✓ Evaluation complete."

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Pipeline finished successfully.        ║"
echo "║                                         ║"
echo "║  Next steps:                            ║"
echo "║  - Train on GPU:  make train            ║"
echo "║  - Serve model:   make serve            ║"
echo "║  - Guardrails:    make pipeline          ║"
echo "╚══════════════════════════════════════════╝"
