.PHONY: generate chunk embed filter data-all \
	train-dry train merge \
	rlhf-reward rlhf-ppo rlhf-dpo rlhf-dry \
	eval-judge eval-metrics eval-report \
	serve client bench quantize-awq export-gguf \
	guard-input guard-output cache-stats pipeline \
	clean help

# ─── Phase 1: Data & Curation ────────────────────────────────────────────────

generate:
	python 01_data_curation/scripts/main.py --config 01_data_curation/configs/defaults.yaml generate

chunk:
	python 01_data_curation/scripts/main.py --config 01_data_curation/configs/defaults.yaml chunk --input ./output/01_generated/synthetic_pairs.jsonl

embed:
	python 01_data_curation/scripts/main.py --config 01_data_curation/configs/defaults.yaml embed --input ./output/02_chunked/chunks.jsonl

filter:
	python 01_data_curation/scripts/main.py --config 01_data_curation/configs/defaults.yaml filter --input ./output/02_chunked/chunks.jsonl

data-all:
	python 01_data_curation/scripts/main.py --config 01_data_curation/configs/defaults.yaml all

# ─── Phase 2: PEFT Tuning ────────────────────────────────────────────────────

train-dry:
	python 02_peft_tuning/scripts/main.py --config 02_peft_tuning/configs/defaults.yaml --dry-run train

train:
	python 02_peft_tuning/scripts/main.py --config 02_peft_tuning/configs/defaults.yaml train

merge:
	python 02_peft_tuning/scripts/main.py --config 02_peft_tuning/configs/defaults.yaml merge

# ─── Phase 3: RLHF Alignment ─────────────────────────────────────────────────

rlhf-dry:
	python 03_alignment_rlhf/scripts/main.py --config 03_alignment_rlhf/configs/defaults.yaml --dry-run reward

rlhf-reward:
	python 03_alignment_rlhf/scripts/main.py --config 03_alignment_rlhf/configs/defaults.yaml reward

rlhf-ppo:
	python 03_alignment_rlhf/scripts/main.py --config 03_alignment_rlhf/configs/defaults.yaml ppo

rlhf-dpo:
	python 03_alignment_rlhf/scripts/main.py --config 03_alignment_rlhf/configs/defaults.yaml dpo

# ─── Phase 4: Eval & Metrics ─────────────────────────────────────────────────

eval-judge:
	python 04_eval_and_metrics/scripts/main.py --config 04_eval_and_metrics/configs/defaults.yaml judge --input ./eval_data.jsonl

eval-metrics:
	python 04_eval_and_metrics/scripts/main.py --config 04_eval_and_metrics/configs/defaults.yaml metrics --input ./eval_data.jsonl

eval-report:
	python 04_eval_and_metrics/scripts/main.py --config 04_eval_and_metrics/configs/defaults.yaml report

# ─── Phase 5: Serving ────────────────────────────────────────────────────────

serve:
	python 05_serving_inference/scripts/main.py --config 05_serving_inference/configs/defaults.yaml serve

client:
	python 05_serving_inference/scripts/main.py --config 05_serving_inference/configs/defaults.yaml client --prompt "Hello, how are you?"

bench:
	python 05_serving_inference/scripts/main.py --config 05_serving_inference/configs/defaults.yaml benchmark

quantize-awq:
	python 05_serving_inference/scripts/main.py --config 05_serving_inference/configs/defaults.yaml quantize-awq --output-path ./awq-model

export-gguf:
	python 05_serving_inference/scripts/main.py --config 05_serving_inference/configs/defaults.yaml export-gguf --output-path ./gguf-model

# ─── Phase 6: Guardrails & Caching ────────────────────────────────────────────

guard-input:
	python 06_cache_guardrails/scripts/main.py --config 06_cache_guardrails/configs/defaults.yaml guard-input --prompt "What is AI?"

guard-output:
	python 06_cache_guardrails/scripts/main.py --config 06_cache_guardrails/configs/defaults.yaml guard-output --text "AI is a field of study."

cache-stats:
	python 06_cache_guardrails/scripts/main.py --config 06_cache_guardrails/configs/defaults.yaml cache-stats

pipeline:
	python 06_cache_guardrails/scripts/main.py --config 06_cache_guardrails/configs/defaults.yaml pipeline --prompt "What is machine learning?"

# ─── Utility ──────────────────────────────────────────────────────────────────

clean:
	rm -rf output/ 01_data_curation/output 02_peft_tuning/output 03_alignment_rlhf/output
	rm -rf 04_eval_and_metrics/output 05_serving_inference/output 06_cache_guardrails/output
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	rm -rf wandb/ */wandb/
	rm -rf .ruff_cache/
	@echo "Cleaned."

help:
	@echo "Frontier LLMOps Core — Makefile"
	@echo ""
	@echo "Phase 1 — Data:       make generate  make chunk  make embed  make filter  make data-all"
	@echo "Phase 2 — Training:   make train-dry  make train  make merge"
	@echo "Phase 3 — RLHF:       make rlhf-dry  make rlhf-reward  make rlhf-ppo  make rlhf-dpo"
	@echo "Phase 4 — Eval:       make eval-judge  make eval-metrics  make eval-report"
	@echo "Phase 5 — Serving:    make serve  make client  make bench  make quantize-awq  make export-gguf"
	@echo "Phase 6 — Guardrails: make guard-input  make guard-output  make cache-stats  make pipeline"
	@echo "Utility:              make clean  make help"
