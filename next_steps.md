# Next Steps

## 1. Runtime Verification (highest priority)

Run each phase end-to-end to shake out real bugs:

- **Phase 1**: `make data-all` with a real `OPENAI_API_KEY` — verify it produces valid JSONL
- **Phase 2**: `make train-dry` — validate the CPU dry-run path with GPT-2
- **Phase 3**: `make rlhf-dry` — validate reward model / DPO dry-run path
- **Phase 4**: Run `eval-metrics` on Phase 1's output to confirm cross-phase data format compatibility
- **Phase 5**: `make client` against a running server (or dry-start to confirm config loads)
- **Phase 6**: `make guard-input` and `make cache-stats` to confirm guardrails + cache init

## 2. Testing Infrastructure

- Add `tests/` directories per phase with unit tests for core logic (chunker, embedder, guardrails, cache, metrics)
- Add root `pyproject.toml` with shared dev dependencies (`pytest`, `ruff`, `mypy`)
- Set up GitHub Actions: `ruff check` on every PR, optionally `pytest` on merge

## 3. Documentation Polish

- Root `README.md` mermaid diagram still references "Llama 3 / Mistral" in Phase 2 — should say "Mistral 7B" only
- Subgraph titles ("Data & Curation", "Adaptation", "Production") use the old roadmap language — could align with directory names

## 4. Infrastructure Niceties (optional)

- `.github/ISSUE_TEMPLATE/` for bug reports / feature requests
- `.pre-commit-config.yaml` for pre-commit hooks (ruff, trailing whitespace)
- `docker-compose.yml` for running Redis (Phase 6) + vLLM serving stack
- `.github/workflows/` for CI/CD
