# Phase 5: Serving & Inference

## Goal

Deploy a fine-tuned model for production inference. This phase covers AWQ quantisation, GGUF export, vLLM serving with continuous batching, and benchmarking.

## Concepts

### vLLM & Continuous Batching

Traditional inference processes one request at a time. vLLM uses **continuous batching** — new requests join an active batch as soon as earlier ones finish — dramatically improving throughput. Key features:

- **PagedAttention**: Manages KV-cache in fixed-size blocks, eliminating fragmentation
- **Prefix caching**: Reuses KV-cache from common prefixes (system prompts, few-shot examples)
- **Tensor parallelism**: Splits the model across multiple GPUs

### Quantisation

| Method | Bits | Quality Loss | Speed | Use Case |
|--------|------|-------------|-------|----------|
| AWQ | 4-bit | Minimal | Fastest | Production GPU serving |
| GGUF | 2–8 bit | Low to moderate | Good | CPU/edge inference via llama.cpp |
| GPTQ | 4-bit | Minimal | Fast | GPU serving (alternative to AWQ) |

## Running

```bash
# Start vLLM server (GPU required)
python scripts/main.py serve

# Query the server
python scripts/main.py client --prompt "What is machine learning?" --stream

# Run benchmark
python scripts/main.py benchmark

# AWQ quantisation (GPU required)
python scripts/main.py quantize-awq --output-path ./awq-model

# GGUF export instructions
python scripts/main.py export-gguf --output-path ./gguf-model
```

## Colab

Open `run_on_colab.ipynb` to quantise and benchmark on a T4 GPU.

## References

- Kwon et al. (2023) — "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Lin et al. (2023) — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- vLLM docs — https://docs.vllm.ai
- llama.cpp — https://github.com/ggml-org/llama.cpp
