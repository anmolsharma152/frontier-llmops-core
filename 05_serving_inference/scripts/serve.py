"""Start a vLLM OpenAI-compatible inference server.

Gracefully falls back to HF Transformers if vLLM is not installed.
"""

try:
    from vllm import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import run_server
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _start_server_hf(config: dict) -> None:
    """Fallback server using HF Transformers (no vLLM installed)."""
    from transformers import pipeline
    import uvicorn
    from fastapi import FastAPI

    model_name = config["model"]["name"]
    gen = pipeline("text-generation", model=model_name, device_map="auto")

    app = FastAPI()

    @app.post("/v1/completions")
    async def complete(request: dict):
        prompt = request.get("prompt", "")
        output = gen(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        return {"choices": [{"text": output}]}

    host = config["server"].get("host", "0.0.0.0")
    port = config["server"].get("port", 8000)
    print(f"[HF fallback] Starting server with {model_name} on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def start_server(config: dict) -> None:
    if not _VLLM_AVAILABLE:
        print("[WARN] vLLM not installed — falling back to HF Transformers (no continuous batching)")
        _start_server_hf(config)
        return

    model_cfg = config["model"]
    server_cfg = config["server"]

    args = AsyncEngineArgs(
        model=model_cfg["name"],
        tokenizer=model_cfg.get("tokenizer", model_cfg["name"]),
        dtype=model_cfg.get("dtype", "auto"),
        quantization=model_cfg.get("quantization"),
        max_model_len=model_cfg.get("max_model_len", 8192),
        tensor_parallel_size=server_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=server_cfg.get("gpu_memory_utilization", 0.9),
        max_num_seqs=server_cfg.get("max_num_seqs", 256),
        enable_prefix_caching=server_cfg.get("enable_prefix_caching", True),
        enforce_eager=server_cfg.get("enforce_eager", False),
    )

    print(f"Starting vLLM server with {model_cfg['name']} on "
          f"{server_cfg.get('host', '0.0.0.0')}:{server_cfg.get('port', 8000)}")

    run_server(
        args=args,
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
    )
