"""Start a vLLM OpenAI-compatible inference server.
"""

from vllm import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import run_server


def start_server(config: dict) -> None:
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
