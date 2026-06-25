import argparse
import yaml

from .serve import start_server
from .client import InferenceClient
from .benchmark import run_benchmark


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_serve(config: dict) -> None:
    start_server(config)


def cmd_client(config: dict, prompt: str, stream: bool) -> None:
    client = InferenceClient(config["client"])
    response = client.generate(prompt, stream=stream)
    if stream:
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
    else:
        print(response)


def cmd_benchmark(config: dict) -> None:
    run_benchmark(config)


def cmd_quantize(config: dict, output_path: str) -> None:
    from .quantize_awq import quantize_awq
    quantize_awq(config, output_path)


def cmd_export_gguf(config: dict, output_path: str) -> None:
    from .export_gguf import export_to_gguf
    export_to_gguf(config, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serving & Inference Pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Start vLLM inference server")
    client_parser = subparsers.add_parser("client", help="Query the inference server")
    client_parser.add_argument("--prompt", required=True)
    client_parser.add_argument("--stream", action="store_true")
    subparsers.add_parser("benchmark", help="Run latency/throughput benchmark")
    quant_parser = subparsers.add_parser("quantize-awq", help="AWQ quantisation (GPU)")
    quant_parser.add_argument("--output-path", required=True)
    gguf_parser = subparsers.add_parser("export-gguf", help="Export to GGUF format")
    gguf_parser.add_argument("--output-path", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    dispatch = {
        "serve": lambda: cmd_serve(config),
        "client": lambda: cmd_client(config, args.prompt, args.stream),
        "benchmark": lambda: cmd_benchmark(config),
        "quantize-awq": lambda: cmd_quantize(config, args.output_path),
        "export-gguf": lambda: cmd_export_gguf(config, args.output_path),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
