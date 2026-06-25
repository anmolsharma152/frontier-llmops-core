"""Export model to GGUF format via the llama.cpp convert script.

Requires the llama.cpp repository cloned locally.
"""

from pathlib import Path


def export_to_gguf(config: dict, output_path: str) -> None:
    model_name = config["model"]["name"]
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_name} to GGUF...")
    print("This requires llama.cpp. Clone: git clone https://github.com/ggml-org/llama.cpp")
    print("Then run: python llama.cpp/convert.py <model-dir> --outfile <output>.gguf")

    instructions = f"""
To export {model_name} to GGUF:

1. Clone llama.cpp:
   git clone https://github.com/ggml-org/llama.cpp

2. Download the model (if not already local):
   huggingface-cli download {model_name} --local-dir ./model

3. Convert:
   python llama.cpp/convert.py ./model --outfile {output_dir / 'model.gguf'}

4. Serve with llama.cpp:
   ./llama.cpp/build/bin/llama-server -m {output_dir / 'model.gguf'} --port 8080
"""
    print(instructions)
    with open(output_dir / "GGUF_EXPORT_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    print(f"Instructions saved to {output_dir / 'GGUF_EXPORT_INSTRUCTIONS.md'}")
