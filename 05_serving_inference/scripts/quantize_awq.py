"""AWQ quantisation (AutoAWQ).

Requires GPU and the `autoawq` package.
Install: pip install autoawq
"""


def quantize_awq(config: dict, output_path: str) -> None:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("autoawq is required. Install: pip install autoawq")

    model_name = config["model"]["name"]
    print(f"Quantising {model_name} with AWQ...")

    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.quantize(
        tokenizer=tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
    )

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"AWQ quantised model saved to {output_path}")
