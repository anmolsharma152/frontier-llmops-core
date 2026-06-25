import argparse
import yaml
from pathlib import Path

from .generator import SyntheticDataGenerator
from .chunker import TextChunker
from .embedder import TextEmbedder
from .quality_filter import QualityFilter


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_generate(config: dict, output_dir: Path) -> None:
    gen = SyntheticDataGenerator(config["generation"])
    pairs = gen.generate(domain=config["domain"], description=config["description"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gen.save(pairs, output_dir / "synthetic_pairs.jsonl")
    print(f"Generated {len(pairs)} pairs → {output_dir / 'synthetic_pairs.jsonl'}")


def cmd_chunk(config: dict, input_path: Path, output_dir: Path) -> None:
    chunker = TextChunker(config["chunking"])
    chunks = chunker.chunk_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunker.save(chunks, output_dir / "chunks.jsonl")
    print(f"Chunked into {len(chunks)} chunks → {output_dir / 'chunks.jsonl'}")


def cmd_embed(config: dict, input_path: Path, output_dir: Path) -> None:
    embedder = TextEmbedder(config["embedding"])
    embeddings = embedder.embed_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder.save(embeddings, output_dir / "embeddings.npy")
    print(f"Embedded {len(embeddings['texts'])} documents → {output_dir / 'embeddings.npy'}")


def cmd_filter(config: dict, input_path: Path, output_dir: Path) -> None:
    flt = QualityFilter(config["quality_filter"])
    filtered = flt.filter_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    flt.save(filtered, output_dir / "filtered.jsonl")
    print(f"Filtered: {len(filtered)} items kept → {output_dir / 'filtered.jsonl'}")


def cmd_all(config: dict, output_dir: Path) -> None:
    gen_output = output_dir / "01_generated"
    chunk_output = output_dir / "02_chunked"
    embed_output = output_dir / "03_embedded"
    filter_output = output_dir / "04_filtered"

    cmd_generate(config, gen_output)
    cmd_chunk(config, gen_output / "synthetic_pairs.jsonl", chunk_output)
    cmd_embed(config, chunk_output / "chunks.jsonl", embed_output)
    cmd_filter(config, chunk_output / "chunks.jsonl", filter_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Curation Pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--input", default=None, help="Input file (for chunk/embed/filter subcommands)")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Generate synthetic data")
    subparsers.add_parser("chunk", help="Chunk text data")
    subparsers.add_parser("embed", help="Generate embeddings")
    subparsers.add_parser("filter", help="Quality-filter data")
    subparsers.add_parser("all", help="Run full pipeline: generate → chunk → embed → filter")

    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    dispatch = {
        "generate": lambda: cmd_generate(config, output_dir),
        "chunk": lambda: cmd_chunk(config, Path(args.input), output_dir),
        "embed": lambda: cmd_embed(config, Path(args.input), output_dir),
        "filter": lambda: cmd_filter(config, Path(args.input), output_dir),
        "all": lambda: cmd_all(config, output_dir),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
