"""Embedding pipeline using Sentence Transformers.

Generates vector embeddings for text chunks for downstream
retrieval, clustering, or semantic cache lookups.
"""

import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, config: dict):
        model_name = config.get("model", "BAAI/bge-small-en-v1.5")
        self.batch_size = config.get("batch_size", 32)
        self.normalize = config.get("normalize_embeddings", True)
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return np.array(embeddings)

    def embed_file(self, path: Path | str) -> dict:
        path = Path(path)
        texts = []
        metadata = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                texts.append(record.get("text", ""))
                metadata.append({k: v for k, v in record.items() if k != "text"})

        embeddings = self.embed_texts(texts)
        return {"embeddings": embeddings, "metadata": metadata, "texts": texts}

    @staticmethod
    def save(data: dict, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data["embeddings"])
        meta_path = path.with_suffix(".jsonl")
        with open(meta_path, "w") as f:
            for text, meta in zip(data["texts"], data["metadata"]):
                f.write(json.dumps({"text": text, **meta}) + "\n")
