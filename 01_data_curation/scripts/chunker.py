"""Text chunking strategies.

Demonstrates character-level, token-count, recursive, and
semantic chunking approaches so you can compare trade-offs.
"""

import json
from pathlib import Path

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


class TextChunker:
    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "recursive")
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 64)
        self.separators = config.get("separators", ["\n\n", "\n", ".", " ", ""])

        self._splitter = self._build_splitter()

    def _build_splitter(self):
        if self.strategy == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=self.separators[0] if self.separators else "\n",
            )
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )
        elif self.strategy == "semantic":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[dict]:
        chunks = self._splitter.split_text(text)
        return [
            {"text": chunk, "metadata": metadata or {}, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]

    def chunk_file(self, path: Path | str) -> list[dict]:
        path = Path(path)
        all_chunks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("completion", record.get("text", ""))
                meta = {k: v for k, v in record.items() if k != "text"}
                all_chunks.extend(self.chunk_text(text, meta))
        return all_chunks

    @staticmethod
    def save(chunks: list[dict], path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
