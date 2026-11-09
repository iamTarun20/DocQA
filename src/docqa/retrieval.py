"""Keyword search, vector search and reciprocal rank fusion."""

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from zlib import crc32

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from docqa.ingest import Chunk
from docqa.text import tokens

BGE_MODEL = "BAAI/bge-small-en-v1.5"
Vector = NDArray[np.float32]


class Mode(StrEnum):
    """Available ranking strategies."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class Embedder(Protocol):
    """Produce a matrix of passage vectors and a compatible query vector."""

    def passages(self, texts: Sequence[str]) -> Vector:
        """Encode passages in index order."""
        ...

    def query(self, text: str) -> Vector:
        """Encode a query with the model's search-query settings."""
        ...


class BGEEmbedder:
    """Load FastEmbed's small BGE model into a persistent local cache."""

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        cache = os.getenv("DOCQA_MODEL_CACHE") or Path.home() / ".cache" / "docqa" / "fastembed"
        self._model = TextEmbedding(
            BGE_MODEL,
            cache_dir=str(cache),
            threads=2,
            local_files_only=os.getenv("DOCQA_LOCAL_FILES_ONLY") == "1",
        )

    def passages(self, texts: Sequence[str]) -> Vector:
        """Encode passages in index order."""
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def query(self, text: str) -> Vector:
        """Encode a search query."""
        return np.asarray(next(self._model.query_embed(text)), dtype=np.float32)


class FakeEmbedder:
    """Deterministic token hashing for tests and the network-free smoke run."""

    def passages(self, texts: Sequence[str]) -> Vector:
        """Stack fake passage vectors."""
        return np.stack([self.query(text) for text in texts])

    def query(self, text: str) -> Vector:
        """Hash content tokens into a fixed-width count vector."""
        vector = np.zeros(256, dtype=np.float32)
        for word in tokens(text):
            vector[crc32(word.encode()) % len(vector)] += 1
        return vector


class VectorStore(Protocol):
    """Index vectors and return row offsets with cosine similarity scores."""

    def index(self, vectors: Vector) -> None:
        """Replace the index with the provided matrix."""
        ...

    def search(self, query: Vector, top_k: int) -> list[tuple[int, float]]:
        """Return at most top_k rows, nearest first."""
        ...


class NumpyVectorStore:
    """Exact cosine search over a small in-memory matrix."""

    def __init__(self) -> None:
        self._vectors: Vector | None = None

    def index(self, vectors: Vector) -> None:
        """Normalize passage vectors once, leaving zero vectors at zero."""
        if vectors.ndim != 2 or not all(vectors.shape) or not np.isfinite(vectors).all():
            raise ValueError("Expected a nonempty finite matrix of passage vectors.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.maximum(norms, 1e-12)

    def search(self, query: Vector, top_k: int) -> list[tuple[int, float]]:
        """Normalize the query and rank by cosine similarity."""
        if self._vectors is None:
            raise ValueError("Index passages before searching.")
        if query.shape != (self._vectors.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Expected a finite query vector with the index dimension.")
        scores = self._vectors @ (query / max(np.linalg.norm(query), 1e-12))
        return [(int(i), float(scores[i])) for i in np.argsort(-scores, kind="stable")[:top_k]]


@dataclass(frozen=True)
class Hit:
    """A chunk and the native score of its retrieval mode."""

    chunk: Chunk
    score: float


def _fuse(*rankings: list[tuple[int, float]]) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (index, _) in enumerate(ranking, start=1):
            scores[index] += 1 / (60 + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    """Build an in-memory index and expose ranked, scored chunks."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        texts = [f"{chunk.section}\n{chunk.text}" for chunk in chunks]
        tokenized = [tokens(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None
        self._embedder = embedder
        self._vectors = vector_store if vector_store is not None else NumpyVectorStore()
        if embedder is not None and texts:
            self._vectors.index(embedder.passages(texts))

    def search(self, question: str, mode: Mode, top_k: int) -> list[Hit]:
        """Rank by BM25, cosine, or reciprocal rank fusion, then take top_k."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_terms = tokens(question)
        if not query_terms or not self.chunks:
            return []
        keyword: list[tuple[int, float]] = []
        dense: list[tuple[int, float]] = []
        if mode != Mode.DENSE and self._bm25 is not None:
            scores = self._bm25.get_scores(query_terms)
            keyword = [
                (int(i), float(scores[i]))
                for i in np.argsort(-scores, kind="stable")
                if scores[i] > 0
            ]
        if mode != Mode.BM25:
            if self._embedder is None:
                raise ValueError("Dense and hybrid retrieval require an embedder.")
            dense = self._vectors.search(self._embedder.query(question), len(self.chunks))
        ranking = {Mode.BM25: keyword, Mode.DENSE: dense}
        ranked = _fuse(keyword, dense) if mode == Mode.HYBRID else ranking[mode]
        return [Hit(self.chunks[index], score) for index, score in ranked[:top_k]]
