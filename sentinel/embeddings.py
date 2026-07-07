"""Text embeddings for Project Sentinel.

Default backend: hashed bag-of-words vectors, fully offline.
Optional backend: sentence-transformers when SENTINEL_EMBEDDINGS=sentence-transformers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from functools import lru_cache


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def embed_text(text: str) -> list[float]:
    backend = os.getenv("SENTINEL_EMBEDDINGS", "auto").strip().lower()

    if backend == "hashed":
        return hashed_embedding(text)

    if backend in {"sentence-transformer", "sentence-transformers", "st"}:
        return sentence_transformer_embedding(text)


    try:
        from sentence_transformers import SentenceTransformer
        return sentence_transformer_embedding(text)
    except ImportError:
        return hashed_embedding(text)



def hashed_embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    counts = Counter(tokenize(text))

    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * float(count)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def sentence_transformer_embedding(text: str) -> list[float]:
    model = _sentence_transformer_model()
    vector = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in vector]


@lru_cache(maxsize=1)
def _sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it or unset SENTINEL_EMBEDDINGS."
        ) from exc

    model_name = os.getenv("SENTINEL_SENTENCE_MODEL", DEFAULT_SENTENCE_TRANSFORMER)
    return SentenceTransformer(model_name)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []

    dimensions = len(vectors[0])
    center = [0.0] * dimensions
    for vector in vectors:
        for index, value in enumerate(vector):
            center[index] += value

    avg_vector = [value / len(vectors) for value in center]


    norm = math.sqrt(sum(v * v for v in avg_vector))
    if norm == 0:
        return avg_vector
    return [v / norm for v in avg_vector]
