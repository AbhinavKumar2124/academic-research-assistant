"""Sentence Transformers runs entirely on the local machine."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str) -> list[list[float]]:
    return get_model(model_name).encode(texts, normalize_embeddings=True).tolist()
