"""Sentence embedding wrapper around all-MiniLM-L6-v2.

Uses sentence-transformers directly. ONNX export is a future optimization.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, prompts: str | Iterable[str], batch_size: int = 32) -> np.ndarray:
        if isinstance(prompts, str):
            prompts = [prompts]
        else:
            prompts = list(prompts)
        embeddings = self.model.encode(
            prompts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
