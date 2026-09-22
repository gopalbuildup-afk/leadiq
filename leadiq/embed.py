from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384


@dataclass
class Embedder:
    model_name: str = MODEL_NAME

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for M3. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        prepared = [
            text if text.startswith("query: ") else f"query: {text}"
            for text in texts
        ]

        vectors = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                f"Expected 2-D embedding matrix, got shape={vectors.shape}"
            )

        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional embeddings, "
                f"got {vectors.shape[1]}"
            )

        return vectors