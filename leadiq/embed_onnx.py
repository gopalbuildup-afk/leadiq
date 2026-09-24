from __future__ import annotations

import os
import numpy as np
from .embed import EMBEDDING_DIM, MODEL_NAME

DEFAULT_ONNX_DIR = os.path.join("artifacts", "m3", "onnx")
DEFAULT_ONNX_PATH = os.path.join(DEFAULT_ONNX_DIR, "model.onnx")


class OnnxEmbedder:
    """ONNX Runtime twin of :class:`leadiq.embed.Embedder`.

    Runs the exported ``model.onnx`` (plus its ``model.onnx.data`` weights)
    with the same pre/post-processing as the offline sentence-transformers
    path: ``query: `` prefix, attention-mask-weighted mean pooling, L2
    normalisation.
    """

    def __init__(
        self,
        onnx_path: str = DEFAULT_ONNX_PATH,
        model_name: str = MODEL_NAME,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX embeddings. "
                "Install it with: pip install onnxruntime"
            ) from exc
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the ONNX tokenizer."
            ) from exc

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}. "
                "Run: python scripts/export_m3_onnx.py"
            )
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        prepared = [
            text if text.startswith("query: ") else f"query: {text}"
            for text in texts
        ]
        out: list[np.ndarray] = []
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            last_hidden = self.session.run(
                ["last_hidden_state"],
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            out.append(_mean_pool(last_hidden, enc["attention_mask"]))
        vectors = (
            np.concatenate(out, axis=0) if out else np.zeros((0, EMBEDDING_DIM))
        )
        return np.asarray(vectors, dtype=np.float32)


def _mean_pool(
    last_hidden_state: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Attention-mask-weighted mean pooling + L2 normalisation."""
    mask = np.asarray(attention_mask, dtype=np.float32)[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (pooled / norms).astype(np.float32)
