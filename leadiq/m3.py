from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .embed import EMBEDDING_DIM, Embedder
from .text_features import build_lead_texts


PCA_COMPONENTS = 16


@dataclass
class M3TextFeatures:
    pca: PCA
    feature_columns: list[str]
    text_feature_matrix: pd.DataFrame
    raw_embeddings: np.ndarray
    text_rows: pd.DataFrame


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Load the embedding model once per Python process.

    This avoids repeated model loading when M3 functions are reused.
    """
    return Embedder()


def build_raw_embeddings(
    leads: pd.DataFrame,
    messages: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build one normalized 384-d embedding per enquiry.

    For enquiries without inbound text, the returned matrix contains
    an internal zero placeholder. Those rows are NOT persisted to
    pgvector; they receive only the explicit text-presence flag.
    """

    embedder = get_embedder()

    text_rows = build_lead_texts(
        leads=leads,
        messages=messages,
        tokenizer=embedder.model.tokenizer,
    )

    active = (
        text_rows["has_inbound_text_before_t"] == 1
    )

    embeddings = np.zeros(
        (
            len(text_rows),
            EMBEDDING_DIM,
        ),
        dtype=np.float32,
    )

    if active.any():
        embeddings[
            active.to_numpy()
        ] = embedder.encode(
            text_rows.loc[
                active,
                "text",
            ].tolist()
        )

    return text_rows, embeddings


def fit_pca_on_training(
    embedding_matrix: np.ndarray,
    train_mask: np.ndarray,
) -> PCA:

    if embedding_matrix.ndim != 2:
        raise ValueError(
            "Expected 2-D embedding matrix, "
            f"got {embedding_matrix.shape}"
        )

    fit_rows = embedding_matrix[
        train_mask
    ]

    if fit_rows.shape[0] < PCA_COMPONENTS:
        raise ValueError(
            "Not enough training text rows to fit "
            f"{PCA_COMPONENTS} PCA components: "
            f"{fit_rows.shape[0]}"
        )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=42,
    )

    pca.fit(fit_rows)

    return pca


def transform_with_pca(
    embedding_matrix: np.ndarray,
    pca: PCA,
    has_text: pd.Series,
) -> pd.DataFrame:

    transformed = np.zeros(
        (
            len(embedding_matrix),
            PCA_COMPONENTS,
        ),
        dtype=np.float32,
    )

    active = has_text.to_numpy(
        dtype=bool
    )

    if active.any():
        transformed[active] = (
            pca.transform(
                embedding_matrix[active]
            )
        )

    columns = [
        f"text_pca_{i:02d}"
        for i in range(PCA_COMPONENTS)
    ]

    result = pd.DataFrame(
        transformed,
        index=has_text.index,
        columns=columns,
    )

    result["has_inbound_text_before_t"] = (
        has_text.astype(int)
    )

    return result


def build_m3_features(
    leads: pd.DataFrame,
    messages: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[M3TextFeatures, pd.DataFrame]:

    text_rows, embeddings = (
        build_raw_embeddings(
            leads,
            messages,
        )
    )

    has_text = text_rows[
        "has_inbound_text_before_t"
    ]

    pca = fit_pca_on_training(
        embedding_matrix=embeddings,
        train_mask=(
            train_mask
            & has_text.to_numpy(
                dtype=bool
            )
        ),
    )

    text_features = transform_with_pca(
        embedding_matrix=embeddings,
        pca=pca,
        has_text=has_text,
    )

    feature_columns = [
        f"text_pca_{i:02d}"
        for i in range(PCA_COMPONENTS)
    ]

    feature_columns.append(
        "has_inbound_text_before_t"
    )

    return (
        M3TextFeatures(
            pca=pca,
            feature_columns=feature_columns,
            text_feature_matrix=text_features,
            raw_embeddings=embeddings,
            text_rows=text_rows,
        ),
        text_rows,
    )