from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from .contract import (
    load_and_validate_leads,
    load_and_validate_messages,
)
import pandas as pd
from .metrics import classification_metrics
from .m3 import build_m3_features
from .model_data import load_m2_dataset
from .pgvector_store import store_lead_embeddings


RANDOM_SEED = 42
M3_MODEL_VERSION = "m3-1.0.0"


def _load_m2_lightgbm_params(
    model_dir: str | Path,
) -> dict[str, Any]:
    """
    Reuse the M2 champion LightGBM configuration for both sides
    of the ablation so the text comparison is fair.

    The M2 artifact is treated only as a source of hyperparameters;
    both ablation models are retrained from scratch.
    """

    bundle_path = (
        Path(model_dir)
        / "model_bundle.joblib"
    )

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"M2 model bundle not found: {bundle_path}\n"
            "Run M2 training first."
        )

    bundle = joblib.load(bundle_path)

    model = bundle.get("lightgbm_model")

    if model is None:
        raise ValueError(
            "M2 bundle does not contain lightgbm_model"
        )

    params = model.get_params()

    allowed = {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "max_bin",
    }

    selected = {
        key: params[key]
        for key in allowed
        if key in params
    }

    selected["objective"] = "binary"
    selected["random_state"] = RANDOM_SEED
    selected["n_jobs"] = -1
    selected["verbosity"] = -1

    best_iteration = int(
        model.best_iteration_
        or params.get("n_estimators", 200)
    )

    selected["n_estimators"] = best_iteration

    return selected


def _train_and_score_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    train_ids: pd.Index,
    validation_ids: pd.Index,
    params: dict[str, Any],
) -> dict[str, float]:

    X_train = X.loc[train_ids]
    y_train = y.loc[train_ids]

    X_validation = X.loc[validation_ids]
    y_validation = y.loc[validation_ids]

    from .train import build_preprocessor

    preprocessor = build_preprocessor(X_train)

    X_train_t = preprocessor.fit_transform(
        X_train
    )

    X_validation_t = preprocessor.transform(
        X_validation
    )

    model = lgb.LGBMClassifier(
        **params
    )

    model.fit(
        X_train_t,
        y_train,
    )

    probability = model.predict_proba(
        X_validation_t
    )[:, 1]

    metrics = classification_metrics(
        y_validation,
        probability,
        include_calibration=False,
    )

    return {
        key: float(value)
        for key, value in metrics.items()
    }


def run_m3_ablation(
    leads_path: str,
    messages_path: str,
    calls_path: str,
    stages_path: str,
    localities_path: str,
    export_date: str,
    model_dir: str,
    output_dir: str,
) -> dict[str, Any]:

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load the exact M2 dataset and exact M2 time split.
    # ------------------------------------------------------------------

    dataset = load_m2_dataset(
        leads_path=leads_path,
        messages_path=messages_path,
        calls_path=calls_path,
        stages_path=stages_path,
        localities_path=localities_path,
        export_date=export_date,
    )

    X_m2 = dataset.X.copy()
    y = dataset.y.copy()
    print("M2 feature rows:", len(X_m2))
    print("M2 unique lead IDs:", X_m2.index.nunique())
    print("M2 index unique:", X_m2.index.is_unique)
    train_ids = dataset.split.train
    validation_ids = dataset.split.validation

    # ------------------------------------------------------------------
    # Load raw data for M3 text construction.
    # ------------------------------------------------------------------

    leads = load_and_validate_leads(
        str(leads_path)
    )

    messages = load_and_validate_messages(
        str(messages_path)
    )

    # Keep exactly the leads used by the M2 dataset.
    mature_ids = dataset.lead_ids

    leads = leads[
        leads["lead_id"].isin(mature_ids)
    ].copy()

    messages = messages[
        messages["lead_id"].isin(mature_ids)
    ].copy()

       # ------------------------------------------------------------------
    # Put leads in EXACTLY the same order as the M2 feature matrix.
    # This guarantees that:
    #   X_m2 row
    #   M3 text features
    #   raw embedding
    # all refer to the same lead.
    # ------------------------------------------------------------------

    ordered_leads = (
        leads
        .set_index("lead_id")
        .reindex(X_m2.index)
        .reset_index()
    )

    if ordered_leads["lead_id"].isna().any():
        raise ValueError(
            "Some M2 lead IDs could not be matched to the raw leads."
        )

    if not ordered_leads["lead_id"].is_unique:
        raise ValueError(
            "Duplicate lead IDs found after ordering M3 leads."
        )

    # ------------------------------------------------------------------
    # Train-only PCA mask.
    # ------------------------------------------------------------------

    train_mask = (
        ordered_leads["lead_id"]
        .isin(train_ids)
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # Build M3 ONCE.
    #
    # This single call:
    #   - loads e5-small once
    #   - builds point-in-time text
    #   - generates embeddings once
    #   - fits PCA on training rows only
    #   - creates PCA text features
    # ------------------------------------------------------------------

    m3_features, text_rows = build_m3_features(
        leads=ordered_leads,
        messages=messages,
        train_mask=train_mask,
    )

    m3_text_features = (
        m3_features
        .text_feature_matrix
        .reindex(X_m2.index)
    )

    if m3_text_features.isna().all(axis=1).any():
        missing_ids = m3_text_features[
            m3_text_features.isna().all(axis=1)
        ].index.tolist()

        raise ValueError(
            "Missing M3 text features for lead IDs: "
            f"{missing_ids[:10]}"
        )

    # ------------------------------------------------------------------
    # Raw embeddings are ALREADY available from build_m3_features().
    # Do NOT call build_raw_embeddings() or build_m3_features() again.
    # ------------------------------------------------------------------

    ordered_embeddings = m3_features.raw_embeddings

    if len(ordered_embeddings) != len(X_m2):
        raise ValueError(
            "Embedding row count does not match M2 rows: "
            f"{len(ordered_embeddings)} != {len(X_m2)}"
        )

    if ordered_embeddings.shape[1] != 384:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{ordered_embeddings.shape[1]}"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        m3_features.text_feature_matrix[
            [f"text_pca_{i:02d}" for i in range(16)]
        ].describe().T[
            ["mean", "std", "min", "max"]
        ]
    )

    print("PCA explained variance:")
    print(
        m3_features.pca.explained_variance_ratio_
    )

    print(
        "Total explained variance:",
        m3_features.pca.explained_variance_ratio_.sum(),
    )

    # ------------------------------------------------------------------
    # Store the SAME embeddings used by M3 in Neon / pgvector.
    # ------------------------------------------------------------------

    asof_times = (
        ordered_leads
        .set_index("lead_id")["created_at"]
        + pd.Timedelta(hours=24)
    )

    stored_count = store_lead_embeddings(
    lead_ids=X_m2.index.tolist(),
    embeddings=ordered_embeddings,
    asof_times=asof_times,
    has_text=(
        m3_features.text_rows[
            "has_inbound_text_before_t"
        ]
        .reindex(X_m2.index)
        .to_numpy()
    ),
    model_version=M3_MODEL_VERSION,
    )

    print(
        f"Embeddings stored in Neon: {stored_count}"
    )
    # ------------------------------------------------------------------
    # Build WITH-TEXT matrix.
    #
    # M2 already has has_inbound_text_before_t.
    # Do not add that same column twice.
    # ------------------------------------------------------------------

    semantic_text_features = m3_text_features.drop(
        columns=[
            "has_inbound_text_before_t"
        ],
        errors="ignore",
    )

    X_with_text = X_m2.join(
        semantic_text_features,
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # Load shared M2 LightGBM configuration.
    # ------------------------------------------------------------------

    lgbm_params = _load_m2_lightgbm_params(
        model_dir
    )

    # ------------------------------------------------------------------
    # WITHOUT TEXT
    # ------------------------------------------------------------------

    without_text_metrics = _train_and_score_lightgbm(
        X=X_m2,
        y=y,
        train_ids=train_ids,
        validation_ids=validation_ids,
        params=lgbm_params,
    )

    # ------------------------------------------------------------------
    # WITH TEXT
    # ------------------------------------------------------------------

    with_text_metrics = _train_and_score_lightgbm(
        X=X_with_text,
        y=y,
        train_ids=train_ids,
        validation_ids=validation_ids,
        params=lgbm_params,
    )

    # ------------------------------------------------------------------
    # Save comparison.
    # ------------------------------------------------------------------

    rows = []

    for model_name, metrics in [
        (
            "without_text",
            without_text_metrics,
        ),
        (
            "with_text",
            with_text_metrics,
        ),
    ]:

        rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        output_path / "ablation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save metadata.
    # ------------------------------------------------------------------

    metadata = {
        "module": "M3",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384,
        "pca_components": 16,
        "model_version": M3_MODEL_VERSION,
        "train_rows": int(len(train_ids)),
        "validation_rows": int(len(validation_ids)),
        "same_time_split": True,
        "pca_fit_only_on_training_rows": True,
        "text_definition": (
            "lead-owned inbound messages where "
            "created_at <= sent_at < created_at + 24h"
        ),
        "lgbm_params": lgbm_params,
        "results": rows,
    }

    (
        output_path / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("M3 TEXT ABLATION")
    print("=" * 70)

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    has_text = (
        m3_features.text_rows[
            "has_inbound_text_before_t"
        ]
        .reindex(X_m2.index)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    print(
        f"Embeddings stored in Neon: {stored_count}"
    )

    print()
    print(
        f"Output: {output_path / 'ablation.csv'}"
    )

    return {
        "comparison": rows,
        "output_dir": str(output_path),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "LeadIQ M3: embeddings, pgvector, "
            "and with/without-text ablation"
        )
    )

    parser.add_argument(
        "--leads",
        required=True,
    )

    parser.add_argument(
        "--messages",
        required=True,
    )

    parser.add_argument(
        "--calls",
        required=True,
    )

    parser.add_argument(
        "--stages",
        required=True,
    )

    parser.add_argument(
        "--localities",
        required=True,
    )

    parser.add_argument(
        "--export-date",
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        default="artifacts/m2",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/m3",
    )

    args = parser.parse_args()

    run_m3_ablation(
        leads_path=args.leads,
        messages_path=args.messages,
        calls_path=args.calls,
        stages_path=args.stages,
        localities_path=args.localities,
        export_date=args.export_date,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()