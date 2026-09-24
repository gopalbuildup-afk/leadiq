from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calibration import assign_band
from .contract import (
    load_and_validate_calls,
    load_and_validate_leads,
    load_and_validate_localities,
    load_and_validate_messages,
    load_and_validate_stage_history,
)
from .features import build_feature_matrix
from .reasons import reason_for_feature
from .versioning import get_duplicate_cluster_size


DEFAULT_MODEL_DIR = Path("artifacts/m2")
DEFAULT_LOCALITIES = Path("data/localities.csv")
DEFAULT_M4_CLUSTERS = Path("artifacts/m4/lead_clusters.csv")


def load_bundle(model_dir: str | Path) -> dict:
    """Load the trained M2 model bundle."""
    model_path = Path(model_dir) / "model_bundle.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {model_path}. "
            "Train M2 first or pass --model-dir."
        )

    bundle = joblib.load(model_path)

    required = {
        "version",
        "champion_name",
        "preprocessor",
        "champion_model",
        "calibrator",
        "band_thresholds",
    }
    missing = required.difference(bundle.keys())

    if missing:
        raise ValueError(
            f"Model bundle is missing required keys: {sorted(missing)}"
        )

    return bundle


def _compute_asof(created_at: pd.Timestamp) -> pd.Timestamp:
    """Scoring moment: t = created_at + 24 hours."""
    return created_at + pd.Timedelta(hours=24)


def _get_top_3_reasons(
    model,
    preprocessor,
    x_row: pd.Series,
    x_transformed_row: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, str]]:
    """Generate top-3 reasons for a single lead's score."""
    try:
        return _get_shap_reasons(
            model, x_transformed_row, feature_names, x_row
        )
    except Exception:
        pass

    return _get_importance_reasons(
        model, preprocessor, x_row, feature_names
    )


def _get_shap_reasons(
    model,
    x_transformed: np.ndarray,
    feature_names: list[str],
    x_row: pd.Series,
) -> list[dict[str, str]]:
    """Try to use SHAP for reason generation."""
    try:
        import shap
    except ImportError:
        raise RuntimeError("shap not available")

    try:
        explainer = shap.TreeExplainer(model)
        dense = x_transformed.toarray() if hasattr(x_transformed, "toarray") else np.asarray(x_transformed)
        explanation = explainer(dense, check_additivity=False)
        values = explanation.values
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[0]
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("Unexpected SHAP shape")
    except Exception:
        raise RuntimeError("SHAP explanation failed")

    contributions = values[0] if values.ndim == 2 else values
    order = np.argsort(np.abs(contributions))[::-1]

    reasons = []
    for feature_idx in order:
        if len(reasons) >= 3:
            break
        if abs(contributions[feature_idx]) < 1e-10:
            continue
        fname = feature_names[feature_idx] if feature_idx < len(feature_names) else str(feature_idx)
        reason = reason_for_feature(fname, x_row)
        if reason is None:
            continue
        reasons.append({"en": reason[0], "hi": reason[1]})

    while len(reasons) < 3:
        reasons.append({"en": "No additional mapped explanation", "hi": "Aur koi mapped explanation nahi"})

    return reasons[:3]


def _get_importance_reasons(
    model,
    preprocessor,
    x_row: pd.Series,
    feature_names: list[str],
) -> list[dict[str, str]]:
    """Fallback reason generation using model feature importances."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            raise RuntimeError("No feature importances available")

        transformed_names = list(preprocessor.get_feature_names_out()) if hasattr(preprocessor, "get_feature_names_out") else feature_names
        ranked_idx = np.argsort(importances)[::-1][:50]

        reasons = []
        for idx in ranked_idx:
            if len(reasons) >= 3:
                break
            fname = transformed_names[idx] if idx < len(transformed_names) else str(idx)
            reason = reason_for_feature(fname, x_row)
            if reason is None:
                continue
            reasons.append({"en": reason[0], "hi": reason[1]})

        while len(reasons) < 3:
            reasons.append({"en": "No additional mapped explanation", "hi": "Aur koi mapped explanation nahi"})

        return reasons[:3]
    except Exception:
        return [
            {"en": "No explanation available", "hi": "Koi explanation nahi"},
            {"en": "No explanation available", "hi": "Koi explanation nahi"},
            {"en": "No explanation available", "hi": "Koi explanation nahi"},
        ]


def _compute_reasons_batch(
    model,
    preprocessor,
    X: pd.DataFrame,
    X_transformed: np.ndarray,
    feature_names: list[str],
) -> list[list[dict[str, str]]]:
    """Compute top-3 reasons for all leads in batch."""
    # Try SHAP for all at once
    try:
        reasons_batch = _get_shap_reasons_batch(
            model, X_transformed, feature_names, X
        )
        return reasons_batch
    except Exception:
        pass

    # Fallback: compute per-row
    return [
        _get_top_3_reasons(model, preprocessor, X.loc[lead_id], X_transformed[i], feature_names)
        for i, lead_id in enumerate(X.index)
    ]


def _get_shap_reasons_batch(
    model,
    X_transformed: np.ndarray,
    feature_names: list[str],
    X: pd.DataFrame,
) -> list[list[dict[str, str]]]:
    """Batch SHAP computation for all rows."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else np.asarray(X_transformed)
        explanation = explainer(dense, check_additivity=False)
        values = explanation.values
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[0]
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("Unexpected SHAP shape")
    except Exception:
        try:
            import numpy as np
            import pandas as pd
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = np.ones(len(feature_names))
            dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else np.asarray(X_transformed)
            values = dense * importances
        except Exception:
            values = np.zeros((len(X), len(feature_names)))

    reasons_batch = []
    for i in range(len(X)):
        contributions = values[i]
        order = np.argsort(np.abs(contributions))[::-1]
        x_row = X.iloc[i]
        reasons = []
        for feature_idx in order:
            if len(reasons) >= 3:
                break
            if abs(contributions[feature_idx]) < 1e-10:
                continue
            fname = feature_names[feature_idx] if feature_idx < len(feature_names) else str(feature_idx)
            reason = reason_for_feature(fname, x_row)
            if reason is None:
                continue
            reasons.append({"en": reason[0], "hi": reason[1]})
        while len(reasons) < 3:
            reasons.append({"en": "No additional mapped explanation", "hi": "Aur koi mapped explanation nahi"})
        reasons_batch.append(reasons[:3])

    return reasons_batch


def build_scoring_features(
    leads_path: str | Path,
    messages_path: str | Path,
    calls_path: str | Path,
    localities_path: str | Path,
    stages_path: str | Path | None = None,
    clusters_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate unseen scoring data and build the feature matrix."""
    leads = load_and_validate_leads(str(leads_path))
    messages = load_and_validate_messages(str(messages_path))
    calls = load_and_validate_calls(str(calls_path))
    localities = load_and_validate_localities(str(localities_path))

    if stages_path is not None:
        load_and_validate_stage_history(str(stages_path))

    # Load M4 clusters for earlier_enquiries_count feature
    m4_features = None
    if clusters_path is not None and Path(clusters_path).exists():
        try:
            import pandas as pd
            from leadiq.m4_features import build_earlier_enquiries_feature
            cluster_df = pd.read_csv(clusters_path)
            if "lead_id" in cluster_df.columns and "cluster_id" in cluster_df.columns:
                m4_features = build_earlier_enquiries_feature(leads, cluster_df)
        except Exception:
            pass

    X = build_feature_matrix(
        leads_df=leads,
        messages_df=messages,
        calls_df=calls,
        localities_df=localities,
        m4_features=m4_features,
    )

    if X.empty:
        raise ValueError("No leads were available for scoring.")

    if not X.index.is_unique:
        duplicated = X.index[X.index.duplicated()].unique().tolist()
        raise ValueError(
            "Duplicate lead_id values found in scoring data: "
            f"{duplicated[:10]}"
        )

    return X, leads


def score_dataframe(
    X: pd.DataFrame,
    leads: pd.DataFrame,
    bundle: dict,
    clusters_path: str | Path | None = None,
) -> pd.DataFrame:
    """Transform features, predict, calibrate, assign bands, and generate reasons."""
    preprocessor = bundle["preprocessor"]
    champion_model = bundle["champion_model"]
    calibrator = bundle["calibrator"]
    thresholds = bundle["band_thresholds"]

    X_transformed = preprocessor.transform(X)

    raw_probability = champion_model.predict_proba(X_transformed)[:, 1]
    calibrated_probability = np.asarray(calibrator.predict(raw_probability), dtype=float)
    calibrated_probability = np.clip(calibrated_probability, 0.0, 1.0)

    bands = assign_band(calibrated_probability, thresholds)
    feature_names = list(preprocessor.get_feature_names_out())

    # Compute reasons in batch
    reasons_list = _compute_reasons_batch(
        champion_model, preprocessor, X, X_transformed, feature_names
    )

    # Build asof timestamps
    leads_indexed = leads.set_index("lead_id")
    asof_times = pd.Series(
        {lid: _compute_asof(leads_indexed.loc[lid, "created_at"]) for lid in X.index},
        index=X.index,
    )

    # Build cluster size lookup
    cluster_sizes: dict[str, int] = {}
    if clusters_path is not None and Path(clusters_path).exists():
        try:
            clusters_df = pd.read_csv(clusters_path)
            if "lead_id" in clusters_df.columns and "cluster_size" in clusters_df.columns:
                cluster_sizes = dict(
                    zip(clusters_df["lead_id"].astype(str), clusters_df["cluster_size"].astype(int))
                )
        except Exception:
            pass

    duplicate_cluster_sizes = []
    for lead_id in X.index:
        lead_str = str(lead_id)
        if lead_str in cluster_sizes:
            duplicate_cluster_sizes.append(cluster_sizes[lead_str])
        else:
            try:
                duplicate_cluster_sizes.append(get_duplicate_cluster_size(lead_str))
            except Exception:
                duplicate_cluster_sizes.append(1)

    scores = pd.DataFrame({
        "lead_id": X.index.astype(str),
        "asof": asof_times.values,
        "probability": calibrated_probability,
        "band": bands,
        "reasons": reasons_list,
        "model_version": bundle["version"],
        "duplicate_cluster_size": duplicate_cluster_sizes,
    })

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeadIQ batch scoring"
    )

    parser.add_argument("--leads", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--calls", required=True)
    parser.add_argument("--stages", required=False, default=None)

    parser.add_argument(
        "--localities",
        default=str(DEFAULT_LOCALITIES),
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
    )

    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--clusters",
        default=str(DEFAULT_M4_CLUSTERS),
    )

    args = parser.parse_args()

    bundle = load_bundle(args.model_dir)

    X, leads = build_scoring_features(
        leads_path=args.leads,
        messages_path=args.messages,
        calls_path=args.calls,
        localities_path=args.localities,
        stages_path=args.stages,
        clusters_path=args.clusters if Path(args.clusters).exists() else None,
    )

    scores = score_dataframe(
        X, leads, bundle,
        clusters_path=args.clusters if Path(args.clusters).exists() else None,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = scores.copy()
    output_df["reasons"] = output_df["reasons"].apply(json.dumps, ensure_ascii=False)
    output_df["asof"] = output_df["asof"].astype(str)
    output_df.to_csv(output_path, index=False)

    print(f"Model: {bundle['version']}")
    print(f"Champion: {bundle['champion_name']}")
    print(f"Scored leads: {len(scores)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
