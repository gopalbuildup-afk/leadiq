from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import psycopg
from fastapi import HTTPException
from pydantic import BaseModel, Field

from leadiq.contract import (
    load_and_validate_calls,
    load_and_validate_leads,
    load_and_validate_localities,
    load_and_validate_messages,
)
from leadiq.features import build_feature_matrix
from leadiq.score import _compute_asof
from leadiq.versioning import (
    get_champion_version,
    get_model_card,
    get_duplicate_cluster_size,
    initialise_schema,
    append_lead_scores,
)


MODEL_VERSION = "m2-1.0.0"


class ScoreRequest(BaseModel):
    lead_id: str = Field(..., min_length=1)


class ReasonResponse(BaseModel):
    en: str
    hi: str


class ScoreResponse(BaseModel):
    lead_id: str
    asof: str
    probability: float = Field(..., ge=0.0, le=1.0)
    band: str
    reasons: list[ReasonResponse]
    model_version: str
    duplicate_cluster_size: int = 1


class BatchScoreRequest(BaseModel):
    lead_ids: list[str] = Field(..., min_length=1, max_length=500)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


class ModelCardResponse(BaseModel):
    version: str
    training_window: str
    data_sha256: str
    metrics: dict[str, Any]
    feature_list: list[str]
    status: str
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Database helpers for single-lead scoring
# ---------------------------------------------------------------------------

def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return database_url


def _get_lead_data(lead_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch a single lead's raw data and build the feature matrix.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leads_path = os.path.join(project_root, "data", "leads.csv")
    messages_path = os.path.join(project_root, "data", "messages.csv")
    calls_path = os.path.join(project_root, "data", "calls.csv")
    localities_path = os.path.join(project_root, "data", "localities.csv")
    clusters_path = os.path.join(project_root, "artifacts", "m4", "lead_clusters.csv")

    try:
        from leadiq.contract import (
            load_and_validate_calls,
            load_and_validate_leads,
            load_and_validate_localities,
            load_and_validate_messages,
        )
        from leadiq.m4_features import build_earlier_enquiries_feature

        leads = load_and_validate_leads(leads_path)
        messages = load_and_validate_messages(messages_path)
        calls = load_and_validate_calls(calls_path)
        localities = load_and_validate_localities(localities_path)

        if lead_id not in leads["lead_id"].values:
            raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")

        leads_filtered = leads[leads["lead_id"] == lead_id]
        messages_filtered = messages[messages["lead_id"] == lead_id]
        calls_filtered = calls[calls["lead_id"] == lead_id]

        # Load M4 clusters for earlier_enquiries_count feature
        m4_features = None
        if os.path.exists(clusters_path):
            try:
                import pandas as pd
                cluster_df = pd.read_csv(clusters_path)
                if "lead_id" in cluster_df.columns and "cluster_id" in cluster_df.columns:
                    m4_features = build_earlier_enquiries_feature(leads_filtered, cluster_df)
            except Exception:
                pass

        # Build feature matrix for the single lead
        X = build_feature_matrix(
            leads_df=leads_filtered,
            messages_df=messages_filtered,
            calls_df=calls_filtered,
            localities_df=localities,
            m4_features=m4_features,
        )

        return X, leads_filtered

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _load_bundle() -> dict:
    """Load the model bundle from artifacts."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "artifacts", "m2", "model_bundle.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model bundle not found. Train M2 first.")
    return joblib.load(model_path)


def _generate_lead_reasons(
    model,
    preprocessor,
    X_row: pd.Series,
    x_transformed_row: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, str]]:
    """Generate top-3 reasons for a single lead."""
    from leadiq.score import _get_top_3_reasons
    return _get_top_3_reasons(
        model, preprocessor, X_row, x_transformed_row, feature_names
    )


def score_single_lead(lead_id: str) -> dict[str, Any]:
    """Score a single enquiry and return the response dict."""
    # Ensure schema exists
    try:
        initialise_schema()
    except Exception:
        pass

    # Load bundle and data
    bundle = _load_bundle()
    X, leads_filtered = _get_lead_data(lead_id)

    if X.empty:
        raise HTTPException(status_code=404, detail=f"No features for lead {lead_id}")

    preprocessor = bundle["preprocessor"]
    champion_model = bundle["champion_model"]
    calibrator = bundle["calibrator"]
    thresholds = bundle["band_thresholds"]

    X_transformed = preprocessor.transform(X)
    raw_probability = champion_model.predict_proba(X_transformed)[:, 1]
    calibrated_probability = np.clip(
        np.asarray(calibrator.predict(raw_probability), dtype=float),
        0.0, 1.0,
    )

    # Assign bands
    bands = []
    for p in calibrated_probability:
        if p >= thresholds["p1_min_probability"]:
            bands.append("P1")
        elif p >= thresholds["p2_min_probability"]:
            bands.append("P2")
        else:
            bands.append("P3")

    feature_names = list(preprocessor.get_feature_names_out())
    lead_str = str(lead_id)

    # Get reasons
    x_row = X.iloc[0]
    x_transformed_row = X_transformed[0]
    reasons = _generate_lead_reasons(
        champion_model, preprocessor, x_row, x_transformed_row, feature_names
    )

    # Get duplicate cluster size
    try:
        cluster_size = get_duplicate_cluster_size(lead_str)
    except Exception:
        cluster_size = 1

    # Compute asof
    created_at = leads_filtered.iloc[0]["created_at"]
    asof = _compute_asof(created_at)

    score_entry = {
        "lead_id": lead_str,
        "asof": asof.isoformat(),
        "probability": float(calibrated_probability[0]),
        "band": bands[0],
        "reasons": reasons,
        "model_version": bundle["version"],
    }

    # Persist to database
    try:
        append_lead_scores([score_entry])
    except Exception:
        pass

    return {
        "lead_id": lead_str,
        "asof": asof.isoformat(),
        "probability": float(calibrated_probability[0]),
        "band": bands[0],
        "reasons": reasons,
        "model_version": bundle["version"],
        "duplicate_cluster_size": cluster_size,
    }


def get_model_card_api() -> dict[str, Any]:
    """Get the model card from the database."""
    initialise_schema()
    card = get_model_card()
    if card is None:
        raise HTTPException(status_code=404, detail="No model registered")
    return card


def healthz_api() -> dict[str, Any]:
    """Health check including model loaded status."""
    try:
        initialise_schema()
        champion = get_champion_version()
        return {
            "status": "ok",
            "database_configured": bool(os.getenv("DATABASE_URL")),
            "model_loaded": champion is not None,
            "champion_version": champion,
        }
    except Exception:
        return {
            "status": "degraded",
            "database_configured": bool(os.getenv("DATABASE_URL")),
            "model_loaded": False,
        }
