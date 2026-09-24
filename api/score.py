from __future__ import annotations

import os
import threading
import time
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

from leadiq.calibration import assign_band
from leadiq.contract import (
    load_and_validate_calls,
    load_and_validate_leads,
    load_and_validate_localities,
    load_and_validate_messages,
)
from leadiq.features import build_feature_matrix
from leadiq.score import _compute_asof, _compute_reasons_batch
from leadiq.versioning import (
    enqueue_scores,
    get_champion_version,
    get_model_card,
    initialise_schema,
    lookup_sizes,
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
# Process-level caches.
#
# Scoring the call list fires hundreds of scores per minute. Reloading the
# model bundle and re-validating every CSV from disk on each request made a
# 500-lead batch take the better part of an hour. Everything below is loaded
# once per API worker process and reused; the underlying files only change
# when a module is re-run, which also restarts the API.
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_schema_ready = False
_bundle: dict | None = None
_frames: dict[str, pd.DataFrame] | None = None
_m4_features: pd.DataFrame | None = None
_m4_attempted = False

_api_log = logging.getLogger("leadiq.api")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_schema_once() -> None:
    """Create Module 5 tables on first use only (DDL per request is slow)."""
    global _schema_ready
    if _schema_ready:
        return
    with _init_lock:
        if _schema_ready:
            return
        try:
            initialise_schema()
        except Exception:
            pass
        _schema_ready = True


def _get_bundle() -> dict:
    """Load the M2 model bundle once per process."""
    global _bundle
    if _bundle is None:
        with _init_lock:
            if _bundle is None:
                model_path = os.path.join(
                    _project_root(), "artifacts", "m2", "model_bundle.joblib"
                )
                if not os.path.exists(model_path):
                    raise HTTPException(
                        status_code=503,
                        detail="Model bundle not found. Train M2 first.",
                    )
                _bundle = joblib.load(model_path)
    assert _bundle is not None
    return _bundle


def _get_frames() -> dict[str, pd.DataFrame]:
    """Load and validate the scoring CSVs once per process."""
    global _frames
    if _frames is None:
        with _init_lock:
            if _frames is None:
                root = _project_root()
                _frames = {
                    "leads": load_and_validate_leads(
                        os.path.join(root, "data", "leads.csv")
                    ),
                    "messages": load_and_validate_messages(
                        os.path.join(root, "data", "messages.csv")
                    ),
                    "calls": load_and_validate_calls(
                        os.path.join(root, "data", "calls.csv")
                    ),
                    "localities": load_and_validate_localities(
                        os.path.join(root, "data", "localities.csv")
                    ),
                }
    assert _frames is not None
    return _frames


def _get_m4_features(leads: pd.DataFrame) -> pd.DataFrame | None:
    """Build the earlier-enquiries feature over the full lead set, once."""
    global _m4_features, _m4_attempted
    if not _m4_attempted:
        with _init_lock:
            if not _m4_attempted:
                _m4_attempted = True
                try:
                    from leadiq.m4_features import (
                        build_earlier_enquiries_feature,
                    )

                    cluster_df = None
                    clusters_path = os.path.join(
                        _project_root(), "artifacts", "m4", "lead_clusters.csv"
                    )
                    if os.path.exists(clusters_path):
                        cluster_df = pd.read_csv(clusters_path)
                    else:
                        try:
                            from leadiq.versioning import _pool_instance
                            with _pool_instance().connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("SELECT cluster_id, lead_id, is_canonical FROM lead_clusters")
                                    rows = cur.fetchall()
                                    if rows:
                                        cluster_df = pd.DataFrame(rows, columns=["cluster_id", "lead_id", "is_canonical"])
                        except Exception:
                            cluster_df = None

                    if (
                        cluster_df is not None
                        and "lead_id" in cluster_df.columns
                        and "cluster_id" in cluster_df.columns
                    ):
                        _m4_features = build_earlier_enquiries_feature(
                            leads, cluster_df
                        )
                except Exception:
                    _m4_features = None

    if _m4_features is None and leads is not None and not leads.empty:
        df_default = pd.DataFrame({
            "lead_id": leads["lead_id"].astype(str),
            "earlier_enquiries_count": 0
        }).set_index("lead_id")
        return df_default

    return _m4_features


# ---------------------------------------------------------------------------
# Shared scoring core: one feature matrix -> one predict -> one batch SHAP.
# ---------------------------------------------------------------------------

def _score_frame(
    X: pd.DataFrame,
    leads_filtered: pd.DataFrame,
    bundle: dict,
) -> list[dict[str, Any]]:
    """Pure compute: no DB. Cluster sizes resolve separately in one checkout."""
    preprocessor = bundle["preprocessor"]
    champion_model = bundle["champion_model"]
    calibrator = bundle["calibrator"]
    thresholds = bundle["band_thresholds"]

    X_transformed = preprocessor.transform(X)
    raw_probability = champion_model.predict_proba(X_transformed)[:, 1]
    calibrated = np.clip(
        np.asarray(calibrator.predict(raw_probability), dtype=float),
        0.0,
        1.0,
    )
    bands = assign_band(calibrated, thresholds)
    feature_names = list(preprocessor.get_feature_names_out())
    reasons_list = _compute_reasons_batch(
        champion_model, preprocessor, X, X_transformed, feature_names
    )

    leads_by_id = leads_filtered.set_index("lead_id")
    results: list[dict[str, Any]] = []
    for i, lead_id in enumerate(X.index):
        lead_str = str(lead_id)
        asof = _compute_asof(leads_by_id.loc[lead_str, "created_at"])
        results.append(
            {
                "lead_id": lead_str,
                "asof": asof.isoformat(),
                "probability": float(calibrated[i]),
                "band": str(bands[i]),
                "reasons": reasons_list[i],
                "model_version": bundle["version"],
                # Placeholder; resolved from lead_clusters in the same
                # checkout as the append (see _resolve_and_persist).
                "duplicate_cluster_size": 1,
            }
        )
    return results


def _db_entry(r: dict[str, Any]) -> dict[str, Any]:
    """Strip a scored result down to the lead_scores columns."""
    return {
        "lead_id": r["lead_id"],
        "asof": r["asof"],
        "model_version": r["model_version"],
        "probability": r["probability"],
        "band": r["band"],
        "reasons": r["reasons"],
    }


def _resolve_and_persist(results: list[dict[str, Any]]) -> None:
    """
    Resolve duplicate cluster sizes synchronously (plain SELECT, ~150ms)
    and hand the INSERT to the background writer (commit latency dominates
    on Neon). Best-effort: scoring never fails over persistence.
    """
    if not results:
        return
    try:
        sizes = lookup_sizes([r["lead_id"] for r in results])
        for r in results:
            r["duplicate_cluster_size"] = sizes.get(r["lead_id"], 1)
    except Exception:
        pass
    enqueue_scores([_db_entry(r) for r in results])


def score_single_lead(lead_id: str) -> dict[str, Any]:
    """Score a single enquiry and return the response dict."""
    _ensure_schema_once()
    bundle = _get_bundle()
    frames = _get_frames()

    lead_str = str(lead_id)
    if lead_str not in set(frames["leads"]["lead_id"].astype(str)):
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")

    mask = frames["leads"]["lead_id"].astype(str) == lead_str
    leads_filtered = frames["leads"][mask]
    messages_filtered = frames["messages"][
        frames["messages"]["lead_id"].astype(str) == lead_str
    ]
    calls_filtered = frames["calls"][
        frames["calls"]["lead_id"].astype(str) == lead_str
    ]

    try:
        X = build_feature_matrix(
            leads_df=leads_filtered,
            messages_df=messages_filtered,
            calls_df=calls_filtered,
            localities_df=frames["localities"],
            m4_features=_get_m4_features(frames["leads"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if X.empty:
        raise HTTPException(status_code=404, detail=f"No features for lead {lead_id}")

    t0 = time.perf_counter()
    result = _score_frame(X, leads_filtered, bundle)[0]
    t_score = time.perf_counter() - t0
    t0 = time.perf_counter()
    _resolve_and_persist([result])
    t_db = time.perf_counter() - t0
    _api_log.info(
        "single %s score=%.2fs db=%.2fs", lead_str, t_score, t_db
    )
    return result


def score_many_leads(lead_ids: list[str]) -> list[dict[str, Any]]:
    """
    Score many enquiries in one vectorized pass.

    Same outputs as calling :func:`score_single_lead` per id, but the
    feature matrix is built once, the model predicts once, SHAP runs once
    over the batch, cluster sizes resolve in one query, and all rows are
    appended in one transaction.
    """
    _ensure_schema_once()
    bundle = _get_bundle()
    frames = _get_frames()

    requested = [str(lid) for lid in lead_ids]
    known = set(frames["leads"]["lead_id"].astype(str))
    for lid in requested:
        if lid not in known:
            raise HTTPException(status_code=404, detail=f"Lead {lid} not found")

    # Score each unique lead once, then map back to request order.
    unique_ids = list(dict.fromkeys(requested))
    leads_mask = frames["leads"]["lead_id"].astype(str).isin(unique_ids)
    leads_filtered = frames["leads"][leads_mask]
    messages_filtered = frames["messages"][
        frames["messages"]["lead_id"].astype(str).isin(unique_ids)
    ]
    calls_filtered = frames["calls"][
        frames["calls"]["lead_id"].astype(str).isin(unique_ids)
    ]

    t0 = time.perf_counter()
    try:
        X = build_feature_matrix(
            leads_df=leads_filtered,
            messages_df=messages_filtered,
            calls_df=calls_filtered,
            localities_df=frames["localities"],
            m4_features=_get_m4_features(frames["leads"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    t_features = time.perf_counter() - t0

    if X.empty:
        raise HTTPException(status_code=404, detail="No features for requested leads")

    t0 = time.perf_counter()
    scored = _score_frame(X, leads_filtered, bundle)
    t_score = time.perf_counter() - t0
    by_id = {r["lead_id"]: r for r in scored}
    results = [by_id[lid] for lid in requested]
    t0 = time.perf_counter()
    _resolve_and_persist(results)
    t_db = time.perf_counter() - t0
    _api_log.info(
        "batch n=%d features=%.2fs score=%.2fs db=%.2fs",
        len(unique_ids),
        t_features,
        t_score,
        t_db,
    )
    return results


def get_model_card_api() -> dict[str, Any]:
    """Get the model card from the database."""
    _ensure_schema_once()
    card = get_model_card()
    if card is None:
        raise HTTPException(status_code=404, detail="No model registered")
    return card


def healthz_api() -> dict[str, Any]:
    """Health check including model loaded status."""
    try:
        _ensure_schema_once()
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
