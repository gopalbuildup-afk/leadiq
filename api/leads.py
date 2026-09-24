from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .score import _ensure_schema_once, _get_frames, _project_root


router = APIRouter(prefix="/v1/leads", tags=["leads"])


class LeadMetaResponse(BaseModel):
    lead_id: str
    full_name: Optional[str] = None
    source: str
    branch: str
    course_interest: str
    age: Optional[float] = None
    home_locality: Optional[str] = None
    created_at: str
    assigned_counselor_id: Optional[str] = None


class LeadListResponse(BaseModel):
    leads: list[LeadMetaResponse]
    count: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


# Only creation-time fields the counselor UI displays. Deliberately
# excluded: phone/email (PII the UI never shows) and every export-time /
# leakage column (current_stage, legacy_score*, updated_at,
# last_contacted_at, education_status).
_META_COLUMNS = [
    "lead_id",
    "full_name",
    "source",
    "branch",
    "course_interest",
    "age",
    "home_locality",
    "created_at",
    "assigned_counselor_id",
]


def _row_to_meta(row: pd.Series) -> dict[str, Any]:
    def _str(v: Any) -> Optional[str]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        return s or None

    def _num(v: Any) -> Optional[float]:
        try:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    created = row["created_at"]
    try:
        createdIso = pd.Timestamp(created).isoformat()
    except Exception:
        createdIso = str(created)
    return {
        "lead_id": str(row["lead_id"]),
        "full_name": _str(row.get("full_name")),
        "source": _str(row.get("source")) or "Unknown",
        "branch": _str(row.get("branch")) or "—",
        "course_interest": _str(row.get("course_interest")) or "—",
        "age": _num(row.get("age")),
        "home_locality": _str(row.get("home_locality")),
        "created_at": createdIso,
        "assigned_counselor_id": _str(row.get("assigned_counselor_id")),
    }


_sorted_leads: pd.DataFrame | None = None


def _all_metas() -> pd.DataFrame:
    global _sorted_leads
    if _sorted_leads is not None:
        return _sorted_leads
    _ensure_schema_once()
    frames = _get_frames()
    leads = frames["leads"].copy()

    # Sort leads by chance percentage (probability) using precomputed scores
    try:
        root = _project_root()
        scores_path = os.path.join(root, "scores.csv")
        if os.path.exists(scores_path):
            scores_df = pd.read_csv(scores_path, usecols=["lead_id", "probability"])
            scores_df["lead_id"] = scores_df["lead_id"].astype(str)
            leads["_lid"] = leads["lead_id"].astype(str)
            merged = leads.merge(
                scores_df,
                left_on="_lid",
                right_on="lead_id",
                how="left",
                suffixes=("", "_score"),
            )
            merged["probability"] = merged["probability"].fillna(0.0)
            merged = merged.sort_values(
                by="probability", ascending=False
            ).reset_index(drop=True)
            _sorted_leads = merged
            return _sorted_leads
    except Exception:
        pass

    try:
        from leadiq.versioning import _pool_instance

        with _pool_instance().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT lead_id, probability FROM lead_scores ORDER BY probability DESC"
                )
                rows = cur.fetchall()
                if rows:
                    scores_df = pd.DataFrame(
                        rows, columns=["lead_id", "probability"]
                    ).drop_duplicates(subset=["lead_id"])
                    scores_df["lead_id"] = scores_df["lead_id"].astype(str)
                    leads["_lid"] = leads["lead_id"].astype(str)
                    merged = leads.merge(
                        scores_df,
                        left_on="_lid",
                        right_on="lead_id",
                        how="left",
                        suffixes=("", "_score"),
                    )
                    merged["probability"] = merged["probability"].fillna(0.0)
                    merged = merged.sort_values(
                        by="probability", ascending=False
                    ).reset_index(drop=True)
                    _sorted_leads = merged
                    return _sorted_leads
    except Exception:
        pass

    _sorted_leads = leads
    return _sorted_leads


@router.get("", response_model=LeadListResponse)
def list_leads(
    limit: Optional[int] = Query(default=None, ge=1, le=50000),
    offset: int = Query(default=0, ge=0),
) -> LeadListResponse:
    """Paginated enquiry metadata for the call list (no PII, no leakage)."""
    try:
        leads = _all_metas()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    total = len(leads)
    if offset >= total:
        window = leads.iloc[0:0]
    elif limit is not None:
        window = leads.iloc[offset : offset + limit]
    else:
        window = leads.iloc[offset:]
    return LeadListResponse(
        leads=[LeadMetaResponse(**_row_to_meta(window.iloc[i])) for i in range(len(window))],
        count=len(window),
        total=total,
    )


@router.get("/{lead_id}", response_model=LeadMetaResponse)
def get_lead(lead_id: str) -> LeadMetaResponse:
    """Single enquiry metadata for the detail page."""
    try:
        leads = _all_metas()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    mask = leads["lead_id"].astype(str) == str(lead_id)
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")
    return LeadMetaResponse(**_row_to_meta(leads[mask].iloc[0]))
