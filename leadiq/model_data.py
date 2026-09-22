from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contract import (
    compute_target_and_maturity,
    load_and_validate_calls,
    load_and_validate_leads,
    load_and_validate_localities,
    load_and_validate_messages,
    load_and_validate_stage_history,
)
from .features import build_feature_matrix
from .split import TimeSplit, build_time_split


@dataclass
class M2Dataset:
    X: pd.DataFrame
    y: pd.Series
    created_at: pd.Series
    lead_ids: pd.Index
    split: TimeSplit


def load_m2_dataset(
    leads_path: str | Path,
    messages_path: str | Path,
    calls_path: str | Path,
    stages_path: str | Path,
    localities_path: str | Path,
    export_date: str | pd.Timestamp,
    hidden_start: str | pd.Timestamp = "2026-07-05 00:00:00+05:30",
    validation_weeks: int = 4,
    calibration_fraction: float = 0.20,
) -> M2Dataset:
    """Load M1 data, build mature labels + features, then make M2 split."""
    leads = load_and_validate_leads(str(leads_path))
    messages = load_and_validate_messages(str(messages_path))
    calls = load_and_validate_calls(str(calls_path))
    localities = load_and_validate_localities(str(localities_path))
    stages = load_and_validate_stage_history(
    str(stages_path)
)

    labels = compute_target_and_maturity(
        leads_df=leads,
        stage_history_df=stages,
        export_date=export_date,
        only_mature=True,
    )

    mature_ids = pd.Index(labels["lead_id"])
    mature_leads = leads[leads["lead_id"].isin(mature_ids)].copy()

    # Preserve raw rows for point-in-time feature construction.
    X = build_feature_matrix(
        mature_leads,
        messages[messages["lead_id"].isin(mature_ids)].copy(),
        calls[calls["lead_id"].isin(mature_ids)].copy(),
        localities,
    )

    y = labels.set_index("lead_id")["target"].reindex(X.index)
    created_at = (
        mature_leads.set_index("lead_id")["created_at"]
        .reindex(X.index)
    )

    if y.isna().any():
        raise ValueError("Some feature rows do not have a target label")
    if created_at.isna().any():
        raise ValueError("Some feature rows do not have created_at")

    split = build_time_split(
        created_at=created_at,
        hidden_start=hidden_start,
        validation_weeks=validation_weeks,
        calibration_fraction=calibration_fraction,
    )

    return M2Dataset(
        X=X,
        y=y.astype(int),
        created_at=created_at,
        lead_ids=X.index.copy(),
        split=split,
    )
