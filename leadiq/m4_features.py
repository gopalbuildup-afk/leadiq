from __future__ import annotations

import pandas as pd

from .contract import TIMEZONE


def build_earlier_enquiries_feature(
    leads_df: pd.DataFrame,
    lead_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Build the M4-derived earlier-enquiries feature for M1.

    For each enquiry, count other enquiries in the same M4 cluster whose
    created_at is strictly earlier than the current enquiry.

    This is inherently before the scoring moment t = created_at + 24h,
    so it cannot use a later enquiry as an "earlier enquiry".
    """
    required_leads = {"lead_id", "created_at"}
    missing = required_leads - set(leads_df.columns)
    if missing:
        raise ValueError(
            f"Missing required lead columns for M4 feature: {sorted(missing)}"
        )

    required_clusters = {"lead_id", "cluster_id"}
    missing = required_clusters - set(lead_clusters.columns)
    if missing:
        raise ValueError(
            f"lead_clusters.csv is missing required columns: {sorted(missing)}"
        )

    leads = leads_df[["lead_id", "created_at"]].copy()
    leads["lead_id"] = leads["lead_id"].astype(str)
    leads["created_at_parsed"] = pd.to_datetime(
        leads["created_at"],
        errors="raise",
        utc=True,
    )

    clusters = lead_clusters[["lead_id", "cluster_id"]].copy()
    clusters["lead_id"] = clusters["lead_id"].astype(str)

    if clusters["lead_id"].duplicated().any():
        duplicated = clusters.loc[
            clusters["lead_id"].duplicated(keep=False), "lead_id"
        ].unique()[:10]
        raise ValueError(
            "lead_clusters.csv must contain exactly one row per lead. "
            f"Duplicate lead_id examples: {duplicated.tolist()}"
        )

    merged = leads.merge(
        clusters,
        on="lead_id",
        how="left",
        validate="one_to_one",
    )

    if merged["cluster_id"].isna().any():
        missing_ids = merged.loc[
            merged["cluster_id"].isna(), "lead_id"
        ].head(10).tolist()
        raise ValueError(
            "Some leads have no M4 cluster assignment. "
            f"Examples: {missing_ids}"
        )

    # Sort chronologically within each M4 cluster.  Because "earlier"
    # means strictly earlier than the current enquiry, cumcount is exactly
    # the number of earlier enquiries in that cluster.
    merged = merged.sort_values(
        ["cluster_id", "created_at_parsed", "lead_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    merged["earlier_enquiries_count"] = (
        merged.groupby("cluster_id", sort=False)
        .cumcount()
    )

    result = merged[["lead_id", "earlier_enquiries_count"]].copy()
    result["earlier_enquiries_count"] = (
        result["earlier_enquiries_count"].astype(int)
    )

    # Restore the original lead order and guarantee one row per lead.
    result = (
        leads[["lead_id"]]
        .merge(result, on="lead_id", how="left", validate="one_to_one")
        .set_index("lead_id")
    )

    if result["earlier_enquiries_count"].isna().any():
        raise ValueError("Failed to construct earlier_enquiries_count.")

    return result
