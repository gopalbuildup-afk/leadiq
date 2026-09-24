from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .contract import TIMEZONE


# ------------------------------------------------------------------------------
# Leakage / modeling policy
# ------------------------------------------------------------------------------

FORBIDDEN_COLUMNS = [
    "current_stage",
    "legacy_score",
    "legacy_score_version",
    "updated_at",
    "last_contacted_at",
]

# assigned_counselor_id is not explicitly listed as a forbidden column in
# the brief, but we exclude it from the model because the source contains
# current ownership rather than a point-in-time assignment history.
NON_MODEL_METADATA_COLUMNS = [
    *FORBIDDEN_COLUMNS,
    "assigned_counselor_id",
]

PII_AND_METADATA_COLS = [
    "full_name",
    "phone",
    "email",
    "home_locality",
    "created_at",
    "t",
    "current_stage",
    "legacy_score",
    "legacy_score_version",
    "updated_at",
    "last_contacted_at",
    "assigned_counselor_id",
]


# ------------------------------------------------------------------------------
# Keyword patterns
# ------------------------------------------------------------------------------

KEYWORD_PATTERNS = {
    "kw_fee": r"\b(fee|fees|cost|price|charge|charges)\b",
    "kw_emi": r"\b(emi|installment|installments|kist)\b",
    "kw_placement": (
        r"\b(placement|placements|job|jobs|salary|package|hiring)\b"
    ),
    "kw_timing": (
        r"\b(time|timing|timings|batch|batches|schedule|"
        r"weekend|weekday)\b"
    ),
    "kw_visit": (
        r"\b(visit|location|address|center|centre|branch|map)\b"
    ),
    "kw_demo": r"\b(demo|trial|class|session)\b",
    "kw_price_objection": (
        r"\b(expensive|zyada|high|costly|budget|kam)\b"
    ),
    "kw_not_interested": (
        r"\b(not interested|no thanks|nahi|cancel|don't want)\b"
    ),
}


# ------------------------------------------------------------------------------
# Safety assertions
# ------------------------------------------------------------------------------

def _assert_output_has_no_forbidden(
    df: pd.DataFrame,
) -> None:
    leaked = [
        column
        for column in FORBIDDEN_COLUMNS
        if column in df.columns
    ]

    if leaked:
        raise ValueError(
            "Forbidden column leakage detected in output "
            f"feature matrix: {leaked}"
        )


def _assert_output_has_no_pii(
    df: pd.DataFrame,
) -> None:
    leaked = [
        column
        for column in [
            "full_name",
            "phone",
            "email",
        ]
        if column in df.columns
    ]

    if leaked:
        raise ValueError(
            "PII columns reached the feature matrix: "
            f"{leaked}"
        )


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _normalise_text_column(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
    )


def _safe_empty_communication_features(
    leads_df: pd.DataFrame,
) -> pd.DataFrame:
    result = leads_df[["lead_id"]].copy()

    result["mins_to_first_lead_reply_after_assistant"] = np.nan
    result["inbound_msg_count_before_t"] = 0
    result["outbound_msg_count_before_t"] = 0
    result["mean_msg_len_before_t"] = 0.0

    for keyword in KEYWORD_PATTERNS:
        result[keyword] = 0

    result["has_inbound_text_before_t"] = 0

    return result


# ------------------------------------------------------------------------------
# Locality features
# ------------------------------------------------------------------------------

def _compute_locality_features(
    leads_df: pd.DataFrame,
    localities_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute nearest-branch distance from the static locality table.

    Missing locality joins are handled using the median locality distance,
    plus an explicit missing-match indicator.
    """
    required_columns = {
        "locality",
        "km_to_riverside",
        "km_to_central",
        "km_to_lakeview",
    }

    missing = required_columns.difference(
        localities_df.columns
    )

    if missing:
        raise ValueError(
            "Missing locality columns: "
            f"{sorted(missing)}"
        )

    loc_df = localities_df.copy()

    loc_df["min_dist_to_branch"] = loc_df[
        [
            "km_to_riverside",
            "km_to_central",
            "km_to_lakeview",
        ]
    ].min(axis=1)

    merged = pd.merge(
        leads_df[
            [
                "lead_id",
                "home_locality",
            ]
        ],
        loc_df[
            [
                "locality",
                "min_dist_to_branch",
            ]
        ],
        left_on="home_locality",
        right_on="locality",
        how="left",
    )

    merged["locality_match_missing"] = (
        merged["min_dist_to_branch"]
        .isna()
        .astype(int)
    )

    median_distance = loc_df[
        "min_dist_to_branch"
    ].median()

    if pd.isna(median_distance):
        raise ValueError(
            "Cannot impute locality distance because "
            "localities.csv contains no valid distances."
        )

    merged["min_dist_to_branch"] = (
        merged["min_dist_to_branch"]
        .fillna(median_distance)
    )

    return merged[
        [
            "lead_id",
            "min_dist_to_branch",
            "locality_match_missing",
        ]
    ]


# ------------------------------------------------------------------------------
# Communication features
# ------------------------------------------------------------------------------

def _compute_communication_features(
    leads_df: pd.DataFrame,
    messages_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute message-derived features strictly before t.

    Source columns are the exact CSV columns:
        sent_at
        direction
        sender_type
        text

    Content-based features are based on the person's own inbound messages,
    not assistant/counselor text.
    """
    required_columns = {
        "message_id",
        "lead_id",
        "sent_at",
        "direction",
        "sender_type",
        "text",
    }

    missing = required_columns.difference(
        messages_df.columns
    )

    if missing:
        raise ValueError(
            "Missing message columns: "
            f"{sorted(missing)}"
        )

    if messages_df.empty:
        return _safe_empty_communication_features(
            leads_df
        )

    msg_df = messages_df.copy()

    msg_df["sent_at"] = pd.to_datetime(
        msg_df["sent_at"],
        utc=True,
    )

    leads_copy = leads_df[
        [
            "lead_id",
            "created_at",
            "t",
        ]
    ].copy()

    leads_copy["created_at"] = pd.to_datetime(
        leads_copy["created_at"],
        utc=True,
    )

    leads_copy["t"] = pd.to_datetime(
        leads_copy["t"],
        utc=True,
    )

    merged = pd.merge(
        msg_df,
        leads_copy,
        on="lead_id",
        how="inner",
    )

    # Point-in-time rule:
    #   created_at <= sent_at < t
    valid_msgs = merged[
        (merged["sent_at"] >= merged["created_at"])
        & (merged["sent_at"] < merged["t"])
    ].copy()

    if valid_msgs.empty:
        return _safe_empty_communication_features(
            leads_df
        )

    valid_msgs["text"] = _normalise_text_column(
        valid_msgs["text"]
    )

    valid_msgs["msg_len"] = (
        valid_msgs["text"]
        .str.len()
    )

    # ------------------------------------------------------------------
    # Message counts
    # ------------------------------------------------------------------

    msg_counts = (
        valid_msgs
        .groupby(
            ["lead_id", "direction"],
            dropna=False,
        )
        .size()
        .unstack(
            fill_value=0,
        )
        .reset_index()
    )

    if "inbound" not in msg_counts.columns:
        msg_counts["inbound"] = 0

    if "outbound" not in msg_counts.columns:
        msg_counts["outbound"] = 0

    msg_counts = msg_counts.rename(
        columns={
            "inbound": "inbound_msg_count_before_t",
            "outbound": "outbound_msg_count_before_t",
        }
    )

    # ------------------------------------------------------------------
    # Person-owned inbound messages
    # ------------------------------------------------------------------

    lead_inbound = valid_msgs[
        (valid_msgs["direction"] == "inbound")
        & (valid_msgs["sender_type"] == "lead")
    ].copy()

    if lead_inbound.empty:
        mean_len = pd.DataFrame(
            columns=[
                "lead_id",
                "mean_msg_len_before_t",
            ]
        )
    else:
        mean_len = (
            lead_inbound
            .groupby("lead_id")["msg_len"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "msg_len": "mean_msg_len_before_t"
                }
            )
        )

    # ------------------------------------------------------------------
    # First lead reply after first assistant message
    # ------------------------------------------------------------------

    assistant_outbound = valid_msgs[
        (valid_msgs["direction"] == "outbound")
        & (valid_msgs["sender_type"] == "assistant")
    ].copy()

    if (
        not assistant_outbound.empty
        and not lead_inbound.empty
    ):
        first_assistant = (
            assistant_outbound
            .groupby("lead_id")["sent_at"]
            .min()
            .reset_index()
            .rename(
                columns={
                    "sent_at": "first_assistant_at"
                }
            )
        )

        response_candidates = pd.merge(
            lead_inbound[
                [
                    "lead_id",
                    "sent_at",
                ]
            ],
            first_assistant,
            on="lead_id",
            how="inner",
        )

        response_candidates = response_candidates[
            response_candidates["sent_at"]
            > response_candidates["first_assistant_at"]
        ].copy()

        if not response_candidates.empty:
            response_candidates[
                "reply_delay_mins"
            ] = (
                response_candidates["sent_at"]
                - response_candidates["first_assistant_at"]
            ).dt.total_seconds() / 60.0

            first_reply = (
                response_candidates
                .groupby("lead_id")[
                    "reply_delay_mins"
                ]
                .min()
                .reset_index()
                .rename(
                    columns={
                        "reply_delay_mins":
                            "mins_to_first_lead_reply_after_assistant"
                    }
                )
            )
        else:
            first_reply = pd.DataFrame(
                columns=[
                    "lead_id",
                    "mins_to_first_lead_reply_after_assistant",
                ]
            )
    else:
        first_reply = pd.DataFrame(
            columns=[
                "lead_id",
                "mins_to_first_lead_reply_after_assistant",
            ]
        )

    # ------------------------------------------------------------------
    # Keyword flags
    # ------------------------------------------------------------------

    if lead_inbound.empty:
        keyword_df = pd.DataFrame(
            columns=[
                "lead_id",
                *KEYWORD_PATTERNS.keys(),
                "has_inbound_text_before_t",
            ]
        )
    else:
        combined_text = (
            lead_inbound
            .groupby("lead_id")["text"]
            .apply(
                lambda values: " ".join(values)
                .lower()
            )
            .reset_index()
        )

        records = []

        for _, row in combined_text.iterrows():
            text = row["text"]

            record = {
                "lead_id": row["lead_id"],
                "has_inbound_text_before_t": int(
                    bool(text.strip())
                ),
            }

            for keyword_name, pattern in KEYWORD_PATTERNS.items():
                record[keyword_name] = int(
                    re.search(
                        pattern,
                        text,
                    )
                    is not None
                )

            records.append(record)

        keyword_df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Merge feature blocks
    # ------------------------------------------------------------------

    comm_df = leads_df[
        ["lead_id"]
    ].copy()

    comm_df = pd.merge(
        comm_df,
        msg_counts[
            [
                "lead_id",
                "inbound_msg_count_before_t",
                "outbound_msg_count_before_t",
            ]
        ],
        on="lead_id",
        how="left",
    )

    comm_df = pd.merge(
        comm_df,
        mean_len,
        on="lead_id",
        how="left",
    )

    comm_df = pd.merge(
        comm_df,
        first_reply,
        on="lead_id",
        how="left",
    )

    comm_df = pd.merge(
        comm_df,
        keyword_df,
        on="lead_id",
        how="left",
    )

    # Structural defaults.
    comm_df[
        "inbound_msg_count_before_t"
    ] = (
        comm_df[
            "inbound_msg_count_before_t"
        ]
        .fillna(0)
        .astype(int)
    )

    comm_df[
        "outbound_msg_count_before_t"
    ] = (
        comm_df[
            "outbound_msg_count_before_t"
        ]
        .fillna(0)
        .astype(int)
    )

    comm_df[
        "mean_msg_len_before_t"
    ] = (
        comm_df[
            "mean_msg_len_before_t"
        ]
        .fillna(0.0)
    )

    comm_df[
        "has_inbound_text_before_t"
    ] = (
        comm_df[
            "has_inbound_text_before_t"
        ]
        .fillna(0)
        .astype(int)
    )

    for keyword in KEYWORD_PATTERNS:
        comm_df[keyword] = (
            comm_df[keyword]
            .fillna(0)
            .astype(int)
        )

    return comm_df


# ------------------------------------------------------------------------------
# Call features
# ------------------------------------------------------------------------------

def _compute_call_features(
    leads_df: pd.DataFrame,
    calls_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute call features from raw calls.csv fields.

    Only calls with:
        created_at <= started_at < t
    are allowed.
    """
    required_columns = {
        "call_id",
        "lead_id",
        "counselor_id",
        "started_at_ms",
        "duration_sec",
        "outcome",
    }

    missing = required_columns.difference(
        calls_df.columns
    )

    if missing:
        raise ValueError(
            "Missing call columns: "
            f"{sorted(missing)}"
        )

    if calls_df.empty:
        return pd.DataFrame(
            {
                "lead_id": leads_df["lead_id"],
                "mins_to_first_counselor_call": np.nan,
                "calls_attempted_before_t": 0,
                "calls_answered_before_t": 0,
            }
        )

    clean_calls = calls_df.copy()

    clean_calls["duration_sec"] = pd.to_numeric(
        clean_calls["duration_sec"],
        errors="raise",
    )

    # Defensive handling of any malformed negative duration.
    clean_calls = clean_calls[
        clean_calls["duration_sec"] >= 0
    ].copy()

    if "started_at" in clean_calls.columns:
        clean_calls["started_at"] = pd.to_datetime(
            clean_calls["started_at"],
            utc=True,
        )
    else:
        clean_calls["started_at"] = pd.to_datetime(
            clean_calls["started_at_ms"],
            unit="ms",
            utc=True,
        )

    leads_copy = leads_df[
        [
            "lead_id",
            "created_at",
            "t",
        ]
    ].copy()

    leads_copy["created_at"] = pd.to_datetime(
        leads_copy["created_at"],
        utc=True,
    )

    leads_copy["t"] = pd.to_datetime(
        leads_copy["t"],
        utc=True,
    )

    merged_calls = pd.merge(
        clean_calls,
        leads_copy,
        on="lead_id",
        how="inner",
    )

    valid_calls = merged_calls[
        (merged_calls["started_at"] >= merged_calls["created_at"])
        & (merged_calls["started_at"] < merged_calls["t"])
    ].copy()

    if valid_calls.empty:
        return pd.DataFrame(
            {
                "lead_id": leads_df["lead_id"],
                "mins_to_first_counselor_call": np.nan,
                "calls_attempted_before_t": 0,
                "calls_answered_before_t": 0,
            }
        )

    attempts = (
        valid_calls
        .groupby("lead_id")
        .size()
        .reset_index(
            name="calls_attempted_before_t"
        )
    )

    answered = (
        valid_calls[
            valid_calls["outcome"] == "answered"
        ]
        .groupby("lead_id")
        .size()
        .reset_index(
            name="calls_answered_before_t"
        )
    )

    valid_calls["call_delay_mins"] = (
        valid_calls["started_at"]
        - valid_calls["created_at"]
    ).dt.total_seconds() / 60.0

    first_call = (
        valid_calls
        .groupby("lead_id")[
            "call_delay_mins"
        ]
        .min()
        .reset_index()
        .rename(
            columns={
                "call_delay_mins":
                    "mins_to_first_counselor_call"
            }
        )
    )

    call_features = leads_df[
        ["lead_id"]
    ].copy()

    call_features = pd.merge(
        call_features,
        attempts,
        on="lead_id",
        how="left",
    )

    call_features = pd.merge(
        call_features,
        answered,
        on="lead_id",
        how="left",
    )

    call_features = pd.merge(
        call_features,
        first_call,
        on="lead_id",
        how="left",
    )

    call_features[
        "calls_attempted_before_t"
    ] = (
        call_features[
            "calls_attempted_before_t"
        ]
        .fillna(0)
        .astype(int)
    )

    call_features[
        "calls_answered_before_t"
    ] = (
        call_features[
            "calls_answered_before_t"
        ]
        .fillna(0)
        .astype(int)
    )

    return call_features


# ------------------------------------------------------------------------------
# Main feature matrix
# ------------------------------------------------------------------------------

def build_feature_matrix(
    leads_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    calls_df: pd.DataFrame,
    localities_df: pd.DataFrame,
    fill_latency_nans: bool = True,
    m4_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the point-in-time feature matrix.

    Scoring moment:
        t = created_at + 24h

    Every event feature satisfies:
        created_at <= event_time < t

    The raw leads dataframe is allowed to contain source/export columns such as
    current_stage and legacy_score. They are explicitly excluded from the
    model matrix.
    """
    required_lead_columns = {
        "lead_id",
        "created_at",
        "source",
        "branch",
        "course_interest",
        "full_name",
        "phone",
        "email",
        "education_status",
        "age",
        "home_locality",
    }

    missing = required_lead_columns.difference(
        leads_df.columns
    )

    if missing:
        raise ValueError(
            "Missing required lead feature columns: "
            f"{sorted(missing)}"
        )

    clean_leads = leads_df.copy()
    created_dt = pd.to_datetime(clean_leads["created_at"])

    if created_dt.dt.tz is None:
        created_dt = created_dt.dt.tz_localize(TIMEZONE)

    clean_leads["created_at"] = created_dt.dt.tz_convert("UTC")

    clean_leads["t"] = clean_leads["created_at"] + pd.Timedelta(hours=24)


    # ------------------------------------------------------    ------------
    # Creation-time features
    # ------------------------------------------------------------------
    base = pd.DataFrame(index=clean_leads["lead_id"])
    base = clean_leads[
        [
            "lead_id",
            "source",
            "branch",
            "course_interest",
            "education_status",
            "age",
        ]
    ].copy()

    base["creation_hour"] = (
        clean_leads["created_at"].dt.hour
    )

    base["creation_weekday"] = (
        clean_leads["created_at"].dt.dayofweek
    )

    base["age_missing"] = (
        clean_leads["age"]
        .isna()
        .astype(int)
    )

    base["has_phone"] = (
        clean_leads["phone"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    base["has_email"] = (
        clean_leads["email"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    base["has_home_locality"] = (
        clean_leads["home_locality"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    base["education_status_missing"] = (
        clean_leads["education_status"]
        .isna()
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Event features
    # ------------------------------------------------------------------

    comm_features = _compute_communication_features(
        clean_leads,
        messages_df,
    )

    call_features = _compute_call_features(
        clean_leads,
        calls_df,
    )

    locality_features = _compute_locality_features(
        clean_leads,
        localities_df,
    )

    if m4_features is not None:
        required_m4 = {"earlier_enquiries_count"}
        missing_m4 = required_m4 - set(m4_features.columns)
        if missing_m4:
            raise ValueError(
                "M4 feature block is missing required columns: "
                f"{sorted(missing_m4)}"
            )

        m4_block = m4_features.copy()
        if m4_block.index.name != "lead_id":
            if "lead_id" in m4_block.columns:
                m4_block = m4_block.set_index("lead_id")
            else:
                raise ValueError(
                    "M4 feature block must use lead_id as its index "
                    "or contain a lead_id column."
                )

        m4_block.index = m4_block.index.astype(str)

        if m4_block.index.duplicated().any():
            raise ValueError("M4 feature block contains duplicate lead_id values.")

        if not clean_leads["lead_id"].astype(str).isin(m4_block.index).all():
            missing_ids = clean_leads.loc[
                ~clean_leads["lead_id"].astype(str).isin(m4_block.index),
                "lead_id",
            ].head(10).tolist()
            raise ValueError(
                "M4 feature block is missing leads. "
                f"Examples: {missing_ids}"
            )

        m4_block = m4_block.reindex(clean_leads["lead_id"].astype(str))
        m4_block.index = clean_leads["lead_id"].values

    # ------------------------------------------------------------------
    # Merge feature blocks
    # ------------------------------------------------------------------

    df = pd.merge(
        base,
        comm_features,
        on="lead_id",
        how="left",
        validate="one_to_one",
    )

    df = pd.merge(
        df,
        call_features,
        on="lead_id",
        how="left",
        validate="one_to_one",
    )

    df = pd.merge(
        df,
        locality_features,
        on="lead_id",
        how="left",
        validate="one_to_one",
    )

    if m4_features is not None:
        m4_merge = m4_block.reset_index().rename(
            columns={"index": "lead_id"}
        )
        df = pd.merge(
            df,
            m4_merge,
            on="lead_id",
            how="left",
            validate="one_to_one",
        )
    else:
        df["earlier_enquiries_count"] = 0

    # ------------------------------------------------------------------
    # Remove metadata / PII
    # ------------------------------------------------------------------

    df = df.drop(
        columns=[
            column
            for column in PII_AND_METADATA_COLS
            if column in df.columns
        ],
        errors="ignore",
    )

    # lead_id remains available as the index, not as a model feature.
    df = df.set_index("lead_id")

    # ------------------------------------------------------------------
    # Structural missing-value handling
    # ------------------------------------------------------------------

    if fill_latency_nans:
        latency_columns = [
            "mins_to_first_lead_reply_after_assistant",
            "mins_to_first_counselor_call",
        ]

        for column in latency_columns:
            if column in df.columns:
                # -1 means the event did not occur within the 24h window.
                df[column] = (
                    df[column]
                    .fillna(-1.0)
                )

    # ------------------------------------------------------------------
    # Final safety checks
    # ------------------------------------------------------------------

    _assert_output_has_no_forbidden(df)
    _assert_output_has_no_pii(df)

    forbidden_present = set(
        FORBIDDEN_COLUMNS
    ).intersection(df.columns)

    if forbidden_present:
        raise ValueError(
            "Forbidden columns reached final feature matrix: "
            f"{sorted(forbidden_present)}"
        )

    return df