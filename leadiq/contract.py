from __future__ import annotations

from typing import Union

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

TIMEZONE = "Asia/Kolkata"

TARGET_STAGE = "Admission Done"
TARGET_WINDOW_DAYS = 30

FORBIDDEN_COLUMNS = [
    "current_stage",
    "legacy_score",
    "legacy_score_version",
    "updated_at",
    "last_contacted_at",
]

IST_DTYPE = pd.DatetimeTZDtype(
    unit="ns",
    tz="Asia/Kolkata",
)
# ------------------------------------------------------------------------------
# Pandera schemas
# ------------------------------------------------------------------------------

LeadsSchema = pa.DataFrameSchema(
    columns={
        "lead_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "created_at": pa.Column(
            IST_DTYPE,
            nullable=False,
        ),
        "source": pa.Column(
            pa.String,
            nullable=False,
        ),
        "branch": pa.Column(
            pa.String,
            nullable=False,
        ),
        "course_interest": pa.Column(
            pa.String,
            nullable=False,
        ),
        "full_name": pa.Column(
            pa.String,
            nullable=True,
        ),
        "phone": pa.Column(
            pa.String,
            nullable=True,
        ),
        "email": pa.Column(
            pa.String,
            nullable=True,
        ),
        "education_status": pa.Column(
            pa.String,
            nullable=True,
        ),
        "age": pa.Column(
            pa.Int64,
            nullable=True,
        ),
        "home_locality": pa.Column(
            pa.String,
            nullable=True,
        ),
        "assigned_counselor_id": pa.Column(
            pa.String,
            nullable=True,
        ),
        "current_stage": pa.Column(
            pa.String,
            nullable=False,
        ),
        "updated_at": pa.Column(
            IST_DTYPE,
            nullable=False,
        ),
        "last_contacted_at": pa.Column(
            IST_DTYPE,
            nullable=True,
        ),
        "legacy_score": pa.Column(
            pa.Int64,
            nullable=True,
        ),
        "legacy_score_version": pa.Column(
            pa.String,
            nullable=True,
        ),
    },
    strict=True,
    coerce=False,
)


MessagesSchema = pa.DataFrameSchema(
    columns={
        "message_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "lead_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "sent_at": pa.Column(
            IST_DTYPE,
            nullable=False,
        ),
        "direction": pa.Column(
            pa.String,
            checks=pa.Check.isin(["inbound", "outbound"]),
            nullable=False,
        ),
        "sender_type": pa.Column(
            pa.String,
            checks=pa.Check.isin(["lead", "assistant", "counselor"]),
            nullable=False,
        ),
        "text": pa.Column(
            pa.String,
            nullable=True,
        ),
    },
    strict=True,
    coerce=False,
)


CallsSchema = pa.DataFrameSchema(
    columns={
        "call_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "lead_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "counselor_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "started_at_ms": pa.Column(
            pa.Int64,
            nullable=False,
        ),
        "duration_sec": pa.Column(
            pa.Int64,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "outcome": pa.Column(
            pa.String,
            checks=pa.Check.isin(
                [
                    "answered",
                    "no_answer",
                    "busy",
                    "switched_off",
                ]
            ),
            nullable=False,
        ),
        "notes": pa.Column(
            pa.String,
            nullable=True,
        ),
    },
    strict=True,
    coerce=False,
)


StageHistorySchema = pa.DataFrameSchema(
    columns={
        "lead_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "stage": pa.Column(
            pa.String,
            checks=pa.Check.isin(
                [
                    "New",
                    "Contacted",
                    "Interested",
                    "Visit Scheduled",
                    "Center Visited",
                    "Demo Scheduled",
                    "Seat Reserved",
                    "Admission Done",
                    "Lost",
                    "Dormant",
                ]
            ),
            nullable=False,
        ),
        "changed_at": pa.Column(
            IST_DTYPE,
            nullable=False,
        ),
        "changed_by": pa.Column(
            pa.String,
            nullable=False,
        ),
    },
    strict=True,
    coerce=False,
)


CounselorsSchema = pa.DataFrameSchema(
    columns={
        "counselor_id": pa.Column(
            pa.String,
            nullable=False,
        ),
        "name": pa.Column(
            pa.String,
            nullable=False,
        ),
        "branch": pa.Column(
            pa.String,
            nullable=False,
        ),
        "languages": pa.Column(
            pa.String,
            nullable=False,
        ),
        "joined_on": pa.Column(
            "datetime64[ns]",
            nullable=False,
        ),
    },
    strict=True,
    coerce=False,
)


LocalitiesSchema = pa.DataFrameSchema(
    columns={
        "locality": pa.Column(
            pa.String,
            nullable=False,
        ),
        "km_to_riverside": pa.Column(
            pa.Float64,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "km_to_central": pa.Column(
            pa.Float64,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "km_to_lakeview": pa.Column(
            pa.Float64,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
    },
    strict=True,
    coerce=False,
)

class LeadsInputSchema(pa.DataFrameModel):
    lead_id: Series[str] = pa.Field(nullable=False)
    created_at: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(nullable=False)
    branch: Series[str] = pa.Field(nullable=False)
    course_interest: Series[str] = pa.Field(nullable=False)
    full_name: Series[str] = pa.Field(nullable=True)
    phone: Series[str] = pa.Field(nullable=True)
    email: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False  # Allows extra columns like legacy_score, current_stage
        coerce = True
# ------------------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------------------
def _ensure_ist_ns(series: pd.Series) -> pd.Series:
    return series.astype(
        "datetime64[ns, Asia/Kolkata]"
    )

def _assert_unique(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    if df[column].duplicated().any():
        duplicates = df.loc[
            df[column].duplicated(keep=False),
            column,
        ].astype(str).unique()[:10]

        raise ValueError(
            f"{dataset_name}: column '{column}' must be unique. "
            f"Example duplicates: {duplicates.tolist()}"
        )


def _to_integer(
    series: pd.Series,
    column_name: str,
    allow_null: bool = False,
) -> pd.Series:
    """
    Convert numeric input explicitly.

    Invalid values raise instead of silently becoming NaN.
    """
    numeric = pd.to_numeric(
        series,
        errors="raise",
    )

    if allow_null:
        return numeric.astype("Int64")

    if numeric.isna().any():
        raise ValueError(
            f"Column '{column_name}' contains null values but is required."
        )

    return numeric.astype("int64")

def _normalize_ist_datetime_precision(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize an already timezone-aware datetime Series to
    Asia/Kolkata with nanosecond precision.

    This makes the dtype deterministic across pandas versions,
    including pandas versions that default to microsecond precision.
    """
    series = pd.to_datetime(
        series,
        errors="raise",
    )

    if series.dt.tz is None:
        raise ValueError(
            "Expected timezone-aware datetime values."
        )

    series = series.dt.tz_convert(TIMEZONE)

    # Pandas 3.x may produce datetime64[us, tz].
    # Pandera's current DateTime validation expects ns here.
    try:
        series = series.dt.as_unit("ns")
    except AttributeError:
        # Compatibility for older pandas versions.
        series = series.astype(
            f"datetime64[ns, {TIMEZONE}]"
        )

    return series
# ------------------------------------------------------------------------------
# Timestamp parsing
# ------------------------------------------------------------------------------

def parse_ist_naive(
    series: pd.Series,
    tz: str = TIMEZONE,
) -> pd.Series:
    """
    Source format:
        local IST timestamp with no offset.

    Example:
        2026-03-04 14:22:10

    The source value is interpreted as IST, not UTC.
    """
    parsed = pd.to_datetime(
        series,
        errors="raise",
    )

    if parsed.dt.tz is not None:
        raise ValueError(
            "Expected naive IST timestamps without timezone offset."
        )

    parsed = parsed.dt.tz_localize(tz)

    return _normalize_ist_datetime_precision(
        parsed
    )


def parse_utc(
    series: pd.Series,
    tz: str = TIMEZONE,
) -> pd.Series:
    """
    Source format:
        UTC timestamp, e.g. ISO-8601 with Z.

    Convert to Asia/Kolkata and normalize precision.
    """
    parsed = pd.to_datetime(
        series,
        utc=True,
        errors="raise",
    )

    parsed = parsed.dt.tz_convert(tz)

    return _normalize_ist_datetime_precision(
        parsed
    )


def parse_offset_timestamp(
    series: pd.Series,
    tz: str = TIMEZONE,
) -> pd.Series:
    """
    Parse ISO-8601 timestamps that already carry an offset,
    then normalize them to Asia/Kolkata.
    """
    try:
        parsed = pd.to_datetime(
            series,
            utc=True,
            errors="raise",
        )
    except Exception as exc:
        raise ValueError(
            "Failed to parse offset-aware timestamp column."
        ) from exc

    return _normalize_ist_datetime_precision(
        parsed
    )


def convert_epoch_ms_to_tz(
    series: pd.Series,
    tz: str = TIMEZONE,
) -> pd.Series:
    """
    Convert Unix epoch milliseconds in UTC to Asia/Kolkata
    and normalize precision.
    """
    parsed = pd.to_datetime(
        series,
        unit="ms",
        utc=True,
        errors="raise",
    )

    parsed = parsed.dt.tz_convert(tz)

    return _normalize_ist_datetime_precision(
        parsed
    )


# ------------------------------------------------------------------------------
# CSV loaders
# ------------------------------------------------------------------------------

def load_and_validate_leads(
    filepath: str,
) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["age"] = pd.to_numeric(
        df["age"],
        errors="raise"
    ).astype("Int64")

    df["legacy_score"] = _to_integer(
        df["legacy_score"],
        "legacy_score",
        allow_null=True,
    )

    for column in [
        "created_at",
        "updated_at",
        "last_contacted_at",
    ]:
        df[column] = parse_ist_naive(df[column])

    validated = LeadsSchema.validate(df)

    _assert_unique(
        validated,
        "lead_id",
        "leads.csv",
    )

    return validated


def load_and_validate_messages(
    filepath: str,
) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["sent_at"] = parse_utc(df["sent_at"])

    validated = MessagesSchema.validate(df)

    _assert_unique(
        validated,
        "message_id",
        "messages.csv",
    )

    return validated


def load_and_validate_calls(
    filepath: str,
) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["started_at_ms"] = _to_integer(
        df["started_at_ms"],
        "started_at_ms",
        allow_null=False,
    )

    df["duration_sec"] = _to_integer(
        df["duration_sec"],
        "duration_sec",
        allow_null=False,
    )

    # Validate the raw CSV contract first.
    validated = CallsSchema.validate(df)

    _assert_unique(
        validated,
        "call_id",
        "calls.csv",
    )

    # Derived convenience timestamp for downstream feature engineering.
    validated["started_at"] = convert_epoch_ms_to_tz(
        validated["started_at_ms"]
    )

    return validated


def load_and_validate_stage_history(
    filepath: str,
) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["changed_at"] = parse_offset_timestamp(df["changed_at"])

    return StageHistorySchema.validate(df)


def load_and_validate_counselors(
    filepath: str,
) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    parsed = pd.to_datetime(
        df["joined_on"],
        errors="raise",
    )

    # joined_on is documented as a DATE, not a timestamp.
    # It must remain timezone-naive.
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError(
            "counselors.csv: joined_on must be timezone-naive."
        )

    # Normalize pandas 3.x microsecond precision to ns
    # for deterministic Pandera validation.
    try:
        parsed = parsed.dt.as_unit("ns")
    except AttributeError:
        parsed = parsed.astype(
            "datetime64[ns]"
        )

    df["joined_on"] = parsed

    return CounselorsSchema.validate(df)


def load_and_validate_localities(
    filepath: str,
) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    for column in [
        "km_to_riverside",
        "km_to_central",
        "km_to_lakeview",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(float)

    return LocalitiesSchema.validate(df)


# ------------------------------------------------------------------------------
# Point-in-time rules
# ------------------------------------------------------------------------------

def compute_scoring_moment(
    created_at: pd.Series,
) -> pd.Series:
    """
    LeadIQ scoring moment:

        t = created_at + 24 hours
    """
    return created_at + pd.Timedelta(hours=24)


# ------------------------------------------------------------------------------
# Target / label construction
# ------------------------------------------------------------------------------

def compute_target_and_maturity(
    leads_df: pd.DataFrame,
    stage_history_df: pd.DataFrame,
    export_date: Union[str, pd.Timestamp],
    only_mature: bool = True,
) -> pd.DataFrame:
    """
    Canonical LeadIQ target definition.

    target = 1
        if and only if the enquiry reaches "Admission Done"
        between created_at and created_at + 30 days.

    target = 0
        otherwise.

    Mature records are those that have a complete 30-day outcome window
    by the export date.
    """
    required_lead_columns = {
        "lead_id",
        "created_at",
    }

    missing_lead = required_lead_columns.difference(leads_df.columns)

    if missing_lead:
        raise ValueError(
            f"Missing required lead columns: {sorted(missing_lead)}"
        )

    required_stage_columns = {
        "lead_id",
        "stage",
        "changed_at",
    }

    missing_stage = required_stage_columns.difference(
        stage_history_df.columns
    )

    if missing_stage:
        raise ValueError(
            "Missing required stage-history columns: "
            f"{sorted(missing_stage)}"
        )

    export_ts = pd.Timestamp(export_date)

    if export_ts.tzinfo is None:
        export_ts = export_ts.tz_localize(TIMEZONE)
    else:
        export_ts = export_ts.tz_convert(TIMEZONE)

    leads = leads_df.copy()

    leads["created_at"] = pd.to_datetime(
        leads["created_at"],
        utc=True,
    )

    leads["t"] = compute_scoring_moment(
        leads["created_at"]
    )

    # An enquiry is mature only when a complete 30-day label window exists.
    maturity_cutoff = export_ts - pd.Timedelta(
        days=TARGET_WINDOW_DAYS
    )

    leads["is_mature"] = (
        leads["created_at"] <= maturity_cutoff
    )

    history = stage_history_df.copy()

    history["changed_at"] = pd.to_datetime(
        history["changed_at"],
        utc=True,
    )

    admission_events = history[
        history["stage"].eq(TARGET_STAGE)
    ].copy()

    merged = pd.merge(
        leads[["lead_id", "created_at"]],
        admission_events[
            ["lead_id", "changed_at"]
        ],
        on="lead_id",
        how="inner",
    )

    window_end = (
        merged["created_at"]
        + pd.Timedelta(days=TARGET_WINDOW_DAYS)
    )

    valid_admissions = merged[
        (merged["changed_at"] >= merged["created_at"])
        & (merged["changed_at"] <= window_end)
    ]

    converted_lead_ids = set(
        valid_admissions["lead_id"]
        .dropna()
        .unique()
    )

    leads["target"] = (
        leads["lead_id"]
        .isin(converted_lead_ids)
        .astype("int64")
    )

    if only_mature:
        return (
            leads.loc[leads["is_mature"]]
            .drop(columns=["is_mature"])
            .reset_index(drop=True)
        )

    return leads


def build_target_vector(
    leads_df: pd.DataFrame,
    stage_history_df: pd.DataFrame,
) -> pd.Series:
    """
    Return the canonical binary target vector.

    This function intentionally has no alternative target definition:
    the LeadIQ task target is always Admission Done within 30 days.
    """
    labeled = compute_target_and_maturity(
        leads_df=leads_df,
        stage_history_df=stage_history_df,
        export_date=(
            pd.to_datetime(
                leads_df["created_at"],
                utc=True,
            ).max()
            + pd.Timedelta(days=TARGET_WINDOW_DAYS)
        ),
        only_mature=False,
    )

    target = labeled.set_index("lead_id")["target"].copy()
    target.name = "target"

    return target