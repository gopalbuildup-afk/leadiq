from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.Index
    validation: pd.Index
    calibration: pd.Index
    validation_start: pd.Timestamp
    calibration_start: pd.Timestamp
    hidden_start: pd.Timestamp


def build_time_split(
    created_at: pd.Series,
    hidden_start: str | pd.Timestamp,
    validation_weeks: int = 4,
    calibration_fraction: float = 0.20,
) -> TimeSplit:
    """
    Build the M2 chronological split required by the task.

    - Train: everything before the validation window.
    - Validation: the weeks immediately before the hidden window, excluding
      the final calibration slice.
    - Calibration: the final part of that validation window.
    - Hidden window: never included.
    """
    if validation_weeks <= 0:
        raise ValueError("validation_weeks must be > 0")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")

    ts = pd.to_datetime(created_at, utc=True)
    hidden_start_ts = pd.Timestamp(hidden_start)

    if hidden_start_ts.tzinfo is None:
        hidden_start_ts = hidden_start_ts.tz_localize("Asia/Kolkata")
    else:
        hidden_start_ts = hidden_start_ts.tz_convert("Asia/Kolkata")

    hidden_start_ts = hidden_start_ts.tz_convert("UTC")

    validation_start = hidden_start_ts - pd.Timedelta(weeks=validation_weeks)
    validation_duration = hidden_start_ts - validation_start
    calibration_duration = validation_duration * calibration_fraction
    calibration_start = hidden_start_ts - calibration_duration

    train_mask = ts < validation_start
    validation_mask = (ts >= validation_start) & (ts < calibration_start)
    calibration_mask = (ts >= calibration_start) & (ts < hidden_start_ts)

    # No split may overlap another split.
    if (train_mask & validation_mask).any():
        raise AssertionError("Train and validation overlap")
    if (train_mask & calibration_mask).any():
        raise AssertionError("Train and calibration overlap")
    if (validation_mask & calibration_mask).any():
        raise AssertionError("Validation and calibration overlap")

    return TimeSplit(
        train=ts.index[train_mask],
        validation=ts.index[validation_mask],
        calibration=ts.index[calibration_mask],
        validation_start=validation_start,
        calibration_start=calibration_start,
        hidden_start=hidden_start_ts,
    )


def describe_split(
    created_at: pd.Series,
    split: TimeSplit,
) -> dict[str, str | int]:
    ts = pd.to_datetime(created_at, utc=True)

    return {
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "calibration_rows": int(len(split.calibration)),
        "train_min": ts.loc[split.train].min().isoformat(),
        "train_max": ts.loc[split.train].max().isoformat(),
        "validation_min": ts.loc[split.validation].min().isoformat(),
        "validation_max": ts.loc[split.validation].max().isoformat(),
        "calibration_min": ts.loc[split.calibration].min().isoformat(),
        "calibration_max": ts.loc[split.calibration].max().isoformat(),
        "validation_start": split.validation_start.isoformat(),
        "calibration_start": split.calibration_start.isoformat(),
        "hidden_start": split.hidden_start.isoformat(),
    }
