from __future__ import annotations

import numpy as np
import pandas as pd

from leadiq.calibration import fit_probability_bands
from leadiq.metrics import (
    classification_metrics,
    expected_calibration_error,
    lift_at,
    recall_at,
)
from leadiq.split import build_time_split
from leadiq.calibration import fit_calibrator

def test_time_split_has_no_overlap():
    created_at = pd.Series(
        pd.date_range(
            "2026-01-01",
            periods=160,
            freq="D",
            tz="UTC",
        )
    )

    split = build_time_split(
        created_at,
        hidden_start="2026-04-20 00:00:00+05:30",
        validation_weeks=2,
        calibration_fraction=0.25,
    )

    assert set(split.train).isdisjoint(set(split.validation))
    assert set(split.train).isdisjoint(set(split.calibration))
    assert set(split.validation).isdisjoint(set(split.calibration))


def test_calibration_is_after_validation():
    created_at = pd.Series(
        pd.date_range(
            "2026-01-01",
            periods=160,
            freq="D",
            tz="UTC",
        )
    )

    split = build_time_split(
        created_at,
        hidden_start="2026-04-20 00:00:00+05:30",
        validation_weeks=2,
        calibration_fraction=0.25,
    )

    validation_dates = created_at.loc[split.validation]
    calibration_dates = created_at.loc[split.calibration]

    assert validation_dates.max() < calibration_dates.min()
    assert calibration_dates.max() < split.hidden_start


def test_required_metrics_are_present():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.10, 0.80, 0.20, 0.70, 0.90, 0.15, 0.60])

    metrics = classification_metrics(y, p, include_calibration=True)

    expected = {
        "pr_auc",
        "roc_auc",
        "lift_at_10pct",
        "lift_at_20pct",
        "recall_at_10pct",
        "recall_at_20pct",
        "brier_score",
        "ece_10bin",
    }
    assert expected.issubset(metrics)


def test_lift_and_recall_are_finite():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.10, 0.80, 0.20, 0.70, 0.90, 0.15, 0.60])

    assert np.isfinite(lift_at(y, p, 0.20))
    assert np.isfinite(recall_at(y, p, 0.20))


def test_ece_is_between_zero_and_one():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.10, 0.80, 0.20, 0.70, 0.90, 0.15, 0.60])

    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_p1_and_p2_probability_bands():
    p = np.linspace(0.01, 0.99, 100)
    thresholds = fit_probability_bands(p)

    assert thresholds["p1_min_probability"] > thresholds["p2_min_probability"]
def test_hidden_window_is_never_assigned_to_training_splits():
    created_at = pd.Series(
        pd.date_range(
            "2026-01-01",
            periods=220,
            freq="D",
            tz="UTC",
        )
    )

    hidden_start = pd.Timestamp(
        "2026-04-20 00:00:00+05:30"
    )

    split = build_time_split(
        created_at,
        hidden_start=hidden_start,
        validation_weeks=2,
        calibration_fraction=0.25,
    )

    assigned = (
        set(split.train)
        | set(split.validation)
        | set(split.calibration)
    )

    hidden_indices = set(
        created_at.index[created_at >= hidden_start.tz_convert("UTC")]
    )

    assert assigned.isdisjoint(hidden_indices)

def test_platt_calibration_returns_valid_probabilities():
    raw = np.array([0.05, 0.15, 0.25, 0.65, 0.75, 0.90])
    y = np.array([0, 0, 0, 1, 1, 1])

    calibrator = fit_calibrator("platt", raw, y)
    calibrated = calibrator.predict(raw)

    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert calibrated.shape == raw.shape


def test_isotonic_calibration_returns_valid_probabilities():
    raw = np.array([0.05, 0.15, 0.25, 0.65, 0.75, 0.90])
    y = np.array([0, 0, 0, 1, 1, 1])

    calibrator = fit_calibrator("isotonic", raw, y)
    calibrated = calibrator.predict(raw)

    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert calibrated.shape == raw.shape