from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling fitted on a held-out calibration slice."""

    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, raw_probability: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        p = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.model.fit(logits, np.asarray(y).astype(int))
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class IsotonicCalibrator:
    """Isotonic calibration fitted on a held-out calibration slice."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

    def fit(self, raw_probability: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(
            np.asarray(raw_probability, dtype=float),
            np.asarray(y).astype(int),
        )
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict(
                np.asarray(raw_probability, dtype=float)
            ),
            dtype=float,
        )


def fit_calibrator(
    method: str,
    raw_probability: np.ndarray,
    y: np.ndarray,
):
    method = method.lower()
    if method == "platt":
        return PlattCalibrator().fit(raw_probability, y)
    if method == "isotonic":
        return IsotonicCalibrator().fit(raw_probability, y)
    raise ValueError("method must be 'platt' or 'isotonic'")


def fit_probability_bands(
    calibrated_probability: np.ndarray,
    p1_fraction: float = 0.10,
    p2_fraction: float = 0.50,
) -> dict[str, float]:
    if not 0 < p1_fraction < p2_fraction < 1:
        raise ValueError("Need 0 < p1_fraction < p2_fraction < 1")

    p = np.asarray(calibrated_probability, dtype=float)
    return {
        "p1_min_probability": float(np.quantile(p, 1.0 - p1_fraction)),
        "p2_min_probability": float(np.quantile(p, 1.0 - p2_fraction)),
    }


def assign_band(
    calibrated_probability: float | np.ndarray,
    thresholds: dict[str, float],
):
    p1 = thresholds["p1_min_probability"]
    p2 = thresholds["p2_min_probability"]
    p = np.asarray(calibrated_probability)

    result = np.where(p >= p1, "P1", np.where(p >= p2, "P2", "P3"))
    if result.ndim == 0:
        return str(result.item())
    return result
