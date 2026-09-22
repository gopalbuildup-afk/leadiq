from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibration import assign_band
from .contract import (
    load_and_validate_calls,
    load_and_validate_leads,
    load_and_validate_localities,
    load_and_validate_messages,
    load_and_validate_stage_history,
)
from .features import build_feature_matrix


DEFAULT_MODEL_DIR = Path("artifacts/m2")
DEFAULT_LOCALITIES = Path("data/localities.csv")


def load_bundle(model_dir: str | Path) -> dict:
    """Load the trained M2 model bundle."""
    model_path = Path(model_dir) / "model_bundle.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {model_path}. "
            "Train M2 first or pass --model-dir."
        )

    bundle = joblib.load(model_path)

    required = {
        "version",
        "champion_name",
        "preprocessor",
        "champion_model",
        "calibrator",
        "band_thresholds",
    }
    missing = required.difference(bundle.keys())

    if missing:
        raise ValueError(
            f"Model bundle is missing required keys: {sorted(missing)}"
        )

    return bundle


def build_scoring_features(
    leads_path: str | Path,
    messages_path: str | Path,
    calls_path: str | Path,
    localities_path: str | Path,
    stages_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load and validate unseen scoring data and build the exact same
    point-in-time feature matrix used by M2.

    stage_history is intentionally not used as a model feature. It is
    loaded/validated when supplied because the batch CLI accepts it and
    the stage history must never affect scoring features.
    """
    leads = load_and_validate_leads(str(leads_path))
    messages = load_and_validate_messages(str(messages_path))
    calls = load_and_validate_calls(str(calls_path))
    localities = load_and_validate_localities(str(localities_path))

    if stages_path is not None:
        load_and_validate_stage_history(str(stages_path))

    X = build_feature_matrix(
        leads_df=leads,
        messages_df=messages,
        calls_df=calls,
        localities_df=localities,
    )

    if X.empty:
        raise ValueError("No leads were available for scoring.")

    if not X.index.is_unique:
        duplicated = X.index[X.index.duplicated()].unique().tolist()
        raise ValueError(
            "Duplicate lead_id values found in scoring data: "
            f"{duplicated[:10]}"
        )

    return X


def score_dataframe(
    X: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """Transform features, predict raw probabilities, calibrate, and assign bands."""
    preprocessor = bundle["preprocessor"]
    champion_model = bundle["champion_model"]
    calibrator = bundle["calibrator"]
    thresholds = bundle["band_thresholds"]

    # The preprocessor was fitted on the M2 raw feature columns.
    # transform() therefore preserves the exact training representation
    # and handle_unknown='ignore' safely handles unseen categories.
    X_transformed = preprocessor.transform(X)

    raw_probability = champion_model.predict_proba(X_transformed)[:, 1]

    calibrated_probability = np.asarray(
        calibrator.predict(raw_probability),
        dtype=float,
    )

    calibrated_probability = np.clip(
        calibrated_probability,
        0.0,
        1.0,
    )

    bands = assign_band(
        calibrated_probability,
        thresholds,
    )

    scores = pd.DataFrame(
        {
            "lead_id": X.index.astype(str),
            "probability": calibrated_probability,
            "band": bands,
            "model_version": bundle["version"],
        }
    )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeadIQ M2 batch scorer"
    )

    parser.add_argument("--leads", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--calls", required=True)
    parser.add_argument("--stages", required=False, default=None)

    parser.add_argument(
        "--localities",
        default=str(DEFAULT_LOCALITIES),
        help="Path to the static localities.csv file. "
        "Defaults to data/localities.csv.",
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing model_bundle.joblib.",
    )

    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    bundle = load_bundle(args.model_dir)

    X = build_scoring_features(
        leads_path=args.leads,
        messages_path=args.messages,
        calls_path=args.calls,
        localities_path=args.localities,
        stages_path=args.stages,
    )

    scores = score_dataframe(X, bundle)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)

    print(f"Model: {bundle['version']}")
    print(f"Champion: {bundle['champion_name']}")
    print(f"Scored leads: {len(scores)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
