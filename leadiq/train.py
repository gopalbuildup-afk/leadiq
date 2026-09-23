from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibration import assign_band, fit_calibrator, fit_probability_bands
from .metrics import classification_metrics, plot_reliability_diagram, reliability_table
from .model_data import load_m2_dataset
from .split import describe_split

RANDOM_SEED = 42
MAX_TRIALS = 20


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical = []
    numeric = []

    for column in X_train.columns:
        dtype = X_train[column].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.StringDtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)
        else:
            numeric.append(column)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def fit_logistic(
    X_train_transformed,
    y_train,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_transformed, y_train)
    return model


def lgbm_search_space() -> dict[str, list[Any]]:
    return {
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 4, 6, 8],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0],
    }


def fit_lightgbm_with_search(
    X_train_transformed,
    y_train,
    X_validation_transformed,
    y_validation,
    n_trials: int = 12,
) -> tuple[lgb.LGBMClassifier, list[dict[str, Any]]]:
    n_trials = min(n_trials, MAX_TRIALS)

    candidates = list(
        ParameterSampler(
            lgbm_search_space(),
            n_iter=n_trials,
            random_state=RANDOM_SEED,
        )
    )

    best_model = None
    best_score = None
    history: list[dict[str, Any]] = []

    for trial_number, params in enumerate(candidates, start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            subsample_freq=1,
            **params,
        )

        model.fit(
            X_train_transformed,
            y_train,
            eval_set=[(X_validation_transformed, y_validation)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=75,
                    verbose=False,
                )
            ],
        )

        probability = model.predict_proba(
            X_validation_transformed,
        )[:, 1]
        metrics = classification_metrics(
            y_validation,
            probability,
            include_calibration=False,
        )

        record = {
            "trial": trial_number,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            **metrics,
            **params,
        }
        history.append(record)

        score = (
            metrics["pr_auc"],
            metrics["lift_at_20pct"],
            metrics["roc_auc"],
        )

        if best_score is None or score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("LightGBM search produced no model")

    return best_model, history


def choose_champion(
    logistic_metrics: dict[str, float],
    lightgbm_metrics: dict[str, float],
) -> str:
    """Deterministic validation-first selection.

    Primary metric: PR-AUC.
    Tie-breaks: Lift@20%, ROC-AUC.
    Exact ties go to logistic regression because the simpler model wins a tie.
    """
    logistic_score = (
        logistic_metrics["pr_auc"],
        logistic_metrics["lift_at_20pct"],
        logistic_metrics["roc_auc"],
    )
    lightgbm_score = (
        lightgbm_metrics["pr_auc"],
        lightgbm_metrics["lift_at_20pct"],
        lightgbm_metrics["roc_auc"],
    )

    if lightgbm_score > logistic_score:
        return "lightgbm"
    return "logistic_regression"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_eval_report(
    output_dir: Path,
    command: str,
    split_info: dict[str, Any],
    logistic_metrics: dict[str, float],
    lightgbm_metrics: dict[str, float],
    champion_name: str,
    calibration_metrics: dict[str, float],
    band_thresholds: dict[str, float],
    n_trials: int,
    calibration_method: str,
) -> None:
    lines = [
        "# M2 Evaluation",
        "",
        f"Champion: **{champion_name}**",
        "",
        "## Reproduction command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Time split",
        "",
    ]

    for key, value in split_info.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(
        [
            "",
            "## Validation: Logistic Regression",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, value in logistic_metrics.items():
        lines.append(f"| {key} | {value:.6f} |" if pd.notna(value) else f"| {key} | NA |")

    lines.extend(
        [
            "",
            "## Validation: LightGBM",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, value in lightgbm_metrics.items():
        lines.append(f"| {key} | {value:.6f} |" if pd.notna(value) else f"| {key} | NA |")

    lines.extend(
        [
            "",
            "## Calibration slice",
            "",
            f"Calibration method: **{calibration_method}**",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, value in calibration_metrics.items():
        lines.append(f"| {key} | {value:.6f} |" if pd.notna(value) else f"| {key} | NA |")

    lines.extend(
        [
            "",
            "## P1 / P2 / P3 thresholds",
            "",
            f"- P1 minimum calibrated probability: `{band_thresholds['p1_min_probability']:.6f}`",
            f"- P2 minimum calibrated probability: `{band_thresholds['p2_min_probability']:.6f}`",
            "- P1 is defined from the top 10% of calibrated probabilities on the calibration slice.",
            "- P2 covers the next 40%; P3 covers the remaining 50%.",
            "",
            f"LightGBM hyperparameter trials: `{n_trials}` (maximum allowed: 20)",
            "",
            "## Reliability diagram",
            "",
            "See `reliability_diagram.png` and `reliability_bins.csv` in this directory.",
            "",
            "> The hidden period is intentionally not used for model selection, calibration, or tuning.",
        ]
    )

    (output_dir / "EVAL_M2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(
    output_dir: Path,
    champion_name: str,
    calibration_method: str,
    split_info: dict[str, Any],
    calibration_metrics: dict[str, float],
) -> None:
    content = f"""# M2 Model Card

## Model
- Champion: `{champion_name}`
- Calibration: `{calibration_method}`
- Random seed: `{RANDOM_SEED}`

## Training/evaluation window
- Train: {split_info['train_min']} → {split_info['train_max']}
- Validation: {split_info['validation_min']} → {split_info['validation_max']}
- Calibration: {split_info['calibration_min']} → {split_info['calibration_max']}
- Hidden window starts: {split_info['hidden_start']}

## Calibration metrics
- Brier score: {calibration_metrics['brier_score']:.6f}
- ECE (10 bins): {calibration_metrics['ece_10bin']:.6f}

## Known limitations
- M3 embedding features are not included until Module 3.
- M4 earlier-enquiry features are not included until Module 4.
- Historical counselor effort can influence outcomes; this is a documented modeling limitation.
"""
    (output_dir / "MODEL_CARD_M2.md").write_text(content, encoding="utf-8")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_m2_dataset(
        leads_path=args.leads,
        messages_path=args.messages,
        calls_path=args.calls,
        stages_path=args.stages,
        localities_path=args.localities,
        export_date=args.export_date,
        hidden_start=args.hidden_start,
        validation_weeks=args.validation_weeks,
        calibration_fraction=args.calibration_fraction,
        lead_clusters_path=args.lead_clusters,
    )

    X = dataset.X
    y = dataset.y

    train_idx = dataset.split.train
    validation_idx = dataset.split.validation
    calibration_idx = dataset.split.calibration

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_validation = X.loc[validation_idx]
    y_validation = y.loc[validation_idx]
    X_calibration = X.loc[calibration_idx]
    y_calibration = y.loc[calibration_idx]

    if y_train.nunique() < 2 or y_validation.nunique() < 2 or y_calibration.nunique() < 2:
        raise ValueError(
            "Each M2 split must contain both classes. "
            f"train={y_train.value_counts().to_dict()}, "
            f"validation={y_validation.value_counts().to_dict()}, "
            f"calibration={y_calibration.value_counts().to_dict()}"
        )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_validation_t = preprocessor.transform(X_validation)
    X_calibration_t = preprocessor.transform(X_calibration)

    logistic_model = fit_logistic(
        X_train_t,
        y_train,
    )
    logistic_validation_probability = logistic_model.predict_proba(
        X_validation_t,
    )[:, 1]
    logistic_validation_metrics = classification_metrics(
        y_validation,
        logistic_validation_probability,
        include_calibration=False,
    )

    lightgbm_model, trial_history = fit_lightgbm_with_search(
        X_train_t,
        y_train,
        X_validation_t,
        y_validation,
        n_trials=args.lightgbm_trials,
    )
    lightgbm_validation_probability = lightgbm_model.predict_proba(
        X_validation_t,
    )[:, 1]
    lightgbm_validation_metrics = classification_metrics(
        y_validation,
        lightgbm_validation_probability,
        include_calibration=False,
    )

    champion_name = choose_champion(
        logistic_validation_metrics,
        lightgbm_validation_metrics,
    )

    if champion_name == "lightgbm":
        champion_model = lightgbm_model
        raw_calibration_probability = lightgbm_model.predict_proba(
            X_calibration_t,
        )[:, 1]
    else:
        champion_model = logistic_model
        raw_calibration_probability = logistic_model.predict_proba(
            X_calibration_t,
        )[:, 1]

    calibrator = fit_calibrator(
        method=args.calibration_method,
        raw_probability=raw_calibration_probability,
        y=y_calibration.to_numpy(),
    )

    calibrated_probability = calibrator.predict(
        raw_calibration_probability,
    )

    calibration_metrics = classification_metrics(
        y_calibration,
        calibrated_probability,
        include_calibration=True,
    )

    band_thresholds = fit_probability_bands(
        calibrated_probability,
        p1_fraction=0.10,
        p2_fraction=0.50,
    )

    split_info = describe_split(
        dataset.created_at,
        dataset.split,
    )

    pd.DataFrame(trial_history).to_csv(
        output_dir / "lightgbm_trials.csv",
        index=False,
    )

    reliability_table(
        y_calibration,
        calibrated_probability,
        n_bins=10,
    ).to_csv(
        output_dir / "reliability_bins.csv",
        index=False,
    )

    plot_reliability_diagram(
        y_calibration,
        calibrated_probability,
        output_dir / "reliability_diagram.png",
    )

    command = " ".join(__import__("sys").argv)

    write_eval_report(
        output_dir=output_dir,
        command=command,
        split_info=split_info,
        logistic_metrics=logistic_validation_metrics,
        lightgbm_metrics=lightgbm_validation_metrics,
        champion_name=champion_name,
        calibration_metrics=calibration_metrics,
        band_thresholds=band_thresholds,
        n_trials=len(trial_history),
        calibration_method=args.calibration_method,
    )

    write_model_card(
        output_dir=output_dir,
        champion_name=champion_name,
        calibration_method=args.calibration_method,
        split_info=split_info,
        calibration_metrics=calibration_metrics,
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    bundle = {
        "version": "m2-1.0.0",
        "champion_name": champion_name,
        "preprocessor": preprocessor,
        "logistic_model": logistic_model,
        "lightgbm_model": lightgbm_model,
        "champion_model": champion_model,
        "calibrator": calibrator,
        "calibration_method": args.calibration_method,
        "band_thresholds": band_thresholds,
        "feature_names": feature_names,
        "random_seed": RANDOM_SEED,
        "split_config": {
            "hidden_start": args.hidden_start,
            "validation_weeks": args.validation_weeks,
            "calibration_fraction": args.calibration_fraction,
        },
        "metrics": {
            "logistic_validation": logistic_validation_metrics,
            "lightgbm_validation": lightgbm_validation_metrics,
            "calibration": calibration_metrics,
        },
        "split_info": split_info,
    }

    joblib.dump(
        bundle,
        output_dir / "model_bundle.joblib",
        compress=3,
    )

    metadata = json_safe(
        {
            "version": "m2-1.0.0",
            "champion_name": champion_name,
            "calibration_method": args.calibration_method,
            "band_thresholds": band_thresholds,
            "metrics": bundle["metrics"],
            "split_info": split_info,
            "feature_names": feature_names,
            "lightgbm_trials": len(trial_history),
            "best_iteration": int(lightgbm_model.best_iteration_ or lightgbm_model.n_estimators),
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Calibration predictions for inspection.
    calibration_predictions = pd.DataFrame(
        {
            "lead_id": X_calibration.index,
            "y_true": y_calibration.to_numpy(),
            "raw_probability": raw_calibration_probability,
            "calibrated_probability": calibrated_probability,
            "band": assign_band(
                calibrated_probability,
                band_thresholds,
            ),
        }
    )
    calibration_predictions.to_csv(
        output_dir / "calibration_predictions.csv",
        index=False,
    )

    return {
        "champion_name": champion_name,
        "logistic_validation_metrics": logistic_validation_metrics,
        "lightgbm_validation_metrics": lightgbm_validation_metrics,
        "calibration_metrics": calibration_metrics,
        "band_thresholds": band_thresholds,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeadIQ M2: train, calibrate and evaluate models"
    )
    parser.add_argument("--leads", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--calls", required=True)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--localities", required=True)
    parser.add_argument("--export-date", required=True)
    parser.add_argument(
        "--hidden-start",
        default="2026-07-05 00:00:00+05:30",
    )
    parser.add_argument(
        "--validation-weeks",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--calibration-method",
        choices=["platt", "isotonic"],
        default="platt",
    )
    parser.add_argument(
        "--lead-clusters",
        default="artifacts/m4/lead_clusters.csv",
        help="M4 lead_clusters.csv used to build earlier_enquiries_count.",
    )
    parser.add_argument(
        "--lightgbm-trials",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/m2",
    )
    args = parser.parse_args()

    if args.lightgbm_trials > MAX_TRIALS:
        raise SystemExit(
            f"lightgbm-trials cannot exceed {MAX_TRIALS}"
        )

    result = run_training(args)
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
