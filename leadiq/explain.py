from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model_data import load_m2_dataset
from .reasons import reason_for_feature


def _get_shap_values(model, X_transformed):
    import shap
    explainer = shap.TreeExplainer(model)
    dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else np.asarray(X_transformed)
    try:
        explanation = explainer(dense, check_additivity=False)
        values = explanation.values
    except TypeError:
        raw = explainer.shap_values(dense)
        values = raw[1] if isinstance(raw, list) and len(raw) == 2 else raw

    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return values


def generate_lightgbm_reasons(
    bundle_path: str | Path,
    leads_path: str,
    messages_path: str,
    calls_path: str,
    stages_path: str,
    localities_path: str,
    export_date: str,
    hidden_start: str,
    output_path: str | Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    bundle = joblib.load(bundle_path)

    if bundle["champion_name"] != "lightgbm":
        raise ValueError(
            "TreeExplainer reasons require LightGBM to be the champion. "
            "The selected champion is: " + bundle["champion_name"]
        )

    dataset = load_m2_dataset(
        leads_path=leads_path,
        messages_path=messages_path,
        calls_path=calls_path,
        stages_path=stages_path,
        localities_path=localities_path,
        export_date=export_date,
        hidden_start=hidden_start,
        validation_weeks=bundle["split_config"]["validation_weeks"],
        calibration_fraction=bundle["split_config"]["calibration_fraction"],
    )

    X = dataset.X
    if max_rows is not None:
        X = X.head(max_rows)

    preprocessor = bundle["preprocessor"]
    model = bundle["lightgbm_model"]

    transformed = preprocessor.transform(X)
    shap_values = _get_shap_values(model, transformed)
    feature_names = list(preprocessor.get_feature_names_out())

    output_rows = []

    for i, lead_id in enumerate(X.index):
        row = X.loc[lead_id]
        contributions = shap_values[i]
        order = np.argsort(np.abs(contributions))[::-1]

        reasons = []
        for feature_idx in order:
            if len(reasons) >= 3:
                break
            if abs(contributions[feature_idx]) < 1e-10:
                continue

            reason = reason_for_feature(
                feature_names[feature_idx],
                row,
            )
            if reason is None:
                continue

            reasons.append(reason)

        while len(reasons) < 3:
            reasons.append(("No additional mapped explanation", "Aur koi mapped explanation nahi"))

        output_rows.append(
            {
                "lead_id": lead_id,
                "reason_1_en": reasons[0][0],
                "reason_1_hi": reasons[0][1],
                "reason_2_en": reasons[1][0],
                "reason_2_hi": reasons[1][1],
                "reason_3_en": reasons[2][0],
                "reason_3_hi": reasons[2][1],
            }
        )

    result = pd.DataFrame(output_rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M2 SHAP reasons")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--leads", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--calls", required=True)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--localities", required=True)
    parser.add_argument("--export-date", required=True)
    parser.add_argument("--hidden-start", default="2026-07-05 00:00:00+05:30")
    parser.add_argument("--output", default="artifacts/m2/shap_reasons.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    result = generate_lightgbm_reasons(
        bundle_path=args.bundle,
        leads_path=args.leads,
        messages_path=args.messages,
        calls_path=args.calls,
        stages_path=args.stages,
        localities_path=args.localities,
        export_date=args.export_date,
        hidden_start=args.hidden_start,
        output_path=args.output,
        max_rows=args.max_rows,
    )

    print(f"Wrote {len(result)} explanation rows to {args.output}")


if __name__ == "__main__":
    main()
