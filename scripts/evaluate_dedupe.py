from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from leadiq.dedupe import (
    RULE_NAMES,
    names_are_token_prefixes,
    pair_is_duplicate,
)


# IMPORTANT:
# This rule was frozen BEFORE the fresh audit was labeled.
# The fresh audit is evaluation-only and must NOT be used to tune
# or select the production rule.
FROZEN_RULE_NAME = "both_keys_or_exact_or_prefix"


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "n": int(len(y_true)),
        "positive_pairs": int((y_true == 1).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate M4 duplicate rules on hand-labeled candidate pairs."
    )
    parser.add_argument(
        "--review",
        required=True,
        help="CSV created by run_dedupe.py; fill same_person with 1 or 0.",
    )
    parser.add_argument("--out-dir", default="artifacts/m4")
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.85)
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype={"lead_a": str, "lead_b": str})

    if "same_person" not in review.columns:
        raise ValueError("Review CSV must contain same_person")

    labels = pd.to_numeric(review["same_person"], errors="coerce")

    if labels.isna().any():
        missing = int(labels.isna().sum())
        raise ValueError(
            f"{missing} rows are unlabeled. "
            "Fill same_person with 1 or 0 for every review row."
        )

    invalid = ~labels.isin([0, 1])
    if invalid.any():
        raise ValueError("same_person must contain only 0 or 1")

    required_features = {
        "phone_match",
        "email_match",
        "name_exact",
        "name_similarity",
        "full_name_a",
        "full_name_b",
    }

    missing = required_features - set(review.columns)

    if missing:
        raise ValueError(
            f"Review CSV missing feature columns: {sorted(missing)}"
        )

    review["same_person"] = labels.astype(int)

    # Candidate recall caveat:
    # this measures recall among the hand-labeled candidate pairs
    # produced by phone/email blocking, not all nC2 pairs.
    results = []

    for rule_name in RULE_NAMES:
        predicted = []

        for _, row in review.iterrows():
            from leadiq.dedupe import PairFeatures

            features = PairFeatures(
                lead_a=str(row["lead_a"]),
                lead_b=str(row["lead_b"]),
                block_reason=str(row.get("block_reason", "")),
                phone_match=bool(row["phone_match"]),
                email_match=bool(row["email_match"]),
                name_exact=bool(row["name_exact"]),
                name_similarity=float(row["name_similarity"]),
                name_prefix_match=names_are_token_prefixes(
                    row.get("full_name_a"),
                    row.get("full_name_b"),
                ),
                source_match=bool(row.get("source_match", False)),
                branch_match=bool(row.get("branch_match", False)),
                course_match=bool(row.get("course_match", False)),
                key_match_count=(
                    int(bool(row["phone_match"]))
                    + int(bool(row["email_match"]))
                ),
                single_key_match=(
                    int(bool(row["phone_match"]))
                    + int(bool(row["email_match"]))
                    == 1
                ),
                evidence_strength="review",
            )

            predicted.append(
                int(pair_is_duplicate(features, rule_name))
            )

        metrics = compute_metrics(
            review["same_person"],
            pd.Series(predicted),
        )

        metrics["rule"] = rule_name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "rule",
            "n",
            "positive_pairs",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
        ]
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        out_dir / "dedupe_rule_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Frozen production rule evaluation
    # ---------------------------------------------------------

    selected_row = results_df[
        results_df["rule"] == FROZEN_RULE_NAME
    ]

    if selected_row.empty:
        raise RuntimeError(
            f"Frozen production rule '{FROZEN_RULE_NAME}' "
            "was not evaluated. Check RULE_NAMES in leadiq/dedupe.py."
        )

    selected = selected_row.iloc[0]

    audit_meets_targets = bool(
        selected["precision"] >= args.min_precision
        and selected["recall"] >= args.min_recall
    )

    selection_note = (
        "Production rule was frozen before the fresh audit was labeled. "
        "The fresh audit labels were NOT used to tune or select the rule."
    )

    rule_payload = {
        "selected_rule": FROZEN_RULE_NAME,
        "status": "frozen",
        "selection_basis": "frozen_before_fresh_manual_audit",
        "audit_used_for_tuning": False,
        "audit_used_for_selection": False,
        "audit_size": int(selected["n"]),
        "audit_positive_pairs": int(selected["positive_pairs"]),
        "audit_tp": int(selected["tp"]),
        "audit_fp": int(selected["fp"]),
        "audit_fn": int(selected["fn"]),
        "audit_tn": int(selected["tn"]),
        "audit_precision": float(selected["precision"]),
        "audit_recall": float(selected["recall"]),
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "audit_meets_targets": audit_meets_targets,
        "selection_note": selection_note,
        "evaluation_scope": (
            "fresh hand-labeled blocked candidate audit"
        ),
        "rule_description": (
            "(same phone AND same email) OR "
            "((same phone OR same email) AND "
            "(exact name OR token-prefix name))"
        ),
        "warning": (
            "Precision/recall are sample estimates and recall is "
            "conditional on the phone/email blocking stage. "
            "The fresh audit was not used to tune the frozen "
            "production rule."
        ),
    }

    (out_dir / "dedupe_rule.json").write_text(
        json.dumps(rule_payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("M4 MANUAL-LABEL EVALUATION")
    print("=" * 72)

    print(results_df.to_string(index=False))

    print("\n" + selection_note)
    print(f"Frozen production rule: {FROZEN_RULE_NAME}")
    print(
        f"Fresh audit precision:  {selected['precision']:.4f}"
    )
    print(
        f"Fresh audit recall:     {selected['recall']:.4f}"
    )
    print(
        "Audit meets targets:    "
        f"{'YES' if audit_meets_targets else 'NO'}"
    )

    print(
        f"Saved: {out_dir / 'dedupe_rule_metrics.csv'}"
    )
    print(
        f"Saved: {out_dir / 'dedupe_rule.json'}"
    )


if __name__ == "__main__":
    main()
