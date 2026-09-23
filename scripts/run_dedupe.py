from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from leadiq.dedupe import (
    DEFAULT_MAX_CLUSTER_SIZE,
    DEFAULT_WEAK_CLUSTER_SIZE,
    RULE_NAMES,
    build_candidate_evidence_frame,
    build_candidate_pairs,
    build_manual_review_sample,
    build_clusters,
    choose_golden_records,
    normalize_email,
    normalize_name,
    normalize_phone,
)


# M4 production rule frozen before the fresh manual audit.
FROZEN_RULE_NAME = "both_keys_or_exact_or_prefix"


def prepare_leads(leads_path: str) -> pd.DataFrame:
    leads = pd.read_csv(leads_path)

    required = {
        "lead_id",
        "created_at",
        "source",
        "full_name",
        "phone",
        "email",
    }

    missing = required - set(leads.columns)

    if missing:
        raise ValueError(
            f"Missing required lead columns: {sorted(missing)}"
        )

    leads["lead_id"] = leads["lead_id"].astype(str)

    if leads["lead_id"].duplicated().any():
        duplicate_ids = (
            leads.loc[
                leads["lead_id"].duplicated(),
                "lead_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"lead_id must be unique. "
            f"Example duplicates: {duplicate_ids}"
        )

    leads["phone_norm"] = leads["phone"].map(normalize_phone)
    leads["email_norm"] = leads["email"].map(normalize_email)
    leads["name_norm"] = leads["full_name"].map(normalize_name)

    return leads


def load_and_validate_rule(rule_file: str | None) -> str:
    """
    Resolve the production deduplication rule.

    The M4 production rule is frozen before the fresh manual audit.
    A rule file may only confirm the same frozen rule; it cannot
    silently replace it with an older or different rule.
    """

    if not rule_file:
        return FROZEN_RULE_NAME

    path = Path(rule_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Rule file not found: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    rule_name = payload.get("selected_rule")

    if not rule_name:
        raise ValueError(
            f"Rule file {path} does not contain selected_rule."
        )

    if rule_name != FROZEN_RULE_NAME:
        raise ValueError(
            "Rule file contains a different production rule.\n"
            f"Expected frozen rule: {FROZEN_RULE_NAME}\n"
            f"Found: {rule_name}\n"
            "Refusing to run with a non-frozen M4 rule."
        )

    if payload.get("audit_used_for_tuning") is True:
        raise ValueError(
            "Rule metadata says the audit was used for tuning. "
            "Refusing to use this rule file."
        )

    if payload.get("audit_used_for_selection") is True:
        raise ValueError(
            "Rule metadata says the audit was used for selection. "
            "Refusing to use this rule file."
        )

    return rule_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LeadIQ M4 deduplication with phone/email blocking, "
            "frozen production rule, union-find and golden records."
        )
    )

    parser.add_argument(
        "--leads",
        required=True,
    )

    parser.add_argument(
        "--stage-history",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        default="artifacts/m4",
    )

    parser.add_argument(
        "--generate-review",
        action="store_true",
        help=(
            "Generate candidate/evidence/manual-review CSVs and "
            "stop before clustering."
        ),
    )

    parser.add_argument(
        "--review-sample-size",
        type=int,
        default=200,
        help=(
            "Manual-review sample size; the M4 brief requires "
            "at least 150."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rule",
        default=FROZEN_RULE_NAME,
        choices=RULE_NAMES,
        help=(
            "Production pair-merge rule. The M4 production rule "
            f"is frozen as {FROZEN_RULE_NAME}."
        ),
    )

    parser.add_argument(
        "--rule-file",
        help=(
            "Optional dedupe_rule.json created by "
            "evaluate_dedupe.py. It must confirm the frozen rule."
        ),
    )

    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=DEFAULT_MAX_CLUSTER_SIZE,
    )

    parser.add_argument(
        "--weak-cluster-size",
        type=int,
        default=DEFAULT_WEAK_CLUSTER_SIZE,
    )

    args = parser.parse_args()

    if args.review_sample_size < 150:
        raise ValueError(
            "--review-sample-size must be >= 150 for the M4 brief."
        )

    # ---------------------------------------------------------
    # Resolve production rule
    # ---------------------------------------------------------

    if args.rule_file:
        args.rule = load_and_validate_rule(args.rule_file)

    elif args.rule != FROZEN_RULE_NAME:
        raise ValueError(
            "The M4 production rule is frozen.\n"
            f"Expected: {FROZEN_RULE_NAME}\n"
            f"Received: {args.rule}\n"
            "Use the frozen production rule."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    leads = prepare_leads(args.leads)

    stage_history = pd.read_csv(
        args.stage_history
    )

    # ---------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------

    candidate_pairs = build_candidate_pairs(
        leads
    )

    evidence = build_candidate_evidence_frame(
        leads,
        candidate_pairs,
    )

    manual_review = build_manual_review_sample(
        leads,
        candidate_pairs,
        sample_size=args.review_sample_size,
        random_seed=args.seed,
    )

    evidence.to_csv(
        out_dir / "candidate_pairs.csv",
        index=False,
    )

    manual_review.to_csv(
        out_dir / "manual_review.csv",
        index=False,
    )

    naive_pairs = (
        len(leads) * (len(leads) - 1) // 2
    )

    avoided_pairs = (
        naive_pairs - len(candidate_pairs)
    )

    reduction = (
        1.0 - (len(candidate_pairs) / naive_pairs)
        if naive_pairs
        else 0.0
    )

    print("=" * 72)
    print("M4 DEDUPLICATION - CANDIDATE GENERATION")
    print("=" * 72)

    print(
        f"Total leads:                   {len(leads):,}"
    )

    print(
        f"Naive nC2 comparisons:         {naive_pairs:,}"
    )

    print(
        f"Candidate pairs:               {len(candidate_pairs):,}"
    )

    print(
        f"Avoided comparisons:           {avoided_pairs:,}"
    )

    print(
        f"Comparison reduction:          {reduction:.2%}"
    )

    print(
        f"Manual review rows:             {len(manual_review):,}"
    )

    print(
        f"Production rule:               {args.rule}"
    )

    print(
        f"Max cluster size:              {args.max_cluster_size}"
    )

    print(
        f"Weak-cluster flag threshold:   {args.weak_cluster_size}"
    )

    print(
        f"Saved candidate pairs:          "
        f"{out_dir / 'candidate_pairs.csv'}"
    )

    print(
        f"Saved manual review sheet:      "
        f"{out_dir / 'manual_review.csv'}"
    )

    if args.generate_review:
        print()
        print(
            "Fill manual_review.csv: "
            "same_person = 1/0 and review_notes."
        )

        print(
            "Then evaluate the labels before final clustering."
        )

        return

    # ---------------------------------------------------------
    # Final clustering using frozen production rule
    # ---------------------------------------------------------

    clusters, accepted_pairs, rejected_pairs = build_clusters(
        leads,
        candidate_pairs,
        rule_name=args.rule,
        max_cluster_size=args.max_cluster_size,
    )

    # ---------------------------------------------------------
    # Golden records
    # ---------------------------------------------------------

    cluster_df = choose_golden_records(
        leads,
        clusters,
        stage_history=stage_history,
        weak_cluster_size=args.weak_cluster_size,
        weak_pair_rows=accepted_pairs,
    )

    cluster_df.to_csv(
        out_dir / "lead_clusters.csv",
        index=False,
    )

    pd.DataFrame(accepted_pairs).to_csv(
        out_dir / "accepted_pairs.csv",
        index=False,
    )

    pd.DataFrame(rejected_pairs).to_csv(
        out_dir / "rejected_by_cluster_cap.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    duplicate_clusters = int(
        cluster_df.loc[
            cluster_df["cluster_size"] > 1,
            "cluster_id",
        ].nunique()
    )

    weak_clusters = int(
        cluster_df.loc[
            cluster_df["weak_cluster"] == True,
            "cluster_id",
        ].nunique()
    )

    duplicate_leads = int(
        (cluster_df["cluster_size"] > 1).sum()
    )

    duplicate_rate = (
        duplicate_leads / len(leads)
        if len(leads)
        else 0.0
    )

    largest = (
        cluster_df
        .groupby("cluster_id")["lead_id"]
        .size()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    summary = pd.DataFrame(
        [
            {
                "total_leads": len(leads),
                "naive_pairs": naive_pairs,
                "candidate_pairs": len(candidate_pairs),
                "avoided_comparisons": avoided_pairs,
                "comparison_reduction": reduction,
                "duplicate_leads": duplicate_leads,
                "duplicate_rate": duplicate_rate,
                "accepted_pairs": len(accepted_pairs),
                "rejected_by_cluster_cap": len(
                    rejected_pairs
                ),
                "clusters": cluster_df[
                    "cluster_id"
                ].nunique(),
                "duplicate_clusters": duplicate_clusters,
                "leads_retained": len(cluster_df),
                "weak_clusters": weak_clusters,
                "rule": args.rule,
                "max_cluster_size": args.max_cluster_size,
                "weak_cluster_size": args.weak_cluster_size,
            }
        ]
    )

    summary.to_csv(
        out_dir / "dedupe_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("M4 DEDUPLICATION - CLUSTERING")
    print("=" * 72)

    print(
        f"Rule:                         {args.rule}"
    )

    print(
        f"Accepted duplicate edges:    "
        f"{len(accepted_pairs):,}"
    )

    print(
        f"Rejected by size cap:         "
        f"{len(rejected_pairs):,}"
    )

    print(
        f"Total clusters (incl. singles): "
        f"{cluster_df['cluster_id'].nunique():,}"
    )

    print(
        f"Duplicate clusters:           "
        f"{duplicate_clusters:,}"
    )

    print(
        f"Leads retained:               "
        f"{len(cluster_df):,}"
    )

    print(
        f"Duplicate leads:              "
        f"{duplicate_leads:,}"
    )

    print(
        f"Duplicate rate:               "
        f"{duplicate_rate:.2%}"
    )

    print(
        f"Weak clusters:                "
        f"{weak_clusters:,}"
    )

    print(
        f"Five largest clusters:        "
        f"{largest}"
    )

    print(
        f"Saved:                        "
        f"{out_dir / 'lead_clusters.csv'}"
    )


if __name__ == "__main__":
    main()