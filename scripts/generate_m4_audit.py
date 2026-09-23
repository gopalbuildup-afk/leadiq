from pathlib import Path
import itertools
import random
import re
from difflib import SequenceMatcher

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

LEADS_PATH = ROOT / "data" / "leads.csv"
OLD_REVIEW_PATH = ROOT / "artifacts" / "m4" / "manual_review.csv"
OUT_PATH = ROOT / "artifacts" / "m4" / "manual_audit.csv"

SEED = 20260922

# Frozen production candidate rule.
# DO NOT CHANGE THIS FOR THE AUDIT.
NAME_PREFIX_THRESHOLD = 0.0


# ---------------------------------------------------------
# Normalisation
# ---------------------------------------------------------

def normalize_phone(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()
    digits = re.sub(r"\D", "", s)

    # Handle Indian formats.
    if digits.startswith("0091"):
        digits = digits[4:]
    elif digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    return digits if len(digits) == 10 else ""


def normalize_email(value):
    if pd.isna(value):
        return ""

    return str(value).strip().casefold()


def normalize_name(value):
    if pd.isna(value):
        return ""

    s = str(value).strip().casefold()
    s = re.sub(r"\s+", " ", s)
    return s


def names_are_token_prefixes(a, b):
    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return False

    ta = a.split()
    tb = b.split()

    if len(ta) > len(tb):
        ta, tb = tb, ta

    return all(
        long.startswith(short)
        for short, long in zip(ta, tb)
    )


def name_similarity(a, b):
    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------
# Frozen rule
# ---------------------------------------------------------

def frozen_rule(row):
    phone_match = bool(row["phone_match"])
    email_match = bool(row["email_match"])
    name_exact = bool(row["name_exact"])
    name_prefix = bool(row["name_prefix_match"])

    return (
        (phone_match and email_match)
        or
        ((phone_match or email_match) and
         (name_exact or name_prefix))
    )


# ---------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------

def make_candidates(df):
    phone_groups = {}
    email_groups = {}

    for idx, row in df.iterrows():
        phone = normalize_phone(row.get("phone"))
        email = normalize_email(row.get("email"))

        if phone:
            phone_groups.setdefault(phone, []).append(idx)

        if email:
            email_groups.setdefault(email, []).append(idx)

    pairs = {}

    def add_pair(a, b, reason):
        if a == b:
            return

        x, y = sorted((a, b))
        key = (x, y)

        if key not in pairs:
            pairs[key] = set()

        pairs[key].add(reason)

    for group in phone_groups.values():
        for a, b in itertools.combinations(group, 2):
            add_pair(a, b, "phone_only")

    for group in email_groups.values():
        for a, b in itertools.combinations(group, 2):
            add_pair(a, b, "email_only")

    rows = []

    for (a, b), reasons in pairs.items():
        if "phone_only" in reasons and "email_only" in reasons:
            block_reason = "both_keys"
        elif "phone_only" in reasons:
            block_reason = "phone_only"
        else:
            block_reason = "email_only"

        rows.append((a, b, block_reason))

    return rows


# ---------------------------------------------------------
# Exclude the previous development/manual sample
# ---------------------------------------------------------

def pair_key(a, b):
    return tuple(sorted((str(a), str(b))))


def load_old_pairs(path):
    if not path.exists():
        return set()

    old = pd.read_csv(path)

    if "lead_a" not in old.columns or "lead_b" not in old.columns:
        return set()

    return {
        pair_key(row.lead_a, row.lead_b)
        for _, row in old.iterrows()
    }


# ---------------------------------------------------------
# Build audit rows
# ---------------------------------------------------------

def build_audit_rows(df, candidates, old_pairs):
    rows = []

    for a_idx, b_idx, block_reason in candidates:
        a = df.loc[a_idx]
        b = df.loc[b_idx]

        lead_a = str(a["lead_id"])
        lead_b = str(b["lead_id"])

        if pair_key(lead_a, lead_b) in old_pairs:
            continue

        phone_a = normalize_phone(a.get("phone"))
        phone_b = normalize_phone(b.get("phone"))

        email_a = normalize_email(a.get("email"))
        email_b = normalize_email(b.get("email"))

        name_a = normalize_name(a.get("full_name"))
        name_b = normalize_name(b.get("full_name"))

        phone_match = bool(phone_a and phone_b and phone_a == phone_b)
        email_match = bool(email_a and email_b and email_a == email_b)
        name_exact = bool(name_a and name_b and name_a == name_b)
        name_prefix = names_are_token_prefixes(name_a, name_b)
        similarity = name_similarity(name_a, name_b)

        row = {
            "lead_a": lead_a,
            "lead_b": lead_b,
            "block_reason": block_reason,

            "full_name_a": a.get("full_name", ""),
            "full_name_b": b.get("full_name", ""),

            "phone_a": a.get("phone", ""),
            "phone_b": b.get("phone", ""),

            "email_a": a.get("email", ""),
            "email_b": b.get("email", ""),

            "source_a": a.get("source", ""),
            "source_b": b.get("source", ""),

            "branch_a": a.get("branch", ""),
            "branch_b": b.get("branch", ""),

            "course_a": a.get("course", ""),
            "course_b": b.get("course", ""),

            "phone_match": phone_match,
            "email_match": email_match,
            "name_exact": name_exact,
            "name_similarity": round(similarity, 6),
            "name_prefix_match": name_prefix,

            "source_match": (
                str(a.get("source", "")).strip().casefold()
                ==
                str(b.get("source", "")).strip().casefold()
            ),

            "branch_match": (
                str(a.get("branch", "")).strip().casefold()
                ==
                str(b.get("branch", "")).strip().casefold()
            ),

            "course_match": (
                str(a.get("course", "")).strip().casefold()
                ==
                str(b.get("course", "")).strip().casefold()
            ),
        }

        row["rule_prediction"] = int(frozen_rule(row))

        # Reviewer fills these only.
        row["same_person"] = ""
        row["review_notes"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Stratified fresh sample
# ---------------------------------------------------------

def main():
    print(f"Reading leads: {LEADS_PATH}")

    df = pd.read_csv(LEADS_PATH)

    required = [
    "lead_id",
    "full_name",
    "phone",
    "email",
    "source",
    "branch",
    "course_interest",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in leads.csv: {missing}"
        )

    old_pairs = load_old_pairs(OLD_REVIEW_PATH)

    print(f"Previous manual pairs excluded: {len(old_pairs)}")

    candidates = make_candidates(df)

    print(f"Total blocked candidate pairs: {len(candidates)}")

    audit = build_audit_rows(df, candidates, old_pairs)

    if audit.empty:
        raise RuntimeError("No fresh candidate pairs remain.")

    print("\nFresh candidate counts:")

    print(
        audit["block_reason"]
        .value_counts()
        .to_string()
    )

    rng = random.Random(SEED)

    targets = {
        "both_keys": 77,
        "phone_only": 77,
        "email_only": 46,
    }

    selected_parts = []

    for reason, n in targets.items():
        part = audit[audit["block_reason"] == reason].copy()

        if len(part) < n:
            raise RuntimeError(
                f"Not enough fresh {reason} candidates. "
                f"Need {n}, found {len(part)}."
            )

        indices = list(part.index)
        rng.shuffle(indices)

        selected_parts.append(part.loc[indices[:n]])

    result = pd.concat(
        selected_parts,
        ignore_index=True
    )

    # Shuffle the final 200 so the reviewer does not see
    # all candidates of one type together.
    indices = list(result.index)
    rng.shuffle(indices)

    result = result.loc[indices].reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUT_PATH, index=False)

    print("\nCreated fresh audit:")
    print(OUT_PATH)

    print("\nAudit size:", len(result))

    print("\nAudit strata:")
    print(
        result["block_reason"]
        .value_counts()
        .to_string()
    )

    print("\nFrozen rule predictions:")
    print(
        result["rule_prediction"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nReviewer instructions:")
    print("  Fill only: same_person and review_notes")
    print("  same_person = 1 if both enquiries appear to be the same person")
    print("  same_person = 0 if they appear to be different people")
    print("  Consider phone, email, name, source, branch and course together.")
    print("  Do NOT change rule_prediction.")


if __name__ == "__main__":
    main()