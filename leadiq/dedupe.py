from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_MAX_CLUSTER_SIZE = 10
DEFAULT_WEAK_CLUSTER_SIZE = 4

# Ordered from weakest to strongest outcome for golden-record reporting.
STAGE_PRIORITY = {
    "new": 0,
    "lost": 0,
    "dormant": 0,
    "contacted": 1,
    "interested": 2,
    "visit scheduled": 3,
    "center visited": 4,
    "demo scheduled": 5,
    "seat reserved": 6,
    "admission done": 7,
}


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def parse_timestamp(value: object, default_timezone: str = DEFAULT_TIMEZONE):
    """Parse a timestamp and return a timezone-aware UTC Timestamp.

    Lead exports in this project contain naive IST timestamps, while some
    other files may contain explicit offsets. Naive values are therefore
    explicitly localized to Asia/Kolkata before conversion to UTC.
    """

    if pd.isna(value):
        return pd.NaT

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT

    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize(default_timezone)

    return ts.tz_convert("UTC")


def parse_timestamp_series(
    series: pd.Series,
    default_timezone: str = DEFAULT_TIMEZONE,
) -> pd.Series:
    return series.map(
        lambda value: parse_timestamp(value, default_timezone=default_timezone)
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_phone(value: object) -> str | None:
    """Normalize phone values to canonical 10-digit form.

    The task requires canonical 10-digit normalization; do not reject a
    number based on its first digit because the supplied dataset contains
    valid test/synthetic numbers outside the usual Indian mobile prefixes.
    """

    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None

    # International dialing prefix without '+'.
    if digits.startswith("0091"):
        digits = digits[4:]

    # +91 / 91 + 10-digit number.
    elif digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]

    # Leading 0 + 10-digit number.
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    if len(digits) != 10:
        return None

    return digits


def normalize_email(value: object) -> str | None:
    if pd.isna(value):
        return None

    normalized = str(value).strip().casefold()
    return normalized if normalized else None


def normalize_name(value: object) -> str | None:
    if pd.isna(value):
        return None

    normalized = str(value).strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized if normalized else None


def names_are_token_prefixes(name_a: object, name_b: object) -> bool:
    """True when the shorter normalized name is a token-wise prefix of the longer.

    Examples: ``megha i`` -> ``megha iyer`` and ``aman`` -> ``aman kapoor``.
    This is supporting evidence only; it is never a standalone duplicate key.
    """
    if not name_a or not name_b or pd.isna(name_a) or pd.isna(name_b):
        return False
    a = str(name_a).strip().split()
    b = str(name_b).strip().split()
    if not a or not b or a == b:
        return False
    if len(a) > len(b):
        a, b = b, a
    if len(a) >= len(b):
        return False
    return all(long.startswith(short) for short, long in zip(a, b))


def name_similarity(name_a: str | None, name_b: str | None) -> float:
    """Lightweight comparison evidence; this is not a trained fuzzy matcher."""

    if not name_a or not name_b:
        return 0.0
    if name_a == name_b:
        return 1.0
    return float(SequenceMatcher(None, name_a, name_b).ratio())


# ---------------------------------------------------------------------------
# Union-Find with cluster-size guard
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, items: Iterable[str]):
        items = [str(item) for item in items]
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}
        self.members = {item: {item} for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def component_size(self, item: str) -> int:
        return len(self.members[self.find(item)])

    def component_members(self, item: str) -> set[str]:
        return set(self.members[self.find(item)])

    def union_if_within_cap(
        self,
        a: str,
        b: str,
        max_cluster_size: int,
    ) -> tuple[bool, str]:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False, "already_connected"

        prospective_size = len(self.members[root_a]) + len(self.members[root_b])
        if prospective_size > max_cluster_size:
            return False, "cluster_size_cap"

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.members[root_a].update(self.members[root_b])
        del self.members[root_b]

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        return True, "accepted"


# ---------------------------------------------------------------------------
# Candidate blocking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePair:
    lead_a: str
    lead_b: str
    block_reason: str


def build_candidate_pairs(df: pd.DataFrame) -> list[CandidatePair]:
    """Block candidates on exact normalized phone OR exact normalized email.

    Names are intentionally not used as a blocking key: a name-only collision
    can create a large number of false candidates. Phone/email are candidate
    keys only; they are not themselves duplicate decisions.
    """

    required = {"lead_id", "phone_norm", "email_norm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required normalized columns: {sorted(missing)}")

    blocks: dict[str, dict[str, set[str]]] = {"phone": {}, "email": {}}

    for _, row in df.iterrows():
        lead_id = str(row["lead_id"])
        for block_type, key_col in (("phone", "phone_norm"), ("email", "email_norm")):
            key = row[key_col]
            if pd.isna(key) or key is None or not str(key).strip():
                continue
            blocks[block_type].setdefault(str(key), set()).add(lead_id)

    pairs: dict[tuple[str, str], set[str]] = {}

    for block_type, values in blocks.items():
        for lead_ids in values.values():
            if len(lead_ids) < 2:
                continue
            for a, b in combinations(sorted(lead_ids), 2):
                pair = (a, b)
                pairs.setdefault(pair, set()).add(block_type)

    return [
        CandidatePair(
            lead_a=a,
            lead_b=b,
            block_reason="+".join(sorted(reasons)),
        )
        for (a, b), reasons in sorted(pairs.items())
    ]


# ---------------------------------------------------------------------------
# Pair comparison evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairFeatures:
    lead_a: str
    lead_b: str
    block_reason: str
    phone_match: bool
    email_match: bool
    name_exact: bool
    name_similarity: float
    name_prefix_match: bool
    source_match: bool
    branch_match: bool
    course_match: bool
    key_match_count: int
    single_key_match: bool
    evidence_strength: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _same_value(row_a: pd.Series, row_b: pd.Series, column: str) -> bool:
    a = row_a.get(column)
    b = row_b.get(column)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return str(a) == str(b)


def compare_pair(
    row_a: pd.Series,
    row_b: pd.Series,
    candidate: CandidatePair | None = None,
) -> PairFeatures:
    """Return auditable evidence; do not make identity decisions here."""

    phone_match = _same_value(row_a, row_b, "phone_norm")
    email_match = _same_value(row_a, row_b, "email_norm")
    name_exact = _same_value(row_a, row_b, "name_norm")

    similarity = name_similarity(
        row_a.get("name_norm"),
        row_b.get("name_norm"),
    )

    name_prefix_match = names_are_token_prefixes(
        row_a.get("name_norm"),
        row_b.get("name_norm"),
    )

    key_match_count = int(phone_match) + int(email_match)

    if key_match_count == 2 and name_exact:
        evidence_strength = "very_strong"
    elif key_match_count == 2:
        evidence_strength = "strong"
    elif key_match_count == 1 and name_exact:
        evidence_strength = "strong"
    elif key_match_count == 1 and similarity >= 0.90:
        evidence_strength = "medium"
    else:
        evidence_strength = "weak"

    return PairFeatures(
        lead_a=str(row_a["lead_id"]),
        lead_b=str(row_b["lead_id"]),
        block_reason=candidate.block_reason if candidate else "",
        phone_match=phone_match,
        email_match=email_match,
        name_exact=name_exact,
        name_similarity=round(similarity, 6),
        name_prefix_match=name_prefix_match,
        source_match=_same_value(row_a, row_b, "source"),
        branch_match=_same_value(row_a, row_b, "branch"),
        course_match=_same_value(row_a, row_b, "course_interest"),
        key_match_count=key_match_count,
        single_key_match=key_match_count == 1,
        evidence_strength=evidence_strength,
    )


DecisionRule = Callable[[PairFeatures], bool]


def decision_rule(name: str) -> DecisionRule:
    """Return one of the documented core M4 deterministic merge rules."""

    if name == "exact_key":
        return lambda f: f.phone_match or f.email_match

    if name == "both_keys":
        return lambda f: f.phone_match and f.email_match

    if name == "key_and_name":
        return lambda f: (f.phone_match or f.email_match) and f.name_exact

    if name.startswith("key_and_name_similarity_"):
        threshold = float(name.rsplit("_", 1)[1])
        return lambda f: (f.phone_match or f.email_match) and (
            f.name_similarity >= threshold
        )

    if name == "both_keys_or_key_and_name":
        return lambda f: (
            (f.phone_match and f.email_match)
            or ((f.phone_match or f.email_match) and f.name_exact)
        )

    if name.startswith("key_and_name_similarity_"):
        threshold = float(name.rsplit("_", 1)[1])
        return lambda f: (f.phone_match or f.email_match) and (
            f.name_similarity >= threshold
        )

    if name == "key_and_name_prefix":
        return lambda f: (f.phone_match or f.email_match) and f.name_prefix_match

    if name == "both_keys_or_name_prefix":
        return lambda f: (f.phone_match and f.email_match) or (
            (f.phone_match or f.email_match) and f.name_prefix_match
        )

    if name == "both_keys_or_exact_or_prefix":
        return lambda f: (f.phone_match and f.email_match) or (
            (f.phone_match or f.email_match)
            and (f.name_exact or f.name_prefix_match)
        )

    raise ValueError(
        "Unknown dedupe rule. Use one of: exact_key, both_keys, key_and_name, "
        "key_and_name_similarity_0.80, key_and_name_similarity_0.85, "
        "key_and_name_similarity_0.90, both_keys_or_key_and_name"
    )


RULE_NAMES = [
    "exact_key",
    "both_keys",
    "key_and_name",
    "key_and_name_similarity_0.50",
    "key_and_name_similarity_0.55",
    "key_and_name_similarity_0.60",
    "key_and_name_similarity_0.65",
    "key_and_name_similarity_0.70",
    "key_and_name_similarity_0.75",
    "key_and_name_similarity_0.80",
    "key_and_name_similarity_0.85",
    "key_and_name_similarity_0.90",
    "key_and_name_prefix",
    "both_keys_or_key_and_name",
    "both_keys_or_name_prefix",
    "both_keys_or_exact_or_prefix",
]


def pair_is_duplicate(features: PairFeatures, rule_name: str) -> bool:
    return bool(decision_rule(rule_name)(features))


# ---------------------------------------------------------------------------
# Review sampling
# ---------------------------------------------------------------------------


def build_candidate_evidence_frame(
    leads: pd.DataFrame,
    candidate_pairs: list[CandidatePair],
) -> pd.DataFrame:
    rows = {
        str(row["lead_id"]): row
        for _, row in leads.iterrows()
    }

    evidence = []
    for candidate in candidate_pairs:
        evidence.append(
            compare_pair(
                rows[candidate.lead_a],
                rows[candidate.lead_b],
                candidate,
            ).to_dict()
        )

    return pd.DataFrame(evidence)


def _review_stratum(row: pd.Series) -> str:
    # Primary strata are based only on the blocking evidence. Name agreement
    # is intentionally kept as a comparison feature so that the reviewer
    # sees both exact-name and name-disagreement cases within each stratum.
    if row["phone_match"] and row["email_match"]:
        return "both_keys"
    if row["phone_match"]:
        return "phone_only"
    if row["email_match"]:
        return "email_only"
    return "other"


def build_manual_review_sample(
    leads: pd.DataFrame,
    candidate_pairs: list[CandidatePair],
    sample_size: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic, balanced manual-review sheet.

    The review sample is deliberately stratified by blocking evidence rather
    than taking a random sample from the candidate list.  The three primary
    strata are:

      * both normalized phone and email match
      * phone-only match
      * email-only match

    Within each single-key stratum, name-disagreement rows are preferred so
    that the review includes the difficult shared-contact/name-variation
    cases.  If a requested stratum does not contain enough rows, its unused
    quota is redistributed deterministically to other non-empty strata.

    IMPORTANT: every row remains a candidate pair.  The blocking key is not
    treated as proof of identity; the human label is the ground truth.
    """

    if sample_size < 1:
        raise ValueError("sample_size must be > 0")

    evidence = build_candidate_evidence_frame(leads, candidate_pairs)
    if evidence.empty:
        return pd.DataFrame()

    evidence["review_stratum"] = evidence.apply(_review_stratum, axis=1)
    evidence = evidence.sort_values(
        ["review_stratum", "lead_a", "lead_b"]
    ).reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    target_strata = ["both_keys", "phone_only", "email_only"]

    # Make the intended 50/50/50 split for the default 200-row review, while
    # keeping the proportions sensible if a different sample size is used.
    base = sample_size // len(target_strata)
    remainder = sample_size % len(target_strata)
    quotas = {
        name: base + (1 if i < remainder else 0)
        for i, name in enumerate(target_strata)
    }

    selected_indices: list[int] = []
    selected_set: set[int] = set()

    def take_from(group: pd.DataFrame, n: int) -> int:
        if n <= 0 or group.empty:
            return 0
        available = group.loc[~group.index.isin(selected_set)]
        if available.empty:
            return 0
        n = min(n, len(available))
        chosen = rng.choice(available.index.to_numpy(), size=n, replace=False)
        for idx in chosen.tolist():
            idx = int(idx)
            selected_indices.append(idx)
            selected_set.add(idx)
        return n

    # First pass: fill each requested blocking stratum, preferring hard
    # name-disagreement cases in the single-key strata.
    for stratum in target_strata:
        group = evidence[evidence["review_stratum"] == stratum]
        quota = quotas[stratum]
        if stratum in {"phone_only", "email_only"}:
            hard = group[~group["name_exact"]]
            easy = group[group["name_exact"]]
            used = take_from(hard, min(quota, len(hard)))
            take_from(easy, quota - used)
        else:
            # Both-key matches are high-confidence cases; include a mix of
            # exact-name and name-variation rows when both exist.
            hard = group[~group["name_exact"]]
            easy = group[group["name_exact"]]
            hard_target = min(len(hard), quota // 2)
            used = take_from(hard, hard_target)
            take_from(easy, quota - used)

    # Redistribute any unavailable quota to remaining candidate strata.
    slots_left = min(sample_size, len(evidence)) - len(selected_indices)
    if slots_left > 0:
        remaining = evidence.loc[~evidence.index.isin(selected_set)].copy()
        # Prefer unrepresented/underrepresented primary strata first, then
        # hard name-disagreement rows, then everything else.
        remaining["_priority"] = 2
        remaining.loc[remaining["review_stratum"].isin(target_strata), "_priority"] = 1
        remaining.loc[~remaining["name_exact"], "_priority"] = 0
        remaining = remaining.sort_values(
            ["_priority", "review_stratum", "lead_a", "lead_b"]
        )
        take_from(remaining, slots_left)

    selected = evidence.loc[sorted(set(selected_indices))].copy()

    # Add human-friendly raw fields needed during review.
    lead_lookup = leads.set_index(leads["lead_id"].astype(str), drop=False)

    review_rows = []
    for _, pair in selected.iterrows():
        a = lead_lookup.loc[str(pair["lead_a"])]
        b = lead_lookup.loc[str(pair["lead_b"])]

        review_rows.append(
            {
                "lead_a": pair["lead_a"],
                "lead_b": pair["lead_b"],
                "block_reason": pair["block_reason"],
                "review_stratum": pair["review_stratum"],
                "full_name_a": a.get("full_name"),
                "full_name_b": b.get("full_name"),
                "phone_a": a.get("phone"),
                "phone_b": b.get("phone"),
                "email_a": a.get("email"),
                "email_b": b.get("email"),
                "source_a": a.get("source"),
                "source_b": b.get("source"),
                "branch_a": a.get("branch"),
                "branch_b": b.get("branch"),
                "course_a": a.get("course_interest"),
                "course_b": b.get("course_interest"),
                "phone_match": bool(pair["phone_match"]),
                "email_match": bool(pair["email_match"]),
                "name_exact": bool(pair["name_exact"]),
                "name_similarity": float(pair["name_similarity"]),
                "source_match": bool(pair["source_match"]),
                "branch_match": bool(pair["branch_match"]),
                "course_match": bool(pair["course_match"]),
                # Human must fill 1 = same person, 0 = different people.
                "same_person": "",
                "review_notes": "",
            }
        )

    return pd.DataFrame(review_rows)


def build_clusters(
    df: pd.DataFrame,
    candidate_pairs: list[CandidatePair],
    rule_name: str = "key_and_name",
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
):
    """Build duplicate clusters using union-find and a hard size cap.

    The cap prevents a chain of accepted pairwise matches from exploding into
    an implausibly large component. Rejected cap attempts are retained in the
    audit output.
    """

    if max_cluster_size < 2:
        raise ValueError("max_cluster_size must be >= 2")

    rows = {
        str(row["lead_id"]): row
        for _, row in df.iterrows()
    }
    uf = UnionFind(rows.keys())
    accepted_pairs: list[dict[str, object]] = []
    rejected_pairs: list[dict[str, object]] = []

    for candidate in candidate_pairs:
        features = compare_pair(
            rows[candidate.lead_a],
            rows[candidate.lead_b],
            candidate,
        )

        if not pair_is_duplicate(features, rule_name):
            continue

        accepted, status = uf.union_if_within_cap(
            candidate.lead_a,
            candidate.lead_b,
            max_cluster_size=max_cluster_size,
        )

        if accepted:
            accepted_pairs.append(
                {
                    **features.to_dict(),
                    "rule_name": rule_name,
                    "cluster_action": "union",
                }
            )
        elif status == "cluster_size_cap":
            rejected_pairs.append(
                {
                    **features.to_dict(),
                    "rule_name": rule_name,
                    "cluster_action": "rejected_cluster_size_cap",
                }
            )

    # Materialize every original lead ID, including singleton components.
    clusters: dict[str, list[str]] = {}
    for lead_id in rows:
        root = uf.find(lead_id)
        clusters.setdefault(root, []).append(lead_id)

    clusters = {
        root: sorted(members)
        for root, members in clusters.items()
    }

    return clusters, accepted_pairs, rejected_pairs


# ---------------------------------------------------------------------------
# Golden records
# ---------------------------------------------------------------------------


def _best_stage_for_members(
    members: list[str],
    stage_history: pd.DataFrame | None,
    stage_history_timezone: str = DEFAULT_TIMEZONE,
) -> tuple[str | None, str]:
    if stage_history is None or stage_history.empty:
        return None, "no_stage_history"

    history = stage_history.copy()
    required = {"lead_id", "stage", "changed_at"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            f"stage_history is missing required columns: {sorted(missing)}"
        )

    history["lead_id"] = history["lead_id"].astype(str)
    history = history[history["lead_id"].isin(set(members))].copy()
    if history.empty:
        return None, "no_stage_history_for_cluster"

    history["stage_norm"] = (
        history["stage"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    history["stage_rank"] = history["stage_norm"].map(STAGE_PRIORITY)

    unknown = history.loc[history["stage_rank"].isna(), "stage"].dropna().unique()
    if len(unknown):
        raise ValueError(
            "Unknown stage value(s) in stage_history.csv: "
            + ", ".join(map(str, sorted(unknown)))
            + ". Add them to STAGE_PRIORITY before producing M4 golden outcomes."
        )

    history["changed_at_parsed"] = parse_timestamp_series(
        history["changed_at"],
        default_timezone=stage_history_timezone,
    )

    best = history.sort_values(
        ["stage_rank", "changed_at_parsed"],
        ascending=[False, False],
    ).iloc[0]

    return str(best["stage"]), "stage_history"


def choose_golden_records(
    df: pd.DataFrame,
    clusters: dict[str, list[str]],
    stage_history: pd.DataFrame | None = None,
    stage_history_timezone: str = DEFAULT_TIMEZONE,
    weak_cluster_size: int = DEFAULT_WEAK_CLUSTER_SIZE,
    weak_pair_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Create a deterministic golden-record/cluster membership table.

    Rules:
      * canonical ID = earliest enquiry by created_at
      * outcome = best stage any member reached, from stage_history.csv
      * sources = union of all member sources
      * every original lead_id is retained, including singleton clusters

    Weak-cluster flags are deliberately explicit rather than silently merging
    or deleting ambiguous records.
    """

    required = {"lead_id", "created_at", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required lead columns: {sorted(missing)}")

    data = df.copy()
    data["lead_id"] = data["lead_id"].astype(str)
    data["created_at_parsed"] = parse_timestamp_series(
        data["created_at"],
        default_timezone=stage_history_timezone,
    )

    weak_pair_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in weak_pair_rows or []:
        key = tuple(sorted((str(row["lead_a"]), str(row["lead_b"]))))
        weak_pair_lookup[key] = row

    lead_lookup = data.set_index("lead_id", drop=False)
    output_rows: list[dict[str, object]] = []

    for members in sorted(clusters.values(), key=lambda values: (min(values), len(values))):
        member_rows = lead_lookup.loc[members].copy().reset_index(drop=True)
        member_rows = member_rows.sort_values(
            ["created_at_parsed", "lead_id"],
            ascending=[True, True],
        )
        canonical_id = str(member_rows.iloc[0]["lead_id"])

        best_stage, stage_source = _best_stage_for_members(
            members,
            stage_history,
            stage_history_timezone=stage_history_timezone,
        )

        sources = sorted(
            {
                str(value).strip()
                for value in member_rows["source"].tolist()
                if pd.notna(value) and str(value).strip()
            },
            key=str.casefold,
        )

        single_key_edges = 0
        cap_relevant = len(members) >= weak_cluster_size
        member_set = set(members)

        for (lead_a, lead_b), pair in weak_pair_lookup.items():
            if {lead_a, lead_b}.issubset(member_set) and pair.get("single_key_match"):
                single_key_edges += 1

        weak_reasons: list[str] = []
        if cap_relevant:
            weak_reasons.append(
                f"cluster_size_{len(members)}_meets_or_exceeds_{weak_cluster_size}"
            )
        if single_key_edges:
            weak_reasons.append(
                f"{single_key_edges}_single_key_edge(s)"
            )

        weak_cluster = bool(weak_reasons)
        cluster_id = f"C_{canonical_id}"

        for lead_id in members:
            output_rows.append(
                {
                    "cluster_id": cluster_id,
                    "lead_id": lead_id,
                    "is_canonical": lead_id == canonical_id,
                    "cluster_size": len(members),
                    "canonical_lead_id": canonical_id,
                    "best_stage": best_stage,
                    "stage_source": stage_source,
                    "sources": json.dumps(sources, ensure_ascii=False),
                    "weak_cluster": weak_cluster,
                    "weak_reason": "; ".join(weak_reasons),
                }
            )

    return pd.DataFrame(output_rows)


def save_json(data: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
