from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
    
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from leadiq.dedupe import (
    build_candidate_pairs,
    build_clusters,
    choose_golden_records,
    normalize_email,
    normalize_name,
    normalize_phone,
)
def sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lead_id": "L1",
                "created_at": "2026-04-01 10:00:00",
                "source": "website",
                "full_name": "Asha Patel",
                "phone": "+91 98765 43210",
                "email": "Asha@Example.com",
                "branch": "Surat",
                "course_interest": "AI",
            },
            {
                "lead_id": "L2",
                "created_at": "2026-04-10 10:00:00",
                "source": "instagram",
                "full_name": "Asha Patel",
                "phone": "09876543210",
                "email": "asha@example.com",
                "branch": "Surat",
                "course_interest": "AI",
            },
            {
                "lead_id": "L3",
                "created_at": "2026-04-11 10:00:00",
                "source": "website",
                "full_name": "Rahul Shah",
                "phone": "9876543210",
                "email": "rahul@example.com",
                "branch": "Surat",
                "course_interest": "AI",
            },
            {
                "lead_id": "L4",
                "created_at": "2026-04-12 10:00:00",
                "source": "website",
                "full_name": "Rohan Shah",
                "phone": "9999999999",
                "email": "rahul@example.com",
                "branch": "Surat",
                "course_interest": "AI",
            },
        ]
    )


def test_normalization():
    assert normalize_phone("+91 98765 43210") == "9876543210"
    assert normalize_phone("09123456789") == "9123456789"
    assert normalize_email(" A@Example.COM ") == "a@example.com"
    assert normalize_name("  Asha   Patel ") == "asha patel"


def test_candidates_block_only_on_phone_or_email():
    leads = sample_leads()
    leads["phone_norm"] = leads["phone"].map(normalize_phone)
    leads["email_norm"] = leads["email"].map(normalize_email)
    leads["name_norm"] = leads["full_name"].map(normalize_name)

    pairs = build_candidate_pairs(leads)
    ids = {(p.lead_a, p.lead_b) for p in pairs}

    assert ("L1", "L2") in ids
    assert ("L3", "L4") in ids


def test_key_and_name_prevents_shared_contact_merge_when_names_differ():
    leads = sample_leads()
    leads["phone_norm"] = leads["phone"].map(normalize_phone)
    leads["email_norm"] = leads["email"].map(normalize_email)
    leads["name_norm"] = leads["full_name"].map(normalize_name)

    pairs = build_candidate_pairs(leads)
    clusters, accepted, _ = build_clusters(
        leads,
        pairs,
        rule_name="key_and_name",
        max_cluster_size=10,
    )

    assert any(item["lead_a"] == "L1" and item["lead_b"] == "L2" for item in accepted)
    assert not any(
        set(["L3", "L4"]).issubset(set(members))
        for members in clusters.values()
    )


def test_cluster_size_cap_and_singletons():
    leads = pd.DataFrame(
        [
            {
                "lead_id": f"L{i}",
                "created_at": f"2026-04-{i+1:02d} 10:00:00",
                "source": "website",
                "full_name": "Same Person",
                "phone": "9876543210",
                "email": f"person{i}@example.com",
                "branch": "Surat",
                "course_interest": "AI",
            }
            for i in range(4)
        ]
    )
    leads["phone_norm"] = leads["phone"].map(normalize_phone)
    leads["email_norm"] = leads["email"].map(normalize_email)
    leads["name_norm"] = leads["full_name"].map(normalize_name)

    pairs = build_candidate_pairs(leads)
    clusters, accepted, rejected = build_clusters(
        leads,
        pairs,
        rule_name="key_and_name",
        max_cluster_size=2,
    )

    assert len(rejected) > 0
    assert max(len(members) for members in clusters.values()) <= 2
    assert sum(len(members) for members in clusters.values()) == len(leads)


def test_golden_record_uses_earliest_and_stage_history():
    leads = sample_leads()
    leads["phone_norm"] = leads["phone"].map(normalize_phone)
    leads["email_norm"] = leads["email"].map(normalize_email)
    leads["name_norm"] = leads["full_name"].map(normalize_name)

    pairs = build_candidate_pairs(leads)
    clusters, accepted, _ = build_clusters(
        leads,
        pairs,
        rule_name="key_and_name",
    )

    stage_history = pd.DataFrame(
        [
            {"lead_id": "L1", "stage": "New", "changed_at": "2026-04-01 11:00:00+05:30", "changed_by": "system"},
            {"lead_id": "L2", "stage": "Admission Done", "changed_at": "2026-04-20 11:00:00+05:30", "changed_by": "C01"},
        ]
    )

    output = choose_golden_records(
        leads,
        clusters,
        stage_history=stage_history,
        weak_pair_rows=accepted,
    )

    l1 = output[output["lead_id"] == "L1"].iloc[0]
    l2 = output[output["lead_id"] == "L2"].iloc[0]

    assert bool(l1["is_canonical"]) is True
    assert bool(l2["is_canonical"]) is False
    assert l1["canonical_lead_id"] == "L1"
    assert l1["best_stage"] == "Admission Done"
    assert set(__import__("json").loads(l1["sources"])) == {"website", "instagram"}
    assert len(output) == len(leads)


def test_manual_review_sample_is_stratified_by_blocking_keys():
    leads = pd.DataFrame(
        [
            {
                "lead_id": f"B{i}",
                "created_at": f"2026-04-{i+1:02d} 10:00:00",
                "source": "website",
                "full_name": f"Person {i}",
                "phone": "9876543210" if i < 4 else f"98{i:08d}",
                "email": f"person{i}@example.com" if i != 4 else "person0@example.com",
                "branch": "Surat",
                "course_interest": "AI",
            }
            for i in range(5)
        ]
    )
    leads["phone_norm"] = leads["phone"].map(normalize_phone)
    leads["email_norm"] = leads["email"].map(normalize_email)
    leads["name_norm"] = leads["full_name"].map(normalize_name)

    from leadiq.dedupe import build_manual_review_sample

    pairs = build_candidate_pairs(leads)
    review = build_manual_review_sample(leads, pairs, sample_size=4, random_seed=42)

    assert len(review) == 4
    assert set(review["review_stratum"]).issubset({"both_keys", "phone_only", "email_only"})
    assert review["same_person"].eq("").all()
    assert review["review_notes"].eq("").all()
