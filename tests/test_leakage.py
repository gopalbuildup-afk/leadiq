from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from leadiq.contract import (
    TARGET_STAGE,
    TIMEZONE,
    compute_target_and_maturity,
)
from leadiq.features import (
    FORBIDDEN_COLUMNS,
    build_feature_matrix,
)


# ------------------------------------------------------------------------------
# Test fixture
# ------------------------------------------------------------------------------

@pytest.fixture
def base_sample_data():
    """
    Synthetic fixture that follows the real source CSV column names.
    """
    created_at = pd.Timestamp(
        "2026-03-01 10:00:00",
        tz=TIMEZONE,
    )

    lead_row = {
        "lead_id": "L10001",
        "created_at": created_at,
        "source": "Website Form",
        "branch": "Riverside",
        "course_interest": "Data Science & AI",
        "full_name": "Test User",
        "phone": "5000000001",
        "email": "test@example.com",
        "education_status": "Graduate",
        "age": 25,
        "home_locality": "Green Park",
        "assigned_counselor_id": "C01",
        "current_stage": "New",
        "updated_at": created_at + pd.Timedelta(hours=5),
        "last_contacted_at": (
            created_at + pd.Timedelta(hours=3)
        ),
        "legacy_score": 50,
        "legacy_score_version": "v1",
    }

    leads_df = pd.DataFrame([lead_row])

    # Assistant sends first message at +2h.
    assistant_time = (
        created_at
        + pd.Timedelta(hours=2)
    )

    # Lead replies seven minutes later.
    lead_reply_time = (
        assistant_time
        + pd.Timedelta(minutes=7)
    )

    # Counselor call at +3h.
    call_time = (
        created_at
        + pd.Timedelta(hours=3)
    )

    messages_df = pd.DataFrame(
        [
            {
                "message_id": "M001",
                "lead_id": "L10001",
                "sent_at": assistant_time.tz_convert("UTC"),
                "direction": "outbound",
                "sender_type": "assistant",
                "text": "Hi, how can I help?",
            },
            {
                "message_id": "M002",
                "lead_id": "L10001",
                "sent_at": lead_reply_time.tz_convert("UTC"),
                "direction": "inbound",
                "sender_type": "lead",
                "text": "What is the fee and EMI structure?",
            },
        ]
    )

    calls_df = pd.DataFrame(
        [
            {
                "call_id": "C001",
                "lead_id": "L10001",
                "counselor_id": "C01",
                "started_at_ms": int(
                    call_time.tz_convert("UTC").timestamp()
                    * 1000
                ),
                "duration_sec": 120,
                "outcome": "answered",
                "notes": "Discussed timing",
            }
        ]
    )

    localities_df = pd.DataFrame(
        [
            {
                "locality": "Green Park",
                "km_to_riverside": 2.1,
                "km_to_central": 6.5,
                "km_to_lakeview": 9.8,
            }
        ]
    )

    return (
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    )


# ------------------------------------------------------------------------------
# Required M1 leakage tests
# ------------------------------------------------------------------------------

def test_no_future_leakage(base_sample_data):
    """
    Required M1 test:
    adding arbitrary messages/calls/stage changes after t
    must not alter the feature vector.
    """
    (
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    ) = base_sample_data

    base_features = build_feature_matrix(
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    )

    created_at = leads_df[
        "created_at"
    ].iloc[0]

    t_moment = (
        created_at
        + pd.Timedelta(hours=24)
    )

    # Future message after t.
    future_message = pd.DataFrame(
        [
            {
                "message_id": "M999_FUTURE",
                "lead_id": "L10001",
                "sent_at": (
                    t_moment
                    + pd.Timedelta(hours=2)
                ).tz_convert("UTC"),
                "direction": "inbound",
                "sender_type": "lead",
                "text": "I want a demo class tomorrow.",
            }
        ]
    )

    # Future call after t.
    future_call_time = (
        t_moment
        + pd.Timedelta(hours=5)
    )

    future_call = pd.DataFrame(
        [
            {
                "call_id": "C999_FUTURE",
                "lead_id": "L10001",
                "counselor_id": "C01",
                "started_at_ms": int(
                    future_call_time
                    .tz_convert("UTC")
                    .timestamp()
                    * 1000
                ),
                "duration_sec": 300,
                "outcome": "answered",
                "notes": "Followup after 24h",
            }
        ]
    )

    # Future stage change.
    # Stage history is deliberately not accepted by the feature builder,
    # because stage changes are not allowed as model features.
    future_stage_history = pd.DataFrame(
        [
            {
                "lead_id": "L10001",
                "stage": TARGET_STAGE,
                "changed_at": (
                    t_moment
                    + pd.Timedelta(hours=6)
                ),
                "changed_by": "C01",
            }
        ]
    )

    assert (
        future_stage_history["changed_at"].iloc[0]
        > t_moment
    )

    dirty_messages = pd.concat(
        [
            messages_df,
            future_message,
        ],
        ignore_index=True,
    )

    dirty_calls = pd.concat(
        [
            calls_df,
            future_call,
        ],
        ignore_index=True,
    )

    post_future_features = build_feature_matrix(
        leads_df,
        dirty_messages,
        dirty_calls,
        localities_df,
    )

    assert_frame_equal(
        base_features,
        post_future_features,
    )


def test_forbidden_columns(base_sample_data):
    """
    Required M1 test:
    forbidden export-time/leakage columns must never
    appear in the final feature matrix.
    """
    (
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    ) = base_sample_data

    clean_features = build_feature_matrix(
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    )

    leaky_leads = leads_df.copy()

    # Deliberately inject obviously predictive future values.
    leaky_leads["current_stage"] = "Admission Done"
    leaky_leads["legacy_score"] = 100
    leaky_leads["legacy_score_version"] = "v2"
    leaky_leads["updated_at"] = (
        leaky_leads["created_at"]
        + pd.Timedelta(days=10)
    )
    leaky_leads["last_contacted_at"] = (
        leaky_leads["created_at"]
        + pd.Timedelta(days=9)
    )

    leaky_features = build_feature_matrix(
        leaky_leads,
        messages_df,
        calls_df,
        localities_df,
    )

    # Changing forbidden fields must not alter features.
    assert_frame_equal(
        clean_features,
        leaky_features,
    )

    for column in FORBIDDEN_COLUMNS:
        assert column not in clean_features.columns
        assert column not in leaky_features.columns


def test_timezone(base_sample_data):
    """
    Required M1 test:
    With created_at in IST (naive) and a message in UTC, a message sent
    one hour before t counts, and one sent one hour after does not.
    """
    leads_df, _, calls_df, localities_df = base_sample_data

    # created_at at 10:00 IST -> t = next day 10:00 IST
    created_ist = pd.Timestamp("2026-03-01 10:00:00", tz=TIMEZONE)
    t_utc = (created_ist + pd.Timedelta(hours=24)).tz_convert("UTC")

    leads_test = leads_df.copy()
    leads_test["created_at"] = "2026-03-01 10:00:00"  # Naive IST string

    messages = pd.DataFrame([
        {
            "message_id": "M_BEFORE",
            "lead_id": "L10001",
            "sent_at": (t_utc - pd.Timedelta(hours=1)).isoformat(),  # 1h before t
            "direction": "inbound",
            "sender_type": "lead",
            "text": "Can I get details on fees?",
        },
        {
            "message_id": "M_AFTER",
            "lead_id": "L10001",
            "sent_at": (t_utc + pd.Timedelta(hours=1)).isoformat(),  # 1h after t
            "direction": "inbound",
            "sender_type": "lead",
            "text": "Following up after deadline",
        },
    ])

    features = build_feature_matrix(
        leads_test, messages, calls_df, localities_df
    )

    # Only M_BEFORE should be counted
    assert features.loc["L10001", "inbound_msg_count_before_t"] == 1
    assert features.loc["L10001", "kw_fee"] == 1


# ------------------------------------------------------------------------------
# Additional M1 correctness tests
# ------------------------------------------------------------------------------

# def test_exact_scoring_moment_boundary(base_sample_data):
#     (
#         leads_df,
#         _,
#         _,
#         _,
#     ) = base_sample_data

#     created_at = leads_df[
#         "created_at"
#     ].iloc[0]

#     expected_t = (
#         created_at
#         + pd.Timedelta(hours=24)
#     )

#     assert expected_t - created_at == pd.Timedelta(
#         hours=24
#     )


def test_first_lead_reply_latency(base_sample_data):
    (
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    ) = base_sample_data

    features = build_feature_matrix(
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    )

    assert (
        features[
            "mins_to_first_lead_reply_after_assistant"
        ].iloc[0]
        == pytest.approx(7.0)
    )


def test_feature_count_at_least_15(base_sample_data):
    (
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    ) = base_sample_data

    features = build_feature_matrix(
        leads_df,
        messages_df,
        calls_df,
        localities_df,
    )

    assert features.shape[1] >= 15


def test_target_definition_is_admission_done_only():
    created_at = pd.Timestamp(
        "2026-03-01 10:00:00",
        tz=TIMEZONE,
    )

    leads_df = pd.DataFrame(
        [
            {
                "lead_id": "L1",
                "created_at": created_at,
            },
            {
                "lead_id": "L2",
                "created_at": created_at,
            },
            {
                "lead_id": "L3",
                "created_at": created_at,
            },
        ]
    )

    stage_history_df = pd.DataFrame(
        [
            # L1: correct target -> 1.
            {
                "lead_id": "L1",
                "stage": "Admission Done",
                "changed_at": (
                    created_at
                    + pd.Timedelta(days=20)
                ),
                "changed_by": "C01",
            },
            # L2: another tempting stage, but NOT the target.
            {
                "lead_id": "L2",
                "stage": "Admitted",
                "changed_at": (
                    created_at
                    + pd.Timedelta(days=10)
                ),
                "changed_by": "C01",
            },
            # L3: Admission Done outside the 30-day window -> 0.
            {
                "lead_id": "L3",
                "stage": "Admission Done",
                "changed_at": (
                    created_at
                    + pd.Timedelta(days=31)
                ),
                "changed_by": "C01",
            },
        ]
    )

    result = compute_target_and_maturity(
        leads_df=leads_df,
        stage_history_df=stage_history_df,
        export_date=(
            created_at
            + pd.Timedelta(days=60)
        ),
        only_mature=False,
    )

    target_by_lead = (
        result
        .set_index("lead_id")["target"]
        .to_dict()
    )

    assert target_by_lead["L1"] == 1
    assert target_by_lead["L2"] == 0
    assert target_by_lead["L3"] == 0


def test_target_maturity_excludes_recent_enquiries():
    export_date = pd.Timestamp(
        "2026-09-15 23:59:00",
        tz=TIMEZONE,
    )

    mature_created_at = (
        export_date
        - pd.Timedelta(days=31)
    )

    immature_created_at = (
        export_date
        - pd.Timedelta(days=29)
    )

    leads_df = pd.DataFrame(
        [
            {
                "lead_id": "MATURE",
                "created_at": mature_created_at,
            },
            {
                "lead_id": "IMMATURE",
                "created_at": immature_created_at,
            },
        ]
    )

    stage_history_df = pd.DataFrame(
        columns=[
            "lead_id",
            "stage",
            "changed_at",
            "changed_by",
        ]
    )

    result = compute_target_and_maturity(
        leads_df=leads_df,
        stage_history_df=stage_history_df,
        export_date=export_date,
        only_mature=True,
    )

    assert result["lead_id"].tolist() == [
        "MATURE"
    ]