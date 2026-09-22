from __future__ import annotations

import re

import pandas as pd


WEEKDAYS_EN = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

WEEKDAYS_HI = {
    0: "Somvaar",
    1: "Mangalvaar",
    2: "Budhvaar",
    3: "Guruvaar",
    4: "Shukravaar",
    5: "Shanivaar",
    6: "Ravivaar",
}

KEYWORD_REASON = {
    "kw_fee": (
        "Asked about fees",
        "Fees ke baare mein poocha",
    ),
    "kw_emi": (
        "Asked about EMI",
        "EMI ke baare mein poocha",
    ),
    "kw_placement": (
        "Asked about placements or jobs",
        "Placement ya job ke baare mein poocha",
    ),
    "kw_timing": (
        "Asked about timing or batches",
        "Timing ya batch ke baare mein poocha",
    ),
    "kw_visit": (
        "Asked about a branch or visit",
        "Branch ya visit ke baare mein poocha",
    ),
    "kw_demo": (
        "Asked about a demo or trial",
        "Demo ya trial ke baare mein poocha",
    ),
    "kw_price_objection": (
        "Raised a price or budget concern",
        "Price ya budget ko lekar concern dikhaya",
    ),
    "kw_not_interested": (
        "Showed a not-interested or cancellation signal",
        "Not interested ya cancellation ka signal diya",
    ),
}


def _strip_prefix(feature_name: str) -> str:
    return re.sub(r"^[^_]+__", "", feature_name)


def _split_one_hot(feature_name: str) -> tuple[str, str] | None:
    name = _strip_prefix(feature_name)

    for base in [
        "source",
        "branch",
        "course",
        "education_status",
    ]:
        prefix = f"{base}_"
        if name.startswith(prefix):
            return base, name[len(prefix):]

    return None


def _is_active(row: pd.Series, feature_name: str) -> bool:
    value = row.get(feature_name)

    if value is None or pd.isna(value):
        return False

    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return bool(value)


def reason_for_feature(
    feature_name: str,
    row: pd.Series,
) -> tuple[str, str] | None:
    name = _strip_prefix(feature_name)

    # ---------------------------------------------------------
    # One-hot categorical features.
    # Only explain the category if it is actually this lead's
    # category. This prevents SHAP selecting an inactive one-hot.
    # ---------------------------------------------------------
    one_hot = _split_one_hot(feature_name)

    if one_hot:
        base, value = one_hot
        actual_value = row.get(base)

        if actual_value is None or pd.isna(actual_value):
            return None

        if str(actual_value) != value:
            return None

        if base == "source":
            return (
                f"Lead came from {value}",
                f"Lead {value} se aaya hai",
            )

        if base == "branch":
            return (
                f"Enquiry is for the {value} branch",
                f"Enquiry {value} branch ke liye hai",
            )

        if base == "course":
            return (
                f"Enquiry is for {value}",
                f"Enquiry {value} course ke liye hai",
            )

        if base == "education_status":
            return (
                f"Education status is {value}",
                f"Education status {value} hai",
            )

    # ---------------------------------------------------------
    # Keyword flags
    # ---------------------------------------------------------
    if name in KEYWORD_REASON:
        if not _is_active(row, name):
            return None
        return KEYWORD_REASON[name]

    # ---------------------------------------------------------
    # Basic lead attributes
    # ---------------------------------------------------------
    if name == "age":
        value = row.get("age")

        if value is None or pd.isna(value):
            return None

        return (
            f"Lead age is {int(value)}",
            f"Lead ki age {int(value)} hai",
        )

    if name == "creation_hour":
        value = row.get(name)

        if value is None or pd.isna(value):
            return None

        return (
            f"Enquiry arrived around {int(value):02d}:00",
            f"Enquiry lagbhag {int(value):02d}:00 baje aayi",
        )

    if name == "creation_weekday":
        value = row.get(name)

        if value is None or pd.isna(value):
            return None

        day_int = int(value)

        return (
            f"Enquiry arrived on {WEEKDAYS_EN.get(day_int, 'a weekday')}",
            f"Enquiry {WEEKDAYS_HI.get(day_int, 'ek din')} ko aayi",
        )

    # ---------------------------------------------------------
    # Messaging
    # ---------------------------------------------------------
    if name == "inbound_msg_count_before_t":
        value = int(row.get(name, 0))

        return (
            f"Person sent {value} WhatsApp message(s) in the first 24 hours",
            f"Person ne pehle 24 ghanton mein {value} WhatsApp message(s) bheje",
        )

    if name == "outbound_msg_count_before_t":
        value = int(row.get(name, 0))

        return (
            f"{value} outbound WhatsApp message(s) were sent in the first 24 hours",
            f"Pehle 24 ghanton mein {value} outbound WhatsApp message(s) bheje gaye",
        )

    if name == "mean_msg_len_before_t":
        value = row.get(name)

        if value is None or pd.isna(value):
            return None

        return (
            f"Average message length was about {float(value):.0f} characters",
            f"Average message length lagbhag {float(value):.0f} characters tha",
        )

    if name == "mins_to_first_lead_reply_after_assistant":
        value = float(row.get(name, -1.0))

        if value < 0:
            return (
                "No lead reply followed the assistant within 24 hours",
                "24 ghanton mein assistant ke baad lead ka reply nahi aaya",
            )

        return (
            f"Replied to the assistant within about {value:.0f} minutes",
            f"Assistant ko lagbhag {value:.0f} minute mein reply kiya",
        )

    if name == "has_inbound_text_before_t":
        if not _is_active(row, name):
            return None

        return (
            "Person sent inbound WhatsApp text in the first 24 hours",
            "Person ne pehle 24 ghanton mein WhatsApp text bheja",
        )

    # ---------------------------------------------------------
    # Calls
    # ---------------------------------------------------------
    if name == "calls_attempted_before_t":
        value = int(row.get(name, 0))

        return (
            f"{value} counselor call attempt(s) happened in the first 24 hours",
            f"Pehle 24 ghanton mein {value} counselor call attempt(s) hue",
        )

    if name == "calls_answered_before_t":
        value = int(row.get(name, 0))

        return (
            f"{value} counselor call(s) were answered in the first 24 hours",
            f"Pehle 24 ghanton mein {value} counselor call(s) answer hue",
        )

    if name == "mins_to_first_counselor_call":
        value = float(row.get(name, -1.0))

        if value < 0:
            return (
                "No counselor call occurred in the first 24 hours",
                "Pehle 24 ghanton mein counselor call nahi hua",
            )

        return (
            f"First counselor call was about {value:.0f} minutes after enquiry",
            f"Pehla counselor call enquiry ke lagbhag {value:.0f} minute baad hua",
        )

    # ---------------------------------------------------------
    # Locality
    # ---------------------------------------------------------
    if name == "min_dist_to_branch":
        value = row.get(name)

        if value is None or pd.isna(value):
            return None

        return (
            f"Nearest branch is about {float(value):.1f} km away",
            f"Nearest branch lagbhag {float(value):.1f} km door hai",
        )

    if name == "locality_match_missing":
        if _is_active(row, name):
            return (
                "Home locality did not match the locality table",
                "Home locality locality table mein match nahi hui",
            )
        return None

    # ---------------------------------------------------------
    # Missingness / availability flags
    # ---------------------------------------------------------
    if name == "has_phone":
        if not _is_active(row, name):
            return None

        return (
            "Phone number is available",
            "Phone number available hai",
        )

    if name == "has_email":
        if not _is_active(row, name):
            return None

        return (
            "Email address is available",
            "Email address available hai",
        )

    if name == "has_home_locality":
        if not _is_active(row, name):
            return None

        return (
            "Home locality is available",
            "Home locality available hai",
        )

    if name == "age_missing":
        if not _is_active(row, name):
            return None

        return (
            "Age is missing from the enquiry",
            "Enquiry mein age missing hai",
        )

    if name == "education_status_missing":
        if not _is_active(row, name):
            return None

        return (
            "Education status is missing",
            "Education status missing hai",
        )

    return None