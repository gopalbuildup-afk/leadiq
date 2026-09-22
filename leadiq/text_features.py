from __future__ import annotations

import re

import pandas as pd


MAX_TEXT_TOKENS = 512


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

# Indian 10-digit mobile numbers, optionally written with +91.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
)


def _mask_pii(text: str) -> str:
    """
    Preserve semantic information while preventing raw phone/email
    values from reaching the embedding model.

    Existing [PHONE] / [EMAIL] placeholders remain unchanged.
    """
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    return text


def _truncate_to_tokens(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TEXT_TOKENS,
    mode: str = "head",
) -> str:
    if mode not in {"head", "tail"}:
        raise ValueError(
            f"Unsupported truncation mode: {mode}. "
            "Expected 'head' or 'tail'."
        )

    if mode == "head":
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )

        token_ids = encoded["input_ids"]

    else:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        token_ids = encoded["input_ids"][-max_tokens:]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_lead_texts(
    leads: pd.DataFrame,
    messages: pd.DataFrame,
    tokenizer,
    truncation_mode: str = "head",
) -> pd.DataFrame:
    """
    Build the M3 text representation.

    Rules:
        - only the person's own inbound messages
        - created_at <= sent_at < created_at + 24h
        - messages ordered chronologically
        - joined into one enquiry text
        - phone/email values masked
        - truncated to 512 tokens
        - no-text enquiries get only a presence flag
    """

    lead_rows = leads[
        [
            "lead_id",
            "created_at",
        ]
    ].copy()

    lead_rows["created_at"] = pd.to_datetime(
        lead_rows["created_at"],
        utc=True,
    )

    lead_rows["scoring_time"] = (
        lead_rows["created_at"]
        + pd.Timedelta(hours=24)
    )

    msg = messages[
        [
            "lead_id",
            "sent_at",
            "direction",
            "sender_type",
            "text",
        ]
    ].copy()

    msg["sent_at"] = pd.to_datetime(
        msg["sent_at"],
        utc=True,
    )

    # Only the person's own inbound messages.
    msg = msg[
        (msg["direction"] == "inbound")
        & (msg["sender_type"] == "lead")
    ].copy()

    joined = msg.merge(
        lead_rows[
            [
                "lead_id",
                "created_at",
                "scoring_time",
            ]
        ],
        on="lead_id",
        how="inner",
    )

    # Exact point-in-time boundary:
    #
    # created_at <= sent_at < t
    #
    # Anything before creation is invalid.
    # Anything at/after the scoring moment is invalid.
    joined = joined[
        (joined["sent_at"] >= joined["created_at"])
        & (joined["sent_at"] < joined["scoring_time"])
    ].copy()

    joined["text"] = (
        joined["text"]
        .fillna("")
        .astype(str)
        .map(_mask_pii)
    )

    joined = joined.sort_values(
        [
            "lead_id",
            "sent_at",
        ]
    )

    grouped: dict[str, list[str]] = {}

    for lead_id, frame in joined.groupby("lead_id"):
        parts = [
            text.strip()
            for text in frame["text"].tolist()
            if text.strip()
        ]

        if parts:
            grouped[lead_id] = parts

    rows = []

    for lead_id in lead_rows["lead_id"]:
        parts = grouped.get(lead_id, [])

        if not parts:
            rows.append(
                {
                    "lead_id": lead_id,
                    "text": "",
                    "has_inbound_text_before_t": 0,
                }
            )
            continue

        joined_text = "\n".join(parts)

        truncated = _truncate_to_tokens(
            joined_text,
            tokenizer=tokenizer,
            max_tokens=MAX_TEXT_TOKENS,
            mode=truncation_mode,
        )

        rows.append(
            {
                "lead_id": lead_id,
                "text": truncated,
                "has_inbound_text_before_t": 1,
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("lead_id")
    )