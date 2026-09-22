from __future__ import annotations

import numpy as np
import pandas as pd

from leadiq.m3 import build_raw_embeddings
from leadiq.text_features import build_lead_texts


def test_future_messages_do_not_enter_text():
    leads = pd.DataFrame(
        {
            "lead_id": ["L1"],
            "created_at": [
                pd.Timestamp(
                    "2026-03-01 10:00:00",
                    tz="Asia/Kolkata",
                )
            ],
        }
    )

    messages = pd.DataFrame(
        {
            "lead_id": ["L1", "L1", "L1"],
            "sent_at": [
                "2026-03-01T04:00:00Z",
                "2026-03-02T03:00:00Z",
                "2026-03-02T05:00:00Z",
            ],
            "direction": [
                "inbound",
                "inbound",
                "inbound",
            ],
            "sender_type": [
                "lead",
                "lead",
                "lead",
            ],
            "text": [
                "before cutoff",
                "just before cutoff",
                "after cutoff",
            ],
        }
    )

    class FakeTokenizer:
        def __call__(
            self,
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        ):
            return {
                "input_ids": text.split()[:max_length]
            }

        def decode(
            self,
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            return " ".join(ids)

    result = build_lead_texts(
        leads=leads,
        messages=messages,
        tokenizer=FakeTokenizer(),
    )

    text = result.loc["L1", "text"]

    assert "before cutoff" in text
    assert "just before cutoff" in text
    assert "after cutoff" not in text


def test_only_lead_inbound_messages_are_used():
    leads = pd.DataFrame(
        {
            "lead_id": ["L1"],
            "created_at": [
                pd.Timestamp(
                    "2026-03-01 10:00:00",
                    tz="Asia/Kolkata",
                )
            ],
        }
    )

    messages = pd.DataFrame(
        {
            "lead_id": ["L1", "L1", "L1"],
            "sent_at": [
                "2026-03-01T05:00:00Z",
                "2026-03-01T06:00:00Z",
                "2026-03-01T07:00:00Z",
            ],
            "direction": [
                "inbound",
                "outbound",
                "inbound",
            ],
            "sender_type": [
                "lead",
                "assistant",
                "counselor",
            ],
            "text": [
                "keep this",
                "do not use this",
                "do not use this",
            ],
        }
    )

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": text.split()}

        def decode(self, ids, **kwargs):
            return " ".join(ids)

    result = build_lead_texts(
        leads,
        messages,
        FakeTokenizer(),
    )

    text = result.loc["L1", "text"]

    assert text == "keep this"

def test_messages_before_created_at_do_not_enter_text():
    import pandas as pd

    from leadiq.text_features import build_lead_texts

    leads = pd.DataFrame(
        {
            "lead_id": ["L1"],
            "created_at": [
                pd.Timestamp(
                    "2026-03-01 10:00:00",
                    tz="Asia/Kolkata",
                )
            ],
        }
    )

    messages = pd.DataFrame(
        {
            "lead_id": ["L1", "L1", "L1"],
            "sent_at": [
                "2026-03-01T03:30:00Z",  # 09:00 IST: before creation
                "2026-03-01T05:00:00Z",  # 10:30 IST: valid
                "2026-03-02T05:00:00Z",  # 10:30 IST next day: after t
            ],
            "direction": [
                "inbound",
                "inbound",
                "inbound",
            ],
            "sender_type": [
                "lead",
                "lead",
                "lead",
            ],
            "text": [
                "must not be used",
                "valid message",
                "after cutoff",
            ],
        }
    )

    class FakeTokenizer:
        def __call__(
            self,
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        ):
            return {
                "input_ids": text.split()[:max_length]
            }

        def decode(
            self,
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            return " ".join(ids)

    result = build_lead_texts(
        leads=leads,
        messages=messages,
        tokenizer=FakeTokenizer(),
    )

    assert result.loc["L1", "text"] == "valid message"
    
def test_no_text_gets_flag_not_text():
    leads = pd.DataFrame(
        {
            "lead_id": ["L1"],
            "created_at": [
                pd.Timestamp(
                    "2026-03-01 10:00:00",
                    tz="Asia/Kolkata",
                )
            ],
        }
    )

    messages = pd.DataFrame(
        columns=[
            "lead_id",
            "sent_at",
            "direction",
            "sender_type",
            "text",
        ]
    )

    class FakeTokenizer:
        pass

    result = build_lead_texts(
        leads,
        messages,
        FakeTokenizer(),
    )

    assert result.loc["L1", "text"] == ""
    assert result.loc["L1", "has_inbound_text_before_t"] == 0