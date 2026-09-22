from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from leadiq.model_data import load_m2_dataset
from dotenv import load_dotenv
load_dotenv()

def main() -> None:

    if not os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    dataset = load_m2_dataset(
        leads_path="data/leads.csv",
        messages_path="data/messages.csv",
        calls_path="data/calls.csv",
        stages_path="data/stage_history.csv",
        localities_path="data/localities.csv",
        export_date="2026-09-15 23:59:00+05:30",
    )

    y = dataset.y

    rows = [
        (
            str(lead_id),
            int(label),
        )
        for lead_id, label in y.items()
    ]

    print(
        f"Prepared {len(rows):,} mature labelled outcomes."
    )

    sql = """
        INSERT INTO lead_outcomes (
            lead_id,
            outcome
        )
        VALUES (%s, %s)
        ON CONFLICT (lead_id)
        DO UPDATE SET
            outcome = EXCLUDED.outcome;
    """

    with psycopg.connect(
        os.environ["DATABASE_URL"]
    ) as connection:

        with connection.cursor() as cursor:

            cursor.executemany(
                sql,
                rows,
            )

        connection.commit()

    print(
        "lead_outcomes populated successfully."
    )


if __name__ == "__main__":
    main()