from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import psycopg
from dotenv import load_dotenv
import pandas as pd
load_dotenv()  

TABLE_NAME = "lead_embeddings"
EMBEDDING_DIM = 384


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set."
        )

    return database_url


def _vector_literal(
    vector: np.ndarray,
) -> str:

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError(
            f"Expected 1-D vector, "
            f"got {vector.shape}"
        )

    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dimensional "
            f"vector, got {len(vector)}"
        )

    return (
        "["
        + ",".join(
            f"{float(value):.8f}"
            for value in vector
        )
        + "]"
    )


def verify_lead_embeddings_table() -> None:

    with psycopg.connect(
        _database_url()
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                """,
                (TABLE_NAME,),
            )

            existing = {
                row[0]
                for row in cur.fetchall()
            }

            required = {
                "lead_id",
                "asof",
                "model_version",
                "embedding",
            }

            missing = required - existing

            if missing:
                raise RuntimeError(
                    "lead_embeddings is missing columns: "
                    f"{sorted(missing)}"
                )

            cur.execute(
                """
                SELECT format_type(
                    a.atttypid,
                    a.atttypmod
                )
                FROM pg_attribute a
                JOIN pg_class c
                    ON c.oid = a.attrelid
                JOIN pg_namespace n
                    ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relname = %s
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """,
                (TABLE_NAME,),
            )

            row = cur.fetchone()

            if row is None:
                raise RuntimeError(
                    "Unable to inspect embedding column."
                )

            if row[0] != "vector(384)":
                raise RuntimeError(
                    "lead_embeddings.embedding must be "
                    f"vector(384), got {row[0]}"
                )


def store_lead_embeddings(
    lead_ids: Iterable[str],
    embeddings: np.ndarray,
    asof_times,
    has_text,
    model_version: str,
) -> int:
    """
    Persist only enquiries that actually have inbound text.

    No-text enquiries are represented by the model's explicit
    has_inbound_text_before_t feature and do not receive a fake
    zero vector in pgvector.
    """

    verify_lead_embeddings_table()

    lead_ids = list(lead_ids)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    has_text = np.asarray(
        has_text,
        dtype=bool,
    )

    asof_times = asof_times.reindex(
        lead_ids
    )

    if len(lead_ids) != len(embeddings):
        raise ValueError(
            "lead_ids and embeddings length mismatch."
        )

    if len(lead_ids) != len(has_text):
        raise ValueError(
            "lead_ids and has_text length mismatch."
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    active_positions = np.flatnonzero(
        has_text
    )

    rows = []

    for position in active_positions:

        lead_id = lead_ids[position]
        vector = embeddings[position]
        asof = asof_times.iloc[position]

        if pd.isna(asof):
            raise ValueError(
                f"Missing asof for lead {lead_id}"
            )

        rows.append(
            (
                str(lead_id),
                _vector_literal(vector),
                asof.to_pydatetime(),
                model_version,
            )
        )

    if not rows:
        return 0

    with psycopg.connect(
        _database_url()
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS
                lead_embeddings_hnsw_idx
                ON public.lead_embeddings
                USING hnsw (
                    embedding vector_cosine_ops
                )
                """
            )

            insert_sql = """
                INSERT INTO public.lead_embeddings
                    (
                        lead_id,
                        embedding,
                        asof,
                        model_version
                    )
                VALUES
                    (
                        %s,
                        %s::vector,
                        %s,
                        %s
                    )
                ON CONFLICT (
                    lead_id,
                    asof,
                    model_version
                )
                DO UPDATE SET
                    embedding = EXCLUDED.embedding
            """

            cur.executemany(
                insert_sql,
                rows,
            )

        conn.commit()

    return len(rows)