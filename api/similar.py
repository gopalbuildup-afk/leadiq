from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import psycopg


MODEL_VERSION = "m3-1.0.0"


class SimilarLeadError(Exception):
    """Base error for similar-lead operations."""


class DatabaseConfigurationError(SimilarLeadError):
    """Raised when DATABASE_URL is missing."""


class SourceEmbeddingNotFoundError(SimilarLeadError):
    """Raised when the requested lead has no stored embedding."""


@dataclass(frozen=True)
class SimilarLead:
    lead_id: str
    similarity: float
    outcome: int


@dataclass(frozen=True)
class SimilarLeadResult:
    lead_id: str
    model_version: str
    k: int
    similar: list[SimilarLead]

    @property
    def admissions(self) -> int:
        return sum(
            1
            for item in self.similar
            if item.outcome == 1
        )

    @property
    def total(self) -> int:
        return len(self.similar)


def _get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise DatabaseConfigurationError(
            "DATABASE_URL environment variable is not configured."
        )

    return database_url


def _connect() -> psycopg.Connection[Any]:
    """
    Create a short-lived database connection.

    This works both locally and with a hosted PostgreSQL/Neon database.
    For local development, DATABASE_URL can be loaded from .env.
    """

    return psycopg.connect(
        _get_database_url()
    )


def get_similar_leads(
    lead_id: str,
    k: int = 10,
) -> SimilarLeadResult:

    if not lead_id or not lead_id.strip():
        raise ValueError(
            "lead_id must not be empty."
        )

    if not 1 <= k <= 50:
        raise ValueError(
            "k must be between 1 and 50."
        )

    lead_id = lead_id.strip()

    query = """
        WITH source_embedding AS (
            SELECT
                embedding
            FROM lead_embeddings
            WHERE lead_id = %s
              AND model_version = %s
            ORDER BY asof DESC
            LIMIT 1
        )
        SELECT
            e.lead_id,
            1 - (e.embedding <=> source_embedding.embedding)
                AS similarity,
            o.outcome
        FROM lead_embeddings AS e
        CROSS JOIN source_embedding
        INNER JOIN lead_outcomes AS o
            ON o.lead_id = e.lead_id
        WHERE e.model_version = %s
          AND e.lead_id <> %s
        ORDER BY e.embedding <=> source_embedding.embedding
        LIMIT %s;
    """

    with _connect() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    lead_id,
                    MODEL_VERSION,
                    MODEL_VERSION,
                    lead_id,
                    k,
                ),
            )

            rows = cursor.fetchall()

    if rows is None:
        rows = []

    # If the source lead has no embedding, the CTE produces no rows.
    if not rows:
        # Check whether the requested source embedding exists.
        source_query = """
            SELECT 1
            FROM lead_embeddings
            WHERE lead_id = %s
              AND model_version = %s
            LIMIT 1;
        """

        with _connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    source_query,
                    (
                        lead_id,
                        MODEL_VERSION,
                    ),
                )

                source_exists = (
                    cursor.fetchone()
                    is not None
                )

        if not source_exists:
            raise SourceEmbeddingNotFoundError(
                f"No M3 embedding found for lead_id={lead_id!r}."
            )

    similar = [
        SimilarLead(
            lead_id=str(row[0]),
            similarity=float(row[1]),
            outcome=int(row[2]),
        )
        for row in rows
    ]

    return SimilarLeadResult(
        lead_id=lead_id,
        model_version=MODEL_VERSION,
        k=k,
        similar=similar,
    )