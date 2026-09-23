from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _database_url() -> str:
    database_url = __import__("os").environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return database_url


def _connect() -> psycopg.Connection[Any]:
    return psycopg.connect(_database_url())


# ---------------------------------------------------------------------------
# Schema: model_registry
# ---------------------------------------------------------------------------

def create_model_registry_table() -> None:
    """Create the model_registry table if it does not exist."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    version        text PRIMARY KEY,
                    created_at     timestamptz NOT NULL,
                    train_window   tstzrange   NOT NULL,
                    data_sha256    text        NOT NULL,
                    metrics        jsonb       NOT NULL,
                    status         text        NOT NULL
                        CHECK (status IN ('champion', 'challenger', 'retired')),
                    feature_list   text[]      NOT NULL
                );
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS one_champion
                ON model_registry (status)
                WHERE status = 'champion';
            """)
        conn.commit()


def create_lead_scores_table() -> None:
    """Create the lead_scores table (append-only) if it does not exist.

    Append-only is enforced at the database level: a
    BEFORE UPDATE OR DELETE trigger rejects any attempt to modify or
    remove an already-scored row. The only allowed write path is INSERT
    via :func:`append_lead_scores`. Every scoring event appends a new
    row -- the primary key includes ``scored_at`` so re-scoring the same
    lead never collides, it just adds another history row.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lead_scores (
                    lead_id        text        NOT NULL,
                    asof           timestamptz NOT NULL,
                    model_version  text        NOT NULL REFERENCES model_registry(version),
                    probability    real        NOT NULL,
                    band           text        NOT NULL,
                    reasons        jsonb       NOT NULL,
                    scored_at      timestamptz DEFAULT now(),
                    PRIMARY KEY (lead_id, asof, model_version, scored_at)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lead_scores_lead_id
                ON lead_scores (lead_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lead_scores_model_version
                ON lead_scores (model_version);
            """)
            # Migration: every scoring event must append a row, so the
            # natural key must NOT be unique on its own.
            # 1. Drop the no-dup constraint if an earlier version added it.
            # 2. Expand a legacy 3-column PK to include scored_at.
            # Tables with a surrogate PK (e.g. score_id) already allow
            # repeats and are left as-is.
            cur.execute("""
                DO $$
                DECLARE
                    _pk_name text;
                    _pk_def text;
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = 'lead_scores'::regclass
                          AND conname = 'lead_scores_no_dup'
                    ) THEN
                        ALTER TABLE lead_scores
                        DROP CONSTRAINT lead_scores_no_dup;
                    END IF;

                    SELECT conname, pg_get_constraintdef(oid)
                    INTO _pk_name, _pk_def
                    FROM pg_constraint
                    WHERE conrelid = 'lead_scores'::regclass
                      AND contype = 'p'
                    LIMIT 1;

                    IF _pk_def LIKE '%(lead_id, asof, model_version)%'
                       AND _pk_def NOT LIKE '%scored_at%' THEN
                        EXECUTE format(
                            'ALTER TABLE lead_scores DROP CONSTRAINT %I',
                            _pk_name
                        );
                        ALTER TABLE lead_scores ADD PRIMARY KEY
                            (lead_id, asof, model_version, scored_at);
                    END IF;
                END;
                $$;
            """)
            # Enforce append-only: reject UPDATE and DELETE of scored rows.
            cur.execute("""
                CREATE OR REPLACE FUNCTION prevent_lead_scores_modify()
                RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION
                        'lead_scores is append-only: % not allowed',
                        TG_OP;
                    RETURN NULL;
                END;
                $$ LANGUAGE plpgsql;
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS lead_scores_no_update_delete
                ON lead_scores;
            """)
            cur.execute("""
                CREATE TRIGGER lead_scores_no_update_delete
                BEFORE UPDATE OR DELETE ON lead_scores
                FOR EACH ROW EXECUTE FUNCTION prevent_lead_scores_modify();
            """)
        conn.commit()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def initialise_schema() -> None:
    """Create all Module 5 tables."""
    create_model_registry_table()
    create_lead_scores_table()


# ---------------------------------------------------------------------------
# Model registry operations
# ---------------------------------------------------------------------------

def compute_data_sha256(leads_path: str, messages_path: str,
                        calls_path: str, stages_path: str) -> str:
    """Compute a SHA-256 hash of the training data files for provenance."""
    sha = hashlib.sha256()
    for filepath in [leads_path, messages_path, calls_path, stages_path]:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
    return sha.hexdigest()


def register_model(
    version: str,
    train_window: tuple[datetime, datetime],
    data_sha256: str,
    metrics: dict[str, float],
    feature_list: list[str],
    status: str = "champion",
) -> None:
    """Register a model version in the model_registry table."""
    create_model_registry_table()

    train_window_str = f"[{train_window[0].isoformat()},{train_window[1].isoformat()}]"

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_registry
                    (version, created_at, train_window, data_sha256,
                     metrics, status, feature_list)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (version) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    train_window = EXCLUDED.train_window,
                    data_sha256 = EXCLUDED.data_sha256,
                    metrics = EXCLUDED.metrics,
                    status = EXCLUDED.status,
                    feature_list = EXCLUDED.feature_list;
                """,
                (
                    version,
                    datetime.now(timezone.utc),
                    train_window_str,
                    data_sha256,
                    json.dumps(metrics),
                    status,
                    feature_list,
                ),
            )
        conn.commit()


def get_champion_version() -> str | None:
    """Return the current champion model version, or None."""
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT version FROM model_registry
                WHERE status = 'champion'
                LIMIT 1;
                """
            )
            row = cursor.fetchone()
    return row[0] if row else None


def _serialize_range(r) -> str | None:
    """Convert a PostgreSQL tstzrange to a JSON-serializable string."""
    if r is None:
        return None
    try:
        lower = r.lower.isoformat() if hasattr(r, 'lower') and r.lower else None
        upper = r.upper.isoformat() if hasattr(r, 'upper') and r.upper else None
        return f"[{lower},{upper}]"
    except Exception:
        return str(r)


def get_model_card(version: str | None = None) -> dict[str, Any] | None:
    """Retrieve a model card as a dictionary."""
    with _connect() as conn:
        with conn.cursor() as cursor:
            if version is None:
                version = get_champion_version()
            if version is None:
                return None
            cursor.execute(
                """
                SELECT version, created_at, train_window, data_sha256,
                       metrics, status, feature_list
                FROM model_registry
                WHERE version = %s;
                """,
                (version,),
            )
            row = cursor.fetchone()
    if row is None:
        return None

    return {
        "version": row[0],
        "created_at": row[1].isoformat() if row[1] else None,
        "train_window": _serialize_range(row[2]),
        "data_sha256": row[3],
        "metrics": row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] else {}),
        "status": row[5],
        "feature_list": list(row[6]) if row[6] else [],
    }


def get_all_versions() -> list[dict[str, Any]]:
    """Return all registered model versions."""
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT version, created_at, train_window, data_sha256,
                       metrics, status, feature_list
                FROM model_registry
                ORDER BY created_at DESC;
                """
            )
            rows = cursor.fetchall()
    return [
        {
            "version": row[0],
            "created_at": row[1].isoformat() if row[1] else None,
            "train_window": _serialize_range(row[2]),
            "data_sha256": row[3],
            "metrics": row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] else {}),
            "status": row[5],
            "feature_list": list(row[6]) if row[6] else [],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Champion / Challenger promotion
# ---------------------------------------------------------------------------

def promote_challenger(
    challenger_version: str,
    min_pr_auc: float = 0.0,
    max_ece: float = float("inf"),
) -> bool:
    """
    Promote a challenger to champion if it beats the current champion
    on PR-AUC without being worse on ECE.

    Returns True if promotion succeeded, False otherwise.
    """
    create_model_registry_table()

    # Get current champion metrics
    champion_version = get_champion_version()
    if champion_version is None:
        # No champion exists; promote directly
        _set_status(challenger_version, "champion")
        return True

    champion_metrics = get_model_card(champion_version)
    if champion_metrics is None:
        return False

    challenger_metrics = get_model_card(challenger_version)
    if challenger_metrics is None:
        return False

    champion_pr_auc = champion_metrics["metrics"].get("pr_auc", 0.0)
    challenger_pr_auc = challenger_metrics["metrics"].get("pr_auc", 0.0)
    champion_ece = champion_metrics["metrics"].get("ece_10bin", float("inf"))
    challenger_ece = challenger_metrics["metrics"].get("ece_10bin", float("inf"))

    # Challenger must beat champion on PR-AUC and not be worse on ECE
    if challenger_pr_auc > champion_pr_auc and challenger_ece <= champion_ece:
        _set_status(challenger_version, "champion")
        if champion_version:
            _set_status(champion_version, "retired")
        return True

    return False


def _set_status(version: str, status: str) -> None:
    """Set the status of a model version."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE model_registry SET status = %s WHERE version = %s;",
                (status, version),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Lead scores (append-only)
# ---------------------------------------------------------------------------

def append_lead_scores(scores: list[dict[str, Any]]) -> int:
    """
    Append scored results to lead_scores table.

    Append-only: every call inserts a new row -- re-scoring the same
    lead appends another history row (distinguished by ``scored_at``),
    it never modifies an existing one.

    Each entry must have: lead_id, asof, model_version, probability, band, reasons.
    """
    create_lead_scores_table()

    if not scores:
        return 0

    with _connect() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for score in scores:
                # Per-row timestamp (not the DB default now(), which is
                # the transaction start and would be identical for every
                # row in a batch).
                scored_at = datetime.now(timezone.utc)
                cur.execute(
                    """
                    INSERT INTO lead_scores
                        (lead_id, asof, model_version, probability, band, reasons, scored_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        score["lead_id"],
                        score["asof"],
                        score["model_version"],
                        score["probability"],
                        score["band"],
                        json.dumps(score["reasons"]),
                        scored_at,
                    ),
                )
                if cur.rowcount == 1:
                    inserted += 1
            conn.commit()
        return inserted


def get_scored_leads(
    lead_id: str | None = None,
    model_version: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve scored leads from the database."""
    create_lead_scores_table()

    query = "SELECT lead_id, asof, model_version, probability, band, reasons FROM lead_scores WHERE 1=1"
    params: list[Any] = []
    param_idx = 0

    if lead_id is not None:
        query += f" AND lead_id = %s"
        params.append(lead_id)
        param_idx += 1
    if model_version is not None:
        query += f" AND model_version = %s"
        params.append(model_version)

    query += f" ORDER BY scored_at DESC LIMIT {limit}"

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [
        {
            "lead_id": row[0],
            "asof": row[1].isoformat() if row[1] else None,
            "model_version": row[2],
            "probability": row[3],
            "band": row[4],
            "reasons": row[5] if isinstance(row[5], list) else (json.loads(row[5]) if row[5] else []),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Duplicate cluster size helper
# ---------------------------------------------------------------------------

def get_duplicate_cluster_size(lead_id: str) -> int:
    """
    Look up the cluster size for a lead from the lead_clusters table.
    Returns 1 if not found (no duplicates).
    """
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cluster_size FROM lead_clusters
                    WHERE lead_id = %s;
                    """,
                    (lead_id,),
                )
                row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------
# Model card for API
# ---------------------------------------------------------------------------

def build_model_card_json() -> dict[str, Any]:
    """Build the model card response JSON for the API."""
    champion_version = get_champion_version()
    card = get_model_card(champion_version)
    if card is None:
        return {"status": "no_model_registered"}

    # Add feature list summary
    feature_list = card.get("feature_list", [])

    return {
        "version": card["version"],
        "training_window": card["train_window"],
        "data_sha256": card["data_sha256"],
        "metrics": card["metrics"],
        "feature_list": feature_list,
        "status": card["status"],
        "created_at": card["created_at"],
    }


def compute_feature_list(bundle: dict) -> list[str]:
    """Extract feature names from a model bundle."""
    return list(bundle.get("feature_names", []))


def compute_data_hash_from_bundle(bundle: dict) -> str:
    """Compute a hash representing the training data used."""
    # Use available metadata from the bundle
    metadata = json.dumps(
        {
            "split_config": bundle.get("split_config", {}),
            "metrics": bundle.get("metrics", {}),
            "random_seed": bundle.get("random_seed"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(metadata.encode()).hexdigest()


def health_check() -> dict[str, Any]:
    """Check if the API is alive and whether a model is loaded."""
    try:
        champion = get_champion_version()
        return {
            "status": "ok",
            "database_configured": True,
            "model_loaded": champion is not None,
            "champion_version": champion,
        }
    except Exception:
        return {
            "status": "degraded",
            "database_configured": False,
            "model_loaded": False,
        }
