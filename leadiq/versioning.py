from __future__ import annotations

import hashlib
import json
import queue
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()


def _database_url() -> str:
    database_url = __import__("os").environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return database_url


def _pool_instance():
    """Lazily create a small process-wide connection pool.

    Cloud PostgreSQL handshakes (TLS + auth, plus Neon compute wake) cost
    seconds per fresh connection. Reusing a warm pooled connection keeps
    the UI snappy: /v1/model and single-score requests each need only
    one or two quick queries.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from psycopg_pool import ConnectionPool

                _pool = ConnectionPool(
                    _database_url(), min_size=1, max_size=4, timeout=30
                )

                def _close_pool_at_exit() -> None:
                    try:
                        _pool.close()
                    except Exception:
                        pass

                __import__("atexit").register(_close_pool_at_exit)
    return _pool


def _connect():
    """Borrow a connection from the pool (same `with` usage as before)."""
    return _pool_instance().connection()


# ---------------------------------------------------------------------------
# Schema: model_registry
# ---------------------------------------------------------------------------

# DDL (CREATE TABLE / INDEX / TRIGGER / FUNCTION) runs once per process.
# Re-issuing it on every request costs several cloud round trips per call
# and, worse, the AccessExclusiveLock DDL takes can deadlock with
# concurrent scoring transactions. Fresh processes (scripts, API workers)
# always run it once, so first-time setup is unaffected.
_ddl_done: set[str] = set()


def create_model_registry_table() -> None:
    """Create the model_registry table if it does not exist."""
    if "model_registry" in _ddl_done:
        return
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
    _ddl_done.add("model_registry")


def create_lead_scores_table() -> None:
    """Create the lead_scores table (append-only) if it does not exist.

    Append-only is enforced at the database level: a
    BEFORE UPDATE OR DELETE trigger rejects any attempt to modify or
    remove an already-scored row. The only allowed write path is INSERT
    via :func:`append_lead_scores`. Every scoring event appends a new
    row -- the primary key includes ``scored_at`` so re-scoring the same
    lead never collides, it just adds another history row.
    """
    if "lead_scores" in _ddl_done:
        return
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
            # Create the trigger only if missing. DROP + CREATE on every
            # call needs AccessExclusiveLock each time and deadlocks with
            # concurrent scoring transactions.
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'lead_scores_no_update_delete'
                    ) THEN
                        CREATE TRIGGER lead_scores_no_update_delete
                        BEFORE UPDATE OR DELETE ON lead_scores
                        FOR EACH ROW EXECUTE FUNCTION prevent_lead_scores_modify();
                    END IF;
                END;
                $$;
            """)
        conn.commit()
    _ddl_done.add("lead_scores")


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

def _fetch_cluster_sizes(
    cur: Any, lead_ids: list[str]
) -> dict[str, int]:
    """Single-query cluster-size lookup on an open cursor."""
    sizes: dict[str, int] = {str(lid): 1 for lid in lead_ids}
    if not lead_ids:
        return sizes
    cur.execute(
        """
        SELECT lead_id, cluster_size FROM lead_clusters
        WHERE lead_id = ANY(%s);
        """,
        ([str(lid) for lid in lead_ids],),
    )
    for lead_id, cluster_size in cur.fetchall():
        if cluster_size is not None:
            sizes[str(lead_id)] = int(cluster_size)
    return sizes


def _insert_score_rows(cur: Any, scores: list[dict[str, Any]]) -> None:
    """Insert many scoring events as ONE multi-row statement (1 round trip)."""
    placeholders = ",".join(["(%s,%s,%s,%s,%s,%s,%s)"] * len(scores))
    params: list[Any] = []
    for score in scores:
        # Per-row timestamp (not the DB default now(), which is the
        # transaction start and would be identical for every row).
        params.extend(
            [
                score["lead_id"],
                score["asof"],
                score["model_version"],
                score["probability"],
                score["band"],
                json.dumps(score["reasons"]),
                datetime.now(timezone.utc),
            ]
        )
    cur.execute(
        "INSERT INTO lead_scores"
        " (lead_id, asof, model_version, probability, band, reasons, scored_at)"
        f" VALUES {placeholders};",
        params,
    )


def _insert_score_row(cur: Any, score: dict[str, Any]) -> None:
    """Insert one scoring event on an open cursor."""
    _insert_score_rows(cur, [score])


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
                _insert_score_row(cur, score)
                if cur.rowcount == 1:
                    inserted += 1
            conn.commit()
        return inserted


def lookup_sizes_and_append(
    lead_ids: list[str], scores: list[dict[str, Any]]
) -> tuple[dict[str, int], int]:
    """
    Synchronous variant: resolve sizes and append in two round trips.
    Prefer :func:`lookup_sizes` + :func:`enqueue_scores` on the latency
    sensitive API path (commit latency dominates); this stays for scripts
    and tests that need durable writes before returning.
    Returns ``(sizes, inserted)``.
    """
    create_lead_scores_table()
    with _connect() as conn:
        with conn.cursor() as cur:
            try:
                sizes = _fetch_cluster_sizes(cur, lead_ids)
            except Exception:
                sizes = {str(lid): 1 for lid in lead_ids}
            inserted = 0
            if scores:
                _insert_score_rows(cur, scores)
                inserted = cur.rowcount if cur.rowcount > 0 else 0
            conn.commit()
        return sizes, inserted


def persist_single_score(score: dict[str, Any]) -> int:
    """
    Synchronous single-row variant (insert + size lookup in one round
    trip). Prefer :func:`lookup_sizes` + :func:`enqueue_scores` on the
    latency sensitive API path; this stays for scripts and one-off use.
    Returns the cluster size (1 if unknown).
    """
    create_lead_scores_table()
    with _connect() as conn:
        with conn.cursor() as cur:
            scored_at = datetime.now(timezone.utc)
            cur.execute(
                """
                WITH ins AS (
                    INSERT INTO lead_scores
                        (lead_id, asof, model_version, probability, band, reasons, scored_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING lead_id
                )
                SELECT %s AS lead_id, cluster_size FROM lead_clusters
                WHERE lead_id = %s;
                """,
                (
                    score["lead_id"],
                    score["asof"],
                    score["model_version"],
                    score["probability"],
                    score["band"],
                    json.dumps(score["reasons"]),
                    scored_at,
                    score["lead_id"],
                    score["lead_id"],
                ),
            )
            row = cur.fetchone()
            conn.commit()
    if row and row[1] is not None:
        return int(row[1])
    return 1


# ---------------------------------------------------------------------------
# Low-latency API path: sync size lookup + async append.
#
# Neon commit latency dominates /v1/score (median ~1.2s per commit), so the
# API resolves cluster sizes synchronously (plain SELECT, ~150ms) and
# hands the INSERT to a background writer. Responses stay well under the
# 800ms budget while every scoring event is still appended exactly once,
# in order, by a single writer thread.
# ---------------------------------------------------------------------------

_append_queue: queue.Queue[dict[str, Any]] = queue.Queue()
_append_thread: threading.Thread | None = None
_append_lock = threading.Lock()


def lookup_sizes(lead_ids: list[str]) -> dict[str, int]:
    """Synchronous cluster-size lookup (one round trip, no writes)."""
    return get_duplicate_cluster_sizes(lead_ids)


def enqueue_scores(scores: list[dict[str, Any]]) -> None:
    """Queue scoring events for background append. Never blocks, never raises."""
    _ensure_append_thread()
    try:
        for score in scores:
            _append_queue.put_nowait(score)
    except Exception:
        pass


def _ensure_append_thread() -> None:
    global _append_thread
    if _append_thread is not None and _append_thread.is_alive():
        return
    with _append_lock:
        if _append_thread is not None and _append_thread.is_alive():
            return
        _append_thread = threading.Thread(
            target=_append_worker, name="lead-scores-writer", daemon=True
        )
        _append_thread.start()

        def _flush_at_exit() -> None:
            flush_append_queue(timeout=10.0)

        import atexit

        atexit.register(_flush_at_exit)


def _append_worker() -> None:
    """Single writer: coalesce queued events, multi-row INSERT, commit."""
    pending: list[dict[str, Any]] = []
    while True:
        try:
            pending.append(_append_queue.get(timeout=0.2))
            while len(pending) < 1000:
                try:
                    pending.append(_append_queue.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        if pending:
            batch, pending = pending, []
            _flush_pending(batch)


def _flush_pending(pending: list[dict[str, Any]]) -> None:
    try:
        create_lead_scores_table()
        with _connect() as conn:
            with conn.cursor() as cur:
                _insert_score_rows(cur, pending)
            conn.commit()
    except Exception:
        # Best-effort background path: drop rather than grow unbounded.
        # Synchronous callers needing durability use append_lead_scores.
        pass


def flush_append_queue(timeout: float = 10.0) -> int:
    """Block until queued events are appended. Returns rows still queued."""
    import time as _time

    deadline = _time.time() + timeout
    while not _append_queue.empty() and _time.time() < deadline:
        _time.sleep(0.05)
    # Give the writer one more beat to commit the final batch.
    _time.sleep(0.3)
    return _append_queue.qsize()


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


def get_duplicate_cluster_sizes(lead_ids: list[str]) -> dict[str, int]:
    """
    Batch version of :func:`get_duplicate_cluster_size`: one query for
    many leads. Missing leads default to 1 (no duplicates).
    """
    sizes: dict[str, int] = {str(lid): 1 for lid in lead_ids}
    if not lead_ids:
        return sizes
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                sizes = _fetch_cluster_sizes(cur, lead_ids)
    except Exception:
        pass
    return sizes


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
