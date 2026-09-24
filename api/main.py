from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .similar import (
    DatabaseConfigurationError,
    SimilarLeadError,
    SourceEmbeddingNotFoundError,
    get_similar_leads,
)
from .score import (
    ScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ModelCardResponse,
    score_single_lead,
    score_many_leads,
    get_model_card_api,
    healthz_api,
)
from .leads import router as leads_router


# ---------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------

class SimilarLeadResponse(BaseModel):
    lead_id: str
    similarity: float = Field(ge=-1.0, le=1.0)
    outcome: int = Field(ge=0, le=1)


class SimilarLeadsResponse(BaseModel):
    lead_id: str
    model_version: str
    k: int
    similar: list[SimilarLeadResponse]
    admissions: int
    total: int


# ---------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # No heavyweight M3 model is loaded at API startup.
    # Embeddings already exist in pgvector.
    # Warm the Module 5 caches (DB pool + schema, model bundle, scoring
    # CSVs, M4 features) so the first UI request is served fast instead
    # of paying the whole cold start. Best-effort: requests lazily
    # initialise anything missed here.
    try:
        from .score import (
            _ensure_schema_once,
            _get_bundle,
            _get_frames,
            _get_m4_features,
        )
        from leadiq.versioning import _pool_instance

        _ensure_schema_once()
        with _pool_instance().connection():
            pass
        _get_bundle()
        frames = _get_frames()
        _get_m4_features(frames["leads"])
    except Exception:
        pass
    yield


app = FastAPI(
    title="LeadIQ API",
    version="0.1.0",
    description=(
        "LeadIQ API for scoring, model versioning, "
        "M3 similarity search and future modules."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")

if allowed_origins.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [
        origin.strip()
        for origin in allowed_origins.split(",")
        if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(leads_router)


# ---------------------------------------------------------------------
# Request log: one line per call with status + duration, so the backend
# terminal visibly shows every request the frontend makes.
# ---------------------------------------------------------------------

_api_logger = logging.getLogger("leadiq.api")
if not _api_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("INFO: %(message)s"))
    _api_logger.addHandler(_handler)
_api_logger.setLevel(logging.INFO)
_api_logger.propagate = False


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _api_logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    database_configured: bool
    model_loaded: bool
    champion_version: str | None = None


@app.get(
    "/healthz",
    response_model=HealthResponse,
)
def healthz() -> HealthResponse:
    """Liveness check and whether the model is loaded."""
    result = healthz_api()
    return HealthResponse(
        status=result["status"],
        database_configured=result["database_configured"],
        model_loaded=result["model_loaded"],
        champion_version=result.get("champion_version"),
    )


# ---------------------------------------------------------------------
# Module 5: Scoring API
# ---------------------------------------------------------------------

@app.post(
    "/v1/score",
    response_model=ScoreResponse,
)
def score_lead(
    request: ScoreRequest,
) -> ScoreResponse:
    """Score a single enquiry by lead_id."""
    try:
        result = score_single_lead(request.lead_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ScoreResponse(
        lead_id=result["lead_id"],
        asof=result["asof"],
        probability=result["probability"],
        band=result["band"],
        reasons=[
            {"en": r["en"], "hi": r["hi"]} for r in result["reasons"]
        ],
        model_version=result["model_version"],
        duplicate_cluster_size=result["duplicate_cluster_size"],
    )


@app.post(
    "/v1/score/batch",
    response_model=BatchScoreResponse,
)
def score_batch(
    request: BatchScoreRequest,
) -> BatchScoreResponse:
    """
    Score up to 500 enquiries per request.

    This is the batch scoring endpoint for hidden set testing.
    """
    if len(request.lead_ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 leads per batch request.",
        )

    try:
        results = score_many_leads(request.lead_ids)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return BatchScoreResponse(
        results=[
            ScoreResponse(
                lead_id=r["lead_id"],
                asof=r["asof"],
                probability=r["probability"],
                band=r["band"],
                reasons=[
                    {"en": reason["en"], "hi": reason["hi"]}
                    for reason in r["reasons"]
                ],
                model_version=r["model_version"],
                duplicate_cluster_size=r["duplicate_cluster_size"],
            )
            for r in results
        ],
        count=len(results),
    )


# ---------------------------------------------------------------------
# Module 5: Model card API
# ---------------------------------------------------------------------

@app.get(
    "/v1/model",
    response_model=ModelCardResponse,
)
def get_model() -> ModelCardResponse:
    """Get the model card as JSON: version, training window, data hash, metrics, feature list."""
    try:
        card = get_model_card_api()
        return ModelCardResponse(
            version=card["version"],
            training_window=str(card["train_window"]),
            data_sha256=card["data_sha256"],
            metrics=card["metrics"],
            feature_list=card["feature_list"],
            status=card["status"],
            created_at=card.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------
# M3 similarity search (existing)
# ---------------------------------------------------------------------

@app.get(
    "/v1/leads/{lead_id}/similar",
    response_model=SimilarLeadsResponse,
)
def similar_leads(
    lead_id: str,
    k: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Number of nearest labelled enquiries to return.",
    ),
) -> SimilarLeadsResponse:
    try:
        result = get_similar_leads(lead_id=lead_id, k=k)
    except SourceEmbeddingNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DatabaseConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except SimilarLeadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    similar = [
        SimilarLeadResponse(
            lead_id=item.lead_id,
            similarity=round(item.similarity, 6),
            outcome=item.outcome,
        )
        for item in result.similar
    ]

    return SimilarLeadsResponse(
        lead_id=result.lead_id,
        model_version=result.model_version,
        k=result.k,
        similar=similar,
        admissions=result.admissions,
        total=result.total,
    )
