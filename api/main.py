from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
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
    get_model_card_api,
    healthz_api,
)


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

    results = []
    for lead_id in request.lead_ids:
        try:
            result = score_single_lead(lead_id)
            results.append(ScoreResponse(
                lead_id=result["lead_id"],
                asof=result["asof"],
                probability=result["probability"],
                band=result["band"],
                reasons=[
                    {"en": r["en"], "hi": r["hi"]} for r in result["reasons"]
                ],
                model_version=result["model_version"],
                duplicate_cluster_size=result["duplicate_cluster_size"],
            ))
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Lead {lead_id} not found",
                ) from exc
            raise

    return BatchScoreResponse(results=results, count=len(results))


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
