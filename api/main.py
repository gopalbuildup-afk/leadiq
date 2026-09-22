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
    # python-dotenv is optional in production.
    pass

from .similar import (
    DatabaseConfigurationError,
    SimilarLeadError,
    SourceEmbeddingNotFoundError,
    get_similar_leads,
)


# ---------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------

class SimilarLeadResponse(BaseModel):
    lead_id: str
    similarity: float = Field(
        ge=-1.0,
        le=1.0,
    )
    outcome: int = Field(
        ge=0,
        le=1,
    )


class SimilarLeadsResponse(BaseModel):
    lead_id: str
    model_version: str
    k: int
    similar: list[SimilarLeadResponse]
    admissions: int
    total: int


class HealthResponse(BaseModel):
    status: str
    database_configured: bool


# ---------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    # No heavyweight M3 model is loaded at API startup.
    # Embeddings already exist in pgvector.
    yield


app = FastAPI(
    title="LeadIQ API",
    version="0.1.0",
    description=(
        "LeadIQ API for M3 similarity search and future modules."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "*",
)

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

@app.get(
    "/healthz",
    response_model=HealthResponse,
)
def healthz() -> HealthResponse:

    return HealthResponse(
        status="ok",
        database_configured=bool(
            os.getenv("DATABASE_URL")
        ),
    )


# ---------------------------------------------------------------------
# M3 similarity search
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

        result = get_similar_leads(
            lead_id=lead_id,
            k=k,
        )

    except SourceEmbeddingNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except DatabaseConfigurationError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except SimilarLeadError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    similar = [
        SimilarLeadResponse(
            lead_id=item.lead_id,
            similarity=round(
                item.similarity,
                6,
            ),
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