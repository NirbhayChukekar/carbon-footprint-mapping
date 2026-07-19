# =============================================================================
# main.py — FastAPI Application  (v3)
# Endpoints:
#   GET  /health
#   POST /analyze   — single year analysis
#   POST /compare   — two-year comparison with change detection
# =============================================================================

import logging
from contextlib import asynccontextmanager

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ee_auth       import initialize_ee
from app.ee_processing import run_analysis, run_comparison
from app.models        import (
    AnalyzeRequest, AnalyzeResponse,
    CompareRequest, CompareResponse,
    ErrorResponse,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("══════════════════════════════════════")
    logger.info("  CarbonLens API v3  |  Starting…")
    logger.info("══════════════════════════════════════")
    try:
        initialize_ee()
        logger.info("[Startup] Earth Engine ready.")
    except Exception as exc:
        logger.error(f"[Startup] EE init failed: {exc}")
        raise
    yield
    logger.info("[Shutdown] CarbonLens API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title     = "CarbonLens API v3",
    version   = "3.0.0",
    lifespan  = lifespan,
    docs_url  = "/docs",
    redoc_url = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _handle_error(exc: Exception):
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ee.EEException):
        raise HTTPException(status_code=500, detail=f"GEE error: {exc}")
    logger.exception(f"Unexpected error: {exc}")
    raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "service": "CarbonLens API", "version": "3.0.0"}


@app.post(
    "/analyze",
    response_model = AnalyzeResponse,
    tags           = ["Analysis"],
    summary        = "Single-year carbon analysis",
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Processes a polygon for one date range.
    Returns RGB, NDVI, Carbon tile URLs + statistics.

    **Optimisations in v3:**
    - Area + scene count fetched in one GEE round-trip
    - Tile URL generation runs in parallel (3 threads)
    - Stats skipped for polygons > 300 km² (returns zeros instead of 40s wait)
    """
    logger.info(
        f"[/analyze] {request.start_date}→{request.end_date} "
        f"| cloud≤{request.cloud_threshold}% | pts={len(request.geometry)}"
    )
    try:
        result = run_analysis(
            coords          = request.geometry,
            start_date      = request.start_date,
            end_date        = request.end_date,
            cloud_threshold = request.cloud_threshold,
        )
        logger.info("[/analyze] Done.")
        return AnalyzeResponse(**result)
    except Exception as exc:
        _handle_error(exc)


@app.post(
    "/compare",
    response_model = CompareResponse,
    tags           = ["Comparison"],
    summary        = "Two-year carbon change comparison",
)
async def compare(request: CompareRequest) -> CompareResponse:
    """
    Processes the SAME polygon for two different years IN PARALLEL.

    Returns:
    - `year1` — tile URLs + mean carbon/NDVI for Year 1
    - `year2` — tile URLs + mean carbon/NDVI for Year 2
    - `change` — change detection tile (red=loss, green=gain) + delta values

    **Display on frontend:**
    Show year1.tile_url_carbon and year2.tile_url_carbon as two separate
    layers (or side-by-side panels). Show change.tile_url_change as the
    third layer. Display delta_carbon_mgcha as the headline number.
    """
    logger.info(
        f"[/compare] {request.year1}→{request.year2} "
        f"| cloud≤{request.cloud_threshold}% | pts={len(request.geometry)}"
    )
    try:
        result = run_comparison(
            coords          = request.geometry,
            year1           = request.year1,
            year2           = request.year2,
            cloud_threshold = request.cloud_threshold,
        )
        logger.info(
            f"[/compare] Done. "
            f"ΔCarbon={result['change']['delta_carbon_mgcha']:+.2f} | "
            f"trend={result['change']['trend']}"
        )
        return CompareResponse(**result)
    except Exception as exc:
        _handle_error(exc)