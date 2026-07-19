# =============================================================================
# models.py — Pydantic schemas  (v3 — adds CompareRequest / CompareResponse)
# =============================================================================

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re


# =============================================================================
# Shared validators
# =============================================================================

def validate_coords(coords):
    if len(coords) < 4:
        raise ValueError("Polygon must have at least 4 coordinate pairs.")
    for i, pt in enumerate(coords):
        if len(pt) != 2:
            raise ValueError(f"Coordinate at index {i} must be [lon, lat].")
        lon, lat = pt
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} out of range.")
        if not (-90  <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of range.")
    if coords[0] != coords[-1]:
        raise ValueError("Ring must be closed: first == last point.")
    return coords


# =============================================================================
# /analyze  — single year
# =============================================================================

class AnalyzeRequest(BaseModel):
    geometry:         List[List[float]] = Field(..., example=[[76.0,11.5],[76.5,11.5],[76.5,12.0],[76.0,12.0],[76.0,11.5]])
    start_date:       str               = Field(..., example="2023-01-01")
    end_date:         str               = Field(..., example="2023-12-31")
    cloud_threshold:  float             = Field(default=20.0, ge=0, le=100)

    @field_validator("geometry")
    @classmethod
    def check_geometry(cls, v): return validate_coords(v)

    @field_validator("start_date", "end_date")
    @classmethod
    def check_date(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Date '{v}' must be YYYY-MM-DD.")
        return v


class BandStats(BaseModel):
    min:  float
    max:  float
    mean: float


class AnalysisStats(BaseModel):
    ndvi:               BandStats
    carbon:             BandStats
    stats_skipped:      bool
    image_count:        int
    polygon_area_km2:   float
    processing_scale_m: int


class AnalyzeResponse(BaseModel):
    tile_url_rgb:    str
    tile_url_ndvi:   str
    tile_url_carbon: str
    stats:           AnalysisStats


# =============================================================================
# /compare  — two years
# =============================================================================

class CompareRequest(BaseModel):
    geometry:         List[List[float]] = Field(..., example=[[76.0,11.5],[76.5,11.5],[76.5,12.0],[76.0,12.0],[76.0,11.5]])
    year1:            int               = Field(..., ge=2017, le=2030, example=2020)
    year2:            int               = Field(..., ge=2017, le=2030, example=2023)
    cloud_threshold:  float             = Field(default=30.0, ge=0, le=100)

    @field_validator("geometry")
    @classmethod
    def check_geometry(cls, v): return validate_coords(v)

    @field_validator("year2")
    @classmethod
    def check_years(cls, v, info):
        if "year1" in info.data and v <= info.data["year1"]:
            raise ValueError("year2 must be greater than year1.")
        return v


class YearResult(BaseModel):
    year:            int
    image_count:     int
    tile_url_rgb:    str
    tile_url_ndvi:   str
    tile_url_carbon: str
    mean_carbon:     float
    mean_ndvi:       float


class ChangeResult(BaseModel):
    tile_url_change:     str
    delta_carbon_mgcha:  float   # year2 − year1, positive = gain
    delta_ndvi:          float
    trend:               str     # "gain" | "loss" | "stable"


class ComparisonStatistics(BaseModel):
    value_year1:         float
    value_year2:         float
    absolute_difference: float
    percentage_change:   float
    gain_loss_status:    str     # "gain" | "loss" | "stable"


class DifferenceData(BaseModel):
    from_year:           int
    to_year:             int
    tile_url_difference: str
    delta_carbon_mgcha:  float
    delta_ndvi:          float
    carbon_stats:        ComparisonStatistics
    ndvi_stats:          ComparisonStatistics


class CompareResponse(BaseModel):
    year1:             YearResult
    year2:             YearResult
    change:            ChangeResult
    year1_data:        YearResult
    year2_data:        YearResult
    difference_data:   DifferenceData
    statistics:        ComparisonStatistics


# =============================================================================
# Error
# =============================================================================

class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None