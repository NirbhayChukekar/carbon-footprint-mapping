 #============================================================================
# utils.py — Utility helpers
# =============================================================================
 
import logging
from typing import List
import ee
from app.config import SCALE_THRESHOLDS, SCALE_VALUES
 
logger = logging.getLogger(__name__)
 
 
def build_ee_geometry(coords: List[List[float]]) -> ee.Geometry.Polygon:
    return ee.Geometry.Polygon(coords)


def get_polygon_area_km2(geometry: ee.Geometry) -> float:
    """Polygon area in km² (GEE geodesic)."""
    area_m2: float = geometry.area(maxError=1).getInfo()
    return area_m2 / 1_000_000.0


def get_dynamic_scale(area_km2: float) -> int:
    if   area_km2 < SCALE_THRESHOLDS["small"]:  scale = SCALE_VALUES["small"]
    elif area_km2 < SCALE_THRESHOLDS["medium"]: scale = SCALE_VALUES["medium"]
    else:                                        scale = SCALE_VALUES["large"]
    logger.info(f"[Scale] {area_km2:.1f} km² → {scale} m")
    return scale
 
 
def extract_band_stats(stats_dict: dict, band_name: str) -> dict:
    return {
        "min":  float(stats_dict.get(f"{band_name}_min",  0.0) or 0.0),
        "max":  float(stats_dict.get(f"{band_name}_max",  0.0) or 0.0),
        "mean": float(stats_dict.get(f"{band_name}_mean", 0.0) or 0.0),
    }
 
 
def build_tile_url(map_id_dict: dict) -> str:
    return map_id_dict["tile_fetcher"].url_format