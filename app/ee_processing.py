# =============================================================================
# ee_processing.py — Google Earth Engine Processing Pipeline
#
# Contains every GEE operation as a standalone, testable function:
#   1. Cloud masking + reflectance scaling
#   2. Image collection builder
#   3. Median composite
#   4. NDVI computation
#   5. Carbon proxy estimation
#   6. Statistics extraction
#   7. Map tile URL generation
# =============================================================================

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import ee

from app.config import (
    S2_DATASET,
    REFLECTANCE_SCALE,
    CARBON_FACTOR,
    CARBON_MID_NDVI,
    REDUCE_REGION_OPTS,
    VIS_RGB,
    VIS_NDVI,
    VIS_CARBON,
    VIS_CHANGE,
)
from app.utils import (
    build_ee_geometry,
    get_polygon_area_km2,
    get_dynamic_scale,
    extract_band_stats,
    build_tile_url,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — Cloud masking + reflectance scaling
# =============================================================================

def mask_and_scale(image: ee.Image) -> ee.Image:
    """
    Applies Sentinel-2 QA60 cloud mask and scales reflectance to [0, 1].

    QA60 bitmask:
        Bit 10 = opaque cloud
        Bit 11 = thin cirrus
    Both bits must be 0 for a pixel to pass.

    Args:
        image: Raw Sentinel-2 SR image.

    Returns:
        Cloud-masked, scaled image with original metadata preserved.
    """
    qa = image.select("QA60")
    cloud_bit_mask  = 1 << 10
    cirrus_bit_mask = 1 << 11

    clear_mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
          .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return (
        image
        .updateMask(clear_mask)
        .divide(REFLECTANCE_SCALE)
        .copyProperties(image, image.propertyNames())
    )


# =============================================================================
# STEP 2 — Image collection builder
# =============================================================================

def build_collection(
    geometry:         ee.Geometry,
    start_date:       str,
    end_date:         str,
    cloud_threshold:  float,
) -> ee.ImageCollection:
    """
    Loads and pre-processes a Sentinel-2 SR image collection.

    Two-stage cloud filtering:
        Stage 1 (scene-level): CLOUDY_PIXEL_PERCENTAGE filter.
            Eliminates whole scenes before pixel-level work begins.
            This is the primary performance optimisation for large polygons.
        Stage 2 (pixel-level): QA60 bitmask cloud mask (see mask_and_scale).

    Args:
        geometry:         Spatial filter (ee.Geometry.Polygon).
        start_date:       ISO date string "YYYY-MM-DD".
        end_date:         ISO date string "YYYY-MM-DD".
        cloud_threshold:  Maximum CLOUDY_PIXEL_PERCENTAGE (0–100).

    Returns:
        Filtered, cloud-masked, scaled ee.ImageCollection.

    Raises:
        ValueError: If the collection is empty after filtering.
    """
    collection = (
        ee.ImageCollection(S2_DATASET)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_and_scale)
    )

    # Evaluate collection size — blocks until GEE responds.
    # Done once per request; not inside a mapped function.
    count: int = collection.size().getInfo()

    if count == 0:
        raise ValueError(
            "No Sentinel-2 images found for the given region and date range. "
            f"Try relaxing cloud_threshold (current: {cloud_threshold}%) "
            "or expanding the date range."
        )

    logger.info(f"[Collection] {count} images found.")
    return collection


# =============================================================================
# STEP 3 — Median composite
# =============================================================================

def build_composite(
    collection: ee.ImageCollection,
    geometry:   ee.Geometry,
) -> ee.Image:
    """
    Creates a median composite clipped to the analysis polygon.

    Median is used instead of mean because it is statistically robust to
    outliers — a single cloud-shadow pixel cannot skew the result.

    Args:
        collection: Pre-processed Sentinel-2 image collection.
        geometry:   Clip boundary.

    Returns:
        Median composite ee.Image.
    """
    return collection.median().clip(geometry)


# =============================================================================
# STEP 4 — NDVI
# =============================================================================

def compute_ndvi(composite: ee.Image) -> ee.Image:
    """
    Computes NDVI from a Sentinel-2 composite.

    Formula: NDVI = (B8 - B4) / (B8 + B4)
        B8 = NIR (842 nm)
        B4 = Red (665 nm)

    GEE's normalizedDifference() handles division-by-zero internally.
    Output range: [-1, 1].

    Args:
        composite: Scaled Sentinel-2 median composite.

    Returns:
        Single-band NDVI image, band name "NDVI".
    """
    return composite.normalizedDifference(["B8", "B4"]).rename("NDVI")


# =============================================================================
# STEP 5 — Carbon proxy estimation
# =============================================================================

def estimate_carbon(ndvi: ee.Image) -> ee.Image:
    """
    Estimates proxy carbon stock density (Mg C/ha) from NDVI.

    Phase 1 piecewise linear model:
        NDVI ≤ 0            → 0 Mg C/ha   (water, shadow, bare rock)
        0 < NDVI < 0.2      → NDVI × (CARBON_FACTOR × 0.1)  [sparse veg, dampened]
        NDVI ≥ 0.2          → NDVI × CARBON_FACTOR           [productive land]

    Rationale for dampening sparse-vegetation pixels:
        A direct linear model overestimates carbon for sandy or degraded land
        (NDVI ≈ 0.1) because it doesn't distinguish low green signal from
        genuine biomass. The 0.1× factor gives a near-zero estimate for these
        pixels while keeping the model numerically simple.

    Phase 2 upgrade path:
        Replace this function with a per-biome regression model trained
        on GEDI LiDAR aboveground biomass data or GlobBiomass AGB maps.
        The output band name and interface remain unchanged so callers
        don't need to be updated.

    Args:
        ndvi: NDVI image, range [-1, 1].

    Returns:
        Single-band Carbon image, band name "Carbon_MgCha".
    """
    ndvi_clamped  = ndvi.max(0)
    low_veg_mask  = ndvi_clamped.lt(CARBON_MID_NDVI)
    carbon_normal = ndvi_clamped.multiply(CARBON_FACTOR)
    carbon_low    = ndvi_clamped.multiply(CARBON_FACTOR * 0.1)

    return (
        carbon_normal
        .where(low_veg_mask, carbon_low)
        .rename("Carbon_MgCha")
    )


# =============================================================================
# STEP 6 — Statistics
# =============================================================================

def compute_stats(
    ndvi:        ee.Image,
    carbon:      ee.Image,
    geometry:    ee.Geometry,
    scale:       int,
) -> dict:
    """
    Computes min / max / mean statistics for NDVI and Carbon over the polygon.

    Uses three memory-safety parameters to prevent "User memory limit exceeded":
        bestEffort=True  → GEE silently raises scale if computation is too heavy.
        tileScale=4      → Subdivides tiles to reduce per-tile memory by 16×.
        maxPixels=1e10   → High pixel budget; actual limit is enforced by scale.

    Args:
        ndvi:     NDVI image.
        carbon:   Carbon proxy image.
        geometry: Analysis polygon.
        scale:    Processing scale in metres (from get_dynamic_scale).

    Returns:
        Dict with keys "ndvi" and "carbon", each containing min/max/mean.
    """
    combined = ndvi.rename("NDVI").addBands(carbon.rename("Carbon_MgCha"))

    reducer = (
        ee.Reducer.min()
          .combine(ee.Reducer.max(),  sharedInputs=True)
          .combine(ee.Reducer.mean(), sharedInputs=True)
    )

    raw_stats: dict = combined.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        **REDUCE_REGION_OPTS,
    ).getInfo()

    logger.debug(f"[Stats] Raw GEE output: {raw_stats}")

    return {
        "ndvi":   extract_band_stats(raw_stats, "NDVI"),
        "carbon": extract_band_stats(raw_stats, "Carbon_MgCha"),
    }


# =============================================================================
# STEP 7 — Map tile URL generation
# =============================================================================

def get_tile_urls(
    composite: ee.Image,
    ndvi:      ee.Image,
    carbon:    ee.Image,
) -> dict:
    """
    Generates XYZ tile URL templates for all three map layers.

    getMapId() contacts the GEE tile server and returns a URL template
    with {x}, {y}, {z} placeholders, ready to plug into any web mapping
    library (Leaflet TileLayer, MapboxGL raster source, OpenLayers, etc.).

    Tile URLs are time-limited (~1 hour). For production use, implement
    a token-refresh mechanism or proxy tiles through your own server.

    Args:
        composite: Scaled median RGB composite.
        ndvi:      NDVI image.
        carbon:    Carbon proxy image.

    Returns:
        Dict with keys "rgb", "ndvi", "carbon" mapping to tile URL strings.
    """
    rgb_map_id    = composite.getMapId(VIS_RGB)
    ndvi_map_id   = ndvi.getMapId(VIS_NDVI)
    carbon_map_id = carbon.getMapId(VIS_CARBON)

    return {
        "rgb":    build_tile_url(rgb_map_id),
        "ndvi":   build_tile_url(ndvi_map_id),
        "carbon": build_tile_url(carbon_map_id),
    }


# =============================================================================
# PUBLIC ENTRY POINT — orchestrates the full pipeline
# =============================================================================

def run_analysis(
    coords:          List[List[float]],
    start_date:      str,
    end_date:        str,
    cloud_threshold: float,
) -> dict:
    """
    Runs the complete Phase 1 GEE processing pipeline.

    Orchestration order:
        1. Build ee.Geometry from coordinates
        2. Compute polygon area and select dynamic scale
        3. Load and filter Sentinel-2 collection
        4. Create median composite
        5. Compute NDVI
        6. Estimate carbon proxy
        7. Extract statistics
        8. Generate tile URLs
        9. Return structured result dict

    Args:
        coords:           Closed polygon ring [[lon, lat], ...].
        start_date:       "YYYY-MM-DD"
        end_date:         "YYYY-MM-DD"
        cloud_threshold:  CLOUDY_PIXEL_PERCENTAGE ceiling (0–100).

    Returns:
        Dict matching the AnalyzeResponse schema:
        {
            "tile_url_rgb":    str,
            "tile_url_ndvi":   str,
            "tile_url_carbon": str,
            "stats": {
                "ndvi":   {"min": float, "max": float, "mean": float},
                "carbon": {"min": float, "max": float, "mean": float},
                "image_count":       int,
                "polygon_area_km2":  float,
                "processing_scale_m": int,
            }
        }

    Raises:
        ValueError:     Empty collection or invalid geometry.
        ee.EEException: Any GEE-side computation error.
    """
    # 1. Geometry
    geometry = build_ee_geometry(coords)

    # 2. Dynamic scale
    area_km2 = get_polygon_area_km2(geometry)
    scale    = get_dynamic_scale(area_km2)

    # 3. Collection
    collection = build_collection(geometry, start_date, end_date, cloud_threshold)
    image_count: int = collection.size().getInfo()

    # 4. Composite
    composite = build_composite(collection, geometry)

    # 5. NDVI
    ndvi = compute_ndvi(composite)

    # 6. Carbon
    carbon = estimate_carbon(ndvi)

    # 7. Statistics
    band_stats = compute_stats(ndvi, carbon, geometry, scale)

    # 8. Tile URLs
    tile_urls = get_tile_urls(composite, ndvi, carbon)

    logger.info("[Pipeline] Analysis complete.")

    # 9. Structured result
    return {
        "tile_url_rgb":    tile_urls["rgb"],
        "tile_url_ndvi":   tile_urls["ndvi"],
        "tile_url_carbon": tile_urls["carbon"],
        "stats": {
            "ndvi":                band_stats["ndvi"],
            "carbon":              band_stats["carbon"],
            "image_count":         image_count,
            "polygon_area_km2":    round(area_km2, 2),
            "processing_scale_m":  scale,
        },
    }


# =============================================================================
# TEMPORAL COMPARISON — exactly two calendar years (FROM → TO)
# =============================================================================

def _year_date_range(year: int) -> Tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def _validate_comparison_years(year1: int, year2: int) -> None:
    if year2 <= year1:
        raise ValueError(
            f"Temporal comparison requires two distinct years with year2 after year1 "
            f"(got year1={year1}, year2={year2})."
        )


def _gain_loss_status(absolute_difference: float, threshold: float = 0.5) -> str:
    if absolute_difference > threshold:
        return "gain"
    if absolute_difference < -threshold:
        return "loss"
    return "stable"


def _comparison_statistics(value_year1: float, value_year2: float) -> dict:
    absolute_difference = value_year2 - value_year1
    if value_year1 > 0:
        percentage_change = (absolute_difference / value_year1) * 100.0
    elif value_year2 > 0:
        percentage_change = 100.0
    else:
        percentage_change = 0.0

    gain_loss_status = _gain_loss_status(absolute_difference)
    return {
        "value_year1":          round(value_year1, 4),
        "value_year2":          round(value_year2, 4),
        "absolute_difference":  round(absolute_difference, 4),
        "percentage_change":    round(percentage_change, 2),
        "gain_loss_status":     gain_loss_status,
    }


def _process_single_year(
    geometry:        ee.Geometry,
    year:            int,
    cloud_threshold: float,
) -> dict:
    """
    Load and process one calendar year independently (no cross-year aggregation).
    """
    start_date, end_date = _year_date_range(year)
    area_km2 = get_polygon_area_km2(geometry)
    scale    = get_dynamic_scale(area_km2)

    collection  = build_collection(geometry, start_date, end_date, cloud_threshold)
    image_count = collection.size().getInfo()
    composite   = build_composite(collection, geometry)
    ndvi        = compute_ndvi(composite)
    carbon      = estimate_carbon(ndvi)
    band_stats  = compute_stats(ndvi, carbon, geometry, scale)
    tile_urls   = get_tile_urls(composite, ndvi, carbon)

    return {
        "year":            year,
        "start_date":      start_date,
        "end_date":        end_date,
        "image_count":     image_count,
        "composite":       composite,
        "ndvi":            ndvi,
        "carbon":          carbon,
        "band_stats":      band_stats,
        "tile_urls":       tile_urls,
        "polygon_area_km2": round(area_km2, 2),
        "processing_scale_m": scale,
    }


def _year_result_payload(year_data: dict) -> dict:
    carbon_stats = year_data["band_stats"]["carbon"]
    ndvi_stats   = year_data["band_stats"]["ndvi"]
    tiles        = year_data["tile_urls"]
    return {
        "year":            year_data["year"],
        "start_date":      year_data["start_date"],
        "end_date":        year_data["end_date"],
        "image_count":     year_data["image_count"],
        "tile_url_rgb":    tiles["rgb"],
        "tile_url_ndvi":   tiles["ndvi"],
        "tile_url_carbon": tiles["carbon"],
        "mean_carbon":     carbon_stats["mean"],
        "mean_ndvi":       ndvi_stats["mean"],
        "stats": {
            "ndvi":   ndvi_stats,
            "carbon": carbon_stats,
        },
    }


def run_comparison(
    coords:          List[List[float]],
    year1:           int,
    year2:           int,
    cloud_threshold: float,
) -> dict:
    """
    Compare exactly two selected years for the same polygon.

    Each year is loaded and processed independently, then differenced at the
    pixel level for the change map and at the mean-carbon level for statistics.
    """
    _validate_comparison_years(year1, year2)
    geometry = build_ee_geometry(coords)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_y1 = executor.submit(_process_single_year, geometry, year1, cloud_threshold)
        future_y2 = executor.submit(_process_single_year, geometry, year2, cloud_threshold)
        data_y1   = future_y1.result()
        data_y2   = future_y2.result()

    year1_payload = _year_result_payload(data_y1)
    year2_payload = _year_result_payload(data_y2)

    carbon_y1 = year1_payload["mean_carbon"]
    carbon_y2 = year2_payload["mean_carbon"]
    ndvi_y1   = year1_payload["mean_ndvi"]
    ndvi_y2   = year2_payload["mean_ndvi"]

    carbon_stats = _comparison_statistics(carbon_y1, carbon_y2)
    ndvi_stats   = _comparison_statistics(ndvi_y1, ndvi_y2)

    change_image = data_y2["carbon"].subtract(data_y1["carbon"]).rename("Carbon_Change")
    change_map_id = change_image.getMapId(VIS_CHANGE)
    change_tile   = build_tile_url(change_map_id)

    delta_carbon = carbon_stats["absolute_difference"]
    delta_ndvi   = ndvi_stats["absolute_difference"]
    trend        = carbon_stats["gain_loss_status"]

    return {
        "year1": year1_payload,
        "year2": year2_payload,
        "change": {
            "tile_url_change":    change_tile,
            "delta_carbon_mgcha": delta_carbon,
            "delta_ndvi":         delta_ndvi,
            "trend":              trend,
        },
        "year1_data": year1_payload,
        "year2_data": year2_payload,
        "difference_data": {
            "from_year":           year1,
            "to_year":             year2,
            "tile_url_difference": change_tile,
            "delta_carbon_mgcha":  delta_carbon,
            "delta_ndvi":          delta_ndvi,
            "carbon_stats":        carbon_stats,
            "ndvi_stats":          ndvi_stats,
        },
        "statistics": carbon_stats,
    }