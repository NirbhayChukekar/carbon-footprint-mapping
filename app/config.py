# =============================================================================
# config.py — Central configuration  (v3)
# =============================================================================

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    GEE_SERVICE_ACCOUNT_KEY:   str = "credentials/service_account.json"
    GEE_SERVICE_ACCOUNT_EMAIL: str = ""
    API_HOST: str  = "0.0.0.0"
    API_PORT: int  = 8000
    DEBUG:    bool = True

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ── Dataset ────────────────────────────────────────────────────────────────
S2_DATASET        = "COPERNICUS/S2_SR_HARMONIZED"
REFLECTANCE_SCALE = 10_000

# ── Carbon proxy ───────────────────────────────────────────────────────────
CARBON_FACTOR   = 80
CARBON_LOW_NDVI = 0.1
CARBON_MID_NDVI = 0.2

# ── Dynamic scale ──────────────────────────────────────────────────────────
SCALE_THRESHOLDS = {"small": 100, "medium": 5_000}
SCALE_VALUES     = {"small": 10,  "medium": 30, "large": 100}

# ── reduceRegion safety params ─────────────────────────────────────────────
REDUCE_REGION_OPTS = {
    "bestEffort": True,
    "tileScale":  4,
    "maxPixels":  1e10,
}

# ── Visualisation params ───────────────────────────────────────────────────
VIS_RGB = {
    "bands": ["B4", "B3", "B2"],
    "min":   0,
    "max":   0.3,
    "gamma": 1.2,
}
VIS_NDVI = {
    "min":     -1,
    "max":      1,
    "palette": ["0000FF", "FFFFFF", "006400"],
}
VIS_CARBON = {
    "min":     0,
    "max":     80,
    "palette": ["FF0000", "FF7F00", "FFFF00", "7CFC00", "006400"],
}
# Change map: red = loss, white = no change, green = gain
VIS_CHANGE = {
    "min":     -40,
    "max":      40,
    "palette": ["FF0000", "FF6B35", "FFFFFF", "85C1E9", "006400"],
}