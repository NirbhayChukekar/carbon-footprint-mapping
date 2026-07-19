# =============================================================================
# ee_auth.py — Google Earth Engine Authentication
#
# Uses a GEE service account + JSON key file for non-interactive auth.
# This is the required pattern for server / CI environments.
#
# How to obtain credentials:
#   1. Go to https://console.cloud.google.com → IAM & Admin → Service Accounts
#   2. Create a service account, download the JSON key.
#   3. Register it with GEE: https://signup.earthengine.google.com/#!/service_accounts
#   4. Place the JSON file at credentials/service_account.json
#   5. Set GEE_SERVICE_ACCOUNT_EMAIL in .env to the email in the JSON.
# =============================================================================

import logging
from pathlib import Path

import ee

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level flag — authentication runs once per process lifetime
_ee_initialized: bool = False


def initialize_ee() -> None:
    """
    Authenticates and initialises the Earth Engine Python client.

    Uses service account credentials loaded from the path defined in .env.
    Safe to call multiple times — re-initialization is skipped after the
    first successful call.

    Raises:
        FileNotFoundError: If the service account JSON key file is missing.
        ee.EEException:    If GEE authentication or initialization fails.
    """
    global _ee_initialized

    if _ee_initialized:
        logger.debug("[EE Auth] Already initialized — skipping.")
        return

    key_path = Path(settings.GEE_SERVICE_ACCOUNT_KEY)
    if not key_path.exists():
        raise FileNotFoundError(
            f"[EE Auth] Service account key not found at: {key_path.resolve()}\n"
            "Create a GEE service account, download its JSON key, and place it at "
            f"'{key_path}'. See: https://developers.google.com/earth-engine/guides/service_account"
        )

    credentials = ee.ServiceAccountCredentials(
        email=settings.GEE_SERVICE_ACCOUNT_EMAIL,
        key_file=str(key_path),
    )

    ee.Initialize(credentials)
    _ee_initialized = True
    logger.info("[EE Auth] Earth Engine initialized successfully via service account.")