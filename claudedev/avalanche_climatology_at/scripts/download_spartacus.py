"""
download_spartacus.py
=====================
Download GeoSphere Austria SPARTACUS-v2 daily gridded temperature and
precipitation data for the Austrian Alps domain.

Scientific rationale
--------------------
SPARTACUS (Spatial and Temporal Climatology and Ranalysis for the
Austrian Climate system) is a daily gridded analysis dataset at ~1 km
resolution, produced by GeoSphere Austria (formerly ZAMG) using optimal
interpolation of the Austrian station network.

For bias correction of ERA5-Land, SPARTACUS provides:
- ``Tl``  – daily mean 2 m air temperature [°C]
- ``RR``  – daily precipitation sum [mm]

Both fields are derived from a dense network of Austrian climate stations
and provide the ground-truth signal for regional temperature offsets and
precipitation scaling factors.

API access
----------
Data are accessed via the GeoSphere Austria OpenData REST API:
    https://dataset.api.hub.zamg.ac.at/v1/grid-hist/spartacus-v2-1d-1km

No authentication is required for open-data access.
Requests are batched by year to respect server-side limitations.

Reference
---------
Hiebl, J. & Frei, C. (2016). Daily temperature grids for Austria since 1961
    — concept, creation and applicability. Theoretical and Applied Climatology,
    124(1–2), 161–177. https://doi.org/10.1007/s00704-015-1411-4

Hiebl, J. & Frei, C. (2018). Daily precipitation grids for Austria since 1961
    — development and evaluation of a spatial dataset for hydroclimatic
    research. Theoretical and Applied Climatology, 132(1–2), 327–345.
    https://doi.org/10.1007/s00704-017-2093-x
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SPARTACUS_PARAMETERS: dict[str, str] = {
    "temperature": "Tl",    # daily mean air temperature
    "precipitation": "RR",  # daily precipitation sum
}

# Maximum number of days in a single API request (server limit)
_MAX_REQUEST_DAYS: int = 365

# Pause between requests to respect rate limits
_REQUEST_PAUSE_SECONDS: float = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_spartacus(config: dict) -> None:
    """
    Download SPARTACUS-v2 daily temperature and precipitation grids.

    Downloads are batched by year and saved as NetCDF files in
    ``config['paths']['spartacus']``.  Existing files are skipped.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    """
    output_dir = Path(config["paths"]["spartacus"])
    output_dir.mkdir(parents=True, exist_ok=True)

    spartacus_cfg = config["spartacus"]
    sim_cfg = config["simulation"]

    from datetime import date, timedelta

    analysis_start = date.fromisoformat(sim_cfg["analysis_start"])
    analysis_end = date.fromisoformat(sim_cfg["analysis_end"])
    spin_up_years = int(sim_cfg["spin_up_years"])

    download_start = date(analysis_start.year - spin_up_years, 1, 1)
    download_end = date(analysis_end.year, 12, 31)

    logger.info(
        "SPARTACUS download: %s → %s", download_start.isoformat(), download_end.isoformat()
    )

    # Download one year at a time for both temperature and precipitation
    current_year = download_start.year
    end_year = download_end.year

    for year in range(current_year, end_year + 1):
        for var_key in ("temperature", "precipitation"):
            _download_spartacus_year(
                year=year,
                variable_key=var_key,
                spartacus_cfg=spartacus_cfg,
                output_dir=output_dir,
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _download_spartacus_year(
    year: int,
    variable_key: str,
    spartacus_cfg: dict,
    output_dir: Path,
) -> None:
    """
    Download a single year of SPARTACUS data for one variable.

    The GeoSphere Austria API delivers data in NetCDF format.  The response
    is written directly to disk without parsing, preserving the native
    grid for subsequent bilinear interpolation in bias_correction.py.

    Parameters
    ----------
    year : int
        Calendar year to download.
    variable_key : str
        One of ``'temperature'`` or ``'precipitation'``.
    spartacus_cfg : dict
        SPARTACUS section of config.yaml.
    output_dir : Path
        Destination directory.
    """
    parameter = _SPARTACUS_PARAMETERS[variable_key]
    out_file = output_dir / f"spartacus_{variable_key}_{year}.nc"

    if out_file.exists():
        logger.info("SPARTACUS %s %d already on disk, skipping.", variable_key, year)
        return

    api_base = spartacus_cfg["api_base"]
    bbox = spartacus_cfg["bbox"]   # [minlon, minlat, maxlon, maxlat]
    start_dt = f"{year}-01-01T00:00"
    end_dt = f"{year}-12-31T00:00"

    params = {
        "parameters": parameter,
        "start": start_dt,
        "end": end_dt,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "output_format": "netcdf",
    }

    logger.info(
        "Requesting SPARTACUS %s (%s) for %d …",
        variable_key,
        parameter,
        year,
    )

    response = _api_get_with_retry(api_base, params=params, timeout=300)

    if response.status_code != 200:
        logger.error(
            "SPARTACUS request failed for %s %d: HTTP %d — %s",
            variable_key,
            year,
            response.status_code,
            response.text[:500],
        )
        raise RuntimeError(
            f"SPARTACUS API error {response.status_code} for {variable_key} {year}"
        )

    out_file.write_bytes(response.content)
    logger.info("SPARTACUS %s %d → %s (%d kB)", variable_key, year, out_file,
                len(response.content) // 1024)

    time.sleep(_REQUEST_PAUSE_SECONDS)


def _api_get_with_retry(
    url: str,
    params: dict,
    timeout: int = 300,
    max_retries: int = 3,
) -> requests.Response:
    """
    Perform a GET request with exponential back-off retry.

    Parameters
    ----------
    url : str
        Full API endpoint URL.
    params : dict
        Query parameters.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    requests.Response
        HTTP response object.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            return response
        except requests.exceptions.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(
                "SPARTACUS GET attempt %d/%d failed: %s — retrying in %d s",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"SPARTACUS API unreachable after {max_retries} retries: {url}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    download_spartacus(cfg)
