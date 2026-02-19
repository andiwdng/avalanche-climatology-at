"""
download_era5.py
================
Download ERA5-Land hourly reanalysis data for the Austrian Alps domain
via the Copernicus Climate Data Store (CDS) API.

Scientific rationale
--------------------
ERA5-Land (Muñoz-Sabater et al. 2021) is a reanalysis that combines ERA5
atmospheric forcing with a land-surface model (HTESSEL) driven at finer
horizontal resolution (~9 km).  For alpine snow modelling, ERA5-Land
is preferred over ERA5 because it better represents orographic
precipitation and land-surface energy balance.

Variables downloaded
--------------------
- 2m_temperature              [K]         → SNOWPACK TA
- 2m_dewpoint_temperature     [K]         → derive relative humidity
- total_precipitation         [m hr⁻¹]   → SNOWPACK PSUM after conversion
- surface_pressure            [Pa]        → SNOWPACK P
- 10m_u_component_of_wind     [m s⁻¹]   → derive wind speed / direction
- 10m_v_component_of_wind     [m s⁻¹]   → derive wind speed / direction
- surface_solar_radiation_downwards  [J m⁻²] → SNOWPACK ISWR
- surface_thermal_radiation_downwards [J m⁻²] → SNOWPACK ILWR
- orography                   [m² s⁻²]  → native terrain height for lapse-rate correction

Temporal strategy
-----------------
Requests are batched by year to respect CDS queue limits and to allow
incremental downloads.  Files are stored per-year in data/era5_raw/.

CDS API credentials
-------------------
Place a valid ~/.cdsapirc file containing:
    url: https://cds.climate.copernicus.eu/api/v2
    key: <UID>:<API-KEY>

Reference
---------
Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., et al. (2021).
    ERA5-Land: A state-of-the-art global reanalysis dataset for land
    applications. Earth System Science Data, 13(9), 4349–4383.
    https://doi.org/10.5194/essd-13-4349-2021
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cdsapi
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Variable names for the atmo-forcing download (excluding orography)
FORCING_VARIABLES: list[str] = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
]

ALL_HOURS: list[str] = [f"{h:02d}:00" for h in range(24)]
ALL_DAYS: list[str] = [f"{d:02d}" for d in range(1, 32)]
ALL_MONTHS: list[str] = [f"{m:02d}" for m in range(1, 13)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_era5(config: dict) -> None:
    """
    Download all required ERA5-Land variables for the simulation period.

    The function requests two separate CDS retrievals per year:
    1. Hourly atmospheric forcing (all timesteps, all forcing variables).
    2. Static orography (single timestep; only downloaded once).

    Downloaded NetCDF files are written to ``config['paths']['era5_raw']``.
    Existing files are skipped to allow resumable downloads.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    """
    output_dir = Path(config["paths"]["era5_raw"])
    output_dir.mkdir(parents=True, exist_ok=True)

    era5_cfg = config["era5"]
    sim_cfg = config["simulation"]

    # Derive full download period including spin-up
    from datetime import date
    analysis_start = date.fromisoformat(sim_cfg["analysis_start"])
    analysis_end = date.fromisoformat(sim_cfg["analysis_end"])
    spin_up_years = int(sim_cfg["spin_up_years"])

    download_start_year = analysis_start.year - spin_up_years
    download_end_year = analysis_end.year

    logger.info(
        "ERA5 download range: %d – %d  (including %d spin-up year(s))",
        download_start_year,
        download_end_year,
        spin_up_years,
    )

    # Initialise CDS client
    rc_path = os.path.expanduser(era5_cfg.get("cds_api_rc", "~/.cdsapirc"))
    client = cdsapi.Client(url=None, key=None, rc_path=rc_path, quiet=True)

    # --- 1. Download static orography (once) ---
    _download_orography(client, era5_cfg, output_dir)

    # --- 2. Download forcing variables year-by-year ---
    for year in range(download_start_year, download_end_year + 1):
        _download_forcing_year(client, year, era5_cfg, output_dir)

    logger.info("ERA5-Land download complete.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _download_orography(
    client: cdsapi.Client,
    era5_cfg: dict,
    output_dir: Path,
) -> None:
    """
    Download ERA5-Land surface geopotential (orography) — a static field.

    The geopotential Φ [m² s⁻²] is divided by g = 9.80665 m s⁻² to obtain
    the geometric surface height in metres above sea level.

    Parameters
    ----------
    client : cdsapi.Client
        Authenticated CDS API client.
    era5_cfg : dict
        ERA5 section of config.yaml.
    output_dir : Path
        Directory where the orography file is saved.
    """
    orography_file = output_dir / "era5land_orography.nc"
    if orography_file.exists():
        logger.info("Orography file already exists, skipping: %s", orography_file)
        return

    logger.info("Downloading ERA5-Land orography → %s", orography_file)
    client.retrieve(
        era5_cfg["product"],
        {
            "variable": ["orography"],
            "year": "2000",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
            "area": era5_cfg["area"],
        },
        str(orography_file),
    )
    logger.info("Orography download complete.")


def _download_forcing_year(
    client: cdsapi.Client,
    year: int,
    era5_cfg: dict,
    output_dir: Path,
) -> None:
    """
    Download hourly ERA5-Land forcing variables for a single calendar year.

    CDS requests all months, days and hours at once for efficiency.
    ERA5-Land time-accumulated variables (ssrd, strd, tp) are accumulated
    since 00:00 UTC each day; de-accumulation is performed in
    :mod:`scripts.interpolate_points`.

    Parameters
    ----------
    client : cdsapi.Client
        Authenticated CDS API client.
    year : int
        Calendar year to download.
    era5_cfg : dict
        ERA5 section of config.yaml.
    output_dir : Path
        Directory where the annual file is saved.
    """
    out_file = output_dir / f"era5land_forcing_{year}.nc"
    if out_file.exists():
        logger.info("ERA5 forcing %d already downloaded, skipping.", year)
        return

    logger.info("Requesting ERA5-Land forcing for year %d …", year)

    request_body = {
        "variable": FORCING_VARIABLES,
        "year": str(year),
        "month": ALL_MONTHS,
        "day": ALL_DAYS,
        "time": ALL_HOURS,
        "format": "netcdf",
        "area": era5_cfg["area"],
    }

    client.retrieve(era5_cfg["product"], request_body, str(out_file))
    logger.info("ERA5-Land forcing %d → %s", year, out_file)


# ---------------------------------------------------------------------------
# Entry-point (for direct invocation / testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    download_era5(cfg)
