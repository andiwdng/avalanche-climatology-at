# snowpack_steiermark/scripts/sno_writer.py
"""
Generate an empty SNO initial profile file for SNOWPACK (no snow on ground).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_empty_sno(
    config: dict,
    station: dict,
    simulation_start: datetime,
    force: bool = False,
) -> Path:
    """
    Write an empty SNO profile file for SNOWPACK initialisation.

    If the file already exists and ``restart_from_sno`` is True in config,
    the existing file is kept unless ``force=True``.

    Parameters
    ----------
    config : dict
        Parsed config.yaml content.
    station : dict
        Station entry from config["stations"] list.
    simulation_start : datetime
        Date/time used as the profile date in the SNO file.
    force : bool
        If True, overwrite an existing SNO file.

    Returns
    -------
    Path
        Path to the SNO file.
    """
    sno_dir = Path(config["paths"]["data"]) / "sno"
    sno_dir.mkdir(parents=True, exist_ok=True)
    sno_path = sno_dir / f"{station['snow_station']}.sno"

    restart = config.get("simulation", {}).get("restart_from_sno", True)
    if sno_path.exists() and restart and not force:
        logger.info("SNO file exists and restart_from_sno=True, keeping: %s", sno_path)
        return sno_path

    profile_date = simulation_start.strftime("%Y-%m-%dT%H:%M:%S")

    content = f"""SMET 1.1 ASCII
[HEADER]
station_id       = {station["snow_station"]}
station_name     = {station["name"]}
latitude         = {station["latitude"]}
longitude        = {station["longitude"]}
altitude         = {station["altitude"]}
nodata           = -999
tz               = 0
ProfileDate      = {profile_date}
HS_Last          = 0.000000
SlopeAngle       = 0.000000
SlopeAzi         = 0.000000
nSoilLayerData   = 0
nSnowLayerData   = 0
SoilAlbedo       = 0.090000
BareSoil_z0      = 0.020000
CanopyHeight     = 0.000000
CanopyLeafAreaIndex = 0.000000
CanopyDirectThroughfall = 1.000000
WindScalingFactor = 1.000000
ErosionLevel     = 0
TimeCountDeltaHS = 0.000000
fields           = timestamp Layer_Thick  T  Vol_Frac_I  Vol_Frac_W  Vol_Frac_WP  Vol_Frac_A  Vol_Frac_S  Rho_S  Conduc_S  HeatCapac_S  rg  rb  dd  sp  mk  mass_hoar  ne  CDot  metamo
[DATA]
"""

    with open(sno_path, "w") as fh:
        fh.write(content)

    logger.info("Written empty SNO file: %s (ProfileDate=%s)", sno_path, profile_date)
    return sno_path
