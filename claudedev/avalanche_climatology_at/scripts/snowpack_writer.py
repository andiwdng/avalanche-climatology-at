"""
snowpack_writer.py
==================
Write SNOWPACK input files in SMET format and generate SNOWPACK INI
configuration files for flat-field point simulations.

SMET format (Snow Meteorological Exchange Tool)
-----------------------------------------------
SMET is the standard ASCII input format for SNOWPACK >= 3.0.  Files
consist of a ``[HEADER]`` section with station metadata and a ``[DATA]``
section with whitespace-delimited time-series.

Reference: Lehning et al. (2002), Computational Geosciences.
SNOWPACK User Manual: https://models.slf.ch/p/snowpack/

Meteorological fields written per timestep
-------------------------------------------
- ``timestamp``  ISO 8601 UTC datetime (e.g. 2000-10-01T00:00)
- ``TA``         Air temperature [K]
- ``RH``         Relative humidity [0–1]
- ``VW``         Wind speed [m s⁻¹]
- ``DW``         Wind direction [° from N, meteorological]
- ``ISWR``       Incoming short-wave radiation [W m⁻²]
- ``ILWR``       Incoming long-wave radiation [W m⁻²]
- ``PSUM``       Precipitation per time step [mm]
- ``PSUM_PH``    Precipitation phase [0 = snow, 1 = rain]
- ``P``          Atmospheric pressure [Pa]

Precipitation phase
-------------------
A simple temperature-based partitioning is used:
    PSUM_PH = 0   (all snow)   if TA < T_snow
    PSUM_PH = 1   (all rain)   if TA > T_rain
    PSUM_PH = (TA − T_snow) / (T_rain − T_snow)  (linear mix) otherwise
Default thresholds: T_snow = 273.65 K (0.5 °C), T_rain = 275.15 K (2 °C).

Initial snow file (.sno)
------------------------
A bare-ground initial state is written for each simulation.  SNOWPACK
starts each simulation with no snow (HS = 0 m).

SNOWPACK INI configuration
--------------------------
A complete INI file is generated for each (region, elevation_band) pair.
Simulation-relevant settings are read from config.yaml; all other
parameters use scientifically validated SNOWPACK defaults for alpine
flat-field simulations.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Precipitation phase thresholds [K]
_T_SNOW: float = 273.65   # all-snow below this temperature
_T_RAIN: float = 275.15   # all-rain above this temperature

# SMET precision format strings
_FLOAT_FMT: str = "{:.6f}"
_INT_FMT: str = "{:d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def write_all_snowpack_inputs(
    config: dict,
    corrected_era5: dict[str, dict[int, pd.DataFrame]],
) -> dict[str, dict[int, Path]]:
    """
    Write SMET meteorological input, SNO initial profile, and INI
    configuration files for every (region, elevation_band) pair.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    corrected_era5 : dict
        Bias-corrected ERA5 data: ``corrected_era5[region][elev] = DataFrame``.

    Returns
    -------
    dict
        ``ini_paths[region][elev] = Path`` to the SNOWPACK INI file
        for each simulation.
    """
    input_dir = Path(config["paths"]["snowpack_input"])
    output_dir = Path(config["paths"]["snowpack_output"])
    regions = config["regions"]
    elevation_bands = config["elevation_bands"]
    sp_cfg = config["snowpack"]
    sim_cfg = config["simulation"]

    ini_paths: dict[str, dict[int, Path]] = {}

    for region_key in regions:
        if region_key not in corrected_era5:
            logger.warning("Region '%s' missing from corrected ERA5 — skipping.", region_key)
            continue

        ini_paths[region_key] = {}

        for elev_m in elevation_bands:
            if elev_m not in corrected_era5[region_key]:
                logger.warning(
                    "Elevation %d m missing for region '%s' — skipping.", elev_m, region_key
                )
                continue

            df = corrected_era5[region_key][elev_m]
            station_id = _make_station_id(region_key, elev_m)
            region_meta = regions[region_key]

            # 1. Write SMET forcing file
            smet_dir = input_dir / "smet"
            smet_dir.mkdir(parents=True, exist_ok=True)
            smet_path = smet_dir / f"{station_id}.smet"
            write_smet(
                df=df,
                station_id=station_id,
                station_name=f"{region_meta['name']} {elev_m}m",
                lat=region_meta["lat"],
                lon=region_meta["lon"],
                altitude=float(elev_m),
                output_path=smet_path,
            )

            # 2. Write initial snow profile (.sno)
            sno_dir = input_dir / "sno"
            sno_dir.mkdir(parents=True, exist_ok=True)
            sno_path = sno_dir / f"{station_id}.sno"
            write_sno(
                station_id=station_id,
                station_name=f"{region_meta['name']} {elev_m}m",
                lat=region_meta["lat"],
                lon=region_meta["lon"],
                altitude=float(elev_m),
                slope_angle=float(sp_cfg["slope_angle"]),
                slope_aspect=float(sp_cfg["aspect"]),
                profile_date=sim_cfg["analysis_start"].replace("-10-01", "-10-01").split("T")[0],
                output_path=sno_path,
            )

            # 3. Write SNOWPACK INI
            ini_dir = input_dir / "ini"
            ini_dir.mkdir(parents=True, exist_ok=True)
            sim_output_dir = output_dir / region_key / f"{elev_m}m"
            sim_output_dir.mkdir(parents=True, exist_ok=True)

            ini_path = ini_dir / f"{station_id}.ini"
            write_snowpack_ini(
                station_id=station_id,
                smet_dir=smet_dir,
                sno_dir=sno_dir,
                output_dir=sim_output_dir,
                slope_angle=float(sp_cfg["slope_angle"]),
                slope_aspect=float(sp_cfg["aspect"]),
                calculation_step=int(sp_cfg["calculation_step_length"]),
                meteo_step=int(sp_cfg["meteo_step_length"]),
                output_path=ini_path,
                sp_cfg=sp_cfg,
            )

            ini_paths[region_key][elev_m] = ini_path
            logger.info("SNOWPACK input ready: %s", station_id)

    return ini_paths


# ---------------------------------------------------------------------------
# SMET writer
# ---------------------------------------------------------------------------
def write_smet(
    df: pd.DataFrame,
    station_id: str,
    station_name: str,
    lat: float,
    lon: float,
    altitude: float,
    output_path: Path,
) -> None:
    """
    Write a SNOWPACK SMET meteorological forcing file.

    Parameters
    ----------
    df : pd.DataFrame
        Bias-corrected hourly ERA5 DataFrame with columns:
        TA, RH, VW, DW, ISWR, ILWR, PSUM, P.
    station_id : str
        Short alphanumeric station identifier.
    station_name : str
        Human-readable station name.
    lat : float
        Station latitude [decimal degrees North].
    lon : float
        Station longitude [decimal degrees East].
    altitude : float
        Station elevation [m a.s.l.].
    output_path : Path
        Destination SMET file path.
    """
    df = df.copy()

    # Compute precipitation phase
    df["PSUM_PH"] = _compute_precip_phase(df["TA"].values)

    # PSUM: convert from mm h⁻¹ to mm (per timestep = 1 h), clip negative
    df["PSUM"] = df["PSUM"].clip(lower=0.0)

    # RH: ensure 0–1 range
    df["RH"] = df["RH"].clip(0.01, 1.0)

    # ISWR / ILWR: cannot be negative
    df["ISWR"] = df["ISWR"].clip(lower=0.0)
    df["ILWR"] = df["ILWR"].clip(lower=0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        # Header
        fh.write("SMET 1.1 ASCII\n")
        fh.write("[HEADER]\n")
        fh.write(f"station_id   = {station_id}\n")
        fh.write(f"station_name = {station_name}\n")
        fh.write(f"latitude     = {lat:.6f}\n")
        fh.write(f"longitude    = {lon:.6f}\n")
        fh.write(f"altitude     = {altitude:.1f}\n")
        fh.write("nodata       = -999\n")
        fh.write("tz           = 0\n")
        fh.write("source       = ERA5-Land-SPARTACUS-BiasCorrection\n")
        fh.write("fields       = timestamp TA RH VW DW ISWR ILWR PSUM PSUM_PH P\n")
        fh.write("[DATA]\n")

        # Data rows
        for ts, row in df.iterrows():
            # Format timestamp to ISO 8601 without seconds
            if isinstance(ts, pd.Timestamp):
                ts_str = ts.strftime("%Y-%m-%dT%H:%M")
            else:
                ts_str = str(ts)[:16]

            line = (
                f"{ts_str} "
                f"{row['TA']:.4f} "
                f"{row['RH']:.4f} "
                f"{row['VW']:.4f} "
                f"{row['DW']:.1f} "
                f"{row['ISWR']:.4f} "
                f"{row['ILWR']:.4f} "
                f"{row['PSUM']:.6f} "
                f"{row['PSUM_PH']:.4f} "
                f"{row['P']:.2f}\n"
            )
            fh.write(line)

    logger.debug("SMET written: %s (%d records)", output_path.name, len(df))


# ---------------------------------------------------------------------------
# Initial snow profile (.sno) writer
# ---------------------------------------------------------------------------
def write_sno(
    station_id: str,
    station_name: str,
    lat: float,
    lon: float,
    altitude: float,
    slope_angle: float,
    slope_aspect: float,
    profile_date: str,
    output_path: Path,
) -> None:
    """
    Write a bare-ground SNOWPACK initial profile file (.sno).

    The simulation starts with no snow (HS = 0).  SNOWPACK will build
    up the snowpack from scratch during the first season.

    Parameters
    ----------
    station_id : str
        Station identifier.
    station_name : str
        Human-readable name.
    lat, lon : float
        Coordinates.
    altitude : float
        Elevation [m].
    slope_angle : float
        Slope inclination [°].  0 for flat-field.
    slope_aspect : float
        Slope aspect [°].
    profile_date : str
        ISO 8601 date string for profile initialisation (typically
        one day before simulation start).
    output_path : Path
        Destination .sno file path.
    """
    # Initialise one day before the simulation start for spin-up
    from datetime import date, timedelta
    pd_date = date.fromisoformat(profile_date)
    profile_timestamp = f"{pd_date.isoformat()}T00:00"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = textwrap.dedent(f"""\
        SMET 1.1 ASCII
        [HEADER]
        station_id               = {station_id}
        station_name             = {station_name}
        latitude                 = {lat:.6f}
        longitude                = {lon:.6f}
        altitude                 = {altitude:.1f}
        nodata                   = -999
        tz                       = 0
        source                   = Generated
        ProfileDate              = {profile_timestamp}
        HS_Last                  = 0.000000
        SlopeAngle               = {slope_angle:.6f}
        SlopeAzi                 = {slope_aspect:.6f}
        nSoilLayerData           = 0
        nSnowLayerData           = 0
        SoilAlbedo               = 0.090000
        BareSoil_z0              = 0.020000
        CanopyHeight             = 0.000000
        CanopyLeafAreaIndex      = 0.000000
        CanopyDirectThroughfall  = 1.000000
        WindScalingFactor        = 1.000000
        ErosionLevel             = 0
        TimeCountDeltaHS         = 0.000000
        fields = timestamp Layer_Thick T Vol_Frac_I Vol_Frac_W Vol_Frac_V Vol_Frac_S R ho_S Conduc_S HeatCapac_S rg rb dd sp mk mass_hoar ne CDot metamo
        [DATA]
    """)

    output_path.write_text(content, encoding="utf-8")
    logger.debug("SNO written: %s", output_path.name)


# ---------------------------------------------------------------------------
# SNOWPACK INI writer
# ---------------------------------------------------------------------------
def write_snowpack_ini(
    station_id: str,
    smet_dir: Path,
    sno_dir: Path,
    output_dir: Path,
    slope_angle: float,
    slope_aspect: float,
    calculation_step: int,
    meteo_step: int,
    output_path: Path,
    sp_cfg: dict,
) -> None:
    """
    Write a SNOWPACK INI configuration file for a flat-field simulation.

    All paths in the INI are written as absolute paths to ensure SNOWPACK
    can be invoked from any working directory.

    Parameters
    ----------
    station_id : str
        Station/simulation identifier (must match SMET ``station_id``).
    smet_dir : Path
        Directory containing SMET input files.
    sno_dir : Path
        Directory containing SNO initial profile files.
    output_dir : Path
        Directory for SNOWPACK output (profiles, met, snow files).
    slope_angle : float
        Slope angle [°].  0 for flat-field.
    slope_aspect : float
        Slope aspect [°].
    calculation_step : int
        Internal SNOWPACK timestep [minutes].
    meteo_step : int
        Meteorological input timestep [minutes].
    output_path : Path
        Destination INI file path.
    sp_cfg : dict
        SNOWPACK section of config.yaml.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PSUM accumulation period in seconds
    psum_period_s = meteo_step * 60

    ini_content = textwrap.dedent(f"""\
        ; SNOWPACK configuration — flat-field avalanche climatology
        ; Station : {station_id}
        ; Generated by avalanche_climatology_at pipeline

        [GENERAL]
        BUFF_CHUNK_SIZE     = 370
        BUFF_BEFORE         = 1.5

        [INPUT]
        COORDSYS            = UTM
        COORDPARAM          = 33T
        TIME_ZONE           = 0
        METEO               = SMET
        METEOPATH           = {smet_dir.resolve()}
        STATION1            = {station_id}
        ISWR_IS_NET         = FALSE
        TSG::create         = CST
        TSG::CST::VALUE     = 273.15
        SNOW                = SMET
        SNOWPATH            = {sno_dir.resolve()}
        SNOWFILE1           = {station_id}.sno

        [OUTPUT]
        COORDSYS            = UTM
        COORDPARAM          = 33T
        TIME_ZONE           = 0
        METEOPATH           = {output_dir.resolve()}
        SNOWPATH            = {output_dir.resolve()}
        PROF_WRITE          = TRUE
        PROF_FORMAT         = PRO
        PROF_START          = 0
        PROF_DAYS_BETWEEN   = 1.0
        SNOW_WRITE          = TRUE
        AGGREGATE_PRO       = FALSE
        OUT_CANOPY          = FALSE
        OUT_HAZ             = TRUE
        OUT_SOILEB          = FALSE
        OUT_HEAT            = FALSE
        OUT_T               = FALSE
        OUT_LW              = FALSE
        OUT_SW              = FALSE
        OUT_MASS            = FALSE
        OUT_METEO           = FALSE
        OUT_STAB            = TRUE
        TS_WRITE            = FALSE

        [SNOWPACK]
        CALCULATION_STEP_LENGTH       = {calculation_step}
        ROUGHNESS_LENGTH              = 0.002
        HEIGHT_OF_METEO_VALUES        = 2.0
        HEIGHT_OF_WIND_VALUE          = 10.0
        ENFORCE_MEASURED_SNOW_HEIGHTS = FALSE
        SW_MODE                       = INCOMING
        ATMOSPHERIC_STABILITY         = NEUTRAL
        CANOPY                        = FALSE
        MEAS_TSS                      = FALSE
        CHANGE_BC                     = FALSE
        SNP_SOIL                      = FALSE

        [SNOWPACKADVANCED]
        ASSUME_RESPONSIBILITY         = AGREE
        VARIANT                       = DEFAULT
        ADJUST_HEIGHT_OF_METEO_VALUES = FALSE
        ADJUST_HEIGHT_OF_WIND_VALUE   = FALSE
        SNOW_EROSION                  = FALSE
        NUMBER_SLOPES                 = 1
        SNOW_REDISTRIBUTION           = FALSE
        PREVAILING_WIND_DIR           = 0
        MEAS_INCOMING_LONGWAVE        = FALSE
        PERP_TO_SLOPE                 = FALSE
        ALLOW_ADAPTIVE_TIMESTEPPING   = TRUE
        THRESH_RAIN                   = 1.9
        FORCE_RH_WATER                = TRUE
        THRESH_RH                     = 0.5
        THRESH_DTEMP_AIR_SNOW         = 3.0
        METAMORPHISM_MODEL            = DEFAULT
        VISCOSITY_MODEL               = DEFAULT
        WATER_LAYER                   = FALSE
        WATERTRANSPORTMODEL_SNOW      = BUCKET
        ALBEDO_AGING                  = TRUE
        COMBINE_ELEMENTS              = TRUE
        HEIGHT_NEW_ELEM               = 0.02
        MINIMUM_L_ELEMENT             = 0.0025

        [STABILITY]
        STABILITY_MODEL      = NIED
        ASSUME_ALWAYS_STABLE = FALSE

        [FILTERS]
        TA::filter1         = min_max
        TA::arg1::MIN       = 240
        TA::arg1::MAX       = 320
        RH::filter1         = min_max
        RH::arg1::MIN       = 0.01
        RH::arg1::MAX       = 1.2
        RH::filter2         = min_max
        RH::arg2::SOFT      = TRUE
        RH::arg2::MIN       = 0.05
        RH::arg2::MAX       = 1.0
        VW::filter1         = min_max
        VW::arg1::MIN       = 0
        VW::arg1::MAX       = 70
        ISWR::filter1       = min_max
        ISWR::arg1::MIN     = -10
        ISWR::arg1::MAX     = 1500
        ISWR::filter2       = min_max
        ISWR::arg2::SOFT    = TRUE
        ISWR::arg2::MIN     = 0
        ISWR::arg2::MAX     = 1500
        ILWR::filter1       = min_max
        ILWR::arg1::MIN     = 180
        ILWR::arg1::MAX     = 600
        PSUM::filter1       = min
        PSUM::arg1::SOFT    = TRUE
        PSUM::arg1::MIN     = 0.0

        [INTERPOLATIONS1D]
        ENABLE_RESAMPLING           = TRUE
        WINDOW_SIZE                 = 86400
        TA::resample                = linear
        TA::linear::window_size     = 86400
        TA::linear::extrapolate     = true
        RH::resample                = linear
        RH::linear::window_size     = 86400
        RH::linear::extrapolate     = true
        VW::resample                = linear
        VW::linear::window_size     = 86400
        VW::linear::extrapolate     = true
        DW::resample                = nearest
        DW::nearest::window_size    = 86400
        DW::nearest::extrapolate    = true
        ISWR::resample              = linear
        ISWR::linear::window_size   = 86400
        ISWR::linear::extrapolate   = true
        ILWR::resample              = linear
        ILWR::linear::window_size   = 86400
        ILWR::linear::extrapolate   = true
        PSUM::resample              = linear
        PSUM::linear::window_size   = 86400
        PSUM::linear::extrapolate   = true
    """)

    output_path.write_text(ini_content, encoding="utf-8")
    logger.debug("INI written: %s", output_path.name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_precip_phase(ta: np.ndarray) -> np.ndarray:
    """
    Compute precipitation phase (PSUM_PH) from air temperature.

    Returns 0 (solid/snow) for TA ≤ T_SNOW, 1 (liquid/rain) for TA ≥ T_RAIN,
    and a linear interpolation in between.

    Parameters
    ----------
    ta : np.ndarray
        Air temperature [K].

    Returns
    -------
    np.ndarray
        Phase values in [0, 1].
    """
    phase = np.zeros_like(ta, dtype=float)
    mixed_mask = (ta > _T_SNOW) & (ta < _T_RAIN)
    phase[ta >= _T_RAIN] = 1.0
    phase[mixed_mask] = (ta[mixed_mask] - _T_SNOW) / (_T_RAIN - _T_SNOW)
    return phase


def _make_station_id(region_key: str, elevation_m: int) -> str:
    """
    Construct a compact, filesystem-safe station identifier.

    Parameters
    ----------
    region_key : str
        Region key from config.yaml.
    elevation_m : int
        Elevation band [m].

    Returns
    -------
    str
        Station ID, e.g. ``'bregenzerwald_2000m'``.
    """
    return f"{region_key}_{elevation_m}m"


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    print("snowpack_writer: module loaded successfully.")
    print("Station ID example:", _make_station_id("bregenzerwald", 2000))
