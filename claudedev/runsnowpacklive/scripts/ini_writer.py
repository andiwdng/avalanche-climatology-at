# snowpack_steiermark/scripts/ini_writer.py
"""
Generate the SNOWPACK INI file for a given station.

Uses the station-validated configuration as a template, substituting file paths,
simulation period, and per-station flags (MEAS_INCOMING_LONGWAVE, TSG generator).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_ini(
    config: dict,
    station: dict,
    smet_path: Path,
    sno_path: Path,
    pro_dir: Path,
    start_date: datetime,
    end_date: datetime,
) -> Path:
    """
    Write the SNOWPACK INI file for the given station.

    Parameters
    ----------
    config : dict
        Parsed config.yaml content.
    station : dict
        Station entry from config["stations"] list.
    smet_path : Path
        Path to the SMET forcing file.
    sno_path : Path
        Path to the initial SNO profile.
    pro_dir : Path
        Directory where SNOWPACK writes PRO / TS output.
    start_date : datetime
        Simulation start.
    end_date : datetime
        Simulation end.

    Returns
    -------
    Path
        Path to the written INI file.
    """
    ini_dir = Path(config["paths"]["data"]) / "ini"
    ini_dir.mkdir(parents=True, exist_ok=True)
    ini_path = ini_dir / f"{station['id'].lower()}.ini"

    smet_dir = smet_path.resolve().parent
    sno_dir = sno_path.resolve().parent
    pro_dir_abs = pro_dir.resolve()
    pro_dir_abs.mkdir(parents=True, exist_ok=True)

    smet_name = smet_path.stem  # e.g. "TAMI2"
    sno_name = sno_path.stem    # e.g. "TAMI2"

    # Per-station flags
    meas_ilwr = "TRUE" if station.get("ilwr") else "FALSE"

    # TSG generator: only add when station does NOT have measured TSG
    tsg_generator_lines = ""
    if not station.get("tsg"):
        tsg_generator_lines = """\
TSG::create         =    CST
TSG::CST::VALUE     =    273.15
"""

    # Extra filters for ILWR and TSG when station has measured ILWR
    ilwr_tsg_filter_lines = ""
    if station.get("ilwr"):
        ilwr_tsg_filter_lines = """\
ILWR::filter1      =    min_max
ILWR::arg1::SOFT   =    TRUE
ILWR::arg1::MIN    =    50
ILWR::arg1::MAX    =    600
TSG::filter1       =    min_max
TSG::arg1::SOFT    =    TRUE
TSG::arg1::MIN     =    200
TSG::arg1::MAX     =    310
"""

    content = f"""\
[GENERAL]
BUFF_CHUNK_SIZE    =    370
BUFF_BEFORE        =    1.5

[INPUT]
COORDSYS    =    UTM
COORDPARAM  =    33T
TIME_ZONE   =    0
METEO       =    SMET
METEOPATH   =    {smet_dir}
STATION1    =    {smet_name}.smet
ISWR_IS_NET =    FALSE
SNOWPATH    =    {sno_dir}
SNOW        =    SMET
SNOWFILE1   =    {sno_name}.sno
{tsg_generator_lines}
[OUTPUT]
COORDSYS             =    UTM
COORDPARAM           =    33T
TIME_ZONE            =    0
METEOPATH            =    {pro_dir_abs}
WRITE_PROCESSED_METEO =   TRUE
EXPERIMENT           =    {station['id']}
SNOW                 =    SMET
SNOWPATH             =    {pro_dir_abs}
BACKUP_DAYS_BETWEEN  =    1
FIRST_BACKUP         =    1
PROF_WRITE           =    TRUE
PROF_FORMAT          =    PRO
SNOW_WRITE           =    TRUE
AGGREGATE_PRO        =    FALSE
AGGREGATE_PRF        =    FALSE
PROF_START           =    0
PROF_DAYS_BETWEEN    =    0.0104167
HARDNESS_IN_NEWTON   =    FALSE
CLASSIFY_PROFILE     =    TRUE
TS_WRITE             =    TRUE
TS_FORMAT            =    SMET
TS_START             =    0
TS_DAYS_BETWEEN      =    0.0104167
CUMSUM_MASS          =    FALSE
PRECIP_RATES         =    TRUE
OUT_CANOPY           =    FALSE
OUT_HAZ              =    TRUE
OUT_SOILEB           =    FALSE
OUT_HEAT             =    TRUE
OUT_T                =    TRUE
OUT_LW               =    TRUE
OUT_SW               =    TRUE
OUT_MASS             =    TRUE
OUT_METEO            =    TRUE
OUT_STAB             =    TRUE

[SNOWPACK]
CALCULATION_STEP_LENGTH       =    15
ROUGHNESS_LENGTH               =    0.002
HEIGHT_OF_METEO_VALUES         =    4
HEIGHT_OF_WIND_VALUE           =    5
ENFORCE_MEASURED_SNOW_HEIGHTS  =    TRUE
SW_MODE                        =    INCOMING
ATMOSPHERIC_STABILITY          =    NEUTRAL
CANOPY                         =    FALSE
MEAS_TSS                       =    FALSE
CHANGE_BC                      =    TRUE
THRESH_CHANGE_BC               =    -1.2
SNP_SOIL                       =    FALSE

[SNOWPACKADVANCED]
ASSUME_RESPONSIBILITY          =    AGREE
VARIANT                        =    DEFAULT
ADJUST_HEIGHT_OF_METEO_VALUES  =    TRUE
ADJUST_HEIGHT_OF_WIND_VALUE    =    TRUE
SNOW_EROSION                   =    TRUE
WIND_SCALING_FACTOR            =    1
NUMBER_SLOPES                  =    1
SNOW_REDISTRIBUTION            =    FALSE
PREVAILING_WIND_DIR            =    0
MEAS_INCOMING_LONGWAVE         =    {meas_ilwr}
PERP_TO_SLOPE                  =    FALSE
ALLOW_ADAPTIVE_TIMESTEPPING    =    TRUE
THRESH_RAIN                    =    1.2
FORCE_RH_WATER                 =    TRUE
THRESH_RH                      =    0.5
THRESH_DTEMP_AIR_SNOW          =    3
HOAR_THRESH_TA                 =    1.2
HOAR_THRESH_RH                 =    0.97
HOAR_THRESH_VW                 =    3.5
HOAR_DENSITY_BURIED            =    125
HOAR_MIN_SIZE_BURIED           =    2
HOAR_DENSITY_SURF              =    100
MIN_DEPTH_SUBSURF              =    0.07
T_CRAZY_MIN                    =    210
T_CRAZY_MAX                    =    340
METAMORPHISM_MODEL             =    DEFAULT
NEW_SNOW_GRAIN_SIZE            =    0.3
STRENGTH_MODEL                 =    DEFAULT
VISCOSITY_MODEL                =    DEFAULT
SALTATION_MODEL                =    SORENSEN
WATERTRANSPORTMODEL_SNOW       =    BUCKET
WATERTRANSPORTMODEL_SOIL       =    BUCKET
ALBEDO_AGING                   =    TRUE
SW_ABSORPTION_SCHEME           =    MULTI_BAND
HARDNESS_PARAMETERIZATION      =    MONTI
DETECT_GRASS                   =    TRUE
PLASTIC                        =    FALSE
JAM                            =    FALSE
WATER_LAYER                    =    FALSE
HEIGHT_NEW_ELEM                =    0.02
MINIMUM_L_ELEMENT              =    0.0025
COMBINE_ELEMENTS               =    TRUE
ADVECTIVE_HEAT                 =    FALSE
SSI_IS_RTA                     =    TRUE
MULTI_LAYER_SK38               =    TRUE

[FILTERS]
TA::filter1        =    min_max
TA::arg1::SOFT     =    TRUE
TA::arg1::MIN      =    240
TA::arg1::MAX      =    320
RH::filter1        =    min_max
RH::arg1::MIN      =    0.01
RH::arg1::MAX      =    1.2
RH::filter2        =    min_max
RH::arg2::SOFT     =    TRUE
RH::arg2::MIN      =    0.05
RH::arg2::MAX      =    1.0
VW::filter1        =    min_max
VW::arg1::MIN      =    0
VW::arg1::MAX      =    70
VW::filter2        =    min_max
VW::arg2::SOFT     =    TRUE
VW::arg2::MIN      =    0.2
VW::arg2::MAX      =    50
ISWR::filter1      =    min_max
ISWR::arg1::MIN    =    -10
ISWR::arg1::MAX    =    1500
ISWR::filter2      =    min_max
ISWR::arg2::SOFT   =    TRUE
ISWR::arg2::MIN    =    0
ISWR::arg2::MAX    =    1500
PSUM::filter1      =    min
PSUM::arg1::SOFT   =    TRUE
PSUM::arg1::MIN    =    0.0
HS::filter1        =    min
HS::arg1::SOFT     =    TRUE
HS::arg1::MIN      =    0.0
HS::filter2        =    rate
HS::arg2::MAX      =    5.55e-5
HS::filter3        =    mad
HS::arg3::SOFT     =    TRUE
HS::arg3::CENTERING     =    left
HS::arg3::MIN_PTS  =    10
HS::arg3::MIN_SPAN =    21600
{ilwr_tsg_filter_lines}
[INTERPOLATIONS1D]
ENABLE_RESAMPLING    =    TRUE
WINDOW_SIZE          =    864000
TA::resample                   =    linear
TA::linear::window_size        =    864000
TA::linear::extrapolate        =    true
RH::resample                   =    linear
RH::linear::window_size        =    864000
RH::linear::extrapolate        =    true
VW::resample                   =    linear
VW::linear::window_size        =    864000
VW::linear::extrapolate        =    true
DW::resample                   =    nearest
DW::nearest::window_size       =    864000
DW::nearest::extrapolate       =    true
ISWR::resample                 =    linear
ISWR::linear::window_size      =    864000
ISWR::linear::extrapolate      =    true
ILWR::resample                 =    linear
ILWR::linear::window_size      =    864000
ILWR::linear::extrapolate      =    true
PSUM::resample                 =    accumulate
PSUM::accumulate::PERIOD       =    900
HS::resample                   =    nearest
HS::nearest::window_size       =    864000
HS::nearest::extrapolate       =    true

[GENERATORS]
ISWR::generators          =    ISWR_ALBEDO ALLSKY_SW
RSWR::generators          =    ISWR_ALBEDO
ILWR::generators          =    ALLSKY_LW CLEARSKY_LW
ILWR::allsky_lw::type     =    Unsworth
ILWR::clearsky_lw::type   =    Brutsaert
"""

    with open(ini_path, "w") as fh:
        fh.write(content)

    logger.info("Written INI: %s (%s → %s)", ini_path,
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return ini_path
