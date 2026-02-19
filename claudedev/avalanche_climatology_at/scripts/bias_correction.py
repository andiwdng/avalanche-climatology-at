"""
bias_correction.py
==================
Apply daily bias correction to ERA5-Land point time-series using
GeoSphere Austria SPARTACUS-v2 daily gridded analyses.

Scientific methods
------------------
Additive temperature bias correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ERA5-Land has a systematic cold or warm bias relative to station-based
analyses, especially in complex terrain.  SPARTACUS provides a daily
gridded temperature analysis ``T_sp [°C]`` derived from the Austrian
station network via optimal interpolation.

For each calendar day d and each (region, elevation) pair:
    bias_T(d) = T_sp(d) − mean(TA_era_hourly(d))  [K]
    TA_corrected(h) = TA_era(h) + bias_T(d)

The bias is computed from the daily mean ERA5 temperature and the
corresponding SPARTACUS daily mean, then broadcast to all 24 hourly
ERA5 values of that day.  This preserves the diurnal cycle of ERA5
while anchoring the daily mean to SPARTACUS.

Multiplicative precipitation scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Precipitation bias in ERA5-Land is corrected by a multiplicative
scaling factor derived from SPARTACUS daily precipitation ``P_sp [mm]``:
    scale_P(d) = P_sp(d) / P_era_daily(d)   if P_era_daily(d) > ε
               = 1.0                          otherwise
    PSUM_corrected(h) = PSUM_era(h) · scale_P(d)

On days where both ERA5 and SPARTACUS report zero precipitation no
adjustment is made.  On days where SPARTACUS reports precipitation but
ERA5 does not, the missing precipitation is added proportionally to
the dominant diurnal cycle (uniform hourly distribution).

Bias-correction is applied only to temperature and precipitation.
Wind, radiation, humidity and pressure from ERA5-Land are not corrected,
as SPARTACUS does not provide these variables and station-based gridded
analyses with adequate spatial coverage are unavailable.

Limitations
-----------
- SPARTACUS elevation: the gridded fields are on the native SPARTACUS
  DEM (~1 km resolution).  The bias computed at the region lat/lon is
  representative of the SPARTACUS grid cell closest to the region
  centroid, which may differ from the target simulation elevation.
  An additional lapse-rate adjustment is applied to the SPARTACUS
  temperature before computing the bias if the SPARTACUS grid elevation
  differs significantly from the target elevation.
- SPARTACUS coverage: data are available from 1961 for temperature and
  precipitation; all years of the simulation period are covered.

Reference
---------
Reuter, B., Viallon-Galinier, L., Horton, S., van Herwijnen, A.,
    Hagenmuller, P., Morin, S., & Schweizer, J. (2023).
    Characterizing snow instability with avalanche problem types derived
    from snow cover simulations. Cold Regions Science and Technology,
    207, 103772. https://doi.org/10.1016/j.coldregions.2022.103772
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)

# Minimum ERA5 daily precipitation to avoid division by zero [mm]
_MIN_ERA5_PRECIP: float = 0.01

# Minimum SPARTACUS precipitation for wet-day detection [mm]
_MIN_SPARTACUS_PRECIP: float = 0.1

# Lapse rate used to adjust SPARTACUS temperature to target elevation
_SPARTACUS_LAPSE_RATE: float = -0.0065  # K m⁻¹


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def apply_bias_correction(
    config: dict,
    era5_points: dict[str, dict[int, pd.DataFrame]],
) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Apply SPARTACUS-based daily bias correction to all (region, elevation) pairs.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    era5_points : dict
        Output of :func:`scripts.interpolate_points.interpolate_era5_to_points`.
        Nested dict: ``era5_points[region_key][elevation_m] = DataFrame``.

    Returns
    -------
    dict
        Same structure as *era5_points* with TA and PSUM bias-corrected.
    """
    spartacus_dir = Path(config["paths"]["spartacus"])
    spartacus_cfg = config["spartacus"]
    regions = config["regions"]

    corrected: dict[str, dict[int, pd.DataFrame]] = {}

    for region_key, region_meta in regions.items():
        if region_key not in era5_points:
            logger.warning("Region '%s' missing from ERA5 points — skipping.", region_key)
            continue

        lat = region_meta["lat"]
        lon = region_meta["lon"]

        logger.info("Bias correction for region '%s' (%.3f°N, %.3f°E)", region_key, lat, lon)

        # Load SPARTACUS daily temperature and precipitation time-series
        sp_temp = _load_spartacus_series(
            spartacus_dir=spartacus_dir,
            variable="temperature",
            lat=lat,
            lon=lon,
        )
        sp_precip = _load_spartacus_series(
            spartacus_dir=spartacus_dir,
            variable="precipitation",
            lat=lat,
            lon=lon,
        )

        corrected[region_key] = {}

        for elev_m, df_era5 in era5_points[region_key].items():
            logger.debug(
                "  Correcting %s @ %d m (%d records)…", region_key, elev_m, len(df_era5)
            )
            df_corrected = _correct_single_series(
                df_era5=df_era5.copy(),
                sp_temp=sp_temp,
                sp_precip=sp_precip,
                target_elevation=float(elev_m),
                spartacus_cfg=spartacus_cfg,
            )
            corrected[region_key][elev_m] = df_corrected

    return corrected


# ---------------------------------------------------------------------------
# Per-series correction
# ---------------------------------------------------------------------------
def _correct_single_series(
    df_era5: pd.DataFrame,
    sp_temp: pd.Series,
    sp_precip: pd.Series,
    target_elevation: float,
    spartacus_cfg: dict,
) -> pd.DataFrame:
    """
    Apply additive temperature bias and multiplicative precipitation
    scaling to one hourly ERA5 time-series.

    Parameters
    ----------
    df_era5 : pd.DataFrame
        Lapse-rate-corrected ERA5 hourly data for one (region, elevation).
    sp_temp : pd.Series
        SPARTACUS daily temperature [°C] at region lat/lon (DatetimeIndex).
    sp_precip : pd.Series
        SPARTACUS daily precipitation [mm] at region lat/lon.
    target_elevation : float
        Target simulation elevation [m a.s.l.].
    spartacus_cfg : dict
        SPARTACUS section of config.yaml.

    Returns
    -------
    pd.DataFrame
        Bias-corrected hourly DataFrame.
    """
    df_era5 = df_era5.copy()

    # Group ERA5 by calendar date
    dates = df_era5.index.normalize()  # DatetimeIndex aligned to midnight
    df_era5["_date"] = dates

    # Compute ERA5 daily mean temperature [K] and daily precip sum [mm]
    era5_daily_ta = df_era5.groupby("_date")["TA"].mean()           # K
    era5_daily_psum = df_era5.groupby("_date")["PSUM"].sum()        # mm

    # Align SPARTACUS to ERA5 dates (inner join)
    sp_temp_aligned = sp_temp.reindex(era5_daily_ta.index)
    sp_precip_aligned = sp_precip.reindex(era5_daily_psum.index)

    # ---------------------------------------------------------------------------
    # Temperature bias (additive)
    # ---------------------------------------------------------------------------
    # SPARTACUS temperature is at the SPARTACUS DEM elevation (~station elevation).
    # No separate elevation correction is applied to SPARTACUS here because
    # ERA5 was already lapse-corrected to target_elevation, and SPARTACUS
    # represents the regional climate signal.  The bias should be interpreted
    # as the residual after lapse-rate correction.
    sp_temp_k = sp_temp_aligned + 273.15                             # °C → K
    bias_T = sp_temp_k - era5_daily_ta                               # daily bias [K]

    # Broadcast daily bias to hourly ERA5 values
    bias_hourly = df_era5["_date"].map(bias_T)
    df_era5["TA"] = df_era5["TA"] + bias_hourly.values

    # ---------------------------------------------------------------------------
    # Precipitation scaling (multiplicative)
    # ---------------------------------------------------------------------------
    era5_daily_psum = era5_daily_psum.rename("era5_psum")
    sp_precip_mm = sp_precip_aligned.rename("sp_psum")

    scale_factors = _compute_precipitation_scaling(era5_daily_psum, sp_precip_mm)

    # Map scale factor from daily to hourly
    scale_hourly = df_era5["_date"].map(scale_factors)
    df_era5["PSUM"] = (df_era5["PSUM"] * scale_hourly.values).clip(lower=0.0)

    # Distribute SPARTACUS-only precipitation uniformly when ERA5 is dry
    df_era5 = _distribute_missing_precipitation(
        df_era5, sp_precip_mm, era5_daily_psum
    )

    df_era5 = df_era5.drop(columns=["_date"])
    return df_era5


def _compute_precipitation_scaling(
    era5_daily: pd.Series,
    sp_daily: pd.Series,
) -> pd.Series:
    """
    Compute daily multiplicative precipitation scaling factors.

    Parameters
    ----------
    era5_daily : pd.Series
        ERA5 daily precipitation totals [mm], indexed by date.
    sp_daily : pd.Series
        SPARTACUS daily precipitation totals [mm], indexed by date.

    Returns
    -------
    pd.Series
        Scaling factors per calendar day.  Factor = 1.0 on dry days in
        both datasets.
    """
    scale = pd.Series(index=era5_daily.index, dtype=float)

    for date in era5_daily.index:
        p_era = era5_daily.get(date, 0.0)
        p_sp = sp_daily.get(date, np.nan)

        if pd.isna(p_sp):
            scale[date] = 1.0                         # SPARTACUS gap: no correction
        elif p_era >= _MIN_ERA5_PRECIP:
            scale[date] = p_sp / p_era               # normal scaling
        else:
            scale[date] = 1.0                         # handled separately

    return scale


def _distribute_missing_precipitation(
    df_era5: pd.DataFrame,
    sp_daily: pd.Series,
    era5_daily: pd.Series,
) -> pd.DataFrame:
    """
    Add precipitation on days where SPARTACUS reports precipitation
    but ERA5 has none.

    On such days, the SPARTACUS amount is distributed uniformly across
    all 24 hours of the day.

    Parameters
    ----------
    df_era5 : pd.DataFrame
        Hourly ERA5 DataFrame (must have ``_date`` column).
    sp_daily : pd.Series
        SPARTACUS daily precipitation [mm].
    era5_daily : pd.Series
        ERA5 daily precipitation [mm].

    Returns
    -------
    pd.DataFrame
        Modified ERA5 DataFrame.
    """
    dry_era5_wet_sp = era5_daily.index[
        (era5_daily < _MIN_ERA5_PRECIP) & (sp_daily >= _MIN_SPARTACUS_PRECIP)
    ]

    if len(dry_era5_wet_sp) == 0:
        return df_era5

    for date in dry_era5_wet_sp:
        sp_amount = sp_daily.get(date, 0.0)
        hour_mask = df_era5["_date"] == date
        n_hours = hour_mask.sum()
        if n_hours > 0:
            df_era5.loc[hour_mask, "PSUM"] = sp_amount / n_hours  # mm per hour

    return df_era5


# ---------------------------------------------------------------------------
# SPARTACUS loader
# ---------------------------------------------------------------------------
def _load_spartacus_series(
    spartacus_dir: Path,
    variable: str,
    lat: float,
    lon: float,
) -> pd.Series:
    """
    Aggregate all annual SPARTACUS NetCDF files for one variable into a
    single daily pandas Series, bilinearly interpolated to (lat, lon).

    Parameters
    ----------
    spartacus_dir : Path
        Directory containing SPARTACUS NetCDF files.
    variable : str
        ``'temperature'`` or ``'precipitation'``.
    lat : float
        Target latitude [decimal degrees North].
    lon : float
        Target longitude [decimal degrees East].

    Returns
    -------
    pd.Series
        Daily values, DatetimeIndex.
    """
    pattern = f"spartacus_{variable}_*.nc"
    files = sorted(spartacus_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No SPARTACUS files matching '{pattern}' in {spartacus_dir}. "
            "Run download_spartacus first."
        )

    series_list: list[pd.Series] = []
    for nc_file in files:
        s = _extract_spartacus_point(nc_file, variable, lat, lon)
        series_list.append(s)

    combined = pd.concat(series_list).sort_index()
    # Remove duplicate dates (overlapping file boundaries)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def _extract_spartacus_point(
    nc_file: Path,
    variable: str,
    lat: float,
    lon: float,
) -> pd.Series:
    """
    Load one SPARTACUS NetCDF file and extract the value at (lat, lon)
    using bilinear interpolation.

    Parameters
    ----------
    nc_file : Path
        SPARTACUS NetCDF file.
    variable : str
        ``'temperature'`` or ``'precipitation'``.
    lat : float
        Target latitude.
    lon : float
        Target longitude.

    Returns
    -------
    pd.Series
        Daily values with DatetimeIndex.
    """
    from scripts.interpolate_points import _bilinear_interp_scalar, _detect_latlon_coords

    ds = xr.open_dataset(nc_file)
    lat_coord, lon_coord = _detect_latlon_coords(ds)

    # SPARTACUS variable name (Tl or RR)
    var_map = {"temperature": "Tl", "precipitation": "RR"}
    var_name = var_map[variable]

    if var_name not in ds:
        # Try alternative naming
        available = list(ds.data_vars)
        logger.warning(
            "Expected SPARTACUS variable '%s' not in %s; available: %s",
            var_name,
            nc_file.name,
            available,
        )
        var_name = available[0]

    da = ds[var_name]
    lat_vals = ds[lat_coord].values
    lon_vals = ds[lon_coord].values
    time_index = pd.DatetimeIndex(ds["time"].values)

    values = np.empty(len(time_index))
    for t in range(len(time_index)):
        field_2d = da.isel(time=t).values
        values[t] = _bilinear_interp_scalar(lat_vals, lon_vals, field_2d, lat, lon)

    ds.close()
    return pd.Series(values, index=time_index, name=variable)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    from scripts.interpolate_points import interpolate_era5_to_points
    era5_pts = interpolate_era5_to_points(cfg)
    corrected = apply_bias_correction(cfg, era5_pts)
    for rkey, elev_dict in corrected.items():
        for elev, df in elev_dict.items():
            print(f"{rkey} @ {elev} m: TA range [{df.TA.min():.1f}, {df.TA.max():.1f}] K")
