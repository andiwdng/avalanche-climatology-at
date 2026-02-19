"""
interpolate_points.py
=====================
Interpolate ERA5-Land hourly gridded fields to point coordinates
(lat/lon of each study region) and apply elevation lapse-rate corrections.

Scientific methods
------------------
1. Bilinear spatial interpolation
   ERA5-Land is a regular lat/lon grid.  For each target point we
   identify the four surrounding grid cells and interpolate linearly
   in both latitude and longitude directions.

2. De-accumulation of radiation and precipitation
   ERA5-Land radiation (ssrd, strd) and precipitation (tp) are
   accumulated since 00:00 UTC each day.  Hourly rates are obtained by
   differencing consecutive accumulations:
       F_rate[t] = (F_acc[t] − F_acc[t−1]) / Δt
   where Δt = 3600 s.  Negative differences (reset artefacts) are
   clipped to zero.

3. Relative humidity from dewpoint
   ERA5-Land provides 2 m dewpoint temperature (d2m) instead of
   relative humidity.  RH is derived via the Magnus formula:
       e(T) = 6.1078 · exp(17.2694 · (T−273.15) / (T−35.85))  [hPa]
       RH = e(d2m) / e(t2m)                                    [0–1]

4. Wind speed and meteorological direction
   u10, v10 → VW = √(u²+v²)  [m s⁻¹]
   DW = (270 − atan2(v, u)·180/π) mod 360       [°, from-direction]

5. Lapse-rate elevation correction
   Temperature and precipitation are corrected for the difference
   between the ERA5-Land native orography height z_era and the target
   simulation elevation z_target:
       T_lapse(z_target) = T_era(z_era) + Γ_T · (z_target − z_era)
       P_lapse(z_target) = P_era(z_era) · (1 + Γ_P · (z_target − z_era))
   where Γ_T = −0.0065 K m⁻¹ and Γ_P = +0.0003 m⁻¹ (3 % / 100 m).
   Pressure is adjusted using the hypsometric equation.

Output
------
For each (region, elevation_band) pair the function returns a
``pandas.DataFrame`` with columns:
    timestamp, TA [K], RH [0–1], VW [m s⁻¹], DW [°], ISWR [W m⁻²],
    ILWR [W m⁻²], PSUM [mm h⁻¹], P [Pa]
ready for SPARTACUS bias correction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)

# Physical constants
_GRAVITY: float = 9.80665          # m s⁻²  (standard gravity)
_GAS_CONSTANT_AIR: float = 287.05  # J kg⁻¹ K⁻¹


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def interpolate_era5_to_points(config: dict) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Interpolate ERA5-Land to all (region, elevation_band) combinations.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.

    Returns
    -------
    dict
        Nested dict: ``result[region_key][elevation_m] = DataFrame``
        Each DataFrame has hourly rows and columns as described above.
    """
    era5_dir = Path(config["paths"]["era5_raw"])
    era5_cfg = config["era5"]
    sim_cfg = config["simulation"]
    regions = config["regions"]
    elevation_bands = config["elevation_bands"]

    from datetime import date
    analysis_start = date.fromisoformat(sim_cfg["analysis_start"])
    analysis_end = date.fromisoformat(sim_cfg["analysis_end"])
    spin_up_years = int(sim_cfg["spin_up_years"])
    download_start_year = analysis_start.year - spin_up_years
    download_end_year = analysis_end.year

    # Load ERA5-Land native orography
    z_era5_grid = _load_orography(era5_dir)

    # Accumulate results
    result: dict[str, dict[int, pd.DataFrame]] = {}

    for region_key, region_meta in regions.items():
        lat_target = region_meta["lat"]
        lon_target = region_meta["lon"]
        result[region_key] = {}

        # Get ERA5 native elevation at this point
        z_era5_point = _bilinear_interp_scalar(
            z_era5_grid["lat"].values,
            z_era5_grid["lon"].values,
            z_era5_grid["z"],          # already a np.ndarray from _load_orography
            lat_target,
            lon_target,
        )
        logger.info(
            "Region '%s' (%.3f°N, %.3f°E): ERA5 orography = %.0f m",
            region_key,
            lat_target,
            lon_target,
            z_era5_point,
        )

        # Collect all annual DataFrames
        annual_frames: list[pd.DataFrame] = []
        for year in range(download_start_year, download_end_year + 1):
            nc_path = era5_dir / f"era5land_forcing_{year}.nc"
            if not nc_path.exists():
                logger.warning("ERA5 forcing file missing: %s — skipping year %d", nc_path, year)
                continue
            df_year = _process_forcing_year(
                nc_path=nc_path,
                lat_target=lat_target,
                lon_target=lon_target,
            )
            annual_frames.append(df_year)

        if not annual_frames:
            logger.error("No ERA5 data found for region '%s' — skipping.", region_key)
            continue

        df_base = pd.concat(annual_frames).sort_index()

        # Apply lapse-rate correction for each elevation band
        for elev_m in elevation_bands:
            df_elev = _apply_lapse_rate(
                df=df_base.copy(),
                z_era5=z_era5_point,
                z_target=float(elev_m),
                gamma_T=float(era5_cfg["lapse_rate_temperature"]),
                gamma_P=float(era5_cfg["lapse_rate_precipitation"]),
            )
            result[region_key][elev_m] = df_elev
            logger.debug(
                "Region '%s' @ %d m: %d hourly records interpolated.",
                region_key,
                elev_m,
                len(df_elev),
            )

    return result


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------
def _process_forcing_year(
    nc_path: Path,
    lat_target: float,
    lon_target: float,
) -> pd.DataFrame:
    """
    Load one year of ERA5-Land forcing, interpolate to point, and derive
    the SNOWPACK-ready met variables.

    Parameters
    ----------
    nc_path : Path
        Path to the annual ERA5-Land NetCDF file.
    lat_target : float
        Target latitude [decimal degrees North].
    lon_target : float
        Target longitude [decimal degrees East].

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with index ``datetime`` (UTC) and columns:
        TA, RH, VW, DW, ISWR, ILWR, PSUM, P.
    """
    ds = xr.open_dataset(nc_path)

    # ERA5-Land coordinate names can vary; normalise
    lat_coord, lon_coord = _detect_latlon_coords(ds)

    lat_vals = ds[lat_coord].values
    lon_vals = ds[lon_coord].values

    def _interp(da: xr.DataArray) -> np.ndarray:
        """Bilinear interpolation of a 2-D field series to a single point."""
        arr = da.values  # shape: (time, lat, lon)  or (lat, lon)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        result = np.empty(arr.shape[0])
        for t in range(arr.shape[0]):
            result[t] = _bilinear_interp_scalar(lat_vals, lon_vals, arr[t], lat_target, lon_target)
        return result

    time_index = pd.DatetimeIndex(ds["time"].values)

    # --- 2 m temperature [K] ---
    ta_raw = _interp(ds["t2m"])

    # --- 2 m dewpoint → relative humidity ---
    d2m_raw = _interp(ds["d2m"])
    rh_raw = _dewpoint_to_rh(ta_raw, d2m_raw)

    # --- Wind speed and direction ---
    u10 = _interp(ds["u10"])
    v10 = _interp(ds["v10"])
    vw = np.sqrt(u10 ** 2 + v10 ** 2)
    dw = _uv_to_met_direction(u10, v10)

    # --- De-accumulate radiation [J m⁻²] → [W m⁻²] ---
    ssrd_acc = _interp(ds["ssrd"])
    strd_acc = _interp(ds["strd"])
    iswr = _deaccumulate(ssrd_acc, time_index)
    ilwr = _deaccumulate(strd_acc, time_index)

    # --- De-accumulate precipitation [m h⁻¹] → [mm h⁻¹] ---
    tp_acc = _interp(ds["tp"])
    psum = _deaccumulate(tp_acc, time_index) * 1000.0  # m → mm

    # --- Surface pressure [Pa] ---
    pressure = _interp(ds["sp"])

    ds.close()

    df = pd.DataFrame(
        {
            "TA": ta_raw,
            "RH": rh_raw,
            "VW": vw,
            "DW": dw,
            "ISWR": iswr,
            "ILWR": ilwr,
            "PSUM": psum,
            "P": pressure,
        },
        index=time_index,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Derived-variable functions
# ---------------------------------------------------------------------------
def _dewpoint_to_rh(ta_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """
    Derive relative humidity from air temperature and dewpoint.

    Uses the Magnus approximation (WMO constants):
        e_s(T [°C]) = 6.1078 · exp(17.2694 · T / (T + 237.29))  [hPa]

    Parameters
    ----------
    ta_k : np.ndarray
        Air temperature [K].
    d2m_k : np.ndarray
        Dewpoint temperature [K].

    Returns
    -------
    np.ndarray
        Relative humidity [0 – 1], clipped to [0.01, 1.0].
    """
    ta_c = ta_k - 273.15
    d2m_c = d2m_k - 273.15

    def _sat_vapour(t_c: np.ndarray) -> np.ndarray:
        return 6.1078 * np.exp(17.2694 * t_c / (t_c + 237.29))

    rh = _sat_vapour(d2m_c) / _sat_vapour(ta_c)
    return np.clip(rh, 0.01, 1.0)


def _uv_to_met_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert U/V wind components to meteorological FROM-direction.

    The meteorological wind direction is the direction FROM which the
    wind blows, measured clockwise from North:
        DW = (270° − atan2(v, u) · 180/π) mod 360°

    Parameters
    ----------
    u : np.ndarray
        Eastward wind component [m s⁻¹].
    v : np.ndarray
        Northward wind component [m s⁻¹].

    Returns
    -------
    np.ndarray
        Meteorological wind direction [°], in [0°, 360°).
    """
    direction = np.degrees(np.arctan2(-u, -v))
    return direction % 360.0


def _deaccumulate(
    acc: np.ndarray,
    time_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Convert ERA5-Land time-accumulated fields to per-hour rates.

    ERA5-Land resets accumulations to zero at 00:00 UTC each day (for
    hourly analysis data).  Hourly differences are computed; any
    negative difference (reset artefact) is set to zero.  Rates are
    then divided by 3600 s for radiation fields.

    Parameters
    ----------
    acc : np.ndarray
        Accumulated field values (time axis).
    time_index : pd.DatetimeIndex
        Corresponding timestamps.

    Returns
    -------
    np.ndarray
        Per-hour rates in SI units (W m⁻² for radiation; m h⁻¹ for tp).
    """
    rates = np.diff(acc, prepend=acc[0])
    # Reset at midnight: ERA5 accumulation restarts; diff at 01:00 is valid
    # but at 00:00 the accumulation is near-zero → use raw value at 00:00
    midnight_mask = (time_index.hour == 0)
    rates[midnight_mask] = acc[midnight_mask]
    # Clip negative artefacts
    rates = np.maximum(rates, 0.0)
    # Convert from J m⁻² per hour to W m⁻² (for radiation)
    # For precipitation (tp) the caller multiplies separately
    return rates / 3600.0


# ---------------------------------------------------------------------------
# Lapse-rate correction
# ---------------------------------------------------------------------------
def _apply_lapse_rate(
    df: pd.DataFrame,
    z_era5: float,
    z_target: float,
    gamma_T: float = -0.0065,
    gamma_P: float = 0.0003,
) -> pd.DataFrame:
    """
    Correct ERA5-Land temperature, precipitation and pressure for
    the elevation difference between ERA5 native orography and the
    target elevation band.

    Temperature correction (additive):
        TA_corrected = TA_era + Γ_T · (z_target − z_era)

    Precipitation correction (multiplicative):
        PSUM_corrected = PSUM_era · max(0, 1 + Γ_P · (z_target − z_era))
        (Precipitation cannot decrease to negative values.)

    Pressure correction (hypsometric equation):
        P_corrected = P_era · exp(−g · Δz / (R_d · T_mean))

    Parameters
    ----------
    df : pd.DataFrame
        Point-interpolated ERA5 DataFrame.
    z_era5 : float
        ERA5-Land surface elevation at the interpolation point [m].
    z_target : float
        Target simulation elevation [m].
    gamma_T : float
        Temperature lapse rate [K m⁻¹]; default −0.0065.
    gamma_P : float
        Precipitation lapse rate [m⁻¹]; default +0.0003.

    Returns
    -------
    pd.DataFrame
        Lapse-rate-corrected DataFrame (same columns).
    """
    dz = z_target - z_era5

    # Temperature: additive correction
    df["TA"] = df["TA"] + gamma_T * dz

    # Precipitation: multiplicative correction, no negatives
    precip_factor = max(0.0, 1.0 + gamma_P * dz)
    df["PSUM"] = df["PSUM"] * precip_factor

    # Pressure: hypsometric equation with mean temperature
    t_mean = df["TA"].mean()
    pressure_factor = np.exp(-_GRAVITY * dz / (_GAS_CONSTANT_AIR * t_mean))
    df["P"] = df["P"] * pressure_factor

    return df


# ---------------------------------------------------------------------------
# ERA5 orography loader
# ---------------------------------------------------------------------------
def _load_orography(era5_dir: Path) -> dict:
    """
    Load ERA5-Land surface geopotential and convert to geometric height.

    Parameters
    ----------
    era5_dir : Path
        Directory containing ``era5land_orography.nc``.

    Returns
    -------
    dict
        Keys: ``'lat'`` (DataArray), ``'lon'`` (DataArray), ``'z'`` (2-D ndarray [m]).
    """
    orog_file = era5_dir / "era5land_orography.nc"
    if not orog_file.exists():
        raise FileNotFoundError(
            f"ERA5 orography not found: {orog_file}. Run download_era5 first."
        )

    ds = xr.open_dataset(orog_file)
    lat_coord, lon_coord = _detect_latlon_coords(ds)

    # Variable name for geopotential in ERA5-Land
    orog_var = "z" if "z" in ds else "Z"

    geopotential = ds[orog_var].isel(time=0) if "time" in ds[orog_var].dims else ds[orog_var]
    z_metres = (geopotential / _GRAVITY).values  # convert Φ [m² s⁻²] → z [m]

    result = {
        "lat": ds[lat_coord],
        "lon": ds[lon_coord],
        "z": z_metres,
    }
    ds.close()
    return result


# ---------------------------------------------------------------------------
# Spatial interpolation
# ---------------------------------------------------------------------------
def _bilinear_interp_scalar(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    field: np.ndarray,
    lat_p: float,
    lon_p: float,
) -> float:
    """
    Bilinear interpolation of a 2-D regular grid to a single point.

    Parameters
    ----------
    lat_grid : np.ndarray, shape (M,)
        Latitude coordinates of the grid, monotone (ascending or descending).
    lon_grid : np.ndarray, shape (N,)
        Longitude coordinates of the grid.
    field : np.ndarray, shape (M, N)
        Grid values.
    lat_p : float
        Target latitude.
    lon_p : float
        Target longitude.

    Returns
    -------
    float
        Bilinearly interpolated value.  Returns ``np.nan`` if the point lies
        outside the grid extent.
    """
    # Ensure latitudes are ascending for searchsorted
    if lat_grid[0] > lat_grid[-1]:
        lat_grid = lat_grid[::-1]
        field = field[::-1, :]

    if lon_grid[0] > lon_grid[-1]:
        lon_grid = lon_grid[::-1]
        field = field[:, ::-1]

    if lat_p < lat_grid[0] or lat_p > lat_grid[-1]:
        return float(np.nan)
    if lon_p < lon_grid[0] or lon_p > lon_grid[-1]:
        return float(np.nan)

    i1 = np.searchsorted(lat_grid, lat_p, side="right") - 1
    j1 = np.searchsorted(lon_grid, lon_p, side="right") - 1
    i1 = int(np.clip(i1, 0, len(lat_grid) - 2))
    j1 = int(np.clip(j1, 0, len(lon_grid) - 2))
    i2, j2 = i1 + 1, j1 + 1

    lat1, lat2 = lat_grid[i1], lat_grid[i2]
    lon1, lon2 = lon_grid[j1], lon_grid[j2]

    # Bilinear weights
    t = (lat_p - lat1) / (lat2 - lat1)
    s = (lon_p - lon1) / (lon2 - lon1)

    value = (
        field[i1, j1] * (1 - t) * (1 - s)
        + field[i2, j1] * t * (1 - s)
        + field[i1, j2] * (1 - t) * s
        + field[i2, j2] * t * s
    )
    return float(value)


def _detect_latlon_coords(ds: xr.Dataset) -> tuple[str, str]:
    """
    Detect latitude and longitude coordinate names in an ERA5 Dataset.

    ERA5-Land commonly uses ``'latitude'``/``'longitude'`` or
    ``'lat'``/``'lon'``.

    Returns
    -------
    tuple[str, str]
        ``(lat_name, lon_name)``
    """
    for lat_cand in ("latitude", "lat", "y"):
        for lon_cand in ("longitude", "lon", "x"):
            if lat_cand in ds.coords and lon_cand in ds.coords:
                return lat_cand, lon_cand
    raise KeyError(f"Cannot find lat/lon coordinates in dataset with coords: {list(ds.coords)}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    result = interpolate_era5_to_points(cfg)
    for rkey, elev_dict in result.items():
        for elev, df in elev_dict.items():
            print(f"{rkey} @ {elev} m: {len(df)} records, columns: {list(df.columns)}")
