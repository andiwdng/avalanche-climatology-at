"""
smoke_test.py
=============
End-to-end pipeline smoke test using entirely synthetic data.

No CDS API account, no SNOWPACK binary, and no AVAPRO installation are
required.  The test exercises every pipeline stage from preprocessing
through figure generation, using:

- Synthetically generated ERA5-Land-style NetCDF files (3 annual files,
  full hourly resolution, physically plausible ranges)
- Synthetically generated SPARTACUS-style NetCDF files
- Mocked SNOWPACK step (synthetic daily problem DataFrames replace PRO output)
- Real calls to interpolate_points, bias_correction, snowpack_writer,
  climatology, and plotting modules

Pass/fail is reported for each stage individually so that partial
failures are immediately attributable.

Usage
-----
Run from the repository root::

    python scripts/smoke_test.py

Optional flags::

    python scripts/smoke_test.py --keep-tmp   # do not delete temp files after run
    python scripts/smoke_test.py --verbose     # DEBUG-level logging
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ============================================================
# Minimal smoke-test configuration
# ============================================================
def _build_config(tmp_dir: Path) -> dict:
    """
    Build a minimal pipeline configuration pointing at *tmp_dir*.

    Three regions and two elevation bands are used.
    Three analysis seasons are generated (Oct 2000 – Apr 2003),
    giving 9 data points for k=2 clustering.

    Parameters
    ----------
    tmp_dir : Path
        Temporary base directory for all synthetic data and outputs.

    Returns
    -------
    dict
        Configuration dictionary compatible with all pipeline modules.
    """
    return {
        "regions": {
            "bregenzerwald": {
                "name": "Bregenzerwald",
                "lat": 47.40,
                "lon": 9.95,
                "province": "Vorarlberg",
                "lwd": "LWD Vorarlberg",
            },
            "tirol": {
                "name": "Tirol",
                "lat": 47.15,
                "lon": 11.40,
                "province": "Tirol",
                "lwd": "LWD Tirol",
            },
            "salzburg": {
                "name": "Salzburg",
                "lat": 47.50,
                "lon": 13.10,
                "province": "Salzburg",
                "lwd": "LAWIS",
            },
        },
        "elevation_bands": [1500, 2000],
        "simulation": {
            "analysis_start": "2000-10-01",
            "analysis_end":   "2003-04-30",
            "spin_up_years": 0,
            "season_start_month": 10,
            "season_end_month": 4,
        },
        "era5": {
            "cds_api_rc": "~/.cdsapirc",
            "product": "reanalysis-era5-land",
            "area": [49.0, 9.5, 46.3, 17.2],
            "variables": [],
            "lapse_rate_temperature": -0.0065,
            "lapse_rate_precipitation": 0.0003,
        },
        "spartacus": {
            "api_base": "https://dataset.api.hub.zamg.ac.at",
            "bbox": [9.5, 46.3, 17.2, 49.0],
            "parameters": {"temperature": "Tl", "precipitation": "RR"},
            "interpolation": "bilinear",
        },
        "snowpack": {
            "binary": "/nonexistent/snowpack",
            "slope_angle": 0.0,
            "aspect": 0.0,
            "calculation_step_length": 15,
            "meteo_step_length": 60,
            "soil_albedo": 0.09,
            "bare_soil_z0": 0.020,
            "profile_output_step": 24,
            "stability": True,
            "meteo_output": False,
        },
        "avapro": {
            "binary": "/nonexistent/avapro",
            "min_profile_depth": 0.10,
            "problem_columns": {
                "new_snow": "new_snow",
                "wind_slab": "wind_slab",
                "persistent_wl": "persistent_wl",
                "wet_snow": "wet_snow",
                "glide_snow": "glide_snow",
            },
        },
        "clustering": {
            "n_clusters": 2,   # reduced: 3 seasons × 3 regions = 9 points → sufficient for k=2
            "random_state": 42,
            "n_init": 10,
            "variables": [
                "new_snow_days",
                "pwl_days",
                "wet_snow_onset_doy",
                "total_problem_days",
            ],
            "reference_elevation": 2000,
        },
        "paths": {
            "era5_raw":        str(tmp_dir / "era5_raw"),
            "spartacus":       str(tmp_dir / "spartacus"),
            "snowpack_input":  str(tmp_dir / "snowpack_input"),
            "snowpack_output": str(tmp_dir / "snowpack_output"),
            "avapro_output":   str(tmp_dir / "avapro_output"),
            "figures":         str(tmp_dir / "figures"),
            "logs":            str(tmp_dir / "logs"),
        },
        "plotting": {
            "dpi": 72,
            "figure_format": "png",
            "colormap_clusters": "tab10",
            "colormap_problems": "Set2",
            "season_labels": True,
        },
    }


# ============================================================
# Synthetic data generators
# ============================================================

# Spatial grid covering Austria with enough extent for bilinear interpolation
_LATS = np.array([46.5, 47.0, 47.5, 48.0, 48.5], dtype="float32")
_LONS = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype="float32")


def _make_era5_orography(out_dir: Path) -> None:
    """
    Write a synthetic ERA5-Land orography NetCDF.

    The surface geopotential Φ = g·z corresponds to a realistic
    Austrian mountain terrain (z ranging from ~400 m in the east to
    ~1800 m in the central Alps).
    """
    g = 9.80665
    nlat, nlon = len(_LATS), len(_LONS)
    # Simulate alpine terrain: higher in the centre, lower at edges
    z_m = np.zeros((nlat, nlon), dtype="float32")
    for i, lat in enumerate(_LATS):
        for j, lon in enumerate(_LONS):
            # Simple terrain model centred on Austrian Alps
            z_m[i, j] = 800 + 1200 * np.exp(-0.5 * ((lat - 47.2) / 0.8) ** 2
                                              - 0.5 * ((lon - 12.0) / 2.5) ** 2)
    geopotential = z_m * g  # Φ [m² s⁻²]

    ds = xr.Dataset(
        {"z": (["time", "latitude", "longitude"], geopotential[None, :, :])},
        coords={
            "time": pd.date_range("2000-01-01", periods=1, freq="h"),
            "latitude": _LATS,
            "longitude": _LONS,
        },
    )
    path = Path(out_dir) / "era5land_orography.nc"
    ds.to_netcdf(path)
    logger.debug("Orography written: %s", path.name)


def _make_era5_forcing_year(out_dir: Path, year: int) -> None:
    """
    Write one year of synthetic ERA5-Land hourly forcing.

    Physical ranges
    ---------------
    - t2m  [K]:  alpine winter/spring temperatures 255 – 290 K
    - d2m  [K]:  2–10 K below t2m
    - tp   [m]:  accumulated precipitation (reset at midnight)
    - sp   [Pa]: 85 000 – 102 000 Pa
    - u10, v10 [m s⁻¹]: −10 to +10
    - ssrd, strd [J m⁻²]: accumulated radiation (reset at midnight)
    """
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    nt = len(times)
    nlat, nlon = len(_LATS), len(_LONS)
    rng = np.random.default_rng(seed=year * 7 + 13)

    doy = np.array([t.day_of_year for t in times], dtype="float32")
    hour = np.array([t.hour for t in times], dtype="float32")

    # --- Temperature: seasonal + diurnal cycle + noise ---
    t_seasonal = 270.0 + 14.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    t_diurnal = 4.0 * np.sin(2 * np.pi * (hour - 6) / 24)
    t_noise = rng.normal(0, 1.5, nt).astype("float32")
    t2m_1d = (t_seasonal + t_diurnal + t_noise)
    t2m = t2m_1d[:, None, None] * np.ones((1, nlat, nlon), dtype="float32")

    # Dewpoint: 3–10 K below air temperature
    d2m = t2m - rng.uniform(3.0, 10.0, (nt, nlat, nlon)).astype("float32")

    # --- Pressure ---
    sp = (93000 + rng.normal(0, 800, (nt, nlat, nlon))).astype("float32")

    # --- Wind ---
    u10 = rng.normal(0, 3, (nt, nlat, nlon)).astype("float32")
    v10 = rng.normal(0, 3, (nt, nlat, nlon)).astype("float32")

    # --- Radiation: accumulated within each day (reset at midnight) ---
    # Peak daytime shortwave ~600 W/m² × 3600 J/hr
    ssrd_instant = np.zeros((nt, nlat, nlon), dtype="float32")
    daylight_mask = (hour >= 6) & (hour <= 18)
    ssrd_instant[daylight_mask] = (
        550.0 * np.sin(np.pi * (hour[daylight_mask] - 6) / 12)[:, None, None]
        * np.ones((1, nlat, nlon), dtype="float32")
        + rng.uniform(-30, 30, (daylight_mask.sum(), nlat, nlon)).astype("float32")
    ).clip(0)

    strd_instant = (270.0 + rng.normal(0, 15, (nt, nlat, nlon))).clip(200, 380).astype("float32")

    ssrd_acc = _accumulate_since_midnight(ssrd_instant * 3600.0, times)
    strd_acc = _accumulate_since_midnight(strd_instant * 3600.0, times)

    # --- Precipitation: accumulated within each day (reset at midnight) ---
    tp_rate = rng.exponential(0.00012, (nt, nlat, nlon)).astype("float32")
    tp_acc = _accumulate_since_midnight(tp_rate, times)

    ds = xr.Dataset(
        {
            "t2m":  (["time", "latitude", "longitude"], t2m),
            "d2m":  (["time", "latitude", "longitude"], d2m),
            "tp":   (["time", "latitude", "longitude"], tp_acc),
            "sp":   (["time", "latitude", "longitude"], sp),
            "u10":  (["time", "latitude", "longitude"], u10),
            "v10":  (["time", "latitude", "longitude"], v10),
            "ssrd": (["time", "latitude", "longitude"], ssrd_acc),
            "strd": (["time", "latitude", "longitude"], strd_acc),
        },
        coords={
            "time": times,
            "latitude": _LATS,
            "longitude": _LONS,
        },
    )
    path = Path(out_dir) / f"era5land_forcing_{year}.nc"
    ds.to_netcdf(path)
    logger.debug("ERA5 forcing %d written: %s (nt=%d)", year, path.name, nt)


def _accumulate_since_midnight(
    instant: np.ndarray,
    times: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Convert an instantaneous rate array to a day-accumulated value
    that resets to the instantaneous rate at 00:00 UTC each day.

    Parameters
    ----------
    instant : np.ndarray, shape (T, ...)
        Instantaneous values.
    times : pd.DatetimeIndex
        Corresponding timestamps.

    Returns
    -------
    np.ndarray
        Accumulated values (same shape), reset at each 00:00 UTC.
    """
    acc = np.zeros_like(instant)
    # Identify unique calendar days
    dates = np.array([t.date() for t in times])
    unique_days, inverse = np.unique(dates, return_inverse=True)

    for day_idx in range(len(unique_days)):
        mask = inverse == day_idx
        # cumsum along the time axis within this day's slice
        day_slice = instant[mask]
        acc[mask] = np.cumsum(day_slice, axis=0)

    return acc


def _make_spartacus_year(out_dir: Path, year: int, variable: str) -> None:
    """
    Write one year of synthetic SPARTACUS daily temperature or precipitation.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    year : int
        Calendar year.
    variable : str
        ``'temperature'`` or ``'precipitation'``.
    """
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    nt = len(times)
    nlat, nlon = len(_LATS), len(_LONS)
    rng = np.random.default_rng(seed=year * 3 + 1001)

    if variable == "temperature":
        doy = np.array([t.day_of_year for t in times], dtype="float32")
        t_seasonal = -4.0 + 13.0 * np.sin(2 * np.pi * (doy - 80) / 365)
        data = (t_seasonal[:, None, None] + rng.normal(0, 2, (nt, nlat, nlon))).astype("float32")
        var_name = "Tl"
    else:
        # Daily precipitation [mm], mostly 0 with occasional events
        data = (rng.exponential(3.0, (nt, nlat, nlon)) * (rng.random((nt, nlat, nlon)) > 0.7)).astype("float32")
        var_name = "RR"

    ds = xr.Dataset(
        {var_name: (["time", "latitude", "longitude"], data)},
        coords={
            "time": times,
            "latitude": _LATS,
            "longitude": _LONS,
        },
    )
    path = Path(out_dir) / f"spartacus_{variable}_{year}.nc"
    ds.to_netcdf(path)
    logger.debug("SPARTACUS %s %d written: %s", variable, year, path.name)


def _make_synthetic_problems(
    config: dict,
) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Generate synthetic daily avalanche problem DataFrames for all
    (region, elevation_band) combinations.

    This replaces the SNOWPACK + AVAPRO steps.  Each season has a
    distinct random pattern so that k-means clustering can find
    meaningful groupings.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    dict
        ``problems[region][elev] = DataFrame`` with boolean columns:
        new_snow, wind_slab, persistent_wl, wet_snow, glide_snow.
    """
    rng = np.random.default_rng(42)
    analysis_start = config["simulation"]["analysis_start"]
    analysis_end = config["simulation"]["analysis_end"]
    index = pd.date_range(analysis_start, analysis_end, freq="D")
    n = len(index)

    problems: dict[str, dict[int, pd.DataFrame]] = {}

    for region_key in config["regions"]:
        problems[region_key] = {}
        for elev_m in config["elevation_bands"]:
            # Elevation-dependent frequencies: more new snow + PWL at high elevation,
            # more wet snow at lower elevation / spring
            elev_factor = (elev_m - 1500) / 1500.0  # 0 at 1500 m, 1 at 3000 m

            doy = index.day_of_year.values.astype(float)
            month = index.month.values

            # New snow: higher frequency in early winter (Oct–Feb)
            new_snow_p = np.where(
                (month >= 10) | (month <= 2),
                0.35 + 0.10 * elev_factor,
                0.10,
            )
            # PWL: higher at high elevation, peaks in Jan–Mar
            pwl_p = np.where(
                (month >= 1) & (month <= 3),
                0.20 + 0.15 * elev_factor,
                0.05,
            )
            # Wet snow: spring, higher at low elevation
            wet_snow_p = np.where(
                (month >= 3) & (month <= 4),
                0.30 - 0.10 * elev_factor,
                0.02,
            )
            # Wind slab: correlated with new snow
            wind_slab_p = new_snow_p * 0.6

            df = pd.DataFrame(
                {
                    "new_snow":    rng.random(n) < new_snow_p,
                    "wind_slab":   rng.random(n) < wind_slab_p,
                    "persistent_wl": rng.random(n) < pwl_p,
                    "wet_snow":    rng.random(n) < wet_snow_p,
                    "glide_snow":  np.zeros(n, dtype=bool),
                },
                index=index,
            )
            problems[region_key][elev_m] = df

    return problems


# ============================================================
# Test runner
# ============================================================
class _StageResult:
    """Record for a single pipeline stage result."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed: bool = False
        self.elapsed: float = 0.0
        self.error: str = ""


def run_smoke_test(tmp_dir: Path, verbose: bool = False) -> bool:
    """
    Execute the full smoke-test pipeline.

    Parameters
    ----------
    tmp_dir : Path
        Temporary directory for all synthetic data and outputs.
    verbose : bool
        If True, set logging to DEBUG level.

    Returns
    -------
    bool
        True if all stages passed.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 65)
    print("  AVALANCHE CLIMATOLOGY PIPELINE — Smoke Test")
    print("  Synthetic data, no external tools required")
    print("=" * 65)

    config = _build_config(tmp_dir)

    # Create all required directories
    for key, path_str in config["paths"].items():
        Path(path_str).mkdir(parents=True, exist_ok=True)
    for sub in ("smet", "sno", "ini"):
        (Path(config["paths"]["snowpack_input"]) / sub).mkdir(parents=True, exist_ok=True)

    results: list[_StageResult] = []

    # ── Stage A: Generate synthetic input data ────────────────────────────────
    stage = _StageResult("A  Generate synthetic NetCDF files")
    t0 = time.perf_counter()
    try:
        era5_dir = Path(config["paths"]["era5_raw"])
        sp_dir = Path(config["paths"]["spartacus"])

        print("\n[A] Generating synthetic ERA5-Land and SPARTACUS files …")
        _make_era5_orography(era5_dir)
        for year in range(2000, 2004):   # 2000, 2001, 2002, 2003
            _make_era5_forcing_year(era5_dir, year)
        for year in range(2000, 2004):
            _make_spartacus_year(sp_dir, year, "temperature")
            _make_spartacus_year(sp_dir, year, "precipitation")

        n_era5 = len(list(era5_dir.glob("*.nc")))
        n_sp = len(list(sp_dir.glob("*.nc")))
        print(f"    ERA5 files: {n_era5}   SPARTACUS files: {n_sp}")
        stage.passed = (n_era5 == 5) and (n_sp == 8)  # 1 orog + 4 forcing; 4 T + 4 RR
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage B: Interpolation + lapse-rate correction ────────────────────────
    stage = _StageResult("B  ERA5 interpolation → points (lapse-rate)")
    t0 = time.perf_counter()
    era5_points = {}
    try:
        from scripts.interpolate_points import interpolate_era5_to_points
        print("\n[B] Interpolating ERA5-Land to point coordinates …")
        era5_points = interpolate_era5_to_points(config)
        n_series = sum(len(v) for v in era5_points.values())
        print(f"    Interpolated (region, elev) pairs: {n_series}")
        # Check all expected keys are present
        expected = len(config["regions"]) * len(config["elevation_bands"])
        stage.passed = (n_series == expected) and all(
            not df.empty for reg in era5_points.values() for df in reg.values()
        )
        if not stage.passed:
            stage.error = f"Expected {expected} series, got {n_series}"
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage C: SPARTACUS bias correction ────────────────────────────────────
    stage = _StageResult("C  SPARTACUS bias correction")
    t0 = time.perf_counter()
    corrected = {}
    try:
        from scripts.bias_correction import apply_bias_correction
        print("\n[C] Applying SPARTACUS bias correction …")
        corrected = apply_bias_correction(config, era5_points)

        # Spot-check: temperature range should be physically plausible (220–300 K)
        sample_df = next(iter(next(iter(corrected.values())).values()))
        ta_min = sample_df["TA"].min()
        ta_max = sample_df["TA"].max()
        print(f"    TA range after correction: [{ta_min:.1f}, {ta_max:.1f}] K")
        stage.passed = (220 < ta_min) and (ta_max < 305)
        if not stage.passed:
            stage.error = f"TA out of plausible range: [{ta_min:.1f}, {ta_max:.1f}] K"
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage D: SNOWPACK writer ───────────────────────────────────────────────
    stage = _StageResult("D  SNOWPACK SMET / SNO / INI writer")
    t0 = time.perf_counter()
    ini_paths = {}
    try:
        from scripts.snowpack_writer import write_all_snowpack_inputs
        print("\n[D] Writing SNOWPACK SMET, SNO, and INI files …")
        ini_paths = write_all_snowpack_inputs(config, corrected)

        smet_dir = Path(config["paths"]["snowpack_input"]) / "smet"
        smet_files = list(smet_dir.glob("*.smet"))
        print(f"    SMET files written: {len(smet_files)}")

        expected = len(config["regions"]) * len(config["elevation_bands"])
        stage.passed = len(smet_files) == expected

        # Validate one SMET file: check header and data rows
        if smet_files:
            _validate_smet(smet_files[0])

        if not stage.passed:
            stage.error = f"Expected {expected} SMET files, found {len(smet_files)}"
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage E: Synthetic avalanche problems (replaces SNOWPACK + AVAPRO) ────
    stage = _StageResult("E  Synthetic avalanche problems (SNOWPACK/AVAPRO mock)")
    t0 = time.perf_counter()
    problems = {}
    try:
        print("\n[E] Generating synthetic avalanche problem DataFrames …")
        problems = _make_synthetic_problems(config)

        # Save to avapro_output/ for reproducibility
        avapro_dir = Path(config["paths"]["avapro_output"])
        for region_key, elev_dict in problems.items():
            for elev_m, df in elev_dict.items():
                station_id = f"{region_key}_{elev_m}m"
                df.to_csv(avapro_dir / f"{station_id}_problems.csv")

        total_problem_days = sum(
            df.any(axis=1).sum()
            for reg in problems.values()
            for df in reg.values()
        )
        print(f"    Total problem-days across all (region, elev) pairs: {total_problem_days}")
        stage.passed = total_problem_days > 0
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage F: Seasonal climatology aggregation ──────────────────────────────
    stage = _StageResult("F  Seasonal climatology aggregation")
    t0 = time.perf_counter()
    df_clim = pd.DataFrame()
    try:
        from scripts.climatology import compute_climatology, save_climatology_outputs
        print("\n[F] Aggregating seasonal avalanche problem statistics …")
        df_clim = compute_climatology(config, problems)
        print(f"    Seasonal records: {len(df_clim)}")
        print(df_clim[["region", "elevation_m", "season",
                        "new_snow_days", "pwl_days", "wet_snow_onset_doy",
                        "total_problem_days"]].head(6).to_string(index=False))

        # Expected: 3 regions × 2 elevations × 3 seasons = 18 records
        expected = (
            len(config["regions"])
            * len(config["elevation_bands"])
            * 3  # Oct2000–Apr2001, Oct2001–Apr2002, Oct2002–Apr2003
        )
        stage.passed = (len(df_clim) == expected)
        if not stage.passed:
            stage.error = f"Expected {expected} records, got {len(df_clim)}"
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage G: K-means clustering ────────────────────────────────────────────
    stage = _StageResult("G  K-means cluster analysis")
    t0 = time.perf_counter()
    df_clusters = pd.DataFrame()
    try:
        from scripts.climatology import perform_clustering, save_climatology_outputs
        print("\n[G] Running k-means cluster analysis (k=2) …")
        df_clusters = perform_clustering(config, df_clim)
        unique_clusters = sorted(df_clusters["cluster"].dropna().unique())
        print(f"    Cluster labels found: {unique_clusters}")

        # Save CSV outputs
        save_climatology_outputs(config, df_clim, df_clusters)

        stage.passed = (
            "cluster" in df_clusters.columns
            and len(unique_clusters) == config["clustering"]["n_clusters"]
        )
        if not stage.passed:
            stage.error = (
                f"Expected {config['clustering']['n_clusters']} clusters, "
                f"got {unique_clusters}"
            )
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Stage H: Figure generation ─────────────────────────────────────────────
    stage = _StageResult("H  Figure generation (all 6 figures)")
    t0 = time.perf_counter()
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for CI / server environments
        from scripts.plotting import create_all_figures
        print("\n[H] Generating figures (headless Agg backend) …")
        create_all_figures(config, df_clim, df_clusters)

        figures_dir = Path(config["paths"]["figures"])
        png_files = list(figures_dir.glob("fig*.png"))
        print(f"    Figures written: {len(png_files)}")
        for f in sorted(png_files):
            size_kb = f.stat().st_size // 1024
            print(f"      {f.name}  ({size_kb} kB)")

        stage.passed = len(png_files) == 6
        if not stage.passed:
            stage.error = f"Expected 6 figures, found {len(png_files)}"
    except Exception:
        stage.error = traceback.format_exc()
    stage.elapsed = time.perf_counter() - t0
    results.append(stage)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SMOKE TEST RESULTS")
    print("=" * 65)
    all_passed = True
    for r in results:
        status = "PASS ✓" if r.passed else "FAIL ✗"
        print(f"  {status}  [{r.elapsed:5.1f}s]  Stage {r.name}")
        if not r.passed:
            all_passed = False
            if r.error:
                # Indent error block
                for line in r.error.strip().splitlines()[-8:]:
                    print(f"           {line}")
    print("=" * 65)
    print(f"  Overall: {'ALL STAGES PASSED' if all_passed else 'SOME STAGES FAILED'}")
    print("=" * 65 + "\n")

    return all_passed


# ============================================================
# SMET file validator
# ============================================================
def _validate_smet(smet_path: Path) -> None:
    """
    Quick structural validation of a SMET file.

    Checks:
    - SMET magic header present
    - [HEADER] and [DATA] sections present
    - Required fields declared
    - Data rows have the expected number of columns

    Parameters
    ----------
    smet_path : Path
        Path to the SMET file to validate.

    Raises
    ------
    AssertionError
        If any structural check fails.
    """
    required_fields = {"timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "PSUM", "PSUM_PH", "P"}
    text = smet_path.read_text(encoding="utf-8")

    assert "SMET 1.1 ASCII" in text, "SMET magic header missing"
    assert "[HEADER]" in text, "[HEADER] section missing"
    assert "[DATA]" in text, "[DATA] section missing"

    # Extract declared fields
    declared_fields = set()
    for line in text.splitlines():
        if line.startswith("fields"):
            parts = line.split("=", 1)[-1].strip().split()
            declared_fields = set(parts)
            break

    missing_fields = required_fields - declared_fields
    assert not missing_fields, f"SMET fields missing: {missing_fields}"

    # Check a data row
    in_data = False
    n_declared = len(declared_fields)  # including timestamp
    n_cols_declared = len(line.split("=", 1)[-1].strip().split()) if "fields" in text else 0
    for raw_line in text.splitlines():
        if raw_line.strip() == "[DATA]":
            in_data = True
            continue
        if in_data and raw_line.strip():
            parts = raw_line.strip().split()
            assert len(parts) == n_cols_declared, (
                f"SMET data row has {len(parts)} columns, expected {n_cols_declared}: {raw_line!r}"
            )
            break

    logger.debug("SMET validation passed: %s", smet_path.name)


# ============================================================
# Entry-point
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalanche climatology pipeline — smoke test (no external tools required)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Do not delete the temporary directory after the test.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="avclim_smoketest_"))
    print(f"\nTemporary directory: {tmp}")

    try:
        passed = run_smoke_test(tmp_dir=tmp, verbose=args.verbose)
    finally:
        if args.keep_tmp:
            print(f"Keeping temp dir: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
