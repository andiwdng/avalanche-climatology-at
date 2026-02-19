"""
main.py
=======
Master pipeline orchestrator for the Austrian Alps avalanche climatology
analysis following Reuter et al. (2023).

Pipeline steps
--------------
1.  Download ERA5-Land hourly reanalysis data (CDS API).
2.  Download SPARTACUS daily gridded temperature and precipitation.
3.  Interpolate ERA5-Land to representative region points with lapse-rate
    elevation correction.
4.  Apply SPARTACUS-based daily bias correction (additive T, multiplicative P).
5.  Write SNOWPACK SMET input files, initial snow profiles, and INI configs.
6.  Run SNOWPACK point simulations in parallel (all regions × elevations).
7.  Run AVAPRO on SNOWPACK PRO profiles to classify avalanche problems.
8.  Aggregate daily problems into seasonal statistics.
9.  K-means cluster analysis of avalanche climate types.
10. Export CSV tables and publication-quality figures.

Usage
-----
    python main.py [--config config.yaml] [--skip-download] [--skip-snowpack]
                   [--skip-avapro] [--only-plot] [--n-jobs N]

Arguments
---------
--config PATH      Path to configuration YAML (default: config.yaml)
--skip-download    Skip ERA5 and SPARTACUS downloads (use existing files)
--skip-snowpack    Skip SNOWPACK simulation (use existing PRO files)
--skip-avapro      Skip AVAPRO classification (use existing CSV files)
--only-plot        Load existing seasonal_stats.csv and re-generate figures
--n-jobs N         Number of parallel SNOWPACK workers (default: -1, all CPUs)
--log-level LEVEL  Logging level: DEBUG, INFO, WARNING (default: INFO)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(level: str, log_dir: Path) -> None:
    """Configure root logger with console and file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load and validate the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml relative to the repository root.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required configuration keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}\n"
            "Run the pipeline from the repository root directory."
        )

    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Validate required top-level keys
    required_keys = ["regions", "elevation_bands", "simulation", "era5", "spartacus",
                     "snowpack", "avapro", "clustering", "paths"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"config.yaml is missing required sections: {missing}")

    # Convert elevation bands to integers
    config["elevation_bands"] = [int(e) for e in config["elevation_bands"]]

    logger.info(
        "Configuration loaded: %d regions, %d elevation bands, period %s → %s",
        len(config["regions"]),
        len(config["elevation_bands"]),
        config["simulation"]["analysis_start"],
        config["simulation"]["analysis_end"],
    )
    return config


# ---------------------------------------------------------------------------
# Step utilities
# ---------------------------------------------------------------------------
def _step(step_number: int, description: str) -> None:
    """Log a prominent step header."""
    separator = "─" * 70
    logger.info(separator)
    logger.info("STEP %d / 10  —  %s", step_number, description)
    logger.info(separator)


def _elapsed(t0: float) -> str:
    """Return elapsed time string from t0 (result of time.perf_counter())."""
    secs = time.perf_counter() - t0
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def stage_download(config: dict) -> None:
    """Download ERA5-Land and SPARTACUS data."""
    from scripts.download_era5 import download_era5
    from scripts.download_spartacus import download_spartacus

    _step(1, "Downloading ERA5-Land via CDS API")
    download_era5(config)

    _step(2, "Downloading SPARTACUS daily grids")
    download_spartacus(config)


def stage_preprocess(config: dict) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Interpolate ERA5 to points, apply lapse-rate correction, and bias-correct.

    Returns
    -------
    dict
        Bias-corrected hourly DataFrames per (region, elevation).
    """
    from scripts.interpolate_points import interpolate_era5_to_points
    from scripts.bias_correction import apply_bias_correction

    _step(3, "Interpolating ERA5-Land to point coordinates (lapse-rate corrected)")
    era5_points = interpolate_era5_to_points(config)

    _step(4, "Applying SPARTACUS daily bias correction")
    corrected = apply_bias_correction(config, era5_points)

    return corrected


def stage_write_snowpack_input(
    config: dict,
    corrected: dict[str, dict[int, pd.DataFrame]],
) -> dict[str, dict[int, Path]]:
    """
    Write SNOWPACK SMET, SNO and INI files.

    Returns
    -------
    dict
        Paths to INI files per (region, elevation).
    """
    from scripts.snowpack_writer import write_all_snowpack_inputs

    _step(5, "Writing SNOWPACK SMET + INI input files")
    ini_paths = write_all_snowpack_inputs(config, corrected)
    return ini_paths


def stage_run_snowpack(
    config: dict,
    ini_paths: dict[str, dict[int, Path]],
    n_jobs: int = -1,
) -> dict[str, dict[int, bool]]:
    """
    Execute SNOWPACK simulations in parallel.

    Returns
    -------
    dict
        Success flags per (region, elevation).
    """
    from scripts.run_snowpack import run_snowpack_simulations

    _step(6, "Running SNOWPACK simulations (flat-field point simulations)")
    success = run_snowpack_simulations(config, ini_paths, n_jobs=n_jobs)
    return success


def stage_run_avapro(
    config: dict,
) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Run AVAPRO on SNOWPACK PRO files.

    Returns
    -------
    dict
        Daily avalanche problem DataFrames per (region, elevation).
    """
    from scripts.run_avapro import run_avapro_all
    from scripts.run_snowpack import find_pro_files

    _step(7, "Running AVAPRO avalanche problem classifier")
    pro_files = find_pro_files(config)
    problems = run_avapro_all(config, pro_files)
    return problems


def stage_climatology(
    config: dict,
    problems: dict[str, dict[int, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute seasonal statistics and k-means clusters.

    Returns
    -------
    tuple
        (df_clim, df_clusters) DataFrames.
    """
    from scripts.climatology import (
        compute_climatology,
        perform_clustering,
        save_climatology_outputs,
    )

    _step(8, "Aggregating seasonal avalanche problem statistics")
    df_clim = compute_climatology(config, problems)
    logger.info("Seasonal records computed: %d", len(df_clim))

    _step(9, "K-means cluster analysis (avalanche climate types)")
    df_clusters = perform_clustering(config, df_clim)

    # Save CSV outputs
    save_climatology_outputs(config, df_clim, df_clusters)
    return df_clim, df_clusters


def stage_plotting(
    config: dict,
    df_clim: pd.DataFrame,
    df_clusters: pd.DataFrame,
) -> None:
    """Generate all six publication-quality figures."""
    from scripts.plotting import create_all_figures

    _step(10, "Generating figures")
    create_all_figures(config, df_clim, df_clusters)


# ---------------------------------------------------------------------------
# Only-plot mode (load existing CSVs)
# ---------------------------------------------------------------------------
def load_existing_results(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously computed seasonal stats and cluster labels from CSV.

    Used when ``--only-plot`` is specified.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    tuple
        (df_clim, df_clusters)
    """
    figures_dir = Path(config["paths"]["figures"])
    stats_path = figures_dir / "seasonal_stats.csv"
    clusters_path = figures_dir / "cluster_labels.csv"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"seasonal_stats.csv not found at {stats_path}. "
            "Run the full pipeline first."
        )

    df_clim = pd.read_csv(stats_path)
    df_clusters = pd.read_csv(clusters_path) if clusters_path.exists() else df_clim.copy()

    logger.info("Loaded %d seasonal records from %s", len(df_clim), stats_path)
    return df_clim, df_clusters


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------
def main() -> int:
    """
    Orchestrate the full avalanche climatology pipeline.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description="Austrian Alps avalanche climatology pipeline — Reuter et al. (2023) method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip ERA5 and SPARTACUS downloads (use existing files)"
    )
    parser.add_argument(
        "--skip-snowpack", action="store_true",
        help="Skip SNOWPACK simulations (use existing PRO files)"
    )
    parser.add_argument(
        "--skip-avapro", action="store_true",
        help="Skip AVAPRO classification (load existing CSVs from avapro_output/)"
    )
    parser.add_argument(
        "--only-plot", action="store_true",
        help="Load existing seasonal_stats.csv and regenerate figures only"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel SNOWPACK workers (-1 = all CPUs)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    # --- Load config ---
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    # --- Setup logging ---
    log_dir = Path(config["paths"]["logs"])
    _setup_logging(args.log_level, log_dir)

    logger.info("=" * 70)
    logger.info("  AVALANCHE CLIMATOLOGY PIPELINE — Austrian Alps")
    logger.info("  Following Reuter et al. (2023) Cold Regions Sci. Tech.")
    logger.info("=" * 70)

    t_start = time.perf_counter()

    try:
        if args.only_plot:
            # Shortcut: load pre-computed results and regenerate figures
            logger.info("--only-plot mode: loading existing results.")
            df_clim, df_clusters = load_existing_results(config)
            stage_plotting(config, df_clim, df_clusters)

        else:
            # Full pipeline
            if not args.skip_download:
                stage_download(config)
            else:
                logger.info("Skipping data download (--skip-download).")

            corrected = stage_preprocess(config)

            ini_paths = stage_write_snowpack_input(config, corrected)

            if not args.skip_snowpack:
                stage_run_snowpack(config, ini_paths, n_jobs=args.n_jobs)
            else:
                logger.info("Skipping SNOWPACK runs (--skip-snowpack).")

            if not args.skip_avapro:
                problems = stage_run_avapro(config)
            else:
                logger.info("Skipping AVAPRO (--skip-avapro); loading existing CSVs.")
                problems = _load_avapro_csvs(config)

            df_clim, df_clusters = stage_climatology(config, problems)
            stage_plotting(config, df_clim, df_clusters)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    logger.info("=" * 70)
    logger.info("  Pipeline complete — elapsed: %s", _elapsed(t_start))
    logger.info(
        "  Outputs: figures/ and %s", Path(config["paths"]["figures"]).resolve()
    )
    logger.info("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# AVAPRO CSV loader (for --skip-avapro mode)
# ---------------------------------------------------------------------------
def _load_avapro_csvs(config: dict) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Load previously saved AVAPRO / heuristic CSV files.

    Used when ``--skip-avapro`` is specified.
    """
    avapro_dir = Path(config["paths"]["avapro_output"])
    problems: dict[str, dict[int, pd.DataFrame]] = {}

    for region_key in config["regions"]:
        problems[region_key] = {}
        for elev_m in config["elevation_bands"]:
            station_id = f"{region_key}_{elev_m}m"
            csv_path = avapro_dir / f"{station_id}_problems.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                df = df.astype(bool)
                problems[region_key][elev_m] = df
            else:
                logger.warning(
                    "AVAPRO CSV not found for %s @ %d m: %s", region_key, elev_m, csv_path
                )

    return problems


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
