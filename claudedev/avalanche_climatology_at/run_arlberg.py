"""
run_arlberg.py
==============
Full pipeline run for the Arlberg region across all available ERA5 years.

Stages
------
1. Interpolate all available ERA5 years to Arlberg elevation bands
2. Write SNOWPACK input files (SMET + SNO + INI)
3. Run SNOWPACK simulations
4. Run heuristic AVAPRO classification
5. Print summary

Usage
-----
    python run_arlberg.py [--n-jobs N]

The script skips years where ERA5 data is missing — run download_era5.py
first to fetch additional years, then re-run this script.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n-jobs", type=int, default=2,
                    help="Parallel SNOWPACK jobs (default 2; use 1 on low-RAM machines)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build Arlberg-only config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

cfg = copy.deepcopy(cfg)
cfg["regions"]         = {"arlberg": cfg["regions"]["arlberg"]}
cfg["elevation_bands"] = [1500, 2000, 2500, 3000]

# Production output directories (separate from test)
cfg["paths"]["snowpack_input"]  = "data/snowpack_input_arlberg"
cfg["paths"]["snowpack_output"] = "data/snowpack_output_arlberg"
cfg["paths"]["avapro_output"]   = "data/avapro_output_arlberg"

cfg["snowpack"]["binary"] = "/Applications/Snowpack/bin/snowpack"

logger.info("=" * 60)
logger.info("Arlberg pipeline run")
logger.info("  Elevation bands : %s", cfg["elevation_bands"])
logger.info("  Analysis period : %s → %s",
            cfg["simulation"]["analysis_start"],
            cfg["simulation"]["analysis_end"])
logger.info("  Spin-up years   : %d", cfg["simulation"]["spin_up_years"])
logger.info("  n_jobs          : %d", args.n_jobs)
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# Stage 1 — Interpolate ERA5 to points
# ---------------------------------------------------------------------------
logger.info("Stage 1: Interpolating ERA5 to Arlberg elevation bands")
from scripts.interpolate_points import interpolate_era5_to_points

era5_data = interpolate_era5_to_points(cfg)

n_years = 0
for rk, elev_dict in era5_data.items():
    for elev, df in elev_dict.items():
        years_present = df.index.year.unique().tolist()
        n_years = max(n_years, len(years_present))
        logger.info("  %s @ %dm : %d hours  years=%s",
                    rk, elev, len(df), years_present)

if n_years == 0:
    logger.error("No ERA5 data available — download ERA5 first with scripts/download_era5.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stage 2 — Write SNOWPACK inputs
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 2: Writing SNOWPACK input files")
from scripts.snowpack_writer import write_all_snowpack_inputs

ini_paths = write_all_snowpack_inputs(cfg, era5_data)

for rk, elev_dict in ini_paths.items():
    for elev, ini_path in elev_dict.items():
        smet = (Path(cfg["paths"]["snowpack_input"]) / "smet"
                / f"{rk}_{elev}m.smet")
        logger.info("  %s_%dm.ini  (smet: %.1f KB)",
                    rk, elev,
                    smet.stat().st_size / 1024 if smet.exists() else 0)

# ---------------------------------------------------------------------------
# Stage 3 — SNOWPACK simulations
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 3: Running SNOWPACK simulations (n_jobs=%d)", args.n_jobs)
from scripts.run_snowpack import run_snowpack_simulations

success = run_snowpack_simulations(cfg, ini_paths, n_jobs=args.n_jobs)

n_ok = 0
for rk, elev_dict in success.items():
    for elev, ok in elev_dict.items():
        status = "OK" if ok else "FAILED"
        logger.info("  %s @ %dm : %s", rk, elev, status)
        if ok:
            n_ok += 1

if n_ok == 0:
    logger.error("All SNOWPACK simulations failed.")
    sys.exit(1)

logger.info("%d / %d simulations succeeded.", n_ok, sum(len(v) for v in success.values()))

# ---------------------------------------------------------------------------
# Stage 4 — AVAPRO classification
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 4: AVAPRO heuristic classification")
from scripts.run_avapro import run_avapro_all
from scripts.run_snowpack import find_pro_files

pro_files = find_pro_files(cfg)
problems  = run_avapro_all(cfg, pro_files)

frames = []
for rk, elev_dict in problems.items():
    for elev, df in elev_dict.items():
        if df is not None and not df.empty:
            frames.append(df.assign(region=rk, elevation=elev))

problems_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ---------------------------------------------------------------------------
# Stage 5 — Summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 5: Summary")

if not problems_df.empty:
    # Per elevation band
    for elev in cfg["elevation_bands"]:
        sub = problems_df[problems_df["elevation"] == elev]
        if sub.empty:
            continue
        ns  = int(sub["napex_sele_trigger"].sum())
        ws  = int(sub["winex"].sum())
        pw  = int(sub["papex_sele_trigger"].sum())
        dpw = int(sub["dapex_sele_trigger"].sum())
        wet = int(sub["wapex_sele"].sum())
        logger.info("  Arlberg @%dm : %d days | NS=%d  WS=%d  PW=%d  deepPW=%d  Wet=%d",
                    elev, len(sub), ns, ws, pw, dpw, wet)

    # Save combined CSV
    out_csv = Path(cfg["paths"]["avapro_output"]) / "arlberg_all_elevations.csv"
    problems_df.to_csv(out_csv, index=False)
    logger.info("Combined results saved to %s", out_csv)

logger.info("=" * 60)
logger.info("Arlberg pipeline run complete.")
