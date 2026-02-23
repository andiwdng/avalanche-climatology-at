"""
test_year2000.py
================
End-to-end pipeline test using the downloaded year 2000 ERA5-Land data.

Runs a reduced pipeline (2 regions × 2 elevation bands) to exercise
every stage up to and including the heuristic AVAPRO classifier, without
requiring SPARTACUS bias correction data.

Stages
------
1. Interpolate ERA5 year 2000 to point locations
2. Write SNOWPACK SMET + SNO + INI files
3. Run SNOWPACK simulations
4. Run heuristic classifier on PRO output
5. Print summary statistics
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build a minimal test config (subset of regions, year 2000 only)
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

# Override simulation period: only year 2000, no spin-up
cfg = copy.deepcopy(cfg)
cfg["simulation"]["analysis_start"] = "2000-01-01"
cfg["simulation"]["analysis_end"]   = "2000-12-31"
cfg["simulation"]["spin_up_years"]  = 0

# Use only 2 representative regions to keep run time manageable
TEST_REGIONS = ["arlberg", "hohe_tauern"]
cfg["regions"] = {k: v for k, v in cfg["regions"].items() if k in TEST_REGIONS}

# 2 elevation bands
cfg["elevation_bands"] = [2000, 3000]

# Fix SNOWPACK binary path (installed at non-default location)
cfg["snowpack"]["binary"] = "/Applications/Snowpack/bin/snowpack"

# Use a separate output prefix so we don't collide with production data
cfg["paths"]["snowpack_input"]  = "data/snowpack_input_test"
cfg["paths"]["snowpack_output"] = "data/snowpack_output_test"
cfg["paths"]["avapro_output"]   = "data/avapro_output_test"

logger.info("Test config: regions=%s  elevations=%s  period=%s → %s",
            list(cfg["regions"]), cfg["elevation_bands"],
            cfg["simulation"]["analysis_start"],
            cfg["simulation"]["analysis_end"])

# ---------------------------------------------------------------------------
# Stage 1: Interpolate ERA5 to points
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 1: Interpolating ERA5 to points")
logger.info("=" * 60)

from scripts.interpolate_points import interpolate_era5_to_points

era5_data = interpolate_era5_to_points(cfg)

for rkey, elev_dict in era5_data.items():
    for elev, df in elev_dict.items():
        ta = df["TA"].mean() - 273.15
        psum = df["PSUM"].sum()
        logger.info("  %s @ %dm : %d h  TA_mean=%.1f°C  PSUM=%.0f mm",
                    rkey, elev, len(df), ta, psum)

# ---------------------------------------------------------------------------
# Stage 2: Write SNOWPACK input files (no bias correction — use raw ERA5)
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 2: Writing SNOWPACK SMET / SNO / INI files")
logger.info("=" * 60)

from scripts.snowpack_writer import write_all_snowpack_inputs

ini_paths = write_all_snowpack_inputs(cfg, era5_data)

for rkey, elev_dict in ini_paths.items():
    for elev, ini_path in elev_dict.items():
        smet = Path(cfg["paths"]["snowpack_input"]) / "smet" / f"{rkey}_{elev}m.smet"
        logger.info("  Written: %s  (smet: %.1f KB)",
                    ini_path.name, smet.stat().st_size / 1024 if smet.exists() else 0)

# ---------------------------------------------------------------------------
# Stage 3: Run SNOWPACK simulations
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 3: Running SNOWPACK simulations")
logger.info("=" * 60)

from scripts.run_snowpack import run_snowpack_simulations

success = run_snowpack_simulations(cfg, ini_paths, n_jobs=1)

for rkey, elev_dict in success.items():
    for elev, ok in elev_dict.items():
        status = "✓ OK" if ok else "✗ FAILED"
        logger.info("  %s @ %dm : %s", rkey, elev, status)

n_ok = sum(v for r in success.values() for v in r.values())
if n_ok == 0:
    logger.error("All SNOWPACK simulations failed — check logs/")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stage 4: Heuristic avalanche problem classification
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 4: Heuristic avalanche problem classification")
logger.info("=" * 60)

from scripts.run_avapro import run_avapro_all
from scripts.run_snowpack import find_pro_files

pro_files = find_pro_files(cfg)
problems = run_avapro_all(cfg, pro_files)

import pandas as pd
problems_df = pd.concat(
    [df.assign(region=rk, elevation=elev)
     for rk, elev_dict in problems.items()
     for elev, df in elev_dict.items()
     if df is not None and not df.empty],
    ignore_index=True,
) if any(df is not None and not df.empty
         for elev_dict in problems.values()
         for df in elev_dict.values()) else pd.DataFrame()

logger.info("Problems DataFrame shape: %s", problems_df.shape)
if not problems_df.empty:
    logger.info("Columns: %s", list(problems_df.columns))

# ---------------------------------------------------------------------------
# Stage 5: Summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Stage 5: Summary")
logger.info("=" * 60)

if not problems_df.empty:
    for rkey in TEST_REGIONS:
        for elev in cfg["elevation_bands"]:
            sub = problems_df[
                (problems_df["region"] == rkey) &
                (problems_df["elevation"] == elev)
            ]
            if sub.empty:
                continue
            # Count problem days using AVAPRO-native column names
            ns_days   = sub["napex_sele_trigger"].sum() if "napex_sele_trigger" in sub else 0
            ws_days   = sub["winex"].sum()              if "winex"              in sub else 0
            pap_days  = sub["papex_sele_trigger"].sum() if "papex_sele_trigger" in sub else 0
            dap_days  = sub["dapex_sele_trigger"].sum() if "dapex_sele_trigger" in sub else 0
            wet_days  = sub["wapex_sele"].sum()         if "wapex_sele"         in sub else 0
            logger.info(
                "  %s @ %dm : %d days | NS=%d  WS=%d  PW=%d  deepPW=%d  Wet=%d",
                rkey, elev, len(sub),
                int(ns_days), int(ws_days), int(pap_days), int(dap_days), int(wet_days),
            )

    # Show PRO file sizes as proxy for snow activity
    for rkey, elev_dict in pro_files.items():
        for elev, pro_path in elev_dict.items():
            if pro_path.exists():
                logger.info("  PRO file: %s  (%.0f KB)", pro_path.name,
                            pro_path.stat().st_size / 1024)

logger.info("=" * 60)
logger.info("Test complete.")
