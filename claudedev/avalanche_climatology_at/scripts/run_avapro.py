"""
run_avapro.py
=============
Wrapper for the AVAPRO (AWSoM toolchain) avalanche problem classifier.
Runs AVAPRO on SNOWPACK PRO profile files and parses the output into
tidy pandas DataFrames.

AVAPRO overview
---------------
AVAPRO (Avalanche Problem Classifier) is part of the AWSoM (Automatic
Warning System Model) toolchain developed by the Austrian Avalanche
Warning Services in collaboration with the WSL Institute for Snow and
Avalanche Research (SLF).

AVAPRO analyses SNOWPACK layered snow profiles (PRO format) and classifies
daily avalanche problem types according to the criteria in:
    Schweizer, J., Mitterer, C., Reuter, B., & Techel, F. (2020).
    Optimizing consistency of avalanche danger rating with a
    statistically-based approach. Cold Regions Science and Technology,
    175, 103030. https://doi.org/10.1016/j.coldregions.2020.103030

Avalanche problem types classified
-----------------------------------
1. new_snow      — recent snowfall with poor bonding
2. wind_slab     — wind-deposited slab on hard faceted or weak layer
3. persistent_wl — persistent weak layer (facets, depth hoar, surface hoar)
4. wet_snow      — free water in snowpack triggering wet avalanches
5. glide_snow    — snow gliding on smooth ground

AVAPRO command line
-------------------
The exact CLI depends on the AWSoM installation version.  The default
invocation assumed here is:

    avapro --input  <pro_file>
           --output <output_csv>
           [--config <config_file>]

Output CSV columns expected
----------------------------
    date, new_snow, wind_slab, persistent_wl, wet_snow, glide_snow
    (plus additional columns that are ignored)

Values are boolean (0/1) or continuous probabilities [0, 1].  This
wrapper normalises both to boolean by thresholding at 0.5.

If AVAPRO is not available or the output format differs, the parser
falls back to a SNOWPACK PRO-based heuristic classifier that estimates
problem types from stability indices embedded in the PRO file (see
``_heuristic_classify_pro``).

PRO file heuristic fallback
----------------------------
When AVAPRO is unavailable, the following heuristics are applied:
- new_snow    : fresh precipitation > 3 cm in 24 h
- wind_slab   : new snow AND |ΔHS| / ΔT differs between aspects (proxy: wind > 5 m/s AND new snow)
- persistent_wl: presence of facets (grain type F, Fk, DH) at depth > 20 cm
- wet_snow    : free water content (LWC) > 1 % in any layer (or surface > 0°C)
- glide_snow  : wet snow AND low slope (< 15°) — not applicable for flat field

References
----------
Reuter, B., Viallon-Galinier, L., Horton, S., van Herwijnen, A.,
    Hagenmuller, P., Morin, S., & Schweizer, J. (2023).
    Characterizing snow instability with avalanche problem types derived
    from snow cover simulations. Cold Regions Science and Technology,
    207, 103772.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Threshold for binarising continuous AVAPRO output
_AVAPRO_THRESHOLD: float = 0.5

# Minimum snow depth for problem classification [m]
_MIN_HS_FOR_CLASSIFICATION: float = 0.10

# New snow threshold for new_snow problem [m per day]
_NEW_SNOW_THRESHOLD: float = 0.03

# Wind speed threshold for wind slab proxy [m/s]
_WIND_SLAB_WIND_THRESHOLD: float = 5.0

# Grain type codes for facets/depth hoar in SNOWPACK PRO (F=8, Fk=9, DH=10)
_PWL_GRAIN_TYPES: set[int] = {8, 9, 10, 11}  # facets, kinetic, depth hoar, surface hoar

# Depth threshold below surface for PWL detection [m]
_PWL_MIN_DEPTH: float = 0.20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_avapro_all(
    config: dict,
    pro_files: dict[str, dict[int, Path]],
) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Run AVAPRO on all SNOWPACK PRO files and return daily problem DataFrames.

    Attempts to use the AVAPRO binary specified in config.yaml.  If the
    binary is not found or returns an error, falls back to the PRO-based
    heuristic classifier.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    pro_files : dict
        ``pro_files[region][elev] = Path`` to SNOWPACK PRO file.

    Returns
    -------
    dict
        ``problems[region][elev] = DataFrame`` with daily boolean columns:
        new_snow, wind_slab, persistent_wl, wet_snow, glide_snow.
    """
    avapro_cfg = config["avapro"]
    avapro_binary = avapro_cfg["binary"]
    avapro_output_dir = Path(config["paths"]["avapro_output"])
    avapro_output_dir.mkdir(parents=True, exist_ok=True)

    avapro_available = _check_avapro_binary(avapro_binary)
    if not avapro_available:
        logger.warning(
            "AVAPRO binary '%s' not found. Using PRO heuristic classifier.", avapro_binary
        )

    problems: dict[str, dict[int, pd.DataFrame]] = {}

    for region_key, elev_dict in pro_files.items():
        problems[region_key] = {}
        for elev_m, pro_path in elev_dict.items():
            station_id = pro_path.stem
            output_csv = avapro_output_dir / f"{station_id}_problems.csv"

            if avapro_available:
                success = _run_avapro_subprocess(
                    avapro_binary=avapro_binary,
                    pro_path=pro_path,
                    output_csv=output_csv,
                )
                if success and output_csv.exists():
                    df = _parse_avapro_csv(output_csv, avapro_cfg)
                else:
                    logger.warning(
                        "AVAPRO failed for %s — falling back to heuristic.",
                        station_id,
                    )
                    df = _heuristic_classify_pro(pro_path)
            else:
                df = _heuristic_classify_pro(pro_path)
                # Save heuristic output to CSV for reproducibility
                df.to_csv(output_csv)

            problems[region_key][elev_m] = df
            logger.info(
                "AVAPRO %s @ %d m: %d problem-days classified.",
                region_key,
                elev_m,
                df.any(axis=1).sum(),
            )

    return problems


# ---------------------------------------------------------------------------
# AVAPRO subprocess
# ---------------------------------------------------------------------------
def _check_avapro_binary(binary_path: str) -> bool:
    """
    Check whether the AVAPRO binary is accessible.

    Parameters
    ----------
    binary_path : str
        Filesystem path or name of the AVAPRO executable.

    Returns
    -------
    bool
        True if the binary can be located.
    """
    try:
        proc = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False


def _run_avapro_subprocess(
    avapro_binary: str,
    pro_path: Path,
    output_csv: Path,
) -> bool:
    """
    Invoke AVAPRO on a single PRO file.

    Parameters
    ----------
    avapro_binary : str
        Path to the AVAPRO executable.
    pro_path : Path
        SNOWPACK PRO file to classify.
    output_csv : Path
        Destination CSV for AVAPRO output.

    Returns
    -------
    bool
        True if AVAPRO exited with return code 0.
    """
    cmd = [
        avapro_binary,
        "--input", str(pro_path.resolve()),
        "--output", str(output_csv.resolve()),
    ]
    logger.debug("AVAPRO command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            logger.error("AVAPRO error for %s:\n%s", pro_path.name, proc.stderr[:1000])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("AVAPRO timed out for %s.", pro_path.name)
        return False


# ---------------------------------------------------------------------------
# AVAPRO CSV parser
# ---------------------------------------------------------------------------
def _parse_avapro_csv(
    csv_path: Path,
    avapro_cfg: dict,
) -> pd.DataFrame:
    """
    Parse AVAPRO output CSV into a standardised boolean DataFrame.

    Parameters
    ----------
    csv_path : Path
        AVAPRO output CSV file.
    avapro_cfg : dict
        AVAPRO section of config.yaml (contains expected column names).

    Returns
    -------
    pd.DataFrame
        Daily DataFrame with DatetimeIndex and boolean columns:
        new_snow, wind_slab, persistent_wl, wet_snow, glide_snow.
    """
    col_map = avapro_cfg.get("problem_columns", {})

    df_raw = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    problem_cols = {
        "new_snow": col_map.get("new_snow", "new_snow"),
        "wind_slab": col_map.get("wind_slab", "wind_slab"),
        "persistent_wl": col_map.get("persistent_wl", "persistent_wl"),
        "wet_snow": col_map.get("wet_snow", "wet_snow"),
        "glide_snow": col_map.get("glide_snow", "glide_snow"),
    }

    result = pd.DataFrame(index=df_raw.index, dtype=bool)
    for std_name, raw_name in problem_cols.items():
        if raw_name in df_raw.columns:
            result[std_name] = df_raw[raw_name].values > _AVAPRO_THRESHOLD
        else:
            logger.warning(
                "Column '%s' not found in AVAPRO output %s; defaulting to False.",
                raw_name,
                csv_path.name,
            )
            result[std_name] = False

    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# Heuristic PRO classifier (fallback)
# ---------------------------------------------------------------------------
def _heuristic_classify_pro(pro_path: Path) -> pd.DataFrame:
    """
    Classify avalanche problems from SNOWPACK PRO data without AVAPRO.

    This is a scientific heuristic, not a replacement for AVAPRO.  It
    implements simplified decision rules based on the stability criteria
    described in Reuter et al. (2023).

    Parsed PRO records
    ------------------
    0500  Date / time
    0501  Snow height HS [cm]
    0502  Air temperature TA [°C]
    0503  Liquid water content [%] per layer
    0508  Layer grain type (SNOWPACK grain type code)
    0510  Layer thickness [m]

    Returns
    -------
    pd.DataFrame
        Daily boolean DataFrame with DatetimeIndex.
    """
    logger.info("Parsing PRO file for heuristic classification: %s", pro_path.name)

    records = _parse_pro_file(pro_path)

    if not records:
        logger.warning("No records parsed from %s.", pro_path.name)
        return _empty_problem_df()

    rows = []
    hs_prev = 0.0

    for date, rec in sorted(records.items()):
        hs = rec.get("hs", 0.0)            # m
        ta = rec.get("ta", -5.0)           # °C
        layers = rec.get("layers", [])

        # --- New snow ---
        delta_hs = max(0.0, hs - hs_prev)
        new_snow = (delta_hs >= _NEW_SNOW_THRESHOLD) and (hs >= _MIN_HS_FOR_CLASSIFICATION)

        # --- Wind slab ---
        vw = rec.get("vw", 0.0)
        wind_slab = new_snow and (vw >= _WIND_SLAB_WIND_THRESHOLD)

        # --- Persistent weak layer ---
        pwl = False
        cumulative_depth = 0.0
        for layer in layers:
            grain_type = int(layer.get("grain_type", 0))
            thickness = float(layer.get("thickness", 0.0))
            cumulative_depth += thickness
            if grain_type in _PWL_GRAIN_TYPES and cumulative_depth >= _PWL_MIN_DEPTH:
                pwl = True
                break

        # --- Wet snow ---
        any_lwc = any(float(l.get("lwc", 0.0)) > 1.0 for l in layers)
        wet_snow = (ta > -0.5) and any_lwc and (hs >= _MIN_HS_FOR_CLASSIFICATION)

        # --- Glide snow (not applicable for flat-field) ---
        glide_snow = False

        rows.append(
            {
                "date": date,
                "new_snow": new_snow,
                "wind_slab": wind_slab,
                "persistent_wl": pwl,
                "wet_snow": wet_snow,
                "glide_snow": glide_snow,
            }
        )
        hs_prev = hs

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df = df.astype(bool)
    return df


def _parse_pro_file(pro_path: Path) -> dict:
    """
    Parse a SNOWPACK PRO file into a dictionary of daily profile records.

    The PRO format uses numeric codes to identify record types.  Only
    the records required for heuristic avalanche problem classification
    are parsed.

    Key record codes
    ----------------
    0500 : Date and time
    0501 : Snow height [cm]  (last column)
    0502 : Air temperature [°C]
    0506 : Wind speed (proxy) [m/s]  — not always present
    0503 : Free water content per layer [%] (comma-separated list)
    0508 : Grain type per layer (comma-separated integer list)
    0510 : Layer thickness per layer [m] (comma-separated)

    Parameters
    ----------
    pro_path : Path
        Path to the SNOWPACK PRO file.

    Returns
    -------
    dict
        ``records[datetime.date] = {hs, ta, vw, layers}``.
    """
    records: dict = {}
    current_date = None
    current_rec: dict = {}

    with open(pro_path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) < 2:
                continue

            code = parts[0].strip()

            if code == "0500":
                # Date/time record: save previous record
                if current_date is not None and current_rec:
                    records[current_date] = current_rec

                # Parse date: format is DD.MM.YYYY HH:MM
                try:
                    import datetime
                    date_str = parts[1].strip() + " " + parts[2].strip() if len(parts) > 2 else parts[1].strip()
                    dt = datetime.datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
                    current_date = dt.date()
                    current_rec = {"layers": [], "ta": -5.0, "hs": 0.0, "vw": 0.0}
                except (ValueError, IndexError):
                    current_date = None
                    current_rec = {}

            elif code == "0501" and current_date is not None:
                # Snow height [cm] — last non-empty column
                try:
                    vals = [p for p in parts[1:] if p.strip() not in ("", "-999")]
                    if vals:
                        current_rec["hs"] = float(vals[-1]) / 100.0  # cm → m
                except (ValueError, IndexError):
                    pass

            elif code == "0502" and current_date is not None:
                # Surface air temperature [°C]
                try:
                    current_rec["ta"] = float(parts[1])
                except (ValueError, IndexError):
                    pass

            elif code == "0506" and current_date is not None:
                # Wind speed [m/s]
                try:
                    current_rec["vw"] = float(parts[1])
                except (ValueError, IndexError):
                    pass

            elif code == "0508" and current_date is not None:
                # Grain type per layer
                grain_types = []
                for val in parts[1:]:
                    try:
                        grain_types.append(int(float(val)))
                    except ValueError:
                        pass
                # Attach to existing layers or create placeholder layers
                for idx, gt in enumerate(grain_types):
                    while len(current_rec["layers"]) <= idx:
                        current_rec["layers"].append({})
                    current_rec["layers"][idx]["grain_type"] = gt

            elif code == "0503" and current_date is not None:
                # Liquid water content [%] per layer
                for idx, val in enumerate(parts[1:]):
                    try:
                        lwc = float(val)
                    except ValueError:
                        continue
                    while len(current_rec["layers"]) <= idx:
                        current_rec["layers"].append({})
                    current_rec["layers"][idx]["lwc"] = lwc

            elif code == "0510" and current_date is not None:
                # Layer thickness [m]
                for idx, val in enumerate(parts[1:]):
                    try:
                        thick = float(val)
                    except ValueError:
                        continue
                    while len(current_rec["layers"]) <= idx:
                        current_rec["layers"].append({})
                    current_rec["layers"][idx]["thickness"] = thick

    # Append last record
    if current_date is not None and current_rec:
        records[current_date] = current_rec

    return records


def _empty_problem_df() -> pd.DataFrame:
    """Return an empty problem DataFrame with the correct column schema."""
    return pd.DataFrame(
        columns=["new_snow", "wind_slab", "persistent_wl", "wet_snow", "glide_snow"],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    from scripts.run_snowpack import find_pro_files
    pro_files = find_pro_files(cfg)
    problems = run_avapro_all(cfg, pro_files)
    for rkey, elev_dict in problems.items():
        for elev, df in elev_dict.items():
            print(f"{rkey} @ {elev} m: {len(df)} days, problem days: {df.any(axis=1).sum()}")
