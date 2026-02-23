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

Note: glide_snow is excluded — AVAPRO cannot reliably classify it.

AVAPRO command line
-------------------
The exact CLI depends on the AWSoM installation version.  The default
invocation assumed here is:

    avapro --input  <pro_file>
           --output <output_csv>
           [--config <config_file>]

Output CSV / DataFrame columns (AVAPRO-native names)
-----------------------------------------------------
    date,
    napex_sele_trigger, napex_sele_natural,   (new snow)
    winex,                                     (wind slab)
    papex_sele_trigger, papex_sele_natural,   (shallow persistent WL: F, Fk, SH)
    dapex_sele_trigger, dapex_sele_natural,   (deep persistent WL: DH)
    wapex_sele                                 (wet snow)
    glide_snow excluded — not reliably classifiable by AVAPRO

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
- glide_snow  : excluded — not classifiable from flat-field profiles

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

# SNOWPACK PRO grain type: Swiss Code F1F2F3 (3-digit integer).
# F1 (hundreds digit) encodes the primary grain form:
#   1=PP, 2=DF, 3=RG, 4=FC, 5=DH, 6=SH, 7=MF, 8=IF
# Shallow persistent WL: FC (4xx) and SH (6xx) → papex
# Deep persistent WL:    DH (5xx)              → dapex
_SHALLOW_PWL_F1: set[int] = {4, 6}   # faceted crystals, surface hoar
_DEEP_PWL_F1: set[int]    = {5}      # depth hoar

# Depth threshold below surface for PWL detection [m]
_PWL_MIN_DEPTH: float = 0.20

# Wet snow (AVAPRO decision tree — Mitterer et al. 2016)
# First-cycle onset threshold for LWCindex [%]
_WET_SNOW_THRESHOLD_1: float = 0.33
# Subsequent-cycle threshold after the snowpack has dried
_WET_SNOW_THRESHOLD_N: float = 1.0
# Number of consecutive isothermal days to end a wet snow cycle
_WET_SNOW_ISOTHERMAL_DAYS: int = 3
# Isothermal: all layers within this many °C of 0
_ISOTHERMAL_TOL: float = 0.5


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

    # Map AVAPRO native column names → our standard names.
    # Native names come from visually_process_aps.py / find_aps.py.
    # col_map in config can override if a different AVAPRO version is used.
    native_cols = {
        "napex_sele_trigger": col_map.get("napex_sele_trigger", "napex_sele_trigger"),
        "napex_sele_natural": col_map.get("napex_sele_natural", "napex_sele_natural"),
        "winex":              col_map.get("winex",              "winex"),
        "papex_sele_trigger": col_map.get("papex_sele_trigger", "papex_sele_trigger"),
        "papex_sele_natural": col_map.get("papex_sele_natural", "papex_sele_natural"),
        "dapex_sele_trigger": col_map.get("dapex_sele_trigger", "dapex_sele_trigger"),
        "dapex_sele_natural": col_map.get("dapex_sele_natural", "dapex_sele_natural"),
        "wapex_sele":         col_map.get("wapex_sele",         "wapex_sele"),
    }

    result = pd.DataFrame(index=df_raw.index)
    for std_name, raw_name in native_cols.items():
        if raw_name in df_raw.columns:
            result[std_name] = (df_raw[raw_name].fillna(0) > _AVAPRO_THRESHOLD).astype(bool)
        else:
            logger.warning(
                "Column '%s' not found in AVAPRO output %s; defaulting to False.",
                raw_name, csv_path.name,
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

    # ── Wet snow state machine (Mitterer et al. 2016) ────────────────────────
    # threshold cycles: first onset at 0.33 %, subsequent cycles at 1.0 %
    wet_threshold   = _WET_SNOW_THRESHOLD_1
    wet_active      = False   # currently flagging wet snow problem
    iso_streak      = 0       # consecutive isothermal days

    rows = []
    hs_prev = 0.0

    for date, rec in sorted(records.items()):
        hs     = rec["hs_m"]
        layers = rec["layers"]

        # ── Snow height guard ──────────────────────────────────────────────
        if hs < _MIN_HS_FOR_CLASSIFICATION:
            hs_prev = hs
            rows.append({
                "date": date,
                "napex_sele_trigger": False, "napex_sele_natural": False,
                "winex":              False,
                "papex_sele_trigger": False, "papex_sele_natural": False,
                "dapex_sele_trigger": False, "dapex_sele_natural": False,
                "wapex_sele":         False,
            })
            continue

        # ── LWCindex: thickness-weighted mean LWC [%] ─────────────────────
        total_thick = sum(l["thickness_m"] for l in layers)
        if total_thick > 0:
            lwc_index = sum(l["lwc_pct"] * l["thickness_m"] for l in layers) / total_thick
        else:
            lwc_index = 0.0

        # ── Isothermal state: all layers within _ISOTHERMAL_TOL of 0 °C ──
        is_isothermal = bool(layers) and all(
            abs(l["temp_c"]) <= _ISOTHERMAL_TOL for l in layers
        )
        iso_streak = (iso_streak + 1) if is_isothermal else 0

        # ── Wet snow state machine ─────────────────────────────────────────
        if not wet_active:
            if lwc_index > wet_threshold:
                wet_active = True
        else:
            # Problem ends if LWCindex drops below threshold OR 3 consecutive
            # isothermal days (snowpack fully wet and stable — AVAPRO criterion)
            if lwc_index <= wet_threshold or iso_streak >= _WET_SNOW_ISOTHERMAL_DAYS:
                wet_active    = False
                wet_threshold = _WET_SNOW_THRESHOLD_N   # raise bar for next cycle

        # ── New snow ───────────────────────────────────────────────────────
        delta_hs = max(0.0, hs - hs_prev)
        new_snow = delta_hs >= _NEW_SNOW_THRESHOLD

        # ── Wind slab proxy (no wind in PRO; use new_snow as stand-in) ────
        # Real AVAPRO uses wind transport index from meteo data.
        wind_slab = new_snow

        # ── Persistent weak layers (from grain type F1 digit) ─────────────
        papex = False
        dapex = False
        depth_from_surface = 0.0
        for layer in reversed(layers):   # top → bottom
            depth_from_surface += layer["thickness_m"]
            if depth_from_surface >= _PWL_MIN_DEPTH:
                f1 = layer["grain_f1"]
                if f1 in _SHALLOW_PWL_F1:
                    papex = True
                elif f1 in _DEEP_PWL_F1:
                    dapex = True

        # Natural release cannot be determined from flat-field heuristic
        rows.append({
            "date":               date,
            "napex_sele_trigger": new_snow,
            "napex_sele_natural": False,
            "winex":              wind_slab,
            "papex_sele_trigger": papex,
            "papex_sele_natural": False,
            "dapex_sele_trigger": dapex,
            "dapex_sele_natural": False,
            "wapex_sele":         wet_active,
        })
        hs_prev = hs

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df = df.astype(bool)
    return df


def _parse_pro_file(pro_path: Path) -> dict:
    """
    Parse a SNOWPACK PRO file into a dictionary of daily profile records.

    PRO record codes used
    ----------------------
    0500 : Date and time  (DD.MM.YYYY HH:MM:SS)
    0501 : Cumulative element heights from base [cm]  — last value = total HS
    0503 : Layer temperature [°C]  (one value per element)
    0506 : Liquid water content by volume [%]  (one per element)
    0513 : Grain type, Swiss Code F1F2F3  (3-digit integer)
           F1 hundreds digit: 1=PP, 2=DF, 3=RG, 4=FC, 5=DH, 6=SH, 7=MF, 8=IF

    Layer thickness is derived from consecutive 0501 heights.
    Elements are ordered bottom → top.

    Returns
    -------
    dict
        ``records[datetime.date] = {hs_m, layers}``.
        Each layer: {thickness_m, temp_c, lwc_pct, grain_f1}
    """
    import datetime

    records: dict = {}
    current_date = None
    current_rec: dict = {}

    with open(pro_path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "[DATA]":
                continue

            parts = line.split(",")
            if len(parts) < 2:
                continue

            code = parts[0].strip()

            # ── Date record ─────────────────────────────────────────────────
            if code == "0500":
                if current_date is not None and current_rec:
                    records[current_date] = current_rec
                try:
                    # Date+time may be in one field ("DD.MM.YYYY HH:MM:SS")
                    # or split across two fields by a comma
                    date_str = (parts[1].strip() + " " + parts[2].strip()
                                if len(parts) > 2 else parts[1].strip())
                    dt = datetime.datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
                    current_date = dt.date()
                    current_rec = {"hs_m": 0.0, "layers": []}
                except (ValueError, IndexError):
                    current_date = None
                    current_rec = {}

            elif current_date is None:
                continue

            # ── Element heights → HS + layer thicknesses ─────────────────
            elif code == "0501":
                try:
                    heights = [float(p) for p in parts[2:] if p.strip() not in ("", "-999")]
                    # heights are cumulative from base [cm], bottom→top
                    if heights:
                        current_rec["hs_m"] = heights[-1] / 100.0
                        thicknesses = []
                        prev = 0.0
                        for h in heights:
                            thicknesses.append((h - prev) / 100.0)  # cm → m
                            prev = h
                        # initialise layer dicts
                        current_rec["layers"] = [{"thickness_m": t, "temp_c": 0.0,
                                                   "lwc_pct": 0.0, "grain_f1": 0}
                                                  for t in thicknesses]
                except (ValueError, IndexError):
                    pass

            # ── Layer temperature [°C] ────────────────────────────────────
            elif code == "0503":
                vals = []
                for p in parts[2:]:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
                for idx, v in enumerate(vals):
                    if idx < len(current_rec["layers"]):
                        current_rec["layers"][idx]["temp_c"] = v

            # ── LWC by volume [%] ─────────────────────────────────────────
            elif code == "0506":
                vals = []
                for p in parts[2:]:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
                for idx, v in enumerate(vals):
                    if idx < len(current_rec["layers"]):
                        current_rec["layers"][idx]["lwc_pct"] = max(0.0, v)

            # ── Grain type (Swiss Code F1F2F3) ────────────────────────────
            elif code == "0513":
                vals = []
                for p in parts[2:]:
                    try:
                        vals.append(int(float(p)))
                    except ValueError:
                        pass
                for idx, v in enumerate(vals):
                    if idx < len(current_rec["layers"]):
                        current_rec["layers"][idx]["grain_f1"] = v // 100  # hundreds digit

    if current_date is not None and current_rec:
        records[current_date] = current_rec

    return records


def _empty_problem_df() -> pd.DataFrame:
    """Return an empty problem DataFrame with AVAPRO-native column schema."""
    return pd.DataFrame(
        columns=[
            "napex_sele_trigger", "napex_sele_natural",
            "winex",
            "papex_sele_trigger", "papex_sele_natural",
            "dapex_sele_trigger", "dapex_sele_natural",
            "wapex_sele",
        ],
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
