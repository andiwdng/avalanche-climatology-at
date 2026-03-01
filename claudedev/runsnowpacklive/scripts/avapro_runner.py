# snowpack_steiermark/scripts/avapro_runner.py
"""
Run AVAPRO (snowpacktools) on SNOWPACK PRO output to classify avalanche problems.

AVAPRO (Assessment and Validation of AVAlanche Problems) processes SNOWPACK
.pro files and assigns prevailing avalanche problem types:
  - NAP  new snow (non-persistent)
  - WSAP wind slab
  - PAP  persistent weak layer
  - DAP  deep slab (was a PAP, now buried deeper)
  - WAP  wet snow

Reference: Reuter et al. (2022), Cold Regions Science and Technology, 194, 103462.
"""
from __future__ import annotations

import configparser
import logging
import os
import pickle
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from snowpacktools.avapro.avapro import avapro as _avapro_func
    _AVAPRO_AVAILABLE = True
except ImportError:
    _AVAPRO_AVAILABLE = False
    logger.warning("snowpacktools not installed — AVAPRO classification unavailable.")


class AvaPRORunner:
    """Runs AVAPRO on a SNOWPACK PRO file and saves daily problem flags to CSV."""

    def __init__(self, config: dict) -> None:
        self.config = config
        out_dir = Path(config["paths"]["data"]) / "avapro_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.csv_path = out_dir / "tamsichbachturm_problems.csv"

    # ------------------------------------------------------------------
    # Season bounds
    # ------------------------------------------------------------------

    def _get_season_bounds(self) -> tuple[str, str]:
        """Return (season_start, season_end) as ISO date strings."""
        now = datetime.now(tz=timezone.utc)
        sm = int(self.config["simulation"]["season_start_month"])
        sd = int(self.config["simulation"]["season_start_day"])
        if now.month >= sm:
            season_start = datetime(now.year, sm, sd)
        else:
            season_start = datetime(now.year - 1, sm, sd)
        season_end = now.replace(tzinfo=None) + timedelta(days=1)
        return season_start.strftime("%Y-%m-%d"), season_end.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # INI generation
    # ------------------------------------------------------------------

    def _write_avapro_ini(self, pro_path: Path) -> Path:
        """
        Write an AVAPRO INI configuration file.

        AVAPRO requires PRO and SMET in the same directory; the file stem
        must match (e.g. TAMI2.pro + TAMI2.smet in data/pro/).

        Parameters
        ----------
        pro_path : Path
            Path to the SNOWPACK PRO file.

        Returns
        -------
        Path
            Path to the written INI file.
        """
        season_start, season_end = self._get_season_bounds()
        figs_dir = self.out_dir / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        ini_dir = Path(self.config["paths"]["data"]) / "ini"
        ini_dir.mkdir(parents=True, exist_ok=True)
        ini_path = ini_dir / "avapro.ini"

        cfg = configparser.ConfigParser()
        # Preserve key case (configparser lowercases by default)
        cfg.optionxform = str

        cfg["AVAPRO"] = {
            "SNP_DIR":              str(pro_path.parent.resolve()),
            "SNP_FILE":             pro_path.name,
            "OUTPUT_DIR":           str(self.out_dir.resolve()),
            "OUTPUT_DIR_FIGS":      str(figs_dir.resolve()),
            "rerun_find_WL":        "1",
            "rerun_assign_avaprobs":"1",
            "run_visualize_avaprobs":"0",   # skip figure generation by default
            "DATE_OPERA":           "TODAY",
            "SEASON_END":           season_end,   # used by avapro.py for date check
            "initilization_type":   "station",
            "drytime":              "6",
            "wettime":              "15",
            "RESOLUTION":           "1d",
            "resolution":           "1d",
            "debug":                "0",
            "scmopt":               "snp",
            "season_start":         season_start,
            "season_end":           season_end,
        }
        cfg["AVAPRO-THOLDS-WL"] = {
            "calcFEM":       "0",
            "owSCtaup":      "0",
            "lookat":        "0",
            "aggrgtDAPs":    "1",
            "aggrgtdiff":    "0.03",
            "outputDAP":     "0",
            "dropini":       "5",
            "droppro":       "0.64",
            "drftthrsh":     "0.4",
            "minSLdens":     "120",
            "minnsthrsh":    "0.05",
            "nsthrsh":       "0.2",
            "minSLthrsh":    "0.2",
            "lwcthrsh_0":    "0.01",
            "lwcthrsh_1":    "0.03",
            "dysisomax":     "3",
            "vw_thrsh":      "5",
            "vw_thrsh_days": "3",
        }
        cfg["AVAPRO-THOLDS-APS-SNP"] = {
            "damthrshnap":   "8",
            "inithrshnap":   "999",
            "propthrshnap":  "0.32",
            "damthrshpap":   "18",
            "inithrshpap":   "1.31",
            "propthrshpap":  "0.42",
            "drftthrsh":     "3",
        }
        cfg["INFO"] = {}

        with open(ini_path, "w") as fh:
            cfg.write(fh)

        logger.info("Written AVAPRO INI: %s", ini_path)
        return ini_path

    # ------------------------------------------------------------------
    # SMET placement
    # ------------------------------------------------------------------

    def _ensure_smet_beside_pro(self, pro_path: Path, smet_path: Path) -> None:
        """
        AVAPRO requires the SMET file to sit in the same directory as the PRO
        file with the same stem.  Create a symlink (or copy as fallback).
        """
        target = pro_path.parent / (pro_path.stem + ".smet")
        if target.exists() or target.is_symlink():
            return
        try:
            target.symlink_to(smet_path.resolve())
            logger.info("Created SMET symlink: %s -> %s", target, smet_path)
        except OSError:
            shutil.copy2(smet_path, target)
            logger.info("Copied SMET to PRO dir: %s", target)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, pro_path: Path, smet_path: Path) -> Optional[pd.DataFrame]:
        """
        Run AVAPRO and return a daily avalanche problems DataFrame.

        Parameters
        ----------
        pro_path : Path
            Path to the SNOWPACK PRO output file.
        smet_path : Path
            Path to the SMET forcing file.

        Returns
        -------
        pd.DataFrame or None
            Columns: [date, new_snow, wind_slab, persistent_weak_layer,
                      deep_slab, wet_snow, glide_snow].
            Returns None on failure.
        """
        if not _AVAPRO_AVAILABLE:
            logger.error("snowpacktools not installed — cannot run AVAPRO.")
            return None
        if not pro_path.exists():
            logger.error("PRO file not found: %s", pro_path)
            return None
        if not smet_path.exists():
            logger.error("SMET file not found: %s", smet_path)
            return None

        self._ensure_smet_beside_pro(pro_path, smet_path)
        ini_path = self._write_avapro_ini(pro_path)

        logger.info("Running AVAPRO on: %s", pro_path.name)
        try:
            _avapro_func(str(ini_path))
        except Exception as exc:
            logger.error("AVAPRO run failed: %s", exc)
            return None

        # Load the post-processed results pickle
        aps_pkl = self.out_dir / f"{pro_path.stem}_df_P_APS.pkl"
        if not aps_pkl.exists():
            logger.error("AVAPRO output pickle not found: %s", aps_pkl)
            return None

        try:
            df_P = pickle.load(open(aps_pkl, "rb"))
        except Exception as exc:
            logger.error("Could not load AVAPRO pickle: %s", exc)
            return None

        df = self._extract_problems(df_P)
        logger.info("AVAPRO classified %d days.", len(df))
        return df

    def save(self, df: pd.DataFrame) -> Path:
        """Save the problems DataFrame to CSV."""
        df.to_csv(self.csv_path, index=False)
        logger.info("Saved AVAPRO results: %s (%d rows)", self.csv_path, len(df))
        return self.csv_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flag(row: "pd.Series", *cols: str) -> bool:
        """Return True if any of the given columns equals 1."""
        for col in cols:
            v = row.get(col, 0)
            if not (isinstance(v, float) and pd.isna(v)) and v == 1:
                return True
        return False

    def _extract_problems(self, df_P: pd.DataFrame) -> pd.DataFrame:
        """
        Convert AVAPRO df_P (with _sele columns) to a simplified CSV-ready
        DataFrame with one boolean column per problem type per day.

        AVAPRO column mapping
        ---------------------
        new_snow             ← napex_sele_natural | napex_sele_trigger
        wind_slab            ← winex
        persistent_weak_layer← papex_sele_natural | papex_sele_trigger
        deep_slab            ← dapex_sele_natural | dapex_sele_trigger
        wet_snow             ← wapex_sele
        glide_snow           ← not computed by AVAPRO (always False)
        """
        rows = []
        for _, row in df_P.iterrows():
            day = row.get("dy")
            if day is None or (isinstance(day, float) and pd.isna(day)):
                continue
            rows.append({
                "date":                   str(day)[:10],
                "new_snow":               self._flag(row, "napex_sele_natural", "napex_sele_trigger"),
                "wind_slab":              self._flag(row, "winex"),
                "persistent_weak_layer":  self._flag(row, "papex_sele_natural", "papex_sele_trigger"),
                "deep_slab":              self._flag(row, "dapex_sele_natural", "dapex_sele_trigger"),
                "wet_snow":               self._flag(row, "wapex_sele"),
                "glide_snow":             False,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def run_avapro(config: dict, pro_path: Path, smet_path: Path) -> Optional[pd.DataFrame]:
    """
    Run AVAPRO and save results.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    pro_path : Path
        Path to the SNOWPACK PRO file.
    smet_path : Path
        Path to the SMET forcing file.

    Returns
    -------
    pd.DataFrame or None
    """
    runner = AvaPRORunner(config)
    df = runner.run(pro_path, smet_path)
    if df is not None and not df.empty:
        runner.save(df)
    return df


def get_today_problems(config: dict) -> dict:
    """
    Read the most recent AVAPRO classification row from CSV.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.

    Returns
    -------
    dict
        Problem flags for the latest classified day, all False if no data.
    """
    csv_path = (
        Path(config["paths"]["data"]) / "avapro_output" / "tamsichbachturm_problems.csv"
    )
    default = {
        "new_snow": False,
        "wind_slab": False,
        "persistent_weak_layer": False,
        "deep_slab": False,
        "wet_snow": False,
        "glide_snow": False,
    }
    if not csv_path.exists():
        return default
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return default
        last = df.iloc[-1].to_dict()
        return {
            "new_snow":              bool(last.get("new_snow", False)),
            "wind_slab":             bool(last.get("wind_slab", False)),
            "persistent_weak_layer": bool(last.get("persistent_weak_layer", False)),
            "deep_slab":             bool(last.get("deep_slab", False)),
            "wet_snow":              bool(last.get("wet_snow", False)),
            "glide_snow":            bool(last.get("glide_snow", False)),
        }
    except Exception as exc:
        logger.warning("Could not read AVAPRO CSV: %s", exc)
        return default
