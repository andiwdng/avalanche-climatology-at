"""
run_snowpack.py
===============
Execute SNOWPACK point simulations for all (region, elevation_band)
combinations using subprocess calls to the SNOWPACK binary.

Simulation strategy
-------------------
- SNOWPACK is invoked once per (region, elevation_band) pair.
- Simulations span the full download period (including spin-up year).
- The SNOWPACK binary is specified in config.yaml and is assumed to be
  installed system-wide (``snowpack`` on PATH or an absolute path).
- Simulations are run in parallel using ``joblib.Parallel`` up to
  ``n_jobs`` workers to reduce wall-clock time.  Default n_jobs = −1
  (use all available CPU cores).

SNOWPACK command
----------------
    snowpack -c <ini_file> -e <end_date>

where ``-c`` is the configuration file and ``-e`` is the end date
(YYYY-MM-DDTHH:MM).  The start date is determined by the earliest
timestamp in the SMET forcing file.

Output files
------------
For each simulation, SNOWPACK writes to the output directory
specified in the INI file:
- ``<station_id>.pro``  — layered snow profiles (PRO format)
- ``<station_id>.sno``  — final snowpack state (for restart)

Error handling
--------------
If SNOWPACK exits with a non-zero return code, the error message is
logged and the simulation is marked as failed.  The pipeline continues
with remaining simulations and reports a summary of failures at the end.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# Maximum simulation wall-clock time per station [seconds]
_SIMULATION_TIMEOUT: int = 7200  # 2 hours


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_snowpack_simulations(
    config: dict,
    ini_paths: dict[str, dict[int, Path]],
    n_jobs: int = -1,
) -> dict[str, dict[int, bool]]:
    """
    Run all SNOWPACK simulations in parallel.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    ini_paths : dict
        ``ini_paths[region][elev] = Path`` to INI file, as returned by
        :func:`scripts.snowpack_writer.write_all_snowpack_inputs`.
    n_jobs : int
        Number of parallel workers.  −1 uses all CPUs.

    Returns
    -------
    dict
        ``success[region][elev] = True/False`` indicating simulation outcome.
    """
    snowpack_binary = config["snowpack"]["binary"]
    sim_cfg = config["simulation"]
    end_date = sim_cfg["analysis_end"] + "T00:00"

    # Build flat list of (region, elev, ini_path) tuples
    tasks: list[tuple[str, int, Path]] = []
    for region_key, elev_dict in ini_paths.items():
        for elev_m, ini_path in elev_dict.items():
            tasks.append((region_key, elev_m, ini_path))

    logger.info(
        "Running %d SNOWPACK simulations with n_jobs=%d …", len(tasks), n_jobs
    )

    # Parallel execution
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_simulation)(
            region_key=region_key,
            elev_m=elev_m,
            ini_path=ini_path,
            snowpack_binary=snowpack_binary,
            end_date=end_date,
        )
        for region_key, elev_m, ini_path in tasks
    )

    # Reconstruct nested dict from flat results
    success: dict[str, dict[int, bool]] = {}
    for (region_key, elev_m, _), ok in zip(tasks, results):
        success.setdefault(region_key, {})[elev_m] = ok

    # Summary
    n_ok = sum(v for reg in success.values() for v in reg.values())
    n_fail = len(tasks) - n_ok
    logger.info(
        "SNOWPACK runs complete: %d succeeded, %d failed.", n_ok, n_fail
    )
    if n_fail > 0:
        for region_key, elev_dict in success.items():
            for elev_m, ok in elev_dict.items():
                if not ok:
                    logger.error("  FAILED: %s @ %d m", region_key, elev_m)

    return success


# ---------------------------------------------------------------------------
# Single simulation runner
# ---------------------------------------------------------------------------
def _run_single_simulation(
    region_key: str,
    elev_m: int,
    ini_path: Path,
    snowpack_binary: str,
    end_date: str,
) -> bool:
    """
    Execute one SNOWPACK simulation as a subprocess.

    Parameters
    ----------
    region_key : str
        Region identifier (for logging).
    elev_m : int
        Elevation band [m] (for logging).
    ini_path : Path
        Absolute path to the SNOWPACK INI file.
    snowpack_binary : str
        Path to the SNOWPACK executable.
    end_date : str
        End date string in format ``YYYY-MM-DDTHH:MM``.

    Returns
    -------
    bool
        True if SNOWPACK exited successfully (return code 0).
    """
    station_id = ini_path.stem
    cmd = [
        snowpack_binary,
        "-c", str(ini_path.resolve()),
        "-e", end_date,
    ]

    logger.info("Starting SNOWPACK: %s @ %d m", region_key, elev_m)
    logger.debug("Command: %s", " ".join(cmd))

    log_file = ini_path.parent.parent.parent / "logs" / f"{station_id}_snowpack.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_file, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=_SIMULATION_TIMEOUT,
                check=False,
            )

        if proc.returncode == 0:
            logger.info("  ✓ %s @ %d m — SNOWPACK finished.", region_key, elev_m)
            return True
        else:
            logger.error(
                "  ✗ %s @ %d m — SNOWPACK returned code %d. See %s",
                region_key,
                elev_m,
                proc.returncode,
                log_file,
            )
            return False

    except subprocess.TimeoutExpired:
        logger.error(
            "  ✗ %s @ %d m — SNOWPACK timed out after %d s.",
            region_key,
            elev_m,
            _SIMULATION_TIMEOUT,
        )
        return False
    except FileNotFoundError:
        logger.error(
            "  ✗ SNOWPACK binary not found: '%s'. "
            "Check 'snowpack.binary' in config.yaml.",
            snowpack_binary,
        )
        return False


# ---------------------------------------------------------------------------
# PRO file locator
# ---------------------------------------------------------------------------
def find_pro_files(
    config: dict,
) -> dict[str, dict[int, Path]]:
    """
    Locate SNOWPACK PRO output files for all (region, elevation) pairs.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.

    Returns
    -------
    dict
        ``pro_files[region][elev] = Path``.  Missing files are logged
        as warnings and excluded.
    """
    output_base = Path(config["paths"]["snowpack_output"])
    regions = config["regions"]
    elevation_bands = config["elevation_bands"]

    pro_files: dict[str, dict[int, Path]] = {}

    for region_key in regions:
        pro_files[region_key] = {}
        for elev_m in elevation_bands:
            station_id = f"{region_key}_{elev_m}m"
            pro_path = output_base / region_key / f"{elev_m}m" / f"{station_id}.pro"
            if pro_path.exists():
                pro_files[region_key][elev_m] = pro_path
            else:
                logger.warning("PRO file not found: %s", pro_path)

    return pro_files


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Locate pre-existing INI files (would normally come from snowpack_writer)
    ini_dir = Path(cfg["paths"]["snowpack_input"]) / "ini"
    ini_paths: dict[str, dict[int, Path]] = {}
    for region_key in cfg["regions"]:
        ini_paths[region_key] = {}
        for elev_m in cfg["elevation_bands"]:
            station_id = f"{region_key}_{elev_m}m"
            ini_path = ini_dir / f"{station_id}.ini"
            if ini_path.exists():
                ini_paths[region_key][elev_m] = ini_path

    success = run_snowpack_simulations(cfg, ini_paths)
    print("Success summary:", success)
