# snowpack_steiermark/scripts/snowpack_runner.py
"""
Run SNOWPACK as a subprocess and manage simulation state.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SnowpackRunner:
    """Executes SNOWPACK and tracks simulation state."""

    def __init__(self, config: dict, station: dict | None = None) -> None:
        self.config = config
        self.binary = config["snowpack"]["binary"]
        self.timeout = int(config["snowpack"]["timeout"])
        state_dir = Path(config["paths"]["state"])
        state_dir.mkdir(parents=True, exist_ok=True)
        if station is not None:
            self.state_path = state_dir / f"{station['id'].lower()}_download.json"
        else:
            # backward compat: fall back to TAMI state
            self.state_path = state_dir / "tami_download.json"
        log_dir = Path(config["paths"]["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

    def check_binary(self) -> bool:
        """
        Check whether the SNOWPACK binary exists and is executable.

        Returns
        -------
        bool
            True if the binary is ready to use.
        """
        p = Path(self.binary)
        if p.exists() and os.access(p, os.X_OK):
            logger.info("SNOWPACK binary found: %s", self.binary)
            return True
        logger.warning("SNOWPACK binary not found or not executable: %s", self.binary)
        return False

    def run(self, ini_path: Path, end_date: datetime) -> tuple[bool, Path]:
        """
        Execute SNOWPACK for the given INI file.

        Parameters
        ----------
        ini_path : Path
            Path to the SNOWPACK INI configuration file.
        end_date : datetime
            Simulation end date (passed as -e argument).

        Returns
        -------
        tuple[bool, Path]
            (success, log_path) where success is True if returncode == 0.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        log_path = self.log_dir / f"snowpack_{ini_path.stem}_{timestamp}.log"

        end_str = end_date.strftime("%Y-%m-%dT%H:%M")
        cmd = [
            self.binary,
            "-c", str(ini_path.resolve()),
            "-e", end_str,
        ]
        logger.info("Running SNOWPACK: %s", " ".join(cmd))
        logger.info("Log: %s", log_path)

        try:
            with open(log_path, "w") as log_fh:
                result = subprocess.run(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    check=False,
                    cwd=str(ini_path.resolve().parent),
                )
            success = result.returncode == 0
            if success:
                logger.info("SNOWPACK finished successfully (rc=0)")
            else:
                logger.error("SNOWPACK failed with return code %d", result.returncode)
            return success, log_path
        except subprocess.TimeoutExpired:
            logger.error("SNOWPACK timed out after %d seconds", self.timeout)
            with open(log_path, "a") as log_fh:
                log_fh.write(f"\n[TIMEOUT after {self.timeout} seconds]\n")
            return False, log_path
        except FileNotFoundError:
            logger.error("SNOWPACK binary not found: %s", self.binary)
            with open(log_path, "w") as log_fh:
                log_fh.write(f"Binary not found: {self.binary}\n")
            return False, log_path
        except Exception as exc:
            logger.error("Unexpected error running SNOWPACK: %s", exc)
            with open(log_path, "a") as log_fh:
                log_fh.write(f"\n[ERROR: {exc}]\n")
            return False, log_path

    def update_state(self, simulation_end: datetime) -> None:
        """
        Persist the simulation end date in the state file.

        Parameters
        ----------
        simulation_end : datetime
            The end date/time of the completed simulation.
        """
        state: dict = {}
        if self.state_path.exists():
            try:
                with open(self.state_path) as fh:
                    state = json.load(fh)
            except Exception:
                state = {}
        state["last_simulation_end"] = simulation_end.isoformat()

        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.state_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(state, fh, indent=2, default=str)
            os.replace(tmp_path, self.state_path)
        except Exception as exc:
            logger.error("Could not update state: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_snowpack(
    config: dict, station: dict, ini_path: Path, end_date: datetime
) -> tuple[bool, Path]:
    """
    Run SNOWPACK using the provided INI file.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    station : dict
        Station entry from config["stations"] list.
    ini_path : Path
        Path to the SNOWPACK INI file.
    end_date : datetime
        Simulation end datetime.

    Returns
    -------
    tuple[bool, Path]
        (success, log_path).
    """
    runner = SnowpackRunner(config, station)
    success, log_path = runner.run(ini_path, end_date)
    if success:
        runner.update_state(end_date)
    return success, log_path
