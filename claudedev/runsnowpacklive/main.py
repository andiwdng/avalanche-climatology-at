# snowpack_steiermark/main.py
"""
CLI entry point for the SNOWPACK Steiermark pipeline.

Usage
-----
    python main.py                        # run all stations
    python main.py --station TAMI         # run only TAMI
    python main.py --full-reset           # restart from scratch (wipe SNO + state)
    python main.py --skip-download        # use existing SMET
    python main.py --skip-snowpack        # skip simulation
    python main.py --web                  # start Flask dashboard instead
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _setup_logging(level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)


def _get_season_start(config: dict) -> datetime:
    now = datetime.now(tz=timezone.utc)
    sm = int(config["simulation"]["season_start_month"])
    sd = int(config["simulation"]["season_start_day"])
    if now.month >= sm:
        return datetime(now.year, sm, sd, 0, 0, tzinfo=timezone.utc)
    return datetime(now.year - 1, sm, sd, 0, 0, tzinfo=timezone.utc)


def _run_station(config: dict, station: dict, args: argparse.Namespace) -> bool:
    """
    Run the full pipeline for a single station.

    Returns True on success, False if a critical stage failed.
    """
    logger = logging.getLogger(__name__)
    station_id = station["id"]
    snow_station = station["snow_station"]

    logger.info("=" * 60)
    logger.info("Station: %s (%s)", station["name"], station_id)
    logger.info("=" * 60)

    smet_path = Path(config["paths"]["data"]) / "smet" / f"{snow_station}.smet"

    # Full reset
    if args.full_reset:
        sno_path = Path(config["paths"]["data"]) / "sno" / f"{snow_station}.sno"
        state_path = Path(config["paths"]["state"]) / f"{station_id.lower()}_download.json"
        for p in [sno_path, state_path]:
            if p.exists():
                p.unlink()
                logger.info("Deleted: %s", p)
        logger.info("Full reset complete for %s.", station_id)

    # --- 1. Download ---
    if not args.skip_download:
        logger.info("Stage 1: Downloading GeoSphere data")
        from scripts.download_geosphere import download_station_data
        df = download_station_data(config, station)
        if df.empty:
            logger.info("  No new data downloaded.")
        else:
            logger.info("  %d new 10-min records.", len(df))
        logger.info("Stage 1b: Writing SMET")
        from scripts.smet_writer import write_smet
        smet_path = write_smet(config, station, df)
        logger.info("  SMET: %s", smet_path)
    else:
        logger.info("Stage 1: Download skipped.")
        if not smet_path.exists():
            logger.error("SMET file missing and --skip-download set. Aborting %s.", station_id)
            return False

    # --- Season timing ---
    season_start = _get_season_start(config)
    end_date = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    logger.info("Season: %s → %s", season_start.date(), end_date.date())

    # --- 2. SNO ---
    sno_path = Path(config["paths"]["data"]) / "sno" / f"{snow_station}.sno"
    if not sno_path.exists() or args.full_reset:
        logger.info("Stage 2: Writing empty SNO file")
        from scripts.sno_writer import write_empty_sno
        sno_path = write_empty_sno(config, station, season_start, force=args.full_reset)
        logger.info("  SNO: %s", sno_path)
    else:
        logger.info("Stage 2: SNO exists, keeping.")

    # --- 3. INI ---
    logger.info("Stage 3: Writing INI file")
    pro_dir = Path(config["paths"]["data"]) / "pro"
    from scripts.ini_writer import write_ini
    ini_path = write_ini(config, station, smet_path, sno_path, pro_dir, season_start, end_date)
    logger.info("  INI: %s", ini_path)

    # --- 4. SNOWPACK ---
    if not args.skip_snowpack:
        logger.info("Stage 4: Running SNOWPACK")
        from scripts.snowpack_runner import SnowpackRunner, run_snowpack
        runner = SnowpackRunner(config, station)
        if not runner.check_binary():
            logger.error("  SNOWPACK binary not found: %s", config["snowpack"]["binary"])
            logger.error("  Adjust snowpack.binary in config.yaml and retry.")
        else:
            success, log_path = run_snowpack(config, station, ini_path, end_date)
            if success:
                logger.info("  SNOWPACK OK. Log: %s", log_path)
            else:
                logger.error("  SNOWPACK FAILED. Log: %s", log_path)
    else:
        logger.info("Stage 4: SNOWPACK skipped.")

    # --- 5. Classify (AVAPRO) ---
    if not args.skip_classify:
        logger.info("Stage 5: Running AVAPRO avalanche problem classification")
        from scripts.avapro_runner import run_avapro
        # PRO file is named {snow_station}_{EXPERIMENT}.pro = {snow_station}_{station_id}.pro
        pro_candidates = sorted(
            pro_dir.glob(f"{snow_station}_*.pro"),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not pro_candidates:
            logger.warning("  No PRO file found for %s, skipping AVAPRO.", station_id)
        else:
            pro_path = pro_candidates[0]
            clf_df = run_avapro(config, station, pro_path, smet_path)
            if clf_df is not None and not clf_df.empty:
                last = clf_df.iloc[-1]
                logger.info("  %d days classified. Latest day %s: NS=%s WS=%s PWL=%s DS=%s Wet=%s",
                            len(clf_df),
                            last.get("date", "?"),
                            last.get("new_snow", False),
                            last.get("wind_slab", False),
                            last.get("persistent_weak_layer", False),
                            last.get("deep_slab", False),
                            last.get("wet_snow", False))
            else:
                logger.warning("  AVAPRO returned no results.")
    else:
        logger.info("Stage 5: Classification skipped.")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="SNOWPACK Steiermark pipeline")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--station", default=None,
                        help="Run only this station ID (e.g. TAMI, LOSE, VEIT, PLAN). "
                             "Default: run all stations.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GeoSphere data download")
    parser.add_argument("--skip-snowpack", action="store_true",
                        help="Skip SNOWPACK simulation")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip heuristic avalanche problem classification")
    parser.add_argument("--skip-git", action="store_true",
                        help="Skip Git commit/push")
    parser.add_argument("--full-reset", action="store_true",
                        help="Delete SNO and state files; restart from season start")
    parser.add_argument("--web", action="store_true",
                        help="Start Flask web server instead of running pipeline")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    log_dir = Path(config["paths"]["logs"])
    _setup_logging(args.log_level, log_dir)
    logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # Web mode
    # -----------------------------------------------------------------------
    if args.web:
        from app import app as flask_app
        host = config["web"]["host"]
        port = int(config["web"]["port"])
        debug = bool(config["web"]["debug"])
        logger.info("Starting Flask dashboard on %s:%d", host, port)
        flask_app.run(host=host, port=port, debug=debug)
        return 0

    # -----------------------------------------------------------------------
    # Pipeline mode
    # -----------------------------------------------------------------------
    stations = config.get("stations", [])
    if not stations:
        logger.error("No stations defined in config.yaml under 'stations:'")
        return 1

    # Filter to requested station
    if args.station:
        targets = [s for s in stations if s["id"].upper() == args.station.upper()]
        if not targets:
            logger.error("Station '%s' not found in config. Available: %s",
                         args.station, [s["id"] for s in stations])
            return 1
    else:
        targets = stations

    any_failed = False
    for station in targets:
        try:
            ok = _run_station(config, station, args)
        except Exception as exc:
            logger.exception("Unhandled error for station %s: %s", station["id"], exc)
            ok = False
        if not ok:
            any_failed = True

    # --- 6. Git ---
    if not args.skip_git and config.get("git", {}).get("auto_push", False):
        logger.info("Stage 6: Git push")
        from scripts.git_sync import git_push
        ok, msg = git_push(config)
        logger.info("  %s %s", "✓" if ok else "✗", msg)
    else:
        logger.info("Stage 6: Git push skipped.")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
