# snowpack_steiermark/app.py
"""
Flask web application for the SNOWPACK Steiermark dashboard.
Provides a browser GUI for controlling the pipeline, displaying status,
downloading PRO files, and triggering Git pushes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request, send_file

from scripts.download_geosphere import download_station_data
from scripts.git_sync import git_push
from scripts.avapro_runner import AvaPRORunner, get_today_problems
from scripts.ini_writer import write_ini
from scripts.pro_parser import parse_pro
from scripts.smet_writer import write_smet
from scripts.sno_writer import write_empty_sno
from scripts.snowpack_runner import SnowpackRunner, run_snowpack

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent / "config.yaml"
with open(_config_path) as _fh:
    config = yaml.safe_load(_fh)

# Global run state (threadsafe enough for single-user dashboard)
run_status: str = "idle"   # "idle" | "running" | "ok" | "error"
run_log: str = ""
run_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helper: find PRO file
# ---------------------------------------------------------------------------
def _find_pro_file() -> Path | None:
    pro_dir = Path(config["paths"]["data"]) / "pro"
    candidates = sorted(pro_dir.glob("*.pro"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_smet_file() -> Path:
    return Path(config["paths"]["data"]) / "smet" / "TAMI2.smet"


def _get_season_start() -> datetime:
    """Return current season start as UTC-aware datetime."""
    now = datetime.now(tz=timezone.utc)
    sm = int(config["simulation"]["season_start_month"])
    sd = int(config["simulation"]["season_start_day"])
    if now.month >= sm:
        return datetime(now.year, sm, sd, 0, 0, tzinfo=timezone.utc)
    return datetime(now.year - 1, sm, sd, 0, 0, tzinfo=timezone.utc)


def _read_state() -> dict:
    state_path = Path(config["paths"]["state"]) / "last_download.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", station_name=config["station"]["name"])


@app.route("/api/status")
def api_status():
    state = _read_state()
    pro_path = _find_pro_file()

    snow_height_cm = None
    surface_temp_c = None
    pro_exists = pro_path is not None and pro_path.exists()

    if pro_exists:
        try:
            parser = parse_pro(pro_path)
            profile = parser.get_latest_profile()
            snow_height_cm = profile.get("hs_cm")
            surface_temp_c = profile.get("tss_c")
        except Exception as exc:
            logger.warning("Could not parse PRO for status: %s", exc)

    problems = get_today_problems(config)
    runner = SnowpackRunner(config)

    return jsonify(
        {
            "last_download": state.get("last_download"),
            "last_simulation": state.get("last_simulation_end"),
            "snow_height_cm": snow_height_cm,
            "surface_temp_c": surface_temp_c,
            "problems": problems,
            "pro_exists": pro_exists,
            "snowpack_binary_ok": runner.check_binary(),
        }
    )


@app.route("/api/smet-data")
def api_smet_data():
    """Return parsed SMET forcing data for GUI visualization."""
    smet_path = _find_smet_file()
    empty = {"dates": [], "ta_c": [], "rh_pct": [], "vw_ms": [],
             "dw_deg": [], "iswr_wm2": [], "psum_mm": [], "hs_cm": []}
    if not smet_path.exists():
        return jsonify(empty)
    try:
        fields: list[str] = []
        dates: list[str] = []
        cols: dict[str, list] = {}
        in_data = False
        with open(smet_path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("fields"):
                    _, _, rhs = s.partition("=")
                    fields = rhs.strip().split()
                    cols = {f: [] for f in fields[1:]}
                    continue
                if s == "[DATA]":
                    in_data = True
                    continue
                if not in_data or not s:
                    continue
                parts = s.split()
                if len(parts) != len(fields):
                    continue
                dates.append(parts[0])
                for i, f in enumerate(fields[1:], 1):
                    try:
                        v = float(parts[i])
                        cols[f].append(None if v == -999.0 else v)
                    except ValueError:
                        cols[f].append(None)

        def _conv(lst, fn):
            return [fn(v) if v is not None else None for v in lst]

        return jsonify({
            "dates":     dates,
            "ta_c":      _conv(cols.get("TA", []),      lambda v: round(v - 273.15, 2)),
            "rh_pct":    _conv(cols.get("RH", []),      lambda v: round(v * 100.0, 1)),
            "vw_ms":     _conv(cols.get("VW", []),      lambda v: round(v, 2)),
            "dw_deg":    _conv(cols.get("DW", []),      lambda v: round(v, 1)),
            "iswr_wm2":  _conv(cols.get("ISWR", []),   lambda v: round(v, 1)),
            "psum_mm":   _conv(cols.get("PSUM", []),   lambda v: round(v, 3)),
            "psum_ph":   _conv(cols.get("PSUM_PH", []), lambda v: round(v, 2)),
            "hs_cm":     _conv(cols.get("HS", []),      lambda v: round(v * 100.0, 1)),
        })
    except Exception as exc:
        logger.error("SMET data parse error: %s", exc)
        return jsonify(empty)


@app.route("/api/timeseries")
def api_timeseries():
    pro_path = _find_pro_file()
    if not pro_path or not pro_path.exists():
        return jsonify({"dates": [], "hs_cm": [], "tss_c": []})
    try:
        parser = parse_pro(pro_path)
        records = parser.get_timeseries_dict()
        return jsonify(
            {
                "dates": [r["date"] for r in records],
                "hs_cm": [r["hs_cm"] for r in records],
                "tss_c": [r["tss_c"] for r in records],
            }
        )
    except Exception as exc:
        logger.error("Timeseries parse error: %s", exc)
        return jsonify({"dates": [], "hs_cm": [], "tss_c": []})


@app.route("/run", methods=["POST"])
def run_pipeline():
    """Start the full update cycle in a background thread."""
    global run_status, run_log
    with run_lock:
        if run_status == "running":
            return jsonify({"status": "already_running"})
        run_status = "running"
        run_log = ""

    thread = threading.Thread(target=_run_pipeline_thread, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


def _run_pipeline_thread() -> None:
    """Execute the full pipeline and update global run_status/run_log."""
    global run_status, run_log

    def log(msg: str) -> None:
        global run_log
        run_log += msg + "\n"
        logger.info(msg)

    try:
        log("=== Pipeline gestartet ===")

        # 1. Download
        log("1. Lade GeoSphere-Daten…")
        df = download_station_data(config)
        if df.empty:
            log("   Keine neuen Daten.")
        else:
            log(f"   {len(df)} neue 10-Minuten-Werte heruntergeladen.")

        # 2. Write SMET
        log("2. Schreibe SMET-Datei…")
        smet_path = write_smet(config, df)
        log(f"   SMET: {smet_path}")

        # 3. SNO (only if missing)
        sno_path = Path(config["paths"]["data"]) / "sno" / "tamsichbachturm.sno"
        if not sno_path.exists():
            log("3. Schreibe leere SNO-Datei…")
            season_start = _get_season_start()
            sno_path = write_empty_sno(config, season_start)
            log(f"   SNO: {sno_path}")
        else:
            log("3. SNO-Datei vorhanden, überspringe.")

        # 4. Write INI
        log("4. Schreibe INI-Datei…")
        pro_dir = Path(config["paths"]["data"]) / "pro"
        season_start = _get_season_start()
        end_date = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        ini_path = write_ini(config, smet_path, sno_path, pro_dir, season_start, end_date)
        log(f"   INI: {ini_path}")

        # 5. Run SNOWPACK
        log("5. Starte SNOWPACK-Simulation…")
        runner = SnowpackRunner(config)
        if not runner.check_binary():
            log("   SNOWPACK-Binary nicht gefunden — Simulation übersprungen.")
            log("   Pfad in config.yaml unter snowpack.binary anpassen.")
        else:
            success, log_path = runner.run(ini_path, end_date)
            if success:
                runner.update_state(end_date)
                log(f"   SNOWPACK erfolgreich. Log: {log_path}")
            else:
                log(f"   SNOWPACK FEHLER. Siehe Log: {log_path}")
                # Read last 20 lines of log
                try:
                    lines = log_path.read_text(errors="replace").splitlines()[-20:]
                    log("\n".join(lines))
                except Exception:
                    pass

        # 6. Classify (AVAPRO)
        log("6. Starte AVAPRO Lawinenproblem-Klassifikation…")
        pro_path = _find_pro_file()
        if pro_path and pro_path.exists():
            runner_ava = AvaPRORunner(config)
            clf_df = runner_ava.run(pro_path, smet_path)
            if clf_df is not None and not clf_df.empty:
                out_path = runner_ava.save(clf_df)
                log(f"   {len(clf_df)} Tage klassifiziert → {out_path}")
            else:
                log("   AVAPRO lieferte keine Ergebnisse.")
        else:
            log("   Kein PRO-File gefunden, Klassifikation übersprungen.")

        # 7. Git push
        if config.get("git", {}).get("auto_push", False):
            log("7. Git Push…")
            ok, msg = git_push(config)
            log(f"   {'✓' if ok else '✗'} {msg}")
        else:
            log("7. Auto-Push deaktiviert.")

        log("=== Pipeline abgeschlossen ===")
        with run_lock:
            run_status = "ok"

    except Exception as exc:
        log(f"FEHLER: {exc}")
        logger.exception("Pipeline error")
        with run_lock:
            run_status = "error"


@app.route("/run/status")
def run_status_endpoint():
    return jsonify({"status": run_status, "log": run_log[-5000:]})


@app.route("/log")
def get_log():
    log_dir = Path(config["paths"]["logs"])
    logs = sorted(log_dir.glob("snowpack_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return Response("Noch keine Simulation gelaufen.", content_type="text/plain")
    try:
        content = logs[0].read_text(errors="replace")
    except Exception as exc:
        content = f"Fehler beim Lesen: {exc}"
    return Response(content, content_type="text/plain; charset=utf-8")


@app.route("/api/pro-files")
def api_pro_files():
    """List all available PRO files sorted by modification time (newest first)."""
    pro_dir = Path(config["paths"]["data"]) / "pro"
    files = []
    if pro_dir.exists():
        for p in sorted(pro_dir.glob("*.pro"), key=lambda f: f.stat().st_mtime, reverse=True):
            stat = p.stat()
            files.append({
                "name": p.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
    return jsonify(files)


@app.route("/pro/download")
def pro_download():
    pro_path = _find_pro_file()
    if not pro_path or not pro_path.exists():
        return Response("Keine PRO-Datei vorhanden. Bitte zuerst eine Simulation starten.", status=404)
    return send_file(
        pro_path,
        as_attachment=True,
        download_name=pro_path.name,
        mimetype="text/plain",
    )


@app.route("/pro/download/<filename>")
def pro_download_named(filename: str):
    """Download a specific PRO file by name."""
    pro_dir = Path(config["paths"]["data"]) / "pro"
    pro_path = (pro_dir / filename).resolve()
    # Security: ensure resolved path is inside pro_dir
    if not str(pro_path).startswith(str(pro_dir.resolve())):
        return Response("Ungültiger Dateiname.", status=400)
    if not pro_path.exists():
        return Response(f"{filename} nicht gefunden.", status=404)
    return send_file(
        pro_path,
        as_attachment=True,
        download_name=pro_path.name,
        mimetype="text/plain",
    )


@app.route("/niviz")
def niviz():
    return render_template("niviz.html")


@app.route("/git-push", methods=["POST"])
def git_push_route():
    try:
        ok, msg = git_push(config)
        return jsonify({"success": ok, "output": msg})
    except Exception as exc:
        return jsonify({"success": False, "output": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = config["web"]["host"]
    port = int(config["web"]["port"])
    debug = bool(config["web"]["debug"])
    logger.info("Starting Flask on %s:%d", host, port)
    app.run(host=host, port=port, debug=debug)
