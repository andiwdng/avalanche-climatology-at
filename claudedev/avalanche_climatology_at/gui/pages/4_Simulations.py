"""
Page 4 — SNOWPACK Simulations
Run all simulations and monitor progress.
"""

import streamlit as st
import yaml
import pandas as pd
import subprocess
from pathlib import Path

st.set_page_config(page_title="Simulations", page_icon="▶️", layout="wide")
st.title("▶️  Step 4 · SNOWPACK Simulations")
st.caption("Run all SNOWPACK point simulations and monitor their status.")

# ── Load config ────────────────────────────────────────────────────────────────
cfg = st.session_state.get("config")
if cfg is None:
    CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)
        st.session_state["config"] = cfg
    else:
        st.error("config.yaml not found.")
        st.stop()

regions = cfg["regions"]
elevations = cfg["elevation_bands"]
snowpack_binary = cfg["snowpack"]["binary"]
input_dir = Path(cfg["paths"]["snowpack_input"])
log_dir = Path(cfg["paths"]["logs"])

st.markdown("---")

# ── Simulation status overview ─────────────────────────────────────────────────
st.subheader("Simulation status")

rows = []
for rk, rm in regions.items():
    for elev in elevations:
        station_id = f"{rk}_{elev}m"
        ini_path = input_dir / "ini" / f"{station_id}.ini"
        log_path = log_dir / f"{station_id}_snowpack.log"
        output_dir = Path(cfg["paths"]["snowpack_output"]) / rk / f"{elev}m"
        pro_path = output_dir / f"{station_id}.pro"

        ini_ok = ini_path.exists()
        pro_ok = pro_path.exists()

        if pro_ok:
            status = "✓  Done"
        elif log_path.exists():
            # Read last line of log
            try:
                last = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
                if "error" in last.lower():
                    status = "✗  Failed"
                else:
                    status = "○  Not run"
            except Exception:
                status = "○  Not run"
        elif ini_ok:
            status = "○  Ready"
        else:
            status = "⚠️  No INI"

        rows.append({
            "Station": station_id,
            "Region": rm["name"],
            "Elev (m)": elev,
            "INI": "✓" if ini_ok else "✗",
            "PRO output": "✓" if pro_ok else "—",
            "Status": status,
        })

df_status = pd.DataFrame(rows)
st.dataframe(df_status, use_container_width=True, hide_index=True)

n_ready = sum(1 for r in rows if "✓" in r["INI"])
n_done  = sum(1 for r in rows if "✓  Done" == r["Status"])
c1, c2, c3 = st.columns(3)
c1.metric("Total simulations", len(rows))
c2.metric("INI files ready", n_ready)
c3.metric("Already completed", n_done)

st.markdown("---")

# ── Run settings ───────────────────────────────────────────────────────────────
st.subheader("Run settings")

col_j, col_b = st.columns([1, 2])
with col_j:
    n_jobs = st.slider(
        "Parallel workers",
        min_value=1, max_value=4, value=1,
        help="How many SNOWPACK simulations to run at the same time. "
             "Keep at 1 on a MacBook Air to avoid memory issues.",
    )
with col_b:
    st.caption(
        "⚠️  On a MacBook Air (8 GB RAM) keep this at **1**. "
        "Each SNOWPACK process uses 1–2 GB of RAM."
    )

end_date = cfg["simulation"]["analysis_end"] + "T00:00"

# ── Run button ─────────────────────────────────────────────────────────────────
if st.button("▶️  Run all simulations", type="primary"):
    ini_paths = st.session_state.get("ini_paths")
    if ini_paths is None:
        # Try to find INI files from disk
        ini_paths = {}
        for rk in regions:
            ini_paths[rk] = {}
            for elev in elevations:
                station_id = f"{rk}_{elev}m"
                p = input_dir / "ini" / f"{station_id}.ini"
                if p.exists():
                    ini_paths[rk][elev] = p

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.run_snowpack import run_snowpack_simulations

    progress_bar = st.progress(0)
    log_box = st.empty()
    success_map = {}

    with st.spinner("Running SNOWPACK simulations …"):
        try:
            results = run_snowpack_simulations(cfg, ini_paths, n_jobs=n_jobs)
            st.session_state["snowpack_results"] = results
            n_ok = sum(v for reg in results.values() for v in reg.values())
            n_fail = sum(1 for reg in results.values() for v in reg.values() if not v)
            progress_bar.progress(100)
            if n_fail == 0:
                st.success(f"✅  All {n_ok} simulations completed successfully.")
            else:
                st.warning(f"⚠️  {n_ok} succeeded, {n_fail} failed. Check logs below.")
            st.rerun()
        except Exception as e:
            st.error(f"Error running simulations: {e}")

st.markdown("---")

# ── Log viewer ─────────────────────────────────────────────────────────────────
st.subheader("Log viewer")
log_files = sorted(log_dir.glob("*_snowpack.log")) if log_dir.exists() else []
if log_files:
    selected_log = st.selectbox(
        "Select simulation log",
        options=log_files,
        format_func=lambda p: p.stem.replace("_snowpack", ""),
    )
    if selected_log:
        content = selected_log.read_text(encoding="utf-8", errors="replace")
        st.code(content if content.strip() else "(empty log)", language=None)
else:
    st.info("No log files yet. Run simulations first.")
