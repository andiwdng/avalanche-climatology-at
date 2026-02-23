"""
Page 5 — AVAPRO
Configure, run and inspect AVAPRO avalanche problem classification.
"""

import sys
import copy
import streamlit as st
import yaml
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="AVAPRO", page_icon="🏔", layout="wide")
st.title("🏔  Step 5 · AVAPRO")
st.caption("Classify avalanche problems from SNOWPACK PRO profiles.")

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

regions    = cfg["regions"]
elevations = cfg["elevation_bands"]
avapro_cfg = cfg.get("avapro", {})

# Collect output directories (production + test)
_prod_out  = Path(cfg["paths"]["snowpack_output"])
_test_out  = _prod_out.parent / (_prod_out.name + "_test")
_prod_avap = Path(cfg["paths"]["avapro_output"])
_test_avap = _prod_avap.parent / (_prod_avap.name + "_test")

output_sources = []
if _prod_out.exists():
    output_sources.append(("production", _prod_out, _prod_avap))
if _test_out.exists():
    output_sources.append(("test", _test_out, _test_avap))

st.markdown("---")

# ── PRO file status ────────────────────────────────────────────────────────────
st.subheader("SNOWPACK PRO output files")

rows = []
for source_label, snowpack_dir, _ in output_sources:
    for rk, rm in regions.items():
        for elev in elevations:
            station_id = f"{rk}_{elev}m"
            pro_path = snowpack_dir / rk / f"{elev}m" / f"{station_id}.pro"
            size_str = f"{pro_path.stat().st_size / 1024:.0f}" if pro_path.exists() else "—"
            rows.append({
                "Source":    source_label,
                "Station":   station_id,
                "Region":    rm["name"],
                "Elev (m)":  elev,
                "PRO file":  "✓" if pro_path.exists() else "✗  Missing",
                "Size (KB)": size_str,
            })

df_status = pd.DataFrame(rows)
st.dataframe(df_status, use_container_width=True, hide_index=True)

n_pro = sum(1 for r in rows if "✓" in r["PRO file"])
st.metric("PRO files available", f"{n_pro} / {len(rows)}")

st.markdown("---")

# ── Run AVAPRO ────────────────────────────────────────────────────────────────
st.subheader("Run AVAPRO")

if n_pro == 0:
    st.warning("No PRO files found. Complete Step 4 (SNOWPACK simulations) first.")
else:
    # Let user pick which source to classify
    source_options = [s[0] for s in output_sources]
    source_choice = st.radio(
        "Run on output from:",
        options=source_options,
        horizontal=True,
    )

    if st.button("▶  Run AVAPRO on all PRO files", type="primary"):
        from scripts.run_avapro import run_avapro_all
        from scripts.run_snowpack import find_pro_files

        # Build a config that points to the chosen source
        run_cfg = copy.deepcopy(cfg)
        for source_label, snowpack_dir, avapro_dir in output_sources:
            if source_label == source_choice:
                run_cfg["paths"]["snowpack_output"] = str(snowpack_dir)
                run_cfg["paths"]["avapro_output"]   = str(avapro_dir)
                break

        with st.spinner("Running AVAPRO …"):
            try:
                pro_files = find_pro_files(run_cfg)
                problems  = run_avapro_all(run_cfg, pro_files)
                st.session_state["avapro_results"]       = problems
                st.session_state["avapro_results_source"] = source_choice
                total_days = sum(
                    df.any(axis=1).sum()
                    for elev_dict in problems.values()
                    for df in elev_dict.values()
                    if df is not None and not df.empty
                )
                st.success(f"Classification complete — {total_days} problem-days across all stations.")
            except Exception as e:
                st.error(f"AVAPRO failed: {e}")

st.markdown("---")

# ── Results ────────────────────────────────────────────────────────────────────
st.subheader("Results")

# Try session state first, then fall back to saved CSVs
avapro_results = st.session_state.get("avapro_results")

if avapro_results is None:
    # Auto-load from saved CSVs (prefer test if available, otherwise production)
    for source_label, _, avapro_dir in reversed(output_sources):
        csv_files = sorted(avapro_dir.glob("*_problems.csv")) if avapro_dir.exists() else []
        if csv_files:
            avapro_results = {}
            for rk in regions:
                avapro_results[rk] = {}
                for elev in elevations:
                    csv = avapro_dir / f"{rk}_{elev}m_problems.csv"
                    if csv.exists():
                        try:
                            avapro_results[rk][elev] = pd.read_csv(
                                csv, parse_dates=["date"], index_col="date"
                            )
                        except Exception:
                            pass
            if any(avapro_results.values()):
                st.info(f"Loaded saved results from **{source_label}** output (`{avapro_dir}`).")
                break

if not avapro_results:
    st.info("No AVAPRO results available yet. Run classification above.")
    st.stop()

# Region / elevation selector
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    region_choice = st.selectbox(
        "Region", options=list(regions.keys()),
        format_func=lambda k: regions[k]["name"],
    )
with col_sel2:
    elev_choice = st.selectbox("Elevation (m)", options=elevations)

df_prob = avapro_results.get(region_choice, {}).get(elev_choice)

if df_prob is None or df_prob.empty:
    st.info(f"No results for {regions[region_choice]['name']} @ {elev_choice} m.")
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
# AVAPRO-native column names + display config (matches visually_process_aps.py)
AVAPRO_PROBLEMS = [
    # (column,               label,               color trigger,  color natural outline)
    ("napex_sele_trigger", "New snow (trigger)",      "#00FF00",  None),
    ("napex_sele_natural", "New snow (natural)",      "#00FF00",  "#FF8C00"),
    ("winex",              "Wind slab",               "#228B22",  None),
    ("papex_sele_trigger", "Persistent WL (trigger)", "#ADD8E6",  None),
    ("papex_sele_natural", "Persistent WL (natural)", "#ADD8E6",  "#FF8C00"),
    ("dapex_sele_trigger", "Deep persist. WL (trig)", "#0000FF",  None),
    ("dapex_sele_natural", "Deep persist. WL (nat.)", "#0000FF",  "#FF8C00"),
    ("wapex_sele",         "Wet snow",                "#FF0000",  None),
]

existing = [(col, lbl, c, o) for col, lbl, c, o in AVAPRO_PROBLEMS if col in df_prob.columns]

# ── Summary metrics ────────────────────────────────────────────────────────────
trigger_cols = [r for r in existing if r[3] is None]   # no natural outline = trigger/main
mcols = st.columns(len(trigger_cols))
for col_ui, (col, lbl, color, _) in zip(mcols, trigger_cols):
    count = int(df_prob[col].sum())
    col_ui.metric(lbl, f"{count} days")

# ── Bar chart: trigger problem days per type ───────────────────────────────────
bar_cols   = [lbl  for col, lbl, c, o in trigger_cols]
bar_vals   = [int(df_prob[col].sum()) for col, lbl, c, o in trigger_cols]
bar_colors = [c    for col, lbl, c, o in trigger_cols]

fig = go.Figure(go.Bar(
    x=bar_cols,
    y=bar_vals,
    marker_color=bar_colors,
    marker_line_color="black",
    marker_line_width=0.5,
))
fig.update_layout(
    title=f"Problem-day counts (trigger) — {regions[region_choice]['name']} @ {elev_choice} m",
    xaxis_title="Problem type",
    yaxis_title="Days",
    height=320,
    margin=dict(t=40, b=20, l=20, r=20),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

# ── Timeline heatmap ───────────────────────────────────────────────────────────
st.markdown("**Daily classification timeline**")

import numpy as np

labels = [lbl for _, lbl, _, _ in existing]
z      = np.array([df_prob[col].astype(int).values for col, _, _, _ in existing], dtype=float)
colors = [c for _, _, c, _ in existing]

# Build one heatmap row per problem, using each problem's own color
fig2 = go.Figure()
for i, (col, lbl, color, outline) in enumerate(existing):
    vals = df_prob[col].astype(float).values
    # Show only present days (mask absent)
    vals_masked = np.where(vals == 1, 1.0, np.nan)
    fig2.add_trace(go.Bar(
        x=df_prob.index.astype(str),
        y=vals_masked,
        base=len(existing) - 1 - i,
        name=lbl,
        marker_color=color,
        marker_line_color=outline if outline else color,
        marker_line_width=2 if outline else 0,
        width=1.0,
        showlegend=True,
    ))

fig2.update_layout(
    barmode="overlay",
    height=max(200, len(existing) * 32),
    margin=dict(t=10, b=60, l=160, r=10),
    xaxis=dict(title="Date", nticks=12),
    yaxis=dict(
        tickvals=list(range(len(existing))),
        ticktext=list(reversed(labels)),
        showgrid=False,
    ),
    legend=dict(orientation="h", y=-0.25),
    plot_bgcolor="white",
)
st.plotly_chart(fig2, use_container_width=True)

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("Raw classification table"):
    st.dataframe(df_prob.reset_index(), use_container_width=True, hide_index=True)
