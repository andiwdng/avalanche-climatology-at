"""
Page 2 — Data Processing
Interpolate ERA5 to region points and apply SPARTACUS bias correction.
"""

import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Processing", page_icon="⚙️", layout="wide")
st.title("⚙️  Step 2 · Data Processing")
st.caption("Interpolate ERA5-Land to region coordinates and apply SPARTACUS bias correction.")

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

st.markdown("---")

# ── What this step does ────────────────────────────────────────────────────────
with st.expander("What does this step do?", expanded=False):
    st.markdown("""
    1. **Interpolation** — ERA5-Land is a regular grid (~9 km). For each study region we pick
       the nearest grid point and apply a bilinear interpolation, then correct temperature and
       precipitation for the difference in elevation (lapse rate).

    2. **Bias correction** — ERA5-Land tends to have systematic errors in precipitation and
       temperature in complex terrain. SPARTACUS provides high-resolution (~1 km) daily
       observations that we use to correct those biases.

    Result: one hourly time series per (region, elevation band) — ready for SNOWPACK.
    """)

st.markdown("---")

# ── Region × elevation overview ───────────────────────────────────────────────
st.subheader("Simulations to prepare")
regions = cfg["regions"]
elevations = cfg["elevation_bands"]
rows = []
for rk, rm in regions.items():
    for elev in elevations:
        rows.append({"Region": rm["name"], "Key": rk, "Elevation (m)": elev})
df = pd.DataFrame(rows)
st.dataframe(df[["Region", "Elevation (m)"]], use_container_width=True, hide_index=True)
st.caption(f"{len(rows)} combinations total ({len(regions)} regions × {len(elevations)} elevation bands)")

st.markdown("---")

# ── Run processing ─────────────────────────────────────────────────────────────
st.subheader("Run processing")

skip_bias = st.checkbox(
    "Skip SPARTACUS bias correction (use raw ERA5)",
    value=False,
    help="Useful if SPARTACUS data is not downloaded yet — only ERA5 interpolation is run.",
)

if st.button("▶️  Run processing", type="primary"):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    with st.spinner("Interpolating ERA5 to region points …"):
        try:
            from scripts.interpolate_points import interpolate_era5_to_points
            era5_points = interpolate_era5_to_points(cfg)
            st.success(f"Interpolation complete — {sum(len(v) for v in era5_points.values())} series ready.")
        except Exception as e:
            st.error(f"Interpolation failed: {e}")
            st.stop()

    if not skip_bias:
        with st.spinner("Applying SPARTACUS bias correction …"):
            try:
                from scripts.bias_correction import apply_bias_correction
                corrected = apply_bias_correction(cfg, era5_points)
                st.session_state["corrected_era5"] = corrected
                st.success("Bias correction complete.")
            except Exception as e:
                st.error(f"Bias correction failed: {e}")
                st.stop()
    else:
        st.session_state["corrected_era5"] = era5_points
        st.info("Skipping bias correction — using raw ERA5 data.")

    st.success("✅  Processing complete. Proceed to Step 3 to write SNOWPACK input files.")

# ── Quick stats if data already in memory ─────────────────────────────────────
if "corrected_era5" in st.session_state:
    st.markdown("---")
    st.subheader("Quick statistics (processed data in memory)")

    corrected = st.session_state["corrected_era5"]
    stat_rows = []
    for rk, elev_dict in corrected.items():
        for elev, df_s in elev_dict.items():
            stat_rows.append({
                "Region": regions[rk]["name"],
                "Elevation (m)": elev,
                "Hours": len(df_s),
                "TA mean (°C)": round(df_s["TA"].mean() - 273.15, 1),
                "PSUM total (mm)": round(df_s["PSUM"].sum(), 0),
            })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
