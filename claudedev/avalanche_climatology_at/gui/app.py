"""
app.py — Home page of the Avalanche Climatology GUI
Run with:  streamlit run gui/app.py
"""

import streamlit as st
import yaml
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Avalanche Climatology",
    page_icon="❄️",
    layout="wide",
)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# ── Load config into session state so all pages can access it ──────────────────
if "config" not in st.session_state:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as fh:
            st.session_state["config"] = yaml.safe_load(fh)
    else:
        st.session_state["config"] = None

cfg = st.session_state["config"]

# ── Home page content ──────────────────────────────────────────────────────────
st.title("❄️  Avalanche Climatology Pipeline")
st.caption("Austrian Alps · Reuter et al. (2023) methodology")

st.markdown("---")

if cfg is None:
    st.error(f"config.yaml not found at `{CONFIG_PATH}`. Make sure you run this from the repository root.")
    st.stop()

# Overview cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Regions", len(cfg.get("regions", {})))
with col2:
    st.metric("Elevation bands", len(cfg.get("elevation_bands", [])))
with col3:
    n_sim = len(cfg.get("regions", {})) * len(cfg.get("elevation_bands", []))
    st.metric("Total simulations", n_sim)
with col4:
    sim = cfg.get("simulation", {})
    period = f"{sim.get('analysis_start','?')} → {sim.get('analysis_end','?')}"
    st.metric("Analysis period", period)

st.markdown("---")

# Pipeline steps overview
st.subheader("Pipeline steps")
st.markdown("""
Use the **sidebar on the left** to navigate between steps. Work through them in order:

| Step | Page | What happens |
|------|------|-------------|
| 1 | 📥 Download | Download ERA5-Land and SPARTACUS data |
| 2 | ⚙️ Processing | Interpolate ERA5 to points, apply bias correction |
| 3 | 📄 SNOWPACK Inputs | Design, preview and **validate** INI + SMET files |
| 4 | ▶️ Simulations | Run all SNOWPACK simulations with a live progress view |
| 5 | 🏔 AVAPRO | Configure and run AVAPRO avalanche problem classifier |
""")

st.markdown("---")

# Region list
st.subheader("Configured regions")
regions = cfg.get("regions", {})
cols = st.columns(3)
for i, (key, meta) in enumerate(regions.items()):
    with cols[i % 3]:
        st.markdown(f"**{meta.get('name', key)}**  \n"
                    f"`{key}` · {meta.get('lat')}°N {meta.get('lon')}°E  \n"
                    f"{meta.get('province', '')}")
