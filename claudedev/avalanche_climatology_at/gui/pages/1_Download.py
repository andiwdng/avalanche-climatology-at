"""
Page 1 — Data Download
Download ERA5-Land (CDS API) and SPARTACUS data.
"""

import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(page_title="Download", page_icon="📥", layout="wide")
st.title("📥  Step 1 · Data Download")
st.caption("Download ERA5-Land hourly reanalysis and SPARTACUS daily gridded data.")

# ── Load config ────────────────────────────────────────────────────────────────
cfg = st.session_state.get("config")
if cfg is None:
    CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)
        st.session_state["config"] = cfg
    else:
        st.error("config.yaml not found. Go back to the Home page first.")
        st.stop()

sim = cfg["simulation"]
era5_dir = Path(cfg["paths"]["era5_raw"])
spartacus_dir = Path(cfg["paths"]["spartacus"])

st.markdown("---")

# ── ERA5-Land status ───────────────────────────────────────────────────────────
st.subheader("ERA5-Land hourly forcing")

col1, col2 = st.columns([2, 1])
with col1:
    import datetime
    start_year = datetime.date.fromisoformat(sim["analysis_start"]).year - int(sim.get("spin_up_years", 1))
    end_year   = datetime.date.fromisoformat(sim["analysis_end"]).year
    years = list(range(start_year, end_year + 1))

    rows = []
    for y in years:
        f = era5_dir / f"era5land_forcing_{y}.nc"
        size_mb = f.stat().st_size / 1e6 if f.exists() else None
        rows.append({
            "Year": y,
            "File": f.name,
            "Status": "✓  Downloaded" if f.exists() else "✗  Missing",
            "Size (MB)": f"{size_mb:.0f}" if size_mb else "—",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

with col2:
    n_ok = sum(1 for r in rows if "✓" in r["Status"])
    st.metric("Downloaded", f"{n_ok} / {len(years)}")
    orog = era5_dir / "era5land_orography.nc"
    st.metric("Orography file", "✓ Present" if orog.exists() else "✗ Missing")

st.markdown("---")

# ── SPARTACUS status ───────────────────────────────────────────────────────────
st.subheader("SPARTACUS daily grids")
spartacus_files = sorted(spartacus_dir.glob("*.nc")) if spartacus_dir.exists() else []
st.metric("Files found", len(spartacus_files))
if spartacus_files:
    st.caption(f"Location: `{spartacus_dir}`")

st.markdown("---")

# ── Download buttons ───────────────────────────────────────────────────────────
st.subheader("Run downloads")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**ERA5-Land via CDS API**")
    st.caption("Requires `~/.cdsapirc` credentials. Downloads can take a long time — run this in the background if needed.")
    if st.button("⬇️  Download ERA5-Land", type="primary"):
        with st.spinner("Downloading ERA5-Land … (this may take a while)"):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from scripts.download_era5 import download_era5
                download_era5(cfg)
                st.success("ERA5-Land download complete.")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")

with col_b:
    st.markdown("**SPARTACUS via GeoSphere API**")
    st.caption("Downloads SPARTACUS-v2 daily temperature and precipitation grids.")
    if st.button("⬇️  Download SPARTACUS", type="primary"):
        with st.spinner("Downloading SPARTACUS …"):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from scripts.download_spartacus import download_spartacus
                download_spartacus(cfg)
                st.success("SPARTACUS download complete.")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")
