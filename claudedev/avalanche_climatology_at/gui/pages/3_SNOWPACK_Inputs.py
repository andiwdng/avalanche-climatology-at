"""
Page 3 — SNOWPACK Input Designer & Validator
Preview SMET files, validate INI keys, write all inputs.
"""

import streamlit as st
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import configparser
import io

st.set_page_config(page_title="SNOWPACK Inputs", page_icon="📄", layout="wide")
st.title("📄  Step 3 · SNOWPACK Inputs")
st.caption("Preview and validate SMET forcing files and INI configuration before running simulations.")

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

# ── Required SNOWPACK INI keys (section → list of required keys) ───────────────
REQUIRED_INI_KEYS = {
    "Snowpack": [
        "CALCULATION_STEP_LENGTH",
        "METEO_STEP_LENGTH",
        "ENFORCE_MEASURED_SNOW_HEIGHTS",
        "CANOPY",
        "HEIGHT_OF_WIND_VALUE",
        "HEIGHT_OF_METEO_VALUES",
        "THRESH_RAIN",
        "THRESH_DTEMP_AIR_SNOW",
        "SW_MODE",
        "SLOPE_ANGLE",
        "SLOPE_ASP",
    ],
    "Input": [
        "METEO",
        "METEOPATH",
        "STATION1",
        "SNOWPATH",
        "NUMBER_OF_STATIONS",
    ],
    "Output": [
        "METEOPATH",
        "SNOWPATH",
        "PROFILE",
    ],
    "General": [
        "CATCHMENT",
        "ALPINE3D",
    ],
}


def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse INI text into {section: {key: value}} (case-insensitive keys, original-case sections)."""
    parser = configparser.RawConfigParser()
    parser.optionxform = str  # preserve key case
    # configparser needs a DEFAULT section — add a dummy one
    try:
        parser.read_string("[__root__]\n" + text)
    except Exception:
        return {}
    result = {}
    for section in parser.sections():
        if section == "__root__":
            continue
        result[section] = dict(parser[section])
    return result


def validate_ini(text: str) -> list[dict]:
    """
    Check that all required keys are present in the correct sections.
    Returns a list of result dicts with keys: section, key, found.
    """
    parsed = parse_ini(text)
    # Build a case-insensitive section lookup
    section_lookup = {s.lower(): s for s in parsed}

    results = []
    for section, keys in REQUIRED_INI_KEYS.items():
        matched_section = section_lookup.get(section.lower())
        for key in keys:
            if matched_section is None:
                found = False
            else:
                # case-insensitive key check
                found = any(k.upper() == key.upper() for k in parsed[matched_section])
            results.append({"Section": f"[{section}]", "Key": key, "Status": "✓" if found else "✗  MISSING"})
    return results


def read_smet(path: Path) -> pd.DataFrame | None:
    """Read a SMET file into a DataFrame."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Find fields line and data start
        fields = None
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("fields"):
                fields = line.split("=", 1)[1].strip().split()
            if line.strip() == "[DATA]":
                data_start = i + 1
                break
        if fields is None or data_start is None:
            return None
        data_lines = [l for l in lines[data_start:] if l.strip()]
        rows = [l.split() for l in data_lines]
        df = pd.DataFrame(rows, columns=fields)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in df.columns:
            if col != "timestamp":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace(-999, float("nan"))
        return df.set_index("timestamp")
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📄 INI Template & Validation", "📊 SMET Preview", "▶️ Write All Inputs"])


# ── TAB 1: INI Template & Validation ─────────────────────────────────────────
with tab1:
    st.subheader("INI template")
    st.markdown(
        "This is the **master template** used for every simulation. "
        "The paths and station ID are filled in automatically per simulation — "
        "everything else is shared."
    )

    # Load an existing INI as starting point if available
    input_dir = Path(cfg["paths"]["snowpack_input"])
    existing_inis = sorted((input_dir / "ini").glob("*.ini")) if (input_dir / "ini").exists() else []

    if existing_inis:
        example_ini = existing_inis[0].read_text(encoding="utf-8")
    else:
        example_ini = """; No INI files generated yet.
; Go to the 'Write All Inputs' tab and generate them first,
; then come back here to review and validate."""

    ini_text = st.text_area(
        "INI content (edit directly here)",
        value=example_ini,
        height=400,
        label_visibility="collapsed",
    )

    col_validate, col_info = st.columns([1, 2])
    with col_validate:
        do_validate = st.button("🔍  Validate INI", type="primary", use_container_width=True)

    if do_validate:
        results = validate_ini(ini_text)
        df_results = pd.DataFrame(results)

        missing = df_results[df_results["Status"] != "✓"]
        ok = df_results[df_results["Status"] == "✓"]

        if missing.empty:
            st.success(f"✅  All {len(ok)} required keys are present.")
        else:
            st.error(f"❌  {len(missing)} required key(s) missing:")
            st.dataframe(missing, use_container_width=True, hide_index=True)
            st.markdown("---")
            st.markdown("**All checks:**")

        # Always show full table
        st.dataframe(df_results, use_container_width=True, hide_index=True)


# ── TAB 2: SMET Climatology Viewer ───────────────────────────────────────────
with tab2:
    st.subheader("SMET forcing — yearly climatology")
    st.caption(
        "SMET is the **final hourly input to SNOWPACK**. "
        "It is derived from ERA5-Land via spatial interpolation and elevation lapse-rate adjustment "
        "(and optionally SPARTACUS bias correction for radiation). "
        "Use this view to check plausibility before running simulations."
    )

    # Find available SMET directories (production + test)
    _smet_dirs = {}
    _prod_smet = input_dir / "smet"
    _test_smet = (input_dir.parent / (input_dir.name + "_test")) / "smet"
    if _prod_smet.exists() and list(_prod_smet.glob("*.smet")):
        _smet_dirs["production"] = _prod_smet
    if _test_smet.exists() and list(_test_smet.glob("*.smet")):
        _smet_dirs["test"] = _test_smet

    if not _smet_dirs:
        st.info("No SMET files found yet. Go to the 'Write All Inputs' tab to generate them first.")
    else:
        regions    = cfg["regions"]
        elevations = cfg["elevation_bands"]

        col_src, col_r, col_e = st.columns(3)
        with col_src:
            src_choice = st.selectbox("Data source", options=list(_smet_dirs.keys()))
        smet_dir = _smet_dirs[src_choice]
        with col_r:
            region_choice = st.selectbox(
                "Region", options=list(regions.keys()),
                format_func=lambda k: regions[k]["name"], key="smet_region",
            )
        with col_e:
            elev_choice = st.selectbox("Elevation (m)", options=elevations, key="smet_elev")

        station_id = f"{region_choice}_{elev_choice}m"
        smet_path  = smet_dir / f"{station_id}.smet"

        if not smet_path.exists():
            st.warning(f"SMET file not found: `{smet_path.name}`")
        else:
            df_h = read_smet(smet_path)
            if df_h is None:
                st.error("Could not parse SMET file.")
            else:
                st.caption(
                    f"`{smet_path.name}` · {len(df_h):,} hourly records · "
                    f"Fields: `{' '.join(df_h.columns.tolist())}`"
                )

                # ── Unit conversions ──────────────────────────────────────────
                df_h = df_h.copy()
                df_h["TA_C"]    = df_h["TA"] - 273.15          # K → °C
                df_h["PSUM_mm"] = df_h["PSUM"]                 # already mm/h
                df_h["P_hPa"]   = df_h["P"] / 100.0            # Pa → hPa

                # ── Daily aggregates ──────────────────────────────────────────
                df_d = pd.DataFrame({
                    "TA_mean":  df_h["TA_C"].resample("D").mean(),
                    "TA_min":   df_h["TA_C"].resample("D").min(),
                    "TA_max":   df_h["TA_C"].resample("D").max(),
                    "PSUM":     df_h["PSUM_mm"].resample("D").sum(),
                    "RH_mean":  df_h["RH"].resample("D").mean(),
                    "VW_mean":  df_h["VW"].resample("D").mean(),
                    "VW_max":   df_h["VW"].resample("D").max(),
                    "ISWR":     df_h["ISWR"].resample("D").mean(),
                    "ILWR":     df_h["ILWR"].resample("D").mean(),
                })

                # ── Summary metrics ───────────────────────────────────────────
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("TA mean",     f"{df_d['TA_mean'].mean():.1f} °C")
                m2.metric("TA min",      f"{df_d['TA_min'].min():.1f} °C")
                m3.metric("TA max",      f"{df_d['TA_max'].max():.1f} °C")
                m4.metric("PSUM total",  f"{df_d['PSUM'].sum():.0f} mm")
                m5.metric("VW max",      f"{df_d['VW_max'].max():.1f} m/s")

                st.markdown("---")

                # ── Main climatology plot ─────────────────────────────────────
                fig = make_subplots(
                    rows=5, cols=1,
                    shared_xaxes=True,
                    subplot_titles=(
                        "Air temperature  [°C]  — daily mean ± range",
                        "Precipitation  [mm/day]",
                        "Relative humidity  [–]  — daily mean",
                        "Wind speed  [m/s]  — daily mean + max",
                        "Shortwave & Longwave radiation  [W/m²]  — daily mean",
                    ),
                    vertical_spacing=0.06,
                    row_heights=[0.25, 0.2, 0.15, 0.2, 0.2],
                )

                x = df_d.index

                # TA with min/max band
                fig.add_trace(go.Scatter(
                    x=list(x) + list(x[::-1]),
                    y=list(df_d["TA_max"]) + list(df_d["TA_min"][::-1]),
                    fill="toself", fillcolor="rgba(74,144,217,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="TA range",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=df_d["TA_mean"], name="TA mean",
                    line=dict(color="#4A90D9", width=1.2)), row=1, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=0.8, row=1, col=1)

                # Precipitation coloured by phase (PSUM_PH: 0=snow,1=rain)
                df_d_rain = df_h[df_h["PSUM_PH"] > 0.5]["PSUM_mm"].resample("D").sum()
                df_d_snow = df_h[df_h["PSUM_PH"] <= 0.5]["PSUM_mm"].resample("D").sum()
                fig.add_trace(go.Bar(x=x, y=df_d["PSUM"], name="Precip total",
                    marker_color="#5BA85A", showlegend=True), row=2, col=1)
                fig.add_trace(go.Bar(x=df_d_rain.index, y=df_d_rain, name="Rain",
                    marker_color="#3399FF"), row=2, col=1)
                fig.add_trace(go.Bar(x=df_d_snow.index, y=df_d_snow, name="Snow",
                    marker_color="#AADDFF"), row=2, col=1)

                # RH
                fig.add_trace(go.Scatter(x=x, y=df_d["RH_mean"], name="RH",
                    line=dict(color="#E67E22", width=1.0)), row=3, col=1)
                fig.add_hline(y=1.0, line_dash="dot", line_color="grey", line_width=0.8, row=3, col=1)

                # Wind
                fig.add_trace(go.Scatter(x=x, y=df_d["VW_max"], name="VW max",
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
                    line=dict(color="rgba(155,89,182,0.3)", width=0.5)), row=4, col=1)
                fig.add_trace(go.Scatter(x=x, y=df_d["VW_mean"], name="VW mean",
                    line=dict(color="#9B59B6", width=1.2)), row=4, col=1)

                # Radiation
                fig.add_trace(go.Scatter(x=x, y=df_d["ISWR"], name="ISWR",
                    line=dict(color="#F39C12", width=1.0)), row=5, col=1)
                fig.add_trace(go.Scatter(x=x, y=df_d["ILWR"], name="ILWR",
                    line=dict(color="#E74C3C", width=1.0)), row=5, col=1)

                fig.update_layout(
                    height=900, showlegend=True,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=-0.04),
                    barmode="overlay",
                    plot_bgcolor="white",
                )
                fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
                st.plotly_chart(fig, use_container_width=True)

                # ── Monthly statistics table ──────────────────────────────────
                st.markdown("**Monthly statistics**")
                MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                monthly_rows = []
                for m in range(1, 13):
                    sub = df_d[df_d.index.month == m]
                    if sub.empty:
                        continue
                    monthly_rows.append({
                        "Month":         MONTH_NAMES[m - 1],
                        "TA mean (°C)":  f"{sub['TA_mean'].mean():.1f}",
                        "TA min (°C)":   f"{sub['TA_min'].min():.1f}",
                        "TA max (°C)":   f"{sub['TA_max'].max():.1f}",
                        "Precip (mm)":   f"{sub['PSUM'].sum():.0f}",
                        "Rain frac (%)": f"{100 * (df_h[df_h.index.month == m]['PSUM_PH'].mean()):.0f}",
                        "RH mean":       f"{sub['RH_mean'].mean():.2f}",
                        "VW mean (m/s)": f"{sub['VW_mean'].mean():.1f}",
                    })
                st.dataframe(pd.DataFrame(monthly_rows), use_container_width=True, hide_index=True)

                # ── Data quality checks ───────────────────────────────────────
                with st.expander("Data quality checks"):
                    checks = [
                        ("Negative PSUM",        int((df_h["PSUM"] < -0.001).sum()),   "0"),
                        ("RH > 1.0",             int((df_h["RH"] > 1.0).sum()),         "0 (soft filter clamps to 1)"),
                        ("RH < 0",               int((df_h["RH"] < 0).sum()),           "0"),
                        ("Negative ISWR (day)",  int((df_h["ISWR"] < -1).sum()),        "0"),
                        ("TA < 220 K",           int((df_h["TA"] < 220).sum()),         "0  (< −53 °C is unrealistic)"),
                        ("TA > 320 K",           int((df_h["TA"] > 320).sum()),         "0  (> +47 °C is unrealistic)"),
                        ("Missing TA",           int(df_h["TA"].isna().sum()),          "0"),
                        ("Missing PSUM",         int(df_h["PSUM"].isna().sum()),        "0"),
                    ]
                    q_df = pd.DataFrame(checks, columns=["Check", "Count", "Expected"])
                    q_df["OK"] = q_df["Count"].apply(lambda v: "✓" if v == 0 else "⚠️")
                    st.dataframe(q_df, use_container_width=True, hide_index=True)


# ── TAB 3: Write All Inputs ───────────────────────────────────────────────────
with tab3:
    st.subheader("Write all SNOWPACK input files")
    st.markdown(
        "This generates one **SMET** (forcing), one **SNO** (initial snowpack state) "
        "and one **INI** (configuration) file for each simulation. "
        "All INI files share the same structure — only paths and station ID differ."
    )

    regions = cfg["regions"]
    elevations = cfg["elevation_bands"]
    n_total = len(regions) * len(elevations)

    st.info(f"Will write **{n_total}** sets of input files ({len(regions)} regions × {len(elevations)} elevation bands).")

    corrected = st.session_state.get("corrected_era5")
    if corrected is None:
        st.warning(
            "No processed ERA5 data found in memory. "
            "Go to **Step 2 · Processing** and run the interpolation first, "
            "or the pipeline will fail here."
        )

    if st.button("📝  Write all input files", type="primary", disabled=(corrected is None)):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.snowpack_writer import write_all_snowpack_inputs

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Writing SMET, SNO and INI files …")
            ini_paths = write_all_snowpack_inputs(cfg, corrected)
            st.session_state["ini_paths"] = ini_paths

            progress.progress(100)
            n_written = sum(len(v) for v in ini_paths.values())
            status.success(f"✅  Written {n_written} sets of input files.")

            # Show what was written
            rows = []
            for rk, elev_dict in ini_paths.items():
                for elev, ini_path in elev_dict.items():
                    smet_path = Path(cfg["paths"]["snowpack_input"]) / "smet" / f"{rk}_{elev}m.smet"
                    rows.append({
                        "Station": f"{rk}_{elev}m",
                        "INI": "✓" if ini_path.exists() else "✗",
                        "SMET (KB)": f"{smet_path.stat().st_size / 1024:.0f}" if smet_path.exists() else "✗",
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info("👉  Switch to the **INI Validation** tab to check all required keys are present.")

        except Exception as e:
            progress.progress(0)
            status.error(f"Failed: {e}")
