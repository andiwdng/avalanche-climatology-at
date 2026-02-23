"""
Page 6 — niViz Profile Viewer
List all .pro output files and open them directly in niViz.
"""

import streamlit as st
import yaml
import subprocess
import threading
import http.server
import socket
import os
import time
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="niViz", page_icon="🔬", layout="wide")
st.title("🔬  niViz · Profile Viewer")
st.caption("Open SNOWPACK profile output directly in niViz — no manual file browsing needed.")

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

NIVIZ_DIR    = Path("/Applications/niviz")
NIVIZ_PORT   = 8000
FILE_PORT    = 8502

# Check if Node.js is available (also search conda paths)
import shutil
_node = shutil.which("node") or shutil.which(
    "node",
    path=os.environ.get("PATH", "") + ":/Users/andreaswedenig/miniconda3/bin"
)
NODE_AVAILABLE = _node is not None
NODE_BIN = _node or "node"

# Collect .pro files from both production and test output directories
_output_dirs = []
for key in ("snowpack_output", "snowpack_output_test"):
    _candidate = Path(cfg["paths"].get("snowpack_output", "data/snowpack_output"))
    if key == "snowpack_output_test":
        _candidate = _candidate.parent / (_candidate.name + "_test")
    if _candidate.exists():
        _output_dirs.append(_candidate)

# Primary root for the file server (parent of all outputs)
_base = Path(cfg["paths"]["snowpack_output"]).parent  # data/
output_root = _base  # served at FILE_PORT

# ── Helper: check if a port is in use ─────────────────────────────────────────
def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) == 0


# ── Helper: start CORS-enabled file server ────────────────────────────────────
class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):
        pass   # silence request logs


def _start_file_server(directory: Path, port: int):
    os.chdir(directory)
    server = http.server.HTTPServer(("localhost", port), _CORSHandler)
    server.serve_forever()


def ensure_file_server(directory: Path, port: int):
    if port_open(port):
        return   # already running
    t = threading.Thread(
        target=_start_file_server,
        args=(directory, port),
        daemon=True,
    )
    t.start()
    # wait up to 2 s for it to come up
    for _ in range(20):
        if port_open(port):
            break
        time.sleep(0.1)


# ── Helper: start niViz ───────────────────────────────────────────────────────
def start_niviz():
    if port_open(NIVIZ_PORT):
        return True, "niViz already running"
    if not (NIVIZ_DIR / "server.js").exists():
        return False, f"niViz not found at {NIVIZ_DIR}"
    if not (NIVIZ_DIR / "node_modules").exists():
        return False, "niViz dependencies not installed. Run `npm install` in /Applications/niviz/"
    subprocess.Popen(
        [NODE_BIN, "server.js"],
        cwd=str(NIVIZ_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if port_open(NIVIZ_PORT):
            return True, "niViz started"
        time.sleep(0.5)
    return False, "niViz did not start in time"


# ── Node.js warning ───────────────────────────────────────────────────────────
st.markdown("---")

if not NODE_AVAILABLE:
    st.warning(
        "**Node.js is not installed** — the local niViz server cannot start.  \n"
        "You can still download .pro files and open them in the **online niViz** at "
        "[run.niviz.org](https://run.niviz.org) (drag & drop the file there).  \n\n"
        "To enable one-click opening, install Node.js from [nodejs.org](https://nodejs.org) "
        "and run `npm install` in `/Applications/niviz/`."
    )

# ── Server status panel ───────────────────────────────────────────────────────
st.subheader("Server status")

col1, col2 = st.columns(2)

with col1:
    niviz_running = port_open(NIVIZ_PORT)
    if niviz_running:
        st.success(f"✓  niViz running at http://localhost:{NIVIZ_PORT}")
    elif NODE_AVAILABLE:
        st.warning(f"○  niViz not running (port {NIVIZ_PORT})")
    else:
        st.error("✗  Node.js not installed — local niViz unavailable")

    if NODE_AVAILABLE:
        if st.button("▶  Start niViz", disabled=niviz_running, type="primary"):
            with st.spinner("Starting niViz …"):
                ok, msg = start_niviz()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

with col2:
    file_srv_running = port_open(FILE_PORT)
    if file_srv_running:
        st.success(f"✓  File server running at http://localhost:{FILE_PORT}")
    elif NODE_AVAILABLE:
        st.info(f"○  File server starts automatically when you open a file (port {FILE_PORT})")
    else:
        st.info("File server not needed without local niViz.")

st.markdown("---")

# ── Scan for .pro files ───────────────────────────────────────────────────────
st.subheader("Available profile files (.pro)")

pro_files = []
for _d in _output_dirs:
    pro_files.extend(_d.rglob("*.pro"))
pro_files = sorted(pro_files)

if not pro_files:
    dirs_str = "\n".join(f"- `{d}`" for d in _output_dirs) if _output_dirs else f"- `{output_root}`"
    st.info(f"No .pro files found yet. Run SNOWPACK simulations in Step 4 first.\n\nLooked in:\n{dirs_str}")
    st.stop()

regions   = cfg["regions"]
elevations = cfg["elevation_bands"]

rows = []
for pro in pro_files:
    size_kb = pro.stat().st_size / 1024
    # Find which output dir this pro belongs to
    parent_dir = next((d for d in _output_dirs if pro.is_relative_to(d)), _output_dirs[0])
    # derive region/elev from path: {output_dir}/{region}/{elev}m/{station}.pro
    inner_parts = pro.relative_to(parent_dir).parts
    region_key  = inner_parts[0] if len(inner_parts) > 1 else "—"
    elev_part   = inner_parts[1] if len(inner_parts) > 2 else "—"
    region_name = regions.get(region_key, {}).get("name", region_key)
    # Label test runs
    is_test = "_test" in parent_dir.name
    label_suffix = " (test)" if is_test else ""

    # relative path from data/ root so the file server can find it
    rel_path = pro.relative_to(output_root)
    file_url = f"http://localhost:{FILE_PORT}/{rel_path.as_posix()}"
    niviz_url = f"http://localhost:{NIVIZ_PORT}?url={file_url}"

    rows.append({
        "Region":      region_name + label_suffix,
        "Elevation":   elev_part,
        "File":        pro.name,
        "Size (KB)":   f"{size_kb:.0f}",
        "_niviz_url":  niviz_url,
        "_pro_path":   pro,
    })

# ── Display table with Open buttons ───────────────────────────────────────────
for i, row in enumerate(rows):
    c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 2])
    c1.markdown(f"**{row['Region']}**")
    c2.markdown(row["Elevation"])
    c3.markdown(f"`{row['File']}`")
    c4.markdown(f"{row['Size (KB)']} KB")

    with c5:
        if NODE_AVAILABLE and niviz_running:
            # One-click: ensure file server is up, then open local niViz with URL
            ensure_file_server(output_root, FILE_PORT)
            st.markdown(
                f'<a href="{row["_niviz_url"]}" target="_blank">'
                f'<button style="background:#e63946;color:white;border:none;'
                f'padding:6px 14px;border-radius:6px;cursor:pointer;font-size:14px;">'
                f'Open in niViz</button></a>',
                unsafe_allow_html=True,
            )
        elif NODE_AVAILABLE:
            st.caption("Start niViz first ↑")
        else:
            # Fallback: open run.niviz.org (user must upload manually)
            st.markdown(
                f'<a href="https://run.niviz.org" target="_blank">'
                f'<button style="background:#457b9d;color:white;border:none;'
                f'padding:6px 14px;border-radius:6px;cursor:pointer;font-size:14px;">'
                f'niViz online</button></a>',
                unsafe_allow_html=True,
            )

    if i < len(rows) - 1:
        st.divider()

st.markdown("---")

# ── Download section ───────────────────────────────────────────────────────────
st.subheader("Download .pro files")
st.caption("Download a file to open it in the online niViz at run.niviz.org or any other tool.")

dl_options = {f"{row['Region']} · {row['Elevation']} · {row['File']}": row["_pro_path"] for row in rows}
selected = st.selectbox("Select file", options=list(dl_options.keys()))
if selected:
    pro_path = dl_options[selected]
    with open(pro_path, "rb") as fh:
        st.download_button(
            label=f"⬇  Download {pro_path.name}",
            data=fh,
            file_name=pro_path.name,
            mime="text/plain",
        )

st.markdown("---")
st.caption(
    "**How it works:** Clicking 'Open in niViz' starts a local file server (port 8502) that serves "
    "your .pro files, then opens niViz (port 8000) with the file URL as a parameter. "
    "niViz reads the URL and loads the profile automatically — no manual browsing needed."
)
