# SNOWPACK Steiermark — Pipeline Documentation

This document explains what the programme does from start to finish: how data are
fetched, cleaned, fed into the snow-cover model, classified for avalanche problems,
and displayed in the web dashboard.

---

## Overview

```
lawinen.at (GeoSphere Austria)
        │
        ▼  download_geosphere.py
    Raw 10-min SMET data
        │
        ▼  smet_writer.py
    Cleaned / gap-filled SMET file
        │
        ▼  ini_writer.py + sno_writer.py
    SNOWPACK configuration (INI + SNO)
        │
        ▼  snowpack_runner.py  →  SNOWPACK binary
    Snow-cover simulation (PRO file)
        │
        ▼  avapro_runner.py  →  AVAPro
    Avalanche-problem classification (CSV)
        │
        ▼  app.py  →  index.html
    Web dashboard
```

The pipeline runs for **four stations** in sequence:

| ID   | Name                    | Snow station | Wind station | ILWR sensor | TSG sensor |
|------|-------------------------|-------------|-------------|-------------|------------|
| TAMI | Gesäuse Tamischbachturm | TAMI2       | TAMI1       | no (generated) | no (0 °C const.) |
| LOSE | Altaussee Loser         | LOSE2       | LOSE1       | yes (lango) | **broken** → 0 °C const. |
| VEIT | Veitsch                 | VEIT2       | VEIT1       | yes (lango) | **broken** → 0 °C const. |
| PLAN | Planneralm              | PLAN2       | PLAN1       | no (generated) | no (0 °C const.) |

---

## Step 1 — Data Download (`scripts/download_geosphere.py`)

### Source

GeoSphere Austria publishes station data on **lawinen.at** in SMET format
(10-minute intervals, rolling 8-day "woche/" window and full-season "winter/" archive).

Two station IDs are fetched per SNOWPACK station:
- **Snow station** (TA, RH, HS, ISWR, possibly ILWR/TSG)
- **Wind station** (VW, DW)

### Field renaming

lawinen.at uses non-standard field names for some sensors. These are renamed on ingest:

| Raw field | Renamed to | Reason |
|-----------|-----------|--------|
| `ISWR2`   | `ISWR`    | Redundant second pyranometer column name |
| `lango`   | `ILWR`    | Long-wave downward radiation (German abbreviation) |
| `tg`      | `TSG`     | Soil/snow surface temperature |

Unused columns (`ISWRu`, `langu`, `slope1*`) are dropped.

### State tracking

Each station stores a `state/{id}_download.json` file with the timestamp of the last
successfully downloaded row. Only data newer than this timestamp is appended on the next
run (incremental updates).

---

## Step 2 — SMET Writing (`scripts/smet_writer.py`)

The raw data are validated, corrected, and written to a SMET 1.1 file that SNOWPACK
(via MeteoIO) can read.

### Base fields (all stations)

```
timestamp  TA  RH  VW  DW  ISWR  PSUM  PSUM_PH  HS
```

### Optional fields

| Station | Additional fields |
|---------|------------------|
| LOSE, VEIT | `ILWR` (long-wave radiation) |

### Unit and value corrections

| Variable | Correction |
|----------|-----------|
| TA | Already in Kelvin from lawinen.at |
| RH | Already fractional (0–1) |
| HS | Already in metres |
| ISWR | Clipped ≥ 0 W/m² |
| ILWR | Values < 50 W/m² are blanked to nodata (`-999`). Real atmospheric ILWR is always ≥ ~100 W/m²; LOSE/VEIT `lango` sensors report 1–2 W/m² (broken) → blanking forces MeteoIO to use its ALLSKY_LW synthetic generator instead |
| TSG (tg) | Values < 200 K are assumed to be in °C and converted: `K = °C + 273.15`. Clipped to 200–310 K. If < 10 % of values are in the valid range, the entire column is replaced with 273.15 K (0 °C constant) |
| PSUM | Derived: `max(0, HS_diff) * 100 * density` approximation for new snow water equivalent |
| PSUM_PH | Phase: 1.0 (snow) when TA < 273.65 K (0.5 °C), else 0.0 (rain) |

### Gap-filling

Gaps in the 10-minute time series (missing data, sensor outages) are filled before
writing so that SNOWPACK never sees holes larger than one MeteoIO interpolation step:

| Variable | Gap-fill strategy | Max gap |
|----------|------------------|---------|
| TA | Linear interpolation | 10 days |
| RH | Linear interpolation | 10 days |
| ISWR | Linear interpolation, clip ≥ 0 | 10 days |
| ILWR | Blank < 50 W/m², then linear interpolation, clip ≥ 0 | 10 days |
| VW | Forward-fill, fallback 2.0 m/s | unlimited |
| DW | Forward-fill | unlimited |
| HS | Forward-fill only (no back-fill; snow height never interpolated backwards) | unlimited |
| TSG | Linear interpolation | 2 days |
| PSUM | Recomputed from gap-filled HS | — |
| PSUM_PH | Recomputed from gap-filled TA | — |

Nodata values remaining after gap-fill are written as `-999` (SMET nodata sentinel).

---

## Step 3 — SNOWPACK Configuration (`scripts/ini_writer.py`, `scripts/sno_writer.py`)

### INI file (`data/ini/{station_id}.ini`)

A MeteoIO + SNOWPACK configuration file is generated for each station. Key parameters:

- **EXPERIMENT** = station ID (determines output PRO filename: `{snow_station}_{ID}.pro`)
- **MEAS_INCOMING_LONGWAVE** = `TRUE` for LOSE/VEIT (measured ILWR in SMET);
  `FALSE` for TAMI/PLAN (MeteoIO ALLSKY_LW generator synthesises ILWR from TA+RH)
- **TSG generator** = `CST 273.15 K` for all stations (broken sensors disabled in
  `config.yaml`; MeteoIO supplies 0 °C constant when no measured TSG is available)
- MeteoIO **filters** applied by INI:
  - `TA`: min/max 200–320 K (SOFT)
  - `RH`: min/max 0.01–1.0 (SOFT)
  - `VW`: min 0 m/s (SOFT)
  - `ISWR`: min 0 W/m² (SOFT)
  - `ILWR` (when present): min/max 50–600 W/m² (SOFT)
  - `TSG` (when present): min/max 200–310 K (SOFT)

### SNO file (`data/sno/{snow_station}.sno`)

An empty initial snow profile is written at the season start (September 1) if one does
not yet exist. It describes a bare ground with no snow layers.

---

## Step 4 — SNOWPACK Simulation (`scripts/snowpack_runner.py`)

The SNOWPACK binary (`/Applications/Snowpack/bin/snowpack`) is run with the generated
INI file. It simulates the snow cover from the season start to the current date.

Output: a **PRO file** (`data/pro/{snow_station}_{station_id}.pro`) containing the full
snow-layer stratigraphy time series (one profile snapshot every output interval).

### Failure recovery

If SNOWPACK exits with an error (e.g., caused by corrupt forcing data), the last 20 lines
of the SNOWPACK log are captured and shown in the web dashboard. On the next run, the
forcing data is re-fetched and the simulation retried.

Past crash causes and fixes:

| Station | Cause | Fix |
|---------|-------|-----|
| LOSE | `lango` sensor reports 1–2 W/m² → MeteoIO wrote 50 W/m² ILWR → surface cooling → crash | ILWR values < 50 W/m² blanked to nodata in smet_writer |
| LOSE | `tg` (TSG) sensor oscillates 200–310 K → catastrophic node-0 temperature swing | `tsg: false` in config.yaml; CST 273.15 K generator used instead |
| VEIT | Same TSG oscillation as LOSE; 52.9 % of values technically in-range but transitions still extreme | `tsg: false` in config.yaml |

---

## Step 5 — Avalanche Problem Classification (`scripts/avapro_runner.py`)

The **AVAPro** library (part of the avalanche warning service Tyrol's `snowpacktools`
package) reads the PRO file and classifies the prevailing avalanche problems for each day
of the simulation.

### Output

`data/avapro_output/{station_id}_problems.csv` — one row per assessment. AVAPro
produces **4 assessments per calendar day** (morning/afternoon × dry/wet conditions).

### Aggregation

For display purposes the 4 daily rows are collapsed to one with `any()`: a problem is
shown as active on a day if at least one of the four assessments flagged it.

### Problem types

| Internal key | Display label |
|-------------|--------------|
| `new_snow` | Neuschnee |
| `wind_slab` | Triebschnee |
| `persistent_weak_layer` | Altschnee |
| `deep_slab` | Altschnee (Tief) |
| `wet_snow` | Nassschnee |
| `glide_snow` | *(not shown in dashboard)* |

---

## Step 6 — Web Dashboard (`app.py`, `templates/index.html`)

A **Flask** web application on port 5001 provides the dashboard.

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/status?station=` | Last simulation/download timestamps, current HS and TSS, today's problems |
| `GET /api/combined-chart?station=` | Daily-aggregated SMET meteo + AVAPRO problem flags for the season chart |
| `GET /api/timeseries?station=` | HS/TSS from the PRO file (SNOWPACK output) |
| `GET /api/smet-data?station=` | Full 10-min SMET data for the raw-data charts |
| `GET /api/avapro-season?station=` | Daily AVAPRO problem flags (all problems incl. glide snow) |
| `GET /api/pro-files` | List of available PRO files |
| `POST /run` | Start the full pipeline for all stations in a background thread |
| `GET /run/status` | Poll pipeline progress and log output |
| `POST /api/avapro-toggle` | Enable/disable AVAPRO classification step |
| `POST /git-push` | Push current data to the Git remote |

### Season chart (`📊 Saisonverlauf`)

A combined interactive chart shows both meteo forcing and avalanche problems for the
full season. Each layer can be toggled on/off independently:

**Meteo (from SMET forcing data, daily aggregates):**
- ❄ **HS** — snow height in cm (daily maximum), bar chart, left y-axis
- 🌡 **TA** — air temperature in °C (daily mean), line, right y-axis
- 💧 **RH** — relative humidity in % (daily mean)
- 💨 **VW** — wind speed in m/s (daily mean)
- ☀ **ISWR** — shortwave incoming radiation in W/m² (daily mean)

**Avalanche problems (from AVAPRO, daily any-aggregation):**
- ❄ Neuschnee / 🌬 Triebschnee / 🏔 Altschnee / ⬇ Altschnee (Tief) / 💧 Nassschnee

The AVAPRO bands are drawn on a second `<canvas>` element directly below the Chart.js
canvas, pixel-aligned to the chart's data area using `chart.chartArea.left`.

---

## Configuration (`config.yaml`)

Key settings:

```yaml
stations:           # list of all stations
  - id: TAMI
    snow_station: TAMI2   # lawinen.at station ID for snow/meteo data
    wind_station: TAMI1   # lawinen.at station ID for wind data
    ilwr: false           # true = measured ILWR in SMET; false = MeteoIO generator
    tsg: false            # true = measured TSG in SMET; false = CST 273.15 K

snowpack:
  binary: /Applications/Snowpack/bin/snowpack
  timeout: 3600

simulation:
  season_start_month: 9
  season_start_day: 1
  restart_from_sno: true

git:
  auto_push: true         # push after every pipeline run

web:
  host: 0.0.0.0
  port: 5001
```

---

## Running

```bash
# Full pipeline for all stations
python main.py

# Single station
python main.py --station LOSE

# Web dashboard
python app.py
# → http://localhost:5001
```

Automated cron job (example):
```
*/30 * * * *  cd /path/to/runsnowpacklive && python main.py >> logs/cron.log 2>&1
```
