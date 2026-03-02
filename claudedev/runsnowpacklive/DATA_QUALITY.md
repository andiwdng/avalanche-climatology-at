# Data Quality — Handling of Sensor Errors in the SNOWPACK Pipeline

This document explains what happens to the raw meteorological data from lawinen.at
at every stage of the pipeline, why certain corrections are applied, and what
specific sensor problems were found at each station. The goal is full transparency:
if a SNOWPACK result looks unexpected, you should be able to trace it back to a
data decision made here.

---

## 1. Data source

All meteorological data comes from **lawinen.at** via SMET 1.2 files:

```
https://lawinen.at/smet/stm/winter/{STATION}.smet.gz   ← full season (~September onwards)
https://lawinen.at/smet/stm/woche/{STATION}.smet.gz    ← last 7 days (incremental update)
```

Each station is split into two sub-stations on the source:

| Sub-station | Contains |
|-------------|----------|
| `{ID}1` — Windmessstation | TA, RH, VW, DW, DW\_MAX, VW\_MAX |
| `{ID}2` — Schneemessstation | TA, RH, HS, ISWR, and optionally ILWR (lango), TSG (tg) |

The pipeline merges both. Wind station VW/DW are used to fill gaps in the snow
station wind data (the wind station is usually at a more exposed location).

**Nodata value on the source:** `-777` → converted to `NaN` internally immediately
after reading.

---

## 2. Field name mapping

The lawinen.at files use non-standard field names for some variables.
These are renamed in `scripts/download_geosphere.py` (`_apply_field_renames`):

| lawinen.at field | SNOWPACK field | Stations | Reason |
|-----------------|---------------|---------|--------|
| `ISWR2` | `ISWR` | LOSE2, VEIT2 | Same quantity, different sensor label |
| `lango` | `ILWR` | LOSE2, VEIT2 | Gegenstrahlung (longwave incoming) |
| `tg` | `TSG` | LOSE2, VEIT2 | Bodentemperatur (ground temperature) |

Columns that SNOWPACK does not use are dropped immediately:
`ISWRu`, `langu`, `DW_MAX`, `VW_MAX`, `slope1*`

**Special case — VEIT2:** The SMET file from lawinen.at contains **both** a column
labelled `TSG` (a pre-existing label) and a column `tg` (the actual sensor value).
If both are present, the pre-labelled `TSG` column is dropped and `tg` is renamed
to `TSG` to ensure only one consistent source is used.

---

## 3. Per-station sensor configuration

The `config.yaml` declares which sensors are expected and usable at each station.
These flags control which columns appear in the SMET file and which MeteoIO options
are set in the INI file.

| Station | `ilwr` | `tsg` | Reason |
|---------|--------|-------|--------|
| TAMI | `false` | `false` | No longwave or ground-temperature sensor |
| LOSE | `true` | `false` | ILWR sensor present but TSG broken (see §7.2) |
| VEIT | `true` | `false` | ILWR sensor present but TSG broken (see §7.5) |
| PLAN | `false` | `false` | No longwave or ground-temperature sensor |

When a sensor is marked `false`, SNOWPACK/MeteoIO supplies a synthetic value
instead (see §6).

---

## 4. Raw value corrections (convert\_dataframe)

Applied in `scripts/smet_writer.py` (`SmetWriter.convert_dataframe`) before
anything is written to the SMET file.

### 4.1 TA (air temperature)
- Unit from source: **Kelvin** (lawinen.at already converts to K)
- Action: fill NaN → nodata (`-999`), no clipping

### 4.2 RH (relative humidity)
- Unit: dimensionless (0–1)
- Action: fill NaN → nodata. A MeteoIO range filter in the INI clips 0.05–1.0
  (SOFT) as a last safety net.

### 4.3 VW / DW (wind speed / direction)
- Action: fill NaN → nodata. Gap-filling applies forward-fill (see §5).

### 4.4 ISWR (shortwave incoming radiation)
- Action: clip to `≥ 0` (radiation cannot be negative), fill NaN → nodata.

### 4.5 ILWR (longwave incoming radiation) — stations with `ilwr: true`
- Unit from source: W/m²
- Physical minimum: atmospheric ILWR is **always ≥ ~100 W/m²** at realistic
  atmospheric temperatures. Values below 50 W/m² are physically impossible and
  indicate sensor noise or a broken sensor.
- Action:
  1. Values `< 50 W/m²` are set to `NaN` (treated as missing, not as valid data).
  2. Remaining values are clipped to `≥ 0`.
  3. NaN → nodata (`-999`) in the SMET file.
- If nodata is written, MeteoIO's `ALLSKY_LW` / `CLEARSKY_LW` generators
  synthesise a physically reasonable ILWR from air temperature and humidity.

### 4.6 TSG (ground/soil temperature) — stations with `tsg: true`
- Unit from lawinen.at: **Celsius** (despite the source SMET header sometimes
  saying "K"). The rule applied is: if a raw value is `< 200`, it is treated as
  Celsius and `+ 273.15` is added to convert to Kelvin. Values already ≥ 200 are
  left unchanged (already in Kelvin).
- After conversion the value is clipped to the range **200 K – 310 K**
  (−73 °C to +37 °C), which covers all physically realistic ground temperatures
  plus sensor saturation extremes.
- **Sensor-broken fallback:** if fewer than **10 %** of all non-NaN values fall
  in the "physically plausible, non-saturated" range (200.5 K – 309.5 K), the
  entire TSG column is replaced with **273.15 K (0 °C)** constant. A warning is
  logged. This protects SNOWPACK from wild oscillations caused by broken sensors.

### 4.7 PSUM / PSUM\_PH (precipitation)
- Always written as nodata (`-999`).
- Reason: SNOWPACK is run with `ENFORCE_MEASURED_SNOW_HEIGHTS = TRUE`. It
  derives precipitation internally from positive changes in snow height (ΔHS > 0).
  Measured precipitation from a heated gauge would be double-counted.

---

## 5. Gap-filling (\_fill\_gaps)

After the raw corrections, the pipeline reindexes the data to a **complete
10-minute grid** (no missing timestamps). Any gap in the measured data is then
filled as described below. The limits here are intentionally generous — SNOWPACK's
MeteoIO layer applies its own stricter filters afterwards.

| Variable | Fill method | Maximum gap filled | Fallback if no data |
|----------|-------------|-------------------|---------------------|
| TA | Linear interpolation | 10 days (1 440 steps) | — (nodata written) |
| RH | Linear interpolation | 10 days | — |
| ISWR | Linear interpolation, clip ≥ 0 | 10 days | — |
| ILWR | **Blank < 50 W/m² first**, then linear interpolation | 10 days | nodata → generator |
| VW | Forward-fill | 10 days | 2.0 m/s constant |
| DW | Forward-fill | 10 days | 180° constant |
| HS | Forward-fill only (never interpolate) | 10 days | 0.0 m |
| TSG | Linear interpolation | 2 days (288 steps) | — |
| PSUM / PSUM\_PH | — | always nodata | — |

**Why HS is only forward-filled (not interpolated):**
Interpolating snow height between two measured values would imply the snowpack
smoothly grew or melted during a data gap. In reality, a multi-day gap could hide
a snowfall event followed by settling. Forward-filling the last known height is
more conservative and avoids introducing phantom snow height changes that would
confuse SNOWPACK's mass balance.

**Why VW falls back to 2.0 m/s:**
Zero wind speed is physically unlikely over multi-day periods at alpine stations
and would suppress turbulent heat exchange in SNOWPACK. A small non-zero wind
speed is more realistic than zero.

---

## 6. MeteoIO filters and generators (INI file)

Even after the Python-side corrections, MeteoIO applies its own quality filters
at runtime before passing data to SNOWPACK. These are configured in the INI file
written by `scripts/ini_writer.py`.

### 6.1 Filters applied to all stations

| Variable | Filter | Limits | Mode |
|----------|--------|--------|------|
| TA | min\_max SOFT | 240 K – 320 K | SOFT: out-of-range → clamped |
| RH | min\_max | 0.01 – 1.2 | Hard reject, then SOFT 0.05 – 1.0 |
| VW | min\_max SOFT | 0.2 – 50 m/s | — |
| ISWR | min\_max SOFT | 0 – 1 500 W/m² | — |
| PSUM | min SOFT | ≥ 0 | — |
| HS | min SOFT + rate + MAD | ≥ 0; max 5.55 × 10⁻⁵ m/s | Spike / rate filter |

### 6.2 Additional filters for stations with `ilwr: true`

| Variable | Filter | Limits | Mode |
|----------|--------|--------|------|
| ILWR | min\_max SOFT | 50 – 600 W/m² | Values below 50 clamped to 50 (last safety net) |
| TSG | min\_max SOFT | 200 – 310 K | — |

### 6.3 Generators (synthetic values when measurement is nodata)

| Variable | Generator | When active |
|----------|-----------|-------------|
| ISWR | ALLSKY\_SW | When measured ISWR is nodata |
| RSWR | ISWR\_ALBEDO | Always (reflected SW derived from ISWR + albedo) |
| ILWR | ALLSKY\_LW (Unsworth) → CLEARSKY\_LW (Brutsaert) | When measured ILWR is nodata |
| TSG | CST = 273.15 K | Stations with `tsg: false` — constant 0 °C |

The ILWR generator chain (Unsworth → Brutsaert fallback) synthesises longwave
radiation from air temperature, humidity, and estimated cloud cover derived from
the ratio of measured ISWR to theoretical clear-sky ISWR. This is a well-validated
approach for alpine stations without longwave sensors.

---

## 7. Station-specific sensor problems found in 2025/26

### 7.1 LOSE — ILWR sensor (lango) completely broken

**What was observed:**
The `lango` sensor at LOSE2 reported values of exactly **1.0 W/m²** for the
entire winter season (2025-09-01 to present), with occasional values of 2.0 W/m².
All 21 213 non-NaN readings were below 3 W/m². Real atmospheric longwave radiation
at this altitude is never below ~150 W/m².

**Effect without correction:**
The SMET was written with ILWR = 1–2 W/m². MeteoIO's SOFT filter clamped these
to 50 W/m². With `MEAS_INCOMING_LONGWAVE = TRUE`, SNOWPACK used this 50 W/m²
value directly in the radiation balance. The snow surface radiated far more energy
than it received, cooling catastrophically to T = 209 K (−64 °C) at snow node 0
by 2025-11-18. SNOWPACK aborted with a runtime error.

**Fix applied:**
- `scripts/smet_writer.py`: ILWR values `< 50 W/m²` are set to NaN in both
  `convert_dataframe` and `_fill_gaps`. Since all LOSE ILWR values are 1–2 W/m²,
  the entire ILWR column in the SMET file is written as nodata (−999).
- MeteoIO's `ALLSKY_LW` generator then synthesises physically reasonable ILWR
  from TA and RH for the full season.
- `config.yaml` keeps `ilwr: true` for LOSE so the generator and MEAS flag
  are still correctly configured; the fix operates at the data layer, not by
  changing the station type.

**Result:** SNOWPACK runs successfully for the full 2025/26 season.

---

### 7.2 LOSE — TSG sensor (tg) broken

**What was observed:**
The `tg` sensor at LOSE2 showed a bimodal distribution: values clustered at
**310 K (37 °C)** and **200 K (−73 °C)**, i.e., constantly at the clip
boundaries. Transitions between these extremes produced values that passed the
10 % validity check (enough intermediate readings existed during the oscillations).

**Effect without correction:**
TSG was written to the SMET with these wild oscillations. In the forcing data
written by SNOWPACK, TSG dropped by ~1 K per 15-minute timestep from 310 K
through 224 K toward 209 K — the same crash temperature as the ILWR problem
above, now driven by the TSG column instead.

**Fix applied:**
`config.yaml`: `tsg: false` for LOSE.

This completely removes the TSG column from the SMET file. The INI file then
includes:
```
TSG::create     = CST
TSG::CST::VALUE = 273.15
```
MeteoIO supplies a constant 0 °C ground temperature for the entire simulation.
This is the standard approach used for TAMI and PLAN, which have no TSG sensor.

**Why not try to repair the broken TSG instead?**
The sensor oscillates between physical impossibility values (±73 °C, +37 °C)
with no period where it appears reliable. There is no valid reference to
interpolate from. A constant 0 °C is a far better model of ground temperature
in November–March at 1 573 m than a random number between −73 °C and +37 °C.

**Result:** SNOWPACK runs successfully. TSG is handled identically to TAMI/PLAN.

---

### 7.3 LOSE — Duplicate timestamps in source SMET

**What was observed:**
The LOSE SMET files from lawinen.at contain duplicate timestamps (same UTC minute
appears more than once with slightly different values, likely caused by the
lawinen.at data ingestion pipeline).

**Effect without correction:**
pandas raises `ValueError: cannot reindex on an axis with duplicate labels`
when the merger tries to align the two SMET files on their timestamp index.

**Fix applied:**
`scripts/download_geosphere.py` (`_merge_stations`): before reindexing, both
dataframes are deduplicated with `.duplicated(keep="last")`. The last occurrence
of any duplicate timestamp is kept (assumed to be the most recently corrected value).

---

### 7.5 VEIT — TSG sensor (tg) broken

**What was observed:**
The `tg` sensor at VEIT2 showed a distribution of 10 303 values at **310 K
(+37 °C)**, 1 378 values at **200 K (−73 °C)**, and 13 176 values (52.9 %)
in the intermediate range 200.5–309.5 K. The intermediate values are transitions
between the two saturation extremes, not real physical ground temperatures.

In the SNOWPACK forcing data, TSG dropped by ~1 K per 15-minute timestep from
310 K through 203 K toward the clip floor, causing node 0 to crash at T = 208 K
(−65 °C) on 2025-11-21.

**Effect without correction:** Same runtime crash as LOSE (§7.2): T out of
bound at node 0, SNOWPACK aborts.

**Why the 10 % fallback did not trigger:**
The fallback checks whether fewer than 10 % of values fall in the valid range
200.5–309.5 K. For VEIT, 52.9 % of values technically land in that range (they
are transitions between the broken extremes, not real temperatures). The threshold
was set conservatively to catch the fully-saturated LOSE case; VEIT's partially-
saturated sensor slipped through.

**Fix applied:** `config.yaml`: `tsg: false` for VEIT. Same CST 273.15 K
generator as TAMI, PLAN, and LOSE.

---

### 7.4 VEIT — Both TSG and tg columns present

**What was observed:**
The VEIT2 SMET file from lawinen.at contains **both** a column already labelled
`TSG` and a separate `tg` column. The pre-labelled `TSG` appears to be an
older sensor reading; `tg` is the current one.

**Effect without correction:**
After renaming `tg → TSG`, pandas had two columns named `TSG`. Assigning a
single computed series to `df["TSG"]` raised:
`ValueError: Cannot set a DataFrame with multiple columns to the single column TSG`

**Fix applied:**
`scripts/download_geosphere.py` (`_apply_field_renames`): if both `TSG` and `tg`
are present, the existing `TSG` column is **dropped first**, then `tg` is renamed
to `TSG`.

---

## 8. What happens when SNOWPACK still fails

The pipeline is designed to continue even when one station's SNOWPACK run fails.
Each station is wrapped in a try/except block in both `main.py` (CLI) and `app.py`
(web). A failed station is logged as an error; the other stations continue normally.

If SNOWPACK fails:
1. Check `logs/snowpack_{station}_{timestamp}.log` for the error message.
2. The most common failure mode is an out-of-bound temperature at a snow node,
   caused by a physically inconsistent combination of forcing data.
3. A `--full-reset` for the affected station (which deletes the SNO profile and
   state file and re-downloads the full winter season) is the standard recovery
   procedure after a code fix.

---

## 9. Summary table

| Issue | Station | Symptom in SNOWPACK | Root cause | Fix location | Fix |
|-------|---------|-------------------|------------|-------------|-----|
| ILWR sensor reports 1–2 W/m² | LOSE | T = 209 K crash at node 0 | `lango` sensor broken | `smet_writer.py` | Blank ILWR < 50 W/m² → nodata → generator |
| TSG sensor oscillates 200–310 K | LOSE, VEIT | T = 208–209 K crash at node 0 | `tg` sensor broken | `config.yaml` | Set `tsg: false` → CST 273.15 K |
| Duplicate timestamps | LOSE | pandas ValueError on merge | lawinen.at data issue | `download_geosphere.py` | Dedup with `keep="last"` |
| Both TSG and tg columns present | VEIT | pandas ValueError on assignment | lawinen.at field labelling | `download_geosphere.py` | Drop pre-labelled TSG before rename |
| TSG < 10 % valid values | Any | Wild TSG driving unrealistic gradients | Broken/saturated sensor | `smet_writer.py` | Auto-fallback to 273.15 K constant |
