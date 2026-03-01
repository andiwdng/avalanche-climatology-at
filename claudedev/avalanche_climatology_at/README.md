# Avalanche Climatology Pipeline — Austrian Alps

A reproducible Python pipeline for computing an avalanche climatology of the
Austrian Alps following the methodology of Reuter et al. (2023), using ERA5-Land
reanalysis, GeoSphere Austria SPARTACUS bias correction, SNOWPACK snow-cover
simulation, and AVAPRO avalanche problem classification.

---

## Current Status (last updated 2026-03-01)

### What is done

| Step | Status | Notes |
|------|--------|-------|
| Pipeline scaffolding | Done | `main.py`, all scripts, `config.yaml`, `requirements.txt` |
| ERA5-Land downloader | Done | Quarterly chunks per variable, ZIP extraction, `already on disk` resume logic |
| ERA5 de-accumulation fix | Done | Carry-over correction for `total_precipitation` and `surface_solar_radiation_downwards`; units converted to mm/h and W/m² |
| ERA5 orography | Done | Downloaded as ERA5 single-levels (not ERA5-Land, which has no orography variable) |
| CDS API compatibility | Done | Works with new Copernicus CDS API v2 (ecmwf-datastores); ZIP extraction instead of direct NetCDF |
| ERA5 years 1999–2001 | **Downloaded & merged** | `data/era5_raw/era5land_forcing_{year}.nc` exist for 1999, 2000, 2001 |
| ERA5 year 2002 | **Partial** — stopped mid-download | See below |
| ERA5 years 2003–2021 | Not started | Will run automatically once 2002 finishes |
| SPARTACUS download | Not started | Script exists (`download_spartacus.py`) |
| Interpolation & bias correction | Not started | Scripts exist |
| SNOWPACK runs | Not started | Scripts exist |
| AVAPRO / classification | Not started | Scripts exist |
| Clustering & figures | Not started | Scripts exist |

### Where we stopped — ERA5 download

The ERA5 download ran from **2026-02-22** and completed years 1999–2001 plus part of 2002.
On **2026-02-26** the machine lost DNS resolution to `cds.climate.copernicus.eu` for ~5 days.
The CDS job for **2002 `total_precipitation Q3`** (job ID `c6cac0cb-3b43-4387-a277-77d27baca18e`)
was submitted successfully but its results expired on the CDS server before the network recovered.
The process crashed on **2026-03-01** with `404 results expired`.

**State of `data/era5_raw/` on stop:**

```
era5land_orography.nc          ← done
era5land_forcing_1999.nc       ← done (merged)
era5land_forcing_2000.nc       ← done (merged)
era5land_forcing_2001.nc       ← done (merged)
_stage_2002/
  2m_temperature_Q1–Q4.nc      ← done
  2m_dewpoint_temperature_Q1–Q4.nc  ← done
  total_precipitation_Q1.nc    ← done
  total_precipitation_Q2.nc    ← done
  total_precipitation_Q3.nc    ← MISSING (job expired)
  ... remaining variables      ← not started
```

### To resume

```bash
cd avalanche_climatology_at/
python scripts/download_era5.py
```

The script checks `already on disk` for every quarterly file before requesting it,
so it will skip the 30+ files already downloaded and continue from 2002 Q3.
Expected remaining download: years 2002–2021 (~19 years × ~7 variables × 4 quarters).

---

## Scientific Background

### Motivation

Avalanche climatology characterises the seasonal and regional distribution of
avalanche problem types — the dominant instability mechanisms that drive avalanche
activity at a given time and place.  Understanding long-term patterns enables:

- Evidence-based avalanche warning service calibration
- Climate-change impact assessment on alpine hazard
- Identification of analogous seasons for operational forecasting

### Methodology (Reuter et al. 2023)

This pipeline closely follows the workflow described in:

> Reuter, B., Viallon-Galinier, L., Horton, S., van Herwijnen, A.,
> Hagenmuller, P., Morin, S., & Schweizer, J. (2023).
> *Characterizing snow instability with avalanche problem types derived
> from snow cover simulations.*
> Cold Regions Science and Technology, 207, 103772.
> https://doi.org/10.1016/j.coldregions.2022.103772

Key adaptations for the Austrian context:

| Reuter et al.        | This pipeline                                   |
|----------------------|------------------------------------------------|
| SAFRAN reanalysis    | ERA5-Land (Muñoz-Sabater et al. 2021)          |
| SAFRAN bias via obs  | SPARTACUS-v2 (GeoSphere Austria) bias correction|
| French Alps regions  | 12 Austrian warning regions (LWD network)      |
| Crocus snow model    | SNOWPACK (SLF)                                 |
| Python / R analysis  | Python 3.11 throughout                        |

### Data Sources

| Dataset     | Variable              | Resolution | Source                        |
|-------------|----------------------|------------|-------------------------------|
| ERA5-Land   | T₂ₘ, Td, tp, P, u₁₀, v₁₀, SSRD, STRD | ~9 km, 1 h | Copernicus CDS |
| SPARTACUS-v2| Daily mean T, daily RR | ~1 km, 1 d | GeoSphere Austria OGD |
| SNOWPACK    | Full snow stratigraphy | Point      | SLF (model) |
| AVAPRO      | Avalanche problem types| Point, daily| AWSoM toolchain |

### Elevation Correction

Temperature is corrected from the ERA5-Land native surface height to each
target elevation band using the ICAO standard atmospheric lapse rate
(Γ = −0.0065 K m⁻¹).  Precipitation increases with elevation at +3 % per
100 m.  Pressure follows the hypsometric equation.

### Bias Correction

1. **Temperature (additive):** For each calendar day, the difference between
   SPARTACUS daily mean temperature and ERA5-Land daily mean temperature is
   computed and applied uniformly to all 24 hourly ERA5 values.  This preserves
   the ERA5 diurnal cycle while anchoring daily means to the observation-based
   SPARTACUS analysis.

2. **Precipitation (multiplicative):** ERA5 daily totals are scaled to
   SPARTACUS daily totals.  On ERA5-dry / SPARTACUS-wet days, the SPARTACUS
   amount is distributed uniformly across 24 hours.

### SNOWPACK Simulation

Flat-field (slope = 0°) point simulations are run at four elevation bands
(1500, 2000, 2500, 3000 m a.s.l.) for each of 12 Austrian study regions.
The simulation period covers 2000–2021 (including 1 spin-up year from 1999).

### Avalanche Problem Classification (AVAPRO)

AVAPRO analyses daily SNOWPACK PRO layer profiles and classifies five
avalanche problem types:

- **New snow** — recent precipitation with poor bonding
- **Wind slab** — wind-deposited slab over weak layer
- **Persistent weak layer (PWL)** — buried facets, depth hoar, or surface hoar
- **Wet snow** — free water in snowpack
- **Glide snow** — snow gliding on smooth ground (not applicable flat-field)

If AVAPRO is not installed, a heuristic fallback classifier based on SNOWPACK
PRO grain types, free water content, and fresh snow depth is used.

### Cluster Analysis

Following Reuter et al. (2023), seasons at the 2000 m reference elevation are
grouped into avalanche climate types using k-means clustering (k = 4) on:

1. Seasonal new-snow problem days
2. Seasonal persistent-weak-layer days
3. Wet-snow onset day-of-year (missing = DOY 180)
4. Total avalanche problem days

Variables are standardised (μ = 0, σ = 1) before clustering.  Clusters are
labelled so that **Type 1** has the highest PWL occurrence.

---

## Repository Structure

```
avalanche_climatology_at/
│
├── data/
│   ├── era5_raw/              # ERA5-Land annual NetCDF files
│   ├── spartacus/             # SPARTACUS annual NetCDF files
│   ├── snowpack_input/
│   │   ├── smet/              # Hourly meteorological forcing (SMET)
│   │   ├── sno/               # Initial snow profiles
│   │   └── ini/               # SNOWPACK configuration files
│   ├── snowpack_output/       # SNOWPACK PRO + SNO output
│   └── avapro_output/         # AVAPRO problem classification CSVs
│
├── scripts/
│   ├── __init__.py
│   ├── download_era5.py       # CDS API downloader
│   ├── download_spartacus.py  # GeoSphere Austria OGD downloader
│   ├── interpolate_points.py  # Bilinear interpolation + lapse-rate
│   ├── bias_correction.py     # SPARTACUS bias correction
│   ├── snowpack_writer.py     # SMET / SNO / INI writer
│   ├── run_snowpack.py        # Parallel SNOWPACK runner
│   ├── run_avapro.py          # AVAPRO wrapper + PRO heuristic fallback
│   ├── climatology.py         # Seasonal aggregation + k-means
│   └── plotting.py            # All figures
│
├── figures/                   # Generated PDF figures + CSV tables
├── logs/                      # Pipeline log files
├── config.yaml                # All user-configurable settings
├── requirements.txt           # Python dependencies
├── main.py                    # Pipeline orchestrator
└── README.md
```

---

## Installation

### System requirements

- Python 3.11
- SNOWPACK ≥ 3.6 (https://models.slf.ch/p/snowpack/)
- AVAPRO / AWSoM toolchain (contact LWD Austria / SLF)
- CDS API account (https://cds.climate.copernicus.eu/)

### Python environment

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### CDS API credentials

Create `~/.cdsapirc` with your Copernicus account credentials:

```ini
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API-KEY>
```

Replace `<UID>` and `<API-KEY>` with values from
https://cds.climate.copernicus.eu/user/login.

### Adjust config.yaml

Update the following fields before running:

```yaml
snowpack:
  binary: "/usr/local/bin/snowpack"   # path to SNOWPACK executable

avapro:
  binary: "/usr/local/bin/avapro"     # path to AVAPRO executable

simulation:
  analysis_start: "2000-10-01"        # adjust study period as needed
  analysis_end:   "2021-04-30"
```

---

## Execution

### Full pipeline

```bash
cd avalanche_climatology_at/
python main.py
```

### Skip downloads (data already on disk)

```bash
python main.py --skip-download
```

### Skip SNOWPACK (use existing PRO files)

```bash
python main.py --skip-download --skip-snowpack
```

### Skip AVAPRO (use existing classification CSVs)

```bash
python main.py --skip-download --skip-snowpack --skip-avapro
```

### Regenerate figures only

```bash
python main.py --only-plot
```

### Control parallelism

```bash
# Use 8 CPU cores for SNOWPACK
python main.py --skip-download --n-jobs 8
```

### Logging

```bash
python main.py --log-level DEBUG
```

The full log is always written to `logs/pipeline.log`.

---

## Outputs

### CSV tables (`figures/`)

| File                   | Content                                                 |
|------------------------|---------------------------------------------------------|
| `seasonal_stats.csv`   | Problem day counts and wet-onset DOY per region × elev × season |
| `cluster_labels.csv`   | Same as seasonal_stats + cluster assignment column       |

### Figures (`figures/`)

| File                               | Content                                        |
|------------------------------------|------------------------------------------------|
| `fig01_problem_frequency_bars.pdf` | Mean problem frequency per region × elevation  |
| `fig02_seasonal_distributions.pdf` | Box plots of seasonal problem counts           |
| `fig03_wet_snow_onset_elevation.pdf`| Wet-snow onset DOY vs. elevation (+ regression)|
| `fig04_cluster_map.pdf`            | Cluster assignment matrix (region × season)    |
| `fig05_cluster_profiles.pdf`       | Mean problem count per avalanche climate type  |
| `fig06_timeseries_by_cluster.pdf`  | Seasonal problem days coloured by cluster      |

---

## Extending the Pipeline

### Adding study regions

Edit `config.yaml` under the `regions:` key.  Each entry requires
`lat`, `lon`, `name`, `province`, and `lwd` fields.

### Changing elevation bands

Modify the `elevation_bands:` list in `config.yaml`.

### Changing the study period

Update `simulation.analysis_start` and `simulation.analysis_end`.
ERA5 and SPARTACUS downloads will automatically cover the new period
plus the spin-up year.

### Using a different number of clusters

Set `clustering.n_clusters` in `config.yaml`.

---

## References

Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., et al. (2021).
ERA5-Land: A state-of-the-art global reanalysis dataset for land applications.
*Earth System Science Data*, 13(9), 4349–4383.
https://doi.org/10.5194/essd-13-4349-2021

Hiebl, J. & Frei, C. (2016). Daily temperature grids for Austria since 1961.
*Theoretical and Applied Climatology*, 124(1–2), 161–177.
https://doi.org/10.1007/s00704-015-1411-4

Hiebl, J. & Frei, C. (2018). Daily precipitation grids for Austria since 1961.
*Theoretical and Applied Climatology*, 132(1–2), 327–345.
https://doi.org/10.1007/s00704-017-2093-x

Lehning, M., Bartelt, P., Brown, B., Fierz, C., & Satyawali, P. (2002).
A physical SNOWPACK model for the Swiss avalanche warning.
*Cold Regions Science and Technology*, 35(3), 147–167.
https://doi.org/10.1016/S0165-232X(02)00074-5

Reuter, B., Viallon-Galinier, L., Horton, S., van Herwijnen, A.,
Hagenmuller, P., Morin, S., & Schweizer, J. (2023).
Characterizing snow instability with avalanche problem types derived from
snow cover simulations. *Cold Regions Science and Technology*, 207, 103772.
https://doi.org/10.1016/j.coldregions.2022.103772

Schweizer, J., Mitterer, C., Reuter, B., & Techel, F. (2020).
Optimizing consistency of avalanche danger rating with a statistically-based
approach. *Cold Regions Science and Technology*, 175, 103030.
https://doi.org/10.1016/j.coldregions.2020.103030

---

## Licence

This code is released under the MIT Licence.
ERA5-Land data: © Copernicus Climate Change Service (C3S).
SPARTACUS data: © GeoSphere Austria, CC BY 4.0.
SNOWPACK / AVAPRO: © SLF / Austrian Avalanche Warning Services — separate licences apply.
