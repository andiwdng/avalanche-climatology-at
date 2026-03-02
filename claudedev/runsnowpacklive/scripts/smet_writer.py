# snowpack_steiermark/scripts/smet_writer.py
"""
Convert GeoSphere Austria station data to SNOWPACK SMET 1.1 format.

Supports per-station field configuration:
  - Base fields:  timestamp TA RH VW DW ISWR PSUM PSUM_PH HS
  - Optional ILWR (after ISWR) when station["ilwr"] is True
  - Optional TSS  (after HS)   when station["tss"] is True

TSS (snow surface temperature) from lawinen.at is in Celsius; converted to
Kelvin with the rule: if value < 200: value_K = value + 273.15
Valid range after conversion: 200–274 K (snow surface ≤ 0 °C).

TSG (ground surface temperature, SNOWPACK bottom boundary) is always generated
as a 273.15 K constant by MeteoIO — never read from sensor data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SMET_HEADER_TEMPLATE = """\
SMET 1.1 ASCII
[HEADER]
station_id       = {station_id}
station_name     = {station_name}
latitude         = {latitude}
longitude        = {longitude}
altitude         = {altitude}
nodata           = -999
tz               = 0
fields           = {fields}
units            = {units}
[DATA]
"""


class SmetWriter:
    """Converts GeoSphere Austria DataFrames to SNOWPACK SMET 1.1 files."""

    def __init__(self, config: dict, station: dict) -> None:
        self.config = config
        self._station = station
        self._build_field_lists()

    def _build_field_lists(self) -> None:
        """Build ordered field/unit lists based on station capabilities."""
        s = self._station
        fields = ["timestamp", "TA", "RH", "VW", "DW", "ISWR"]
        units  = ["ISO8601",   "K",  "-",  "m/s", "°", "W/m2"]
        if s.get("ilwr"):
            fields.append("ILWR"); units.append("W/m2")
        fields += ["PSUM", "PSUM_PH", "HS"]
        units  += ["kg/m2", "-", "m"]
        if s.get("tss"):
            fields.append("TSS"); units.append("K")
        self._fields = fields          # includes "timestamp"
        self._units = units
        self._numeric_cols = fields[1:]  # excludes "timestamp"

    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw lawinen.at DataFrame to SNOWPACK SMET format.

        Source data already uses SNOWPACK-native field names and units for most
        fields. ILWR and TSG need clipping/conversion; everything else just
        needs NaN → nodata fill.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with at minimum: [timestamp, TA, RH, VW, DW, ISWR, HS].
            May also contain ILWR and/or TSG (Celsius) depending on station.

        Returns
        -------
        pd.DataFrame
            Data ready for SMET output with all configured fields present.
        """
        out = pd.DataFrame()
        out["timestamp"] = df["timestamp"]

        def safe(series: pd.Series, fill: float = -999.0) -> pd.Series:
            return series.where(series.notna(), fill)

        out["TA"]   = safe(df["TA"].round(6)   if "TA"   in df.columns else pd.Series(np.nan, index=df.index))
        out["RH"]   = safe(df["RH"].round(6)   if "RH"   in df.columns else pd.Series(np.nan, index=df.index))
        out["VW"]   = safe(df["VW"].round(6)   if "VW"   in df.columns else pd.Series(np.nan, index=df.index))
        out["DW"]   = safe(df["DW"].round(6)   if "DW"   in df.columns else pd.Series(np.nan, index=df.index))

        iswr = df["ISWR"].copy() if "ISWR" in df.columns else pd.Series(np.nan, index=df.index)
        out["ISWR"] = safe(iswr.clip(lower=0).round(6))

        if self._station.get("ilwr"):
            ilwr = df["ILWR"].copy() if "ILWR" in df.columns else pd.Series(np.nan, index=df.index)
            # Atmospheric ILWR is physically always >= ~100 W/m²; values below 50
            # are sensor noise/errors — treat as nodata so generators supply values.
            ilwr = ilwr.where(ilwr.isna() | (ilwr >= 50), np.nan)
            out["ILWR"] = safe(ilwr.clip(lower=0).round(6))

        # PSUM/PSUM_PH: always nodata — SNOWPACK derives precipitation from HS
        # changes internally when ENFORCE_MEASURED_SNOW_HEIGHTS = TRUE.
        out["PSUM"]    = -999.0
        out["PSUM_PH"] = -999.0

        hs = df["HS"].copy() if "HS" in df.columns else pd.Series(np.nan, index=df.index)
        out["HS"] = safe(hs.clip(lower=0.0).round(4))

        if self._station.get("tss"):
            tss_raw = df["TSS"].copy() if "TSS" in df.columns else pd.Series(np.nan, index=df.index)
            # Convert Celsius → Kelvin for values that look like Celsius
            tss_raw = tss_raw.where(tss_raw.isna() | (tss_raw >= 200), tss_raw + 273.15)
            # Clip to plausible range; MeteoIO SOFT filter removes outliers
            tss_raw = tss_raw.clip(lower=210, upper=280)
            out["TSS"] = safe(tss_raw.round(4))

        return out

    def write(self, df: pd.DataFrame, smet_path: Path, mode: str = "w") -> None:
        """
        Write SMET file.

        Parameters
        ----------
        df : pd.DataFrame
            Converted data (output of convert_dataframe).
        smet_path : Path
            Destination file path.
        mode : str
            "w" writes full file including header; "a" appends data rows only.
        """
        smet_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "w":
            header = SMET_HEADER_TEMPLATE.format(
                station_id=self._station["snow_station"],
                station_name=self._station["name"],
                latitude=self._station["latitude"],
                longitude=self._station["longitude"],
                altitude=self._station["altitude"],
                fields=" ".join(self._fields),
                units=" ".join(self._units),
            )
            with open(smet_path, "w") as fh:
                fh.write(header)
                self._write_rows(fh, df)
        else:
            with open(smet_path, "a") as fh:
                self._write_rows(fh, df)

    def _write_rows(self, fh, df: pd.DataFrame) -> None:
        """Write DATA rows to an open file handle."""
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                ts_str = str(ts)
            vals = []
            for col in self._numeric_cols:
                v = row.get(col, -999.0)
                if pd.isna(v):
                    v = -999.0
                vals.append(f"{v:.6f}")
            fh.write(ts_str + " " + " ".join(vals) + "\n")

    def write_or_append(self, df: pd.DataFrame, smet_path: Path) -> None:
        """
        Merge new data with existing SMET, fill temporal gaps, and rewrite.

        Parameters
        ----------
        df : pd.DataFrame
            Raw GeoSphere data (pre-conversion).
        smet_path : Path
            SMET file path.
        """
        converted = self.convert_dataframe(df) if not df.empty else pd.DataFrame()

        if smet_path.exists():
            existing = self._read_smet_data(smet_path)
            if not existing.empty and not converted.empty:
                merged = (
                    pd.concat([existing, converted])
                    .drop_duplicates("timestamp", keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            elif not existing.empty:
                merged = existing
            else:
                merged = converted
        else:
            merged = converted

        if merged.empty:
            logger.info("No data to write to SMET.")
            return

        gap_filled = self._fill_gaps(merged)
        logger.info(
            "Writing SMET file: %s (%d measured rows, %d after gap-fill)",
            smet_path, len(merged), len(gap_filled),
        )
        self.write(gap_filled, smet_path, mode="w")

    def _read_smet_data(self, smet_path: Path) -> pd.DataFrame:
        """Read an existing SMET file back into a DataFrame (NaN for nodata)."""
        fields: list[str] = []
        rows: list[list] = []
        in_data = False
        with open(smet_path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("fields"):
                    _, _, rhs = s.partition("=")
                    fields = rhs.strip().split()
                    continue
                if s == "[DATA]":
                    in_data = True
                    continue
                if not in_data or not s:
                    continue
                parts = s.split()
                if len(parts) == len(fields):
                    rows.append(parts)
        if not fields or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=fields)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        for col in fields[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == -999.0, col] = float("nan")
        return df.sort_values("timestamp").reset_index(drop=True)

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex to a complete 10-minute grid and fill data gaps.

        Strategy per variable
        ---------------------
        TA, RH     : linear interpolation (up to 10 days)
        ISWR       : linear interpolation, clipped >= 0 (up to 10 days)
        ILWR       : linear interpolation, clipped >= 0 (up to 10 days)
        VW         : forward-fill, fallback 2.0 m/s
        DW         : forward-fill, fallback 180 deg
        HS         : forward-fill only (carry snow height, never interpolate)
        TSS        : linear interpolation (up to 2 days), clipped 200–274 K
        PSUM/PH    : always nodata (SNOWPACK derives from delta-HS)
        """
        if df.empty:
            return df
        idx = df.set_index("timestamp").sort_index()
        full_idx = pd.date_range(
            idx.index.min(), idx.index.max(), freq="10min", tz="UTC", name="timestamp"
        )
        idx = idx.reindex(full_idx)
        limit = 1440  # 10 days at 10-minute resolution

        for col in ["TA", "RH"]:
            if col in idx.columns:
                idx[col] = idx[col].interpolate(method="time", limit=limit)

        if "ISWR" in idx.columns:
            idx["ISWR"] = idx["ISWR"].interpolate(method="time", limit=limit).clip(lower=0.0)

        if "ILWR" in idx.columns:
            # Atmospheric ILWR is physically always >= ~100 W/m²; values below 50
            # are sensor errors (e.g. LOSE lango sensor reports 1-2 W/m²).
            # Blank them so MeteoIO ALLSKY_LW / CLEARSKY_LW generators supply values.
            idx["ILWR"] = idx["ILWR"].where(idx["ILWR"] >= 50, np.nan)
            idx["ILWR"] = idx["ILWR"].interpolate(method="time", limit=limit).clip(lower=0.0)

        if "VW" in idx.columns:
            idx["VW"] = idx["VW"].ffill(limit=limit).fillna(2.0)

        if "DW" in idx.columns:
            idx["DW"] = idx["DW"].ffill(limit=limit).fillna(180.0)

        if "HS" in idx.columns:
            idx["HS"] = idx["HS"].ffill(limit=limit).clip(lower=0.0).fillna(0.0)

        if "TSS" in idx.columns:
            tss_limit = 288  # 2 days at 10-minute resolution
            idx["TSS"] = idx["TSS"].interpolate(method="time", limit=tss_limit)

        idx["PSUM"] = float("nan")
        idx["PSUM_PH"] = float("nan")
        return idx.reset_index()


def write_smet(config: dict, station: dict, df: pd.DataFrame) -> Path:
    """
    Write or append GeoSphere data to the station SMET file.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    station : dict
        Station entry from config["stations"] list.
    df : pd.DataFrame
        New data from GeoSphereDownloader.

    Returns
    -------
    Path
        Path to the SMET file.
    """
    smet_path = Path(config["paths"]["data"]) / "smet" / f"{station['snow_station']}.smet"
    writer = SmetWriter(config, station)
    writer.write_or_append(df, smet_path)
    return smet_path
