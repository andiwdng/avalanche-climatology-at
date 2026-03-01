# snowpack_steiermark/scripts/smet_writer.py
"""
Convert GeoSphere Austria hourly data to SNOWPACK SMET 1.1 format.
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
fields           = timestamp TA RH VW DW ISWR PSUM PSUM_PH HS
units            = ISO8601 K - m/s ° W/m2 kg/m2 - m
[DATA]
"""

SMET_FIELDS = "timestamp TA RH VW DW ISWR PSUM PSUM_PH"


class SmetWriter:
    """Converts GeoSphere Austria DataFrames to SNOWPACK SMET 1.1 files."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.station_cfg = config["station"]

    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pass-through conversion from lawinen.at SMET DataFrame to SNOWPACK format.

        Source data from lawinen.at already uses SNOWPACK-native field names and units:
        TA [K], RH [0-1], VW [m/s], DW [°], ISWR [W/m²], HS [m].
        No unit conversion is needed — just fill NaN with nodata sentinel.

        Parameters
        ----------
        df : pd.DataFrame
            Hourly data with columns [timestamp, TA, RH, VW, DW, ISWR, HS].

        Returns
        -------
        pd.DataFrame
            Data with columns [timestamp, TA, RH, VW, DW, ISWR, PSUM, PSUM_PH, HS].
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

        # PSUM/PSUM_PH: always nodata — SNOWPACK derives precipitation from HS
        # changes internally when ENFORCE_MEASURED_SNOW_HEIGHTS = TRUE.
        out["PSUM"]    = -999.0
        out["PSUM_PH"] = -999.0

        hs = df["HS"].copy() if "HS" in df.columns else pd.Series(np.nan, index=df.index)
        out["HS"] = safe(hs.clip(lower=0.0).round(4))

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
        numeric_cols = ["TA", "RH", "VW", "DW", "ISWR", "PSUM", "PSUM_PH", "HS"]

        if mode == "w":
            header = SMET_HEADER_TEMPLATE.format(
                station_id=self.station_cfg["id"],
                station_name=self.station_cfg["name"],
                latitude=self.station_cfg["latitude"],
                longitude=self.station_cfg["longitude"],
                altitude=self.station_cfg["altitude"],
            )
            with open(smet_path, "w") as fh:
                fh.write(header)
                self._write_rows(fh, df, numeric_cols)
        else:
            with open(smet_path, "a") as fh:
                self._write_rows(fh, df, numeric_cols)

    def _write_rows(self, fh, df: pd.DataFrame, numeric_cols: list[str]) -> None:
        """Write DATA rows to an open file handle."""
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                ts_str = str(ts)
            vals = []
            for col in numeric_cols:
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
        ISWR       : linear interpolation, clipped >= 0
        VW         : forward-fill, fallback 2.0 m/s
        DW         : forward-fill, fallback 180 deg
        HS (Fix 3) : forward-fill only (carry snow height, never interpolate)
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

        if "VW" in idx.columns:
            idx["VW"] = idx["VW"].ffill(limit=limit).fillna(2.0)

        if "DW" in idx.columns:
            idx["DW"] = idx["DW"].ffill(limit=limit).fillna(180.0)

        if "HS" in idx.columns:
            idx["HS"] = idx["HS"].ffill(limit=limit).clip(lower=0.0).fillna(0.0)

        idx["PSUM"] = float("nan")
        idx["PSUM_PH"] = float("nan")
        return idx.reset_index()


def write_smet(config: dict, df: pd.DataFrame) -> Path:
    """
    Write or append GeoSphere data to the station SMET file.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    df : pd.DataFrame
        New hourly data from GeoSphereDownloader.

    Returns
    -------
    Path
        Path to the SMET file.
    """
    smet_path = Path(config["paths"]["data"]) / "smet" / "TAMI2.smet"
    writer = SmetWriter(config)
    writer.write_or_append(df, smet_path)
    return smet_path
