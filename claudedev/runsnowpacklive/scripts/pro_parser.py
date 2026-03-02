# snowpack_steiermark/scripts/pro_parser.py
"""
Parse SNOWPACK PRO output files for timeseries extraction and GUI display.

SNOWPACK PRO format:
  [STATION_PARAMETERS] section — station metadata
  [HEADER] section — field code descriptors, e.g.:
      0500,Date
      0501,nElems,height [> 0: top, < 0: bottom of elem.] (cm)
      0503,nElems,element temperature (degC)
  Data starts immediately after the header (no [DATA] marker).
  Each timestep occupies multiple lines:
      0500,DD.MM.YYYY HH:MM:SS
      0501,nElems,h1,h2,...,hN    ← last value = total HS [cm]
      0503,nElems,t1,t2,...,tN    ← last value = surface temp [°C]
      ...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ProParser:
    """Parses a SNOWPACK PRO file and exposes timeseries and profile data."""

    def __init__(self, pro_path: Path) -> None:
        self.pro_path = pro_path
        self._header_meta: Optional[dict] = None
        self._timeseries: Optional[pd.DataFrame] = None

    def parse_header(self) -> dict:
        """Parse [STATION_PARAMETERS] section."""
        if self._header_meta is not None:
            return self._header_meta
        meta: dict = {}
        if not self.pro_path.exists():
            return meta
        in_station = False
        try:
            with open(self.pro_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line == "[STATION_PARAMETERS]":
                        in_station = True
                        continue
                    if line.startswith("[") and in_station:
                        break
                    if in_station and "=" in line:
                        k, _, v = line.partition("=")
                        meta[k.strip()] = v.strip()
        except Exception as exc:
            logger.warning("Could not parse PRO header: %s", exc)
        self._header_meta = meta
        return meta

    def parse_timeseries(self) -> pd.DataFrame:
        """
        Parse all timestep blocks and extract date, HS, TSS.

        The PRO data section starts after the [HEADER] block.
        Each timestep has lines:
          0500,DD.MM.YYYY HH:MM:SS      ← date
          0501,nElems,v1,...,vN         ← last vN = total HS [cm]
          0503,nElems,v1,...,vN         ← last vN = surface temp [°C]

        Returns
        -------
        pd.DataFrame
            Columns: [date, hs_cm, tss_c]. Missing values are NaN.
        """
        if self._timeseries is not None:
            return self._timeseries

        empty = pd.DataFrame(columns=["date", "hs_cm", "tss_c"])

        if not self.pro_path.exists() or self.pro_path.stat().st_size == 0:
            self._timeseries = empty
            return empty

        rows: list[dict] = []
        current_date = None
        current_hs = float("nan")
        current_tss = float("nan")

        try:
            past_header = False
            with open(self.pro_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue

                    # The header section ends when we see the first data line
                    # (a line starting with "0500," followed by a date, not "Date")
                    if line.startswith("["):
                        past_header = False
                        continue
                    if not past_header:
                        # Lines in HEADER section start with 4-digit codes like "0500,Date"
                        # Data lines start with "0500,DD.MM.YYYY"
                        if line.startswith("0500,") and not line.startswith("0500,Date"):
                            past_header = True
                        else:
                            continue

                    if not line.startswith("0"):
                        continue

                    parts = line.split(",")
                    code = parts[0].strip()

                    if code == "0500":
                        # Save previous timestep
                        if current_date is not None:
                            rows.append({
                                "date": current_date,
                                "hs_cm": current_hs,
                                "tss_c": current_tss,
                            })
                        # Parse new date: "DD.MM.YYYY HH:MM:SS"
                        date_str = parts[1].strip() if len(parts) > 1 else ""
                        try:
                            current_date = pd.to_datetime(date_str, format="%d.%m.%Y %H:%M:%S")
                        except Exception:
                            current_date = None
                        current_hs = float("nan")
                        current_tss = float("nan")

                    elif code == "0501" and current_date is not None:
                        # 0501,nElems,v1,...,vN — last value = total HS [cm]
                        vals = parts[2:]  # skip code and nElems
                        if vals:
                            try:
                                current_hs = float(vals[-1].strip())
                            except ValueError:
                                current_hs = float("nan")

                    elif code == "0503" and current_date is not None:
                        # 0503,nElems,v1,...,vN — last value = surface temp [°C]
                        vals = parts[2:]  # skip code and nElems
                        if vals:
                            try:
                                current_tss = float(vals[-1].strip())
                            except ValueError:
                                current_tss = float("nan")

            # Save final timestep
            if current_date is not None:
                rows.append({
                    "date": current_date,
                    "hs_cm": current_hs,
                    "tss_c": current_tss,
                })

        except Exception as exc:
            logger.error("Failed to parse PRO file: %s", exc)

        if rows:
            df = pd.DataFrame(rows)
            # Derive new snow (hn_cm) from positive HS increments
            df["hn_cm"] = df["hs_cm"].diff().clip(lower=0.0).fillna(0.0)
        else:
            logger.warning("PRO timeseries empty, returning empty classification.")
            df = pd.DataFrame(columns=["date", "hs_cm", "tss_c", "hn_cm"])

        self._timeseries = df
        return df

    def get_latest_profile(self) -> dict:
        """Return the most recent profile as a summary dict."""
        df = self.parse_timeseries()
        if df.empty:
            return {"date": None, "hs_cm": None, "tss_c": None}
        last = df.iloc[-1]

        def _round(val, n):
            return round(float(val), n) if pd.notna(val) else None

        return {
            "date": last["date"].isoformat() if hasattr(last["date"], "isoformat") else str(last["date"]),
            "hs_cm": _round(last["hs_cm"], 1),
            "tss_c": _round(last["tss_c"], 2),
        }

    def get_timeseries_dict(self) -> list[dict]:
        """Return timeseries as a list of dicts for JSON serialisation."""
        df = self.parse_timeseries()
        if df.empty:
            return []
        records = []
        for _, row in df.iterrows():
            date_str = (
                row["date"].isoformat()
                if hasattr(row["date"], "isoformat")
                else str(row["date"])
            )
            hs = round(float(row["hs_cm"]), 1) if pd.notna(row["hs_cm"]) else None
            tss = round(float(row["tss_c"]), 2) if pd.notna(row["tss_c"]) else None
            records.append({"date": date_str, "hs_cm": hs, "tss_c": tss})
        return records


def parse_pro(pro_path: Path) -> ProParser:
    """Create and return a ProParser for the given PRO file."""
    return ProParser(pro_path)
