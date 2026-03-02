# snowpack_steiermark/scripts/download_geosphere.py
"""
Download station SMET data from lawinen.at for a given station pair.

Each station is split into two sub-stations on lawinen.at/smet/stm/:
  {ID}1 — Windmessstation:  TA RH DW VW DW_MAX VW_MAX
  {ID}2 — Schneemessstation: TA RH DW VW ISWR HS [TSG] [ILWR] ...

Strategy:
  - Snow station is the primary source (has ISWR and HS).
  - VW/DW from wind station (higher elevation, better wind exposure) are used when
    snow station wind is missing.
  - PSUM is derived from positive ΔHS (snow height increase) as a proxy.
  - Data is 10-minute resolution; MeteoIO handles resampling to simulation step.

Field renames applied after fetching:
  - ISWR2  → ISWR  (LOSE2, VEIT2; only when ISWR not already present)
  - lango  → ILWR  (LOSE2, VEIT2)
  - tg     → TSG   (LOSE2, VEIT2; raw Celsius kept as-is; SmetWriter converts to K)

Unused columns dropped: ISWRu, langu, slope1* etc.

URLs:
  winter/{ID}*.smet.gz — full season from ~September (updated weekly)
  woche/{ID}*.smet.gz  — last 7 days (updated ~hourly)

nodata value on the source: -777  →  converted to NaN internally.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://lawinen.at/smet/stm"
NODATA = -777.0

# Columns from lawinen.at that are not used by SNOWPACK and should be dropped
_DROP_COLS = {"ISWRu", "langu", "DW_MAX", "VW_MAX"}
_DROP_PREFIX = "slope"  # drop any column starting with "slope"


class GeoSphereDownloader:
    """Downloads and merges wind+snow SMET data from lawinen.at for a station pair."""

    def __init__(self, config: dict, station: dict) -> None:
        self.config = config
        self._station = station
        station_id = station["id"].lower()
        state_dir = Path(config["paths"]["state"])
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = state_dir / f"{station_id}_download.json"
        data_dir = Path(config["paths"]["data"])
        self.raw_csv = data_dir / "smet" / f"{station_id}_raw.csv"

        # Migrate old TAMI state file on first run
        if station["id"].upper() == "TAMI" and not self.state_path.exists():
            old_state = state_dir / "last_download.json"
            if old_state.exists():
                try:
                    shutil.copy2(old_state, self.state_path)
                    logger.info("Migrated state: %s → %s", old_state, self.state_path)
                except Exception as exc:
                    logger.warning("Could not migrate state file: %s", exc)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path) as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Could not read state file: %s", exc)
            return {}

    def save_state(self, state: dict) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.state_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(state, fh, indent=2, default=str)
            os.replace(tmp_path, self.state_path)
        except Exception as exc:
            logger.error("Could not save state: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def get_season_start(self) -> datetime:
        now = datetime.now(tz=timezone.utc)
        sm = int(self.config["simulation"]["season_start_month"])
        sd = int(self.config["simulation"]["season_start_day"])
        if now.month >= sm:
            return datetime(now.year, sm, sd, 0, 0, tzinfo=timezone.utc)
        return datetime(now.year - 1, sm, sd, 0, 0, tzinfo=timezone.utc)

    # ------------------------------------------------------------------
    # SMET fetching
    # ------------------------------------------------------------------

    def _fetch_smet_gz(self, url: str) -> pd.DataFrame:
        """
        Download and parse a gzip-compressed SMET 1.2 file from lawinen.at.

        Returns
        -------
        pd.DataFrame
            Parsed data with timestamp as UTC-aware datetime index,
            other columns as floats with NaN for nodata.
        """
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Could not fetch %s: %s", url, exc)
            return pd.DataFrame()

        try:
            raw = gzip.decompress(resp.content).decode("utf-8")
        except Exception as exc:
            logger.error("Could not decompress %s: %s", url, exc)
            return pd.DataFrame()

        fields: list[str] = []
        rows: list[list] = []
        in_data = False

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line == "[DATA]":
                in_data = True
                continue
            if line.startswith("fields"):
                _, _, rhs = line.partition("=")
                fields = rhs.strip().split()
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) != len(fields):
                continue
            rows.append(parts)

        if not fields or not rows:
            logger.warning("Empty or unparseable SMET from %s", url)
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=fields)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        for col in fields[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == NODATA, col] = float("nan")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Field renames / cleanup
    # ------------------------------------------------------------------

    def _apply_field_renames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename non-standard lawinen.at fields to SNOWPACK names and drop
        columns that are not used by SNOWPACK.

        Renames:
          ISWR2  → ISWR   (only when ISWR not already present)
          lango  → ILWR
          tg     → TSG    (stays in Celsius; SmetWriter converts to K)

        Drops: ISWRu, langu, DW_MAX, VW_MAX, slope1* …
        """
        if df.empty:
            return df

        # Rename ISWR2 → ISWR only if ISWR is absent
        if "ISWR2" in df.columns and "ISWR" not in df.columns:
            df = df.rename(columns={"ISWR2": "ISWR"})
        elif "ISWR2" in df.columns:
            df = df.drop(columns=["ISWR2"])

        if "lango" in df.columns:
            df = df.rename(columns={"lango": "ILWR"})

        if "tg" in df.columns:
            # VEIT2 has both a pre-labelled TSG column and tg — drop the duplicate
            if "TSG" in df.columns:
                df = df.drop(columns=["TSG"])
            df = df.rename(columns={"tg": "TSG"})

        # Drop unused columns
        drop = [c for c in df.columns
                if c in _DROP_COLS or c.startswith(_DROP_PREFIX)]
        if drop:
            df = df.drop(columns=drop)

        return df

    # ------------------------------------------------------------------
    # Merge & resample
    # ------------------------------------------------------------------

    def _merge_stations(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge wind station (df1) and snow station (df2) on timestamp.

        Snow station is primary; wind station VW/DW fill gaps in snow station.
        """
        if df1.empty and df2.empty:
            return pd.DataFrame()
        if df1.empty:
            return df2
        if df2.empty:
            return df1

        df1 = df1.set_index("timestamp")
        df2 = df2.set_index("timestamp")

        # Drop duplicate timestamps (can occur in lawinen.at data)
        df1 = df1[~df1.index.duplicated(keep="last")]
        df2 = df2[~df2.index.duplicated(keep="last")]

        merged = df2.copy()

        for col in ["VW", "DW"]:
            col1 = df1[col] if col in df1.columns else None
            col2 = merged.get(col)
            if col1 is not None and col2 is not None:
                col1_aligned = col1.reindex(merged.index, method="nearest", tolerance="5min")
                merged[col] = merged[col].fillna(col1_aligned)
            elif col1 is not None:
                merged[col] = col1.reindex(merged.index, method="nearest", tolerance="5min")

        merged = merged.reset_index()
        return merged

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(self) -> pd.DataFrame:
        """
        Download wind+snow station pair from lawinen.at and return merged
        10-minute DataFrame.

        Uses winter/ for full season on first run, woche/ for incremental
        updates (last 7 days) when state exists.

        Returns
        -------
        pd.DataFrame
            10-minute merged data with SNOWPACK-native field names.
        """
        state = self.load_state()
        now = datetime.now(tz=timezone.utc)
        station = self._station

        use_winter = not state.get("last_download")
        source = "winter" if use_winter else "woche"
        logger.info("Fetching %s+%s from lawinen.at/smet/stm/%s/",
                    station["wind_station"], station["snow_station"], source)

        url1 = f"{BASE_URL}/{source}/{station['wind_station']}.smet.gz"
        url2 = f"{BASE_URL}/{source}/{station['snow_station']}.smet.gz"

        df1 = self._fetch_smet_gz(url1)
        df2 = self._fetch_smet_gz(url2)

        if df1.empty and df2.empty:
            logger.error("Both %s and %s returned empty data.",
                         station["wind_station"], station["snow_station"])
            return pd.DataFrame()

        logger.info("%s: %d rows, %s: %d rows",
                    station["wind_station"], len(df1),
                    station["snow_station"], len(df2))

        # Apply field renames to snow station data
        df2 = self._apply_field_renames(df2)

        # Filter to only new data when doing incremental update
        if state.get("last_download") and not use_winter:
            cutoff = datetime.fromisoformat(state["last_download"])
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            if not df1.empty:
                df1 = df1[df1["timestamp"] > cutoff]
            if not df2.empty:
                df2 = df2[df2["timestamp"] > cutoff]
            if df1.empty and df2.empty:
                logger.info("No new data since %s", cutoff)
                return pd.DataFrame()

        # Filter to season start
        season_start = self.get_season_start()
        if not df1.empty:
            df1 = df1[df1["timestamp"] >= season_start]
        if not df2.empty:
            df2 = df2[df2["timestamp"] >= season_start]

        merged = self._merge_stations(df1, df2)
        if merged.empty:
            return pd.DataFrame()

        # Mark last download timestamp
        last_ts = merged["timestamp"].max()
        state["last_download"] = last_ts.isoformat() if pd.notna(last_ts) else now.isoformat()
        self.save_state(state)

        # Persist raw CSV
        self._append_raw_csv(merged)

        logger.info("Prepared %d 10-min records (%s → %s)",
                    len(merged),
                    merged["timestamp"].min().date(),
                    merged["timestamp"].max().date())
        return merged

    def _append_raw_csv(self, df: pd.DataFrame) -> None:
        self.raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_write = df.copy()
        df_write["timestamp"] = df_write["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        if self.raw_csv.exists():
            try:
                existing = pd.read_csv(self.raw_csv)
                existing_ts = set(existing["timestamp"].astype(str))
                df_write = df_write[~df_write["timestamp"].astype(str).isin(existing_ts)]
                if df_write.empty:
                    return
                df_write.to_csv(self.raw_csv, mode="a", header=False, index=False)
            except Exception as exc:
                logger.warning("Could not append to raw CSV: %s", exc)
                df_write.to_csv(self.raw_csv, index=False)
        else:
            df_write.to_csv(self.raw_csv, index=False)


def download_station_data(config: dict, station: dict) -> pd.DataFrame:
    """
    Download wind+snow station pair from lawinen.at and return merged
    10-minute DataFrame.

    Parameters
    ----------
    config : dict
        Parsed config.yaml content.
    station : dict
        Station entry from config["stations"] list.

    Returns
    -------
    pd.DataFrame
        10-minute merged data, or empty DataFrame on failure.
    """
    downloader = GeoSphereDownloader(config, station)
    return downloader.download()
