"""
climatology.py
==============
Seasonal aggregation of AVAPRO avalanche problem classifications and
k-means cluster analysis following the methodology of Reuter et al. (2023).

Seasonal aggregation
--------------------
An avalanche season spans October–April (configurable).  For each
(region, elevation_band, season) combination the following metrics
are computed:

- ``new_snow_days``     : number of days classified as new-snow problem
- ``wind_slab_days``    : number of days classified as wind-slab problem
- ``pwl_days``          : number of days classified as persistent-weak-layer problem
- ``wet_snow_days``     : number of days classified as wet-snow problem
- ``glide_snow_days``   : number of days classified as glide-snow problem
- ``total_problem_days``: total days with at least one problem active
- ``wet_snow_onset_doy``: day-of-year of first wet-snow problem occurrence
                          (NaN if no wet snow in season → imputed with 180)

K-means cluster analysis
------------------------
Following Reuter et al. (2023), seasons are grouped into avalanche
climate types using k-means clustering on:
    1. seasonal new-snow days
    2. seasonal persistent-weak-layer days
    3. wet-snow onset DOY  (missing seasons imputed to DOY 180)
    4. total problem days

Variables are standardised (zero mean, unit variance) before clustering.
The number of clusters k = 4 (configurable in config.yaml).
Clustering is applied at the reference elevation (default 2000 m).

Multiple random initialisations are used to mitigate local optima.
Final cluster labels are re-ordered so that Cluster 1 has the highest
persistent-weak-layer occurrence (aligned with Reuter et al. labelling
convention).

Output
------
- ``seasonal_stats.csv``  : per-season aggregated statistics
- ``cluster_labels.csv``  : cluster assignment per (region, season)

Reference
---------
Reuter, B., Viallon-Galinier, L., Horton, S., van Herwijnen, A.,
    Hagenmuller, P., Morin, S., & Schweizer, J. (2023).
    Characterizing snow instability with avalanche problem types derived
    from snow cover simulations. Cold Regions Science and Technology,
    207, 103772. https://doi.org/10.1016/j.coldregions.2022.103772
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# DOY imputed for seasons without a wet-snow problem
_MISSING_WET_ONSET_DOY: float = 180.0

# Minimum number of snow-covered days per season for inclusion in clustering
_MIN_SEASON_DAYS: int = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_climatology(
    config: dict,
    problems: dict[str, dict[int, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Aggregate AVAPRO daily problem classifications into seasonal statistics.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    problems : dict
        ``problems[region][elev] = DataFrame`` as returned by
        :func:`scripts.run_avapro.run_avapro_all`.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        region, elevation_m, season, new_snow_days, wind_slab_days,
        pwl_days, wet_snow_days, glide_snow_days, total_problem_days,
        wet_snow_onset_doy.
        Indexed by integer RangeIndex.
    """
    sim_cfg = config["simulation"]
    season_start_month = int(sim_cfg.get("season_start_month", 10))
    season_end_month = int(sim_cfg.get("season_end_month", 4))

    rows: list[dict] = []

    for region_key, elev_dict in problems.items():
        for elev_m, df_problems in elev_dict.items():
            if df_problems.empty:
                logger.warning(
                    "Empty problem DataFrame for %s @ %d m — skipping.", region_key, elev_m
                )
                continue

            # Split into seasons and aggregate
            season_groups = _split_into_seasons(
                df_problems,
                season_start_month=season_start_month,
                season_end_month=season_end_month,
            )

            for season_label, df_season in season_groups.items():
                if len(df_season) < _MIN_SEASON_DAYS:
                    logger.debug(
                        "Season %s too short for %s @ %d m (%d days < %d) — skipping.",
                        season_label,
                        region_key,
                        elev_m,
                        len(df_season),
                        _MIN_SEASON_DAYS,
                    )
                    continue

                stats = _aggregate_season(df_season, season_label)
                stats["region"] = region_key
                stats["elevation_m"] = int(elev_m)
                stats["season"] = season_label
                rows.append(stats)

    df_clim = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "region",
        "elevation_m",
        "season",
        "new_snow_days",
        "wind_slab_days",
        "pwl_days",
        "wet_snow_days",
        "glide_snow_days",
        "total_problem_days",
        "wet_snow_onset_doy",
    ]
    df_clim = df_clim[[c for c in col_order if c in df_clim.columns]]
    df_clim = df_clim.reset_index(drop=True)

    logger.info(
        "Climatology aggregated: %d (region × elevation × season) records.", len(df_clim)
    )
    return df_clim


def perform_clustering(
    config: dict,
    df_clim: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign seasonal avalanche climate type clusters using k-means.

    Clustering is performed on the reference elevation (default 2000 m)
    and cluster labels are then propagated to all elevation bands for
    consistency.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    df_clim : pd.DataFrame
        Output of :func:`compute_climatology`.

    Returns
    -------
    pd.DataFrame
        Copy of *df_clim* with an additional ``cluster`` column
        (integer, 1-based following Reuter et al. convention).
    """
    clust_cfg = config["clustering"]
    k = int(clust_cfg["n_clusters"])
    random_state = int(clust_cfg["random_state"])
    n_init = int(clust_cfg["n_init"])
    cluster_vars = clust_cfg["variables"]
    reference_elevation = int(clust_cfg["reference_elevation"])

    # Impute missing wet-snow onset DOY
    df_work = df_clim.copy()
    df_work["wet_snow_onset_doy"] = df_work["wet_snow_onset_doy"].fillna(_MISSING_WET_ONSET_DOY)

    # Filter to reference elevation for training
    df_ref = df_work[df_work["elevation_m"] == reference_elevation].copy()

    if df_ref.empty:
        raise ValueError(
            f"No data at reference elevation {reference_elevation} m. "
            "Check 'clustering.reference_elevation' in config.yaml."
        )

    # Verify all clustering variables are present
    missing_vars = [v for v in cluster_vars if v not in df_ref.columns]
    if missing_vars:
        raise KeyError(f"Clustering variables not found in climatology: {missing_vars}")

    X = df_ref[cluster_vars].values.astype(float)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means with multiple initialisations
    kmeans = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd",
    )
    raw_labels = kmeans.fit_predict(X_scaled)

    # Re-order clusters: Cluster 1 = highest PWL occurrence (Reuter convention)
    label_order = _order_clusters_by_pwl(raw_labels, df_ref["pwl_days"].values, k)
    relabelled = np.array([label_order[lb] for lb in raw_labels]) + 1  # 1-based

    df_ref = df_ref.copy()
    df_ref["cluster"] = relabelled

    # Log cluster statistics
    for cl in range(1, k + 1):
        cl_mask = df_ref["cluster"] == cl
        n_seasons = cl_mask.sum()
        logger.info(
            "Cluster %d: %d seasons  |  "
            "NS=%.1f  WS=%.1f  PWL=%.1f  WetOnset=%.0f  Total=%.1f",
            cl,
            n_seasons,
            df_ref.loc[cl_mask, "new_snow_days"].mean(),
            df_ref.loc[cl_mask, "wet_snow_days"].mean(),
            df_ref.loc[cl_mask, "pwl_days"].mean(),
            df_ref.loc[cl_mask, "wet_snow_onset_doy"].mean(),
            df_ref.loc[cl_mask, "total_problem_days"].mean(),
        )

    # Merge cluster labels back to all elevations via (region, season)
    cluster_map = df_ref.set_index(["region", "season"])["cluster"]
    df_result = df_work.copy()
    df_result["cluster"] = df_result.set_index(["region", "season"]).index.map(cluster_map)

    return df_result


def save_climatology_outputs(
    config: dict,
    df_clim: pd.DataFrame,
    df_clusters: pd.DataFrame,
) -> None:
    """
    Save seasonal statistics and cluster assignments to CSV files.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    df_clim : pd.DataFrame
        Seasonal statistics from :func:`compute_climatology`.
    df_clusters : pd.DataFrame
        Clustered seasonal statistics from :func:`perform_clustering`.
    """
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats_path = figures_dir / "seasonal_stats.csv"
    df_clim.to_csv(stats_path, index=False, float_format="%.2f")
    logger.info("Seasonal statistics saved: %s", stats_path)

    clusters_path = figures_dir / "cluster_labels.csv"
    df_clusters.to_csv(clusters_path, index=False, float_format="%.2f")
    logger.info("Cluster labels saved: %s", clusters_path)


# ---------------------------------------------------------------------------
# Season splitting
# ---------------------------------------------------------------------------
def _split_into_seasons(
    df: pd.DataFrame,
    season_start_month: int = 10,
    season_end_month: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Partition a daily problem DataFrame into hydrological seasons.

    A season labelled ``'YYYY/YYYY+1'`` begins on the first day of
    *season_start_month* in calendar year YYYY and ends on the last day
    of *season_end_month* in calendar year YYYY+1.

    Parameters
    ----------
    df : pd.DataFrame
        Daily boolean DataFrame with DatetimeIndex.
    season_start_month : int
        Month number of season start (default 10 = October).
    season_end_month : int
        Month number of season end in the following calendar year
        (default 4 = April).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{season_label: subset_DataFrame}``.
    """
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)

    # Assign each date to a season start year
    def _season_year(dt: pd.Timestamp) -> int:
        """Return the calendar year in which this season started."""
        if dt.month >= season_start_month:
            return dt.year
        else:
            return dt.year - 1

    df["_season_year"] = [_season_year(ts) for ts in df.index]

    seasons: dict[str, pd.DataFrame] = {}
    for year, group in df.groupby("_season_year"):
        label = f"{year}/{year + 1}"
        season_data = group.drop(columns=["_season_year"])
        seasons[label] = season_data

    return seasons


# ---------------------------------------------------------------------------
# Per-season aggregation
# ---------------------------------------------------------------------------
def _aggregate_season(
    df_season: pd.DataFrame,
    season_label: str,
) -> dict:
    """
    Compute avalanche problem metrics for a single season.

    Parameters
    ----------
    df_season : pd.DataFrame
        Boolean daily problem DataFrame for one season.
    season_label : str
        Season identifier string (e.g. ``'2010/2011'``).

    Returns
    -------
    dict
        Keys: new_snow_days, wind_slab_days, pwl_days, wet_snow_days,
        glide_snow_days, total_problem_days, wet_snow_onset_doy.
    """
    stats: dict = {}

    problem_cols = {
        "new_snow_days": "new_snow",
        "wind_slab_days": "wind_slab",
        "pwl_days": "persistent_wl",
        "wet_snow_days": "wet_snow",
        "glide_snow_days": "glide_snow",
    }

    for stat_col, prob_col in problem_cols.items():
        if prob_col in df_season.columns:
            stats[stat_col] = int(df_season[prob_col].sum())
        else:
            stats[stat_col] = 0

    # Total days with at least one problem
    active_cols = [c for c in df_season.columns if c in problem_cols.values()]
    if active_cols:
        stats["total_problem_days"] = int(df_season[active_cols].any(axis=1).sum())
    else:
        stats["total_problem_days"] = 0

    # Wet-snow onset DOY
    if "wet_snow" in df_season.columns:
        wet_days = df_season.index[df_season["wet_snow"] == True]
        if len(wet_days) > 0:
            first_wet = pd.Timestamp(wet_days[0])
            stats["wet_snow_onset_doy"] = float(first_wet.day_of_year)
        else:
            stats["wet_snow_onset_doy"] = float("nan")
    else:
        stats["wet_snow_onset_doy"] = float("nan")

    return stats


# ---------------------------------------------------------------------------
# Cluster ordering
# ---------------------------------------------------------------------------
def _order_clusters_by_pwl(
    raw_labels: np.ndarray,
    pwl_counts: np.ndarray,
    k: int,
) -> dict[int, int]:
    """
    Re-order cluster labels so that Cluster 0 has the highest mean
    persistent-weak-layer count (descending order).

    Parameters
    ----------
    raw_labels : np.ndarray
        Raw k-means cluster assignments (0-based).
    pwl_counts : np.ndarray
        Seasonal PWL day counts.
    k : int
        Number of clusters.

    Returns
    -------
    dict[int, int]
        Mapping from raw label → new label (both 0-based).
    """
    mean_pwl = {
        cl: pwl_counts[raw_labels == cl].mean()
        for cl in range(k)
    }
    # Sort clusters by descending mean PWL
    sorted_clusters = sorted(mean_pwl, key=mean_pwl.get, reverse=True)
    return {old: new for new, old in enumerate(sorted_clusters)}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Demonstrate with synthetic data
    import datetime
    idx = pd.date_range("2010-10-01", "2011-04-30", freq="D")
    rng = np.random.default_rng(42)
    synthetic_df = pd.DataFrame(
        {
            "new_snow": rng.random(len(idx)) > 0.7,
            "wind_slab": rng.random(len(idx)) > 0.8,
            "persistent_wl": rng.random(len(idx)) > 0.85,
            "wet_snow": rng.random(len(idx)) > 0.75,
            "glide_snow": np.zeros(len(idx), dtype=bool),
        },
        index=idx,
    )
    stats = _aggregate_season(synthetic_df, "2010/2011")
    print("Sample season stats:", stats)
