"""
plotting.py
===========
Generate publication-quality figures for the Austrian avalanche climatology
analysis following the style conventions of Reuter et al. (2023).

Figures produced
----------------
1. ``fig01_problem_frequency_bars.pdf``
   Bar charts of avalanche problem frequency (fraction of season days)
   per region and elevation band.

2. ``fig02_seasonal_distributions.pdf``
   Box plots of seasonal problem day counts for each avalanche problem type,
   stratified by elevation band.

3. ``fig03_wet_snow_onset_elevation.pdf``
   Scatter plot of wet-snow onset DOY vs. elevation for each region,
   with linear regression per region and a climatological mean.

4. ``fig04_cluster_map.pdf``
   Panel showing cluster assignment per region and season.  Regions
   are sorted geographically (W→E) along the x-axis; seasons along y.
   Colour-coded by cluster label.

5. ``fig05_cluster_profiles.pdf``
   Mean avalanche problem frequency profiles for each cluster,
   displayed as stacked bar charts or radar plots.

6. ``fig06_timeseries_by_cluster.pdf``
   Time-series of total problem days per season, coloured by cluster
   assignment for the reference elevation.

Design notes
------------
- All figures use the ``viridis`` / ``tab10`` / ``Set2`` colormaps as
  configured in config.yaml.
- Figure size follows Nature journal guidelines (single column: 89 mm,
  double column: 183 mm) converted to inches.
- Font sizes: 8 pt body, 9 pt axes labels, 10 pt titles.
- Tick marks inward, axes spine visible (box=False).
- Colour-blind-friendly problem-type colours.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Problem-type colour palette (colour-blind-friendly)
# ---------------------------------------------------------------------------
PROBLEM_COLORS: dict[str, str] = {
    "new_snow_days": "#2196F3",       # blue
    "wind_slab_days": "#9C27B0",      # purple
    "pwl_days": "#F44336",            # red
    "wet_snow_days": "#FF9800",       # orange
    "glide_snow_days": "#4CAF50",     # green
}

PROBLEM_LABELS: dict[str, str] = {
    "new_snow_days": "New snow",
    "wind_slab_days": "Wind slab",
    "pwl_days": "Pers. weak layer",
    "wet_snow_days": "Wet snow",
    "glide_snow_days": "Glide snow",
}

# Cluster colours (tab10 first 4)
CLUSTER_COLORS: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Elevation band line styles
ELEVATION_LINESTYLES: dict[int, str] = {
    1500: "--",
    2000: "-",
    2500: "-.",
    3000: ":",
}

# Figure sizes in inches (Nature single / double column)
_SINGLE_COL: float = 3.5    # 89 mm
_DOUBLE_COL: float = 7.2    # 183 mm
_ROW_HEIGHT: float = 2.2

# Font sizes
_BODY_FS: int = 8
_LABEL_FS: int = 9
_TITLE_FS: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_all_figures(
    config: dict,
    df_clim: pd.DataFrame,
    df_clusters: pd.DataFrame,
) -> None:
    """
    Generate all six climatology figures and save them to the figures directory.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    df_clim : pd.DataFrame
        Seasonal statistics from :func:`scripts.climatology.compute_climatology`.
    df_clusters : pd.DataFrame
        Clustered statistics from :func:`scripts.climatology.perform_clustering`.
    """
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = config.get("plotting", {})
    dpi = int(plot_cfg.get("dpi", 150))
    fmt = plot_cfg.get("figure_format", "pdf")

    plt.rcParams.update(_rcparams())

    fig_funcs = [
        ("fig01_problem_frequency_bars", _fig_problem_frequency, (config, df_clim)),
        ("fig02_seasonal_distributions", _fig_seasonal_distributions, (config, df_clim)),
        ("fig03_wet_snow_onset_elevation", _fig_wet_snow_onset, (config, df_clim)),
        ("fig04_cluster_map", _fig_cluster_map, (config, df_clusters)),
        ("fig05_cluster_profiles", _fig_cluster_profiles, (config, df_clusters)),
        ("fig06_timeseries_by_cluster", _fig_timeseries_cluster, (config, df_clusters)),
    ]

    for basename, func, args in fig_funcs:
        out_path = figures_dir / f"{basename}.{fmt}"
        try:
            fig = func(*args)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("Figure saved: %s", out_path)
        except Exception as exc:
            logger.error("Failed to produce %s: %s", basename, exc)


# ---------------------------------------------------------------------------
# Figure 1 — Problem frequency bar charts
# ---------------------------------------------------------------------------
def _fig_problem_frequency(
    config: dict,
    df_clim: pd.DataFrame,
) -> plt.Figure:
    """
    Bar chart of mean seasonal problem frequency per region.

    X-axis: region (sorted W→E by longitude)
    Y-axis: fraction of season days with each problem active
    Bars: grouped by problem type; colours from PROBLEM_COLORS.
    One subplot per elevation band.
    """
    regions = config["regions"]
    elevation_bands = config["elevation_bands"]

    # Sort regions west→east by longitude
    region_order = sorted(regions.keys(), key=lambda r: regions[r]["lon"])
    region_labels = [regions[r]["name"] for r in region_order]

    n_elev = len(elevation_bands)
    fig, axes = plt.subplots(
        1, n_elev,
        figsize=(_DOUBLE_COL, _ROW_HEIGHT * 1.5),
        sharey=True,
    )
    if n_elev == 1:
        axes = [axes]

    problem_cols = list(PROBLEM_COLORS.keys())
    n_problems = len(problem_cols)
    bar_width = 0.12
    x = np.arange(len(region_order))

    for ax_idx, (ax, elev_m) in enumerate(zip(axes, elevation_bands)):
        df_elev = df_clim[df_clim["elevation_m"] == elev_m]

        for p_idx, prob_col in enumerate(problem_cols):
            # Mean fractional frequency per region
            freq = []
            for region_key in region_order:
                df_reg = df_elev[df_elev["region"] == region_key]
                if df_reg.empty or prob_col not in df_reg.columns:
                    freq.append(0.0)
                else:
                    # Fraction: mean(problem_days) / ~212 days per season
                    season_len = df_reg["total_problem_days"].add(1).mean()
                    freq.append(df_reg[prob_col].mean() / max(season_len, 1))

            offsets = (p_idx - n_problems / 2 + 0.5) * bar_width
            ax.bar(
                x + offsets,
                freq,
                width=bar_width,
                color=PROBLEM_COLORS[prob_col],
                label=PROBLEM_LABELS[prob_col] if ax_idx == 0 else None,
                alpha=0.85,
                edgecolor="none",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=_BODY_FS - 1)
        ax.set_title(f"{elev_m} m a.s.l.", fontsize=_TITLE_FS)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="both", labelsize=_BODY_FS)
        _despine(ax)

    axes[0].set_ylabel("Fraction of season days", fontsize=_LABEL_FS)

    # Legend on first subplot
    handles = [
        Patch(facecolor=PROBLEM_COLORS[c], label=PROBLEM_LABELS[c])
        for c in problem_cols
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=n_problems,
        fontsize=_BODY_FS,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Avalanche problem frequency — Austrian Alps",
        fontsize=_TITLE_FS,
        y=1.06,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Seasonal distributions (box plots)
# ---------------------------------------------------------------------------
def _fig_seasonal_distributions(
    config: dict,
    df_clim: pd.DataFrame,
) -> plt.Figure:
    """
    Box plots of seasonal problem day counts for each problem type,
    stratified by elevation band.
    """
    elevation_bands = config["elevation_bands"]
    problem_cols = list(PROBLEM_COLORS.keys())
    n_problems = len(problem_cols)

    fig, axes = plt.subplots(
        1, n_problems,
        figsize=(_DOUBLE_COL, _ROW_HEIGHT),
        sharey=False,
    )
    if n_problems == 1:
        axes = [axes]

    for ax, prob_col in zip(axes, problem_cols):
        data_by_elev = [
            df_clim.loc[df_clim["elevation_m"] == elev, prob_col].dropna().values
            for elev in elevation_bands
        ]

        bp = ax.boxplot(
            data_by_elev,
            labels=[f"{e} m" for e in elevation_bands],
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(to_rgba(PROBLEM_COLORS[prob_col], alpha=0.6))

        ax.set_title(PROBLEM_LABELS[prob_col], fontsize=_TITLE_FS, pad=3)
        ax.set_xlabel("Elevation", fontsize=_LABEL_FS)
        ax.set_xticklabels(
            [f"{e} m" for e in elevation_bands],
            rotation=30,
            ha="right",
            fontsize=_BODY_FS,
        )
        ax.tick_params(axis="y", labelsize=_BODY_FS)
        _despine(ax)

    axes[0].set_ylabel("Days per season", fontsize=_LABEL_FS)
    fig.suptitle(
        "Seasonal problem day distributions by elevation",
        fontsize=_TITLE_FS,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Wet-snow onset vs. elevation
# ---------------------------------------------------------------------------
def _fig_wet_snow_onset(
    config: dict,
    df_clim: pd.DataFrame,
) -> plt.Figure:
    """
    Scatter and regression plot of wet-snow onset DOY vs. elevation.
    Each region is a separate colour; overall linear regression shown.
    """
    regions = config["regions"]
    elevation_bands = config["elevation_bands"]

    cmap = plt.get_cmap("tab20")
    region_colors = {r: cmap(i / len(regions)) for i, r in enumerate(regions)}

    fig, ax = plt.subplots(figsize=(_SINGLE_COL * 1.4, _ROW_HEIGHT * 1.3))

    all_elev, all_doy = [], []

    for region_key, region_meta in regions.items():
        df_reg = df_clim[df_clim["region"] == region_key].copy()
        df_reg = df_reg.dropna(subset=["wet_snow_onset_doy"])

        x = df_reg["elevation_m"].values
        y = df_reg["wet_snow_onset_doy"].values

        ax.scatter(
            x, y,
            color=region_colors[region_key],
            s=15,
            alpha=0.6,
            linewidths=0,
            label=region_meta["name"],
            zorder=3,
        )

        all_elev.extend(x.tolist())
        all_doy.extend(y.tolist())

        # Per-region regression
        if len(x) >= 3:
            coef = np.polyfit(x, y, 1)
            x_range = np.linspace(min(elevation_bands), max(elevation_bands), 50)
            ax.plot(
                x_range,
                np.polyval(coef, x_range),
                color=region_colors[region_key],
                linewidth=0.8,
                alpha=0.5,
            )

    # Overall regression
    if len(all_elev) > 3:
        coef_all = np.polyfit(all_elev, all_doy, 1)
        x_range = np.linspace(min(elevation_bands), max(elevation_bands), 100)
        ax.plot(
            x_range,
            np.polyval(coef_all, x_range),
            color="black",
            linewidth=1.5,
            label=f"Overall  (slope = {coef_all[0]:.2f} d / 100 m)",
            zorder=5,
        )

    # Month labels on y-axis
    _add_month_ticks(ax, axis="y")

    ax.set_xlabel("Elevation (m a.s.l.)", fontsize=_LABEL_FS)
    ax.set_ylabel("Wet-snow onset (day of year)", fontsize=_LABEL_FS)
    ax.set_title("Wet-snow onset DOY vs. elevation", fontsize=_TITLE_FS)
    ax.tick_params(labelsize=_BODY_FS)
    ax.legend(
        fontsize=_BODY_FS - 1,
        ncol=2,
        frameon=False,
        loc="upper left",
    )
    _despine(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Cluster map (region × season heat map)
# ---------------------------------------------------------------------------
def _fig_cluster_map(
    config: dict,
    df_clusters: pd.DataFrame,
) -> plt.Figure:
    """
    Heat-map matrix of cluster assignment per (region, season).

    Rows: seasons (chronological)
    Columns: regions (sorted W→E by longitude)
    Colour: cluster label (1–k)
    """
    regions = config["regions"]
    region_order = sorted(regions.keys(), key=lambda r: regions[r]["lon"])
    region_labels = [regions[r]["name"] for r in region_order]

    if "cluster" not in df_clusters.columns:
        logger.warning("No 'cluster' column in df_clusters — skipping cluster map.")
        fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _ROW_HEIGHT))
        ax.text(0.5, 0.5, "Cluster labels not available", ha="center", va="center")
        return fig

    # Pivot: seasons × regions
    df_ref = df_clusters[df_clusters["elevation_m"] == config["clustering"]["reference_elevation"]]
    pivot = df_ref.pivot_table(index="season", columns="region", values="cluster", aggfunc="first")
    pivot = pivot.reindex(columns=region_order)

    seasons = pivot.index.tolist()
    k = int(config["clustering"]["n_clusters"])

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(CLUSTER_COLORS[:k])
    bounds = np.arange(0.5, k + 1.5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_height = max(_ROW_HEIGHT, len(seasons) * 0.22)
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, fig_height))

    mat = pivot.values.astype(float)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(region_order)))
    ax.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=_BODY_FS)
    ax.set_yticks(range(len(seasons)))
    ax.set_yticklabels(seasons, fontsize=_BODY_FS - 1)
    ax.set_xlabel("Region (W→E)", fontsize=_LABEL_FS)
    ax.set_ylabel("Season", fontsize=_LABEL_FS)
    ax.set_title(
        f"Avalanche climate type cluster assignment ({config['clustering']['reference_elevation']} m)",
        fontsize=_TITLE_FS,
    )

    cb = fig.colorbar(im, ax=ax, ticks=range(1, k + 1), shrink=0.6)
    cb.set_ticklabels([f"Type {cl}" for cl in range(1, k + 1)], fontsize=_BODY_FS)
    cb.ax.tick_params(labelsize=_BODY_FS)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Cluster profiles (mean problem frequency per cluster)
# ---------------------------------------------------------------------------
def _fig_cluster_profiles(
    config: dict,
    df_clusters: pd.DataFrame,
) -> plt.Figure:
    """
    Grouped bar chart of mean problem day count per cluster type.
    """
    if "cluster" not in df_clusters.columns:
        logger.warning("No cluster column — skipping cluster profiles.")
        fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _ROW_HEIGHT))
        ax.text(0.5, 0.5, "Cluster labels not available", ha="center", va="center")
        return fig

    k = int(config["clustering"]["n_clusters"])
    ref_elev = int(config["clustering"]["reference_elevation"])
    problem_cols = list(PROBLEM_COLORS.keys())
    n_problems = len(problem_cols)

    df_ref = df_clusters[df_clusters["elevation_m"] == ref_elev]
    cluster_means = df_ref.groupby("cluster")[problem_cols].mean()

    fig, ax = plt.subplots(figsize=(_SINGLE_COL * 1.5, _ROW_HEIGHT * 1.2))

    x = np.arange(k)
    bar_width = 0.12
    for p_idx, prob_col in enumerate(problem_cols):
        offsets = (p_idx - n_problems / 2 + 0.5) * bar_width
        vals = [cluster_means.loc[cl, prob_col] if cl in cluster_means.index else 0.0
                for cl in range(1, k + 1)]
        ax.bar(
            x + offsets,
            vals,
            width=bar_width,
            color=PROBLEM_COLORS[prob_col],
            label=PROBLEM_LABELS[prob_col],
            alpha=0.85,
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {cl}" for cl in range(1, k + 1)], fontsize=_BODY_FS)
    ax.set_xlabel("Avalanche climate type", fontsize=_LABEL_FS)
    ax.set_ylabel("Mean seasonal days", fontsize=_LABEL_FS)
    ax.set_title(
        f"Mean problem count per cluster type ({ref_elev} m)",
        fontsize=_TITLE_FS,
    )
    ax.tick_params(labelsize=_BODY_FS)
    _despine(ax)

    ax.legend(
        fontsize=_BODY_FS - 1,
        frameon=False,
        loc="upper right",
        ncol=1,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Time series coloured by cluster
# ---------------------------------------------------------------------------
def _fig_timeseries_cluster(
    config: dict,
    df_clusters: pd.DataFrame,
) -> plt.Figure:
    """
    Bar chart of total problem days per season, coloured by cluster label.
    One subplot per region.
    """
    regions = config["regions"]
    region_order = sorted(regions.keys(), key=lambda r: regions[r]["lon"])
    ref_elev = int(config["clustering"]["reference_elevation"])
    k = int(config["clustering"]["n_clusters"])

    df_ref = df_clusters[df_clusters["elevation_m"] == ref_elev]

    n_regions = len(region_order)
    ncols = min(3, n_regions)
    nrows = (n_regions + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_DOUBLE_COL, _ROW_HEIGHT * nrows * 0.85),
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax_idx, region_key in enumerate(region_order):
        ax = axes_flat[ax_idx]
        df_reg = df_ref[df_ref["region"] == region_key].sort_values("season")

        if df_reg.empty:
            ax.set_visible(False)
            continue

        seasons = df_reg["season"].values
        totals = df_reg["total_problem_days"].values
        clusters = df_reg["cluster"].values if "cluster" in df_reg.columns else np.ones(len(df_reg))

        bar_colors = [
            CLUSTER_COLORS[int(cl) - 1] if not np.isnan(cl) else "gray"
            for cl in clusters
        ]

        ax.bar(range(len(seasons)), totals, color=bar_colors, width=0.8, alpha=0.85, edgecolor="none")
        ax.set_xticks(range(len(seasons)))
        ax.set_xticklabels(
            [s.split("/")[0] for s in seasons],
            rotation=60,
            ha="right",
            fontsize=_BODY_FS - 2,
        )
        ax.set_title(regions[region_key]["name"], fontsize=_BODY_FS + 1, pad=2)
        ax.tick_params(axis="y", labelsize=_BODY_FS - 1)
        _despine(ax)

    # Hide unused subplots
    for ax in axes_flat[len(region_order):]:
        ax.set_visible(False)

    # Shared y-axis label
    fig.text(0.02, 0.5, "Total problem days per season", va="center", rotation="vertical",
             fontsize=_LABEL_FS)

    # Cluster legend
    handles = [Patch(facecolor=CLUSTER_COLORS[cl], label=f"Type {cl + 1}") for cl in range(k)]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=k,
        fontsize=_BODY_FS,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        f"Seasonal problem days — coloured by cluster ({ref_elev} m)",
        fontsize=_TITLE_FS,
        y=1.04,
    )
    fig.tight_layout(rect=(0.05, 0, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rcparams() -> dict:
    """Return matplotlib rcParams for consistent scientific style."""
    return {
        "font.size": _BODY_FS,
        "axes.titlesize": _TITLE_FS,
        "axes.labelsize": _LABEL_FS,
        "xtick.labelsize": _BODY_FS,
        "ytick.labelsize": _BODY_FS,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "pdf.fonttype": 42,   # embed fonts in PDF
        "ps.fonttype": 42,
    }


def _despine(ax: plt.Axes) -> None:
    """Remove top and right spines from an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_month_ticks(ax: plt.Axes, axis: str = "y") -> None:
    """
    Add abbreviated month labels on the DOY axis.

    Parameters
    ----------
    ax : Axes
        Target axes.
    axis : str
        ``'x'`` or ``'y'``.
    """
    import calendar
    # DOY for the 1st of each month (non-leap year)
    month_doys = [
        (sum(calendar.monthrange(2001, m)[1] for m in range(1, m_)) + 1, calendar.month_abbr[m_])
        for m_ in range(1, 13)
    ]
    ticks = [d for d, _ in month_doys]
    labels = [l for _, l in month_doys]

    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=_BODY_FS)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=_BODY_FS)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    print("plotting: module loaded. Run create_all_figures(config, df_clim, df_clusters) to produce figures.")
