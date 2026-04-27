from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PLOT_PALETTE

sns.set_theme(style="whitegrid")


def get_ordered_tech_list(analyzer: PerformanceAnalyzer, df: pd.DataFrame) -> List[str]:
    all_techs = set(df["server_type"].unique())
    ordered = [t for t in analyzer.chart_order if t in all_techs]
    remaining = sorted([t for t in all_techs if t not in set(ordered)])
    return ordered + remaining


def create_scorecard_heatmap(analyzer: PerformanceAnalyzer) -> Optional[Path]:
    if analyzer.scorecard_ranks_df.empty or analyzer.scorecard_values_df.empty:
        return None

    plt.figure(figsize=(20, 10))
    avg_ranks = analyzer.scorecard_ranks_df.mean().sort_values()
    ordered_techs = avg_ranks.index.tolist()
    ordered_metrics = analyzer.scorecard_ranks_df.index
    ranks_ordered = analyzer.scorecard_ranks_df.reindex(
        index=ordered_metrics, columns=ordered_techs
    )
    values_ordered = analyzer.scorecard_values_df.reindex(
        index=ordered_metrics, columns=ordered_techs
    )

    ax = sns.heatmap(
        ranks_ordered,
        annot=values_ordered,
        fmt=".2f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Performance Rank (1 is best)"},
    )

    plt.title("Performance Scorecard", fontsize=16)
    plt.xlabel("Technology (Ordered Best to Worst Overall)")
    plt.ylabel("Metric")

    # Color x-axis labels by group
    ax.set_xticks(np.arange(len(ordered_techs)) + 0.5)
    ax.set_xticklabels(ordered_techs, rotation=45, ha="right")

    # Map tech to group color
    for label in ax.get_xticklabels():
        tech = label.get_text()
        # Find group for this tech from raw_df
        if not analyzer.raw_df.empty:
            match = analyzer.raw_df[analyzer.raw_df["server_type"] == tech]
            if not match.empty:
                group = match["group"].iloc[0]
                label.set_color(PLOT_PALETTE.get(group, "black"))
                label.set_weight("bold")

    plt.yticks(rotation=0)
    plt.tight_layout()

    path = Path(analyzer.plots_dir) / "performance_scorecard.png"
    plt.savefig(path)
    plt.close()
    return path


def create_comparison_plot(
    analyzer: PerformanceAnalyzer,
    df: pd.DataFrame,
    stat_col: str,
    metric_name: str,
    plot_type: str = "box",
) -> Optional[Path]:
    if df.empty:
        return None
    plot_order = get_ordered_tech_list(analyzer, df)
    plt.figure(figsize=(14, 8))

    if plot_type == "box":
        sns.boxplot(
            data=df, x="server_type", y=stat_col, hue="group", dodge=False, order=plot_order
        )
    elif plot_type == "violin":
        sns.violinplot(
            data=df,
            x="server_type",
            y=stat_col,
            hue="group",
            dodge=False,
            inner="quartile",
            cut=0,
            order=plot_order,
        )

    plt.title(f"Distribution of {metric_name}", fontsize=16)
    plt.ylabel(metric_name)
    plt.xlabel("Technology")

    # Capture axis for tick manipulation
    ax = plt.gca()
    plt.xticks(rotation=45, ha="right")

    # Map tech to group color
    for label in ax.get_xticklabels():
        tech = label.get_text()
        if not df.empty:
            match = df[df["server_type"] == tech]
            if not match.empty:
                group = match["group"].iloc[0]
                label.set_color(PLOT_PALETTE.get(group, "black"))
                label.set_weight("bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    filename = f"{metric_name.lower().replace(' ', '_')}_{plot_type}_comparison.png"
    path = Path(analyzer.plots_dir) / re.sub(r"[^a-z0-9_.-]", "", filename)
    plt.savefig(path)
    plt.close()
    return path


def create_timeseries_plot(
    analyzer: PerformanceAnalyzer, metric: str, metric_name: str, group_filter: Optional[str] = None
) -> Optional[Path]:
    if group_filter:
        df_metric = analyzer.raw_df[
            (analyzer.raw_df["metric"] == metric) & (analyzer.raw_df["group"] == group_filter)
        ]
    else:
        df_metric = analyzer.raw_df[analyzer.raw_df["metric"] == metric]

    if df_metric.empty:
        return None
    max_time = df_metric["time_sec"].max()
    if pd.isna(max_time):
        return None

    full_time_index = pd.to_timedelta(np.arange(int(max_time) + 1), unit="s")
    processed_dfs = []
    for group, group_df in df_metric.groupby(["server_type", "run_number"]):
        temp_df = group_df.set_index(pd.to_timedelta(group_df["time_sec"], unit="s"))
        temp_df = temp_df.reindex(full_time_index).ffill().bfill()
        temp_df["server_type"] = group[0]
        temp_df["run_number"] = group[1]
        temp_df["time_sec"] = temp_df.index.total_seconds()
        processed_dfs.append(temp_df.reset_index(drop=True))

    if not processed_dfs:
        return None
    plot_df = pd.concat(processed_dfs, ignore_index=True)
    agg_df = (
        plot_df.groupby(["server_type", "time_sec"])["metric_value"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    plot_order = get_ordered_tech_list(analyzer, agg_df)

    plt.figure(figsize=(14, 8))

    # Use a varied palette to distinguish frameworks
    palette = sns.color_palette("tab10", n_colors=len(plot_order))
    tech_colors = {tech: palette[i] for i, tech in enumerate(plot_order)}

    for tech in plot_order:
        tech_df = agg_df[agg_df["server_type"] == tech]
        if tech_df.empty:
            continue

        color = tech_colors[tech]

        plt.plot(tech_df["time_sec"], tech_df["mean"], label=tech, color=color, linewidth=2)
        plt.fill_between(
            tech_df["time_sec"], tech_df["min"], tech_df["max"], color=color, alpha=0.15
        )

    title_suffix = f" ({group_filter})" if group_filter else ""
    plt.title(f"Time-Series: {metric_name}{title_suffix}", fontsize=16)
    plt.xlabel("Time (seconds)")
    plt.ylabel(metric_name)
    leg = plt.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Color legend labels to match line colors
    for text in leg.get_texts():
        tech = text.get_text()
        if tech in tech_colors:
            text.set_color(tech_colors[tech])
            text.set_weight("bold")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    group_suffix = f"_{group_filter.lower().replace('-', '_')}" if group_filter else ""
    base_filename = (
        f"{metric_name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
    )
    path = Path(analyzer.plots_dir) / re.sub(r"[^a-z0-9_.-]", "", base_filename)
    plt.savefig(path)
    plt.close()
    return path
