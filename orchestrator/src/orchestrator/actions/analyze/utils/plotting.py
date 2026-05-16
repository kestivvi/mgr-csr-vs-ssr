from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from orchestrator.shared.research import Column

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PLOT_PALETTE

# Ensure Gap is transparent
PLOT_PALETTE["Gap"] = "none"

sns.set_theme(style="whitegrid")


def get_ordered_tech_list(analyzer: PerformanceAnalyzer, df: pd.DataFrame) -> List[str]:
    all_techs = list(df[Column.SERVER_TYPE].unique())

    def get_sort_key(tech: str):
        tech_lower = tech.lower()

        # Lookup manifest from structured metadata
        subjects = analyzer.experiment.subject_metadata if analyzer.experiment else {}
        manifest = subjects.get(tech_lower, {})

        # 1. Family Priority (from config.yaml)
        family = manifest.get("family", "unknown")
        family_score = 999
        for i, family_keywords in enumerate(analyzer.families):
            keywords = [family_keywords] if isinstance(family_keywords, str) else family_keywords
            # Match by explicit family name or fallback to keyword matching
            if family in keywords or any(k in tech_lower for k in keywords):
                family_score = i
                break

        # 2. Strategy Priority (CSR < SSR)
        strategy = manifest.get("strategy", "csr" if tech_lower.startswith("csr-") else "ssr")
        strategy_score = 0 if strategy == "csr" else 1

        # 3. Meta-framework Priority (Pure first, then Alphabetical)
        meta = manifest.get("meta_framework")
        meta_sort_key = "" if meta is None else str(meta)

        # 4. Runtime Priority (from config.yaml)
        runtime = manifest.get("runtime", "node")
        runtime_score = 999
        for i, r in enumerate(analyzer.runtime_priority):
            if runtime == r:
                runtime_score = i
                break

        return (family_score, strategy_score, meta_sort_key, runtime_score, tech_lower)

    return sorted(all_techs, key=get_sort_key)


def clean_tech_names_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if Column.SERVER_TYPE not in df.columns or Column.GROUP not in df.columns:
        return df

    # Create display names
    df["display_name"] = df[Column.SERVER_TYPE].str.replace(r"^(csr|ssr)-", "", regex=True)

    # Handle duplicates across groups
    counts = df.groupby("display_name")[Column.GROUP].nunique()
    dupes = counts[counts > 1].index

    mask = df["display_name"].isin(dupes)
    df.loc[mask, "display_name"] = (
        df.loc[mask, "display_name"] + " (" + df.loc[mask, Column.GROUP] + ")"
    )
    return df


def create_scorecard_heatmap(
    analyzer: PerformanceAnalyzer, scorecard_ranks_df: pd.DataFrame
) -> Optional[Path]:
    # We also need values, but we can re-derive them if needed or pass them.
    # For now, let's assume the ranks are enough or we pass them too.
    if scorecard_ranks_df.empty:
        return None

    plt.figure(figsize=(20, 10))
    avg_ranks = scorecard_ranks_df.mean(axis=1).sort_values()
    ordered_techs = avg_ranks.index.tolist()
    ordered_metrics = scorecard_ranks_df.index
    ranks_ordered = scorecard_ranks_df.reindex(index=ordered_metrics, columns=ordered_techs)

    # Get display names mapping
    temp_df = pd.DataFrame({Column.SERVER_TYPE: ordered_techs})
    # Get groups from experiment
    if not analyzer.experiment:
        return None
    metrics_df = analyzer.experiment.metrics
    tech_to_group = metrics_df.groupby(Column.SERVER_TYPE)[Column.GROUP].first().to_dict()
    temp_df[Column.GROUP] = temp_df[Column.SERVER_TYPE].map(tech_to_group).fillna("Uncategorized")
    temp_df = clean_tech_names_df(temp_df)
    display_name_map = temp_df.set_index(Column.SERVER_TYPE)["display_name"].to_dict()

    ranks_ordered.columns = [display_name_map.get(c, c) for c in ranks_ordered.columns]

    ax = sns.heatmap(
        ranks_ordered,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Performance Rank (1 is best)"},
    )

    plt.title("Karta wyników (Scorecard)", fontsize=16)
    plt.xlabel("")
    plt.ylabel("Metryka")

    # Color x-axis labels by group
    ax.set_xticks(np.arange(len(ordered_techs)) + 0.5)
    ax.set_xticklabels(ranks_ordered.columns, rotation=45, ha="right")

    # Map tech to group color
    for label in ax.get_xticklabels():
        name = label.get_text()
        match = temp_df[temp_df["display_name"] == name]
        if not match.empty:
            group = match[Column.GROUP].iloc[0]
            label.set_color(PLOT_PALETTE.get(group, "black"))
            label.set_fontweight("bold")

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
    df = clean_tech_names_df(df)
    plot_order_orig = get_ordered_tech_list(analyzer, df)

    # Create mapping from orig to display for order
    order_map = df.set_index(Column.SERVER_TYPE)["display_name"].to_dict()
    plot_order = [order_map[t] for t in plot_order_orig if t in order_map]

    plt.figure(figsize=(14, 8))

    if plot_type == "box":
        sns.boxplot(
            data=df,
            x="display_name",
            y=stat_col,
            hue=Column.GROUP,
            dodge=False,
            order=plot_order,
        )
    elif plot_type == "violin":
        sns.violinplot(
            data=df,
            x="display_name",
            y=stat_col,
            hue=Column.GROUP,
            dodge=False,
            inner="quartile",
            cut=0,
            order=plot_order,
        )

    plt.title(f"Rozkład: {metric_name}", fontsize=16)
    plt.ylabel(metric_name)
    plt.xlabel("")

    # Capture axis for tick manipulation
    ax = plt.gca()
    plt.xticks(rotation=45, ha="right")

    # Map tech to group color
    for label in ax.get_yticklabels():
        name = label.get_text()
        if not df.empty:
            match = df[df["display_name"] == name]
            if not match.empty:
                group = match[Column.GROUP].iloc[0]
                label.set_color(PLOT_PALETTE.get(group, "black"))
                label.set_fontweight("bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    filename = f"{metric_name.lower().replace(' ', '_')}_{plot_type}_comparison.png"
    path = Path(analyzer.plots_dir) / re.sub(r"[^a-z0-9_.-]", "", filename)
    plt.savefig(path)
    plt.close()
    return path


def create_timeseries_plot(
    analyzer: PerformanceAnalyzer,
    metric: str,
    metric_name: str,
    group_filter: Optional[str] = None,
) -> Optional[Path]:
    if not analyzer.experiment:
        return None
    metrics_df = analyzer.experiment.metrics
    if group_filter:
        df_metric = metrics_df[
            (metrics_df[Column.METRIC] == metric) & (metrics_df[Column.GROUP] == group_filter)
        ]
    else:
        df_metric = metrics_df[metrics_df[Column.METRIC] == metric]

    if df_metric.empty:
        return None
    max_time = df_metric[Column.TIME_SEC].max()
    if pd.isna(max_time):
        return None

    full_time_index = pd.to_timedelta(np.arange(int(max_time) + 1), unit="s")
    processed_dfs = []
    for group, group_df in df_metric.groupby([Column.SERVER_TYPE, Column.RUN_NUMBER]):
        temp_df = group_df.set_index(pd.to_timedelta(group_df[Column.TIME_SEC], unit="s"))
        temp_df = temp_df.reindex(full_time_index).ffill().bfill()
        temp_df[Column.SERVER_TYPE] = group[0]
        temp_df[Column.RUN_NUMBER] = group[1]
        temp_df[Column.TIME_SEC] = temp_df.index.total_seconds()
        processed_dfs.append(temp_df.reset_index(drop=True))

    if not processed_dfs:
        return None
    plot_df = pd.concat(processed_dfs, ignore_index=True)
    agg_df = (
        plot_df.groupby([Column.SERVER_TYPE, Column.TIME_SEC])[Column.VALUE]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    plot_order = get_ordered_tech_list(analyzer, agg_df)

    plt.figure(figsize=(14, 8))

    # Use a varied palette to distinguish frameworks
    palette = sns.color_palette("tab10", n_colors=len(plot_order))
    tech_colors = {tech: palette[i] for i, tech in enumerate(plot_order)}

    for tech in plot_order:
        tech_df = agg_df[agg_df[Column.SERVER_TYPE] == tech]
        if tech_df.empty:
            continue

        color = tech_colors[tech]

        plt.plot(tech_df[Column.TIME_SEC], tech_df["mean"], label=tech, color=color, linewidth=2)
        plt.fill_between(
            tech_df[Column.TIME_SEC], tech_df["min"], tech_df["max"], color=color, alpha=0.15
        )

    title_suffix = f" ({group_filter})" if group_filter else ""
    plt.title(f"Serie czasowe: {metric_name}{title_suffix}", fontsize=16)
    plt.xlabel("Czas (sekundy)")
    plt.ylabel(metric_name)
    leg = plt.legend(title="Technologia", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Color legend labels to match line colors
    for text in leg.get_texts():
        tech = text.get_text()
        if tech in tech_colors:
            text.set_color(tech_colors[tech])
            text.set_fontweight("bold")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    group_suffix = f"_{group_filter.lower().replace('-', '_')}" if group_filter else ""
    base_filename = f"{metric_name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
    path = Path(analyzer.plots_dir) / re.sub(r"[^a-z0-9_.-]", "", base_filename)
    plt.savefig(path)
    plt.close()
    return path
