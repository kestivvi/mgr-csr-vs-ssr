from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from orchestrator.shared.research import Column, MetricName

from ..config import PLOT_PALETTE
from ..utils.plotting import get_ordered_tech_list
from ..utils.reporting import write_report

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_capacity_k6_analysis(analyzer: PerformanceAnalyzer) -> None:
    if not analyzer.experiment or analyzer.experiment.metrics.empty:
        return
    raw_results = compute_capacity_metrics(analyzer)
    if raw_results is not None:
        generate_capacity_plots(analyzer, raw_results)
        content = generate_capacity_report(analyzer, raw_results)
        write_report(analyzer, content)


def compute_capacity_metrics(analyzer: PerformanceAnalyzer) -> Optional[pd.DataFrame]:
    required = [
        MetricName.K6_SUCCESSFUL_RPS,
        MetricName.K6_TOTAL_RPS,
        MetricName.CPU,
        MetricName.MEMORY,
    ]
    metrics_df = analyzer.experiment.metrics
    available = metrics_df[Column.METRIC].unique()
    if not all(m in available for m in required):
        return None

    results = []
    for (tech, run), run_df in metrics_df.groupby([Column.SERVER_TYPE, Column.RUN_NUMBER]):
        pivot = (
            run_df.pivot_table(index=Column.TIME_SEC, columns=Column.METRIC, values=Column.VALUE)
            .reindex(columns=required)
            .fillna(0)
        )
        rolling_mins = pivot[MetricName.K6_SUCCESSFUL_RPS].rolling(window=30, min_periods=1).min()
        sustained_rps = float(rolling_mins.max())
        peak_rps = float(pivot[MetricName.K6_SUCCESSFUL_RPS].max())
        sustained_time = rolling_mins.idxmax()
        cpu_at = (
            float(pivot.loc[sustained_time, MetricName.CPU])
            if sustained_time in pivot.index
            else 0.0
        )
        ram_at = (
            float(pivot.loc[sustained_time, MetricName.MEMORY])
            if sustained_time in pivot.index
            else 0.0
        )
        group = run_df[Column.GROUP].iloc[0]
        results.append(
            {
                Column.GROUP: group,
                Column.SERVER_TYPE: tech,
                Column.RUN_NUMBER: run,
                "sustained_rps": sustained_rps,
                "peak_rps": peak_rps,
                "cpu_at_sustained": cpu_at,
                "ram_at_sustained": ram_at,
            }
        )

    return pd.DataFrame(results)


def generate_capacity_plots(analyzer: PerformanceAnalyzer, raw_results: pd.DataFrame) -> None:
    # 1. Timeseries Plots
    metrics_to_plot = {
        MetricName.K6_SUCCESSFUL_RPS: "successful_throughput_rps_timeseries",
        MetricName.CPU: "cpu_usage_timeseries",
        MetricName.MEMORY: "ram_usage_timeseries",
    }
    labels = {
        MetricName.K6_SUCCESSFUL_RPS: "Przepustowość (RPS)",
        MetricName.CPU: "Zużycie CPU (%)",
        MetricName.MEMORY: "Zużycie pamięci (MB)",
    }
    metrics_df = analyzer.experiment.metrics
    for metric, filename in metrics_to_plot.items():
        m_df = metrics_df[metrics_df[Column.METRIC] == metric]
        if m_df.empty:
            continue

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=m_df, x=Column.TIME_SEC, y=Column.VALUE, hue=Column.SERVER_TYPE, errorbar="sd"
        )
        plt.title(f"Test pojemnościowy: {labels[metric]} w czasie")
        plt.xlabel("Czas (sekundy)")
        plt.ylabel(labels[metric])
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Technologia")
        plt.tight_layout()
        plt.savefig(analyzer.plots_dir / f"{filename}.png")
        plt.close()

    # 2. Comparison Bar Charts
    if raw_results.empty:
        return

    # Work on a copy for plotting
    plot_df = raw_results.copy()
    # Note: 'group' is already added by the ExperimentLoader
    plot_df["display_name"] = plot_df[Column.SERVER_TYPE].str.replace(
        r"^(csr|ssr)-", "", regex=True
    )

    # Handle duplicate names by adding group suffix
    name_counts = plot_df.groupby("display_name")[Column.GROUP].nunique()
    duplicate_names = name_counts[name_counts > 1].index
    plot_df.loc[plot_df["display_name"].isin(duplicate_names), "display_name"] += (
        " (" + plot_df[Column.GROUP] + ")"
    )

    comparisons = {
        "sustained_rps": "capacity_rps_comparison.png",
        "cpu_at_sustained": "capacity_cpu_at_sustained_usage.png",
        "ram_at_sustained": "capacity_ram_at_sustained_usage.png",
    }
    titles = {
        "sustained_rps": "Maksymalna utrzymana przepustowość (utrzymany RPS)",
        "cpu_at_sustained": "Zużycie CPU przy utrzymanej przepustowości (%)",
        "ram_at_sustained": "Zużycie pamięci przy utrzymanej przepustowości (MB)",
    }

    for col, filename in comparisons.items():
        if col not in plot_df.columns:
            continue
        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        if col == "sustained_rps":
            # Grouped bar chart for RPS (Peak vs Sustained)
            rps_df = plot_df.melt(
                id_vars=["display_name", Column.GROUP, Column.RUN_NUMBER],
                value_vars=["sustained_rps", "peak_rps"],
                var_name="metric_type",
                value_name="rps",
            )
            rps_df["metric_type"] = rps_df["metric_type"].map(
                {"peak_rps": "Szczytowy", "sustained_rps": "Utrzymany"}
            )

            # Use global family-based order
            full_order = get_ordered_tech_list(analyzer, plot_df)
            order_map = plot_df.set_index(Column.SERVER_TYPE)["display_name"].to_dict()
            order = [order_map[t] for t in full_order if t in order_map]

            sns.barplot(
                data=rps_df,
                y="display_name",
                x="rps",
                hue="metric_type",
                hue_order=["Szczytowy", "Utrzymany"],
                order=order,
                palette={"Szczytowy": "#4C72B0", "Utrzymany": "#A0C4FF"},
                capsize=0.1,
                alpha=0.8,
                ax=ax,
            )
            sns.stripplot(
                data=rps_df,
                y="display_name",
                x="rps",
                hue="metric_type",
                hue_order=["Szczytowy", "Utrzymany"],
                palette={"Szczytowy": "#222222", "Utrzymany": "#222222"},
                order=order,
                size=3,
                alpha=0.4,
                dodge=True,
                jitter=True,
                legend=False,
                ax=ax,
            )

            # Add value labels for both bars in the group
            for i, name in enumerate(order):
                for j, _m_type in enumerate(["Szczytowy", "Utrzymany"]):
                    m_key = "peak_rps" if j == 0 else "sustained_rps"
                    subset = plot_df[plot_df["display_name"] == name][m_key]
                    m = subset.mean()
                    s = subset.std() if len(subset) > 1 else 0

                    # Offset y position for grouped bars (default width is 0.8, so offset is 0.2)
                    y_pos = i - 0.2 if j == 0 else i + 0.2
                    extent = max(m + s, subset.max())

                    label_text = f"{m:.0f} (±{s:.1f})"
                    ax.annotate(
                        label_text,
                        (extent, y_pos),
                        ha="left",
                        va="center",
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color="#222222",
                    )
            # Manual coloring for SSR (Red) and CSR (Blue)
            n_techs = len(order)
            for i, patch in enumerate(ax.patches):
                if i >= 2 * n_techs:
                    break
                tech_idx = i % n_techs
                is_sustained = i >= n_techs
                tech_name = order[tech_idx]
                group = plot_df[plot_df["display_name"] == tech_name][Column.GROUP].iloc[0].lower()

                if group == "ssr":
                    color = "#d62728" if not is_sustained else "#ff9f9b"
                else:
                    color = "#1f77b4" if not is_sustained else "#a1c9f4"
                patch.set_facecolor(color)
                patch.set_edgecolor("white")

            # Custom legend for the complex color scheme
            legend_elements = [
                Patch(facecolor="#d62728", label="SSR - Szczytowy"),
                Patch(facecolor="#ff9f9b", label="SSR - Utrzymany"),
                Patch(facecolor="#1f77b4", label="CSR - Szczytowy"),
                Patch(facecolor="#a1c9f4", label="CSR - Utrzymany"),
            ]
            plt.legend(
                handles=legend_elements,
                title="Technologia i typ RPS",
                loc="lower right",
                fontsize=8,
            )
        else:
            # Standard bar chart for CPU/RAM - use global family-based order
            full_order = get_ordered_tech_list(analyzer, plot_df)
            order_map = plot_df.set_index(Column.SERVER_TYPE)["display_name"].to_dict()
            order = [order_map[t] for t in full_order if t in order_map]
            sns.barplot(
                data=plot_df,
                y="display_name",
                x=col,
                hue=Column.GROUP,
                order=order,
                palette=PLOT_PALETTE,
                dodge=False,
                errorbar="sd",
                capsize=0.1,
                alpha=0.8,
                ax=ax,
            )
            sns.stripplot(
                data=plot_df,
                y="display_name",
                x=col,
                color="#333333",
                order=order,
                size=4,
                alpha=0.6,
                dodge=False,
                jitter=True,
                ax=ax,
            )

            # Add labels for single bars
            means = plot_df.groupby("display_name")[col].mean()
            stds = plot_df.groupby("display_name")[col].std().fillna(0)
            for i, name in enumerate(order):
                m, s = means[name], stds[name]
                group_data = plot_df[plot_df["display_name"] == name][col]
                extent = max(m + s, group_data.max())
                label_text = f"{m:.1f} (±{s:.2f})" if s > 0 else f"{m:.1f}"
                ax.annotate(
                    label_text,
                    (extent, i),
                    ha="left",
                    va="center",
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=9,
                )
            plt.legend(title="Grupa", loc="lower right")

        plt.title(titles[col])
        plt.xlabel(titles[col].split("(")[-1].replace(")", ""))
        plt.ylabel("")

        # Color y-axis labels by group
        for label in ax.get_yticklabels():
            name = label.get_text()
            match = plot_df[plot_df["display_name"] == name]
            if not match.empty:
                group = match[Column.GROUP].iloc[0]
                label.set_color(PLOT_PALETTE.get(group, "black"))
                label.set_fontweight("bold")

        plt.tight_layout()
        plt.savefig(analyzer.plots_dir / filename)
        plt.close()


def generate_capacity_report(analyzer: PerformanceAnalyzer, raw_results: pd.DataFrame) -> str:
    summary = raw_results.groupby(Column.SERVER_TYPE).agg(
        {
            "sustained_rps": ["mean", "std"],
            "peak_rps": ["mean", "std"],
            "cpu_at_sustained": ["mean", "std"],
            "ram_at_sustained": ["mean", "std"],
        }
    )

    # Flatten multi-index columns
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    # Sort summary by global family-based order
    full_order = get_ordered_tech_list(analyzer, summary)
    summary[Column.SERVER_TYPE] = pd.Categorical(summary[Column.SERVER_TYPE], categories=full_order, ordered=True)
    summary = summary.sort_values(Column.SERVER_TYPE)

    # Create a user-friendly table
    rows = []
    for _, row in summary.iterrows():
        cpu_val = f"{row['cpu_at_sustained_mean']:.1f} (±{row['cpu_at_sustained_std']:.2f})"
        ram_val = f"{row['ram_at_sustained_mean']:.1f} (±{row['ram_at_sustained_std']:.2f})"
        sustained_rps_str = f"{row['sustained_rps_mean']:.1f} (±{row['sustained_rps_std']:.2f})"
        peak_rps_str = f"{row['peak_rps_mean']:.1f} (±{row['peak_rps_std']:.2f})"
        rows.append(
            {
                "Technologia": row[Column.SERVER_TYPE],
                "Utrzymany RPS": sustained_rps_str,
                "Szczytowy RPS": peak_rps_str,
                "CPU @ Sustained (%)": cpu_val,
                "RAM @ Sustained (MB)": ram_val,
            }
        )

    report = [
        f"# Capacity Report for `{analyzer.input_dir.name}`",
        "\n### Podsumowanie wyników zagregowanych",
        pd.DataFrame(rows).to_markdown(index=False),
        "\n*Wartości w nawiasach oznaczają odchylenie standardowe "
        "(standard deviation) z wielu prób.*",
    ]
    return "\n".join(report)
