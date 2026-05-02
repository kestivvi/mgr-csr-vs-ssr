from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import PLOT_PALETTE
from ..utils.reporting import write_report

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_capacity_k6_analysis(analyzer: PerformanceAnalyzer) -> None:
    summary = compute_capacity_metrics(analyzer)
    if summary is not None:
        generate_capacity_plots(analyzer, summary)
        content = generate_capacity_report(analyzer, summary)
        write_report(analyzer, content)


def compute_capacity_metrics(analyzer: PerformanceAnalyzer) -> Optional[pd.DataFrame]:
    required = ["k6_successful_html_reqs_rate", "k6_total_html_reqs_rate", "cpu", "memory"]
    available = analyzer.raw_df["metric"].unique()
    if not all(m in available for m in required):
        return None

    results = []
    for (tech, run), run_df in analyzer.raw_df.groupby(["server_type", "run_number"]):
        pivot = (
            run_df.pivot_table(index="time_sec", columns="metric", values="metric_value")
            .reindex(columns=required)
            .fillna(0)
        )
        rolling_mins = pivot["k6_successful_html_reqs_rate"].rolling(window=30, min_periods=1).min()
        sustained_rps = float(rolling_mins.max())
        peak_rps = float(pivot["k6_successful_html_reqs_rate"].max())
        sustained_time = rolling_mins.idxmax()
        cpu_at = float(pivot.loc[sustained_time, "cpu"]) if sustained_time in pivot.index else 0.0
        ram_at = (
            float(pivot.loc[sustained_time, "memory"]) if sustained_time in pivot.index else 0.0
        )
        results.append(
            {
                "server_type": tech,
                "run_number": run,
                "sustained_rps": sustained_rps,
                "peak_rps": peak_rps,
                "cpu_at_sustained": cpu_at,
                "ram_at_sustained": ram_at,
            }
        )

    return (
        pd.DataFrame(results).groupby("server_type").mean().drop(columns="run_number").reset_index()
    )


def generate_capacity_plots(analyzer: PerformanceAnalyzer, summary: pd.DataFrame) -> None:
    # 1. Timeseries Plots
    metrics_to_plot = {
        "k6_successful_html_reqs_rate": "successful_throughput_rps_timeseries",
        "cpu": "cpu_usage_timeseries",
        "memory": "ram_usage_timeseries",
    }
    labels = {
        "k6_successful_html_reqs_rate": "Przepustowość (RPS)",
        "cpu": "Zużycie CPU (%)",
        "memory": "Zużycie pamięci (MB)",
    }
    for metric, filename in metrics_to_plot.items():
        m_df = analyzer.raw_df[analyzer.raw_df["metric"] == metric]
        if m_df.empty:
            continue

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=m_df, x="time_sec", y="metric_value", hue="server_type", errorbar="sd")
        plt.title(f"Test pojemnościowy: {labels[metric]} w czasie")
        plt.xlabel("Czas (sekundy)")
        plt.ylabel(labels[metric])
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Technologia")
        plt.tight_layout()
        plt.savefig(analyzer.plots_dir / f"{filename}.png")
        plt.close()

    # 2. Comparison Bar Charts (from summary)
    if summary.empty:
        return

    # Map server_type to group for coloring
    tech_to_group = {
        tech.lower(): group for group, techs in analyzer.groups_config.items() for tech in techs
    }
    summary["group"] = summary["server_type"].str.lower().map(tech_to_group).fillna("Uncategorized")
    summary["display_name"] = summary["server_type"].str.replace(r"^(csr|ssr)-", "", regex=True)
    
    # Handle duplicate names by adding group suffix
    name_counts = summary["display_name"].value_counts()
    duplicate_names = name_counts[name_counts > 1].index
    summary.loc[summary["display_name"].isin(duplicate_names), "display_name"] += " (" + summary["group"] + ")"

    comparisons = {
        "sustained_rps": "capacity_rps_comparison.png",
        "cpu_at_sustained": "capacity_cpu_at_sustained_usage.png",
        "ram_at_sustained": "capacity_ram_at_sustained_usage.png",
    }
    titles = {
        "sustained_rps": "Maksymalna utrzymana przepustowość (utrzymany / szczytowy RPS)",
        "cpu_at_sustained": "Zużycie CPU przy utrzymanej przepustowości (%)",
        "ram_at_sustained": "Zużycie pamięci przy utrzymanej przepustowości (MB)",
    }

    for col, filename in comparisons.items():
        if col not in summary.columns:
            continue
        plt.figure(figsize=(12, 8))
        # Sort by value for better comparison
        sorted_summary = summary.sort_values(col, ascending=False)

        sns.barplot(
            data=sorted_summary,
            y="display_name",
            x=col,
            hue="group",
            palette=PLOT_PALETTE,
            dodge=False,
        )

        plt.title(titles[col])
        plt.xlabel(titles[col].split("(")[-1].replace(")", ""))
        plt.ylabel("")

        # Color y-axis labels by group
        ax = plt.gca()
        for label in ax.get_yticklabels():
            name = label.get_text()
            # Find original tech name to get group
            match = sorted_summary[sorted_summary["display_name"] == name]
            if not match.empty:
                group = match["group"].iloc[0]
                label.set_color(PLOT_PALETTE.get(group, "black"))
                label.set_fontweight("bold")

        # Add value labels to bars
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            if width <= 0: continue
            
            if col == "sustained_rps":
                # Get the corresponding peak_rps
                peak = sorted_summary.iloc[i]["peak_rps"]
                sust = sorted_summary.iloc[i]["sustained_rps"]
                label_text = f"{sust:.0f} / {peak:.0f}"
            else:
                label_text = f"{width:.1f}"
            
            ax.annotate(label_text, 
                        (width, p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', 
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=9)

        plt.legend(title="Grupa", loc="lower right")
        plt.tight_layout()
        plt.savefig(analyzer.plots_dir / filename)
        plt.close()


def generate_capacity_report(analyzer: PerformanceAnalyzer, summary: pd.DataFrame) -> str:
    report = [f"# Capacity Report for `{analyzer.input_dir.name}`", summary.to_markdown()]
    return "\n".join(report)
