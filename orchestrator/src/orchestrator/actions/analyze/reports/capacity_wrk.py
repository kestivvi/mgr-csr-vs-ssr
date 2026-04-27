from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..config import PLOT_PALETTE
from ..utils.reporting import write_report

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_capacity_wrk_analysis(analyzer: PerformanceAnalyzer) -> None:
    generate_capacity_wrk_plots(analyzer)
    content = f"# Capacity WRK Report\n\n{analyzer.wrk_summary.to_markdown()}"
    write_report(analyzer, content)


def generate_capacity_wrk_plots(analyzer: PerformanceAnalyzer) -> None:
    if analyzer.wrk_summary.empty:
        return

    # 1. Base WRK Plots (RPS, Latency)
    metrics = {
        "rps_mean": ("Maximum Throughput (RPS)", "capacity_wrk_rps_comparison.png"),
        "lat_mean": ("Average Latency (ms)", "capacity_wrk_latency_comparison.png"),
    }

    for col, (title, filename) in metrics.items():
        plt.figure(figsize=(12, 8))
        sorted_df = analyzer.wrk_summary.sort_values(col, ascending=(col == "lat_mean"))
        sns.barplot(
            data=sorted_df,
            y="server_type",
            x=col,
            hue="group",
            palette=PLOT_PALETTE,
            dodge=False,
        )
        plt.title(title)
        plt.xlabel(title.split("(")[-1].replace(")", ""))
        plt.ylabel("Framework")
        plt.legend(title="Group", loc="lower right")
        plt.tight_layout()
        plt.savefig(analyzer.plots_dir / filename)
        plt.close()

    # 2. Resource & Efficiency Plots (if available)
    if not analyzer.summary_df.empty:
        # Get mean resource usage per framework
        resource_means = (
            analyzer.summary_df.groupby(["server_type", "metric"])["mean"].mean().unstack()
        )

        # Merge with wrk_summary
        merged = analyzer.wrk_summary.merge(resource_means, on="server_type", how="left")

        # Efficiency: RPS / CPU %
        if "cpu" in merged.columns:
            merged["efficiency"] = merged["rps_mean"] / merged["cpu"].replace(0, np.nan)

        resource_metrics = {
            "cpu": ("Mean CPU Usage (%)", "capacity_wrk_cpu_comparison.png"),
            "memory": ("Mean Memory Usage (MB)", "capacity_wrk_memory_comparison.png"),
        }
        if "efficiency" in merged.columns:
            resource_metrics["efficiency"] = (
                "Resource Efficiency (RPS / CPU %)",
                "capacity_wrk_efficiency_comparison.png",
            )

        for col, (title, filename) in resource_metrics.items():
            if col not in merged.columns:
                continue
            plt.figure(figsize=(12, 8))
            sorted_merged = merged.sort_values(col, ascending=(col != "efficiency"))
            sns.barplot(
                data=sorted_merged,
                y="server_type",
                x=col,
                hue="group",
                palette=PLOT_PALETTE,
                dodge=False,
            )
            plt.title(title)
            plt.xlabel(title.split("(")[-1].replace(")", ""))
            plt.ylabel("Framework")
            plt.legend(title="Group", loc="lower right")
            plt.tight_layout()
            plt.savefig(analyzer.plots_dir / filename)
            plt.close()

        # 3. Timeseries (Resource usage over time)
        for metric, filename in {"cpu": "cpu_timeseries", "memory": "ram_timeseries"}.items():
            m_df = analyzer.raw_df[analyzer.raw_df["metric"] == metric]
            if not m_df.empty:
                plt.figure(figsize=(12, 7))
                sns.lineplot(
                    data=m_df, x="time_sec", y="metric_value", hue="server_type", errorbar="sd"
                )
                plt.title(f"WRK Test: {metric.upper()} Over Time")
                plt.xlabel("Time (seconds)")
                plt.ylabel(metric.upper())
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(analyzer.plots_dir / f"capacity_wrk_{filename}.png")
                plt.close()
