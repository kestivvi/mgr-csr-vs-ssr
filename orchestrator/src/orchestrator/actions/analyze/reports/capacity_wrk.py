from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from orchestrator.shared.research import Column, MetricName

from ..config import PLOT_PALETTE
from ..utils.group_summary import (
    MetricSpec,
    load_family_map,
    render_group_summary_section,
    render_per_family_group_summary_section,
)
from ..utils.plotting import get_ordered_tech_list
from ..utils.reporting import write_report

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_capacity_wrk_analysis(analyzer: PerformanceAnalyzer) -> None:
    if (
        not analyzer.experiment
        or analyzer.experiment.wrk_results is None
        or analyzer.experiment.wrk_results.empty
    ):
        return

    wrk_df = analyzer.experiment.wrk_results
    # Compute summary locally
    wrk_summary = (
        wrk_df.groupby([Column.GROUP, Column.SERVER_TYPE])
        .agg({"rps": ["mean", "std", "max"], "latency_ms": ["mean", "std", "max"]})
        .reset_index()
    )
    wrk_summary.columns = [
        Column.GROUP,
        Column.SERVER_TYPE,
        "rps_mean",
        "rps_std",
        "rps_max",
        "lat_mean",
        "lat_std",
        "lat_max",
    ]

    generate_capacity_wrk_plots(analyzer, wrk_summary)

    group_summary_md = _render_wrk_group_summary_md(wrk_summary)
    parts = ["# Capacity WRK Report (DIRTY TEST)"]
    if group_summary_md:
        parts.append(group_summary_md)
    parts.append(wrk_summary.to_markdown())
    write_report(analyzer, "\n\n".join(parts))


def _render_wrk_group_summary_md(wrk_summary: pd.DataFrame) -> str:
    per_app_values: dict[str, dict[str, list[float]]] = {}
    per_app_values_by_family: dict[str, dict[str, dict[str, list[float]]]] = {}
    family_map = load_family_map()
    metric_specs: list[MetricSpec] = []
    for col, name, unit, decimals, higher_is in [
        ("rps_mean", "Utrzymany RPS", "RPS", 0, "CSR"),
        ("rps_max", "Szczytowy RPS", "RPS", 0, "CSR"),
    ]:
        groups: dict[str, list[float]] = {}
        for group, sub in wrk_summary.groupby(Column.GROUP):
            groups[str(group)] = [float(v) for v in sub[col].tolist()]
        for _, row in wrk_summary.iterrows():
            family = family_map.get(str(row[Column.SERVER_TYPE]))
            if family:
                per_app_values_by_family.setdefault(family, {}).setdefault(name, {}).setdefault(
                    str(row[Column.GROUP]), []
                ).append(float(row[col]))
        if "CSR" not in groups or "SSR" not in groups:
            continue
        per_app_values[name] = groups
        metric_specs.append(
            MetricSpec(name=name, unit=unit, decimals=decimals, higher_is=higher_is)  # type: ignore[arg-type]
        )
    if not metric_specs:
        return ""
    overall = render_group_summary_section(per_app_values, metric_specs)
    per_family = render_per_family_group_summary_section(per_app_values_by_family, metric_specs)
    return overall + ("\n\n" + per_family if per_family else "")


def generate_capacity_wrk_plots(analyzer: PerformanceAnalyzer, wrk_summary: pd.DataFrame) -> None:
    # 1. Base WRK Plots (RPS, Latency)
    metrics = {
        "rps_mean": ("Maximum Throughput (RPS)", "capacity_wrk_rps_comparison.png"),
        "lat_mean": ("Average Latency (ms)", "capacity_wrk_latency_comparison.png"),
    }

    full_order = get_ordered_tech_list(analyzer, wrk_summary)
    for col, (title, filename) in metrics.items():
        plt.figure(figsize=(12, 8))
        sorted_df = wrk_summary.copy()
        display_name_map = (
            sorted_df.set_index(Column.SERVER_TYPE).index.to_series().to_dict()
        )  # Use server_type as display for now
        order = [t for t in full_order if t in display_name_map]
        sns.barplot(
            data=sorted_df,
            y=Column.SERVER_TYPE,
            x=col,
            hue=Column.GROUP,
            palette=PLOT_PALETTE,
            dodge=False,
            order=order,
        )
        plt.title(title)
        plt.xlabel(title.split("(")[-1].replace(")", ""))
        plt.ylabel("Framework")

        # Color y-axis labels by group
        ax = plt.gca()
        for label in ax.get_yticklabels():
            tech = label.get_text()
            match = sorted_df[sorted_df[Column.SERVER_TYPE] == tech]
            if not match.empty:
                group = match[Column.GROUP].iloc[0]
                label.set_color(PLOT_PALETTE.get(group, "black"))
                label.set_fontweight("bold")

        plt.legend(title="Group", loc="lower right")
        plt.tight_layout()
        plt.savefig(analyzer.plots_dir / filename)
        plt.close()

    # 2. Resource & Efficiency Plots (if available)
    if analyzer.experiment and not analyzer.experiment.metrics.empty:
        # Get mean resource usage per framework
        metrics_df = analyzer.experiment.metrics
        resource_means = (
            metrics_df.groupby([Column.SERVER_TYPE, Column.METRIC])[Column.VALUE].mean().unstack()
        )

        # Merge with wrk_summary
        merged = wrk_summary.merge(resource_means, on=Column.SERVER_TYPE, how="left")

        # Efficiency: RPS / CPU %
        if MetricName.CPU in merged.columns:
            merged["efficiency"] = merged["rps_mean"] / merged[MetricName.CPU].replace(0, np.nan)

        resource_metrics = {
            MetricName.CPU: ("Mean CPU Usage (%)", "capacity_wrk_cpu_comparison.png"),
            MetricName.MEMORY: ("Mean Memory Usage (MB)", "capacity_wrk_memory_comparison.png"),
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
            sorted_merged = merged.copy()
            display_name_map = (
                sorted_merged.set_index(Column.SERVER_TYPE).index.to_series().to_dict()
            )
            order = [t for t in full_order if t in display_name_map]
            sns.barplot(
                data=sorted_merged,
                y=Column.SERVER_TYPE,
                x=col,
                hue=Column.GROUP,
                palette=PLOT_PALETTE,
                dodge=False,
                order=order,
            )
            plt.title(title)
            plt.xlabel(title.split("(")[-1].replace(")", ""))
            plt.ylabel("Framework")

            # Color y-axis labels by group
            ax = plt.gca()
            for label in ax.get_yticklabels():
                tech = label.get_text()
                match = sorted_merged[sorted_merged[Column.SERVER_TYPE] == tech]
                if not match.empty:
                    group = match[Column.GROUP].iloc[0]
                    label.set_color(PLOT_PALETTE.get(group, "black"))
                    label.set_fontweight("bold")

            plt.legend(title="Group", loc="lower right")
            plt.tight_layout()
            plt.savefig(analyzer.plots_dir / filename)
            plt.close()

        # 3. Timeseries (Resource usage over time)
        for metric, filename in {
            MetricName.CPU: "cpu_timeseries",
            MetricName.MEMORY: "ram_timeseries",
        }.items():
            m_df = metrics_df[metrics_df[Column.METRIC] == metric]
            if not m_df.empty:
                plt.figure(figsize=(12, 7))
                sns.lineplot(
                    data=m_df,
                    x=Column.TIME_SEC,
                    y=Column.VALUE,
                    hue=Column.SERVER_TYPE,
                    errorbar="sd",
                )
                plt.title(f"WRK Test: {metric.upper()} Over Time")
                plt.xlabel("Time (seconds)")
                plt.ylabel(metric.upper())
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(analyzer.plots_dir / f"capacity_wrk_{filename}.png")
                plt.close()
