from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pandas as pd

from ..config import METRIC_CONFIG
from ..utils.plotting import (
    create_comparison_plot,
    create_scorecard_heatmap,
    create_timeseries_plot,
)
from ..utils.reporting import render_metadata_md, write_report
from ..utils.stats import calculate_confidence_interval

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_load_analysis(analyzer: PerformanceAnalyzer) -> None:
    compute_rankings(analyzer)
    compute_scorecard_and_winner(analyzer)
    generate_load_plots(analyzer)
    content = generate_load_report(analyzer)
    write_report(analyzer, content)


def compute_rankings(analyzer: PerformanceAnalyzer) -> None:
    for stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            m_df = analyzer.summary_df[analyzer.summary_df["metric"] == metric]
            if m_df.empty:
                continue
            agg = (
                m_df.groupby(["group", "server_type"])[stat_col]
                .agg(["mean", calculate_confidence_interval])
                .reset_index()
            )
            agg[["ci_lower", "ci_upper"]] = pd.DataFrame(
                agg["calculate_confidence_interval"].tolist(), index=agg.index
            )
            analyzer.ranking_results[str(config["name"])] = agg.drop(
                columns=["calculate_confidence_interval"]
            )


def compute_scorecard_and_winner(analyzer: PerformanceAnalyzer) -> None:
    all_ranks = {}
    all_values = {}
    for name, df in analyzer.ranking_results.items():
        sort_asc = True
        for sm in METRIC_CONFIG.values():
            for mc in sm.values():
                if mc["name"] == name:
                    sort_asc = bool(mc["sort_ascending"])
                    break
        ranked = df.sort_values("mean", ascending=sort_asc).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        all_ranks[name] = ranked.set_index("server_type")["rank"]
        all_values[name] = ranked.set_index("server_type")["mean"]
    analyzer.scorecard_ranks_df = pd.DataFrame(all_ranks).transpose()
    analyzer.scorecard_values_df = pd.DataFrame(all_values).transpose()

    if analyzer.scorecard_ranks_df.empty:
        analyzer.executive_summary_text = (
            "Could not generate a summary as no ranking data was available."
        )
        return

    first_place_ranks = analyzer.scorecard_ranks_df[analyzer.scorecard_ranks_df == 1].count()
    if first_place_ranks.sum() == 0:
        analyzer.executive_summary_text = (
            "No technology achieved a #1 rank in any category, "
            "so no overall winner could be determined."
        )
        return

    winner_counts = Counter(first_place_ranks[first_place_ranks > 0].to_dict())
    winners = winner_counts.most_common()

    num_metrics = len(analyzer.scorecard_ranks_df)
    num_runs = analyzer.metadata.get("parameters", {}).get("num_runs", "multiple")

    if len(winners) > 0 and (len(winners) == 1 or winners[0][1] > winners[1][1]):
        winner_tech, win_count = winners[0]
        analyzer.executive_summary_text = (
            f"Based on an analysis of **{num_metrics} key metrics** across "
            f"**{num_runs} runs**, **`{winner_tech}`** emerges as the top overall "
            f"performer, achieving the #1 rank in **{win_count} categories**. "
            "The performance scorecard below provides a detailed breakdown "
            "of all technologies."
        )
    else:
        top_contenders = [tech for tech, count in winners if count == winners[0][1]]
        contender_str = "`, `".join(top_contenders)
        analyzer.executive_summary_text = (
            f"The analysis of **{num_metrics} key metrics** across **{num_runs} runs** "
            "did not yield a single clear winner. Several technologies showed "
            "top-tier performance in different areas, with "
            f"**`{contender_str}`** leading in an equal number of categories. "
            "This suggests a performance trade-off, which can be explored in "
            "the detailed scorecard and analysis below."
        )


def generate_load_plots(analyzer: PerformanceAnalyzer) -> None:
    create_scorecard_heatmap(analyzer)

    for stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            m_df = analyzer.summary_df[analyzer.summary_df["metric"] == metric]
            if m_df.empty:
                continue
            create_comparison_plot(
                analyzer, m_df, stat_col, str(config["name"]), plot_type="violin"
            )

    for metric, config in METRIC_CONFIG["mean"].items():
        for group in analyzer.raw_df["group"].unique():
            create_timeseries_plot(analyzer, metric, str(config["name"]), group_filter=group)


def generate_load_report(analyzer: PerformanceAnalyzer) -> str:
    report = [f"# Performance Analysis Report for `{analyzer.input_dir.name}`"]
    report.append(render_executive_summary_md(analyzer))
    report.append("\n## Detailed Analysis")
    report.append(render_ranking_tables_md(analyzer))
    report.append(render_visual_overview_md(analyzer))
    report.append(render_temporal_analysis_md(analyzer))
    report.append(render_metadata_md(analyzer))
    return "\n".join(report)


def render_executive_summary_md(analyzer: PerformanceAnalyzer) -> str:
    md = ["\n## Executive Summary", analyzer.executive_summary_text]
    path = analyzer.plots_dir / "performance_scorecard.png"
    if path.exists():
        md.append("\n### Performance Scorecard")
        md.append("![Performance Scorecard](./plots/performance_scorecard.png)")
    return "\n".join(md)


def render_ranking_tables_md(analyzer: PerformanceAnalyzer) -> str:
    md = ["\n### Intra-Group Rankings"]
    stat_name_map = {"mean": "Mean", "std": "Mean of Std Devs", "p95": "Mean of p95s"}
    emoji_map = {1: "🥇", 2: "🥈", 3: "🥉"}

    for stat_col, metrics in METRIC_CONFIG.items():
        for _metric, config in metrics.items():
            name = str(config["name"])
            if name not in analyzer.ranking_results:
                continue
            ranking_df = analyzer.ranking_results[name]
            md.append(f"#### {name}")

            for group in sorted(ranking_df["group"].unique()):
                md.append(f"\n##### Group: {group}\n")
                group_data = (
                    ranking_df[ranking_df["group"] == group]
                    .sort_values(by="mean", ascending=config["sort_ascending"])
                    .reset_index(drop=True)
                )

                rows = []
                for i, row in group_data.iterrows():
                    tech_label = f"{row['server_type']} {emoji_map.get(i + 1, '')}".strip()
                    val_str = (
                        f"{row['mean']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                        if pd.notna(row["ci_lower"])
                        else f"{row['mean']:.4f}"
                    )
                    rows.append(
                        {
                            "Technology": tech_label,
                            stat_name_map.get(stat_col, stat_col): val_str,
                        }
                    )

                md.append(pd.DataFrame(rows).to_markdown(index=False))
    return "\n".join(md)


def render_visual_overview_md(analyzer: PerformanceAnalyzer) -> str:
    md = ["\n### Metric Distributions"]
    for _stat_col, metrics in METRIC_CONFIG.items():
        for _metric, config in metrics.items():
            name = str(config["name"])
            filename = f"{name.lower().replace(' ', '_')}_violin_comparison.png"
            if (analyzer.plots_dir / filename).exists():
                md.append(f"#### {name}")
                md.append(f"![{name}](./plots/{filename})")
    return "\n".join(md)


def render_temporal_analysis_md(analyzer: PerformanceAnalyzer) -> str:
    md = ["\n### Temporal Analysis"]
    for group in sorted(analyzer.raw_df["group"].unique()):
        md.append(f"#### Group: {group}")
        for _metric, config in METRIC_CONFIG["mean"].items():
            name = str(config["name"])
            group_suffix = f"_{group.lower().replace('-', '_')}"
            filename = f"{name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
            if (analyzer.plots_dir / filename).exists():
                md.append(f"![{name}](./plots/{filename})")
    return "\n".join(md)
