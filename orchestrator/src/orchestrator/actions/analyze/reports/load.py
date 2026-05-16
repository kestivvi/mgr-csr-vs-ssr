from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Dict

import pandas as pd

from orchestrator.shared.research import Column

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
    if not analyzer.experiment or analyzer.experiment.metrics.empty:
        return
    metrics_df = analyzer.experiment.metrics
    # Prepare session state for rankings
    ranking_results: Dict[str, pd.DataFrame] = {}

    # Compute summary_df locally (Mean/Std/P95 per tech/run/metric)
    summary_df = (
        metrics_df.groupby([Column.GROUP, Column.SERVER_TYPE, Column.RUN_NUMBER, Column.METRIC])[
            Column.VALUE
        ]
        .agg(["mean", "std", lambda x: x.quantile(0.99)])
        .reset_index()
    )
    summary_df.rename(columns={"<lambda_0>": "p99"}, inplace=True)

    compute_rankings(analyzer, summary_df, ranking_results)
    scorecard_ranks_df, scorecard_values_df, executive_summary = compute_scorecard_and_winner(
        analyzer, ranking_results
    )

    generate_load_plots(analyzer, summary_df, ranking_results, scorecard_ranks_df)

    # Render report
    content = generate_load_report(
        analyzer, ranking_results, scorecard_ranks_df, scorecard_values_df, executive_summary
    )
    write_report(analyzer, content)


def compute_rankings(
    analyzer: PerformanceAnalyzer,
    summary_df: pd.DataFrame,
    ranking_results: Dict[str, pd.DataFrame],
) -> None:
    for stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            m_df = summary_df[summary_df[Column.METRIC] == metric]
            if m_df.empty:
                continue
            agg = (
                m_df.groupby([Column.GROUP, Column.SERVER_TYPE])[stat_col]
                .agg(["mean", calculate_confidence_interval])
                .reset_index()
            )
            agg[["ci_lower", "ci_upper"]] = pd.DataFrame(
                agg["calculate_confidence_interval"].tolist(), index=agg.index
            )
            ranking_results[str(config["name"])] = agg.drop(
                columns=["calculate_confidence_interval"]
            )


def compute_scorecard_and_winner(
    analyzer: PerformanceAnalyzer, ranking_results: Dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    all_ranks = {}
    all_values = {}
    for name, df in ranking_results.items():
        sort_asc = True
        for sm in METRIC_CONFIG.values():
            for mc in sm.values():
                if mc["name"] == name:
                    sort_asc = bool(mc["sort_ascending"])
                    break
        ranked = df.sort_values("mean", ascending=sort_asc).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        all_ranks[name] = ranked.set_index(Column.SERVER_TYPE)["rank"]
        all_values[name] = ranked.set_index(Column.SERVER_TYPE)["mean"]

    scorecard_ranks_df = pd.DataFrame(all_ranks).transpose()
    scorecard_values_df = pd.DataFrame(all_values).transpose()

    if scorecard_ranks_df.empty:
        return (
            scorecard_ranks_df,
            scorecard_values_df,
            "Could not generate a summary as no ranking data was available.",
        )

    first_place_ranks = scorecard_ranks_df[scorecard_ranks_df == 1].count()
    if first_place_ranks.sum() == 0:
        summary_text = (
            "No technology achieved a #1 rank in any category, "
            "so no overall winner could be determined."
        )
    else:
        winner_counts = Counter(first_place_ranks[first_place_ranks > 0].to_dict())
        winners = winner_counts.most_common()

        num_metrics = len(scorecard_ranks_df)
        num_runs = (
            analyzer.experiment.metadata.get("num_runs", "multiple")
            if analyzer.experiment
            else "multiple"
        )

        if len(winners) > 0 and (len(winners) == 1 or winners[0][1] > winners[1][1]):
            winner_tech, win_count = winners[0]
            summary_text = (
                f"Based on an analysis of **{num_metrics} key metrics** across "
                f"**{num_runs} runs**, **`{winner_tech}`** emerges as the top overall "
                f"performer, achieving the #1 rank in **{win_count} categories**. "
                "The performance scorecard below provides a detailed breakdown "
                "of all technologies."
            )
        else:
            top_contenders = [tech for tech, count in winners if count == winners[0][1]]
            contender_str = "`, `".join(top_contenders)
            summary_text = (
                f"The analysis of **{num_metrics} key metrics** across **{num_runs} runs** "
                "did not yield a single clear winner. Several technologies showed "
                "top-tier performance in different areas, with "
                f"**`{contender_str}`** leading in an equal number of categories. "
                "This suggests a performance trade-off, which can be explored in "
                "the detailed scorecard and analysis below."
            )

    return scorecard_ranks_df, scorecard_values_df, summary_text


def generate_load_plots(
    analyzer: PerformanceAnalyzer,
    summary_df: pd.DataFrame,
    ranking_results: Dict[str, pd.DataFrame],
    scorecard_ranks_df: pd.DataFrame,
) -> None:
    # Heatmap needs the ranks
    create_scorecard_heatmap(analyzer, scorecard_ranks_df)

    for stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            m_df = summary_df[summary_df[Column.METRIC] == metric]
            if m_df.empty:
                continue
            create_comparison_plot(
                analyzer, m_df, stat_col, str(config["name"]), plot_type="violin"
            )

    for metric, config in METRIC_CONFIG["mean"].items():
        for group in summary_df[Column.GROUP].unique():
            create_timeseries_plot(analyzer, metric, str(config["name"]), group_filter=group)


def generate_load_report(
    analyzer: PerformanceAnalyzer,
    ranking_results: Dict[str, pd.DataFrame],
    scorecard_ranks_df: pd.DataFrame,
    scorecard_values_df: pd.DataFrame,
    executive_summary: str,
) -> str:
    report = [f"# Performance Analysis Report for `{analyzer.input_dir.name}`"]
    report.append(render_executive_summary_md(analyzer, executive_summary))
    report.append("\n## Detailed Analysis")
    report.append(render_ranking_tables_md(ranking_results))
    report.append(render_visual_overview_md(analyzer))
    report.append(render_temporal_analysis_md(analyzer))
    report.append(render_metadata_md(analyzer))
    return "\n".join(report)


def render_executive_summary_md(analyzer: PerformanceAnalyzer, executive_summary: str) -> str:
    md = ["\n## Executive Summary", executive_summary]
    path = analyzer.plots_dir / "performance_scorecard.png"
    if path.exists():
        md.append("\n### Performance Scorecard")
        md.append("![Performance Scorecard](./plots/performance_scorecard.png)")
    return "\n".join(md)


def render_ranking_tables_md(ranking_results: Dict[str, pd.DataFrame]) -> str:
    md = ["\n### Intra-Group Rankings"]
    stat_name_map = {"mean": "Mean", "std": "Mean of Std Devs", "p99": "Mean of p99s"}
    emoji_map = {1: "🥇", 2: "🥈", 3: "🥉"}

    for stat_col, metrics in METRIC_CONFIG.items():
        for _metric, config in metrics.items():
            name = str(config["name"])
            if name not in ranking_results:
                continue
            ranking_df = ranking_results[name]
            md.append(f"#### {name}")

            for group in sorted(ranking_df[Column.GROUP].unique()):
                md.append(f"\n##### Group: {group}\n")
                group_data = (
                    ranking_df[ranking_df[Column.GROUP] == group]
                    .sort_values(by="mean", ascending=config["sort_ascending"])
                    .reset_index(drop=True)
                )

                rows = []
                for i, row in group_data.iterrows():
                    tech_label = f"{row[Column.SERVER_TYPE]} {emoji_map.get(i + 1, '')}".strip()
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
    if not analyzer.experiment:
        return ""
    groups = sorted(analyzer.experiment.metrics[Column.GROUP].unique())
    for group in groups:
        md.append(f"#### Group: {group}")
        for _metric, config in METRIC_CONFIG["mean"].items():
            name = str(config["name"])
            group_suffix = f"_{group.lower().replace('-', '_')}"
            filename = f"{name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
            if (analyzer.plots_dir / filename).exists():
                md.append(f"![{name}](./plots/{filename})")
    return "\n".join(md)
