from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import pandas as pd
from scipy import stats

from orchestrator.shared.research import Column

from ..config import METRIC_CONFIG
from ..models import ChampionResult
from ..utils.reporting import write_report
from ..utils.stats import cohen_d

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_champions_analysis(analyzer: PerformanceAnalyzer) -> None:
    if not analyzer.experiment or analyzer.experiment.metrics.empty:
        return
    metrics_df = analyzer.experiment.metrics
    # Prepare local state
    champion_results: Dict[str, ChampionResult] = {}

    # Compute summary_df locally
    summary_df = (
        metrics_df.groupby(
            [Column.GROUP, Column.SERVER_TYPE, Column.RUN_NUMBER, Column.METRIC]
        )[Column.VALUE]
        .agg(["mean", "std", lambda x: x.quantile(0.99)])
        .reset_index()
    )
    summary_df.rename(columns={"<lambda_0>": "p99"}, inplace=True)

    compute_champion_stats(analyzer, summary_df, champion_results)
    content = generate_champions_report(analyzer, champion_results)
    write_report(analyzer, content)


def compute_champion_stats(
    analyzer: PerformanceAnalyzer,
    summary_df: pd.DataFrame,
    champion_results: Dict[str, ChampionResult],
) -> None:
    if len(analyzer.champions_list) != 2:
        return
    c1, c2 = analyzer.champions_list
    for _stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            m_df = summary_df[summary_df[Column.METRIC] == metric]
            g1 = m_df[m_df[Column.SERVER_TYPE] == c1]["mean"].dropna()
            g2 = m_df[m_df[Column.SERVER_TYPE] == c2]["mean"].dropna()
            res: ChampionResult = {"name": str(config["name"]), "champ1": c1, "champ2": c2}
            if len(g1) > 1 and len(g2) > 1:
                test = (
                    stats.ttest_ind(g1, g2, equal_var=False)
                    if stats.shapiro(g1).pvalue > 0.05
                    else stats.mannwhitneyu(g1, g2)
                )
                res.update({"p_value": float(test.pvalue), "cohen_d": cohen_d(g1, g2)})
            champion_results[str(config["name"])] = res


def generate_champions_report(
    analyzer: PerformanceAnalyzer, champion_results: Dict[str, ChampionResult]
) -> str:
    # Basic implementation
    return f"# Champions Comparison: {analyzer.champions_list}\n\n" + str(champion_results)
