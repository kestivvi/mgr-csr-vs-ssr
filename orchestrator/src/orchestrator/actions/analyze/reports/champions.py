from __future__ import annotations

from typing import TYPE_CHECKING

from scipy import stats

from ..config import METRIC_CONFIG
from ..models import ChampionResult
from ..utils.reporting import write_report
from ..utils.stats import cohen_d

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def run_champions_analysis(analyzer: PerformanceAnalyzer) -> None:
    compute_champion_stats(analyzer)
    content = generate_champions_report(analyzer)
    write_report(analyzer, content)


def compute_champion_stats(analyzer: PerformanceAnalyzer) -> None:
    if len(analyzer.champions_list) != 2:
        return
    c1, c2 = analyzer.champions_list
    for _stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            m_df = analyzer.summary_df[analyzer.summary_df["metric"] == metric]
            g1 = m_df[m_df["server_type"] == c1]["mean"].dropna()
            g2 = m_df[m_df["server_type"] == c2]["mean"].dropna()
            res: ChampionResult = {"name": str(config["name"]), "champ1": c1, "champ2": c2}
            if len(g1) > 1 and len(g2) > 1:
                test = (
                    stats.ttest_ind(g1, g2, equal_var=False)
                    if stats.shapiro(g1).pvalue > 0.05
                    else stats.mannwhitneyu(g1, g2)
                )
                res.update({"p_value": float(test.pvalue), "cohen_d": cohen_d(g1, g2)})
            analyzer.champion_results[str(config["name"])] = res


def generate_champions_report(analyzer: PerformanceAnalyzer) -> str:
    # Basic implementation as seen in engine.py
    return "# Champions Comparison"
