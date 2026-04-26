import glob
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from scipy import stats

console = Console()

# --- CONFIGURATION (Global) ---
METRIC_CONFIG: Dict[str, Dict[str, Dict[str, Union[str, bool]]]] = {
    "mean": {
        "cpu": {"name": "Mean CPU Usage (%)", "sort_ascending": True},
        "memory": {"name": "Mean Memory Usage (MB)", "sort_ascending": True},
        "latency": {"name": "Mean p95 Latency (ms)", "sort_ascending": True},
        "network_tx": {"name": "Mean Network Transmit Rate (MB/s)", "sort_ascending": True},
    },
    "std": {
        "cpu": {"name": "CPU Usage Stability (Std Dev)", "sort_ascending": True},
        "memory": {"name": "Memory Usage Stability (Std Dev)", "sort_ascending": True},
        "latency": {"name": "Latency Stability (Std Dev)", "sort_ascending": True},
        "network_tx": {"name": "Network Transmit Stability (Std Dev)", "sort_ascending": True},
    },
    "p95": {
        "cpu": {"name": "Peak CPU Usage (95th Percentile)", "sort_ascending": True},
        "memory": {"name": "Peak Memory Usage (95th Percentile)", "sort_ascending": True},
        "latency": {"name": "Peak Latency (95th Percentile)", "sort_ascending": True},
        "network_tx": {
            "name": "Peak Network Transmit Rate (95th Percentile) (MB/s)",
            "sort_ascending": True,
        },
    },
}

# --- STATISTICAL HELPERS ---


def calculate_confidence_interval(data: pd.Series) -> Tuple[float, float]:
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(data))
    std_err = stats.sem(data)
    if std_err == 0 or np.isnan(std_err):
        return (mean, mean)
    interval = stats.t.interval(0.95, df=n - 1, loc=mean, scale=std_err)
    return (max(0.0, float(interval[0])), float(interval[1]))


def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    n1, n2 = len(group1), len(group2)
    s1, s2 = float(np.var(group1, ddof=1)), float(np.var(group2, ddof=1))
    if (n1 + n2 - 2) == 0:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


class PerformanceAnalyzer:
    def __init__(self, input_dir: Path, report_type: str, champions: Optional[List[str]] = None):
        self.input_dir = input_dir
        self.report_type = report_type
        self.champions_list = champions or []
        self.plots_dir = self.input_dir / "plots"

        report_names = {
            "capacity": "capacity_report.md",
            "capacity_wrk": "capacity_report_wrk.md",
            "champions": "champions_report.md",
        }
        self.report_path = self.input_dir / report_names.get(
            self.report_type, f"report_{self.report_type}.md"
        )

        self.metadata: Dict[str, Any] = {}
        self.groups_config: Dict[str, List[str]] = {}
        self.chart_order: List[str] = []
        self.raw_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.ranking_results: Dict[str, pd.DataFrame] = {}
        self.champion_results: Dict[str, Dict[str, Any]] = {}
        self.scorecard_ranks_df = pd.DataFrame()
        self.scorecard_values_df = pd.DataFrame()
        self.executive_summary_text = ""
        self.wrk_df = pd.DataFrame()
        self.wrk_summary = pd.DataFrame()

    def run(self) -> None:
        console.print(f"[bold cyan]Starting analysis for:[/bold cyan] {self.input_dir}")
        if not self._load_configuration():
            return

        self.plots_dir.mkdir(exist_ok=True)

        if self.report_type == "capacity_wrk":
            if not self._load_wrk_data():
                return
            self._run_capacity_wrk_analysis()
        else:
            if not self._load_and_prepare_data():
                return
            if self.report_type == "capacity":
                self._run_capacity_analysis_pipeline()
            elif self.report_type == "load":
                self._run_load_analysis()
            elif self.report_type == "champions":
                self._run_champions_analysis()

        console.print(
            f"[bold green]Analysis complete. Report saved to: {self.report_path}[/bold green]"
        )

    def _load_configuration(self) -> bool:
        metadata_path = self.input_dir / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = yaml.safe_load(f)

        config_path = Path(__file__).parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                self.groups_config = config_data.get("groups", {})
                self.chart_order = config_data.get("chart_order", [])
            shutil.copy(config_path, self.input_dir / "config.yaml")
            return True
        except Exception as e:
            console.print(f"[bold red]Failed to load configuration:[/bold red] {e}")
            return False

    def _load_and_prepare_data(self) -> bool:
        metrics_dir = self.input_dir / "metrics"
        if not metrics_dir.is_dir():
            return False
        all_files = glob.glob(str(metrics_dir / "*.csv"))
        if not all_files:
            return False

        tech_to_group = {
            tech.lower(): group for group, techs in self.groups_config.items() for tech in techs
        }
        all_long_dfs = []
        filename_regex = re.compile(r"^(\d+)_(.*)$")

        for f in all_files:
            p = Path(f)
            match = filename_regex.match(p.stem)
            if not match:
                continue
            run_num, tech_raw = match.groups()

            wide_df = pd.read_csv(f)
            long_df = pd.melt(
                wide_df, id_vars=["timestamp"], var_name="metric", value_name="metric_value"
            )
            long_df["run_number"] = int(run_num)
            tech = tech_raw.replace("_", "-")
            long_df["server_type"] = tech
            long_df["group"] = (
                long_df["server_type"].str.lower().map(tech_to_group).fillna("Uncategorized")
            )
            all_long_dfs.append(long_df)

        self.raw_df = pd.concat(all_long_dfs, ignore_index=True)
        self.raw_df.loc[self.raw_df["metric"] == "cpu", "metric_value"] *= 100
        self.raw_df.loc[self.raw_df["metric"] == "memory", "metric_value"] /= 1024 * 1024
        self.raw_df.loc[self.raw_df["metric"] == "network_tx", "metric_value"] /= 1024 * 1024

        self.raw_df["timestamp"] = pd.to_datetime(self.raw_df["timestamp"])
        self.raw_df["time_sec"] = self.raw_df.groupby(["server_type", "run_number", "metric"])[
            "timestamp"
        ].transform(lambda x: (x - x.min()).dt.total_seconds())

        self.summary_df = (
            self.raw_df.groupby(["group", "server_type", "run_number", "metric"])["metric_value"]
            .agg(["mean", "std", lambda x: x.quantile(0.95)])
            .reset_index()
        )
        self.summary_df.rename(columns={"<lambda_0>": "p95"}, inplace=True)
        return True

    def _load_wrk_data(self) -> bool:
        import json

        all_files = glob.glob(str(self.input_dir / "*_wrk_client_results.json"))
        if not all_files:
            return False
        tech_to_group = {
            tech.lower(): group for group, techs in self.groups_config.items() for tech in techs
        }
        records = []
        for f in all_files:
            run_num = int(Path(f).stem.split("_")[0])
            with open(f, "r") as jf:
                data = json.load(jf)
                for tech, res in data.items():
                    lat_str = res["latency_avg"]
                    lat_match = re.search(r"[\d.]+", lat_str)
                    lat_val = float(lat_match.group()) if lat_match else 0.0
                    if "us" in lat_str:
                        lat_val /= 1000
                    elif "s" in lat_str and "ms" not in lat_str:
                        lat_val *= 1000
                    records.append(
                        {
                            "run_number": run_num,
                            "server_type": tech,
                            "group": tech_to_group.get(tech.lower(), "Uncategorized"),
                            "rps": float(res["rps"]),
                            "latency_ms": lat_val,
                        }
                    )
        self.wrk_df = pd.DataFrame(records)
        self.wrk_summary = (
            self.wrk_df.groupby(["group", "server_type"])
            .agg({"rps": ["mean", "std", "max"], "latency_ms": ["mean", "std", "max"]})
            .reset_index()
        )
        self.wrk_summary.columns = [
            "group",
            "server_type",
            "rps_mean",
            "rps_std",
            "rps_max",
            "lat_mean",
            "lat_std",
            "lat_max",
        ]
        return True

    def _run_load_analysis(self) -> None:
        self._compute_rankings()
        self._compute_scorecard_and_winner()
        content = self._generate_load_report()
        self._write_report(content)

    def _run_capacity_analysis_pipeline(self) -> None:
        summary = self._compute_capacity_metrics()
        if summary is not None:
            content = self._generate_capacity_report(summary)
            self._write_report(content)

    def _run_capacity_wrk_analysis(self) -> None:
        content = f"# Capacity WRK Report\n\n{self.wrk_summary.to_markdown()}"
        self._write_report(content)

    def _run_champions_analysis(self) -> None:
        self._compute_champion_stats()
        content = self._generate_champions_report()
        self._write_report(content)

    def _compute_capacity_metrics(self) -> Optional[pd.DataFrame]:
        required = ["k6_successful_html_reqs_rate", "k6_total_html_reqs_rate", "cpu", "memory"]
        available = self.raw_df["metric"].unique()
        if not all(m in available for m in required):
            return None

        results = []
        for (tech, run), run_df in self.raw_df.groupby(["server_type", "run_number"]):
            pivot = (
                run_df.pivot_table(index="time_sec", columns="metric", values="metric_value")
                .reindex(columns=required)
                .fillna(0)
            )
            rolling_mins = (
                pivot["k6_successful_html_reqs_rate"].rolling(window=30, min_periods=1).min()
            )
            sustained_rps = float(rolling_mins.max())
            peak_rps = float(pivot["k6_successful_html_reqs_rate"].max())
            sustained_time = rolling_mins.idxmax()
            cpu_at = (
                float(pivot.loc[sustained_time, "cpu"]) if sustained_time in pivot.index else 0.0
            )
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
            pd.DataFrame(results)
            .groupby("server_type")
            .mean()
            .drop(columns="run_number")
            .reset_index()
        )

    def _compute_rankings(self) -> None:
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                m_df = self.summary_df[self.summary_df["metric"] == metric]
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
                self.ranking_results[str(config["name"])] = agg.drop(
                    columns=["calculate_confidence_interval"]
                )

    def _compute_scorecard_and_winner(self) -> None:
        all_ranks: Dict[str, pd.Series] = {}
        all_values: Dict[str, pd.Series] = {}
        for name, df in self.ranking_results.items():
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
        self.scorecard_ranks_df = pd.DataFrame(all_ranks).transpose()
        self.scorecard_values_df = pd.DataFrame(all_values).transpose()

        if self.scorecard_ranks_df.empty:
            return
        wins = Counter(self.scorecard_ranks_df[self.scorecard_ranks_df == 1].count().to_dict())
        top = wins.most_common()
        if top:
            self.executive_summary_text = (
                f"**{top[0][0]}** is the winner with {top[0][1]} #1 ranks."
            )

    def _compute_champion_stats(self) -> None:
        if len(self.champions_list) != 2:
            return
        c1, c2 = self.champions_list
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                m_df = self.summary_df[self.summary_df["metric"] == metric]
                g1 = m_df[m_df["server_type"] == c1][stat_col].dropna()
                g2 = m_df[m_df["server_type"] == c2][stat_col].dropna()
                res: Dict[str, Any] = {"name": str(config["name"]), "champ1": c1, "champ2": c2}
                if len(g1) > 1 and len(g2) > 1:
                    test = (
                        stats.ttest_ind(g1, g2, equal_var=False)
                        if stats.shapiro(g1).pvalue > 0.05
                        else stats.mannwhitneyu(g1, g2)
                    )
                    res.update({"p_value": float(test.pvalue), "cohen_d": cohen_d(g1, g2)})
                self.champion_results[str(config["name"])] = res

    # --- REPORT GENERATORS ---

    def _generate_load_report(self) -> str:
        report = [f"# Load Test Report for `{self.input_dir.name}`", self.executive_summary_text]
        return "\n".join(report)

    def _generate_capacity_report(self, summary: pd.DataFrame) -> str:
        report = [f"# Capacity Report for `{self.input_dir.name}`", summary.to_markdown()]
        return "\n".join(report)

    def _generate_champions_report(self) -> str:
        return "# Champions Comparison"

    def _write_report(self, content: str) -> None:
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(content)
