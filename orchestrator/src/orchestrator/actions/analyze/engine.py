import datetime
import glob
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from rich.console import Console
from scipy import stats

console = Console()
sns.set_theme(style="whitegrid")

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

        # Create timestamped output directory
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.input_dir / f"analysis_{ts}"
        self.plots_dir = self.output_dir / "plots"

        report_names = {
            "capacity_k6": "capacity_report_k6.md",
            "capacity_wrk": "capacity_report_wrk.md",
            "champions": "champions_report.md",
        }
        self.report_path = self.output_dir / report_names.get(
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        if not self._load_configuration():
            console.print("[bold red]Analysis aborted: Configuration loading failed.[/bold red]")
            return

        if self.report_type == "capacity_wrk":
            # Load wrk-specific data (RPS, Latency)
            if not self._load_wrk_data():
                return
            # Also load resource metrics if available
            self._load_and_prepare_data()
            self._run_capacity_wrk_analysis()
        else:
            if not self._load_and_prepare_data():
                console.print(f"[bold red]Analysis aborted: No valid metric data found in {self.input_dir / 'metrics'}.[/bold red]")
                return
            if self.report_type == "capacity_k6":
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
            shutil.copy(config_path, self.output_dir / "config.yaml")
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

        all_files = glob.glob(str(self.input_dir / "tool_results" / "*_wrk.json"))
        if not all_files:
            console.print(f"[bold red]No wrk result files found in {self.input_dir / 'tool_results'}[/bold red]")
            return False

        tech_to_group = {
            tech.lower(): group for group, techs in self.groups_config.items() for tech in techs
        }
        records = []
        for f in all_files:
            p = Path(f)
            run_num = int(p.stem.split("_")[0])
            with open(f, "r") as jf:
                res = json.load(jf)

                # New format only: filename contains tech, data is the metrics dict
                parts = p.stem.split("_")
                tech = "-".join(parts[1:-1])

                lat_str = res.get("latency_avg", "0ms")
                lat_match = re.search(r"[\d.]+", str(lat_str))
                lat_val = float(lat_match.group()) if lat_match else 0.0
                if "us" in str(lat_str):
                    lat_val /= 1000
                elif "s" in str(lat_str) and "ms" not in str(lat_str):
                    lat_val *= 1000

                records.append(
                    {
                        "run_number": run_num,
                        "server_type": tech,
                        "group": tech_to_group.get(tech.lower(), "Uncategorized"),
                        "rps": float(res.get("rps", 0.0)),
                        "latency_ms": lat_val,
                    }
                )
        if not records:
            console.print("[bold red]No valid records found in wrk result files.[/bold red]")
            return False
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
        self._generate_load_plots()
        content = self._generate_load_report()
        self._write_report(content)

    def _run_capacity_analysis_pipeline(self) -> None:
        summary = self._compute_capacity_metrics()
        if summary is not None:
            self._generate_capacity_plots(summary)
            content = self._generate_capacity_report(summary)
            self._write_report(content)

    def _run_capacity_wrk_analysis(self) -> None:
        self._generate_capacity_wrk_plots()
        content = f"# Capacity WRK Report\n\n{self.wrk_summary.to_markdown()}"
        self._write_report(content)

    def _generate_capacity_wrk_plots(self) -> None:
        if self.wrk_summary.empty:
            return

        # 1. Base WRK Plots (RPS, Latency)
        palette = {"CSR": "#1f77b4", "SSR": "#d62728", "Uncategorized": "gray"}
        metrics = {
            "rps_mean": ("Maximum Throughput (RPS)", "capacity_wrk_rps_comparison.png"),
            "lat_mean": ("Average Latency (ms)", "capacity_wrk_latency_comparison.png"),
        }

        for col, (title, filename) in metrics.items():
            plt.figure(figsize=(12, 8))
            sorted_df = self.wrk_summary.sort_values(col, ascending=(col == "lat_mean"))
            sns.barplot(data=sorted_df, y="server_type", x=col, hue="group", palette=palette, dodge=False)
            plt.title(title)
            plt.xlabel(title.split("(")[-1].replace(")", ""))
            plt.ylabel("Framework")
            plt.legend(title="Group", loc="lower right")
            plt.tight_layout()
            plt.savefig(self.plots_dir / filename)
            plt.close()

        # 2. Resource & Efficiency Plots (if available)
        if not self.summary_df.empty:
            # Get mean resource usage per framework
            resource_means = self.summary_df.groupby(["server_type", "metric"])["mean"].mean().unstack()

            # Merge with wrk_summary
            merged = self.wrk_summary.merge(resource_means, on="server_type", how="left")

            # Efficiency: RPS / CPU %
            if "cpu" in merged.columns:
                merged["efficiency"] = merged["rps_mean"] / merged["cpu"].replace(0, np.nan)

            resource_metrics = {
                "cpu": ("Mean CPU Usage (%)", "capacity_wrk_cpu_comparison.png"),
                "memory": ("Mean Memory Usage (MB)", "capacity_wrk_memory_comparison.png"),
            }
            if "efficiency" in merged.columns:
                resource_metrics["efficiency"] = ("Resource Efficiency (RPS / CPU %)", "capacity_wrk_efficiency_comparison.png")

            for col, (title, filename) in resource_metrics.items():
                if col not in merged.columns:
                    continue
                plt.figure(figsize=(12, 8))
                sorted_merged = merged.sort_values(col, ascending=(col != "efficiency"))
                sns.barplot(data=sorted_merged, y="server_type", x=col, hue="group", palette=palette, dodge=False)
                plt.title(title)
                plt.xlabel(title.split("(")[-1].replace(")", ""))
                plt.ylabel("Framework")
                plt.legend(title="Group", loc="lower right")
                plt.tight_layout()
                plt.savefig(self.plots_dir / filename)
                plt.close()

            # 3. Timeseries (Resource usage over time)
            for metric, filename in {"cpu": "cpu_timeseries", "memory": "ram_timeseries"}.items():
                m_df = self.raw_df[self.raw_df["metric"] == metric]
                if not m_df.empty:
                    plt.figure(figsize=(12, 7))
                    sns.lineplot(data=m_df, x="time_sec", y="metric_value", hue="server_type", errorbar="sd")
                    plt.title(f"WRK Test: {metric.upper()} Over Time")
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(metric.upper())
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"capacity_wrk_{filename}.png")
                    plt.close()

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
            self.executive_summary_text = "Could not generate a summary as no ranking data was available."
            return

        first_place_ranks = self.scorecard_ranks_df[self.scorecard_ranks_df == 1].count()
        if first_place_ranks.sum() == 0:
            self.executive_summary_text = "No technology achieved a #1 rank in any category, so no overall winner could be determined."
            return

        winner_counts = Counter(first_place_ranks[first_place_ranks > 0].to_dict())
        winners = winner_counts.most_common()

        num_metrics = len(self.scorecard_ranks_df)
        num_runs = self.metadata.get('parameters', {}).get('num_runs', 'multiple')

        if len(winners) > 0 and (len(winners) == 1 or winners[0][1] > winners[1][1]):
            winner_tech, win_count = winners[0]
            self.executive_summary_text = (
                f"Based on an analysis of **{num_metrics} key metrics** across **{num_runs} runs**, "
                f"**`{winner_tech}`** emerges as the top overall performer, achieving the #1 rank in **{win_count} categories**. "
                "The performance scorecard below provides a detailed breakdown of all technologies."
            )
        else:
            top_contenders = [tech for tech, count in winners if count == winners[0][1]]
            contender_str = "`, `".join(top_contenders)
            self.executive_summary_text = (
                f"The analysis of **{num_metrics} key metrics** across **{num_runs} runs** did not yield a single clear winner. "
                f"Several technologies showed top-tier performance in different areas, with **`{contender_str}`** leading in an equal number of categories. "
                "This suggests a performance trade-off, which can be explored in the detailed scorecard and analysis below."
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

    # --- PLOT GENERATORS ---

    def _generate_load_plots(self) -> None:
        # Ported logic to create all required plots for the load report
        self._create_scorecard_heatmap()

        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                m_df = self.summary_df[self.summary_df['metric'] == metric]
                if m_df.empty: continue
                self._create_comparison_plot(m_df, stat_col, str(config['name']), plot_type='violin')

        for metric, config in METRIC_CONFIG['mean'].items():
            for group in self.raw_df['group'].unique():
                self._create_timeseries_plot(metric, str(config['name']), group_filter=group)

    def _create_scorecard_heatmap(self) -> Optional[Path]:
        if self.scorecard_ranks_df.empty or self.scorecard_values_df.empty:
            return None

        plt.figure(figsize=(16, 10))
        avg_ranks = self.scorecard_ranks_df.mean().sort_values()
        ordered_techs = avg_ranks.index.tolist()
        ordered_metrics = self.scorecard_ranks_df.index
        ranks_ordered = self.scorecard_ranks_df.reindex(index=ordered_metrics, columns=ordered_techs)
        values_ordered = self.scorecard_values_df.reindex(index=ordered_metrics, columns=ordered_techs)

        sns.heatmap(
            ranks_ordered, annot=values_ordered, fmt=".2f", cmap="RdYlGn_r",
            linewidths=.5, cbar_kws={'label': 'Performance Rank (1 is best)'}
        )

        plt.title('Performance Scorecard', fontsize=16)
        plt.xlabel('Technology (Ordered Best to Worst Overall)')
        plt.ylabel('Metric')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        path = self.plots_dir / "performance_scorecard.png"
        plt.savefig(path)
        plt.close()
        return path

    def _create_comparison_plot(self, df: pd.DataFrame, stat_col: str, metric_name: str, plot_type: str = 'box') -> Optional[Path]:
        if df.empty: return None
        plot_order = self._get_ordered_tech_list(df)
        plt.figure(figsize=(14, 8))

        if plot_type == 'box':
            sns.boxplot(data=df, x='server_type', y=stat_col, hue='group', dodge=False, order=plot_order)
        elif plot_type == 'violin':
            sns.violinplot(data=df, x='server_type', y=stat_col, hue='group', dodge=False, inner='quartile', cut=0, order=plot_order)

        plt.title(f'Distribution of {metric_name}', fontsize=16)
        plt.ylabel(metric_name)
        plt.xlabel('Technology')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{metric_name.lower().replace(' ', '_')}_{plot_type}_comparison.png"
        path = self.plots_dir / re.sub(r'[^a-z0-9_.-]', '', filename)
        plt.savefig(path)
        plt.close()
        return path

    def _create_timeseries_plot(self, metric: str, metric_name: str, group_filter: Optional[str] = None) -> Optional[Path]:
        if group_filter:
            df_metric = self.raw_df[(self.raw_df['metric'] == metric) & (self.raw_df['group'] == group_filter)]
        else:
            df_metric = self.raw_df[self.raw_df['metric'] == metric]

        if df_metric.empty: return None
        max_time = df_metric['time_sec'].max()
        if pd.isna(max_time): return None

        full_time_index = pd.to_timedelta(np.arange(int(max_time) + 1), unit='s')
        processed_dfs = []
        for group, group_df in df_metric.groupby(['server_type', 'run_number']):
            temp_df = group_df.set_index(pd.to_timedelta(group_df['time_sec'], unit='s'))
            temp_df = temp_df.reindex(full_time_index).ffill().bfill()
            temp_df['server_type'] = group[0]
            temp_df['run_number'] = group[1]
            temp_df['time_sec'] = temp_df.index.total_seconds()
            processed_dfs.append(temp_df.reset_index(drop=True))

        if not processed_dfs: return None
        plot_df = pd.concat(processed_dfs, ignore_index=True)
        agg_df = plot_df.groupby(['server_type', 'time_sec'])['metric_value'].agg(['mean', 'min', 'max']).reset_index()
        plot_order = self._get_ordered_tech_list(agg_df)

        plt.figure(figsize=(14, 8))
        palette = {"CSR": "#1f77b4", "SSR": "#d62728", "Uncategorized": "gray"}

        for tech in plot_order:
            tech_df = agg_df[agg_df['server_type'] == tech]
            if tech_df.empty: continue

            # Map tech to group color
            group = self.raw_df[self.raw_df['server_type'] == tech]['group'].iloc[0]
            color = palette.get(group, 'gray')

            plt.plot(tech_df['time_sec'], tech_df['mean'], label=tech, color=color, linewidth=2)
            plt.fill_between(tech_df['time_sec'], tech_df['min'], tech_df['max'], color=color, alpha=0.15)

        title_suffix = f" ({group_filter})" if group_filter else ""
        plt.title(f"Time-Series: {metric_name}{title_suffix}", fontsize=16)
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric_name)
        plt.legend(title='Technology', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        group_suffix = f"_{group_filter.lower().replace('-', '_')}" if group_filter else ""
        base_filename = f"{metric_name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
        path = self.plots_dir / re.sub(r'[^a-z0-9_.-]', '', base_filename)
        plt.savefig(path)
        plt.close()
        return path

    def _get_ordered_tech_list(self, df: pd.DataFrame) -> List[str]:
        all_techs = set(df['server_type'].unique())
        ordered = [t for t in self.chart_order if t in all_techs]
        remaining = sorted([t for t in all_techs if t not in set(ordered)])
        return ordered + remaining

    def _generate_capacity_plots(self, summary: pd.DataFrame) -> None:
        # 1. Timeseries Plots
        metrics_to_plot = {
            "k6_successful_html_reqs_rate": "successful_throughput_rps_timeseries",
            "cpu": "cpu_usage_timeseries",
            "memory": "ram_usage_timeseries",
        }
        labels = {
            "k6_successful_html_reqs_rate": "RPS (Throughput)",
            "cpu": "CPU Usage (%)",
            "memory": "Memory Usage (MB)",
        }
        for metric, filename in metrics_to_plot.items():
            m_df = self.raw_df[self.raw_df["metric"] == metric]
            if m_df.empty:
                continue

            plt.figure(figsize=(12, 7))
            sns.lineplot(data=m_df, x="time_sec", y="metric_value", hue="server_type", errorbar="sd")
            plt.title(f"Capacity Test: {labels[metric]} Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel(labels[metric])
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{filename}.png")
            plt.close()

        # 2. Comparison Bar Charts (from summary)
        if summary.empty:
            return

        # Map server_type to group for coloring
        tech_to_group = {
            tech.lower(): group for group, techs in self.groups_config.items() for tech in techs
        }
        summary["group"] = (
            summary["server_type"].str.lower().map(tech_to_group).fillna("Uncategorized")
        )

        comparisons = {
            "sustained_rps": "capacity_rps_comparison.png",
            "cpu_at_sustained": "capacity_cpu_at_sustained_usage.png",
            "ram_at_sustained": "capacity_ram_at_sustained_usage.png",
        }
        titles = {
            "sustained_rps": "Maximum Sustained Throughput (RPS)",
            "cpu_at_sustained": "CPU Usage at Sustained Throughput (%)",
            "ram_at_sustained": "Memory Usage at Sustained Throughput (MB)",
        }
        palette = {"CSR": "#1f77b4", "SSR": "#d62728", "Uncategorized": "gray"}

        for col, filename in comparisons.items():
            if col not in summary.columns:
                continue
            plt.figure(figsize=(12, 8))
            # Sort by value for better comparison
            sorted_summary = summary.sort_values(col, ascending=False)

            sns.barplot(
                data=sorted_summary,
                y="server_type",
                x=col,
                hue="group",
                palette=palette,
                dodge=False
            )

            plt.title(titles[col])
            plt.xlabel(titles[col].split("(")[-1].replace(")", ""))
            plt.ylabel("Framework")
            plt.legend(title="Group", loc="lower right")
            plt.tight_layout()
            plt.savefig(self.plots_dir / filename)
            plt.close()

    # --- REPORT GENERATORS ---

    def _generate_load_report(self) -> str:
        report = [f"# Performance Analysis Report for `{self.input_dir.name}`"]
        report.append(self._render_executive_summary_md())
        report.append("\n## Detailed Analysis")
        report.append(self._render_ranking_tables_md())
        report.append(self._render_visual_overview_md())
        report.append(self._render_temporal_analysis_md())
        report.append(self._render_metadata_md())
        return "\n".join(report)

    def _render_executive_summary_md(self) -> str:
        md = ["\n## Executive Summary", self.executive_summary_text]
        path = self.plots_dir / "performance_scorecard.png"
        if path.exists():
            md.append("\n### Performance Scorecard")
            md.append("![Performance Scorecard](./plots/performance_scorecard.png)")
        return "\n".join(md)

    def _render_ranking_tables_md(self) -> str:
        md = ["\n### Intra-Group Rankings"]
        stat_name_map = {'mean': 'Mean', 'std': 'Mean of Std Devs', 'p95': 'Mean of p95s'}
        emoji_map = {1: '🥇', 2: '🥈', 3: '🥉'}

        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                name = str(config['name'])
                if name not in self.ranking_results: continue
                ranking_df = self.ranking_results[name]
                md.append(f"#### {name}")

                for group in sorted(ranking_df['group'].unique()):
                    md.append(f"\n##### Group: {group}\n")
                    group_data = ranking_df[ranking_df['group'] == group].sort_values(by='mean', ascending=config['sort_ascending']).reset_index(drop=True)

                    rows = []
                    for i, row in group_data.iterrows():
                        tech_label = f"{row['server_type']} {emoji_map.get(i+1, '')}".strip()
                        val_str = f"{row['mean']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]" if pd.notna(row['ci_lower']) else f"{row['mean']:.4f}"
                        rows.append({"Technology": tech_label, stat_name_map.get(stat_col, stat_col): val_str})

                    md.append(pd.DataFrame(rows).to_markdown(index=False))
        return "\n".join(md)

    def _render_visual_overview_md(self) -> str:
        md = ["\n### Metric Distributions"]
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                name = str(config['name'])
                filename = f"{name.lower().replace(' ', '_')}_violin_comparison.png"
                if (self.plots_dir / filename).exists():
                    md.append(f"#### {name}")
                    md.append(f"![{name}](./plots/{filename})")
        return "\n".join(md)

    def _render_temporal_analysis_md(self) -> str:
        md = ["\n### Temporal Analysis"]
        for group in sorted(self.raw_df['group'].unique()):
            md.append(f"#### Group: {group}")
            for metric, config in METRIC_CONFIG['mean'].items():
                name = str(config['name'])
                group_suffix = f"_{group.lower().replace('-', '_')}"
                filename = f"{name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
                if (self.plots_dir / filename).exists():
                    md.append(f"![{name}](./plots/{filename})")
        return "\n".join(md)

    def _render_metadata_md(self) -> str:
        md = ["\n## Appendix: Experiment Parameters"]
        params = self.metadata.get('parameters', {})
        durations = self.metadata.get('calculated_durations_sec', {})
        rows = [
            ["Run Timestamp (UTC)", f"`{self.metadata.get('run_timestamp_utc', 'N/A')}`"],
            ["Runs per Technology", params.get('num_runs', 'N/A')],
            ["Target RPS per Instance", params.get('rate', 'N/A')],
            ["k6 Test Duration", f"`{params.get('k6_duration', 'N/A')}`"],
            ["Warm-up Duration", f"`{params.get('warmup_duration', 'N/A')}`"],
            ["Measurement Duration (sec)", durations.get('measurement', 'N/A')]
        ]
        md.append(pd.DataFrame(rows, columns=["Parameter", "Value"]).to_markdown(index=False))
        return "\n".join(md)

    def _generate_capacity_report(self, summary: pd.DataFrame) -> str:
        report = [f"# Capacity Report for `{self.input_dir.name}`", summary.to_markdown()]
        return "\n".join(report)

    def _generate_champions_report(self) -> str:
        return "# Champions Comparison"

    def _write_report(self, content: str) -> None:
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(content)
