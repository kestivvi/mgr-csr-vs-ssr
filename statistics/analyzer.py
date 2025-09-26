# statistics/analyzer.py
import argparse
import glob
import shutil
import sys
from pathlib import Path
import re
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- CONFIGURATION (Global) ---
METRIC_CONFIG = {
    'mean': {
        'cpu':        {'name': 'Mean CPU Usage (%)', 'sort_ascending': True},
        'memory':     {'name': 'Mean Memory Usage (MB)', 'sort_ascending': True},
        'latency':    {'name': 'Mean p95 Latency (ms)', 'sort_ascending': True},
        'network_tx': {'name': 'Mean Network Transmit Rate (MB/s)', 'sort_ascending': True}
    },
    'std': {
        'cpu':        {'name': 'CPU Usage Stability (Std Dev)', 'sort_ascending': True},
        'memory':     {'name': 'Memory Usage Stability (Std Dev)', 'sort_ascending': True},
        'latency':    {'name': 'Latency Stability (Std Dev)', 'sort_ascending': True},
        'network_tx': {'name': 'Network Transmit Stability (Std Dev)', 'sort_ascending': True}
    },
    'p95': {
        'cpu':        {'name': 'Peak CPU Usage (95th Percentile)', 'sort_ascending': True},
        'memory':     {'name': 'Peak Memory Usage (95th Percentile)', 'sort_ascending': True},
        'latency':    {'name': 'Peak Latency (95th Percentile)', 'sort_ascending': True},
        'network_tx': {'name': 'Peak Network Transmit Rate (95th Percentile) (MB/s)', 'sort_ascending': True}
    }
}

# --- PURE STATISTICAL HELPER FUNCTIONS (No Class State) ---

def calculate_confidence_interval(data: pd.Series) -> Tuple[float, float]:
    """Calculates the 95% confidence interval for the mean of a given dataset."""
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    mean = np.mean(data)
    std_err = stats.sem(data)
    if std_err == 0 or np.isnan(std_err):
        return (mean, mean)
    
    interval = stats.t.interval(0.95, df=n-1, loc=mean, scale=std_err)
    
    ci_lower = max(0, interval[0])
    return (ci_lower, interval[1])

def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    if len(group1) < 2 or len(group2) < 2: return np.nan
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    if (n1 + n2 - 2) == 0: return np.nan
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0 or np.isnan(pooled_std): return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    return (mean1 - mean2) / pooled_std

# --- MAIN ANALYSIS CLASS ---

class PerformanceAnalyzer:
    """
    Orchestrates the entire performance analysis pipeline from data loading to report generation.
    """
    def __init__(self, input_dir: Path, report_type: str, champions: Optional[List[str]] = None):
        self.input_dir = input_dir
        self.report_type = report_type
        self.champions_list = champions or []
        
        self.plots_dir = self.input_dir / "plots"
        if self.report_type == 'capacity':
            self.report_path = self.input_dir / "capacity_report.md"
        else:
            self.report_path = self.input_dir / f"report_{self.report_type}.md"

        self.metadata: Dict[str, Any] = {}
        self.groups_config: Dict[str, Any] = {}
        self.chart_order: list = []
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.summary_df: pd.DataFrame = pd.DataFrame()
        self.ranking_results: Dict[str, pd.DataFrame] = {}
        self.champion_results: Dict[str, Dict[str, Any]] = {}
        
        self.scorecard_ranks_df = pd.DataFrame()
        self.scorecard_values_df = pd.DataFrame()
        self.executive_summary_text = ""

    def run(self):
        """Executes the full analysis pipeline based on the report type."""
        print(f"Starting analysis for directory: {self.input_dir}")
        print(f"Report Type: {self.report_type}")
        self.plots_dir.mkdir(exist_ok=True)

        if not self._load_configuration(): return
        if not self._load_and_prepare_data(): return

        report_content = ""
        if self.report_type == 'all_apps':
            self._compute_rankings()
            self._compute_scorecard_and_winner()
            report_content = self._generate_all_apps_report()
        elif self.report_type == 'champions':
            self._compute_champion_stats()
            report_content = self._generate_champions_report()
        elif self.report_type == 'capacity':
            capacity_summary_df = self._run_capacity_analysis()
            if capacity_summary_df is not None and not capacity_summary_df.empty:
                report_content = self._generate_capacity_report(capacity_summary_df)
            else:
                print("ERROR: Capacity analysis did not produce a summary. Report generation skipped.")
                return
        else:
            print(f"ERROR: Unknown report type '{self.report_type}'")
            return

        self._write_report(report_content)

        print(f"\n\n{'='*25} ANALYSIS COMPLETE {'='*26}")
        print(f"All plots and archived configuration have been saved in: {self.input_dir}")
        print(f"Markdown report saved to: {self.report_path}")

    # --------------------------------------------------------------------------
    # STAGE 1: LOADING AND PREPARATION
    # --------------------------------------------------------------------------

    def _load_configuration(self) -> bool:
        """Loads metadata.yaml and config.yaml configurations."""
        metadata_path = self.input_dir / "metadata.yaml"
        if metadata_path.exists():
            print(f"INFO: Loading experiment metadata from {metadata_path}...")
            with open(metadata_path, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            print("WARNING: metadata.yaml not found.")

        config_path = Path(__file__).parent / "config.yaml"
        print(f"INFO: Loading group and chart order configuration from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if not config_data:
                    print("ERROR: Configuration file is empty.")
                    return False
                
                self.groups_config = config_data.get('groups', {})
                self.chart_order = config_data.get('chart_order', [])

                if not self.groups_config:
                    print("WARNING: 'groups' section not found in config.yaml. All technologies will be 'Uncategorized'.")
                if not self.chart_order:
                    print("WARNING: 'chart_order' section not found in config.yaml. Charts will be ordered alphabetically.")

            shutil.copy(config_path, self.input_dir / "config.yaml")
            print(f"INFO: Archived configuration to {self.input_dir / 'config.yaml'}")
            return True
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"ERROR: Failed to load configuration from {config_path}: {e}")
            return False

    def _load_and_prepare_data(self) -> bool:
        """
        Loads aggregated wide-format CSV files from the 'metrics' subdirectory,
        converts them to a long-format DataFrame, and performs initial processing.
        """
        metrics_dir = self.input_dir / "metrics"
        if not metrics_dir.is_dir():
            print(f"ERROR: 'metrics' subdirectory not found in {self.input_dir}.")
            return False

        all_files = glob.glob(str(metrics_dir / "*.csv"))
        if not all_files:
            print(f"ERROR: No CSV data files found in {metrics_dir}.")
            return False

        filename_regex = re.compile(r"^(\d+)_(.*)$")
        
        tech_to_group = {tech: group for group, techs in self.groups_config.items() for tech in techs}
        all_long_dfs = []
        
        for f in all_files:
            p = Path(f)
            match = filename_regex.match(p.stem)
            
            if not match:
                print(f"WARNING: Skipping file with unexpected name format: {p.name}")
                continue

            run_number, server_type_raw = match.groups()
            
            wide_df = pd.read_csv(f)
            
            long_df = pd.melt(
                wide_df,
                id_vars=['timestamp'],
                var_name='metric',
                value_name='metric_value'
            )
            
            long_df['run_number'] = int(run_number)
            server_type = server_type_raw.replace('_', '-')
            long_df['server_type'] = server_type
            long_df['group'] = long_df['server_type'].map(tech_to_group).fillna('Uncategorized')
            all_long_dfs.append(long_df)
        
        if not all_long_dfs:
            print("ERROR: No data could be loaded and processed from CSV files.")
            return False

        self.raw_df = pd.concat(all_long_dfs, ignore_index=True)
        
        cpu_mask = self.raw_df['metric'] == 'cpu'
        self.raw_df.loc[cpu_mask, 'metric_value'] *= 100
        
        memory_mask = self.raw_df['metric'] == 'memory'
        self.raw_df.loc[memory_mask, 'metric_value'] /= (1024 * 1024)
        
        network_mask = self.raw_df['metric'] == 'network_tx'
        self.raw_df.loc[network_mask, 'metric_value'] /= (1024 * 1024)
        
        self.raw_df['timestamp'] = pd.to_datetime(self.raw_df['timestamp'])
        self.raw_df['time_sec'] = self.raw_df.groupby(['server_type', 'run_number', 'metric'])['timestamp'].transform(lambda x: (x - x.min()).dt.total_seconds())
        
        self.summary_df = self.raw_df.groupby(['group', 'server_type', 'run_number', 'metric'])['metric_value'].agg(['mean', 'std', lambda x: x.quantile(0.95)]).reset_index()
        self.summary_df.rename(columns={'<lambda_0>': 'p95'}, inplace=True)
        
        print("INFO: Data loading and preparation complete.")
        return True

    # --------------------------------------------------------------------------
    # STAGE 2: STATISTICAL COMPUTATION
    # --------------------------------------------------------------------------

    def _run_capacity_analysis(self) -> Optional[pd.DataFrame]:
        """
        Performs the entire analysis for the capacity test (stress test) data.
        """
        print("INFO: Starting capacity test analysis...")

        required_metrics = [
            'k6_successful_html_reqs_rate', 'k6_total_html_reqs_rate',
            'cpu', 'memory'
        ]
        available_metrics = self.raw_df['metric'].unique()
        if not all(m in available_metrics for m in required_metrics):
            print("ERROR: Missing required metrics for capacity analysis.", file=sys.stderr)
            print(f"Required: {required_metrics}", file=sys.stderr)
            print(f"Found: {list(available_metrics)}", file=sys.stderr)
            return None

        all_run_results = []
        for (server_type, run_number), run_df in self.raw_df.groupby(['server_type', 'run_number']):
            print(f"  -> Analyzing {server_type} (Run {run_number})...")

            pivot_df = run_df.pivot_table(
                index='time_sec', columns='metric', values='metric_value'
            ).reindex(columns=required_metrics).fillna(0)

            # --- 1. Calculate Sustained RPS (Max of rolling 30s minimums) ---
            rolling_mins = pivot_df['k6_successful_html_reqs_rate'].rolling(window=30, min_periods=1).min()
            sustained_rps = rolling_mins.max() if not rolling_mins.empty else 0

            # --- 2. Calculate Peak RPS (Absolute maximum over the entire run) ---
            peak_rps = pivot_df['k6_successful_html_reqs_rate'].max() if not pivot_df.empty else 0

            # --- 3. Extract Resource Usage at the Sustained RPS point ---
            sustained_rps_time = rolling_mins.idxmax() if not rolling_mins.empty else 0
            cpu_at_sustained = pivot_df.loc[sustained_rps_time, 'cpu'] if sustained_rps_time in pivot_df.index else 0
            ram_at_sustained = pivot_df.loc[sustained_rps_time, 'memory'] if sustained_rps_time in pivot_df.index else 0

            all_run_results.append({
                'server_type': server_type, 'run_number': run_number,
                'sustained_rps': sustained_rps, 'peak_rps': peak_rps,
                'cpu_at_sustained': cpu_at_sustained, 'ram_at_sustained': ram_at_sustained,
            })

        if not all_run_results:
            print("ERROR: No capacity results could be calculated.")
            return None

        results_df = pd.DataFrame(all_run_results)
        final_summary_df = results_df.groupby('server_type').mean().drop(columns='run_number').reset_index()
        
        print("INFO: Capacity analysis computation complete.")
        return final_summary_df

    def _compute_rankings(self):
        """Calculates rankings with confidence intervals for all metrics."""
        print("INFO: Computing statistical rankings...")
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_df = self.summary_df[self.summary_df['metric'] == metric]
                if metric_df.empty:
                    continue
                
                agg_df = metric_df.groupby(['group', 'server_type'])[stat_col].agg(['mean', calculate_confidence_interval]).reset_index()
                agg_df[['ci_lower', 'ci_upper']] = pd.DataFrame(agg_df['calculate_confidence_interval'].tolist(), index=agg_df.index)
                self.ranking_results[config['name']] = agg_df.drop(columns=['calculate_confidence_interval'])
        print("INFO: Ranking computation complete.")

    def _compute_scorecard_and_winner(self):
        """Computes ranks for all metrics to build a scorecard and determine an overall winner."""
        print("INFO: Computing scorecard and determining winner...")
        all_ranks = {}
        all_values = {}
        
        for metric_name, df in self.ranking_results.items():
            sort_asc = True
            for stat_metrics in METRIC_CONFIG.values():
                for m_config in stat_metrics.values():
                    if m_config['name'] == metric_name:
                        sort_asc = m_config['sort_ascending']
                        break
            
            ranked_df = df.sort_values('mean', ascending=sort_asc).reset_index(drop=True)
            ranked_df['rank'] = ranked_df.index + 1
            
            all_ranks[metric_name] = ranked_df.set_index('server_type')['rank']
            all_values[metric_name] = ranked_df.set_index('server_type')['mean']

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

    def _compute_champion_stats(self):
        """Computes all pairwise statistical tests between the two provided champions."""
        print("INFO: Computing champion statistics...")
        if len(self.champions_list) != 2:
            print("ERROR: Champion comparison requires exactly two technologies.")
            return

        all_techs = set(self.summary_df['server_type'].unique())
        for champ in self.champions_list:
            if champ not in all_techs:
                print(f"\nERROR: Champion technology '{champ}' not found in the dataset.", file=sys.stderr)
                print("Please check for typos. Available technologies are:", file=sys.stderr)
                for tech in sorted(list(all_techs)):
                    print(f"  - {tech}", file=sys.stderr)
                sys.exit(1)

        champ1_tech, champ2_tech = self.champions_list[0], self.champions_list[1]
        print(f"INFO: Comparing '{champ1_tech}' vs '{champ2_tech}'")

        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_df = self.summary_df[self.summary_df['metric'] == metric]
                if metric_df.empty: continue

                group1 = metric_df[metric_df['server_type'] == champ1_tech][stat_col].dropna()
                group2 = metric_df[metric_df['server_type'] == champ2_tech][stat_col].dropna()
                
                results = {'name': config['name'], 'champ1': champ1_tech, 'champ2': champ2_tech}
                if len(group1) > 1 and len(group2) > 1:
                    results['shapiro_p1'] = stats.shapiro(group1).pvalue
                    results['shapiro_p2'] = stats.shapiro(group2).pvalue
                    results['levene_p'] = stats.levene(group1, group2).pvalue
                    
                    if results['shapiro_p1'] > 0.05 and results['shapiro_p2'] > 0.05:
                        test = stats.ttest_ind(group1, group2, equal_var=False)
                        results.update({'test_name': "Welch's t-test", 'statistic': test.statistic, 'p_value': test.pvalue})
                    else:
                        test = stats.mannwhitneyu(group1, group2)
                        results.update({'test_name': 'Mann-Whitney U', 'statistic': test.statistic, 'p_value': test.pvalue})
                
                results['cohen_d'] = cohen_d(group1, group2)
                self.champion_results[config['name']] = results
        print("INFO: Champion statistics complete.")

    # --------------------------------------------------------------------------
    # STAGE 3: REPORT GENERATION
    # --------------------------------------------------------------------------

    def _generate_capacity_report(self, summary_df: pd.DataFrame) -> str:
        """Assembles all parts of the capacity test report."""
        print("INFO: Generating 'capacity' report content...")
        
        report_parts = [
            f"# Raport z Testu Pojemnościowego dla `{self.input_dir.name}`",
            "\nRaport prezentuje maksymalną wydajność systemów pod rosnącym obciążeniem. Analiza rozróżnia dwie kluczowe metryki:",
            "- **Utrzymana Przepustowość (Sustained RPS):** Najwyższy poziom zapytań, który system był w stanie obsługiwać w sposób stabilny i ciągły.",
            "- **Szczytowa Przepustowość (Peak RPS):** Chwilowe, absolutne maksimum wydajności zarejestrowane w stabilnym okresie pracy.",
            "\n## 1. Wizualne Porównanie Przepustowości"
        ]

        rps_plot_path = self._create_capacity_rps_plot(summary_df)
        if rps_plot_path:
            report_parts.append(f"![Porównanie Przepustowości]({rps_plot_path.relative_to(self.input_dir)})")

        report_parts.append("\n## 2. Szczegółowe Wyniki Numeryczne")
        report_parts.append(self._render_capacity_tables_md(summary_df))

        report_parts.append("\n## 3. Analiza Kosztu Zasobowego")
        cpu_plot_path = self._create_capacity_resource_plot(summary_df, 'cpu_at_sustained', 'Zużycie CPU przy Utrzymanej Przepustowości', 'Średnie Zużycie CPU (%)')
        if cpu_plot_path:
            report_parts.append(f"![Zużycie CPU]({cpu_plot_path.relative_to(self.input_dir)})")
        
        ram_plot_path = self._create_capacity_resource_plot(summary_df, 'ram_at_sustained', 'Zużycie RAM przy Utrzymanej Przepustowości', 'Średnie Zużycie RAM (MB)')
        if ram_plot_path:
            report_parts.append(f"![Zużycie RAM]({ram_plot_path.relative_to(self.input_dir)})")

        # --- START: DODANA SEKCJA 4 ---
        report_parts.append("\n## 4. Analiza Temporalna (Szeregi Czasowe)")
        report_parts.append("Poniższe wykresy przedstawiają przebieg kluczowych metryk na przestrzeni czasu, co pozwala zidentyfikować momenty nasycenia zasobów i awarii systemu.")

        # 4a. RPS Time Series
        rps_ts_path = self._create_timeseries_plot(
            'k6_successful_html_reqs_rate', 
            'Udana Przepustowość (RPS)', 
            group_filter=None
        )
        if rps_ts_path:
            report_parts.append(f"### Przebieg Przepustowości (RPS)\n")
            report_parts.append(f"![Przebieg RPS]({rps_ts_path.relative_to(self.input_dir)})")

        # 4b. CPU Time Series
        cpu_ts_path = self._create_timeseries_plot(
            'cpu', 
            'Zużycie CPU', 
            group_filter=None
        )
        if cpu_ts_path:
            report_parts.append(f"### Przebieg Zużycia CPU\n")
            report_parts.append(f"![Przebieg CPU]({cpu_ts_path.relative_to(self.input_dir)})")

        # 4c. RAM Time Series
        ram_ts_path = self._create_timeseries_plot(
            'memory', 
            'Zużycie RAM', 
            group_filter=None
        )
        if ram_ts_path:
            report_parts.append(f"### Przebieg Zużycia RAM\n")
            report_parts.append(f"![Przebieg RAM]({ram_ts_path.relative_to(self.input_dir)})")
        # --- END: DODANA SEKCJA 4 ---

        report_parts.append("\n## 5. Załącznik: Metodologia Obliczeń")
        report_parts.append(self._render_capacity_methodology_md())
        
        return "\n".join(report_parts)

    def _generate_all_apps_report(self) -> str:
        """Assembles all parts of the main comparison report."""
        print("INFO: Generating 'all_apps' report content...")
        report_parts = [f"# Performance Analysis Report for `{self.input_dir.name}`"]
        report_parts.append(self._render_executive_summary_md())
        report_parts.append("\n## Detailed Analysis")
        report_parts.append("The following sections provide a detailed breakdown of each performance metric through rankings, distributions, and time-series analysis.")
        report_parts.append(self._render_ranking_tables_md())
        report_parts.append(self._render_visual_overview_md())
        report_parts.append(self._render_temporal_analysis_md())
        report_parts.append(self._render_metadata_md())
        return "\n".join(report_parts)

    def _generate_champions_report(self) -> str:
        """Assembles the focused champion vs. champion report."""
        print("INFO: Generating 'champions' report content...")
        report_parts = [f"# Champion Comparison Report for `{self.input_dir.name}`"]
        report_parts.append(self._render_champion_analysis_md())
        report_parts.append(self._render_metadata_md())
        return "\n".join(report_parts)

    def _write_report(self, content: str):
        """Writes the final report content to a file."""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # --- "Render" Methods for Markdown Sections ---

    def _render_capacity_tables_md(self, summary_df: pd.DataFrame) -> str:
        """Generates Markdown tables for the capacity report."""
        
        sorted_df = summary_df.sort_values('sustained_rps', ascending=False)

        rps_table = sorted_df[['server_type', 'sustained_rps', 'peak_rps']].copy()
        # Ensure we don't divide by zero if sustained_rps is 0
        rps_table['difference_perc'] = np.divide(
            (rps_table['peak_rps'] - rps_table['sustained_rps']),
            rps_table['sustained_rps'],
            out=np.zeros_like(rps_table['sustained_rps'], dtype=float),
            where=(rps_table['sustained_rps'] != 0)
        ) * 100
        
        rps_table.rename(columns={
            'server_type': 'Technologia',
            'sustained_rps': 'Utrzymany RPS (Sustained)',
            'peak_rps': 'Szczytowy RPS (Peak)',
            'difference_perc': 'Różnica (%)'
        }, inplace=True)
        
        resource_table = sorted_df[['server_type', 'cpu_at_sustained', 'ram_at_sustained']].copy()
        resource_table.rename(columns={
            'server_type': 'Technologia',
            'cpu_at_sustained': 'CPU @ Sustained RPS (%)',
            'ram_at_sustained': 'RAM @ Sustained RPS (MB)'
        }, inplace=True)

        md_parts = [
            "### Tabela 1: Ranking Przepustowości (RPS dla zapytań HTML)",
            rps_table.to_markdown(index=False, floatfmt=".2f"),
            "\n### Tabela 2: Koszt Zasobowy przy Utrzymanej Przepustowości",
            resource_table.to_markdown(index=False, floatfmt=".2f")
        ]
        return "\n".join(md_parts)

    def _render_capacity_methodology_md(self) -> str:
        """Returns a static string explaining the capacity calculation methodology."""
        return (
            "Metryki w tym raporcie zostały obliczone na podstawie surowych danych szeregów czasowych z Prometheus w następujący sposób:\n\n"
            "**Utrzymany RPS (Sustained):**\n"
            "1. Na całym szeregu czasowym udanych zapytań HTML na sekundę zastosowano 30-sekundowe, przesuwane okno.\n"
            "2. W każdym oknie znaleziono wartość **minimalną**.\n"
            "3. Finalny wynik to **najwyższa (maksymalna)** wartość znaleziona spośród wszystkich minimów z okien. Reprezentuje to najwyższą przepustowość, jaką system mógł zagwarantować w dowolnym 30-sekundowym okresie.\n\n"
            "**Szczytowy RPS (Peak):**\n"
            "1. Finalny wynik to **absolutnie najwyższa (maksymalna)** wartość udanych zapytań HTML na sekundę, zarejestrowana w dowolnym, jednosekundowym momencie całego testu."
        )

    def _render_executive_summary_md(self) -> str:
        """Renders the high-level summary and scorecard heatmap."""
        md = ["\n## Executive Summary", self.executive_summary_text]
        
        path = self._create_scorecard_heatmap()
        if path:
            md.append(f"\n### Performance Scorecard")
            md.append("The table below shows the performance of each technology across all metrics, ordered from best to worst overall. The color indicates the rank (Green = Best, Red = Worst), and the number in the cell is the measured value.")
            md.append(f"![Performance Scorecard]({path.relative_to(self.input_dir)})")
        
        return "\n".join(md)

    def _render_metadata_md(self) -> str:
        """Renders the experiment parameters as an appendix."""
        if not self.metadata:
            return "\n## Appendix: Experiment Parameters\n\n*Metadata file (metadata.yaml) not found.*\n"
        
        md = ["\n## Appendix: Experiment Parameters", "This report was generated from data collected using the following parameters:\n"]
        
        params = self.metadata.get('parameters', {})
        durations = self.metadata.get('calculated_durations_sec', {})
        
        md.extend([
            "| Parameter                   | Value |",
            "|-----------------------------|-------|",
            f"| Run Timestamp (UTC)         | `{self.metadata.get('run_timestamp_utc', 'N/A')}` |",
            f"| Runs per Technology         | {params.get('num_runs', 'N/A')} |",
            f"| Target RPS per Instance     | {params.get('rate', 'N/A')} |",
            f"| k6 Test Duration            | `{params.get('k6_duration', 'N/A')}` |",
            f"| Warm-up Duration            | `{params.get('warmup_duration', 'N/A')}` |",
            f"| Cooldown Duration           | `{params.get('cooldown_duration', 'N/A')}` |",
            f"| Measurement Duration (sec)  | {durations.get('measurement', 'N/A')} |"
        ])
        return "\n".join(md)

    def _render_visual_overview_md(self) -> str:
        md = ["\n### Metric Distributions", "These plots show the distribution of performance metrics across all technologies and runs.\n"]
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_df = self.summary_df[self.summary_df['metric'] == metric]
                if metric_df.empty: continue
                
                md.append(f"#### {config['name']}")
                path = self._create_comparison_plot(metric_df, stat_col, config['name'], plot_type='violin')
                if path: md.append(f"![Violin Plot: {config['name']}]({path.relative_to(self.input_dir)})")
        return "\n".join(md)

    def _render_temporal_analysis_md(self) -> str:
        md = ["\n### Temporal Analysis", "Time-series plots showing the mean metric value over the duration of the test runs, separated by technology group.\n"]
        
        groups = sorted(self.raw_df['group'].unique())
        for group in groups:
            md.append(f"#### Group: {group}")
            group_has_plot = False
            for metric, config in METRIC_CONFIG['mean'].items():
                path = self._create_timeseries_plot(metric, config['name'], group_filter=group)
                if path:
                    md.append(f"![{config['name']} Time-Series for {group}]({path.relative_to(self.input_dir)})")
                    group_has_plot = True
            if not group_has_plot:
                md.append("*No time-series data available for this group.*")
            md.append("\n")
        return "\n".join(md)

    def _render_ranking_tables_md(self) -> str:
        md = ["\n### Intra-Group Rankings", "Tables ranking each technology within its group. The metric shown is the mean across all runs, with the 95% confidence interval.\n"]
        
        stat_name_map = {'mean': 'Mean', 'std': 'Mean of Std Devs', 'p95': 'Mean of p95s'}
        emoji_map = {1: '🥇', 2: '🥈', 3: '🥉'}

        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_name = config['name']
                if metric_name not in self.ranking_results: continue
                
                ranking_df = self.ranking_results[metric_name]
                md.append(f"#### {metric_name}")
                
                sort_ascending = config['sort_ascending']
                header_name = f"{stat_name_map.get(stat_col, stat_col.title())} (95% CI)"

                for group_name in sorted(ranking_df['group'].unique()):
                    md.append(f"\n##### Group: {group_name}\n")
                    group_data = ranking_df[ranking_df['group'] == group_name].sort_values(by='mean', ascending=sort_ascending).reset_index()
                    
                    if group_data.empty: continue

                    def add_emoji(row):
                        rank = row.name + 1
                        emoji = emoji_map.get(rank, '')
                        return f"{row['server_type']} {emoji}".strip()

                    group_data['Technology'] = group_data.apply(add_emoji, axis=1)
                    group_data['formatted_metric'] = group_data.apply(
                        lambda r: f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]" if pd.notna(r['ci_lower']) else f"{r['mean']:.4f}", axis=1)
                    
                    output_df = group_data[['Technology', 'formatted_metric']].rename(columns={'formatted_metric': header_name})
                    md.append(output_df.to_markdown(index=False))
        return "\n".join(md)

    def _interpret_stats(self, p_value: Optional[float], cohen_d: Optional[float]) -> str:
        """Helper to translate statistical results into English."""
        if p_value is None or np.isnan(p_value):
            return "Statistical test could not be performed (e.g., insufficient data)."

        if p_value < 0.05:
            significance = "The observed difference is **statystycznie istotna**."
        else:
            significance = "The observed difference is **statystycznie nieistotna**."

        if cohen_d is None or np.isnan(cohen_d):
            effect = "Effect size could not be calculated."
        else:
            d = abs(cohen_d)
            if d < 0.2: effect_label = "znikoma"
            elif d < 0.5: effect_label = "mała"
            elif d < 0.8: effect_label = "średnia"
            else: effect_label = "duża"
            effect = f"Wielkość tej różnicy (effect size) jest **{effect_label}**."

        return f"{significance} {effect}"

    def _render_champion_analysis_md(self) -> str:
        if not self.champion_results:
            return "\n## Champion vs. Champion Analysis\n\n*Statistical comparison could not be completed. Ensure data exists for the specified champions.*\n"
        
        champ1_tech, champ2_tech = self.champions_list[0], self.champions_list[1]

        md = ["\n## Champion vs. Champion Analysis"]
        md.append(f"This report provides a direct statistical comparison between two selected technologies:\n")
        md.append(f"*   **Technology 1:** `{champ1_tech}`")
        md.append(f"*   **Technology 2:** `{champ2_tech}`\n")

        for metric_name, results in self.champion_results.items():
            md.append(f"### Statistical Comparison: {metric_name}")
            md.append(f"**Comparison:** `{results.get('champ1', 'N/A')}` vs. `{results.get('champ2', 'N/A')}`\n")
            md.append("| Test | Value |")
            md.append("|---|---|")
            md.append(f"| Normality (Shapiro-Wilk p-value) | `{results.get('shapiro_p1', 0):.4f}` (Tech 1), `{results.get('shapiro_p2', 0):.4f}` (Tech 2) |")
            md.append(f"| Variance (Levene's p-value) | `{results.get('levene_p', 0):.4f}` |")
            md.append(f"| Hypothesis Test Used | **{results.get('test_name', 'N/A')}** |")
            md.append(f"| Test Statistic | `{results.get('statistic', 0):.4f}` |")
            md.append(f"| p-value | `{results.get('p_value', 0):.4f}` |")
            md.append(f"| Effect Size (Cohen's d) | `{results.get('cohen_d', 0):.4f}` |")
            
            interpretation = self._interpret_stats(results.get('p_value'), results.get('cohen_d'))
            md.append(f"\n**Interpretation:** {interpretation}\n")
        
        return "\n".join(md)

    # --- Plotting Methods ---

    def _create_capacity_rps_plot(self, summary_df: pd.DataFrame) -> Optional[Path]:
        """Creates the main capacity bar chart with sustained and peak RPS."""
        print("INFO: Generating capacity RPS comparison plot...")
        if summary_df.empty: return None

        sorted_df = summary_df.sort_values('sustained_rps', ascending=True)
        
        plt.figure(figsize=(12, 10))
        
        bars = plt.barh(sorted_df['server_type'], sorted_df['sustained_rps'], color='skyblue', label='Utrzymany RPS (Sustained)')
        
        # Use scatter plot for peak markers for robustness
        plt.scatter(y=sorted_df['server_type'], x=sorted_df['peak_rps'], color='salmon', 
                    marker='|', s=100, zorder=10, label='Szczytowy RPS (Peak)')

        plt.xlabel('Requests Per Second (dla zapytań HTML)')
        plt.ylabel('Technologia')
        plt.title('Porównanie Przepustowości w Teście Pojemnościowym')
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        filepath = self.plots_dir / "capacity_rps_comparison.png"
        plt.savefig(filepath)
        plt.close()
        return filepath

    def _create_capacity_resource_plot(self, summary_df: pd.DataFrame, metric: str, title: str, xlabel: str) -> Optional[Path]:
        """Creates a bar chart for resource usage at sustained RPS."""
        print(f"INFO: Generating capacity resource plot for {metric}...")
        if summary_df.empty or metric not in summary_df.columns: return None

        sorted_df = summary_df.sort_values('sustained_rps', ascending=True)

        plt.figure(figsize=(12, 10))
        plt.barh(sorted_df['server_type'], sorted_df[metric], color='lightgreen')
        
        plt.xlabel(xlabel)
        plt.ylabel('Technologia')
        plt.title(title)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"capacity_{metric}_usage.png"
        filepath = self.plots_dir / filename
        plt.savefig(filepath)
        plt.close()
        return filepath

    def _get_ordered_tech_list(self, df: pd.DataFrame) -> list:
        """
        Gets a list of technologies from the dataframe, ordered according to the
        'chart_order' list in the config file.
        """
        if not self.chart_order:
            return sorted(df['server_type'].unique())

        all_techs_in_data = set(df['server_type'].unique())
        ordered_techs_from_config = [t for t in self.chart_order if t in all_techs_in_data]
        config_techs_set = set(ordered_techs_from_config)
        remaining_techs = sorted([t for t in all_techs_in_data if t not in config_techs_set])
        return ordered_techs_from_config + remaining_techs

    def _create_scorecard_heatmap(self) -> Path | None:
        """Creates a heatmap with columns ordered by average rank."""
        print("INFO: Generating scorecard heatmap...")
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
        
        filepath = self.plots_dir / "performance_scorecard.png"
        plt.savefig(filepath)
        plt.close()
        return filepath

    def _create_comparison_plot(self, df: pd.DataFrame, stat_col: str, metric_name: str, plot_type: str = 'box') -> Path | None:
        """Creates a comparison plot (box or violin)."""
        print(f"INFO: Generating {plot_type} plot for '{metric_name}'...")
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
        filepath = self.plots_dir / re.sub(r'[^a-z0-9_.-]', '', filename)
        plt.savefig(filepath)
        plt.close()
        return filepath

    def _create_timeseries_plot(self, metric: str, metric_name: str, group_filter: Optional[str] = None) -> Path | None:
        """Creates a time-series plot, optionally filtered by group."""
        title_suffix = f" ({group_filter})" if group_filter else ""
        print(f"INFO: Generating time-series plot for '{metric_name}{title_suffix}'...")
        
        if group_filter:
            df_metric = self.raw_df[(self.raw_df['metric'] == metric) & (self.raw_df['group'] == group_filter)]
        else:
            df_metric = self.raw_df[self.raw_df['metric'] == metric]
            
        if df_metric.empty: return None

        max_time = df_metric['time_sec'].max()
        if pd.isna(max_time):
            print(f"WARNING: No valid time data for metric '{metric_name}{title_suffix}'. Skipping plot.")
            return None
            
        full_time_index = pd.to_timedelta(np.arange(int(max_time) + 1), unit='s')

        processed_dfs = []
        for group, group_df in df_metric.groupby(['server_type', 'run_number']):
            temp_df = group_df.set_index(pd.to_timedelta(group_df['time_sec'], unit='s'))
            temp_df = temp_df.reindex(full_time_index).ffill().bfill()
            temp_df['server_type'] = group[0]
            temp_df['run_number'] = group[1]
            temp_df['time_sec'] = temp_df.index.total_seconds()
            processed_dfs.append(temp_df.reset_index(drop=True))
        
        if not processed_dfs:
            print(f"WARNING: Could not process time-series data for metric '{metric_name}{title_suffix}'.")
            return None

        plot_df = pd.concat(processed_dfs, ignore_index=True)
        agg_df = plot_df.groupby(['server_type', 'time_sec'])['metric_value'].agg(['mean', 'min', 'max']).reset_index()
        plot_order = self._get_ordered_tech_list(agg_df)

        plt.figure(figsize=(14, 8))
        palette = sns.color_palette("husl", len(plot_order))
        color_map = {tech: color for tech, color in zip(plot_order, palette)}

        for tech in plot_order:
            tech_df = agg_df[agg_df['server_type'] == tech]
            if tech_df.empty: continue
            color = color_map[tech]
            plt.plot(tech_df['time_sec'], tech_df['mean'], label=tech, color=color)
            plt.fill_between(tech_df['time_sec'], tech_df['min'], tech_df['max'], color=color, alpha=0.2)
        
        plt.title(f"Time-Series Analysis: {metric_name}{title_suffix}", fontsize=16)
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric_name)
        plt.legend(title='Technology')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        group_suffix = f"_{group_filter.lower().replace('-', '_')}" if group_filter else ""
        base_filename = f"{metric_name.lower().replace(' ', '_')}{group_suffix}_timeseries_overview.png"
        filepath = self.plots_dir / re.sub(r'[^a-z0-9_.-]', '', base_filename)
        plt.savefig(filepath)
        plt.close()
        return filepath

# --- SCRIPT EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performance analysis script that generates a Markdown report.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input-dir', required=True, type=Path, help="Directory containing the CSV data files.")
    parser.add_argument('--report-type', required=True, choices=['all_apps', 'champions', 'capacity'],
                        help="Choose the type of report to generate: 'all_apps' (load test), 'champions' (statistical comparison), 'capacity' (stress test).")
    parser.add_argument('--champions', nargs=2, metavar=('TECH1', 'TECH2'),
                        help="Specify two technologies to compare. Required for '--report-type champions'.")
    args = parser.parse_args()

    if args.report_type == 'champions' and not args.champions:
        parser.error("--champions is required when --report-type is 'champions'.")

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)
        
    try:
        import pandas, numpy, scipy, matplotlib, seaborn, yaml
    except ImportError as e:
        print(f"ERROR: Missing dependency - {e.name}. Please install it.", file=sys.stderr)
        print("You can install all dependencies with: pip install pandas numpy scipy matplotlib seaborn pyyaml", file=sys.stderr)
        sys.exit(1)
    
    analyzer = PerformanceAnalyzer(args.input_dir, args.report_type, args.champions)
    analyzer.run()
