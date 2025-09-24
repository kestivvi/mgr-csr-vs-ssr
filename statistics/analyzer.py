import argparse
import glob
import shutil
from pathlib import Path
import re
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- CONFIGURATION (Global) ---
METRIC_CONFIG = {
    'mean': {
        'cpu':     {'name': 'Mean CPU Usage (%)', 'sort_ascending': True},
        'memory':  {'name': 'Mean Memory Usage (%)', 'sort_ascending': True},
        'latency': {'name': 'Mean p95 Latency (ms)', 'sort_ascending': True}
    },
    'std': {
        'cpu':     {'name': 'CPU Usage Stability (Std Dev)', 'sort_ascending': True},
        'memory':  {'name': 'Memory Usage Stability (Std Dev)', 'sort_ascending': True},
        'latency': {'name': 'Latency Stability (Std Dev)', 'sort_ascending': True}
    },
    'p95': {
        'cpu':     {'name': 'Peak CPU Usage (95th Percentile)', 'sort_ascending': True},
        'memory':  {'name': 'Peak Memory Usage (95th Percentile)', 'sort_ascending': True},
        'latency': {'name': 'Peak Latency (95th Percentile)', 'sort_ascending': True}
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
    if std_err == 0:
        return (mean, mean)
    
    interval = stats.t.interval(0.95, df=n-1, loc=mean, scale=std_err)
    
    # --- FIX #2: Clamp the lower bound at 0 for non-negative metrics ---
    # This prevents nonsensical negative CIs for metrics like standard deviation.
    ci_lower = max(0, interval[0])
    return (ci_lower, interval[1])

def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    if len(group1) < 2 or len(group2) < 2: return np.nan
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    if (n1 + n2 - 2) == 0: return np.nan
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0: return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    return (mean1 - mean2) / pooled_std

# --- MAIN ANALYSIS CLASS ---

class PerformanceAnalyzer:
    """
    Orchestrates the entire performance analysis pipeline from data loading to report generation.
    """
    def __init__(self, input_dir: Path, primary_metric: str):
        self.input_dir = input_dir
        self.primary_metric_str = primary_metric
        self.plots_dir = self.input_dir / "plots"
        self.report_path = self.input_dir / "report.md"

        self.metadata: Dict[str, Any] = {}
        self.groups_config: Dict[str, Any] = {}
        self.chart_order: list = []
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.summary_df: pd.DataFrame = pd.DataFrame()
        self.ranking_results: Dict[str, pd.DataFrame] = {}
        self.champions: Dict[str, str] = {}
        self.champion_results: Dict[str, Dict[str, Any]] = {}

    def run(self):
        """Executes the full analysis pipeline."""
        print(f"Starting analysis for directory: {self.input_dir}")
        self.plots_dir.mkdir(exist_ok=True)

        if not self._load_configuration():
            return

        if not self._load_and_prepare_data():
            return

        self._compute_statistical_summaries()
        
        report_content = self._generate_report()
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
        """Loads all CSV files and creates the initial raw and summary DataFrames."""
        all_files = glob.glob(str(self.input_dir / "[0-9][0-9]_*_*.csv"))
        if not all_files:
            print(f"ERROR: No valid CSV data files found in {self.input_dir} matching the pattern 'NN_...'.")
            return False

        filename_regex = re.compile(r"^(\d+)_([a-zA-Z0-9_-]+)_(cpu|memory|latency)$")
        
        tech_to_group = {tech: group for group, techs in self.groups_config.items() for tech in techs}
        all_dfs = []
        
        for f in all_files:
            p = Path(f)
            match = filename_regex.match(p.stem)
            
            if not match:
                print(f"WARNING: Skipping file with unexpected name format: {p.name}")
                continue

            run_number, server_type_raw, metric = match.groups()
            
            df = pd.read_csv(f)
            df['run_number'] = int(run_number)
            server_type = server_type_raw.replace('_', '-')
            df['server_type'] = server_type
            df['metric'] = metric
            df['group'] = df['server_type'].map(tech_to_group).fillna('Uncategorized')
            all_dfs.append(df)
        
        if not all_dfs:
            print("ERROR: No data could be loaded from CSV files.")
            return False

        self.raw_df = pd.concat(all_dfs, ignore_index=True)
        
        cpu_mem_mask = self.raw_df['metric'].isin(['cpu', 'memory'])
        self.raw_df.loc[cpu_mem_mask, 'metric_value'] *= 100
        
        self.raw_df['timestamp'] = pd.to_datetime(self.raw_df['timestamp'])
        self.raw_df['time_sec'] = self.raw_df.groupby(['server_type', 'run_number', 'metric'])['timestamp'].transform(lambda x: (x - x.min()).dt.total_seconds())
        
        self.summary_df = self.raw_df.groupby(['group', 'server_type', 'run_number', 'metric'])['metric_value'].agg(['mean', 'std', lambda x: x.quantile(0.95)]).reset_index()
        self.summary_df.rename(columns={'<lambda_0>': 'p95'}, inplace=True)
        
        print("INFO: Data loading and preparation complete.")
        return True

    # --------------------------------------------------------------------------
    # STAGE 2: STATISTICAL COMPUTATION
    # --------------------------------------------------------------------------

    def _compute_statistical_summaries(self):
        """Performs all statistical calculations and stores them in instance attributes."""
        print("INFO: Computing statistical summaries...")
        self._compute_rankings()
        self._compute_champion_stats()
        print("INFO: Statistical computations complete.")

    def _compute_rankings(self):
        """Calculates rankings with confidence intervals for all metrics."""
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_df = self.summary_df[self.summary_df['metric'] == metric]
                if metric_df.empty:
                    continue
                
                agg_df = metric_df.groupby(['group', 'server_type'])[stat_col].agg(['mean', calculate_confidence_interval]).reset_index()
                agg_df[['ci_lower', 'ci_upper']] = pd.DataFrame(agg_df['calculate_confidence_interval'].tolist(), index=agg_df.index)
                self.ranking_results[config['name']] = agg_df.drop(columns=['calculate_confidence_interval'])

    def _compute_champion_stats(self):
        """Selects champions and computes all pairwise statistical tests between them."""
        try:
            primary_stat, primary_metric = self.primary_metric_str.split('_')
            primary_metric_name = METRIC_CONFIG[primary_stat][primary_metric]['name']
        except (ValueError, KeyError):
            print(f"ERROR: Invalid --primary-metric format '{self.primary_metric_str}'.")
            return

        primary_df = self.ranking_results.get(primary_metric_name)
        if primary_df is None or primary_df.empty:
            print(f"ERROR: No data for primary metric '{primary_metric_name}'. Cannot select champions.")
            return

        for group in self.groups_config.keys():
            group_techs = primary_df[primary_df['group'] == group]
            if not group_techs.empty:
                sort_asc = METRIC_CONFIG[primary_stat][primary_metric]['sort_ascending']
                best = group_techs.loc[group_techs['mean'].idxmin() if sort_asc else group_techs['mean'].idxmax()]
                self.champions[group] = best['server_type']
        
        if len(self.champions) < 2:
            print("INFO: Fewer than two groups have champions. Skipping champion vs. champion analysis.")
            return
        
        champ_keys = list(self.champions.keys())
        champ1_group, champ2_group = champ_keys[0], champ_keys[1]
        champ1_tech, champ2_tech = self.champions[champ1_group], self.champions[champ2_group]

        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_df = self.summary_df[self.summary_df['metric'] == metric]
                if metric_df.empty: continue

                group1 = metric_df[metric_df['server_type'] == champ1_tech][stat_col]
                group2 = metric_df[metric_df['server_type'] == champ2_tech][stat_col]
                
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

    # --------------------------------------------------------------------------
    # STAGE 3: REPORT GENERATION
    # --------------------------------------------------------------------------

    def _generate_report(self) -> str:
        """Assembles all parts of the Markdown report."""
        print("INFO: Generating report content...")
        report_parts = [f"# Performance Analysis Report for `{self.input_dir.name}`"]
        report_parts.append(self._render_metadata_md())
        report_parts.append(self._render_visual_overview_md())
        report_parts.append(self._render_temporal_analysis_md())
        report_parts.append(self._render_ranking_tables_md())
        report_parts.append(self._render_champion_analysis_md())
        return "\n".join(report_parts)

    def _write_report(self, content: str):
        """Writes the final report content to a file."""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # --- "Render" Methods for Markdown Sections ---

    def _render_metadata_md(self) -> str:
        if not self.metadata:
            return "## Experiment Parameters\n\n*Metadata file (metadata.yaml) not found.*\n"
        
        md = ["## Experiment Parameters", "This report was generated from data collected using the following parameters:\n"]
        
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
        md = ["\n## Stage 1: Visual Overview", "These plots show the distribution of performance metrics across all technologies and runs.\n"]
        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_df = self.summary_df[self.summary_df['metric'] == metric]
                if metric_df.empty: continue
                
                md.append(f"### {config['name']}")
                path = self._create_comparison_plot(metric_df, stat_col, config['name'], plot_type='violin')
                if path: md.append(f"![Violin Plot: {config['name']}]({path.relative_to(self.input_dir)})")
        return "\n".join(md)

    def _render_temporal_analysis_md(self) -> str:
        md = ["\n## Stage 2: Temporal Analysis", "Time-series plots showing the mean metric value over the duration of the test runs.\n"]
        for metric, config in METRIC_CONFIG['mean'].items():
            path = self._create_timeseries_plot(metric, config['name'])
            if path:
                md.append(f"### {config['name']}")
                md.append(f"![{config['name']} Time-Series]({path.relative_to(self.input_dir)})")
        return "\n".join(md)

    def _render_ranking_tables_md(self) -> str:
        md = ["\n## Stage 3: Intra-Group Rankings", "Tables ranking each technology within its group. The metric shown is the mean across all runs, with the 95% confidence interval.\n"]
        
        # --- FIX #3: Refactor loop to generate dynamic headers ---
        stat_name_map = {
            'mean': 'Mean',
            'std': 'Mean of Std Devs',
            'p95': 'Mean of p95s'
        }

        for stat_col, metrics in METRIC_CONFIG.items():
            for metric, config in metrics.items():
                metric_name = config['name']
                if metric_name not in self.ranking_results:
                    continue
                
                ranking_df = self.ranking_results[metric_name]
                md.append(f"### {metric_name}")
                
                sort_ascending = config['sort_ascending']
                header_name = f"{stat_name_map.get(stat_col, stat_col.title())} (95% CI)"

                for group_name in sorted(ranking_df['group'].unique()):
                    md.append(f"\n#### Group: {group_name}\n")
                    group_data = ranking_df[ranking_df['group'] == group_name].sort_values(by='mean', ascending=sort_ascending)
                    group_data['formatted_metric'] = group_data.apply(
                        lambda r: f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]" if pd.notna(r['ci_lower']) else f"{r['mean']:.4f}", axis=1)
                    
                    output_df = group_data[['server_type', 'formatted_metric']].rename(columns={'formatted_metric': header_name})
                    md.append(output_df.to_markdown(index=False))
        # --- END FIX ---
        return "\n".join(md)

    def _render_champion_analysis_md(self) -> str:
        if len(self.champions) < 2:
            return "\n## Stage 4: Champion vs. Champion Analysis\n\n*This analysis is skipped because data from fewer than two distinct technology groups was found.*\n"
        
        champ_keys = list(self.champions.keys())
        champ1_group, champ2_group = champ_keys[0], champ_keys[1]
        champ1_tech, champ2_tech = self.champions[champ1_group], self.champions[champ2_group]

        md = ["\n## Stage 4: Champion vs. Champion Analysis"]
        md.append(f"Champions are selected based on the primary metric: **{self.primary_metric_str}**.\n")
        md.append(f"*   **{champ1_group} Champion:** `{champ1_tech}`")
        md.append(f"*   **{champ2_group} Champion:** `{champ2_tech}`\n")

        for metric_name, results in self.champion_results.items():
            md.append(f"### Statistical Comparison: {metric_name}")
            md.append(f"**Comparison:** `{results.get('champ1', 'N/A')}` vs. `{results.get('champ2', 'N/A')}`\n")
            md.append("#### Assumption Checks")
            md.append(f"*   Shapiro-Wilk p-values: `{results.get('shapiro_p1', 0):.4f}` ({champ1_group}), `{results.get('shapiro_p2', 0):.4f}` ({champ2_group})")
            md.append(f"*   Levene's Test p-value: `{results.get('levene_p', 0):.4f}`\n")
            md.append("#### Hypothesis Testing")
            md.append(f"Using **{results.get('test_name', 'N/A')}**:")
            md.append(f"*   Statistic: `{results.get('statistic', 0):.4f}`")
            md.append(f"*   p-value: `{results.get('p_value', 0):.4f}`\n")
            md.append("#### Effect Size")
            md.append(f"*   Cohen's d: `{results.get('cohen_d', 0):.4f}`")
        
        return "\n".join(md)

    # --- Plotting Methods ---

    def _get_ordered_tech_list(self, df: pd.DataFrame) -> list:
        """
        Gets a list of technologies from the dataframe, ordered according to the
        'chart_order' list in the config file. Any technologies in the dataframe
        not present in the config list are appended alphabetically.
        """
        if not self.chart_order:
            return sorted(df['server_type'].unique())

        all_techs_in_data = set(df['server_type'].unique())
        
        # Filter chart_order to only include techs present in the current data
        ordered_techs_from_config = [t for t in self.chart_order if t in all_techs_in_data]
        
        # Find techs in data that were not in the config's order list
        config_techs_set = set(ordered_techs_from_config)
        remaining_techs = sorted([t for t in all_techs_in_data if t not in config_techs_set])
        
        # The final order is the ordered list from config plus the sorted remainder
        return ordered_techs_from_config + remaining_techs

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
        
        plt.title(f'{plot_type.capitalize()} Plot Comparison of {metric_name}', fontsize=16)
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

    def _create_timeseries_plot(self, metric: str, metric_name: str) -> Path | None:
        """Creates a time-series plot for a given metric."""
        print(f"INFO: Generating time-series plot for '{metric_name}'...")
        df_metric = self.raw_df[self.raw_df['metric'] == metric]
        if df_metric.empty: return None

        max_time = df_metric['time_sec'].max()
        if pd.isna(max_time):
            print(f"WARNING: No valid time data for metric '{metric_name}'. Skipping time-series plot.")
            return None
            
        full_time_index = pd.to_timedelta(np.arange(int(max_time) + 1), unit='s')

        processed_dfs = []
        for group, group_df in df_metric.groupby(['server_type', 'run_number']):
            temp_df = group_df.set_index(pd.to_timedelta(group_df['time_sec'], unit='s'))
            temp_df = temp_df.reindex(full_time_index)
            temp_df['metric_value'] = temp_df['metric_value'].ffill()
            temp_df['metric_value'] = temp_df['metric_value'].bfill()
            temp_df['server_type'] = group[0]
            temp_df['run_number'] = group[1]
            temp_df['time_sec'] = temp_df.index.total_seconds()
            processed_dfs.append(temp_df.reset_index(drop=True))
        
        if not processed_dfs:
            print(f"WARNING: Could not process time-series data for metric '{metric_name}'.")
            return None

        plot_df = pd.concat(processed_dfs, ignore_index=True)
        
        agg_df = plot_df.groupby(['server_type', 'time_sec'])['metric_value'].agg(['mean', 'min', 'max']).reset_index()
        
        plot_order = self._get_ordered_tech_list(agg_df)

        plt.figure(figsize=(14, 8))
        palette = sns.color_palette("husl", len(plot_order))
        color_map = {tech: color for tech, color in zip(plot_order, palette)}

        for tech in plot_order:
            tech_df = agg_df[agg_df['server_type'] == tech]
            if tech_df.empty:
                continue
            color = color_map[tech]
            plt.plot(tech_df['time_sec'], tech_df['mean'], label=tech, color=color)
            plt.fill_between(tech_df['time_sec'], tech_df['min'], tech_df['max'], color=color, alpha=0.2)
        
        plt.title(f"Time-Series Analysis: {metric_name}", fontsize=16)
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric_name)
        plt.legend(title='Technology')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        filename = f"{metric_name.lower().replace(' ', '_')}_timeseries_overview.png"
        filepath = self.plots_dir / re.sub(r'[^a-z0-9_.-]', '', filename)
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
    parser.add_argument('--primary-metric', default='mean_cpu', 
                        help="Metric for champion selection (e.g., 'mean_cpu', 'p95_latency').")
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"ERROR: Input directory not found: {args.input_dir}")
    else:
        try:
            import pandas, numpy, scipy, matplotlib, seaborn, yaml
        except ImportError as e:
            print(f"ERROR: Missing dependency - {e.name}. Please install it.")
            print("You can install all dependencies with: pip install pandas numpy scipy matplotlib seaborn pyyaml")
        else:
            analyzer = PerformanceAnalyzer(args.input_dir, args.primary_metric)
            analyzer.run()
