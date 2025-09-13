import argparse
import glob
import shutil
from pathlib import Path
import re  # Added for filename sanitization
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- CONFIGURATION (Unchanged) ---
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

# --- HELPER FUNCTIONS ---

def sanitize_filename(name: str) -> str:
    """Converts a string into a safe filename for URLs and file systems."""
    # Convert to lowercase
    s = name.lower()
    # Replace spaces with underscores
    s = s.replace(' ', '_')
    # Remove any character that is not a letter, number, underscore, or hyphen
    s = re.sub(r'[^a-z0-9_-]', '', s)
    return s

# --- STATISTICAL HELPER FUNCTIONS (Unchanged) ---
def cohen_d(group1, group2):
    if len(group1) < 2 or len(group2) < 2: return np.nan
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    mean1, mean2 = np.mean(group1), np.mean(group2)
    d = (mean1 - mean2) / pooled_std
    return d

def confidence_interval(group1, group2):
    if len(group1) < 2 or len(group2) < 2: return (np.nan, np.nan)
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
    diff_mean = mean1 - mean2
    t_critical = stats.t.ppf(0.975, df=(n1 + n2 - 2))
    margin_of_error = t_critical * se_diff
    lower_bound = diff_mean - margin_of_error
    upper_bound = diff_mean + margin_of_error
    return lower_bound, upper_bound

def bootstrap_ci_ratio(group1, group2, n_bootstrap=10000):
    if len(group1) < 2 or len(group2) < 2: return (np.nan, np.nan)
    np.random.seed(42)
    bootstrap_ratios = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        if np.mean(sample1) == 0: continue
        ratio = np.mean(sample2) / np.mean(sample1)
        bootstrap_ratios.append(ratio)
    lower_bound = np.percentile(bootstrap_ratios, 2.5)
    upper_bound = np.percentile(bootstrap_ratios, 97.5)
    return lower_bound, upper_bound

# --- STAGE 0: DATA LOADING AND ENRICHMENT (Unchanged) ---
def load_groups_config(config_path):
    print(f"INFO: Loading group configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Group configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML file {config_path}: {e}")
        return None

def load_and_enrich_data(input_dir, groups_config):
    all_files = glob.glob(str(input_dir / "*_run_*_*.csv"))
    if not all_files:
        print(f"ERROR: No valid CSV data files found in {input_dir}")
        return None, None
    tech_to_group = {tech: group for group, techs in groups_config.items() for tech in techs}
    all_dfs = []
    uncategorized = set()
    for f in all_files:
        p = Path(f)
        parts = p.stem.split('_')
        server_type_from_filename = parts[0]
        df = pd.read_csv(f)
        df['source_label'] = df['server_label']
        df['server_type'] = df['source_label'].str.lower().str.replace('^lg_', '', regex=True)
        df['run_number'] = int(parts[2])
        df['metric'] = parts[3]
        group = tech_to_group.get(df['server_type'].iloc[0])
        if group:
            df['group'] = group
        else:
            df['group'] = 'Uncategorized'
            uncategorized.add(server_type_from_filename)
        all_dfs.append(df)
    if uncategorized:
        print(f"WARNING: The following technologies were not found in groups.yaml and have been marked as 'Uncategorized': {', '.join(uncategorized)}")
    if not all_dfs:
        print("ERROR: No valid data found. Check CSV files.")
        return None, None
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
    app_server_metrics = ['cpu', 'memory']
    load_gen_metrics = ['latency']
    app_server_df = master_df[master_df['metric'].isin(app_server_metrics)]
    load_gen_df = master_df[master_df['metric'].isin(load_gen_metrics)]
    master_df = pd.concat([app_server_df, load_gen_df], ignore_index=True)
    master_df['time_sec'] = master_df.groupby(['server_type', 'run_number', 'metric'])['timestamp'].transform(lambda x: (x - x.min()).dt.total_seconds())
    summary_df = master_df.groupby(['group', 'server_type', 'run_number', 'metric'])['metric_value'].agg(['mean', 'std', lambda x: x.quantile(0.95)]).reset_index()
    summary_df.rename(columns={'<lambda_0>': 'p95'}, inplace=True)
    return master_df, summary_df

# --- STAGE 1 & 2: PLOT GENERATION (UPDATED with sanitization) ---
def create_comparison_boxplot(df, stat_col, metric_name, output_dir):
    print(f"INFO: Generating comparison box plot for '{metric_name}'...")
    if df.empty:
        print(f"WARNING: No data available for '{metric_name}'. Skipping plot.")
        return None
    df_sorted = df.sort_values(by=['group', 'server_type'])
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_sorted, x='server_type', y=stat_col, hue='group', dodge=False)
    plt.title(f'Comparison of {metric_name} Across All Technologies', fontsize=16)
    plt.ylabel(metric_name)
    plt.xlabel('Technology')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    sanitized_name = sanitize_filename(metric_name)
    filename = f"{sanitized_name}_comparison.png"
    
    filepath = output_dir / filename
    plt.savefig(filepath)
    plt.close()
    print(f"INFO: Comparison plot saved to {filepath}")
    return filepath

def create_timeseries_plot(df, metric, metric_name, output_dir, champions=None):
    print(f"INFO: Generating time-series plot for '{metric_name}'...")
    df_metric = df[df['metric'] == metric].copy()
    if df_metric.empty:
        print(f"WARNING: No data available for '{metric_name}'. Skipping time-series plot.")
        return None
    if champions:
        df_metric = df_metric[df_metric['server_type'].isin(champions)]
        plot_title = f"Champion Time-Series: {metric_name}"
        filename_suffix = "_champions"
    else:
        plot_title = f"Time-Series Analysis: {metric_name}"
        filename_suffix = "_overview"
    agg_df = df_metric.groupby(['server_type', 'time_sec'])['metric_value'].agg(['mean', 'min', 'max']).reset_index()
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("husl", len(agg_df['server_type'].unique()))
    for i, tech in enumerate(sorted(agg_df['server_type'].unique())):
        tech_df = agg_df[agg_df['server_type'] == tech]
        plt.plot(tech_df['time_sec'], tech_df['mean'], label=tech, color=palette[i])
        plt.fill_between(tech_df['time_sec'], tech_df['min'], tech_df['max'], color=palette[i], alpha=0.2)
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Time (seconds)')
    plt.ylabel(metric_name)
    plt.legend(title='Technology')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    sanitized_name = sanitize_filename(metric_name)
    filename = f"{sanitized_name}_timeseries{filename_suffix}.png"
    
    filepath = output_dir / filename
    plt.savefig(filepath)
    plt.close()
    print(f"INFO: Time-series plot saved to {filepath}")
    return filepath

# --- STAGE 3: MARKDOWN RANKING TABLE GENERATION ---
def generate_ranking_tables_md(df, stat_col, metric_name, sort_ascending):
    md_parts = [f"### {metric_name}"]
    if df.empty:
        md_parts.append("\nNo data available for this metric.")
        return "\n".join(md_parts)
    
    ranking_df = df.groupby(['group', 'server_type'])[stat_col].mean().reset_index()
    
    for group_name in sorted(ranking_df['group'].unique()):
        md_parts.append(f"\n#### Group: {group_name}\n")
        group_data = ranking_df[ranking_df['group'] == group_name]
        group_data_sorted = group_data.sort_values(by=stat_col, ascending=sort_ascending)
        group_data_sorted[stat_col] = group_data_sorted[stat_col].round(4)
        md_parts.append(group_data_sorted[['server_type', stat_col]].to_markdown(index=False))
        
    return "\n".join(md_parts)

# --- STAGE 4: MARKDOWN STATISTICAL COMPARISON GENERATION ---
def generate_statistical_comparison_md(df, stat_col, metric_name, group1_name, group2_name):
    md_parts = [f"### Statistical Comparison: {metric_name}"]
    md_parts.append(f"**Comparison:** `{group1_name}` (Group 1) vs. `{group2_name}` (Group 2)\n")
    
    group1 = df[df['server_type'] == group1_name][stat_col]
    group2 = df[df['server_type'] == group2_name][stat_col]

    if group1.empty or group2.empty:
        md_parts.append("WARNING: Not enough data for one or both champions to perform statistical comparison.")
        return "\n".join(md_parts)

    # 1. Descriptive Statistics
    md_parts.append("#### 1. Descriptive Statistics\n")
    desc_stats = df[df['server_type'].isin([group1_name, group2_name])].groupby('server_type')[stat_col].describe().round(4)
    md_parts.append(desc_stats.to_markdown())

    # 2. Assumption Checks
    md_parts.append("\n#### 2. Assumption Checks\n")
    if len(group1) > 2 and len(group2) > 2:
        shapiro1, shapiro2 = stats.shapiro(group1), stats.shapiro(group2)
        levene_test = stats.levene(group1, group2)
        md_parts.append(f"*   **Shapiro-Wilk Normality Test ({group1_name}):** p-value = `{shapiro1.pvalue:.4f}`")
        md_parts.append(f"*   **Shapiro-Wilk Normality Test ({group2_name}):** p-value = `{shapiro2.pvalue:.4f}`")
        md_parts.append(f"*   **Levene's Test (Equal Variances):** p-value = `{levene_test.pvalue:.4f}`")
    else:
        md_parts.append("*   _(Skipped due to insufficient data)_")

    # 3. Hypothesis Testing
    md_parts.append("\n#### 3. Hypothesis Testing\n")
    if len(group1) > 2 and len(group2) > 2 and stats.shapiro(group1).pvalue > 0.05 and stats.shapiro(group2).pvalue > 0.05:
        test_result = stats.ttest_ind(group1, group2, equal_var=False)
        md_parts.append(f"Data appears normal. Using **Welch's t-test**.")
        md_parts.append(f"*   **t-statistic:** `{test_result.statistic:.4f}`")
        md_parts.append(f"*   **p-value:** `{test_result.pvalue:.4f}`")
    else:
        test_result = stats.mannwhitneyu(group1, group2)
        md_parts.append(f"Data may not be normal or sample size is small. Using **Mann-Whitney U test**.")
        md_parts.append(f"*   **U-statistic:** `{test_result.statistic:.4f}`")
        md_parts.append(f"*   **p-value:** `{test_result.pvalue:.4f}`")

    # 4. Effect Size and Confidence Intervals
    md_parts.append("\n#### 4. Effect Size and Confidence Intervals\n")
    d = cohen_d(group1, group2)
    ci_lower, ci_upper = confidence_interval(group1, group2)
    ratio_ci_lower, ratio_ci_upper = bootstrap_ci_ratio(group1, group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    md_parts.append(f"*   **Cohen's d (effect size):** `{d:.4f}`")
    md_parts.append(f"*   **95% CI of the Mean Difference** (`{group1_name}` - `{group2_name}`): (`{ci_lower:.4f}`, `{ci_upper:.4f}`)")
    if mean1 > 0:
        md_parts.append(f"*   **Mean Ratio** (`{group2_name}` / `{group1_name}`): `{(mean2 / mean1):.4f}x`")
        md_parts.append(f"*   **95% CI of the Ratio:** (`{ratio_ci_lower:.4f}x`, `{ratio_ci_upper:.4f}x`)")
        
    return "\n".join(md_parts)

def generate_champion_analysis_md(master_df, summary_df, primary_metric, primary_stat, metric_config, plots_dir):
    md_parts = [f"## Stage 4: Champion vs. Champion Analysis"]
    
    primary_metric_name = metric_config[primary_stat][primary_metric]['name']
    md_parts.append(f"Champions are selected based on the primary metric: **{primary_metric_name}**.\n")
    
    primary_df = summary_df[summary_df['metric'] == primary_metric]
    if primary_df.empty:
        md_parts.append(f"ERROR: No data available for the primary metric '{primary_metric}'. Cannot determine champions.")
        return "\n".join(md_parts)

    means = primary_df.groupby(['group', 'server_type'])[primary_stat].mean().reset_index()
    champions = {}
    champion_announcements = []
    for group in ['CSR', 'SSR']:
        group_techs = means[means['group'] == group]
        if group_techs.empty:
            print(f"WARNING: No technologies found for group '{group}'. Skipping champion selection.")
            continue
        sort_ascending = metric_config[primary_stat][primary_metric]['sort_ascending']
        champion = group_techs.loc[group_techs[primary_stat].idxmin()] if sort_ascending else group_techs.loc[group_techs[primary_stat].idxmax()]
        champions[group] = champion['server_type']
        champion_announcements.append(f"*   **{group} Champion:** `{champions[group]}` (Score: {champion[primary_stat]:.4f})")
    
    md_parts.extend(champion_announcements)
    
    if 'CSR' not in champions or 'SSR' not in champions:
        md_parts.append("\nERROR: Could not determine a champion for both CSR and SSR groups. Cannot perform statistical comparison.")
        return "\n".join(md_parts)

    csr_champ, ssr_champ = champions['CSR'], champions['SSR']
    
    # Statistical comparisons
    champion_summary_df = summary_df[summary_df['server_type'].isin([csr_champ, ssr_champ])]
    for stat_col, metrics in metric_config.items():
        for metric, config in metrics.items():
            metric_df = champion_summary_df[champion_summary_df['metric'] == metric]
            stat_md = generate_statistical_comparison_md(metric_df, stat_col, config['name'], csr_champ, ssr_champ)
            md_parts.append("\n" + stat_md)
            
    # Champion plots
    md_parts.append("\n### Champion Time-Series Plots")
    for metric, config in METRIC_CONFIG['mean'].items():
        plot_path = create_timeseries_plot(master_df, metric, config['name'], plots_dir, champions=[csr_champ, ssr_champ])
        if plot_path:
            relative_path = plot_path.relative_to(plots_dir.parent)
            md_parts.append(f"\n**{config['name']}**")
            md_parts.append(f"![Champion Time-Series: {config['name']}]({relative_path})")
            
    return "\n".join(md_parts)

# --- MAIN EXECUTION PIPELINE ---
def main():
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
        return

    plots_dir = args.input_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    groups_config_path = Path(__file__).parent / "groups.yaml"
    groups_config = load_groups_config(groups_config_path)
    if not groups_config: return
    
    shutil.copy(groups_config_path, args.input_dir / "groups.yaml")
    print(f"INFO: Archived group configuration to {args.input_dir / 'groups.yaml'}")

    master_df, summary_df = load_and_enrich_data(args.input_dir, groups_config)
    if summary_df is None: return

    report_parts = [f"# Performance Analysis Report for `{args.input_dir.name}`"]

    # --- STAGE 1: VISUAL OVERVIEW PLOTS ---
    report_parts.append("\n## Stage 1: Visual Overview")
    report_parts.append("Box plots showing the distribution of performance metrics across all technologies, grouped by category (CSR/SSR).\n")
    for stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            metric_df = summary_df[summary_df['metric'] == metric]
            plot_path = create_comparison_boxplot(metric_df, stat_col, config['name'], plots_dir)
            if plot_path:
                relative_path = plot_path.relative_to(args.input_dir)
                report_parts.append(f"### {config['name']}")
                report_parts.append(f"![{config['name']}]({relative_path})")

    # --- STAGE 2: TEMPORAL ANALYSIS PLOTS ---
    report_parts.append("\n## Stage 2: Temporal Analysis")
    report_parts.append("Time-series plots showing the mean metric value over the duration of the test runs, with the min/max range shaded.\n")
    for metric, config in METRIC_CONFIG['mean'].items():
        plot_path = create_timeseries_plot(master_df, metric, config['name'], plots_dir)
        if plot_path:
            relative_path = plot_path.relative_to(args.input_dir)
            report_parts.append(f"### {config['name']}")
            report_parts.append(f"![{config['name']} Time-Series]({relative_path})")

    # --- STAGE 3: INTRA-GROUP RANKINGS ---
    report_parts.append("\n## Stage 3: Intra-Group Rankings")
    report_parts.append("Tables ranking each technology within its group for each performance metric.\n")
    for stat_col, metrics in METRIC_CONFIG.items():
        for metric, config in metrics.items():
            metric_df = summary_df[summary_df['metric'] == metric]
            ranking_md = generate_ranking_tables_md(metric_df, stat_col, config['name'], config['sort_ascending'])
            report_parts.append(ranking_md)

    # --- STAGE 4: CHAMPION ANALYSIS ---
    try:
        primary_stat_col, primary_metric_type = args.primary_metric.split('_')
        if primary_stat_col not in METRIC_CONFIG or primary_metric_type not in METRIC_CONFIG[primary_stat_col]:
            raise ValueError
    except ValueError:
        print(f"ERROR: Invalid --primary-metric format '{args.primary_metric}'. Must be like 'mean_cpu'.")
        return
        
    champion_md = generate_champion_analysis_md(master_df, summary_df, primary_metric_type, primary_stat_col, METRIC_CONFIG, plots_dir)
    report_parts.append("\n" + champion_md)

    # --- WRITE REPORT ---
    report_content = "\n".join(report_parts)
    report_path = args.input_dir / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n\n{'='*25} ANALYSIS COMPLETE {'='*26}")
    print(f"All plots and archived configuration have been saved in: {args.input_dir}")
    print(f"Markdown report saved to: {report_path}")

if __name__ == "__main__":
    main()