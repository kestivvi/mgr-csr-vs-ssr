import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

def cohen_d(group1, group2):
    """Calculates Cohen's d for independent samples."""
    if len(group1) < 2 or len(group2) < 2:
        return np.nan # Not enough data to calculate
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    mean1, mean2 = np.mean(group1), np.mean(group2)
    d = (mean1 - mean2) / pooled_std
    return d

def confidence_interval(group1, group2):
    """Calculates the 95% confidence interval for the difference between two independent samples."""
    if len(group1) < 2 or len(group2) < 2:
        return (np.nan, np.nan) # Not enough data to calculate
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
    """Calculates the bootstrap 95% CI for the ratio of means (mean(group2)/mean(group1))."""
    if len(group1) < 2 or len(group2) < 2:
        return (np.nan, np.nan) # Not enough data to calculate
        
    np.random.seed(42) # for reproducibility
    bootstrap_ratios = []
    for _ in range(n_bootstrap):
        # Sample with replacement from each group
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        
        # Calculate the ratio of means for the bootstrap sample
        # Avoid division by zero, though unlikely with resource usage data
        if np.mean(sample1) == 0:
            continue
        ratio = np.mean(sample2) / np.mean(sample1)
        bootstrap_ratios.append(ratio)
        
    # Calculate the confidence interval from the percentiles of the bootstrap distribution
    lower_bound = np.percentile(bootstrap_ratios, 2.5)
    upper_bound = np.percentile(bootstrap_ratios, 97.5)
    
    return lower_bound, upper_bound

def analyze_metric(df, metric_col, metric_name):
    """Performs and prints a full statistical analysis for a given metric."""
    print(f"\n\n{'='*20} Analysis for: {metric_name.upper()} {'='*20}")

    group_csr = df[df['server_type'] == 'csr'][metric_col]
    group_ssr = df[df['server_type'] == 'ssr'][metric_col]

    # 1. Descriptive Statistics
    print("\n--- 1. Descriptive Statistics ---")
    print(df.groupby('server_type')[metric_col].describe().round(4))

    # 2. Assumption Checks
    print("\n--- 2. Assumption Checks ---")
    shapiro_csr = stats.shapiro(group_csr)
    shapiro_ssr = stats.shapiro(group_ssr)
    levene_test = stats.levene(group_csr, group_ssr)

    print(f"Shapiro-Wilk Test for Normality (CSR): Statistic={shapiro_csr.statistic:.4f}, p-value={shapiro_csr.pvalue:.4f}")
    print(f"Shapiro-Wilk Test for Normality (SSR): Statistic={shapiro_ssr.statistic:.4f}, p-value={shapiro_ssr.pvalue:.4f}")
    print(f"Levene's Test for Equal Variances: Statistic={levene_test.statistic:.4f}, p-value={levene_test.pvalue:.4f}")

    is_normal_csr = shapiro_csr.pvalue > 0.05
    is_normal_ssr = shapiro_ssr.pvalue > 0.05
    has_equal_variance = levene_test.pvalue > 0.05

    # 3. Inferential Statistics (Hypothesis Testing)
    print("\n--- 3. Hypothesis Testing ---")
    if is_normal_csr and is_normal_ssr:
        if has_equal_variance:
            print("INFO: Data is normal with equal variances. Using Student's t-test.")
            test_result = stats.ttest_ind(group_csr, group_ssr, equal_var=True)
            print(f"Student's t-test: t-statistic={test_result.statistic:.4f}, p-value={test_result.pvalue:.4f}")
        else:
            print("INFO: Data is normal with unequal variances. Using Welch's t-test.")
            test_result = stats.ttest_ind(group_csr, group_ssr, equal_var=False)
            print(f"Welch's t-test: t-statistic={test_result.statistic:.4f}, p-value={test_result.pvalue:.4f}")
    else:
        print("INFO: Data is not normally distributed. Using Mann-Whitney U test.")
        test_result = stats.mannwhitneyu(group_csr, group_ssr)
        print(f"Mann-Whitney U test: U-statistic={test_result.statistic:.4f}, p-value={test_result.pvalue:.4f}")

    # 4. Effect Size and Confidence Interval
    print("\n--- 4. Effect Size and Confidence Interval ---")
    d = cohen_d(group_csr, group_ssr)
    ci_lower, ci_upper = confidence_interval(group_csr, group_ssr)
    ratio_ci_lower, ratio_ci_upper = bootstrap_ci_ratio(group_csr, group_ssr)

    mean_csr = np.mean(group_csr)
    mean_ssr = np.mean(group_ssr)

    print(f"Cohen's d (effect size): {d:.4f}")
    print(f"95% CI of the Difference (CSR - SSR): ({ci_lower:.4f}, {ci_upper:.4f})")
    
    # Calculate and print the ratio of means, checking for division by zero
    if mean_csr > 0:
        print(f"Mean Ratio (SSR/CSR): {(mean_ssr / mean_csr):.4f}x")
        print(f"95% CI of the Ratio (SSR uses X times more resources): ({ratio_ci_lower:.4f}x, {ratio_ci_upper:.4f}x)")

def main():
    parser = argparse.ArgumentParser(description="Analyze performance data from experiment runs.")
    parser.add_argument('--input-dir', required=True, type=Path, help="Directory containing the raw CSV data files.")
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return

    # 1. Load and aggregate data
    all_files = glob.glob(str(args.input_dir / "*.csv"))
    if not all_files:
        print(f"ERROR: No CSV files found in {args.input_dir}")
        return

    all_dfs = []
    for f in all_files:
        p = Path(f)
        parts = p.stem.split('_') # e.g., ['csr', 'run', '01', 'cpu']
        df = pd.read_csv(f)
        df['server_type'] = parts[0]
        df['run_number'] = int(parts[2])
        df['metric'] = parts[3]
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    # 2. Pivot and calculate summary stats for each run
    summary_df = master_df.groupby(['server_type', 'run_number', 'metric'])['metric_value'].agg(['mean', 'std', lambda x: x.quantile(0.95)]).reset_index()
    summary_df.rename(columns={'<lambda_0>': 'p95'}, inplace=True)

    # 3. Perform analysis for each metric
    analyze_metric(summary_df[summary_df['metric'] == 'cpu'], 'mean', 'Mean CPU Usage')
    analyze_metric(summary_df[summary_df['metric'] == 'memory'], 'mean', 'Mean Memory Usage')

    analyze_metric(summary_df[summary_df['metric'] == 'cpu'], 'std', 'CPU Usage Stability (Std Dev)')
    analyze_metric(summary_df[summary_df['metric'] == 'memory'], 'std', 'Memory Usage Stability (Std Dev)')

    analyze_metric(summary_df[summary_df['metric'] == 'cpu'], 'p95', 'Peak CPU Usage (95th Percentile)')
    analyze_metric(summary_df[summary_df['metric'] == 'memory'], 'p95', 'Peak Memory Usage (95th Percentile)')


if __name__ == "__main__":
    main() 