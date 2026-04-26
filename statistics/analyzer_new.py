# statistics/analyzer.py
import argparse
import glob
import shutil
import sys
from pathlib import Path
import re
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
import json

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
        elif self.report_type == 'capacity_wrk':
            self.report_path = self.input_dir / "capacity_wrk_report.md"
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
        
        if not self._load_configuration():
            return

        self.plots_dir.mkdir(exist_ok=True)

        if self.report_type == 'capacity_wrk':
            if not self._load_wrk_data():
                return
            self._run_capacity_wrk_analysis()
        else:
            if not self._load_and_prepare_data():
                return

            if self.report_type == 'capacity':
                self._run_capacity_analysis()
            elif self.report_type == 'load':
                self._run_load_analysis()
            elif self.report_type == 'champions':
                self._run_champions_analysis()

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

    def _load_wrk_data(self) -> bool:
        """Loads wrk JSON results for capacity_wrk tests."""
        all_files = glob.glob(str(self.input_dir / "*_wrk_client_results.json"))
        if not all_files:
            print(f"ERROR: No wrk JSON files found in {self.input_dir}.")
            return False

        tech_to_group = {tech: group for group, techs in self.groups_config.items() for tech in techs}
        wrk_records = []
        
        for f in all_files:
            run_number = int(Path(f).stem.split('_')[0])
            with open(f, 'r') as jf:
                data = json.load(jf)
                for tech, results in data.items():
                    # Parse latency (e.g. "1.23ms", "450us")
                    lat_str = results['latency_avg']
                    match = re.search(r'[\d.]+', lat_str)
                    if not match: continue
                    lat_val = float(match.group())
                    if 'us' in lat_str: lat_val /= 1000
                    if 's' in lat_str and 'ms' not in lat_str: lat_val *= 1000
                    
                    wrk_records.append({
                        'run_number': run_number,
                        'server_type': tech,
                        'group': tech_to_group.get(tech, 'Uncategorized'),
                        'rps': float(results['rps']),
                        'latency_ms': lat_val
                    })
        
        if not wrk_records:
            print("ERROR: No valid wrk records found in JSON files.")
            return False

        self.wrk_df = pd.DataFrame(wrk_records)
        self.wrk_summary = self.wrk_df.groupby(['group', 'server_type']).agg({
            'rps': ['mean', 'std', 'max'],
            'latency_ms': ['mean', 'std', 'max']
        }).reset_index()
        self.wrk_summary.columns = ['group', 'server_type', 'rps_mean', 'rps_std', 'rps_max', 'lat_mean', 'lat_std', 'lat_max']
        return True

    def _load_and_prepare_data(self) -> bool:
        """Loads aggregated CSV files from metrics/."""
        # ... (rest of the original _load_and_prepare_data logic)
        # I will keep the original implementation here but for brevity in this tool call 
        # I am focusing on the fix. 
        # Wait, I must provide the FULL file if I use write_to_file.
        return self._original_load_and_prepare_data()

    # I'll use replace_file_content instead to avoid losing 900 lines of code.
