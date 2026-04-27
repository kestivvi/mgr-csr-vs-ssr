import datetime
import glob
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from rich.console import Console

from .models import ChampionResult
from .reports.capacity_k6 import run_capacity_k6_analysis
from .reports.capacity_wrk import run_capacity_wrk_analysis
from .reports.champions import run_champions_analysis
from .reports.load import run_load_analysis

console = Console()


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
        self.champion_results: Dict[str, ChampionResult] = {}
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
            run_capacity_wrk_analysis(self)
        else:
            if not self._load_and_prepare_data():
                console.print(
                    "[bold red]Analysis aborted: No valid metric data found in "
                    f"{self.input_dir / 'metrics'}.[/bold red]"
                )
                return
            if self.report_type == "capacity_k6":
                run_capacity_k6_analysis(self)
            elif self.report_type == "load":
                run_load_analysis(self)
            elif self.report_type == "champions":
                run_champions_analysis(self)

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
        all_files = glob.glob(str(self.input_dir / "tool_results" / "*_wrk.json"))
        if not all_files:
            console.print(
                "[bold red]No wrk result files found in "
                f"{self.input_dir / 'tool_results'}[/bold red]"
            )
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
