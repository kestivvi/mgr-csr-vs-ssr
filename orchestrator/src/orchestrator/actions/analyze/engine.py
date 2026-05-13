import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from rich.console import Console

from orchestrator.shared.research import Experiment, ExperimentLoader

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

        self.groups_config: Dict[str, List[str]] = {}
        self.chart_order: List[str] = []
        self.experiment: Optional[Experiment] = None

        # Legacy properties for report compatibility (will be refactored)
        self.raw_df = pd.DataFrame()
        self.wrk_df = pd.DataFrame()

    def run(self) -> None:
        console.print(f"[bold cyan]Starting analysis for:[/bold cyan] {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        if not self._load_configuration():
            console.print("[bold red]Analysis aborted: Configuration loading failed.[/bold red]")
            return

        # Use the deep loader
        loader = ExperimentLoader(groups_config=self.groups_config)
        try:
            self.experiment = loader.load(self.input_dir)
            self.raw_df = self.experiment.metrics
            if self.experiment.wrk_results is not None:
                self.wrk_df = self.experiment.wrk_results
        except Exception as e:
            console.print(f"[bold red]Failed to load experiment data:[/bold red] {e}")
            return

        if self.report_type == "capacity_wrk":
            run_capacity_wrk_analysis(self)
        elif self.report_type == "capacity_k6":
            run_capacity_k6_analysis(self)
        elif self.report_type == "load":
            run_load_analysis(self)
        elif self.report_type == "champions":
            run_champions_analysis(self)

        console.print(
            f"[bold green]Analysis complete. Report saved to: {self.report_path}[/bold green]"
        )

    def _load_configuration(self) -> bool:
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
