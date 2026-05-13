import datetime
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console

from orchestrator.config import RESULTS_DIR
from orchestrator.shared.research import ResearchArtifact

console = Console()


# SourceSpec will be simplified to just handle filters,
# while ResearchArtifact handles the paths.


class DataAggregator:
    def __init__(
        self,
        source_specs: List[str],
        output_dir: Optional[Path] = None,
        lax: bool = False,
        copy_logs: bool = True,
    ):
        self.source_specs_raw = source_specs
        self.lax = lax
        self.copy_logs = copy_logs

        if output_dir:
            self.output_dir = output_dir
        else:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = RESULTS_DIR / f"aggregated_{ts}"

        self.global_run_counter = 0
        self.master_metadata: Dict[str, Any] = {}
        self.lineage: List[Dict[str, Any]] = []
        self.app_counts: Dict[str, int] = {}
        self.inconsistency_detected = False

    def run(self) -> None:
        console.print(f"[bold blue]Starting aggregation into:[/bold blue] {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "tool_results").mkdir(exist_ok=True)
        if self.copy_logs:
            (self.output_dir / "logs").mkdir(exist_ok=True)

        for spec_raw in self.source_specs_raw:
            # Parse spec: path[include,!exclude]
            match = re.match(r"^([^\[\]]+)(?:\[(.*)\])?$", spec_raw)
            if not match:
                console.print(f"[bold red]Invalid source specification:[/bold red] {spec_raw}")
                continue
            path_str, filter_str = match.groups()
            src_path = Path(path_str)

            # Use the deep Research Artifact
            src_artifact = ResearchArtifact(src_path)
            if not src_path.exists():
                console.print(f"[bold red]Source path does not exist:[/bold red] {src_path}")
                continue

            # 1. Consistency Check (Research Contract)
            if not self.master_metadata:
                self.master_metadata = src_artifact.metadata
            else:
                diffs = src_artifact.check_compatibility(self.master_metadata)
                if diffs:
                    if not self.lax:
                        console.print(f"[bold red]Inconsistency in {src_path}:[/bold red]")
                        for d in diffs:
                            console.print(f"  - {d}")
                        raise ValueError("Research Contract violated. Use --lax to override.")
                    else:
                        console.print(f"[yellow]Warning: Overriding inconsistency in {src_path}[/yellow]")
                        self.inconsistency_detected = True

            # 2. Process Runs
            include = None
            exclude = None
            if filter_str:
                filters = [f.strip() for f in filter_str.split(",")]
                include = {f for f in filters if not f.startswith("!")}
                exclude = {f[1:] for f in filters if f.startswith("!")}

            runs = src_artifact.get_runs(include_apps=include, exclude_apps=exclude)

            # Map local run IDs to global sequence
            local_runs = sorted(list(set(r.run_id for r in runs)))
            run_mapping = {
                local: (i + 1 + self.global_run_counter) for i, local in enumerate(local_runs)
            }

            for run in runs:
                global_id = run_mapping[run.run_id]

                # Copy Metrics
                if run.metrics_path:
                    new_name = f"{global_id:02d}_{run.server_type.replace('-', '_')}.csv"
                    shutil.copy2(run.metrics_path, self.output_dir / "metrics" / new_name)
                    self.app_counts[run.server_type] = self.app_counts.get(run.server_type, 0) + 1

                # Copy Results
                if run.results_path:
                    new_name = f"{global_id:02d}_{run.server_type.replace('-', '_')}_wrk.json"
                    shutil.copy2(run.results_path, self.output_dir / "tool_results" / new_name)

                # Copy Logs
                if self.copy_logs and run.logs_path:
                    new_name = f"{global_id:02d}_{run.server_type.replace('-', '_')}.log"
                    shutil.copy2(run.logs_path, self.output_dir / "logs" / new_name)

            if local_runs:
                self.global_run_counter = max(run_mapping.values())

            self.lineage.append(
                {
                    "source": str(src_path.absolute()),
                    "filter": filter_str,
                    "mapped_runs": run_mapping,
                }
            )

        self._save_metadata()
        self._print_summary()

    def _save_metadata(self) -> None:
        if not self.master_metadata:
            # Fallback if no metadata was ever found
            self.master_metadata = {"test_type": "unknown"}

        new_meta = self.master_metadata.copy()
        new_meta["aggregated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_meta["lineage"] = self.lineage
        new_meta["num_runs"] = self.global_run_counter
        new_meta["is_consistent"] = not self.inconsistency_detected

        with open(self.output_dir / "metadata.yaml", "w") as f:
            yaml.dump(new_meta, f)

    def _print_summary(self) -> None:
        console.print("\n[bold green]Aggregation Complete![/bold green]")
        console.print(f"Total runs: {self.global_run_counter}")
        console.print("Sample counts per app:")
        if not self.app_counts:
            console.print(" [yellow]No apps were aggregated![/yellow]")
        for app, count in sorted(self.app_counts.items()):
            console.print(f" - {app}: {count} runs")
        console.print(f"\nResults saved to: [cyan]{self.output_dir}[/cyan]")
