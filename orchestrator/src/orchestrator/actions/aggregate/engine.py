import datetime
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from rich.console import Console

from orchestrator.config import RESULTS_DIR

console = Console()


class SourceSpec:
    def __init__(self, raw_spec: str):
        self.raw = raw_spec
        self.path: Path = Path()
        self.include_apps: Optional[Set[str]] = None
        self.exclude_apps: Optional[Set[str]] = None
        self._parse()

    def _parse(self) -> None:
        # Matches: path/to/dir or path/to/dir[app1,app2] or path/to/dir[!app3]
        match = re.match(r"^([^\[\]]+)(?:\[(.*)\])?$", self.raw)
        if not match:
            raise ValueError(f"Invalid source spec: {self.raw}")

        path_str, filter_str = match.groups()
        self.path = Path(path_str)

        if filter_str:
            filters = [f.strip() for f in filter_str.split(",")]
            include = {f for f in filters if not f.startswith("!")}
            exclude = {f[1:] for f in filters if f.startswith("!")}

            if include:
                self.include_apps = include
            if exclude:
                self.exclude_apps = exclude

    def should_include(self, app_name: str) -> bool:
        # Standardize app name (dashes to underscores often happen in filenames)
        normalized_app = app_name.replace("_", "-")

        if self.include_apps is not None:
            return (
                app_name in self.include_apps
                or normalized_app in self.include_apps
                or app_name.lower() in self.include_apps
                or normalized_app.lower() in self.include_apps
            )

        if self.exclude_apps is not None:
            return not (
                app_name in self.exclude_apps
                or normalized_app in self.exclude_apps
                or app_name.lower() in self.exclude_apps
                or normalized_app.lower() in self.exclude_apps
            )

        return True


class DataAggregator:
    def __init__(
        self,
        source_specs: List[str],
        output_dir: Optional[Path] = None,
        lax: bool = False,
        copy_logs: bool = True,
    ):
        self.specs = [SourceSpec(s) for s in source_specs]
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

    def run(self) -> None:
        console.print(f"[bold blue]Starting aggregation into:[/bold blue] {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "tool_results").mkdir(exist_ok=True)
        if self.copy_logs:
            (self.output_dir / "logs").mkdir(exist_ok=True)

        for spec in self.specs:
            self._process_source(spec)

        self._save_metadata()
        self._print_summary()

    def _process_source(self, spec: SourceSpec) -> None:
        if not spec.path.exists():
            console.print(f"[bold red]Source path does not exist:[/bold red] {spec.path}")
            return

        # 1. Validate/Load Metadata
        meta_path = spec.path / "metadata.yaml"
        if not meta_path.exists():
            console.print(
                f"[yellow]Warning: No metadata.yaml in {spec.path}. Skipping validation.[/yellow]"
            )
            meta = {}
        else:
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)

            if not self.master_metadata:
                self.master_metadata = meta
            else:
                self._check_consistency(spec.path, meta)

        # 2. Discover Runs
        metrics_dir = spec.path / "metrics"
        if not metrics_dir.exists():
            console.print(f"[yellow]No metrics found in {spec.path}[/yellow]")
            return

        # Map local run numbers to global sequential
        filename_regex = re.compile(r"^(\d+)_(.*)\.csv$")
        local_files = list(metrics_dir.glob("*.csv"))

        run_mapping: Dict[int, int] = {}
        local_runs_set: Set[int] = set()
        for file_path in local_files:
            m = filename_regex.match(file_path.name)
            if m:
                local_runs_set.add(int(m.group(1)))

        local_runs = sorted(local_runs_set)

        for local_run in local_runs:
            self.global_run_counter += 1
            run_mapping[local_run] = self.global_run_counter

        # 3. Copy Files
        self._copy_data(spec, metrics_dir, run_mapping, "metrics", ".csv")
        self._copy_data(spec, spec.path / "tool_results", run_mapping, "tool_results", "_wrk.json")
        if self.copy_logs:
            self._copy_data(spec, spec.path / "logs", run_mapping, "logs", ".log")

        self.lineage.append(
            {
                "source": str(spec.path.absolute()),
                "filter": spec.raw.split("[")[1][:-1] if "[" in spec.raw else None,
                "mapped_runs": run_mapping,
            }
        )

    def _copy_data(
        self,
        spec: SourceSpec,
        src_dir: Path,
        mapping: Dict[int, int],
        dest_subdir: str,
        suffix: str,
    ) -> None:
        if not src_dir.exists():
            return

        # Regex to match run_app_suffix
        # e.g. 01_nextjs_bun.csv
        # e.g. 01_nextjs_bun_wrk.json
        regex = re.compile(rf"^(\d+)_(.*){re.escape(suffix)}$")

        for file_path in src_dir.iterdir():
            match = regex.match(file_path.name)
            if not match:
                continue

            local_run = int(match.group(1))
            app_name = match.group(2)

            if local_run not in mapping:
                continue

            if not spec.should_include(app_name):
                continue

            global_run = mapping[local_run]
            new_name = f"{global_run:02d}_{app_name}{suffix}"
            dest_path = self.output_dir / dest_subdir / new_name

            shutil.copy2(file_path, dest_path)

            if dest_subdir == "metrics":
                self.app_counts[app_name] = self.app_counts.get(app_name, 0) + 1

    def _check_consistency(self, path: Path, meta: Dict[str, Any]) -> None:
        if self.lax:
            return

        # Compare parameters
        p1 = self.master_metadata.get("parameters", {})
        p2 = meta.get("parameters", {})

        # Basic consistency check of the experiment configuration
        keys_to_check = [
            "test_type",
            "capacity_k6_options",
            "load_options",
            "capacity_wrk_options",
        ]
        for k in keys_to_check:
            if p1.get(k) != p2.get(k):
                console.print(
                    f"[bold red]Consistency Error in {path}:[/bold red] parameter '{k}' differs."
                )
                console.print(f"Master: {p1.get(k)}")
                console.print(f"Source: {p2.get(k)}")
                raise ValueError(f"Inconsistent metadata in {path}. Use --lax to bypass.")

    def _save_metadata(self) -> None:
        if not self.master_metadata:
            # Fallback if no metadata was ever found
            self.master_metadata = {"test_type": "unknown"}

        new_meta = self.master_metadata.copy()
        new_meta["aggregated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_meta["lineage"] = self.lineage
        new_meta["num_runs"] = self.global_run_counter

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
