from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResearchRun:
    """
    A logical unit of an experiment, grouping all files for one technology instance.
    This is a "Deep" representation of a (run_number, server_type) pair.
    """

    run_id: int
    server_type: str
    metrics_path: Path | None = None
    results_path: Path | None = None
    logs_path: Path | None = None

    @property
    def is_complete(self) -> bool:
        """A run is complete if it has both metrics and tool results."""
        return self.metrics_path is not None and self.results_path is not None


class ResearchArtifact:
    """
    A deep module managing the physical storage of experimental data.
    Acts as the "File Surgeon" for aggregation and the "Guard" for analysis.
    """

    path: Path
    metadata: dict[str, Any]
    warnings: list[str]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata = self._load_metadata()
        self.warnings = []

    def _load_metadata(self) -> dict[str, Any]:
        meta_path = self.path / "metadata.yaml"
        if not meta_path.exists():
            return {}
        with open(meta_path, "r") as f:
            return yaml.safe_load(f) or {}

    def get_runs(
        self, include_apps: set[str] | None = None, exclude_apps: set[str] | None = None
    ) -> list[ResearchRun]:
        """
        Discovers and pairs all runs within the artifact.
        Hides the complexity of filename regexes from the rest of the system.
        """
        runs_map: dict[tuple[int, str], ResearchRun] = {}

        # 1. Scan Metrics
        metrics_dir = self.path / "metrics"
        if metrics_dir.exists():
            for p in metrics_dir.glob("*.csv"):
                run_id, tech = self._parse_filename(p.name, ".csv")
                if (
                    run_id is not None
                    and tech is not None
                    and self._should_include(tech, include_apps, exclude_apps)
                ):
                    key = (run_id, tech)
                    runs_map[key] = ResearchRun(run_id=run_id, server_type=tech, metrics_path=p)

        # 2. Scan Tool Results (wrk/k6)
        results_dir = self.path / "tool_results"
        if results_dir.exists():
            for p in results_dir.glob("*_wrk.json"):
                run_id, tech = self._parse_filename(p.name, "_wrk.json")
                if (
                    run_id is not None
                    and tech is not None
                    and self._should_include(tech, include_apps, exclude_apps)
                ):
                    key = (run_id, tech)
                    if key not in runs_map:
                        runs_map[key] = ResearchRun(run_id=run_id, server_type=tech)
                    runs_map[key].results_path = p

        # 3. Scan Logs
        logs_dir = self.path / "logs"
        if logs_dir.exists():
            for p in logs_dir.glob("*.log"):
                run_id, tech = self._parse_filename(p.name, ".log")
                if (
                    run_id is not None
                    and tech is not None
                    and self._should_include(tech, include_apps, exclude_apps)
                ):
                    key = (run_id, tech)
                    if key not in runs_map:
                        runs_map[key] = ResearchRun(run_id=run_id, server_type=tech)
                    runs_map[key].logs_path = p

        return sorted(runs_map.values(), key=lambda r: (r.run_id, r.server_type))

    def _parse_filename(self, filename: str, suffix: str) -> tuple[int | None, str | None]:
        """Internal helper to extract (run_id, server_type) from a filename."""
        regex = re.compile(rf"^(\d+)_(.*){re.escape(suffix)}$")
        match = regex.match(filename)
        if match:
            return int(match.group(1)), match.group(2).replace("_", "-")
        return None, None

    def _should_include(
        self, tech: str, include: set[str] | None, exclude: set[str] | None
    ) -> bool:
        if include and tech not in include:
            return False
        if exclude and tech in exclude:
            return False
        return True

    @property
    def is_consistent(self) -> bool:
        """The 'Vouching' flag for the Research Contract."""
        return self.metadata.get("is_consistent", True) and not self.warnings

    def check_compatibility(self, other: ResearchArtifact | dict[str, Any]) -> list[str]:
        """
        Compares this artifact against another to find Research Contract violations.
        """
        m1 = self.metadata.get("parameters", {})
        if isinstance(other, dict):
            m2 = other.get("parameters", {})
        else:
            m2 = other.metadata.get("parameters", {})

        diffs = []
        # 1. Global Research Contract
        keys = ["test_type", "rate", "warmup_duration"]
        for k in keys:
            if m1.get(k) != m2.get(k):
                diffs.append(f"Global Contract violation: {k} differs")

        # 2. Tool-specific Contract (Deep check)
        # Check capacity_k6_options
        opt1 = m1.get("capacity_k6_options", {})
        opt2 = m2.get("capacity_k6_options", {})
        k6_keys = ["peak_rate", "max_vus", "ramp_up", "sustain"]
        for k in k6_keys:
            if opt1.get(k) != opt2.get(k):
                diffs.append(f"K6 Contract violation: {k} differs ({opt1.get(k)} vs {opt2.get(k)})")

        return diffs
