import datetime
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import ansible_runner
import yaml
from rich.console import Console

from orchestrator.actions.test.collector import collect_metrics
from orchestrator.config import (
    ANSIBLE_DIR,
    ANSIBLE_INVENTORY,
    RESULTS_DIR,
)

console = Console()

# Playbook paths (relative to ANSIBLE_DIR/project)
LOAD_PLAYBOOK = "ops/test_load_run.yml"
CAPACITY_PLAYBOOK = "ops/test_capacity_run.yml"
WRK_PLAYBOOK = "ops/test_wrk_run.yml"
TEARDOWN_PLAYBOOK = "ops/test_teardown.yml"

TIMESTAMP_MARKER = "ORCHESTRATOR_TIMESTAMPS::"
WRK_RESULT_MARKER = "WRK_RESULTS::"


def parse_duration_to_seconds(duration_str: str) -> int:
    if not duration_str:
        return 0
    match = re.match(r"(\d+)([smh])", duration_str)
    if not match:
        return 0
    value, unit = int(match.group(1)), match.group(2)
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    return 0


class TestRunner:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.test_type = self.config.get("test_type")
        self.num_runs = self.config.get("num_runs", 1)
        self.results_base_dir = (
            RESULTS_DIR
            / f"{self.test_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self.results_base_dir.mkdir(parents=True, exist_ok=True)

        self.manager = Manager()
        self.shutdown_event = self.manager.Event()

    def _extract_marker(self, output: str, marker: str) -> Optional[Dict[str, Any]]:
        for line in output.splitlines():
            if marker in line:
                try:
                    json_str = line.split(marker, 1)[1].strip().strip("'\"")
                    return cast(Dict[str, Any], json.loads(json_str))
                except Exception:
                    continue
        return None

    def run_scenario(self, run_number: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
        if self.shutdown_event.is_set():
            return {"success": False}

        scenario_name = scenario["name"]
        run_prefix = f"{run_number:02d}"
        tool = "wrk" if self.test_type == "capacity_wrk" else "k6"

        console.print(f"[{run_prefix}:{scenario_name}] Starting {tool} test.")

        extra_vars = {
            "target_host_group": scenario["load_generator_group"],
            "target_url": f"https://{scenario['app_server_ip']}",
            "server_type": scenario_name,
            "prometheus_url": f"http://{scenario['monitoring_private_ip']}:9090",
        }

        # Logic for tool-specific vars (simplified for now)
        playbook = LOAD_PLAYBOOK  # Default
        if tool == "wrk":
            playbook = WRK_PLAYBOOK
            extra_vars.update(self.config.get("wrk_options", {}))
        elif self.test_type == "capacity_k6":
            playbook = CAPACITY_PLAYBOOK
            extra_vars.update(self.config.get("capacity_options", {}))
        else:
            extra_vars.update(self.config.get("load_options", {}))

        # Run via ansible-runner
        r = ansible_runner.run(
            private_data_dir=str(ANSIBLE_DIR),
            playbook=playbook,
            extravars=extra_vars,
            quiet=True,
        )

        output = r.stdout.read()
        timestamps = self._extract_marker(output, TIMESTAMP_MARKER)
        wrk_results = self._extract_marker(output, WRK_RESULT_MARKER) if tool == "wrk" else None

        if r.rc != 0 or not timestamps:
            console.print(f"[bold red][{run_prefix}:{scenario_name}] Failed.[/bold red]")
            # Run teardown
            ansible_runner.run(
                private_data_dir=str(ANSIBLE_DIR),
                playbook=TEARDOWN_PLAYBOOK,
                extravars={
                    "target_host_group": scenario["load_generator_group"],
                    "server_type": scenario_name,
                },
                quiet=True,
            )
            return {"success": False, "name": scenario_name}

        return {
            "success": True,
            "name": scenario_name,
            "timestamps": timestamps,
            "wrk_results": wrk_results,
            "scenario": scenario,
        }

    def run_all(self) -> None:
        # 1. Parse Inventory
        # (Assuming parse_inventory logic is moved here or shared)
        scenarios = self._parse_inventory()

        for run in range(1, self.num_runs + 1):
            if self.shutdown_event.is_set():
                break

            console.print(f"--- Starting Run {run} of {self.num_runs} ---")

            run_results = []
            with ProcessPoolExecutor(max_workers=len(scenarios)) as executor:
                futures = [executor.submit(self.run_scenario, run, s) for s in scenarios]
                for f in as_completed(futures):
                    run_results.append(f.result())

            successful = [r for r in run_results if r.get("success") and r.get("timestamps")]
            if not successful:
                continue

            # Sync timestamps
            start_ts = max(r["timestamps"]["start"] for r in successful)
            end_ts = min(r["timestamps"]["end"] for r in successful)

            if start_ts >= end_ts:
                console.print("[red]No overlapping time window found.[/red]")
                continue

            for res in successful:
                collect_metrics(
                    prometheus_url=f"http://{res['scenario']['monitoring_public_ip']}:9090",
                    start_epoch=start_ts,
                    end_epoch=end_ts,
                    server_type=res["name"],
                    run_number=run,
                    output_dir=self.results_base_dir,
                )

        console.print(
            f"[bold green]Experiment complete. Results in {self.results_base_dir}[/bold green]"
        )

    def _parse_inventory(self) -> List[Dict[str, Any]]:
        # Minimal port of the inventory parsing logic
        with open(ANSIBLE_INVENTORY, "r") as f:
            inv = yaml.safe_load(f)

        scenarios = []
        all_hosts = inv.get("all", {}).get("children", {})
        mon = list(all_hosts.get("role_monitoring_server", {}).get("hosts", {}).values())[0]

        for group, content in all_hosts.items():
            if group.startswith("app_server_"):
                name = group.replace("app_server_", "")
                app_ip = list(content.get("hosts", {}).values())[0].get("private_ip")
                scenarios.append(
                    {
                        "name": name,
                        "app_server_ip": app_ip,
                        "load_generator_group": f"role_load_generator_{name}",
                        "monitoring_public_ip": mon["public_ip"],
                        "monitoring_private_ip": mon["private_ip"],
                    }
                )
        return scenarios
