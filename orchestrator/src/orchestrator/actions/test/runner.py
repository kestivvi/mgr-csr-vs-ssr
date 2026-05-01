import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from orchestrator.shared.ansible import get_ansible_env

console = Console()

# Playbook paths (relative to ANSIBLE_DIR/project)
LOAD_PLAYBOOK_PATH = "ops/test_load_run.yml"
CAPACITY_PLAYBOOK_PATH = "ops/test_capacity_run.yml"
WRK_PLAYBOOK_PATH = "ops/test_wrk_run.yml"
TEARDOWN_PLAYBOOK_PATH = "ops/test_teardown.yml"

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
    def __init__(
        self,
        config_path: Optional[Path],
        overrides: Optional[Dict[str, Any]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        from orchestrator.actions.test.models import ExperimentConfig

        raw_config: Dict[str, Any] = {}
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)
        elif config_dict:
            raw_config = config_dict
        else:
            raw_config = {"test_type": "load", "num_runs": 1}

        # Apply overrides to raw_config before validation
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    console.print(f"[bold yellow]Override:[/bold yellow] Setting {key} to {value}")
                    if key == "num_runs":
                        raw_config["num_runs"] = value
                    else:
                        # Auto-detect which options block to update
                        for opt_key in [
                            "load_options",
                            "capacity_k6_options",
                            "capacity_wrk_options",
                        ]:
                            if opt_key not in raw_config:
                                raw_config[opt_key] = {}
                            raw_config[opt_key][key] = value

        self.config = ExperimentConfig(**raw_config)
        self.test_type = self.config.test_type
        self.num_runs = self.config.num_runs
        self.results_base_dir = (
            RESULTS_DIR
            / f"{self.test_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self.results_base_dir.mkdir(parents=True, exist_ok=True)

        self.manager = Manager()
        self.shutdown_event = self.manager.Event()

    def _strip_ansi(self, text: str) -> str:
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def _extract_marker(self, output: str, marker: str) -> Optional[Dict[str, Any]]:
        for line in output.splitlines():
            clean_line = self._strip_ansi(line)
            if marker in clean_line:
                try:
                    # Use a more flexible regex to find the JSON block { ... }
                    # It looks for the marker, then anything until the first '{',
                    # then captures everything until the last '}'
                    match = re.search(f"{re.escape(marker)}.*?({{.*}})", clean_line)
                    if match:
                        json_str = match.group(1).replace('\\"', '"')
                        return cast(Dict[str, Any], json.loads(json_str))
                except Exception as e:
                    console.print(f"[dim red]Extraction error for {marker}: {e}[/dim red]")
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

        # Logic for tool-specific vars
        playbook = LOAD_PLAYBOOK_PATH  # Default
        measurement_window: Optional[Dict[str, int]] = None

        if tool == "wrk":
            playbook = WRK_PLAYBOOK_PATH
            if self.config.capacity_wrk_options:
                extra_vars.update(self.config.capacity_wrk_options.model_dump(exclude_none=True))
        elif self.test_type == "capacity_k6":
            playbook = CAPACITY_PLAYBOOK_PATH
            if self.config.capacity_k6_options:
                opts = self.config.capacity_k6_options.model_dump(exclude_none=True)
                mapped = {
                    "capacity_start_rate": opts.get("start_rate"),
                    "capacity_warmup": opts.get("warmup"),
                    "capacity_peak_rate": opts.get("peak_rate"),
                    "capacity_ramp_up": opts.get("ramp_up"),
                    "capacity_peak_rate_2": opts.get("peak_rate_2"),
                    "capacity_ramp_up_2": opts.get("ramp_up_2"),
                    "capacity_sustain": opts.get("sustain"),
                    "capacity_ramp_down": opts.get("ramp_down"),
                    "max_vus": opts.get("max_vus"),
                    "k6_path_type": opts.get("path_type"),
                    "k6_request_timeout": opts.get("timeout"),
                }
                extra_vars.update({k: v for k, v in mapped.items() if v is not None})
        else:
            # Default 'load' test
            if self.config.load_options:
                load_opts = self.config.load_options
                w_sec = parse_duration_to_seconds(load_opts.warmup)
                d_sec = parse_duration_to_seconds(load_opts.duration)
                a_sec = parse_duration_to_seconds(load_opts.after)
                total_sec = w_sec + d_sec + a_sec

                extra_vars.update(
                    {
                        "k6_rate": load_opts.rps,
                        "k6_duration": f"{total_sec}s",
                        "max_vus": load_opts.vus,
                        "k6_path_type": load_opts.path_type,
                        "k6_request_timeout": load_opts.timeout,
                    }
                )
                measurement_window = {"warmup": w_sec, "duration": d_sec}

        # Run via ansible-runner
        r = ansible_runner.run(
            private_data_dir=str(ANSIBLE_DIR),
            playbook=playbook,
            extravars=extra_vars,
            envvars=get_ansible_env(),
            quiet=True,
        )

        output = r.stdout.read()
        timestamps = self._extract_marker(output, TIMESTAMP_MARKER)
        wrk_results = self._extract_marker(output, WRK_RESULT_MARKER) if tool == "wrk" else None

        if timestamps:
            timestamps["start"] = float(timestamps["start"])
            timestamps["end"] = float(timestamps["end"])
            if measurement_window:
                # Adjust timestamps to the middle measurement window
                # Ensure they are numbers as k6 might return them as strings
                base_start = timestamps["start"]
                timestamps["start"] = base_start + measurement_window["warmup"]
                timestamps["end"] = timestamps["start"] + measurement_window["duration"]

        if not r.status == "successful" or not timestamps:
            msg = (
                f"[{run_prefix}:{scenario_name}] Failed. "
                f"Status: {r.status}, Timestamps: {bool(timestamps)}"
            )
            console.print(msg)
            if not timestamps:
                console.print(f"[dim yellow]Raw Output Snippet:[/dim yellow]\n{output[-500:]}")
            # Run teardown
            ansible_runner.run(
                private_data_dir=str(ANSIBLE_DIR),
                playbook=TEARDOWN_PLAYBOOK_PATH,
                extravars={
                    "target_host_group": scenario["load_generator_group"],
                    "server_type": scenario_name,
                },
                envvars=get_ansible_env(),
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

        try:
            for run in range(1, self.num_runs + 1):
                if self.shutdown_event.is_set():
                    break

                console.print(
                    f"\n[bold blue]--- Starting Run {run} of {self.num_runs} ---[/bold blue]"
                )
                run_results = []

                # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
                # with Manager/Event objects. Ansible is I/O bound anyway.
                with ThreadPoolExecutor(max_workers=len(scenarios)) as executor:
                    futures = [executor.submit(self.run_scenario, run, s) for s in scenarios]
                    try:
                        for fut in as_completed(futures):
                            run_results.append(fut.result())
                    except KeyboardInterrupt:
                        console.print(
                            "\n[bold red]Interrupt received! Stopping all runs...[/bold red]"
                        )
                        self.shutdown_event.set()
                        # We don't break here, we let the teardown in run_scenario handle it
                        # or we trigger a global stop below.
                        raise

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
                    # 1. Collect Prometheus metrics
                    collect_metrics(
                        prometheus_url=f"http://{res['scenario']['monitoring_public_ip']}:9090",
                        start_epoch=res["timestamps"]["start"],
                        end_epoch=res["timestamps"]["end"],
                        server_type=res["name"],
                        run_number=run,
                        output_dir=self.results_base_dir,
                    )

                    # 2. Save tool-specific results (like wrk)
                    if res.get("wrk_results"):
                        wrk_dir = self.results_base_dir / "tool_results"
                        wrk_dir.mkdir(parents=True, exist_ok=True)
                        sanitized_name = res["name"].lower().replace("-", "_")
                        res_path = wrk_dir / f"{run:02d}_{sanitized_name}_wrk.json"
                        with open(res_path, "w") as out_file:
                            json.dump(res["wrk_results"], out_file, indent=2)
                        console.print(f"[green]Saved wrk results to {res_path}[/green]")

            # 3. Save metadata for analyzer
            metadata = {
                "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "test_type": self.test_type,
                "parameters": self.config.model_dump(exclude_none=True),
                "calculated_durations_sec": {
                    "measurement": end_ts - start_ts,
                },
            }
            # Flatten parameters for easier access in analyzer appendix
            if self.test_type == "load" and self.config.load_options:
                l_opts = self.config.load_options
                params_dict = cast(dict[str, Any], metadata["parameters"])
                params_dict.update(l_opts.model_dump())
                total_k6_sec = (
                    parse_duration_to_seconds(l_opts.warmup)
                    + parse_duration_to_seconds(l_opts.duration)
                    + parse_duration_to_seconds(l_opts.after)
                )
                params_dict["k6_duration"] = f"{total_k6_sec}s"
                params_dict["warmup_duration"] = l_opts.warmup

            with open(self.results_base_dir / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)

            console.print(
                f"[bold green]Experiment complete. Results in {self.results_base_dir}[/bold green]"
            )
        except KeyboardInterrupt:
            self.global_teardown(scenarios)
            return

    def global_teardown(self, scenarios: Optional[List[Dict[str, Any]]] = None) -> None:
        console.print("[bold yellow]Initiating global emergency stop...[/bold yellow]")
        # Run the stop_all playbook to be sure
        ansible_runner.run(
            private_data_dir=str(ANSIBLE_DIR),
            playbook="ops/test_stop_all.yml",
            envvars=get_ansible_env(),
            quiet=True,
        )
        console.print("[bold green]All tests stopped and containers removed.[/bold green]")

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
