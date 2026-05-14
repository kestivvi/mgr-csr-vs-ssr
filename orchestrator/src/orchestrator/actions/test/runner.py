import datetime
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path
from typing import Any, cast

import ansible_runner
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from orchestrator.actions.test.collector import collect_metrics
from orchestrator.actions.test.models import (
    ExperimentConfig,
    ScenarioMetadata,
    ScenarioResult,
)
from orchestrator.config import (
    ANSIBLE_DIR,
    ANSIBLE_INVENTORY,
    RESULTS_DIR,
)
from orchestrator.shared.ansible import get_ansible_env

console = Console()

# Resource Estimation Constants
IDLE_LOAD_GEN_RAM_MB = 310
RAM_PER_K6_VU_MB = 4.85
RAM_PER_K6_VU_ASSETS_MB = 0.42  # Additional MB per VU when assets are enabled
INSTANCE_RAM_MAP = {
    # t4g family (Burstable)
    "t4g.nano": 512,
    "t4g.micro": 1024,
    "t4g.small": 2048,
    "t4g.medium": 4096,
    "t4g.large": 8192,
    "t4g.xlarge": 16384,
    "t4g.2xlarge": 32768,
    # c8g family (Compute Optimized - Graviton4)
    "c8g.medium": 2048,
    "c8g.large": 4096,
    "c8g.xlarge": 8192,
    "c8g.2xlarge": 16384,
    "c8g.4xlarge": 32768,
    "c8g.8xlarge": 65536,
    "c8g.12xlarge": 98304,
    "c8g.16xlarge": 131072,
    "c8g.24xlarge": 196608,
    "c8g.metal": 196608,
}

# Playbook paths (relative to ANSIBLE_DIR/project)
LOAD_PLAYBOOK_PATH = "ops/test_load_run.yml"
CAPACITY_PLAYBOOK_PATH = "ops/test_capacity_run.yml"
WRK_PLAYBOOK_PATH = "ops/test_wrk_run.yml"
SYNC_SCRIPT_PLAYBOOK_PATH = "ops/test_sync_script.yml"
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
        config_path: Path | None,
        overrides: dict[str, Any] | None = None,
        config_dict: dict[str, Any] | None = None,
        output_dir: Path | None = None,
        apps: str | None = None,
    ) -> None:
        raw_config: dict[str, Any] = {}
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
                    console.print(f"[dim yellow]Override: {key} set to {value}[/dim yellow]")
                    if key in ["num_runs", "inter_run_delay", "auto_approve"]:
                        raw_config[key] = value
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

        if output_dir:
            self.results_base_dir = output_dir
        else:
            self.results_base_dir = (
                RESULTS_DIR
                / f"{self.test_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )

        self.results_base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.results_base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.apps_filter = apps

        self.manager = Manager()
        self.shutdown_event = self.manager.Event()

    def _strip_ansi(self, text: str) -> str:
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def _extract_marker(self, output: str, marker: str) -> dict[str, Any] | None:
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
                        return cast(dict[str, Any], json.loads(json_str))
                except Exception as e:
                    console.print(f"[dim red]Extraction error for {marker}: {e}[/dim red]")
                    continue
        return None

    def run_scenario(self, run_number: int, scenario: ScenarioMetadata) -> ScenarioResult:
        if self.shutdown_event.is_set():
            return {"success": False}

        scenario_name = scenario["name"]
        run_prefix = f"{run_number:02d}"
        tool = "wrk" if self.test_type == "capacity_wrk" else "k6"

        console.print(f"[{run_prefix}:{scenario_name}] Starting {tool} test.")

        extra_vars: dict[str, Any] = {
            "target_host_group": scenario["load_generator_group"],
            "target_url": f"https://{scenario['app_server_ip']}",
            "server_type": scenario_name,
            "prometheus_url": f"http://{scenario['monitoring_private_ip']}:9090",
        }

        # Logic for tool-specific vars
        playbook = LOAD_PLAYBOOK_PATH  # Default
        measurement_window: dict[str, int] | None = None

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
                    "k6_log_path": str(
                        self.logs_dir
                        / f"{run_prefix}_{scenario_name.lower().replace('-', '_')}.log"
                    ),
                    "k6_skip_assets": opts.get("skip_assets"),
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
                        "k6_log_path": str(
                            self.logs_dir
                            / f"{run_prefix}_{scenario_name.lower().replace('-', '_')}.log"
                        ),
                        "k6_skip_assets": load_opts.skip_assets,
                    }
                )
                measurement_window = {"warmup": w_sec, "duration": d_sec}

        # Run via ansible-runner (async to allow for cancellation).
        # cancel_callback is polled by ansible-runner between events; returning
        # True triggers a clean cancellation of the underlying ansible-playbook.
        thread, r = ansible_runner.run_async(
            private_data_dir=str(ANSIBLE_DIR),
            playbook=playbook,
            extravars=extra_vars,
            envvars=get_ansible_env(),
            cancel_callback=self.shutdown_event.is_set,
            quiet=True,
        )

        try:
            thread.join()
        except Exception:
            r.cancel()
            raise

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

        # Save log file (Note: For k6, logs are now fetched directly by the playbook to avoid OOM)
        log_path = None
        if tool == "wrk":
            log_filename = f"{run_prefix}_{scenario_name.lower().replace('-', '_')}.log"
            log_path = self.logs_dir / log_filename
            with open(log_path, "w") as f:
                f.write(output)

        if not r.status == "successful" or not timestamps:
            msg = (
                f"[{run_prefix}:{scenario_name}] Failed. "
                f"Status: {r.status}, Timestamps: {bool(timestamps)}"
            )
            console.print(msg)
            if log_path:
                console.print(f"[dim yellow]Full log saved to:[/dim yellow] {log_path}")
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
                cancel_callback=self.shutdown_event.is_set,
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
            self._print_summary(scenarios)

            if not self.config.auto_approve:
                if not Confirm.ask("Do you want to proceed with the experiment?"):
                    console.print("[bold yellow]Aborting experiment...[/bold yellow]")
                    return

            # 2. Sync k6 script to all load generators
            if self.test_type in ["load", "capacity_k6"]:
                if not self._sync_k6_script():
                    console.print("[bold red]Critical Error: Failed to sync k6 script.[/bold red]")
                    return

            start_ts: float = 0.0
            end_ts: float = 0.0

            for run in range(1, self.num_runs + 1):
                if self.shutdown_event.is_set():
                    break

                console.print(
                    f"\n[bold blue]--- Starting Run {run} of {self.num_runs} ---[/bold blue]"
                )
                run_results = []

                # Poll with a timeout so the main thread regularly returns to
                # Python bytecode and KeyboardInterrupt can be delivered.
                # (concurrent.futures' no-timeout waits sit in a C-level lock
                # that swallows SIGINT until a future completes.)
                executor = ThreadPoolExecutor(max_workers=len(scenarios))
                futures = [executor.submit(self.run_scenario, run, s) for s in scenarios]
                try:
                    pending = set(futures)
                    while pending:
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        for fut in done:
                            run_results.append(fut.result())
                except KeyboardInterrupt:
                    console.print("\n[bold red]Interrupt received! Stopping all runs...[/bold red]")
                    self.shutdown_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    executor.shutdown(wait=False)

                successful = [r for r in run_results if r.get("success") and r.get("timestamps")]
                if not successful:
                    continue

                # Sync timestamps
                # We use cast here because we already filtered for non-None timestamps above
                start_ts = max(cast(dict[str, float], r["timestamps"])["start"] for r in successful)
                end_ts = min(cast(dict[str, float], r["timestamps"])["end"] for r in successful)

                if start_ts >= end_ts:
                    console.print("[red]No overlapping time window found.[/red]")
                    continue

                for res in successful:
                    # 1. Collect Prometheus metrics
                    ts_dict = cast(dict[str, float], res["timestamps"])
                    collect_metrics(
                        prometheus_url=f"http://{res['scenario']['monitoring_public_ip']}:9090",
                        start_epoch=int(ts_dict["start"]),
                        end_epoch=int(ts_dict["end"]),
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

                if run < self.num_runs:
                    delay_sec = parse_duration_to_seconds(self.config.inter_run_delay)
                    console.print(
                        f"\n[bold yellow]Wait period:[/bold yellow] "
                        f"Sleeping for {self.config.inter_run_delay} ({delay_sec}s) "
                        "before next run..."
                    )
                    time.sleep(delay_sec)

            # 3. Save metadata for analyzer
            if end_ts > start_ts:
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
                Panel(
                    f"[bold green]Experiment complete![/bold green]\n"
                    f"Results saved to: [cyan]{self.results_base_dir}[/cyan]\n"
                    f"Run [bold]mgr analyze[/bold] to generate reports.",
                    title="Success",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            self.global_teardown(scenarios)
            console.print(
                "\n",
                Panel(
                    "[bold red]Experiment aborted by user.[/bold red]\n"
                    "Some data may have been collected but the run is incomplete.",
                    title="Aborted",
                    border_style="red",
                ),
            )
            return

    def _sync_k6_script(self) -> bool:
        with console.status("[bold cyan]Syncing k6 script to load generators...", spinner="dots"):
            # cancel_callback must be passed explicitly; otherwise ansible-runner
            # installs a silent SIGINT/SIGTERM handler that breaks Ctrl+C for the
            # rest of the process lifetime.
            r = ansible_runner.run(
                private_data_dir=str(ANSIBLE_DIR),
                playbook=SYNC_SCRIPT_PLAYBOOK_PATH,
                envvars=get_ansible_env(),
                cancel_callback=self.shutdown_event.is_set,
                quiet=True,
            )
        if r.status != "successful":
            console.print("[red]Ansible Sync Output:[/red]")
            console.print(r.stdout.read())
        return bool(r.status == "successful")

    def global_teardown(self, scenarios: list[ScenarioMetadata] | None = None) -> None:
        try:
            console.print("[bold yellow]Initiating global emergency stop...[/bold yellow]")
            # Run the stop_all playbook to be sure
            # cancel_callback=lambda: False keeps teardown un-cancellable AND
            # prevents ansible-runner from installing its silent SIGINT handler.
            ansible_runner.run(
                private_data_dir=str(ANSIBLE_DIR),
                playbook="ops/test_stop_all.yml",
                envvars=get_ansible_env(),
                cancel_callback=lambda: False,
                quiet=True,
            )
            console.print("[bold green]All tests stopped and containers removed.[/bold green]")
        except KeyboardInterrupt:
            # If interrupted again, we just skip the message and let the process die
            # but we try to avoid crashing mid-ansible run if possible.
            console.print("[dim red](Further interrupt ignored during cleanup...)[/dim red]")
            # We don't re-raise here to allow the process to exit naturally after this call

    def _print_summary(self, scenarios: list[ScenarioMetadata]) -> None:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Parameter", style="dim", width=25)
        table.add_column("Value", style="bold cyan")

        # --- Section: General ---
        table.add_row("[bold white]General[/bold white]", "")
        table.add_row("  Test Type", self.test_type.upper())
        table.add_row("  Num Runs", str(self.num_runs))
        table.add_row("  Scenarios", str(len(scenarios)))

        skip_assets = False
        if self.test_type == "load" and self.config.load_options:
            skip_assets = self.config.load_options.skip_assets
        elif self.test_type == "capacity_k6" and self.config.capacity_k6_options:
            skip_assets = self.config.capacity_k6_options.skip_assets

        table.add_row("  Skip Assets", str(skip_assets))
        table.add_section()

        # Get Load Generator RAM from infra.yaml
        lg_type = "unknown"
        lg_ram = 0
        try:
            from orchestrator.config import INFRA_YAML

            if INFRA_YAML.exists():
                with open(INFRA_YAML, "r") as f:
                    infra = yaml.safe_load(f)
                    lg_type = infra.get("load_generator_instance_type", "unknown")
                    lg_ram = INSTANCE_RAM_MAP.get(lg_type, 0)
        except Exception:
            pass

        vus = 0
        # --- Section: Workload ---
        table.add_row("[bold white]Workload[/bold white]", "")
        if self.test_type == "load" and self.config.load_options:
            l_opts = self.config.load_options
            vus = l_opts.vus
            table.add_row("  Target RPS", str(l_opts.rps))
            table.add_row("  Max VUs", str(l_opts.vus))
            if l_opts.rps > 0:
                theoretical_max_latency = (l_opts.vus / l_opts.rps) * 1000
                table.add_row("  Max Latency (theo)", f"{theoretical_max_latency:.2f}ms")
            table.add_section()

            # --- Section: Timeline ---
            table.add_row("[bold white]Timeline[/bold white]", "")
            table.add_row("  Warmup", f"{l_opts.warmup} (0 ➔ {l_opts.rps} RPS)")
            table.add_row("  Sustain", f"{l_opts.duration} ({l_opts.rps} RPS)")
            table.add_row("  Cooldown", f"{l_opts.after} ({l_opts.rps} ➔ 0 RPS)")

        elif self.test_type == "capacity_k6" and self.config.capacity_k6_options:
            c_opts = self.config.capacity_k6_options
            vus = c_opts.max_vus
            table.add_row("  Peak Rate", str(c_opts.peak_rate))
            table.add_row("  Max VUs", str(c_opts.max_vus))
            if c_opts.peak_rate > 0:
                theoretical_max_latency = (c_opts.max_vus / c_opts.peak_rate) * 1000
                table.add_row("  Max Latency (theo)", f"{theoretical_max_latency:.2f}ms")
            table.add_section()

            # --- Section: Timeline ---
            table.add_row("[bold white]Timeline[/bold white]", "")
            table.add_row(
                "  Ramp Up", f"{c_opts.ramp_up} ({c_opts.start_rate} ➔ {c_opts.peak_rate} RPS)"
            )
            table.add_row("  Sustain", f"{c_opts.sustain} ({c_opts.peak_rate} RPS)")
            table.add_row("  Ramp Down", f"{c_opts.ramp_down} ({c_opts.peak_rate} ➔ 0 RPS)")

        table.add_section()
        # --- Section: Resources ---
        table.add_row("[bold white]Resources (Est.)[/bold white]", "")
        if vus > 0:
            vu_ram = RAM_PER_K6_VU_MB
            if not skip_assets:
                vu_ram += RAM_PER_K6_VU_ASSETS_MB

            est_ram = IDLE_LOAD_GEN_RAM_MB + (vus * vu_ram)
            ram_str = f"{est_ram:.0f}MB"
            if lg_ram > 0:
                usage_pct = (est_ram / lg_ram) * 100
                ram_style = (
                    "bold red"
                    if usage_pct > 85
                    else "bold yellow"
                    if usage_pct > 70
                    else "bold green"
                )
                table.add_row("  RAM Usage", f"[{ram_style}]{ram_str}[/] / {lg_ram}MB ({lg_type})")
            else:
                table.add_row("  RAM Usage", ram_str)

        console.print(
            Panel(table, title="[bold white]Experiment Plan[/bold white]", border_style="blue")
        )

    def _parse_inventory(self) -> list[ScenarioMetadata]:
        # Minimal port of the inventory parsing logic
        with open(ANSIBLE_INVENTORY, "r") as f:
            inv = yaml.safe_load(f)

        scenarios = []
        all_hosts = inv.get("all", {}).get("children", {})
        mon = list(all_hosts.get("role_monitoring_server", {}).get("hosts", {}).values())[0]

        for group, content in all_hosts.items():
            if group.startswith("app_server_"):
                name = group.replace("app_server_", "")

                # Filter by app name if specified
                if self.apps_filter:
                    allowed = [a.strip().lower() for a in self.apps_filter.split(",")]
                    if not any(a in name.lower() for a in allowed):
                        continue

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
        return cast(list[ScenarioMetadata], scenarios)
