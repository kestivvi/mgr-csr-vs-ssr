import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Optional, Dict, List

import yaml

# --- CONFIGURATION ---
RESULTS_BASE_DIR = Path("results")
INVENTORY_PATH = Path("ansible/inventory.yml")
ANSIBLE_CONFIG_PATH = Path("ansible/ansible.cfg")
# Playbook paths
CONSTANT_PLAYBOOK_PATH = "ansible/test_constant_run.yml"
STRESS_PLAYBOOK_PATH = "ansible/test_stress_run.yml"
TEARDOWN_PLAYBOOK_PATH = "ansible/test_teardown.yml"
COLLECTOR_SCRIPT_PATH = "statistics/collector.py"
TIMESTAMP_MARKER = "ORCHESTRATOR_TIMESTAMPS::"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- SHARED STATE FOR SIGNAL HANDLING ---
shutdown_event = None

def init_worker(event):
    """Initializer for worker processes to ignore SIGINT."""
    global shutdown_event
    shutdown_event = event
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main_signal_handler(signum, frame):
    """Signal handler for the main process to set the shutdown event."""
    global shutdown_event
    if shutdown_event and not shutdown_event.is_set():
        logging.warning("\nCTRL+C detected! Initiating graceful shutdown. Please wait...")
        shutdown_event.set()

def parse_duration_to_seconds(duration_str: str) -> int:
    """Converts a duration string (e.g., '5m', '30s', '1h') to seconds."""
    if not duration_str:
        return 0
    match = re.match(r"(\d+)([smh])", duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Use 's', 'm', or 'h'.")
    
    value, unit = int(match.group(1)), match.group(2)
    if unit == 's': return value
    if unit == 'm': return value * 60
    if unit == 'h': return value * 3600
    return 0

def parse_inventory(inventory_file: Path) -> list:
    """Parses the Ansible inventory to find valid test scenarios."""
    logging.info(f"Parsing inventory file: {inventory_file}")
    if not inventory_file.exists():
        logging.error(f"Inventory file not found at '{inventory_file}'")
        sys.exit(1)

    with open(inventory_file, 'r') as f:
        inventory = yaml.safe_load(f)

    scenarios = []
    all_hosts = inventory.get("all", {}).get("children", {})
    
    monitoring_group = all_hosts.get("role_monitoring_server", {}).get("hosts", {})
    if not monitoring_group:
        logging.error("Could not find 'role_monitoring_server' in inventory.")
        sys.exit(1)
    monitoring_host_details = list(monitoring_group.values())[0]
    monitoring_public_ip = monitoring_host_details.get("public_ip")
    monitoring_private_ip = monitoring_host_details.get("private_ip")

    if not all([monitoring_public_ip, monitoring_private_ip]):
        logging.error("Monitoring server is missing 'public_ip' or 'private_ip'.")
        sys.exit(1)

    for group_name, group_content in all_hosts.items():
        if group_name.startswith("app_server_"):
            scenario_name = group_name.replace("app_server_", "")
            lg_group_name = f"role_load_generator_{scenario_name}"

            if lg_group_name in all_hosts:
                app_hosts = group_content.get("hosts", {})
                if not app_hosts:
                    logging.warning(f"Scenario '{scenario_name}' has no app server hosts defined. Skipping.")
                    continue
                
                first_app_host_details = list(app_hosts.values())[0]
                app_server_private_ip = first_app_host_details.get("private_ip")

                if not app_server_private_ip:
                    logging.warning(f"App server for '{scenario_name}' is missing a private_ip. Skipping.")
                    continue

                scenarios.append({
                    "name": scenario_name,
                    "app_server_ip": app_server_private_ip,
                    "load_generator_group": lg_group_name,
                    "monitoring_public_ip": monitoring_public_ip,
                    "monitoring_private_ip": monitoring_private_ip,
                })
                logging.info(f"Discovered valid scenario: '{scenario_name}'")

    if not scenarios:
        logging.error("No valid test scenarios discovered in inventory. Aborting.")
        sys.exit(1)
        
    return scenarios

def extract_timestamps_from_output(output: str) -> Optional[Dict[str, int]]:
    """Finds the timestamp marker in Ansible output and parses the JSON."""
    for line in output.splitlines():
        if TIMESTAMP_MARKER in line:
            try:
                json_str = line.split(TIMESTAMP_MARKER, 1)[1].strip().strip("'\"")
                data = json.loads(json_str)
                return {"start": int(data["start"]), "end": int(data["end"])}
            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                logging.error(f"Failed to parse timestamp data from Ansible line '{line}': {e}")
                return None
    return None

def run_single_scenario_lifecycle(
    run_number: int,
    scenario: dict,
    results_dir: Path,
    args: argparse.Namespace
) -> Dict:
    """
    Executes a single test scenario and returns its result, including precise timestamps.
    It does NOT collect metrics; that is handled by the main process after synchronization.
    """
    global shutdown_event
    scenario_name = scenario["name"]
    run_prefix = f"{run_number:02d}"
    
    logging.info(f"[{run_prefix}:{scenario_name}] Starting k6 test ({args.test_type} type).")
    
    ansible_log_path = results_dir / f"{run_prefix}_ansible_k6_{scenario_name}.txt"
    
    playbook_path = CONSTANT_PLAYBOOK_PATH if args.test_type == 'constant' else STRESS_PLAYBOOK_PATH
    extra_vars = {
        "target_host_group": scenario["load_generator_group"],
        "target_url": f"http://{scenario['app_server_ip']}",
        "server_type": scenario_name,
        "prometheus_url": f"http://{scenario['monitoring_private_ip']}:9090",
        "k6_path_type": args.path_type,
    }
    if args.test_type == 'constant':
        extra_vars.update({"k6_rate": args.rate, "k6_duration": args.duration})
    else: # stress
        extra_vars.update({
            "stress_start_rate": args.start_rate, "stress_peak_rate": args.peak_rate,
            "stress_ramp_up": args.ramp_up, "stress_sustain": args.sustain,
            "stress_ramp_down": args.ramp_down, "max_vus": args.max_vus,
        })

    if args.backoff_timeout_s is not None: extra_vars["k6_backoff_timeout_s"] = args.backoff_timeout_s
    if args.backoff_5xx_s is not None: extra_vars["k6_backoff_5xx_s"] = args.backoff_5xx_s

    ansible_command = ["ansible-playbook", "-i", str(INVENTORY_PATH), playbook_path, "--extra-vars", json.dumps(extra_vars)]
    ansible_env = os.environ.copy()
    ansible_env["ANSIBLE_CONFIG"] = str(ANSIBLE_CONFIG_PATH)
    
    timestamps = None
    interrupted = False
    
    try:
        with open(ansible_log_path, 'w') as log_file:
            process = subprocess.Popen(
                ansible_command,
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=ansible_env
            )

            while process.poll() is None:
                if shutdown_event and shutdown_event.is_set():
                    logging.warning(f"[{run_prefix}:{scenario_name}] Shutdown signal received. Terminating Ansible...")
                    process.terminate()
                    interrupted = True
                    break
                time.sleep(0.5)

        with open(ansible_log_path, 'r') as f:
            output_log = f.read()

        if interrupted:
            logging.info(f"[{run_prefix}:{scenario_name}] Ansible terminated. Running teardown playbook to get final timestamps.")
            teardown_command = [
                "ansible-playbook", "-i", str(INVENTORY_PATH), TEARDOWN_PLAYBOOK_PATH,
                "--extra-vars", json.dumps({"target_host_group": scenario["load_generator_group"], "server_type": scenario_name})
            ]
            teardown_process = subprocess.run(teardown_command, capture_output=True, check=True, text=True, env=ansible_env)
            with open(ansible_log_path, 'a') as log_file:
                log_file.write(f"\n\n--- TEARDOWN INITIATED BY CTRL+C ---\n{teardown_process.stdout}")
            timestamps = extract_timestamps_from_output(teardown_process.stdout)
        
        elif process.returncode == 0:
            timestamps = extract_timestamps_from_output(output_log)
        else:
            logging.error(f"[{run_prefix}:{scenario_name}] Ansible playbook failed with exit code {process.returncode}. See log: {ansible_log_path}")
            return {"name": scenario_name, "success": False, "timestamps": None, "scenario_details": scenario}

    except Exception as e:
        logging.error(f"[{run_prefix}:{scenario_name}] An unexpected error occurred: {e}")
        return {"name": scenario_name, "success": False, "timestamps": None, "scenario_details": scenario}
        
    if not timestamps:
        logging.error(f"[{run_prefix}:{scenario_name}] Could not extract test timestamps from Ansible output.")
        return {"name": scenario_name, "success": False, "timestamps": None, "scenario_details": scenario}

    logging.info(f"[{run_prefix}:{scenario_name}] Test execution phase finished.")
    return {"name": scenario_name, "success": True, "timestamps": timestamps, "scenario_details": scenario}

def collect_metrics_for_scenario(run_number: int, result: Dict, results_dir: Path, start_epoch: int, end_epoch: int, args: argparse.Namespace):
    """Calls the collector script for a single scenario using a synchronized time window."""
    scenario_name = result["name"]
    scenario = result["scenario_details"]
    run_prefix = f"{run_number:02d}"
    
    logging.info(f"[{run_prefix}:{scenario_name}] Collecting metrics for synchronized window.")
    collector_log_path = results_dir / f"{run_prefix}_collector_{scenario_name}.txt"
    
    metric_start = start_epoch
    metric_end = end_epoch
    if args.test_type == 'constant' and not shutdown_event.is_set():
        warmup_sec = parse_duration_to_seconds(args.warmup)
        cooldown_sec = parse_duration_to_seconds(args.cooldown)
        
        effective_start = start_epoch + warmup_sec
        effective_end = end_epoch - cooldown_sec

        if effective_start >= effective_end:
            logging.warning(f"[{run_prefix}:{scenario_name}] Warmup/cooldown is longer than the synchronized window. Using full window.")
        else:
            metric_start, metric_end = effective_start, effective_end

    collector_command = [
        sys.executable, str(COLLECTOR_SCRIPT_PATH),
        "--prometheus-url", f"http://{scenario['monitoring_public_ip']}:9090",
        "--start-epoch", str(metric_start),
        "--end-epoch", str(metric_end),
        "--server-type", scenario_name,
        "--run-number", str(run_number),
        "--output-dir", str(results_dir)
    ]

    try:
        with open(collector_log_path, 'w') as log_file:
            subprocess.run(collector_command, stdout=log_file, stderr=subprocess.STDOUT, check=True, text=True)
        logging.info(f"[{run_prefix}:{scenario_name}] Metric collection successful.")
    except subprocess.CalledProcessError:
        logging.error(f"[{run_prefix}:{scenario_name}] Collector script failed. See log: {collector_log_path}")

def create_metadata_file(results_dir: Path, args: argparse.Namespace):
    """Creates a metadata.yaml file with the experiment parameters."""
    metadata = {
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameters": {
            "test_type": args.test_type, "num_runs": args.num_runs, "path_type": args.path_type,
        }
    }
    
    # --- FIX #4: Calculate and add measurement duration ---
    calculated_durations = {}
    if args.test_type == 'constant':
        metadata["parameters"].update({"rate": args.rate, "k6_duration": args.duration, "warmup_duration": args.warmup, "cooldown_duration": args.cooldown})
        total_s = parse_duration_to_seconds(args.duration)
        warmup_s = parse_duration_to_seconds(args.warmup)
        cooldown_s = parse_duration_to_seconds(args.cooldown)
        measurement_s = total_s - warmup_s - cooldown_s
        calculated_durations['measurement'] = max(0, measurement_s) # Ensure it's not negative
    else: # stress
        metadata["parameters"].update({"start_rate": args.start_rate, "peak_rate": args.peak_rate, "ramp_up_duration": args.ramp_up, "sustain_duration": args.sustain, "ramp_down_duration": args.ramp_down, "max_vus": args.max_vus})
        ramp_up_s = parse_duration_to_seconds(args.ramp_up)
        sustain_s = parse_duration_to_seconds(args.sustain)
        ramp_down_s = parse_duration_to_seconds(args.ramp_down)
        # For stress tests, the intended measurement duration is the full scenario
        calculated_durations['measurement'] = ramp_up_s + sustain_s + ramp_down_s

    if args.backoff_timeout_s is not None: metadata["parameters"]["backoff_timeout_s"] = args.backoff_timeout_s
    if args.backoff_5xx_s is not None: metadata["parameters"]["backoff_5xx_s"] = args.backoff_5xx_s
    
    metadata['calculated_durations_sec'] = calculated_durations
    # --- END FIX ---

    with open(results_dir / "metadata.yaml", 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logging.info(f"Experiment metadata saved to {results_dir / 'metadata.yaml'}")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate k6 test runs and metric collection.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--test-type', type=str, required=True, choices=['constant', 'stress'])
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--path-type', type=str, default='dynamic', choices=['static', 'dynamic'])
    constant_group = parser.add_argument_group('constant test options')
    constant_group.add_argument('--rate', type=int)
    constant_group.add_argument('--duration', type=str)
    constant_group.add_argument('--warmup', type=str, default='30s')
    constant_group.add_argument('--cooldown', type=str, default='15s')
    stress_group = parser.add_argument_group('stress test options')
    stress_group.add_argument('--start-rate', type=int, default=10)
    stress_group.add_argument('--peak-rate', type=int, default=2000)
    stress_group.add_argument('--ramp-up', type=str, default='10m')
    stress_group.add_argument('--sustain', type=str, default='5m')
    stress_group.add_argument('--ramp-down', type=str, default='1m')
    stress_group.add_argument('--max-vus', type=int, default=200)
    backoff_group = parser.add_argument_group('backoff options')
    backoff_group.add_argument('--backoff-timeout-s', dest='backoff_timeout_s', type=float)
    backoff_group.add_argument('--backoff-5xx-s', dest='backoff_5xx_s', type=float)
    args = parser.parse_args()

    if args.test_type == 'constant' and not all([args.rate, args.duration]):
        parser.error("--rate and --duration are required for 'constant' test type")

    results_dir = RESULTS_BASE_DIR / f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(results_dir / "orchestrator.txt")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Starting new experiment. Results will be saved in: {results_dir}")
    create_metadata_file(results_dir, args)
    scenarios = parse_inventory(INVENTORY_PATH)

    global shutdown_event
    manager = Manager()
    shutdown_event = manager.Event()
    signal.signal(signal.SIGINT, main_signal_handler)

    for run in range(1, args.num_runs + 1):
        if shutdown_event.is_set():
            logging.warning(f"Skipping run {run} due to shutdown signal.")
            break
        
        logging.info(f"--- Starting Run {run} of {args.num_runs} ---")
        
        tasks = [(run, scenario, results_dir, args) for scenario in scenarios]
        run_results = []
        with ProcessPoolExecutor(initializer=init_worker, initargs=(shutdown_event,)) as executor:
            futures = {executor.submit(run_single_scenario_lifecycle, *task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    run_results.append(future.result())
                except Exception as exc:
                    logging.error(f"A scenario task generated an exception: {exc}")

        successful_results = [r for r in run_results if r["success"] and r["timestamps"]]
        
        if not successful_results:
            logging.error(f"[Run {run:02d}] No scenarios completed successfully. Skipping metric collection.")
            continue

        all_starts = [r["timestamps"]["start"] for r in successful_results]
        all_ends = [r["timestamps"]["end"] for r in successful_results]
        sync_start_epoch = max(all_starts)
        sync_end_epoch = min(all_ends)

        if sync_start_epoch >= sync_end_epoch:
            logging.error(f"[Run {run:02d}] Test windows did not overlap. Cannot collect synchronized metrics.")
            continue
        
        sync_duration = sync_end_epoch - sync_start_epoch
        logging.info(f"--- Synchronized measurement window for Run {run} is {sync_duration}s ---")
        logging.info(f"Start: {datetime.datetime.fromtimestamp(sync_start_epoch, tz=datetime.timezone.utc).isoformat()}")
        logging.info(f"End:   {datetime.datetime.fromtimestamp(sync_end_epoch, tz=datetime.timezone.utc).isoformat()}")

        for result in successful_results:
            collect_metrics_for_scenario(run, result, results_dir, sync_start_epoch, sync_end_epoch, args)
        
        logging.info(f"--- Finished Run {run} of {args.num_runs} ---")

    if shutdown_event.is_set():
        logging.info("Experiment terminated gracefully by user.")
    else:
        logging.info("Experiment finished.")

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed. Please run: pip install PyYAML", file=sys.stderr)
        sys.exit(1)
    main()
