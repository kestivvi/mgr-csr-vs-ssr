import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict

import yaml

# --- CONFIGURATION ---
RESULTS_BASE_DIR = Path("results")
INVENTORY_PATH = Path("ansible/inventory.yml")
ANSIBLE_CONFIG_PATH = Path("ansible/ansible.cfg")
# Playbook paths are now selected dynamically
CONSTANT_PLAYBOOK_PATH = "ansible/test_constant_run.yml"
STRESS_PLAYBOOK_PATH = "ansible/test_stress_run.yml"
COLLECTOR_SCRIPT_PATH = "statistics/collector.py"
TIMESTAMP_MARKER = "ORCHESTRATOR_TIMESTAMPS::"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_duration_to_seconds(duration_str: str) -> int:
    """Converts a duration string (e.g., '5m', '30s', '1h') to seconds."""
    if not duration_str:
        return 0
    match = re.match(r"(\d+)([smh])", duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Use 's', 'm', or 'h'.")
    
    value, unit = int(match.group(1)), match.group(2)
    if unit == 's':
        return value
    if unit == 'm':
        return value * 60
    if unit == 'h':
        return value * 3600
    return 0

def parse_inventory(inventory_file: Path) -> list:
    """
    Parses the Ansible inventory to find matching app servers and load generators.
    """
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
                json_str = line.split(TIMESTAMP_MARKER, 1)[1]
                json_str = json_str.strip().strip("'\"")
                
                data = json.loads(json_str)
                return {
                    "start": int(data["start"]),
                    "end": int(data["end"])
                }
            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                logging.error(f"Failed to parse timestamp data from Ansible line '{line}': {e}")
                return None
    return None

def run_single_scenario_lifecycle(
    run_number: int,
    scenario: dict,
    results_dir: Path,
    args: argparse.Namespace
) -> tuple[str, bool]:
    """
    The full lifecycle for one scenario in one run: k6 test + metric collection.
    """
    scenario_name = scenario["name"]
    run_prefix = f"{run_number:02d}"
    
    logging.info(f"[{run_prefix}:{scenario_name}] Starting k6 test ({args.test_type} type).")
    
    ansible_log_path = results_dir / f"{run_prefix}_ansible_k6_{scenario_name}.txt"
    
    # --- DYNAMICALLY build playbook path and extra_vars based on test type ---
    if args.test_type == 'constant':
        playbook_path = CONSTANT_PLAYBOOK_PATH
        extra_vars = {
            "target_host_group": scenario["load_generator_group"],
            "target_url": f"http://{scenario['app_server_ip']}",
            "server_type": scenario_name,
            "prometheus_url": f"http://{scenario['monitoring_private_ip']}:9090",
            "k6_rate": args.rate,
            "k6_duration": args.duration,
        }
        if args.backoff_timeout_s is not None:
            extra_vars["k6_backoff_timeout_s"] = args.backoff_timeout_s
        if args.backoff_5xx_s is not None:
            extra_vars["k6_backoff_5xx_s"] = args.backoff_5xx_s
    elif args.test_type == 'stress':
        playbook_path = STRESS_PLAYBOOK_PATH
        extra_vars = {
            "target_host_group": scenario["load_generator_group"],
            "target_url": f"http://{scenario['app_server_ip']}",
            "server_type": scenario_name,
            "prometheus_url": f"http://{scenario['monitoring_private_ip']}:9090",
            "stress_start_rate": args.start_rate,
            "stress_peak_rate": args.peak_rate,
            "stress_ramp_up": args.ramp_up,
            "stress_sustain": args.sustain,
            "stress_ramp_down": args.ramp_down,
            "max_vus": args.max_vus,
        }
        if args.backoff_timeout_s is not None:
            extra_vars["k6_backoff_timeout_s"] = args.backoff_timeout_s
        if args.backoff_5xx_s is not None:
            extra_vars["k6_backoff_5xx_s"] = args.backoff_5xx_s
    else:
        logging.error(f"Internal error: Unknown test type '{args.test_type}'")
        return scenario_name, False

    ansible_command = [
        "ansible-playbook",
        "-i", str(INVENTORY_PATH),
        playbook_path,
        "--extra-vars",
        json.dumps(extra_vars)
    ]
    
    ansible_env = os.environ.copy()
    ansible_env["ANSIBLE_CONFIG"] = str(ANSIBLE_CONFIG_PATH)
    
    try:
        process = subprocess.run(
            ansible_command,
            capture_output=True,
            check=True,
            text=True,
            env=ansible_env
        )
        with open(ansible_log_path, 'w') as log_file:
            log_file.write(process.stdout)
            if process.stderr:
                log_file.write("\n--- STDERR ---\n")
                log_file.write(process.stderr)

    except subprocess.CalledProcessError as e:
        with open(ansible_log_path, 'w') as log_file:
            log_file.write(e.stdout)
            if e.stderr:
                log_file.write("\n--- STDERR ---\n")
                log_file.write(e.stderr)
        logging.error(f"[{run_prefix}:{scenario_name}] Ansible playbook failed with exit code {e.returncode}. See log: {ansible_log_path}")
        return scenario_name, False
    except FileNotFoundError:
        logging.error("`ansible-playbook` command not found. Is Ansible installed and in your PATH?")
        return scenario_name, False
        
    timestamps = extract_timestamps_from_output(process.stdout)
    if not timestamps:
        logging.error(f"[{run_prefix}:{scenario_name}] Could not extract test timestamps from Ansible output. Aborting metric collection.")
        return scenario_name, False

    actual_duration = timestamps["end"] - timestamps["start"]
    logging.info(f"[{run_prefix}:{scenario_name}] k6 test finished successfully after {actual_duration:.1f}s.")

    # --- Metric collection is only supported for constant tests for now ---
    if args.test_type == 'constant':
        logging.info(f"[{run_prefix}:{scenario_name}] Starting metric collection.")
        
        collector_log_path = results_dir / f"{run_prefix}_collector_{scenario_name}.txt"
        
        warmup_sec = parse_duration_to_seconds(args.warmup)
        cooldown_sec = parse_duration_to_seconds(args.cooldown)
        total_duration_sec = parse_duration_to_seconds(args.duration)
        
        metric_start_epoch = timestamps["start"] + warmup_sec
        metric_end_epoch = (timestamps["start"] + total_duration_sec) - cooldown_sec

        if metric_start_epoch >= metric_end_epoch:
            logging.error(
                f"[{run_prefix}:{scenario_name}] Warmup ({warmup_sec}s) and cooldown ({cooldown_sec}s) "
                f"period is longer than the configured test duration ({total_duration_sec}s). Cannot collect metrics."
            )
            return scenario_name, False

        collector_command = [
            sys.executable,
            str(COLLECTOR_SCRIPT_PATH),
            "--prometheus-url", f"http://{scenario['monitoring_public_ip']}:9090",
            "--start-epoch", str(metric_start_epoch),
            "--end-epoch", str(metric_end_epoch),
            "--server-type", scenario_name,
            "--run-number", str(run_number),
            "--output-dir", str(results_dir)
        ]

        try:
            with open(collector_log_path, 'w') as log_file:
                subprocess.run(
                    collector_command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"[{run_prefix}:{scenario_name}] Collector script failed with exit code {e.returncode}. See log: {collector_log_path}")
            return scenario_name, False

        logging.info(f"[{run_prefix}:{scenario_name}] Metric collection finished successfully.")
    
    return scenario_name, True

def create_metadata_file(results_dir: Path, args: argparse.Namespace):
    """Creates a metadata.yaml file with the experiment parameters."""
    metadata = {
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameters": {
            "test_type": args.test_type,
            "num_runs": args.num_runs,
        }
    }

    if args.test_type == 'constant':
        total_duration_sec = parse_duration_to_seconds(args.duration)
        warmup_sec = parse_duration_to_seconds(args.warmup)
        cooldown_sec = parse_duration_to_seconds(args.cooldown)
        measurement_sec = total_duration_sec - warmup_sec - cooldown_sec
        
        metadata["parameters"].update({
            "rate": args.rate,
            "k6_duration": args.duration,
            "warmup_duration": args.warmup,
            "cooldown_duration": args.cooldown,
        })
        if args.backoff_timeout_s is not None:
            metadata["parameters"]["backoff_timeout_s"] = args.backoff_timeout_s
        if args.backoff_5xx_s is not None:
            metadata["parameters"]["backoff_5xx_s"] = args.backoff_5xx_s
        metadata["calculated_durations_sec"] = {
            "total": total_duration_sec,
            "warmup": warmup_sec,
            "cooldown": cooldown_sec,
            "measurement": measurement_sec if measurement_sec > 0 else "Invalid"
        }
    elif args.test_type == 'stress':
        metadata["parameters"].update({
            "start_rate": args.start_rate,
            "peak_rate": args.peak_rate,
            "ramp_up_duration": args.ramp_up,
            "sustain_duration": args.sustain,
            "ramp_down_duration": args.ramp_down,
            "max_vus": args.max_vus,
        })
        if args.backoff_timeout_s is not None:
            metadata["parameters"]["backoff_timeout_s"] = args.backoff_timeout_s
        if args.backoff_5xx_s is not None:
            metadata["parameters"]["backoff_5xx_s"] = args.backoff_5xx_s

    metadata_path = results_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logging.info(f"Experiment metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate k6 test runs and metric collection.", formatter_class=argparse.RawTextHelpFormatter)
    
    # --- Main arguments ---
    parser.add_argument('--test-type', type=str, required=True, choices=['constant', 'stress'], help="The type of test to run.")
    parser.add_argument('--num-runs', type=int, default=1, help="Number of times to repeat the experiment.")

    # --- Constant test arguments ---
    constant_group = parser.add_argument_group('constant test options')
    constant_group.add_argument('--rate', type=int, help="Required for constant test: Number of iterations to START per second.")
    constant_group.add_argument('--duration', type=str, help="Required for constant test: Duration for the k6 test (e.g., '5m', '1h').")
    constant_group.add_argument('--warmup', type=str, default='30s', help="For constant test: Time to exclude from the start for metrics.")
    constant_group.add_argument('--cooldown', type=str, default='15s', help="For constant test: Time to exclude from the end for metrics.")

    # --- Stress test arguments (with defaults) ---
    stress_group = parser.add_argument_group('stress test options')
    stress_group.add_argument('--start-rate', type=int, default=10, help="For stress test: The initial arrival rate. (Default: 10)")
    stress_group.add_argument('--peak-rate', type=int, default=2000, help="For stress test: The peak arrival rate to ramp up to. (Default: 2000)")
    stress_group.add_argument('--ramp-up', type=str, default='10m', help="For stress test: Duration of the ramp-up phase. (Default: '10m')")
    stress_group.add_argument('--sustain', type=str, default='5m', help="For stress test: Duration to sustain the peak rate. (Default: '5m')")
    stress_group.add_argument('--ramp-down', type=str, default='1m', help="For stress test: Duration of the ramp-down phase. (Default: '1m')")
    stress_group.add_argument('--max-vus', type=int, default=200, help="For stress test: Max VUs to pre-allocate. (Default: 200)")

    # --- Backoff options (apply to both test types) ---
    backoff_group = parser.add_argument_group('backoff options')
    backoff_group.add_argument('--backoff-timeout-s', dest='backoff_timeout_s', type=float, help="Sleep seconds after timeout/network error (default 0.5 if unset)")
    backoff_group.add_argument('--backoff-5xx-s', dest='backoff_5xx_s', type=float, help="Sleep seconds after 5xx/server errors (default 0.2 if unset)")

    args = parser.parse_args()

    # --- Argument validation ---
    if args.test_type == 'constant':
        if not all([args.rate, args.duration]):
            parser.error("--rate and --duration are required when --test-type is 'constant'")
    elif args.test_type == 'stress':
        if args.num_runs > 1:
            parser.error("--num-runs cannot be greater than 1 for a 'stress' test.")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = RESULTS_BASE_DIR / f"experiment_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(results_dir / "orchestrator.txt")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Starting new experiment. Results will be saved in: {results_dir}")
    
    create_metadata_file(results_dir, args)
    
    scenarios = parse_inventory(INVENTORY_PATH)

    for run in range(1, args.num_runs + 1):
        logging.info(f"--- Starting Run {run} of {args.num_runs} ---")
        
        tasks = [(run, scenario, results_dir, args) for scenario in scenarios]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_single_scenario_lifecycle, *task) for task in tasks]
            
            for future in as_completed(futures):
                scenario_name, success = future.result()
                if success:
                    logging.info(f"[Run {run:02d}] Scenario '{scenario_name}' completed successfully.")
                else:
                    logging.error(f"[Run {run:02d}] Scenario '{scenario_name}' FAILED.")
        
        logging.info(f"--- Finished Run {run} of {args.num_runs} ---")

    logging.info("Experiment finished.")


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed. Please run: pip install PyYAML", file=sys.stderr)
        sys.exit(1)
        
    main()
