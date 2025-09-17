#!/bin/bash
#
# Orchestrates a series of k6 load tests against different server types in parallel.
# It handles dependency checks, setup, test execution via Ansible, and metrics
# collection via a Python script.

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipe commands will fail if any command in the pipe fails.
set -o pipefail

# --- SCRIPT CONSTANTS & PATHS ---
# These define the script's integration with the project structure and tools.
# Avoid changing these unless you are modifying the project structure itself.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$SCRIPT_DIR"

# --- ANSIBLE CONFIGURATION ---
# Explicitly set the path to the ansible.cfg file. This is the crucial fix.
# It ensures Ansible always finds its configuration, including the correct remote_user.
export ANSIBLE_CONFIG="${PROJECT_ROOT}/ansible/ansible.cfg"
# Disable host key checking for non-interactive execution
export ANSIBLE_HOST_KEY_CHECKING=False

# --- USER CONFIGURATION ---
# These are the primary variables you might want to change between test runs.
# SERVER_TYPES is now discovered automatically from the inventory.
NUM_RUNS=2
K6_RPS=500
K6_DURATION="3m"
# Test path: 'static' for "/" or 'dynamic' for "/dynamic/{random-number}"
TEST_PATH="static"
# Margin to trim from the start and end of the test window for stable metrics.
METRICS_TIME_MARGIN="1 minute"

# --- TOOL INTEGRATION & FILE PATHS ---
PROMETHEUS_HOST_QUERY='.all.children.role_monitoring_server.hosts.*.ansible_host'
ANSIBLE_GROUP_TEMPLATE="role_load_generator_"
INVENTORY_FILE="${PROJECT_ROOT}/ansible/inventory.yml"
PLAYBOOK="${PROJECT_ROOT}/ansible/run_constant_load_test.yml"
K6_BUILD_DIR="${PROJECT_ROOT}/k6"
RESULTS_BASE_DIR="${PROJECT_ROOT}/results"
PYTHON_COLLECTOR_SCRIPT="${PROJECT_ROOT}/statistics/collector.py"
PYTHON_ANALYZER_SCRIPT="${PROJECT_ROOT}/statistics/analyzer.py"

# Global variables populated by the setup function.
PROMETHEUS_URL=""
OUTPUT_DIR=""
# Declare SERVER_TYPES as an array to be populated dynamically
declare -a SERVER_TYPES

# --- TRAP FOR CLEANUP ---
# Ensure background jobs are killed if the script exits unexpectedly
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# --- HELPER FUNCTIONS ---

# Centralized logging function for consistent output format.
log() {
    echo "INFO: $(date -u --iso-8601=seconds) | $*"
}

execute_k6_test() {
    local server_type_lower=$1
    # prometheus_url is no longer needed here, Ansible will handle it.
    local ansible_group="${ANSIBLE_GROUP_TEMPLATE}${server_type_lower}"
    local server_type_upper=${server_type_lower^^}
    
    log "[${server_type_upper}] Running k6 test via Ansible on group '$ansible_group'..."

    # Build --extra-vars as a single JSON string for robustness.
    local extra_vars_json
    # --- THIS IS THE FIX ---
    # REMOVED prometheus_url from this command. The playbook now correctly
    # uses its internal definition based on the private IP.
    extra_vars_json=$(printf '{"k6_rps": "%s", "k6_duration": "%s", "test_path": "%s", "project_root": "%s"}' \
        "$K6_RPS" "$K6_DURATION" "$TEST_PATH" "$PROJECT_ROOT")

    # Build the full command in an array to prevent quoting issues.
    local cmd=(
        ansible-playbook
        -i "$INVENTORY_FILE"
        "$PLAYBOOK"
        --extra-vars "$extra_vars_json"
        -l "$ansible_group"
    )

    # Execute the command.
    "${cmd[@]}"

    # The Ansible playbook returns immediately after starting the container.
    # We must explicitly wait for the test duration to complete.
    log "[${server_type_upper}] Test started. Waiting for K6 duration (${K6_DURATION}) to complete..."
    sleep "$K6_DURATION"
}

collect_prometheus_metrics() {
    local server_type_upper=$1
    local run_num=$2
    local start_time=$3
    local end_time=$4
    local prometheus_url=$5
    local output_dir=$6

    log "[${server_type_upper}] Collecting metrics from Prometheus..."
    python3 "$PYTHON_COLLECTOR_SCRIPT" \
        --prometheus-url "$prometheus_url" \
        --start "$start_time" \
        --end "$end_time" \
        --server-type "$server_type_upper" \
        --run-number "$run_num" \
        --output-dir "$output_dir"
}

run_test_series_for_type() {
    local server_type_lower=$1
    local prometheus_url=$2
    local output_dir=$3
    local server_type_upper=${server_type_lower^^} # Convert to uppercase for display/external use
    
    log "Starting test series for server type: $server_type_upper"
    for (( run_num=1; run_num<=NUM_RUNS; run_num++ )); do
        log "--- Starting Run $run_num / $NUM_RUNS for $server_type_upper ---"

        local real_start_time
        real_start_time=$(date -u --iso-8601=seconds)
        log "[$server_type_upper] Actual test start time: $real_start_time"

        # Pass the prometheus_url (public) to the execute function,
        # but it will only be used for metrics collection, not the test itself.
        execute_k6_test "$server_type_lower"

        local real_end_time
        real_end_time=$(date -u --iso-8601=seconds)
        log "[$server_type_upper] Actual test end time: $real_end_time"

        log "[$server_type_upper] Adjusting time window by a ${METRICS_TIME_MARGIN} margin on each end."
        local metrics_start_time
        metrics_start_time=$(date -d "$real_start_time + ${METRICS_TIME_MARGIN}" -u --iso-8601=seconds)
        local metrics_end_time
        metrics_end_time=$(date -d "$real_end_time - ${METRICS_TIME_MARGIN}" -u --iso-8601=seconds)
        log "[$server_type_upper] Metrics collection window: $metrics_start_time to $metrics_end_time"

        collect_prometheus_metrics "$server_type_upper" "$run_num" "$metrics_start_time" "$metrics_end_time" "$prometheus_url" "$output_dir"
        
        log "[$server_type_upper] Run $run_num completed."
    done
    log "Test series for $server_type_upper completed successfully."
}

# --- MAIN LOGIC FUNCTIONS ---

preflight_checks() {
    log "Starting pre-flight checks..."
    local dependencies=("yq" "ansible-playbook" "python3")
    for cmd in "${dependencies[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "ERROR: Required command '$cmd' could not be found." >&2
            echo "Please ensure it is installed and in your PATH." >&2
            exit 1
        fi
        log "[✓] $cmd is installed."
    done

    # Check for required Python packages to fail fast.
    log "Checking for required Python packages..."
    if ! python3 -c "import pandas, requests, numpy" &> /dev/null; then
        echo "ERROR: Missing required Python packages (e.g., pandas, requests, numpy)." >&2
        echo "Please install them in your active Python environment (e.g., 'pip install pandas requests numpy')." >&2
        exit 1
    fi
    log "[✓] Required Python packages are installed."

    # Specifically check for GNU date, as its flags are used for time arithmetic.
    if ! date -d "now + 1 min" >/dev/null 2>&1; then
        echo "ERROR: This script requires GNU 'date' for time calculations." >&2
        exit 1
    fi
    log "[✓] GNU date is available."

    if [ ! -f "$INVENTORY_FILE" ]; then
        echo "ERROR: Ansible inventory not found at $INVENTORY_FILE." >&2
        exit 1
    fi
    log "[✓] Ansible inventory found."
}

setup() {
    log "Starting setup..."
    
    log "Building k6 test archive..."
    if ! (cd "$K6_BUILD_DIR" && ./build.sh); then
        echo "ERROR: Failed to build k6 test archive." >&2
        exit 1
    fi
    log "[✓] k6 test archive built successfully."

    local prometheus_ip
    prometheus_ip=$(yq "$PROMETHEUS_HOST_QUERY" "$INVENTORY_FILE" | head -n 1)
    if [ -z "$prometheus_ip" ] || [ "$prometheus_ip" == "null" ]; then
        echo "ERROR: Could not parse Prometheus IP from inventory file using query." >&2
        exit 1
    fi
    PROMETHEUS_URL="http://${prometheus_ip}:9090"
    log "Prometheus URL set to: $PROMETHEUS_URL"

    OUTPUT_DIR="${RESULTS_BASE_DIR}/run_$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
    mkdir -p "$OUTPUT_DIR"
    log "Results will be saved in: $OUTPUT_DIR"

    # --- DYNAMICALLY DISCOVER SERVER TYPES ---
    log "Discovering server types from inventory..."
    
    # Use mapfile (readarray) to correctly read multi-line output into an array.
    # This is the robust way to handle discovery of one or more server types.
    # This is the critical fix.
    mapfile -t SERVER_TYPES < <(yq '.all.children.role_load_generators.children | keys | .[]' "$INVENTORY_FILE" | sed "s/${ANSIBLE_GROUP_TEMPLATE}//")

    if [ ${#SERVER_TYPES[@]} -eq 0 ]; then
        echo "ERROR: No server types discovered from inventory. Check inventory structure and yq query." >&2
        exit 1
    fi
    log "[✓] Discovered server types: ${SERVER_TYPES[*]}"
}

run_experiments() {
    local prometheus_url=$1
    local output_dir=$2

    log "========================================================="
    log "Starting test series for all server types in parallel."
    log "Types to be tested: ${SERVER_TYPES[*]}"
    log "========================================================="

    local pids=()
    declare -A pid_map
    for type in "${SERVER_TYPES[@]}"; do
        local log_file="${output_dir}/${type,,}_run.log"
        log "Launching test for '$type'. Log file: $log_file"
        
        # Launch the entire test series for this type in the background
        run_test_series_for_type "$type" "$prometheus_url" "$output_dir" > "$log_file" 2>&1 &
        
        # Capture the Process ID (PID) of the background job
        local pid=$!
        pids+=($pid)
        pid_map[$pid]="Test for '$type' (Log: $log_file)"
    done

    log "All test series launched. Waiting for completion..."
    local has_failed=0
    # Loop through all the captured PIDs and wait for them to finish
    for pid in "${pids[@]}"; do
        # 'wait' will return a non-zero exit code if the background job failed
        if ! wait "$pid"; then
            echo "ERROR: ${pid_map[$pid]} FAILED." >&2
            has_failed=1
        fi
    done

    if [ "$has_failed" -ne 0 ]; then
        echo "ERROR: One or more test series failed. Please review the logs in $output_dir" >&2
        exit 1
    fi
}

print_summary() {
    local output_dir=$1
    local venv_activate_path="${PROJECT_ROOT}/statistics/venv/bin/activate"

    log "========================================================="
    log "All experiments completed successfully."
    log "Raw data saved in $output_dir"
    # Use plain echo for user instructions that are meant to be copied.
    echo ""
    echo "To analyze the results, first activate your Python virtual environment:"
    echo "source $venv_activate_path"
    echo ""
    echo "Then, run the analyzer script:"
    echo "python3 $PYTHON_ANALYZER_SCRIPT --input-dir $output_dir"
    echo "========================================================="
}

# --- MAIN EXECUTION ---

main() {
    preflight_checks
    setup
    run_experiments "$PROMETHEUS_URL" "$OUTPUT_DIR"
    print_summary "$OUTPUT_DIR"
}

# Run the main function
main