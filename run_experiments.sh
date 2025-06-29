#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- ANSIBLE CONFIGURATION ---
# Disable host key checking for non-interactive execution
export ANSIBLE_HOST_KEY_CHECKING=False

# --- CONFIGURATION ---
NUM_RUNS=15
K6_RPS=100
K6_DURATION="5m"
# SERVER_TYPES="CSR SSR" # No longer needed, handled explicitly
INVENTORY_FILE="ansible/inventory.yml"
PLAYBOOK="ansible/run_constant_load_test.yml"

# --- TRAP FOR CLEANUP ---
# Ensure background jobs are killed if the script exits unexpectedly
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# --- PRE-FLIGHT CHECKS ---
echo "INFO: Starting pre-flight checks..."

# Check 1: Ensure yq is installed
if ! command -v yq &> /dev/null
then
    echo "ERROR: yq command could not be found. Please install yq to proceed. (sudo snap install yq)"
    echo "Installation instructions: https://github.com/mikefarah/yq/#install"
    exit 1
fi
echo "INFO: [✓] yq is installed."

# Check 2: Ensure Ansible inventory exists
if [ ! -f "$INVENTORY_FILE" ]; then
    echo "ERROR: Ansible inventory not found at $INVENTORY_FILE."
    echo "Please run Terraform to create the infrastructure first."
    exit 1
fi
echo "INFO: [✓] Ansible inventory found."

# --- SETUP ---

# Define the project root for absolute path references
PROJECT_ROOT=$(pwd)

# Build the k6 test archive. This is a prerequisite for the Ansible playbook.
echo "INFO: Building k6 test archive..."
if ! (cd k6 && ./build.sh); then
    echo "ERROR: Failed to build k6 test archive."
    exit 1
fi
echo "INFO: [✓] k6 test archive built successfully."

# Get Prometheus URL from inventory
PROMETHEUS_IP=$(yq '.all.children.role_monitoring_server.hosts.*.ansible_host' "$INVENTORY_FILE" | head -n 1)
if [ -z "$PROMETHEUS_IP" ] || [ "$PROMETHEUS_IP" == "null" ]; then
    echo "ERROR: Could not parse Prometheus IP from inventory file. 'yq' returned an empty or null value."
    exit 1
fi
PROMETHEUS_URL="http://${PROMETHEUS_IP}:9090"
echo "INFO: Prometheus URL set to: $PROMETHEUS_URL"

# Create a unique, timestamped output directory for this experiment
OUTPUT_DIR="results/run_$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
mkdir -p "$OUTPUT_DIR"
echo "INFO: Results will be saved in: $OUTPUT_DIR"

# Activate Python virtual environment
VENV_PATH="statistics/venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "ERROR: Python virtual environment not found at $VENV_PATH"
    echo "Please run 'python3 -m venv statistics/venv' to create it."
    exit 1
fi
source "$VENV_PATH"
echo "INFO: Python virtual environment activated."

# --- HELPER FUNCTION ---
run_test_series() {
    local server_type=$1
    local output_dir=$2
    
    echo "INFO: Starting test series for server type: $server_type"
    for (( run_num=1; run_num<=NUM_RUNS; run_num++ )); do
        echo -e "\n--- Starting Run $run_num / $NUM_RUNS for $server_type ---"

        # 1. Capture real start time
        local REAL_START_TIME=$(date -u --iso-8601=seconds)
        echo "INFO: [$server_type] Actual test start time: $REAL_START_TIME"

        # 2. Run the Ansible playbook to execute the k6 test
        echo "INFO: [$server_type] Running k6 test via Ansible..."
        ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK" \
            --extra-vars "k6_rps=${K6_RPS} k6_duration=${K6_DURATION} prometheus_url=${PROMETHEUS_URL} project_root=${PROJECT_ROOT}" \
            -l "role_load_generator_${server_type,,}"

        # 3. Capture real end time
        local REAL_END_TIME=$(date -u --iso-8601=seconds)
        echo "INFO: [$server_type] Actual test end time: $REAL_END_TIME"

        # 4. Calculate the adjusted time window for metric collection
        echo "INFO: [$server_type] Adjusting time window by a 1-minute margin on each end."
        local METRICS_START_TIME=$(date -d "$REAL_START_TIME + 1 minute" -u --iso-8601=seconds)
        local METRICS_END_TIME=$(date -d "$REAL_END_TIME - 1 minute" -u --iso-8601=seconds)
        echo "INFO: [$server_type] Metrics collection window: $METRICS_START_TIME to $METRICS_END_TIME"

        # 5. Run the Python collector script
        echo "INFO: [$server_type] Collecting metrics from Prometheus..."
        python3 statistics/collector.py \
            --prometheus-url "$PROMETHEUS_URL" \
            --start "$METRICS_START_TIME" \
            --end "$METRICS_END_TIME" \
            --server-type "$server_type" \
            --run-number "$run_num" \
            --output-dir "$output_dir"
        
        echo "INFO: [$server_type] Run $run_num completed."
    done
    echo "INFO: Test series for $server_type completed successfully."
}

# --- EXPERIMENT EXECUTION ---

echo -e "\n========================================================="
echo "INFO: Starting test series for CSR and SSR in parallel."
echo "INFO: CSR run log will be saved to $OUTPUT_DIR/csr_run.log"
echo "========================================================="

# Run CSR tests in the background, redirecting output to a log file
run_test_series "CSR" "$OUTPUT_DIR" > "$OUTPUT_DIR/csr_run.log" 2>&1 &
CSR_PID=$!

# Run SSR tests in the foreground
run_test_series "SSR" "$OUTPUT_DIR"

# Wait for the background CSR process to finish and check its exit code
echo "INFO: SSR test series finished. Waiting for CSR series to complete..."
wait $CSR_PID
echo "INFO: CSR test series completed."

# Deactivate Python environment
deactivate
echo -e "\n========================================================="
echo "INFO: All experiments completed successfully."
echo "INFO: Raw data saved in $OUTPUT_DIR"
echo "INFO: To analyze the results, run:"
echo "source $VENV_PATH"
echo "python3 statistics/analyzer.py --input-dir $OUTPUT_DIR"
echo "=========================================================" 