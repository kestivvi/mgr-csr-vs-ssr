# MGR Code

Repository for Master's Thesis project.

## Setup

### Prerequisites

- **Linux Environment**: Ubuntu/Debian (WSL2 supported)
- **AWS Account**: With EC2 permissions and a key pair (I named it "MGR1")

### Required Tools

#### Core Infrastructure

- **Terraform** - Infrastructure as Code (AWS provider ~> 5.0)
- **Ansible** - Configuration management
- **ansible-lint** - Ansible code linting
- **AWS CLI** - AWS authentication and configuration

#### Development and Testing

- **Docker** - Containerization platform
- **Docker Compose** - Multi-container orchestration
- **yq** - YAML processor for inventory parsing
- **k6** - Load testing tool (via Docker)

#### Python Environment

- **Python 3** - Runtime environment

### Configuration Requirements

#### AWS Setup

- AWS credentials configured (`aws configure`)
- EC2 key pair named "MGR1" with proper file permissions
- Terraform variables updated in `terraform/variables.tf`:
  - `my_ip`: Your public IP address
  - `key_name`: AWS key pair name
  - `aws_region`: AWS region (default: eu-central-1)

## Workflow

1. Install locally Terraform and Ansible.
2. Configure AWS CLI and Terraform variables (see [Configuration Requirements](#configuration-requirements)).
3. Setup Python virtual environment for experiments:
   ```bash
   # Note: If you moved the project directory, delete the old venv first:
   # rm -rf statistics/venv
   python3 -m venv statistics/venv
   source statistics/venv/bin/activate
   pip install -r statistics/requirements.txt
   ```
4. Run the setup script to provision infrastructure and configure servers:
   ```bash
   ./scripts/setup.sh
   ```

To remove all provisioned infrastructure, run:

```bash
./scripts/destroy.sh
```

## Frequent Commands

### 🏗️ Infrastructure (Terraform)

_Run inside `terraform/` directory._

- **Initial setup**: `terraform init`
- **Preview changes**: `terraform plan`
- **Deploy infrastructure**: `terraform apply -auto-approve`
- **Deploy with custom instances**: `terraform apply -var="app_server_instance_type=c8g.large" -var="load_generator_instance_type=c8g.2xlarge"`
- **Destroy everything**: `terraform destroy`

### ⚙️ Configuration (Ansible)

_Run inside `ansible/` directory._

- **Check connectivity**: `ansible-playbook ping.yml`
- **Full configuration (Setup containers/exporters)**: `ansible-playbook site.yml`
- **Clean all containers**: `ansible-playbook test_teardown.yml`

### 🧪 Running Experiments (Orchestrator)

The main tool for conducting research is the `scripts/experiments.py` orchestrator. It automates the process of running tests across all defined application servers and collecting metrics from Prometheus.

#### Prerequisites

1.  **Infrastructure**: Ensure AWS resources are deployed (`terraform apply`) and configured (`ansible-playbook site.yml`).
2.  **Environment**: Activate the Python virtual environment:
    ```bash
    source statistics/venv/bin/activate
    ```

#### 1. Capacity Test

This test implements the **ramping-arrival-rate** executor in k6. It gradually increases the load to identify the server's breaking point (maximum throughput).

```bash
python ./scripts/experiments.py --test-type capacity_k6 \
  --num-runs 1 \
  --peak-rate 1000 \
  --ramp-up 5m \
  --sustain 1m \
  --ramp-down 1m
```

**Parameters:**

- `--peak-rate`: The target requests per second (RPS) at the peak of the ramp-up.
- `--ramp-up`: Duration to linearly increase load from 0 to `peak-rate`.
- `--sustain`: How long to maintain the `peak-rate` before ramping down.
- `--max-vus`: (Optional) Maximum number of pre-allocated Virtual Users (default: 200). Increase if you hit VU limits at high RPS.

#### 2. Load Test

This test implements the **constant-arrival-rate** executor. It maintains a steady, predefined load throughout the test duration. It is ideal for studying application stability, resource consumption under sustained pressure, and determining precise latency percentiles.

```bash
python ./scripts/experiments.py --test-type load \
  --num-runs 3 \
  --rate 100 \
  --duration 5m \
  --warmup 30s \
  --cooldown 15s
```

**Parameters:**

- `--num-runs`: Number of times the entire experiment (for all apps) should be repeated. Essential for statistical significance (default: 1).
- `--rate`: Target requests per second (RPS).
- `--duration`: Duration of the main measurement phase.
- `--warmup`: Warm-up duration (default 30s). In `load` tests, this period is automatically excluded from the final metrics to ensure data represents steady-state performance. In `capacity` tests, it is included to capture the full ramp-up curve.
- `--cooldown`: Cooldown duration after the test (default 15s). Similar to warm-up, this is excluded from metrics in `load` tests.
- `--path-type`: (`static` or `dynamic`) Determines whether k6 should hit a static path or generate dynamic parameters (e.g., random IDs).

#### 3. Capacity Benchmark (wrk)

This test uses the **wrk** tool on Load Generators to measure maximum throughput and latency. It includes a mandatory warmup phase to ensure JIT optimization (essential for Node.js/V8 frameworks).

```bash
python ./scripts/experiments.py --test-type capacity_wrk \
  --num-runs 3 \
  --duration 30s \
  --warmup 30s \
  --threads 2 \
  --connections 100
```

**Parameters:**

- `--test-type capacity_wrk`: Explicitly selects the wrk-based scientific test.
- `--duration`: Duration of the measurement phase (e.g., `30s`, `1m`).
- `--warmup`: Duration of the warmup phase (results are discarded, but JIT is warmed up).
- `--threads`: Number of threads wrk should use.
- `--connections`: Number of open connections.

**Results:**

- **Client-Side**: Saved to `[run]_wrk_client_results.json` (includes RPS, Average Latency, and Transfer Rate).
- **Server-Side**: CPU, RAM, and Network metrics are collected from Prometheus as usual and saved to `[run]_[type]_[server].csv`.

### 🎓 Master's Thesis Standard Benchmark

For a Master's Thesis, reproducibility and statistical significance are paramount. The following configuration is the **recommended scientific standard** for your results chapter. It balances JIT warm-up, steady-state measurement, and AWS environmental noise reduction.

```bash
cd ./scripts
# Recommended Command for Thesis Results
sh ./setup.sh \
&& python3 ./experiments.py \
  --test-type capacity_wrk \
  --num-runs 5 \
  --duration 2m \
  --warmup 1m \
  --threads 2 \
  --connections 100 \
&& sh ./destroy.sh

# 4. Generate the scientific report
statistics/venv/bin/python3 statistics/analyzer.py \
  --input-dir results/capacity_wrk_YYYY-MM-DD_HH-MM-SS \
  --report-type capacity_wrk
```

**Scientific Rationale:**

- **Warmup (1m)**: Ensures that JIT compilers (V8 for Node, Bun) have fully optimized hot code paths and that TCP connection pools are stabilized.
- **Duration (2m)**: Provides enough time to capture multiple Garbage Collection (GC) cycles and average out micro-fluctuations in cloud network latency.
- **Runs (5x)**: The academic minimum for calculating the **Mean** and **95% Confidence Intervals**, effectively filtering out AWS "noisy neighbor" effects.

#### 🚀 Full Experiment Cycle (Automation)

For a fully automated run (Provision → Test → Destroy), you can chain the scripts together from the `scripts/` directory:

```bash
cd scripts
sh ./setup.sh && \
python3 ./experiments.py --test-type capacity_wrk \
  --num-runs 3 \
  --duration 30s \
  --warmup 30s \
  --threads 2 \
  --connections 100 && \
sh ./destroy.sh
```

#### 4. Champion Comparison (A/B Test)

#### 📊 Results and Monitoring

- **Results Directory**: Every experiment creates a unique folder in `results/[prefix]_[datetime]/`.
- **Artifacts**:
  - `metadata.yaml`: Parameters used for the run.
  - `orchestrator.txt`: Detailed logs of the orchestration process.
  - `[run]_[type]_[server].csv`: Raw metrics collected from Prometheus for each instance.
- **Emergency Stop**: To immediately terminate all running k6 containers on all load generators:
  ```bash
  ansible-playbook ./ansible/test_stop_all.yml
  ```

### 📊 Analysis

_Run from the project root._

1. **Activate Environment**:

   ```bash
   source statistics/venv/bin/activate
   ```

2. **Generate Capacity Report**:

   ```bash
   # Analyze specific experiment
   python statistics/analyzer.py --input-dir results/capacity_k6_YYYY-MM-DD_HH-MM-SS --report-type capacity

   # Analyze latest capacity_k6 experiment (Bash)
   python statistics/analyzer.py --input-dir $(ls -td results/capacity_k6_* | head -1) --report-type capacity
   ```

3. **Generate Load Report**:
   Use the `load` report type to compare all tested technologies in a ranking table and box plots.

   ```bash
   python statistics/analyzer.py --input-dir results/load_k6_YYYY-MM-DD_HH-MM-SS --report-type load
   ```

**Artifacts**:

- `capacity_report.md` or `report_all_apps.md`: Markdown summary with tables and charts.
- `plots/`: Subdirectory containing all generated `.png` charts.
