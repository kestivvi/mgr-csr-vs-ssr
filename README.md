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

#### 1. Capacity Test (Stress Test)

This test implements the **ramping-arrival-rate** executor in k6. It gradually increases the load to identify the server's breaking point (maximum throughput).

```bash
python ./scripts/experiments.py --test-type stress \
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

#### 2. Load Test (Constant Rate)

This test implements the **constant-arrival-rate** executor. It maintains a steady, predefined load throughout the test duration. It is ideal for studying application stability, resource consumption under constant pressure, and determining precise latency percentiles.

```bash
python ./scripts/experiments.py --test-type constant \
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
- `--warmup`: Warm-up duration (default 30s). In `constant` tests, this period is automatically excluded from the final metrics to ensure data represents steady-state performance. In `stress` tests, it is included to capture the full ramp-up curve.
- `--cooldown`: Cooldown duration after the test (default 15s). Similar to warm-up, this is excluded from metrics in `constant` tests.
- `--path-type`: (`static` or `dynamic`) Determines whether k6 should hit a static path or generate dynamic parameters (e.g., random IDs).

#### 3. Champion Comparison (A/B Test)

If you want to perform a deep statistical analysis (significance tests, effect size) between exactly two technologies, you can use a dedicated report. First, run a `constant` type test for the selected applications, and then:

```bash
python statistics/analyzer.py --input-dir results/experiment_... \
  --report-type champions \
  --champions CSR-Vanilla SSR-NextJS
```

#### 📊 Results and Monitoring

- **Results Directory**: Every experiment creates a unique folder in `results/experiment_YYYY-MM-DD_HH-MM-SS/`.
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

2. **Generate Capacity Report (Stress Test)**:

   ```bash
   # Analyze specific experiment
   python statistics/analyzer.py --input-dir results/experiment_YYYY-MM-DD_HH-MM-SS --report-type capacity

   # Analyze latest experiment (Bash)
   python statistics/analyzer.py --input-dir $(ls -td results/experiment_* | head -1) --report-type capacity
   ```

3. **Generate Comparison Report (Constant Load Test)**:
   Use the `all_apps` report type to compare all tested technologies in a ranking table and box plots.

   ```bash
   python statistics/analyzer.py --input-dir results/experiment_YYYY-MM-DD_HH-MM-SS --report-type all_apps
   ```

**Artifacts**:

- `capacity_report.md` or `report_all_apps.md`: Markdown summary with tables and charts.
- `plots/`: Subdirectory containing all generated `.png` charts.
