# MGR Orchestrator

Unified Python orchestrator for managing the Master's Thesis experiment lifecycle.

## Prerequisites

- Python 3.10+
- Terraform
- Ansible

## Installation

1. **Create and activate the virtual environment**:

   ```bash
   cd orchestrator
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the package**:
   Installing in editable mode installs all dependencies and registers the `mgr` command.
   ```bash
   pip install -e .
   ```

## Usage

Once installed, you can use the `mgr` command from anywhere (within the virtual environment).

### 1. Infrastructure Setup

Provision AWS resources via Terraform and configure them via Ansible.

```bash
mgr setup
```

### 2. Run Experiments

Execute performance tests. There are three modes:

- **Load Tests**: Standard fixed-load tests.
  ```bash
  mgr test load --vus 20 --duration 1m
  ```
- **Capacity Tests**: Ramping-up tests to find breaking points.
  ```bash
  mgr test capacity --vus 200 --step-vus 20
  ```
- **Custom Files**: Use a YAML file for complex scenarios.
  ```bash
  mgr test file my_experiment.yaml
  ```

#### Custom Experiment File Schema

Custom YAML files are **strictly validated** using Pydantic. Here is an exhaustive example showing all available fields for different `test_type` modes.

```yaml
# my_experiment.yaml
test_type: capacity_k6 # options: load, capacity_k6, capacity_wrk
num_runs: 1

# --- If test_type is 'load' ---
load_options:
  rps: 100
  duration: 5m
  vus: 200
  path_type: dynamic # static or dynamic
  timeout: 0.4s

# --- If test_type is 'capacity_k6' ---
capacity_k6_options:
  peak_rate: 1000
  ramp_up: 5m
  sustain: 1m
  ramp_down: 1m
  start_rate: 1
  warmup: 0s
  max_vus: 200
  path_type: dynamic
  timeout: 0.4s

# --- If test_type is 'capacity_wrk' ---
capacity_wrk_options:
  threads: 2
  connections: 10
  duration: 30s
  warmup: 30s
```

> [!IMPORTANT]
> **Auto-Discovery**: The orchestrator automatically finds all provisioned app servers in your inventory. You do not need to specify hostnames or IPs in your configuration files.

#### Configuration Reference

| Block            | Field         | Type  | Default      | Description                              |
| :--------------- | :------------ | :---- | :----------- | :--------------------------------------- |
| **Root**         | `test_type`   | `str` | **Required** | `load`, `capacity_k6`, `capacity_wrk`    |
|                  | `num_runs`    | `int` | `1`          | Number of times to repeat the experiment |
| **Load**         | `rps`         | `int` | `100`        | Target Requests Per Second               |
|                  | `duration`    | `str` | `5m`         | Duration of the test                     |
|                  | `vus`         | `int` | `200`        | Number of virtual users                  |
|                  | `path_type`   | `str` | `dynamic`    | `static` or `dynamic` paths              |
|                  | `timeout`     | `str` | `0.4s`       | HTTP request timeout                     |
| **Capacity K6**  | `peak_rate`   | `int` | `1000`       | Target RPS at peak                       |
|                  | `ramp_up`     | `str` | `5m`         | Ramp up duration (e.g., `10m`)           |
|                  | `sustain`     | `str` | `1m`         | Sustain duration at peak                 |
|                  | `ramp_down`   | `str` | `1m`         | Ramp down duration                       |
|                  | `start_rate`  | `int` | `1`          | Starting RPS                             |
|                  | `warmup`      | `str` | `0s`         | Initial stay duration at `start_rate`    |
|                  | `max_vus`     | `int` | `200`        | Max pre-allocated VUs                    |
|                  | `path_type`   | `str` | `dynamic`    | `static` or `dynamic` paths              |
|                  | `timeout`     | `str` | `0.4s`       | HTTP request timeout                     |
| **Capacity Wrk** | `threads`     | `int` | `2`          | Number of threads                        |
|                  | `connections` | `int` | `10`         | Number of connections                    |
|                  | `duration`    | `str` | `30s`        | Duration of the test                     |
|                  | `warmup`      | `str` | `30s`        | Warmup duration                          |

### 3. Analyze Results

Generate statistical reports and plots.

```bash
mgr analyze results/load_YYYY-MM-DD
```

### 4. Teardown

Destroy all infrastructure.

```bash
mgr destroy
```

## Project Structure

- `src/orchestrator/`: Core package source code.
  - `actions/`: Domain-specific logic (setup, test, analyze, destroy).
  - `shared/`: Shared utilities (runner, logging).
  - `config.py`: Centralized path resolution (CWD independent).
  - `main.py`: CLI entry point definition.
- `pyproject.toml`: Modern Python project configuration and linter settings.
- `requirements.txt`: Flat dependency list.

## Development

This project uses **Ruff** for linting/formatting and **Mypy** for strict type checking.

```bash
# 1. Format code (auto-fixes line wrapping, indentation, etc.)
./venv/bin/ruff format .

# 2. Lint code (checks for logic errors and style violations)
./venv/bin/ruff check .

# 3. Type checking
./venv/bin/mypy .
```
