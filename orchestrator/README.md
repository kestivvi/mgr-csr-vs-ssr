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

Execute performance tests based on a YAML configuration.

```bash
mgr test --config experiments/load_test.yaml
```

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
# Run quality checks
ruff check .
mypy .
```
