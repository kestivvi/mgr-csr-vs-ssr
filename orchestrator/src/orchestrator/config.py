import os
from pathlib import Path

# Base project structure
# Assumes this file is in mgr-code/orchestrator/src/orchestrator/config.py
ORCHESTRATOR_DIR = Path(__file__).parent.resolve()
ROOT_DIR = ORCHESTRATOR_DIR.parent.parent.parent.resolve()

# Infrastructure paths
TERRAFORM_DIR = ROOT_DIR / "terraform"
ANSIBLE_DIR = ROOT_DIR / "ansible"
ANSIBLE_INVENTORY = ANSIBLE_DIR / "inventory" / "inventory.yml"
ANSIBLE_PROJECT = ANSIBLE_DIR / "project"
INFRA_YAML = ROOT_DIR / "orchestrator" / "infra.yaml"
INFRA_EXAMPLE_YAML = ROOT_DIR / "orchestrator" / "infra.example.yaml"

# Application paths
APPLICATIONS_DIR = ROOT_DIR / "applications"
COMPOSE_DIR = APPLICATIONS_DIR / "_infra" / "compose"

# Universal Master Files (Docker Compose templates)
COMPOSE_CSR_NGINX = COMPOSE_DIR / "csr-nginx.yml"
COMPOSE_CSR_APACHE = COMPOSE_DIR / "csr-apache.yml"
COMPOSE_SSR_NGINX = COMPOSE_DIR / "ssr-nginx.yml"

# Output and results
RESULTS_DIR = ROOT_DIR / "results"
VERIFY_LOGS_BASE_DIR = ROOT_DIR / "verify_logs"

# Execution settings
DEFAULT_CONFIG_PATH = ROOT_DIR / "experiments" / "default.yaml"


# Playbook paths
ANSIBLE_OPS = ANSIBLE_PROJECT / "ops"
LOAD_PLAYBOOK = ANSIBLE_OPS / "test_load_run.yml"
CAPACITY_PLAYBOOK = ANSIBLE_OPS / "test_capacity_run.yml"
WRK_PLAYBOOK = ANSIBLE_OPS / "test_wrk_run.yml"


def resolve_path(path: str | Path) -> Path:
    """Helper to ensure we have an absolute path.
    Relative paths are resolved against the current working directory.
    """
    p = Path(path)
    return p.resolve()


# Environment overrides
ANSIBLE_CONFIG = ANSIBLE_DIR / "ansible.cfg"
os.environ["ANSIBLE_CONFIG"] = str(ANSIBLE_CONFIG)
