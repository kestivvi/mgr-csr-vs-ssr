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

# Application paths
APPS_DIR = ROOT_DIR / "apps"

# Output and results
RESULTS_DIR = ROOT_DIR / "results"

# Execution settings
DEFAULT_CONFIG_PATH = ROOT_DIR / "experiments" / "default.yaml"


def resolve_path(path: str | Path) -> Path:
    """Helper to ensure we have an absolute path."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT_DIR / p).resolve()


# Environment overrides
ANSIBLE_CONFIG = ANSIBLE_DIR / "ansible.cfg"
os.environ["ANSIBLE_CONFIG"] = str(ANSIBLE_CONFIG)
