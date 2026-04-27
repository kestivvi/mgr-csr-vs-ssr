import json
import os
from pathlib import Path
from typing import Any

import ansible_runner
import yaml
from rich.console import Console

from orchestrator.config import ANSIBLE_DIR, TERRAFORM_DIR
from orchestrator.shared.runner import run_command

console = Console()

DEFAULT_INFRA = {
    "aws_region": "ap-south-1",
    "key_name": "MGR-M",
    "my_ip": "46.205.195.40/32",
    "app_server_instance_type": "t4g.micro",
    "load_generator_instance_type": "t4g.micro",
    "monitoring_server_instance_type": "t4g.micro",
    "technologies": {
        "CSR-Vanilla": {
            "description": "Application Server (CSR-Vanilla)",
            "purpose": "Hosts Client-Side Rendered application",
            "app_dir": "apps/csr-vanilla-nginx",
        }
    },
}


def load_infra_config(path: Path) -> dict[str, Any]:
    """Loads and merges infrastructure configuration."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Merge with defaults
    merged = DEFAULT_INFRA.copy()
    if config:
        merged.update(config)
    return merged


def run_setup(infra_path: Path, force: bool = False) -> None:
    """Orchestrates the infrastructure setup."""

    # 0. Load Configuration
    config = load_infra_config(infra_path)

    # 1. Terraform Init & Apply
    console.print("[bold green]Step 1: Terraform Provisioning[/bold green]")
    run_command(["terraform", "init"], cwd=str(TERRAFORM_DIR))

    apply_cmd = ["terraform", "apply", "-auto-approve"]

    # Construct -var arguments
    for key, value in config.items():
        if isinstance(value, (dict, list)):
            # Complex types must be passed as JSON strings for Terraform
            tf_value = json.dumps(value)
        else:
            tf_value = str(value)
        apply_cmd.extend(["-var", f"{key}={tf_value}"])

    if force:
        # Note: Handled by Terraform -replace if needed, but for now we just run apply
        pass

    env = os.environ.copy()
    env["TF_IN_AUTOMATION"] = "true"

    rc = run_command(apply_cmd, cwd=str(TERRAFORM_DIR), env=env)
    if rc != 0:
        console.print("[bold red]Terraform failed. Aborting.[/bold red]")
        return

    # 2. Ansible Configuration
    console.print("[bold green]Step 2: Ansible Configuration[/bold green]")

    from orchestrator.shared.ansible import get_ansible_env

    # Using ansible-runner with dynamic environment
    r = ansible_runner.run(
        private_data_dir=str(ANSIBLE_DIR),
        playbook="site.yml",
        quiet=False,
        envvars=get_ansible_env(),
    )

    if r.rc != 0:
        console.print(f"[bold red]Ansible failed with return code {r.rc}[/bold red]")
    else:
        console.print("[bold green]Infrastructure is ready![/bold green]")
