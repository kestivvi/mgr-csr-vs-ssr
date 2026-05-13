import json
import os
from pathlib import Path
from typing import Any

import ansible_runner
import yaml
from rich.console import Console

from orchestrator.config import ANSIBLE_DIR, TERRAFORM_DIR
from orchestrator.shared.infra import CloudEnvironment, InfrastructureError

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


def run_setup(infra_path: Path, force: bool = False, verbose: bool = False) -> None:
    """Orchestrates the infrastructure setup using deep adapters."""

    # 0. Load Configuration
    config = load_infra_config(infra_path)
    env = CloudEnvironment()

    try:
        if force:
            console.print("[bold red]Step 0: Destroying existing infrastructure (force)...[/bold red]")
            env.teardown(verbose=verbose)

        console.print("[bold green]Step 1: Provisioning and Configuring Environment...[/bold green]")
        # CloudEnvironment handles both Terraform and Ansible
        env.setup(config, verbose=verbose)
        
        console.print("[bold green]Infrastructure is ready![/bold green]")

    except InfrastructureError as e:
        console.print(f"\n[bold red]Setup failed: {e}[/bold red]")
        if e.logs:
            # For non-verbose runs, we can still show the last bit of logs on failure
            console.print("[dim yellow]Tail of failure logs:[/dim yellow]")
            last_lines = "\n".join(e.logs.splitlines()[-20:])
            console.print(last_lines)
        return
