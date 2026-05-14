from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from orchestrator.config import INFRA_EXAMPLE_YAML, INFRA_YAML
from orchestrator.shared.infra import CloudEnvironment, InfrastructureError

console = Console()


def load_infra_config(path: Path | None = None) -> dict[str, Any]:
    """
    Loads infrastructure configuration.

    Order of precedence:
    1. Provided path (CLI Argument)
    2. Default infra.yaml in orchestrator root

    If none found, raises FileNotFoundError with instruction to use infra.example.yaml.
    """
    target_path = path or INFRA_YAML

    if not target_path.exists():
        if target_path == INFRA_YAML:
            raise FileNotFoundError(
                f"Infrastructure config not found at {INFRA_YAML}.\n"
                f"Please copy {INFRA_EXAMPLE_YAML} to {INFRA_YAML} and configure it."
            )
        else:
            raise FileNotFoundError(f"Specified infrastructure config not found at {target_path}")

    with open(target_path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if not config:
        return {}

    return config


def run_setup(infra_path: Path | None = None, force: bool = False, verbose: bool = False) -> None:
    """Orchestrates the infrastructure setup using deep adapters."""

    try:
        # 0. Load Configuration
        config = load_infra_config(infra_path)
        env = CloudEnvironment()

        if force:
            console.print(
                "[bold red]Step 0: Destroying existing infrastructure (force)...[/bold red]"
            )
            env.teardown(verbose=verbose)

        with console.status(
            "[bold green]Step 1: Provisioning and Configuring Environment...[/bold green]",
            spinner="dots",
        ):
            # CloudEnvironment handles both Terraform and Ansible
            env.setup(config, verbose=verbose)

        console.print("[bold green]Infrastructure is ready![/bold green]")

    except FileNotFoundError as e:
        console.print(f"\n[bold red]Configuration Error:[/bold red] {e}")
        return
    except InfrastructureError as e:
        console.print(f"\n[bold red]Setup failed: {e}[/bold red]")
        if e.logs:
            console.print("[dim yellow]Tail of failure logs:[/dim yellow]")
            last_lines = "\n".join(e.logs.splitlines()[-20:])
            console.print(last_lines)
        return
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error during setup:[/bold red] {e}")
        return
