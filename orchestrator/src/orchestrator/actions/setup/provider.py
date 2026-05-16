from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from orchestrator.config import INFRA_EXAMPLE_YAML, INFRA_YAML
from orchestrator.shared.infra import CloudEnvironment, InfrastructureError

console = Console()


def load_infra_config(
    path: Path | None = None,
    subjects: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """
    Loads infrastructure configuration and injects discovered subjects.
    """
    from orchestrator.config import SUBJECTS_DIR
    from orchestrator.shared.research.subject import SubjectRegistry

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
        config = {}

    # --- Subject Autodiscovery ---
    registry = SubjectRegistry(SUBJECTS_DIR)
    discovered_techs = {}
    for s in registry.all():
        # Filter if subjects list is provided (Inclusion)
        if subjects and s.id not in subjects:
            continue

        # Filter if exclude list is provided (Exclusion)
        if exclude and s.id in exclude:
            continue

        discovered_techs[s.id] = {
            "description": s.display_name,
            "purpose": f"Hosts {s.family.capitalize()} {s.strategy.upper()} subject",
            "subject_dir": f"subjects/{s.id}",
            "metadata": s.to_dict(),
        }

    # Merge: Manual entries in infra.yaml can still override/augment discovered ones
    if "technologies" not in config:
        config["technologies"] = {}

    # We prioritize discovered techs to enforce the naming contract
    config["technologies"].update(discovered_techs)

    # Final filtering pass to ensure consistency
    final_techs = config["technologies"]
    if subjects:
        final_techs = {k: v for k, v in final_techs.items() if k in subjects}
    if exclude:
        final_techs = {k: v for k, v in final_techs.items() if k not in exclude}

    config["technologies"] = final_techs

    return config


def run_setup(
    infra_path: Path | None = None,
    force: bool = False,
    verbose: bool = False,
    subjects: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    """Orchestrates the infrastructure setup using deep adapters."""

    try:
        # 0. Load Configuration
        config = load_infra_config(infra_path, subjects=subjects, exclude=exclude)
        env = CloudEnvironment()

        if force:
            console.print(
                "[bold red]Step 0: Destroying existing infrastructure (force)...[/bold red]"
            )
            env.teardown(verbose=verbose)

        msg = "[bold green]Step 1: Provisioning and Configuring Environment...[/bold green]"
        if verbose:
            console.print(msg)
            env.setup(config, verbose=verbose)
        else:
            with console.status(msg, spinner="dots"):
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
