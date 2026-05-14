import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from orchestrator.actions.setup.provider import load_infra_config
from orchestrator.actions.test.runner import TestRunner
from orchestrator.config import RESULTS_DIR
from orchestrator.shared.infra import CloudEnvironment, InfrastructureError

console = Console()

CAMPAIGN_STATE_FILE = "campaign_state.json"


def run_campaign(
    path: Path,
    app_filter: str | None = None,
    resume: Path | None = None,
    infra_path: Path | None = None,
    verbose: bool = False,
) -> None:
    """
    Orchestrates a research campaign: Provision -> Warmup -> Test -> Rotate.
    """
    # 1. Load Campaign Config
    if not path.exists():
        console.print(f"[bold red]Error: Campaign config file not found at {path}[/bold red]")
        return

    with open(path, "r") as f:
        campaign_config = yaml.safe_load(f)

    # 2. Load Infrastructure Defaults
    try:
        base_infra_config = load_infra_config(infra_path)
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Configuration Error:[/bold red] {e}")
        return

    state: dict[str, Any] = {}
    if resume:
        campaign_dir = resume
        state_path = campaign_dir / CAMPAIGN_STATE_FILE
        if not state_path.exists():
            console.print(f"[bold red]Error: State file not found in {campaign_dir}[/bold red]")
            return
        with open(state_path, "r") as f:
            state = json.load(f)
        console.print(f"[yellow]Resuming campaign in {campaign_dir}...[/yellow]")
    else:
        # Create a new timestamped directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        campaign_dir = RESULTS_DIR / f"campaign_{timestamp}"
        campaign_dir.mkdir(parents=True, exist_ok=True)
        state_path = campaign_dir / CAMPAIGN_STATE_FILE
        state = {"completed_apps": [], "failed_apps": []}
        console.print(
            f"[bold magenta]Created new campaign directory: {campaign_dir}[/bold magenta]"
        )

    # 3. Filter Apps from YAML
    # The campaign YAML should define the experiment profile and the list of apps
    experiment_config = campaign_config.get("experiment", {})
    apps_in_config = experiment_config.get("apps", [])

    if app_filter:
        filters = [f.strip() for f in app_filter.split(",")]
        apps_to_test = [a for a in apps_in_config if any(f in a for f in filters)]
    else:
        apps_to_test = apps_in_config

    if not apps_to_test:
        console.print("[bold yellow]No apps found to test in campaign config.[/bold yellow]")
        return

    # Filter out completed apps if resuming
    apps_to_run = [a for a in apps_to_test if a not in state["completed_apps"]]

    if not apps_to_run:
        console.print("[bold green]All requested apps have already been completed.[/bold green]")
        return

    console.print(
        f"[bold magenta]Starting Campaign: {len(apps_to_run)} subjects to test.[/bold magenta]"
    )

    env = CloudEnvironment()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            disable=verbose,
        ) as progress:
            campaign_task = progress.add_task("[magenta]Campaign Progress", total=len(apps_to_test))
            progress.update(campaign_task, completed=len(state["completed_apps"]))

            for app_name in apps_to_run:
                progress.update(
                    campaign_task, description=f"[magenta]Processing [bold]{app_name}[/bold]..."
                )

                try:
                    # --- Step 0: Strict Whitelist Check ---
                    infra_technologies = base_infra_config.get("technologies", {})
                    if app_name not in infra_technologies:
                        raise ValueError(
                            f"App '{app_name}' is not defined in infrastructure config.\n"
                            "Campaign aborted for this subject."
                        )

                    # --- Step 1: Provision & Configure ---
                    progress.console.print(
                        f"[cyan][{app_name}] Provisioning infrastructure...[/cyan]"
                    )
                    infra_config = base_infra_config.copy()
                    # We only provision ONE app at a time
                    infra_config["technologies"] = {app_name: infra_technologies[app_name]}
                    env.setup(infra_config, verbose=verbose)

                    # --- Step 2: Warmup (50 RPS with assets) ---
                    progress.console.print(
                        f"[cyan][{app_name}] Starting Warmup (50 RPS, assets ENABLED)...[/cyan]"
                    )
                    warmup_config = {
                        "test_type": "load",
                        "num_runs": 1,
                        "load_options": {
                            "rps": 50,
                            "warmup": "0s",
                            "duration": "1m",
                            "after": "0s",
                            "vus": 10,
                            "skip_assets": False,
                        },
                    }
                    # Warmup results go to a hidden/separate folder to avoid cluttering aggregation
                    warmup_runner = TestRunner(
                        None,
                        config_dict=warmup_config,
                        output_dir=campaign_dir / ".warmup" / app_name,
                    )
                    warmup_runner.run_all()

                    # --- Step 3: Experiment (Profile from YAML) ---
                    progress.console.print(f"[cyan][{app_name}] Starting Experiment...[/cyan]")

                    # The campaign experiment config is shared across all apps in the loop
                    # All apps save to the SAME FLAT metrics directory for aggregation compatibility
                    runner = TestRunner(
                        None, config_dict=experiment_config, output_dir=campaign_dir
                    )
                    runner.run_all()

                    # --- Step 4: Finalize Subject ---
                    state["completed_apps"].append(app_name)
                    with open(state_path, "w") as f:
                        json.dump(state, f, indent=2)

                    progress.console.print(
                        f"[bold green]\u2713 {app_name} completed successfully.[/bold green]"
                    )

                except Exception as e:
                    progress.console.print(f"[bold red]\u2717 {app_name} failed: {e}[/bold red]")
                    state["failed_apps"].append({"name": app_name, "error": str(e)})
                    with open(state_path, "w") as f:
                        json.dump(state, f, indent=2)

                    # Continue to next app unless we hit a critical infra error
                    if isinstance(e, InfrastructureError):
                        progress.console.print(
                            "[red]Critical Infrastructure Error. Stopping campaign.[/red]"
                        )
                        break

                progress.advance(campaign_task)

    except KeyboardInterrupt:
        console.print("\n[bold red]Campaign interrupted. State saved.[/bold red]")
    finally:
        # Final cleanup: Destroy all infrastructure
        console.print(
            "[bold yellow]Campaign finished or stopped. Tearing down final subject...[/bold yellow]"
        )
        try:
            env.teardown(verbose=verbose)
        except Exception as e:
            console.print(f"[red]Teardown failed: {e}[/red]")

    console.print("\n[bold magenta]Campaign Summary:[/bold magenta]")
    console.print(f"  Completed: {len(state['completed_apps'])}")
    console.print(f"  Failed:    {len(state['failed_apps'])}")
    console.print(f"Results available in: [bold]{campaign_dir}[/bold]")
