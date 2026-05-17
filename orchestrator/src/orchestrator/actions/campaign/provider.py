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
    subject_filter: str | None = None,
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
        state = {"completed_subjects": [], "failed_subjects": []}
        console.print(
            f"[bold magenta]Created new campaign directory: {campaign_dir}[/bold magenta]"
        )

    # 3. Filter Subjects from YAML
    # The campaign YAML should define the experiment profile and the list of subjects
    experiment_config = campaign_config.get("experiment", {})
    subjects_in_config = experiment_config.get("subjects", [])

    if subject_filter:
        filters = [f.strip() for f in subject_filter.split(",")]
        subjects_to_test = [s for s in subjects_in_config if any(f in s for f in filters)]
    else:
        subjects_to_test = subjects_in_config

    if not subjects_to_test:
        console.print("[bold yellow]No subjects found to test in campaign config.[/bold yellow]")
        return

    # Filter out completed subjects if resuming
    subjects_to_run = [s for s in subjects_to_test if s not in state["completed_subjects"]]

    if not subjects_to_run:
        console.print(
            "[bold green]All requested subjects have already been completed.[/bold green]"
        )
        return

    console.print(
        f"[bold magenta]Starting Campaign: {len(subjects_to_run)} subjects to test.[/bold magenta]"
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
            campaign_task = progress.add_task(
                "[magenta]Campaign Progress", total=len(subjects_to_test)
            )
            progress.update(campaign_task, completed=len(state["completed_subjects"]))

            for subject_id in subjects_to_run:
                progress.update(
                    campaign_task, description=f"[magenta]Processing [bold]{subject_id}[/bold]..."
                )

                try:
                    # --- Step 0: Strict Whitelist Check ---
                    infra_technologies = base_infra_config.get("technologies", {})
                    if subject_id not in infra_technologies:
                        raise ValueError(
                            f"Subject '{subject_id}' is not defined in infrastructure config.\n"
                            "Campaign aborted for this subject."
                        )

                    # --- Step 1: Provision & Configure ---
                    progress.console.print(
                        f"[cyan][{subject_id}] Provisioning infrastructure...[/cyan]"
                    )
                    infra_config = base_infra_config.copy()
                    # We only provision ONE subject at a time
                    infra_config["technologies"] = {subject_id: infra_technologies[subject_id]}
                    env.setup(infra_config, verbose=verbose)

                    # --- Step 2: Warmup (50 RPS with assets) ---
                    progress.console.print(
                        f"[cyan][{subject_id}] Starting Warmup (50 RPS, assets ENABLED)...[/cyan]"
                    )
                    warmup_config = {
                        "test_type": "load",
                        "num_repetitions": 1,
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
                        output_dir=campaign_dir / ".warmup" / subject_id,
                    )
                    warmup_runner.run_all()

                    # --- Step 3: Experiment (Profile from YAML) ---
                    progress.console.print(f"[cyan][{subject_id}] Starting Experiment...[/cyan]")

                    # The campaign experiment config is shared across all subjects in the loop.
                    # All subjects save to the SAME FLAT directory for aggregation compatibility.
                    runner = TestRunner(
                        None, config_dict=experiment_config, output_dir=campaign_dir
                    )
                    runner.run_all()

                    # --- Step 4: Finalize Subject ---
                    state["completed_subjects"].append(subject_id)
                    with open(state_path, "w") as f:
                        json.dump(state, f, indent=2)

                    progress.console.print(
                        f"[bold green]\u2713 {subject_id} completed successfully.[/bold green]"
                    )

                except Exception as e:
                    progress.console.print(f"[bold red]\u2717 {subject_id} failed: {e}[/bold red]")
                    state["failed_subjects"].append({"name": subject_id, "error": str(e)})
                    with open(state_path, "w") as f:
                        json.dump(state, f, indent=2)

                    # Continue to next subject unless we hit a critical infra error
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
    console.print(f"  Completed: {len(state['completed_subjects'])}")
    console.print(f"  Failed:    {len(state['failed_subjects'])}")
    console.print(f"Results available in: [bold]{campaign_dir}[/bold]")
