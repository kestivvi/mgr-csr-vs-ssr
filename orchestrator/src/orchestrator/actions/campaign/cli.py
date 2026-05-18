from pathlib import Path
from typing import Annotated, Optional

import typer

from orchestrator.actions.campaign.provider import run_campaign


def run(
    path: Annotated[Path, typer.Argument(help="Path to campaign experiment configuration YAML")],
    apps: Annotated[
        Optional[str],
        typer.Option(
            "--apps",
            help="Comma-separated list of apps to include (overrides YAML)",
        ),
    ] = None,
    resume: Annotated[
        Optional[Path],
        typer.Option("--resume", help="Path to campaign directory to resume from"),
    ] = None,
    infra: Annotated[
        Optional[Path],
        typer.Option("--infra", help="Path to infrastructure config (defaults to infra.yaml)"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
) -> None:
    """
    Run a sequential [bold magenta]Campaign[/bold magenta] across multiple applications.
    """
    run_campaign(path=path, app_filter=apps, resume=resume, infra_path=infra, verbose=verbose)
