from typing import Annotated, Optional

import typer
from rich.console import Console

# Import sub-commands
from orchestrator.actions.analyze import cli as analyze_cli
from orchestrator.actions.destroy import cli as destroy_cli
from orchestrator.actions.setup import cli as setup_cli
from orchestrator.actions.test import cli as test_cli

app = typer.Typer(
    help="MGR Orchestrator: Unified CLI for infrastructure, testing, and analysis.",
    rich_markup_mode="rich",
)
console = Console()


# Placeholder commands for now
@app.command()
def setup(
    force: Annotated[bool, typer.Option(help="Force recreation of infrastructure")] = False,
) -> None:
    """
    [bold green]Setup[/bold green]: Provision infrastructure via Terraform
    and configure via Ansible.
    """
    console.print("[yellow]Setup command initiated...[/yellow]")
    setup_cli.run(force=force)


@app.command()
def test(
    config: Annotated[
        str, typer.Option(help="Path to experiment config")
    ] = "experiments/default.yaml",
) -> None:
    """
    [bold blue]Test[/bold blue]: Run performance experiments (k6/wrk).
    """
    console.print(f"[yellow]Test command initiated with config: {config}[/yellow]")
    test_cli.run(config=config)


@app.command()
def analyze(
    results_dir: Annotated[str, typer.Argument(help="Directory containing experiment results")],
    report_type: Annotated[
        str, typer.Option(help="Type of report: load, capacity, champions, capacity_wrk")
    ] = "load",
    champions: Annotated[
        Optional[list[str]],
        typer.Option(help="Two technologies to compare for 'champions' report"),
    ] = None,
) -> None:
    """
    [bold magenta]Analyze[/bold magenta]: Generate statistical reports and plots from results.
    """
    console.print(f"[yellow]Analysis initiated for: {results_dir}[/yellow]")
    analyze_cli.run(results_dir=results_dir, report_type=report_type, champions=champions)


@app.command()
def destroy() -> None:
    """
    [bold red]Destroy[/bold red]: Tear down all infrastructure via Terraform.
    """
    console.print("[red]Teardown initiated...[/red]")
    destroy_cli.run()


if __name__ == "__main__":
    app()
