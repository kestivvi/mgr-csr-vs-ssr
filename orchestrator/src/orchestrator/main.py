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


test_app = typer.Typer(
    help="[bold blue]Test[/bold blue]: Run performance experiments (k6/wrk).",
    rich_markup_mode="rich",
)
app.add_typer(test_app, name="test")


@test_app.command(name="load")
def test_load(
    num_runs: Annotated[
        int, typer.Option("--num-runs", help="Number of runs", show_default=True)
    ] = 1,
    duration: Annotated[
        str, typer.Option("--duration", help="Duration", show_default=True)
    ] = "30s",
    vus: Annotated[
        int, typer.Option("--vus", help="Number of virtual users", show_default=True)
    ] = 10,
    rps: Annotated[Optional[int], typer.Option("--rps", help="Requests per second")] = None,
) -> None:
    """Run standard [cyan]load[/cyan] tests."""
    test_cli.run(mode="load", num_runs=num_runs, duration=duration, vus=vus, rps=rps)


@test_app.command(name="capacity")
def test_capacity(
    num_runs: Annotated[
        int, typer.Option("--num-runs", help="Number of runs", show_default=True)
    ] = 1,
    peak_rate: Annotated[
        int, typer.Option("--peak-rate", help="Target RPS at peak", show_default=True)
    ] = 1000,
    ramp_up: Annotated[
        str, typer.Option("--ramp-up", help="Ramp up duration", show_default=True)
    ] = "5m",
    sustain: Annotated[
        str, typer.Option("--sustain", help="Sustain duration at peak", show_default=True)
    ] = "1m",
    ramp_down: Annotated[
        str, typer.Option("--ramp-down", help="Ramp down duration", show_default=True)
    ] = "1m",
    start_rate: Annotated[
        int, typer.Option("--start-rate", help="Starting RPS", show_default=True)
    ] = 1,
) -> None:
    """Run [cyan]capacity[/cyan] tests (Thesis Standard RPS Ramping)."""
    test_cli.run(
        mode="capacity",
        num_runs=num_runs,
        peak_rate=peak_rate,
        ramp_up=ramp_up,
        sustain=sustain,
        ramp_down=ramp_down,
        start_rate=start_rate,
    )


@test_app.command(name="file")
def test_file(
    path: Annotated[str, typer.Argument(help="Path to custom experiment config")],
    num_runs: Annotated[
        Optional[int], typer.Option("--num-runs", help="Override number of runs")
    ] = None,
    duration: Annotated[Optional[str], typer.Option("--duration", help="Override duration")] = None,
    vus: Annotated[Optional[int], typer.Option("--vus", help="Override VUs")] = None,
) -> None:
    """Run experiments from a custom [cyan]YAML file[/cyan]."""
    test_cli.run(mode="file", path=path, num_runs=num_runs, duration=duration, vus=vus)


@test_app.command(name="stop")
def test_stop() -> None:
    """[bold red]Emergency Stop[/bold red]: Kill all running k6 containers on all generators."""
    from orchestrator.actions.test.runner import TestRunner

    runner = TestRunner(config_path=None)
    runner.global_teardown()


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
