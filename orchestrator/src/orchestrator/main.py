from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

# Import sub-commands
# Sub-command CLIs are imported lazily inside functions to speed up startup/completion

app = typer.Typer(
    help="MGR Orchestrator: Unified CLI for infrastructure, testing, and analysis.",
    rich_markup_mode="rich",
)
console = Console()


# Placeholder commands for now
@app.command()
def setup(
    infra_path: Annotated[Path, typer.Argument(help="Path to infrastructure configuration YAML")],
    force: Annotated[bool, typer.Option(help="Force recreation of infrastructure")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")] = False,
) -> None:
    """
    [bold green]Setup[/bold green]: Provision infrastructure via Terraform
    and configure via Ansible.
    """
    console.print(f"[yellow]Setup command initiated with config: {infra_path}...[/yellow]")
    from orchestrator.actions.setup import cli as setup_cli

    setup_cli.run(infra_path=infra_path, force=force, verbose=verbose)


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
    inter_run_delay: Annotated[
        str,
        typer.Option(
            "--inter-run-delay",
            "--delay",
            help="Break duration between runs (e.g. 1m, 30s)",
            show_default=True,
        ),
    ] = "1m",
    duration: Annotated[
        str, typer.Option("--duration", help="Test (middle) duration", show_default=True)
    ] = "1m",
    warmup: Annotated[
        str, typer.Option("--warmup", help="Warmup duration", show_default=True)
    ] = "30s",
    after: Annotated[
        str, typer.Option("--after", help="After-test duration", show_default=True)
    ] = "30s",
    vus: Annotated[
        int, typer.Option("--vus", help="Number of virtual users", show_default=True)
    ] = 10,
    rps: Annotated[Optional[int], typer.Option("--rps", help="Requests per second")] = None,
) -> None:
    """Run standard [cyan]load[/cyan] tests."""
    from orchestrator.actions.test import cli as test_cli

    test_cli.run(
        mode="load",
        num_runs=num_runs,
        inter_run_delay=inter_run_delay,
        duration=duration,
        warmup=warmup,
        after=after,
        vus=vus,
        rps=rps,
    )


@test_app.command(name="capacity")
def test_capacity(
    num_runs: Annotated[
        int, typer.Option("--num-runs", help="Number of runs", show_default=True)
    ] = 1,
    inter_run_delay: Annotated[
        str,
        typer.Option(
            "--inter-run-delay",
            "--delay",
            help="Break duration between runs (e.g. 1m, 30s)",
            show_default=True,
        ),
    ] = "1m",
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
    peak_rate_2: Annotated[
        Optional[int], typer.Option("--peak-rate-2", help="Target RPS at peak 2")
    ] = None,
    ramp_up_2: Annotated[
        Optional[str], typer.Option("--ramp-up-2", help="Ramp up 2 duration")
    ] = None,
) -> None:
    """Run [cyan]capacity[/cyan] tests (Thesis Standard RPS Ramping)."""
    from orchestrator.actions.test import cli as test_cli

    test_cli.run(
        mode="capacity",
        num_runs=num_runs,
        inter_run_delay=inter_run_delay,
        peak_rate=peak_rate,
        ramp_up=ramp_up,
        sustain=sustain,
        ramp_down=ramp_down,
        start_rate=start_rate,
        peak_rate_2=peak_rate_2,
        ramp_up_2=ramp_up_2,
    )


@test_app.command(name="file")
def test_file(
    path: Annotated[str, typer.Argument(help="Path to custom experiment config")],
    num_runs: Annotated[
        Optional[int], typer.Option("--num-runs", help="Override number of runs")
    ] = None,
    inter_run_delay: Annotated[
        Optional[str],
        typer.Option(
            "--inter-run-delay",
            "--delay",
            help="Override break duration between runs (e.g. 1m, 30s)",
        ),
    ] = None,
    duration: Annotated[Optional[str], typer.Option("--duration", help="Override duration")] = None,
    warmup: Annotated[Optional[str], typer.Option("--warmup", help="Override warmup")] = None,
    after: Annotated[Optional[str], typer.Option("--after", help="Override after")] = None,
    vus: Annotated[Optional[int], typer.Option("--vus", help="Override VUs")] = None,
    peak_rate_2: Annotated[
        Optional[int], typer.Option("--peak-rate-2", help="Override peak rate 2")
    ] = None,
    ramp_up_2: Annotated[
        Optional[str], typer.Option("--ramp-up-2", help="Override ramp up 2 duration")
    ] = None,
) -> None:
    """Run experiments from a custom [cyan]YAML file[/cyan]."""
    from orchestrator.actions.test import cli as test_cli

    test_cli.run(
        mode="file",
        path=path,
        num_runs=num_runs,
        inter_run_delay=inter_run_delay,
        duration=duration,
        warmup=warmup,
        after=after,
        vus=vus,
        peak_rate_2=peak_rate_2,
        ramp_up_2=ramp_up_2,
    )


@test_app.command(name="capacity_local_wrk")
def test_capacity_local_wrk(
    apps: Annotated[
        Optional[str],
        typer.Option(
            "--apps",
            "--app",
            help="Filter apps by name (partial matches, comma-separated).",
        ),
    ] = None,
    num_runs: Annotated[
        int, typer.Option("--num-runs", help="Number of test runs per app", show_default=True)
    ] = 1,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")] = False,
) -> None:
    """Run local [cyan]capacity benchmarks[/cyan] using wrk."""
    from orchestrator.actions.test import cli as test_cli

    test_cli.run_local_wrk(app_filter=apps, num_runs=num_runs, verbose=verbose)


@test_app.command(name="stop")
def test_stop() -> None:
    """[bold red]Emergency Stop[/bold red]: Kill all running k6 containers on all generators."""
    from orchestrator.actions.test.runner import TestRunner

    runner = TestRunner(config_path=None)
    runner.global_teardown()


@app.command()
def analyze(
    report_type: Annotated[
        str, typer.Argument(help="Type of report: load, capacity_k6, champions, capacity_wrk")
    ],
    results_dir: Annotated[str, typer.Argument(help="Directory containing experiment results")],
    champions: Annotated[
        Optional[list[str]],
        typer.Option(help="Two technologies to compare for 'champions' report"),
    ] = None,
) -> None:
    """
    [bold magenta]Analyze[/bold magenta]: Generate statistical reports and plots from results.
    """
    console.print(f"[yellow]Analysis initiated for: {results_dir}[/yellow]")
    from orchestrator.actions.analyze import cli as analyze_cli

    analyze_cli.run(results_dir=results_dir, report_type=report_type, champions=champions)


@app.command()
def destroy(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")] = False,
) -> None:
    """
    [bold red]Destroy[/bold red]: Tear down all infrastructure via Terraform.
    """
    console.print("[red]Teardown initiated...[/red]")
    from orchestrator.actions.destroy import cli as destroy_cli

    destroy_cli.run(verbose=verbose)


@app.command()
def verify(
    apps: Annotated[
        Optional[str],
        typer.Option(
            "--apps",
            "--app",
            help=(
                "Filter apps by name. Supports [bold]partial matches[/bold] and "
                "[bold]comma-separated lists[/bold] (e.g., 'nextjs,react')."
            ),
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")] = False,
) -> None:
    """
    [bold green]Verify[/bold green]: Build and test apps locally to ensure functionality.
    """
    from orchestrator.actions.verify import cli as verify_cli

    verify_cli.run(app_filter=apps, verbose=verbose)


if __name__ == "__main__":
    app()
