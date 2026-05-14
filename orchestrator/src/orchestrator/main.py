from pathlib import Path
from typing import Annotated

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
    infra_path: Annotated[
        Path,
        typer.Argument(
            help="Path to infrastructure configuration YAML (e.g., infra.yaml).",
            show_default=False,
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Tear down existing infrastructure before setup",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
    auto_approve: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompts (useful for CI/CD)",
        ),
    ] = False,
) -> None:
    """
    [bold green]Setup[/bold green]: Provision and configure the research environment.
    """
    if force and not auto_approve:
        confirm = typer.confirm(
            "[bold red]WARNING: --force will DESTROY existing infrastructure "
            "before rebuilding. Continue?[/bold red]"
        )
        if not confirm:
            raise typer.Abort()

    from orchestrator.actions.setup import provider as setup_provider

    setup_provider.run_setup(infra_path=infra_path, force=force, verbose=verbose)


test_app = typer.Typer(
    help="[bold blue]Test[/bold blue]: Run performance experiments (k6/wrk).",
    rich_markup_mode="rich",
)
app.add_typer(test_app, name="test")


@test_app.command(name="load")
def test_load(
    apps: Annotated[
        str | None,
        typer.Option(
            "--apps",
            "--app",
            "-a",
            help="Filter apps by name (partial matches, comma-separated).",
            rich_help_panel="Scope",
        ),
    ] = None,
    num_runs: Annotated[
        int,
        typer.Option(
            "--num-runs",
            "-n",
            help="Number of times to repeat the entire experiment cycle.",
            show_default=True,
            rich_help_panel="General",
        ),
    ] = 1,
    inter_run_delay: Annotated[
        str,
        typer.Option(
            "--inter-run-delay",
            "--delay",
            help="Wait duration between successive runs (e.g., 1m, 30s).",
            show_default=True,
            rich_help_panel="General",
        ),
    ] = "1m",
    duration: Annotated[
        str,
        typer.Option(
            "--duration",
            "-d",
            help="Duration of the steady-state load phase.",
            show_default=True,
            rich_help_panel="Timeline",
        ),
    ] = "1m",
    warmup: Annotated[
        str,
        typer.Option(
            "--warmup",
            "-w",
            help="Duration of the initial ramp-up phase.",
            show_default=True,
            rich_help_panel="Timeline",
        ),
    ] = "30s",
    after: Annotated[
        str,
        typer.Option(
            "--after",
            help="Duration of the cooldown/quiescence phase after the test.",
            show_default=True,
            rich_help_panel="Timeline",
        ),
    ] = "30s",
    vus: Annotated[
        int,
        typer.Option(
            "--vus",
            "-v",
            help="Maximum number of concurrent Virtual Users (VUs) to allocate.",
            show_default=True,
            rich_help_panel="Workload",
        ),
    ] = 10,
    rps: Annotated[
        int | None,
        typer.Option(
            "--rps",
            "-r",
            help="Target requests per second (RPS) for the steady phase.",
            rich_help_panel="Workload",
        ),
    ] = None,
    skip_assets: Annotated[
        bool,
        typer.Option(
            "--skip-assets",
            help="Disable static asset fetching (JS/CSS) to reduce load generator overhead.",
            rich_help_panel="Workload",
        ),
    ] = False,
    auto_approve: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the experiment plan confirmation prompt.",
            rich_help_panel="General",
        ),
    ] = False,
) -> None:
    """Run standard [cyan]load[/cyan] tests."""
    from orchestrator.actions.test import cli as test_cli

    test_cli.run(
        mode="load",
        apps=apps,
        num_runs=num_runs,
        inter_run_delay=inter_run_delay,
        duration=duration,
        warmup=warmup,
        after=after,
        vus=vus,
        rps=rps,
        skip_assets=skip_assets,
        auto_approve=auto_approve,
    )


@test_app.command(name="capacity")
def test_capacity(
    apps: Annotated[
        str | None,
        typer.Option(
            "--apps",
            "--app",
            "-a",
            help="Filter apps by name (partial matches, comma-separated).",
            rich_help_panel="Scope",
        ),
    ] = None,
    num_runs: Annotated[
        int,
        typer.Option(
            "--num-runs",
            "-n",
            help="Number of times to repeat the entire experiment cycle.",
            show_default=True,
            rich_help_panel="General",
        ),
    ] = 1,
    inter_run_delay: Annotated[
        str,
        typer.Option(
            "--inter-run-delay",
            "--delay",
            help="Wait duration between successive runs (e.g., 1m, 30s).",
            show_default=True,
            rich_help_panel="General",
        ),
    ] = "1m",
    peak_rate: Annotated[
        int,
        typer.Option(
            "--peak-rate",
            "-r",
            help="Target RPS at the top of the ramp.",
            show_default=True,
            rich_help_panel="Workload",
        ),
    ] = 1000,
    ramp_up: Annotated[
        str,
        typer.Option(
            "--ramp-up",
            help="Duration to ramp up from start_rate to peak_rate.",
            show_default=True,
            rich_help_panel="Timeline",
        ),
    ] = "5m",
    sustain: Annotated[
        str,
        typer.Option(
            "--sustain",
            help="Duration to maintain the peak load.",
            show_default=True,
            rich_help_panel="Timeline",
        ),
    ] = "1m",
    ramp_down: Annotated[
        str,
        typer.Option(
            "--ramp-down",
            help="Duration to ramp down from peak_rate to zero.",
            show_default=True,
            rich_help_panel="Timeline",
        ),
    ] = "1m",
    start_rate: Annotated[
        int,
        typer.Option(
            "--start-rate",
            help="RPS to start the ramp from.",
            show_default=True,
            rich_help_panel="Workload",
        ),
    ] = 1,
    peak_rate_2: Annotated[
        int | None,
        typer.Option(
            "--secondary-peak-rate",
            "--peak-rate-2",
            help="Target RPS for an optional second peak (e.g., spike testing).",
            rich_help_panel="Workload",
        ),
    ] = None,
    ramp_up_2: Annotated[
        str | None,
        typer.Option(
            "--secondary-ramp-up",
            "--ramp-up-2",
            help="Duration to ramp from peak_rate to secondary_peak_rate.",
            rich_help_panel="Timeline",
        ),
    ] = None,
    skip_assets: Annotated[
        bool,
        typer.Option(
            "--skip-assets",
            help="Disable static asset fetching (JS/CSS) to reduce load generator overhead.",
            rich_help_panel="Workload",
        ),
    ] = False,
    auto_approve: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the experiment plan confirmation prompt.",
            rich_help_panel="General",
        ),
    ] = False,
) -> None:
    """Run [cyan]capacity[/cyan] tests (Thesis Standard RPS Ramping)."""
    from orchestrator.actions.test import cli as test_cli

    test_cli.run(
        mode="capacity",
        apps=apps,
        num_runs=num_runs,
        inter_run_delay=inter_run_delay,
        peak_rate=peak_rate,
        ramp_up=ramp_up,
        sustain=sustain,
        ramp_down=ramp_down,
        start_rate=start_rate,
        peak_rate_2=peak_rate_2,
        ramp_up_2=ramp_up_2,
        skip_assets=skip_assets,
        auto_approve=auto_approve,
    )


@test_app.command(name="file")
def test_file(
    path: Annotated[str, typer.Argument(help="Path to custom experiment configuration YAML file.")],
    num_runs: Annotated[
        int | None,
        typer.Option("--num-runs", "-n", help="Override the number of runs specified in the YAML."),
    ] = None,
    inter_run_delay: Annotated[
        str | None,
        typer.Option(
            "--inter-run-delay",
            "--delay",
            help="Override wait duration between runs (e.g., 1m, 30s).",
        ),
    ] = None,
    duration: Annotated[
        str | None, typer.Option("--duration", "-d", help="Override steady-state duration.")
    ] = None,
    warmup: Annotated[
        str | None, typer.Option("--warmup", "-w", help="Override warmup duration.")
    ] = None,
    after: Annotated[
        str | None, typer.Option("--after", help="Override cooldown/quiescence duration.")
    ] = None,
    vus: Annotated[
        int | None, typer.Option("--vus", "-v", help="Override maximum Virtual Users.")
    ] = None,
    peak_rate_2: Annotated[
        int | None,
        typer.Option(
            "--secondary-peak-rate",
            "--peak-rate-2",
            help="Override secondary peak RPS.",
        ),
    ] = None,
    ramp_up_2: Annotated[
        str | None,
        typer.Option(
            "--secondary-ramp-up",
            "--ramp-up-2",
            help="Override secondary ramp-up duration.",
        ),
    ] = None,
    skip_assets: Annotated[
        bool | None,
        typer.Option(
            "--skip-assets", help="Override asset skipping (True to disable asset fetching)."
        ),
    ] = None,
    auto_approve: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the experiment plan confirmation prompt.",
        ),
    ] = False,
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
        skip_assets=skip_assets,
        auto_approve=auto_approve,
    )


@test_app.command(name="wrk")
def test_wrk(
    apps: Annotated[
        str | None,
        typer.Option(
            "--apps",
            "--app",
            "-a",
            help="Filter apps by name (partial matches, comma-separated).",
        ),
    ] = None,
    num_runs: Annotated[
        int,
        typer.Option("--num-runs", "-n", help="Number of test runs per app", show_default=True),
    ] = 1,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
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
        list[str] | None,
        typer.Option(help="Two technologies to compare for 'champions' report"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Force analysis of inconsistent data (Research Contract violation)"
        ),
    ] = False,
) -> None:
    """
    [bold magenta]Analyze[/bold magenta]: Generate statistical reports and plots from results.
    """
    console.print(f"[yellow]Analysis initiated for: {results_dir}[/yellow]")
    from orchestrator.actions.analyze import cli as analyze_cli

    analyze_cli.run(
        results_dir=results_dir, report_type=report_type, champions=champions, force=force
    )


@app.command()
def destroy(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
) -> None:
    """
    [bold red]Destroy[/bold red]: Tear down all infrastructure via Terraform.
    """
    console.print("[red]Teardown initiated...[/red]")
    from orchestrator.actions.destroy import cli as destroy_cli

    destroy_cli.run(verbose=verbose)


@app.command()
def campaign(
    path: Annotated[Path, typer.Argument(help="Path to campaign experiment configuration YAML")],
    apps: Annotated[
        str | None,
        typer.Option(
            "--apps",
            help="Comma-separated list of apps to include (overrides YAML)",
        ),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Path to campaign directory to resume from"),
    ] = None,
    infra: Annotated[
        Path | None,
        typer.Option("--infra", help="Path to infrastructure config (defaults to infra.yaml)"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
) -> None:
    """
    [bold magenta]Campaign[/bold magenta]: Run a sequential research campaign
    (Provision -> Warmup -> Test -> Rotate).
    """
    from orchestrator.actions.campaign import cli as campaign_cli

    campaign_cli.run(path=path, apps=apps, resume=resume, infra=infra, verbose=verbose)


@app.command()
def aggregate(
    sources: Annotated[
        list[str],
        typer.Argument(help="Source directories with filters, e.g. 'path[app1,app2]'"),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory. Defaults to a timestamped folder in results/",
        ),
    ] = None,
    lax: Annotated[
        bool,
        typer.Option(
            "--lax", help="Allow aggregation of runs with different parameters (dangerous!)"
        ),
    ] = False,
    no_logs: Annotated[
        bool,
        typer.Option("--no-logs", help="Do not copy log files"),
    ] = False,
) -> None:
    """
    [bold blue]Aggregate[/bold blue]: Combine multiple test runs into a single sequential dataset.
    """
    from orchestrator.actions.aggregate import cli as aggregate_cli

    aggregate_cli.run(sources=sources, output=output, lax=lax, copy_logs=not no_logs)


@app.command()
def verify(
    apps: Annotated[
        str | None,
        typer.Option(
            "--apps",
            "--app",
            help=(
                "Filter apps by name. Supports [bold]partial matches[/bold] and "
                "[bold]comma-separated lists[/bold] (e.g., 'nextjs,react')."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
) -> None:
    """
    [bold green]Verify[/bold green]: Build and test apps locally to ensure functionality.
    """
    from orchestrator.actions.verify import cli as verify_cli

    verify_cli.run(app_filter=apps, verbose=verbose)


@app.command()
def run(
    app: Annotated[
        str | None,
        typer.Argument(
            help=(
                "App name or filter. If provided, runs that app directly. "
                "Otherwise shows interactive selection menu."
            ),
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind (default: 3000). Finds available automatically if in use.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show real-time output from tools")
    ] = False,
) -> None:
    """
    [bold blue]Run[/bold blue]: Start a selected app locally for manual testing in browser.

    Interactive selection if no app specified. Streams logs and cleans up on Ctrl+C.
    """
    from orchestrator.actions.run import cli as run_cli

    run_cli.run(app_filter=app, port=port, verbose=verbose)


if __name__ == "__main__":
    app()
