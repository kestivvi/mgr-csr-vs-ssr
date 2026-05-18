from datetime import datetime

from rich.console import Console

from orchestrator.config import APPLICATIONS_DIR, VERIFY_LOGS_BASE_DIR
from orchestrator.shared.infra import InfrastructureError, LocalEnvironment
from orchestrator.shared.verifier import CSR_PROFILE, SSR_PROFILE, ApplicationVerifier

console = Console()


def run_verify(app_filter: str | None = None, verbose: bool = False) -> None:
    """Orchestrates the verification of applications using deep adapters."""
    # 1. Prepare log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_log_dir = VERIFY_LOGS_BASE_DIR / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Starting verification run: {timestamp}[/bold green]")
    console.print(f"[yellow]Logs will be stored in: {run_log_dir}[/yellow]")

    # 2. Find applications
    apps = sorted(
        [
            d
            for d in APPLICATIONS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "Dockerfile").exists()
        ]
    )
    if app_filter:
        filters = [f.strip() for f in app_filter.split(",")]
        apps = [s for s in apps if any(f in s.name for f in filters)]

    if not apps:
        console.print("[bold red]No applications found to verify![/bold red]")
        return

    # 3. Manifest Integrity Guard (Fail-Fast)
    for app_path in apps:
        if not (app_path / "application.json").exists():
            raise ValueError(
                f"Missing application.json in {app_path.name}. "
                "All applications must have a manifest."
            )

    console.print(f"[bold blue]Found {len(apps)} applications to verify:[/bold blue]")
    for s in apps:
        console.print(f" - {s.name}")

    results = []

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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
            task = progress.add_task("[cyan]Verifying applications...", total=len(apps))

            for app_path in apps:
                app_id = app_path.name
                env = LocalEnvironment(app_path)

                # Determine health profile
                profile = SSR_PROFILE if app_id.startswith("ssr-") else CSR_PROFILE

                build_log = run_log_dir / f"{app_id}-build.txt"
                run_log = run_log_dir / f"{app_id}-run.txt"
                test_log = run_log_dir / f"{app_id}-test.txt"

                app_result = {
                    "Application": app_id,
                    "Build": "SKIP",
                    "Run": "SKIP",
                    "Test": "SKIP",
                    "Status": "UNKNOWN",
                }

                try:
                    # Build
                    progress.update(task, description=f"[cyan]Building [bold]{app_id}[/bold]...")
                    try:
                        env.docker.build(log_path=build_log, verbose=verbose)
                        app_result["Build"] = "PASS"
                    except InfrastructureError:
                        app_result["Build"] = "FAIL"

                    if app_result["Build"] == "PASS":
                        # Up
                        progress.update(
                            task, description=f"[cyan]Starting [bold]{app_id}[/bold]..."
                        )
                        try:
                            env.docker.up(log_path=run_log, verbose=verbose)
                            app_result["Run"] = "PASS"
                        except InfrastructureError:
                            app_result["Run"] = "FAIL"

                        if app_result["Run"] == "PASS":
                            # Test
                            progress.update(
                                task,
                                description=f"[cyan]Testing [bold]{app_id}[/bold]...",
                            )

                            # Use the new deep Verifier
                            verifier = ApplicationVerifier(
                                workdir=app_path,
                                on_output=lambda msg: progress.console.print(f"  [dim]{msg}[/dim]"),
                            )

                            # We check both HTTP and HTTPS (standard for MGR verify)
                            success = True
                            for base_url in ["http://localhost:80", "https://localhost:443"]:
                                if not verifier.wait_until_healthy(
                                    base_url=base_url, profile=profile, log_path=test_log
                                ):
                                    success = False
                                    break

                            app_result["Test"] = "PASS" if success else "FAIL"

                            # Append container logs to run log
                            try:
                                env.docker.logs(log_path=run_log, verbose=verbose)
                            except InfrastructureError:
                                pass
                finally:
                    # Always try to Down if we attempted Build/Up
                    if app_result["Build"] != "SKIP":
                        try:
                            env.teardown(verbose=verbose)
                        except InfrastructureError:
                            pass

                # Overall Status
                if (
                    app_result["Build"] == "PASS"
                    and app_result["Run"] == "PASS"
                    and app_result["Test"] == "PASS"
                ):
                    app_result["Status"] = "PASS"
                    progress.console.print(
                        f"[bold green]\u2713 {app_id} passed verification.[/bold green]"
                    )
                else:
                    app_result["Status"] = "FAIL"
                    progress.console.print(
                        f"[bold red]\u2717 {app_id} failed verification.[/bold red]"
                    )

                results.append(app_result)
                progress.advance(task)
    except KeyboardInterrupt:
        console.print("\n[bold red]Verification interrupted by user. Cleaning up...[/bold red]")

    # 4. Generate results.md (even for partial runs)
    if results:
        from tabulate import tabulate

        summary_table = tabulate(results, headers="keys", tablefmt="github")
        results_file = run_log_dir / "results.md"
        with open(results_file, "w") as f:
            f.write(f"# Verification Results - {timestamp}\n\n")
            f.write(summary_table)
            f.write("\n")

        # Summary console output
        console.print("\n[bold green]Verification Summary:[/bold green]")
        console.print(summary_table)
        console.print(f"\n[yellow]Logs available in: {run_log_dir}[/yellow]")
        console.print(f"[yellow]Results table: {results_file}[/yellow]")
    else:
        console.print("[yellow]No results to show.[/yellow]")
