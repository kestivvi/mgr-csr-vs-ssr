import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from orchestrator.config import APPS_DIR, VERIFY_LOGS_BASE_DIR
from orchestrator.shared.runner import run_command

console = Console()


def run_verify(app_filter: str | None = None) -> None:
    """Orchestrates the verification of applications."""
    # 1. Prepare log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_log_dir = VERIFY_LOGS_BASE_DIR / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Starting verification run: {timestamp}[/bold green]")
    console.print(f"[yellow]Logs will be stored in: {run_log_dir}[/yellow]")

    # 2. Find apps
    apps = sorted(
        [d for d in APPS_DIR.iterdir() if d.is_dir() and (d / "docker-compose.yml").exists()]
    )
    if app_filter:
        filters = [f.strip() for f in app_filter.split(",")]
        apps = [a for a in apps if any(f in a.name for f in filters)]

    if not apps:
        console.print("[bold red]No apps found to verify![/bold red]")
        return

    console.print(f"[bold blue]Found {len(apps)} apps to verify:[/bold blue]")
    for app in apps:
        console.print(f" - {app.name}")

    results = []

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Verifying apps...", total=len(apps))

        for app_path in apps:
            app_name = app_path.name

            build_log = run_log_dir / f"{app_name}-build.txt"
            run_log = run_log_dir / f"{app_name}-run.txt"
            test_log = run_log_dir / f"{app_name}-test.txt"

            app_result = {
                "App": app_name,
                "Build": "SKIP",
                "Run": "SKIP",
                "Test": "SKIP",
                "Status": "UNKNOWN",
            }

            # Build
            progress.update(task, description=f"[cyan]Building [bold]{app_name}[/bold]...")
            rc = run_command(
                ["docker-compose", "build"], cwd=str(app_path), log_path=build_log, quiet=True
            )
            app_result["Build"] = "PASS" if rc == 0 else "FAIL"

            if rc == 0:
                # Up
                progress.update(task, description=f"[cyan]Starting [bold]{app_name}[/bold]...")
                rc = run_command(
                    ["docker-compose", "up", "-d", "--force-recreate"],
                    cwd=str(app_path),
                    log_path=run_log,
                    quiet=True,
                )
                app_result["Run"] = "PASS" if rc == 0 else "FAIL"

                if rc == 0:
                    # Test
                    progress.update(task, description=f"[cyan]Testing [bold]{app_name}[/bold]...")
                    test_result = test_app_with_curl(
                        app_path, test_log, quiet=False, progress=progress
                    )
                    app_result["Test"] = "PASS" if test_result else "FAIL"

                    # Append container logs to run log
                    run_command(
                        ["docker-compose", "logs"],
                        cwd=str(app_path),
                        log_path=run_log,
                        quiet=True,
                    )

                # Down
                run_command(
                    ["docker-compose", "down"], cwd=str(app_path), log_path=run_log, quiet=True
                )

            # Overall Status
            if (
                app_result["Build"] == "PASS"
                and app_result["Run"] == "PASS"
                and app_result["Test"] == "PASS"
            ):
                app_result["Status"] = "PASS"
                progress.console.print(
                    f"[bold green]✓ {app_name} passed verification.[/bold green]"
                )
            else:
                app_result["Status"] = "FAIL"
                progress.console.print(f"[bold red]✗ {app_name} failed verification.[/bold red]")

            results.append(app_result)
            progress.advance(task)

    # 4. Generate results.md
    from tabulate import tabulate  # type: ignore[import-untyped]

    summary_table = tabulate(results, headers="keys", tablefmt="github")
    results_file = run_log_dir / "results.md"
    with open(results_file, "w") as f:
        f.write(f"# Verification Results - {timestamp}\n\n")
        f.write(summary_table)
        f.write("\n")

    # Summary console output
    console.print("\n[bold green]Verification Summary:[/bold green]")
    console.print(summary_table)
    console.print(f"\n[yellow]Full logs available in: {run_log_dir}[/yellow]")
    console.print(f"[yellow]Results table: {results_file}[/yellow]")


def test_app_with_curl(
    app_path: Path, log_path: Path, quiet: bool = False, progress: Any = None
) -> bool:
    """Tests the app's root endpoint using curl with retries."""
    app_name = app_path.name
    output = progress.console if progress else console

    if not quiet:
        output.print(f"[cyan]Testing {app_name} at localhost:80...[/cyan]")

    max_retries = 15
    delay = 2

    # Initialize log file with a header
    with open(log_path, "w") as f:
        f.write(f"Testing {app_name} at localhost:80\n")

    for i in range(max_retries):
        # -i for headers, -s for silent, -L to follow redirects
        rc = run_command(
            ["curl", "-isL", "http://localhost:80"],
            cwd=str(app_path),
            log_path=log_path,
            quiet=True,  # Always quiet for the curl command itself to avoid clutter
        )

        if rc == 0:
            if not quiet:
                output.print(f"[bold green]Success! App {app_name} responded.[/bold green]")
            return True

        if not quiet:
            output.print(
                f"[yellow]Attempt {i + 1}/{max_retries}: {app_name} not ready, retrying...[/yellow]"
            )
        time.sleep(delay)

    if not quiet:
        output.print(
            f"[bold red]Test failed for {app_name} after {max_retries} attempts.[/bold red]"
        )
    return False
