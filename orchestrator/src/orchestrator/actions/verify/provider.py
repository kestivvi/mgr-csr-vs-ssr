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

    try:
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

                try:
                    # Build
                    progress.update(
                        task, description=f"[cyan]Building [bold]{app_name}[/bold]..."
                    )
                    rc = run_command(
                        ["docker-compose", "build"],
                        cwd=str(app_path),
                        log_path=build_log,
                        quiet=True,
                    )
                    app_result["Build"] = "PASS" if rc == 0 else "FAIL"

                    if rc == 0:
                        # Up
                        progress.update(
                            task, description=f"[cyan]Starting [bold]{app_name}[/bold]..."
                        )
                        rc = run_command(
                            ["docker-compose", "up", "-d", "--force-recreate"],
                            cwd=str(app_path),
                            log_path=run_log,
                            quiet=True,
                        )
                        app_result["Run"] = "PASS" if rc == 0 else "FAIL"

                        if rc == 0:
                            # Test
                            progress.update(
                                task,
                                description=f"[cyan]Testing [bold]{app_name}[/bold]...",
                            )
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
                finally:
                    # Always try to Down if we attempted Build/Up
                    if app_result["Build"] != "SKIP":
                        run_command(
                            ["docker-compose", "down"],
                            cwd=str(app_path),
                            log_path=run_log,
                            quiet=True,
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
                    progress.console.print(
                        f"[bold red]✗ {app_name} failed verification.[/bold red]"
                    )

                results.append(app_result)
                progress.advance(task)
    except KeyboardInterrupt:
        console.print(
            "\n[bold red]Verification interrupted by user. Cleaning up...[/bold red]"
        )

    # 4. Generate results.md (even for partial runs)
    if results:
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
        console.print(f"\n[yellow]Logs available in: {run_log_dir}[/yellow]")
        console.print(f"[yellow]Results table: {results_file}[/yellow]")
    else:
        console.print("[yellow]No results to show.[/yellow]")


def test_app_with_curl(
    app_path: Path, log_path: Path, quiet: bool = False, progress: Any = None
) -> bool:
    """Tests the app's root endpoint and favicon using curl over HTTP and HTTPS."""
    app_name = app_path.name
    output = progress.console if progress else console

    if not quiet:
        output.print(
            f"[cyan]Testing {app_name} (root + favicon) at localhost:80 & 443...[/cyan]"
        )

    max_retries = 15
    delay = 2

    # Protocols and ports to test
    targets = [
        ("HTTP", "http://localhost:80"),
        ("HTTPS", "https://localhost:443"),
    ]

    for i in range(max_retries):
        # Clear log file for this attempt
        with open(log_path, "w") as f:
            f.write(f"--- Attempt {i+1} for {app_name} ---\n")

        all_protocols_reachable = True
        for proto, base_url in targets:
            # 1. Test Root Page (using -k for self-signed certs)
            rc_root = run_command(
                ["curl", "-isLk", base_url],
                cwd=str(app_path),
                log_path=log_path,
                quiet=True,
            )

            # 2. Test Favicon (using -k and -I for headers only)
            rc_fav = run_command(
                ["curl", "-IsLk", f"{base_url}/favicon.ico"],
                cwd=str(app_path),
                log_path=log_path,
                quiet=True,
            )

            if rc_root != 0 or rc_fav != 0:
                if not quiet:
                    output.print(
                        f"[yellow]Attempt {i+1}: {proto} connection failed, retrying...[/yellow]"
                    )
                all_protocols_reachable = False
                break

        if not all_protocols_reachable:
            time.sleep(delay)
            continue

        content = log_path.read_text()

        # Check for HTTP 200 in the responses.
        # We expect 4 successful responses (root + favicon for both HTTP and HTTPS).
        # We check for various formats of 200 OK across HTTP/1.1 and HTTP/2.
        success_count = (
            content.count("200 OK")
            + content.count("HTTP/2 200")
            + content.count("HTTP/1.1 200")
        )

        if success_count >= 4:
            if not quiet:
                output.print(
                    f"[bold green]Success! {app_name} HTTP and HTTPS are OK.[/bold green]"
                )
            return True

        if not quiet:
            output.print(
                f"[yellow]Attempt {i+1}: Status check failed (found {success_count}/4 OKs), retrying...[/yellow]"
            )

        time.sleep(delay)

    if not quiet:
        output.print(
            f"[bold red]Test failed for {app_name} after {max_retries} attempts.[/bold red]"
        )
    return False
