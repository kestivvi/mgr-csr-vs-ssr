import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from orchestrator.config import APPS_DIR, VERIFY_LOGS_BASE_DIR
from orchestrator.shared.runner import run_command

console = Console()

MAX_RETRIES = 15
RETRY_DELAY = 2


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
                    progress.update(task, description=f"[cyan]Building [bold]{app_name}[/bold]...")
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
                        f"[bold green]\u2713 {app_name} passed verification.[/bold green]"
                    )
                else:
                    app_result["Status"] = "FAIL"
                    progress.console.print(
                        f"[bold red]\u2717 {app_name} failed verification.[/bold red]"
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


def test_app_with_curl(
    app_path: Path, log_path: Path, quiet: bool = False, progress: Any = None
) -> bool:
    """Refactored entry point for app verification using curl."""
    app_name = app_path.name
    is_ssr = app_name.startswith("ssr-")
    output = progress.console if progress else console

    if not quiet:
        output.print(
            f"[cyan]Testing {app_name} (root, favicon, dynamic) at localhost:80 & 443...[/cyan]"
        )

    for i in range(MAX_RETRIES):
        with open(log_path, "a") as f:
            f.write(f"\n\n--- Attempt {i + 1} for {app_name} ---\n")

        success = False
        try:
            if is_ssr:
                success = _run_ssr_verification(app_path, log_path)
            else:
                success = _run_csr_verification(app_path, log_path)

            if success:
                if not quiet:
                    tech = "SSR" if is_ssr else "CSR"
                    msg = f"Success! {app_name} ({tech}) HTTP, HTTPS, and Dynamic Path are OK"
                    if is_ssr:
                        msg += ", and Content matches"
                    output.print(f"[bold green]{msg}.[/bold green]")
                return True

        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"Error during verification: {e}\n")

        if not quiet:
            output.print(
                f"[yellow]Attempt {i + 1}: Verification failed for {app_name}, "
                f"retrying in {RETRY_DELAY}s...[/yellow]"
            )
        time.sleep(RETRY_DELAY)

    if not quiet:
        output.print(
            f"[bold red]Test failed for {app_name} after {MAX_RETRIES} attempts.[/bold red]"
        )
    return False


def _run_ssr_verification(app_path: Path, log_path: Path) -> bool:
    """Runs specific tests for SSR applications (Connectivity + Content + Gzip)."""
    targets = [
        ("HTTP", "http://localhost:80"),
        ("HTTPS", "https://localhost:443"),
    ]

    for _proto, base_url in targets:
        # 1. Root Page (200 OK + Hello World + Gzip)
        if not _verify_endpoint(base_url, app_path, log_path, check_content=True, verify_gzip=True):
            return False

        # 2. Favicon (200 OK - Gzip usually disabled for small assets)
        if not _verify_endpoint(f"{base_url}/favicon.ico", app_path, log_path, headers_only=True):
            return False

        # 3. Dynamic Page (200 OK + Hello World + Gzip)
        if not _verify_endpoint(
            f"{base_url}/dynamic/verify",
            app_path,
            log_path,
            check_content=True,
            verify_gzip=True,
        ):
            return False

    return True


def _run_csr_verification(app_path: Path, log_path: Path) -> bool:
    """Runs specific tests for CSR applications (Connectivity + Gzip)."""
    targets = [
        ("HTTP", "http://localhost:80"),
        ("HTTPS", "https://localhost:443"),
    ]

    for _proto, base_url in targets:
        # 1. Root Page (200 OK + Gzip)
        if not _verify_endpoint(base_url, app_path, log_path, verify_gzip=True):
            return False

        # 2. Favicon (200 OK)
        if not _verify_endpoint(f"{base_url}/favicon.ico", app_path, log_path, headers_only=True):
            return False

        # 3. Dynamic Page (200 OK + Gzip)
        if not _verify_endpoint(f"{base_url}/dynamic/verify", app_path, log_path, verify_gzip=True):
            return False

    return True


def _verify_endpoint(
    url: str,
    app_path: Path,
    log_path: Path,
    check_content: bool = False,
    headers_only: bool = False,
    verify_gzip: bool = False,
) -> bool:
    """Executes curl request(s) and validates the result, optionally checking gzip."""
    modes: list[tuple[str, list[str]]] = [("standard", [])]
    if verify_gzip:
        # In gzip mode, we request gzip encoding and tell curl to decompress for content check
        modes.append(("gzip", ["--compressed", "-H", "Accept-Encoding: gzip"]))

    for mode_name, extra_flags in modes:
        flags = ["-IsLk"] if headers_only else ["-isLk"]
        cmd = ["curl"] + flags + extra_flags + [url]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(app_path),
                capture_output=True,
                text=True,
                timeout=10,
            )

            with open(log_path, "a") as f:
                f.write(
                    f"\n>>> Request ({mode_name}): {url} (Flags: {' '.join(flags + extra_flags)})\n"
                )
                if result.stdout:
                    f.write(result.stdout)
                if result.stderr:
                    f.write(f"STDERR: {result.stderr}\n")

            # 1. Check Status Code
            if "HTTP/1.1 200" not in result.stdout and "HTTP/2 200" not in result.stdout:
                return False

            # 2. Check Content (if required)
            if check_content:
                if "hello world" not in result.stdout.lower():
                    return False

            # 3. Check Gzip Header (if in gzip mode)
            if mode_name == "gzip":
                if "content-encoding: gzip" not in result.stdout.lower():
                    return False

        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"CURL EXECUTION ERROR for {url} ({mode_name}): {e}\n")
            return False

    return True
