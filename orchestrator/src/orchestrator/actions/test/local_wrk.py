import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from orchestrator.config import SUBJECTS_DIR, RESULTS_DIR
from orchestrator.shared.infra import BaseAdapter, InfrastructureError, LocalEnvironment

console = Console()


def parse_wrk_output(output: str) -> float:
    """Parses Requests/sec from wrk output."""
    match = re.search(r"Requests/sec:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    return 0.0


def wait_for_subject_ready(
    env: LocalEnvironment, log_path: Path, progress: Progress | None = None
) -> bool:
    """Waits for the subject to be reachable via HTTPS."""
    max_retries = 15
    delay = 2

    # We use a temporary adapter for the curl check
    curl_adapter = BaseAdapter(env.docker.workdir)

    for _ in range(max_retries):
        try:
            curl_adapter._run(
                ["curl", "-isLk", "https://localhost"],
                log_path=log_path,
            )
            content = log_path.read_text()
            if "200 OK" in content or "HTTP/2 200" in content or "HTTP/1.1 200" in content:
                return True
        except InfrastructureError:
            pass

        time.sleep(delay)
    return False


def run_capacity_local_wrk(
    subject_filter: str | None = None, num_runs: int = 1, verbose: bool = False
) -> None:
    """Orchestrates local capacity testing using wrk."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_log_dir = RESULTS_DIR / f"capacity_local_wrk_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Starting local capacity benchmark (wrk): {timestamp}[/bold green]")
    console.print(f"[yellow]Logs will be stored in: {run_log_dir}[/yellow]")
    console.print(f"[yellow]Number of test runs: {num_runs}[/yellow]")

    # 1. Find subjects
    subjects = sorted(
        [
            d
            for d in SUBJECTS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "Dockerfile").exists()
        ]
    )
    if subject_filter:
        filters = [f.strip() for f in subject_filter.split(",")]
        subjects = [s for s in subjects if any(f in s.name for f in filters)]

    if not subjects:
        console.print("[bold red]No subjects found to benchmark![/bold red]")
        return

    console.print(f"[bold blue]Found {len(subjects)} subjects to benchmark:[/bold blue]")
    for s in subjects:
        console.print(f" - {s.name}")

    results = []

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
            task = progress.add_task("[cyan]Benchmarking subjects...", total=len(subjects))

            for subject_path in subjects:
                subject_id = subject_path.name
                env = LocalEnvironment(subject_path)
                wrk_adapter = BaseAdapter(subject_path)

                subject_result: dict[str, Any] = {
                    "Subject": subject_id,
                    "Status": "FAIL",
                    "RPS_Avg": 0.0,
                }

                build_log = run_log_dir / f"{subject_id}-build.txt"
                run_log = run_log_dir / f"{subject_id}-run.txt"
                benchmark_log = run_log_dir / f"{subject_id}-wrk.txt"

                try:
                    # Build
                    progress.update(
                        task, description=f"[cyan]Building [bold]{subject_id}[/bold]..."
                    )
                    try:
                        env.docker.build(log_path=build_log, verbose=verbose)

                        # Up
                        progress.update(
                            task, description=f"[cyan]Starting [bold]{subject_id}[/bold]..."
                        )
                        env.docker.up(log_path=run_log, verbose=verbose)

                        # Wait for ready
                        if wait_for_subject_ready(env, run_log, progress):
                            rps_values = []

                            # Warmup
                            progress.update(
                                task,
                                description=f"[cyan]Warmup (10s) [bold]{subject_id}[/bold]...",
                            )
                            wrk_adapter._run(
                                ["wrk", "-t2", "-c100", "-d10s", "https://localhost"],
                                log_path=benchmark_log,
                                verbose=verbose,
                            )

                            # N Runs
                            for i in range(1, num_runs + 1):
                                progress.update(
                                    task,
                                    description=(
                                        f"[cyan]Test Run {i}/{num_runs} (10s) "
                                        f"[bold]{subject_id}[/bold]..."
                                    ),
                                )

                                run_tmp_log = run_log_dir / f"{subject_id}-wrk-run-{i}.txt"
                                wrk_adapter._run(
                                    ["wrk", "-t2", "-c100", "-d10s", "https://localhost"],
                                    log_path=run_tmp_log,
                                    verbose=verbose,
                                )

                                rps = parse_wrk_output(run_tmp_log.read_text())
                                rps_values.append(rps)

                                with open(benchmark_log, "a") as f:
                                    f.write(f"\n--- RUN {i} ---\n")
                                    f.write(run_tmp_log.read_text())

                            if rps_values:
                                avg_rps = sum(rps_values) / len(rps_values)
                                subject_result["RPS_Avg"] = avg_rps
                                subject_result["Status"] = "PASS"
                                progress.console.print(
                                    f"[bold green]✓ {subject_id}: {avg_rps:.2f} req/s "
                                    f"(avg of {num_runs})[/bold green]"
                                )
                        else:
                            progress.console.print(
                                f"[bold red]✗ {subject_id} failed to become ready.[/bold red]"
                            )
                    except InfrastructureError as e:
                        progress.console.print(f"[bold red]✗ {subject_id} failed: {e}[/bold red]")
                finally:
                    # Always try to Down
                    try:
                        env.teardown(verbose=verbose)
                    except InfrastructureError:
                        pass

                results.append(subject_result)
                progress.advance(task)

    except KeyboardInterrupt:
        console.print("\n[bold red]Benchmark interrupted by user. Cleaning up...[/bold red]")

    # 4. Generate report
    if results:
        from tabulate import tabulate

        # Sort results by RPS_Avg descending
        results.sort(key=lambda x: x["RPS_Avg"], reverse=True)

        # Format RPS for display
        display_results = []
        for r in results:
            display_results.append(
                {"Subject": r["Subject"], "Status": r["Status"], "RPS Avg": f"{r['RPS_Avg']:.2f}"}
            )

        summary_table = tabulate(display_results, headers="keys", tablefmt="github")
        results_file = run_log_dir / "results.md"
        with open(results_file, "w") as f:
            f.write(f"# Local Capacity Benchmark Results (wrk) - {timestamp}\n\n")
            f.write("Parameters: `-t2 -c100 -d10s`\n")
            f.write(f"Runs per subject: {num_runs}\n\n")
            f.write(summary_table)
            f.write("\n")

        console.print("\n[bold green]Benchmark Summary:[/bold green]")
        console.print(summary_table)
        console.print(f"\n[yellow]Logs available in: {run_log_dir}[/yellow]")
        console.print(f"[yellow]Results table: {results_file}[/yellow]")
    else:
        console.print("[yellow]No results to show.[/yellow]")
