import os
import socket
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from orchestrator.config import SUBJECTS_DIR
from orchestrator.shared.infra import InfrastructureError, LocalEnvironment

console = Console()


def _is_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _find_available_port(preferred_port: int = 3000) -> int:
    """Find an available port starting from preferred_port."""
    port = preferred_port
    while port < 9000:
        if _is_port_available(port):
            return port
        port += 1
    raise RuntimeError(f"No available ports found between {preferred_port} and 9000")


def _discover_subjects() -> list[Path]:
    """Discover all available subjects with Dockerfiles."""
    subjects = sorted(
        [
            d
            for d in SUBJECTS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "Dockerfile").exists()
        ]
    )
    return subjects


def _select_subject(subjects: list[Path], filter_str: str | None = None) -> Path:
    """Select a subject interactively or by filter."""
    if filter_str:
        filters = [f.strip().lower() for f in filter_str.split(",")]
        filtered = [s for s in subjects if any(f in s.name.lower() for f in filters)]
        if len(filtered) == 1:
            return filtered[0]
        elif len(filtered) > 1:
            console.print(f"\n[yellow]Multiple subjects match '{filter_str}':[/yellow]")
            for i, subject in enumerate(filtered, 1):
                console.print(f"  {i}. {subject.name}")
            choice = Prompt.ask(
                "Select subject", choices=[str(i) for i in range(1, len(filtered) + 1)]
            )
            return filtered[int(choice) - 1]
        else:
            console.print(f"[red]No subjects match '{filter_str}'[/red]")
            sys.exit(1)

    # Interactive selection
    console.print("\n[bold cyan]Available subjects:[/bold cyan]")

    ssr_subjects = [s for s in subjects if s.name.startswith("ssr-")]
    csr_subjects = [s for s in subjects if s.name.startswith("csr-")]

    index = 1
    subject_map: dict[int, Path] = {}

    if ssr_subjects:
        console.print("\n[bold]Server-Side Rendering (SSR):[/bold]")
        for s in ssr_subjects:
            console.print(f"  {index:2d}. {s.name}")
            subject_map[index] = s
            index += 1

    if csr_subjects:
        console.print("\n[bold]Client-Side Rendering (CSR):[/bold]")
        for s in csr_subjects:
            console.print(f"  {index:2d}. {s.name}")
            subject_map[index] = s
            index += 1

    choice = Prompt.ask("\nSelect subject number", choices=[str(i) for i in subject_map.keys()])
    return subject_map[int(choice)]


def _stream_logs_interactive(
    subject_path: Path, compose_file: Path, env_vars: dict[str, str]
) -> None:
    """Stream docker-compose logs interactively until Ctrl+C."""
    try:
        cmd = ["docker-compose", "-f", str(compose_file), "logs", "-f"]
        process = subprocess.Popen(
            cmd,
            env={**os.environ, **env_vars},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if line:
                    console.print(f"[dim]{line.rstrip()}[/dim]")

        process.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping logs stream...[/yellow]")


def preview_subject(
    subject_filter: str | None = None, port: int | None = None, verbose: bool = False
) -> None:
    """Preview a selected subject locally using docker-compose."""
    # Discover subjects
    subjects = _discover_subjects()
    if not subjects:
        console.print("[bold red]No subjects found![/bold red]")
        sys.exit(1)

    # Select subject
    selected_subject = _select_subject(subjects, subject_filter)
    subject_id = selected_subject.name

    # Determine port
    if port is None:
        port = 3000

    if not _is_port_available(port):
        available = _find_available_port(port)
        console.print(f"[yellow]Port {port} is in use. Using {available} instead.[/yellow]")
        port = available

    console.print(f"\n[bold green]Starting {subject_id}...[/bold green]")

    try:
        env = LocalEnvironment(selected_subject)

        # Customize port in env
        env_vars = env.docker._get_env()
        env_vars["HOST_PORT"] = str(port)
        # We keep APP_PORT at default (3000) because it's used for internal backend connection

        # Build
        console.print("[cyan]Building image...[/cyan]")
        try:
            env.docker.build(verbose=verbose)
        except InfrastructureError as e:
            console.print(f"[bold red]Build failed:[/bold red] {e}")
            sys.exit(1)

        # Start containers
        console.print("[cyan]Starting containers...[/cyan]")
        try:
            # Run docker-compose with custom port
            cmd = [
                "docker-compose",
                "-f",
                str(env.docker.compose_file),
                "up",
                "-d",
                "--force-recreate",
            ]
            subprocess.run(
                cmd,
                env={**os.environ, **env_vars},
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Failed to start subject:[/bold red] {e.stderr.decode()}")
            sys.exit(1)

        # Show access info
        console.print("\n[bold green]✓ Subject started successfully![/bold green]")
        console.print(f"[cyan]Access the subject at:[/cyan] [bold]http://localhost:{port}[/bold]")
        container_suffix = "runner" if subject_id.startswith("ssr-") else "webserver"
        container_name = f"mgr-{env.docker.subject_id}-{container_suffix}"
        console.print(f"[gray]Container:[/gray] {container_name}")
        console.print("[gray]Press Ctrl+C to stop and clean up...[/gray]\n")

        # Stream logs
        _stream_logs_interactive(selected_subject, env.docker.compose_file, env_vars)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping subject...[/yellow]")
    finally:
        # Cleanup: stop and remove containers
        console.print("[cyan]Cleaning up containers...[/cyan]")
        try:
            env = LocalEnvironment(selected_subject)
            env_vars = env.docker._get_env()
            env_vars["HOST_PORT"] = str(port)

            cmd = ["docker-compose", "-f", str(env.docker.compose_file), "down"]
            subprocess.run(
                cmd,
                env={**os.environ, **env_vars},
                capture_output=True,
            )
            console.print("[bold green]✓ Cleanup complete[/bold green]")
        except Exception as e:
            console.print(f"[yellow]Warning: cleanup failed: {e}[/yellow]")
