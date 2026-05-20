import os
import socket
import subprocess
import sys
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel

from orchestrator.config import APPLICATIONS_DIR
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


def _discover_apps() -> list[Path]:
    """Discover all available applications with Dockerfiles."""
    apps = sorted(
        [
            d
            for d in APPLICATIONS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "Dockerfile").exists()
        ]
    )
    return apps


def _select_app(apps: list[Path], filter_str: str | None = None) -> Path:
    """Select an application interactively or by filter."""
    if filter_str:
        filters = [f.strip().lower() for f in filter_str.split(",")]
        filtered = [s for s in apps if any(f in s.name.lower() for f in filters)]
        if len(filtered) == 1:
            return filtered[0]
        elif len(filtered) == 0:
            console.print(f"[red]No applications match '{filter_str}'[/red]")
            sys.exit(1)
        # Multiple matches — narrow list fed into selector below
        apps = filtered

    ssr_apps = [s for s in apps if s.name.startswith("ssr-")]
    csr_apps = [s for s in apps if s.name.startswith("csr-")]

    choices: list[questionary.Choice] = []
    if ssr_apps:
        choices.append(questionary.Choice(title="── SSR ──", value=None, disabled=""))
        choices.extend(questionary.Choice(title=s.name, value=s) for s in ssr_apps)
    if csr_apps:
        choices.append(questionary.Choice(title="── CSR ──", value=None, disabled=""))
        choices.extend(questionary.Choice(title=s.name, value=s) for s in csr_apps)

    selected = questionary.select("Select application:", choices=choices).ask()

    if selected is None:
        sys.exit(0)

    return selected  # type: ignore[no-any-return]


def _stream_logs_interactive(compose_file: Path, env_vars: dict[str, str]) -> None:
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


def preview_app(
    app_filter: str | None = None, port: int | None = None, verbose: bool = False
) -> None:
    """Preview a selected application locally using docker-compose."""
    # Discover applications
    apps = _discover_apps()
    if not apps:
        console.print("[bold red]No applications found![/bold red]")
        sys.exit(1)

    # Select application
    selected_app = _select_app(apps, app_filter)
    app_id = selected_app.name

    # Determine port
    if port is None:
        port = 3000

    if not _is_port_available(port):
        available = _find_available_port(port)
        console.print(
            Panel(
                f"Port [bold]{port}[/bold] is already in use.\n"
                f"Binding to [bold green]{available}[/bold green] instead.",
                title="[yellow]Port conflict[/yellow]",
                border_style="yellow",
            )
        )
        port = available

    console.print(f"\n[bold green]Starting {app_id}...[/bold green]")

    try:
        env = LocalEnvironment(selected_app)

        # Customize port in env
        env_vars = env.docker._get_env()
        env_vars["HOST_PORT"] = str(port)
        # We keep APP_PORT at default (3000) because it's used for internal backend connection

        # Build
        if verbose:
            console.print("[cyan]Building image...[/cyan]")
            try:
                env.docker.build(verbose=True)
            except InfrastructureError as e:
                console.print(f"[bold red]Build failed:[/bold red] {e}")
                sys.exit(1)
        else:
            with console.status("[cyan]Building image...[/cyan]", spinner="dots"):
                try:
                    env.docker.build(verbose=False)
                except InfrastructureError as e:
                    console.print(f"[bold red]Build failed:[/bold red] {e}")
                    sys.exit(1)

        # Start containers
        with console.status("[cyan]Starting containers...[/cyan]", spinner="dots"):
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
                console.print(
                    f"[bold red]Failed to start application:[/bold red] {e.stderr.decode()}"
                )
                sys.exit(1)

        # Show access info
        container_suffix = "runner" if app_id.startswith("ssr-") else "webserver"
        container_name = f"mgr-{env.docker.app_id}-{container_suffix}"
        console.print(
            Panel(
                f"[bold]http://localhost:{port}[/bold]\n"
                f"[dim]Container: {container_name}[/dim]\n\n"
                "[dim]Press Ctrl+C to stop and clean up.[/dim]",
                title="[bold green]✓ Running[/bold green]",
                border_style="green",
            )
        )

        # Stream logs
        _stream_logs_interactive(env.docker.compose_file, env_vars)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping application...[/yellow]")
    finally:
        # Cleanup: stop and remove containers
        console.print("[cyan]Cleaning up containers...[/cyan]")
        try:
            cleanup_env_vars = env.docker._get_env()
            cleanup_env_vars["HOST_PORT"] = str(port)
            cmd = ["docker-compose", "-f", str(env.docker.compose_file), "down"]
            subprocess.run(
                cmd,
                env={**os.environ, **cleanup_env_vars},
                capture_output=True,
            )
            console.print("[bold green]✓ Cleanup complete[/bold green]")
        except Exception as e:
            console.print(f"[yellow]Warning: cleanup failed: {e}[/yellow]")
