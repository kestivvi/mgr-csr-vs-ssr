import subprocess
from pathlib import Path

from rich.console import Console

console = Console()


def run_command(
    command: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    quiet: bool = False,
) -> int:
    """Runs a command and streams output to the console (if not quiet) and optionally to a file."""
    if not quiet:
        console.print(f"[bold cyan]Executing:[/bold cyan] {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        log_file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "a")

        if process.stdout:
            from rich.text import Text

            while True:
                # Read chunk by chunk to avoid line buffering issues
                chunk = process.stdout.read(1024)
                if not chunk:
                    break

                if not quiet:
                    console.print(Text.from_ansi(chunk), end="")

                if log_file:
                    log_file.write(chunk)
                    log_file.flush()

        if log_file:
            log_file.close()

        process.wait()
        return process.returncode
    except Exception as e:
        if not quiet:
            console.print(f"[bold red]Error executing command:[/bold red] {e}")
        return 1
