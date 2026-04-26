import subprocess

from rich.console import Console

console = Console()


def run_command(
    command: list[str], cwd: str | None = None, env: dict[str, str] | None = None
) -> int:
    """Runs a command and streams output to the console."""
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

        if process.stdout:
            from rich.text import Text

            for line in process.stdout:
                console.print(Text.from_ansi(line), end="")

        process.wait()
        return process.returncode
    except Exception as e:
        console.print(f"[bold red]Error executing command:[/bold red] {e}")
        return 1
