from rich.console import Console

from orchestrator.config import TERRAFORM_DIR
from orchestrator.shared.runner import run_command

console = Console()


def run_destroy() -> None:
    """Tears down infrastructure."""
    console.print("[bold red]Destroying Infrastructure[/bold red]")
    run_command(["terraform", "destroy", "-auto-approve"], cwd=str(TERRAFORM_DIR))
