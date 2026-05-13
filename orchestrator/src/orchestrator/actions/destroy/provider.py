from rich.console import Console
from orchestrator.shared.infra import CloudEnvironment, InfrastructureError

console = Console()


def run_destroy(verbose: bool = False) -> None:
    """Tears down infrastructure using deep adapters."""
    console.print("[bold red]Destroying Infrastructure[/bold red]")
    env = CloudEnvironment()
    
    try:
        env.teardown(verbose=verbose)
        console.print("[bold green]Infrastructure destroyed successfully.[/bold green]")
    except InfrastructureError as e:
        console.print(f"[bold red]Teardown failed: {e}[/bold red]")
