import ansible_runner
from rich.console import Console

from orchestrator.config import ANSIBLE_DIR, TERRAFORM_DIR
from orchestrator.shared.runner import run_command

console = Console()


def run_setup(force: bool = False) -> None:
    """Orchestrates the infrastructure setup."""

    # 1. Terraform Init & Apply
    console.print("[bold green]Step 1: Terraform Provisioning[/bold green]")
    run_command(["terraform", "init"], cwd=str(TERRAFORM_DIR))

    apply_cmd = ["terraform", "apply", "-auto-approve"]
    if force:
        apply_cmd.append("-replace=...")  # Optional: handle force logic if needed

    import os

    env = os.environ.copy()
    env["TF_IN_AUTOMATION"] = "true"

    rc = run_command(apply_cmd, cwd=str(TERRAFORM_DIR), env=env)
    if rc != 0:
        console.print("[bold red]Terraform failed. Aborting.[/bold red]")
        return

    # 2. Ansible Configuration
    console.print("[bold green]Step 2: Ansible Configuration[/bold green]")

    from orchestrator.shared.ansible import get_ansible_env

    # Using ansible-runner with dynamic environment
    r = ansible_runner.run(
        private_data_dir=str(ANSIBLE_DIR),
        playbook="site.yml",
        quiet=False,
        envvars=get_ansible_env(),
    )

    if r.rc != 0:
        console.print(f"[bold red]Ansible failed with return code {r.rc}[/bold red]")
    else:
        console.print("[bold green]Infrastructure is ready![/bold green]")
