import os


def get_ansible_env() -> dict[str, str]:
    """
    Returns a dictionary of environment variables for Ansible.
    Dynamically detects and enables Mitogen if it's installed in the environment.
    """
    env = os.environ.copy()

    # Enable colored output for Ansible
    env["ANSIBLE_FORCE_COLOR"] = "true"

    from rich.console import Console

    console = Console()

    try:
        if os.environ.get("MGR_DISABLE_MITOGEN") == "true":
            raise ImportError("Mitogen disabled by user")
        import ansible_mitogen

        mitogen_dir = os.path.dirname(ansible_mitogen.__file__)
        plugin_path = os.path.join(mitogen_dir, "plugins", "strategy")

        if os.path.exists(plugin_path):
            env["ANSIBLE_STRATEGY_PLUGINS"] = plugin_path
            env["ANSIBLE_STRATEGY"] = "mitogen_host_pinned"
            console.print(
                "[bold cyan]Performance:[/bold cyan] Mitogen detected. "
                "Using [bold magenta]mitogen_host_pinned[/bold magenta] strategy."
            )
        else:
            console.print(
                "[yellow]Warning:[/yellow] ansible_mitogen found but strategy "
                "plugins path is missing. Falling back to [bold green]free[/bold green] strategy."
            )
            env["ANSIBLE_STRATEGY"] = "free"
    except ImportError:
        # Fallback to 'free' strategy which is generally faster than 'linear'
        env["ANSIBLE_STRATEGY"] = "free"
        console.print(
            "[bold yellow]Performance Tip:[/bold yellow] Mitogen not found. "
            "Using [bold green]free[/bold green] strategy. "
            "Install mitogen for 3x-5x speedup: [italic]pip install mitogen[/italic]"
        )

    return env
