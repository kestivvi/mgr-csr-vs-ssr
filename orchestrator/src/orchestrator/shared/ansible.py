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
                "plugins path is missing. Using default strategy."
            )
            env["ANSIBLE_STRATEGY"] = "linear"
    except ImportError:
        env["ANSIBLE_STRATEGY"] = "linear"

    return env
