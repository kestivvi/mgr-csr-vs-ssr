from orchestrator.actions.setup.provider import run_setup


def run(force: bool = False) -> None:
    run_setup(force=force)
