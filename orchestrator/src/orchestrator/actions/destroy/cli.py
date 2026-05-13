from orchestrator.actions.destroy.provider import run_destroy


def run(verbose: bool = False) -> None:
    run_destroy(verbose=verbose)
