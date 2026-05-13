from orchestrator.actions.verify.provider import run_verify


def run(app_filter: str | None = None, verbose: bool = False) -> None:
    """Entry point for the verify command."""
    run_verify(app_filter=app_filter, verbose=verbose)
