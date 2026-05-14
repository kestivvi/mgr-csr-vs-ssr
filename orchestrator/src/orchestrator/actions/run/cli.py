from orchestrator.actions.run.provider import run_app


def run(app_filter: str | None = None, port: int | None = None, verbose: bool = False) -> None:
    """Entry point for the run command."""
    run_app(app_filter=app_filter, port=port, verbose=verbose)
