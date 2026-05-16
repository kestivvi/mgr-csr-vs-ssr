from orchestrator.actions.run.provider import run_subject


def run(subject_filter: str | None = None, port: int | None = None, verbose: bool = False) -> None:
    """Entry point for the run command."""
    run_subject(subject_filter=subject_filter, port=port, verbose=verbose)
