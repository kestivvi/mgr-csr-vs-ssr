from orchestrator.actions.verify.provider import run_verify


def run(subject_filter: str | None = None, verbose: bool = False) -> None:
    """Entry point for the verify command."""
    run_verify(subject_filter=subject_filter, verbose=verbose)
