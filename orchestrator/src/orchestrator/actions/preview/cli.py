from orchestrator.actions.preview.provider import preview_subject


def run(subject_filter: str | None = None, port: int | None = None, verbose: bool = False) -> None:
    """Entry point for the preview command."""
    preview_subject(subject_filter=subject_filter, port=port, verbose=verbose)
