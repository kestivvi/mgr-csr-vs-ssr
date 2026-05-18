from orchestrator.actions.preview.provider import preview_app


def run(app_filter: str | None = None, port: int | None = None, verbose: bool = False) -> None:
    """Entry point for the preview command."""
    preview_app(app_filter=app_filter, port=port, verbose=verbose)
