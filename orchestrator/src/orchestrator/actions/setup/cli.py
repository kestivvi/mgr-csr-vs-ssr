from pathlib import Path

from orchestrator.actions.setup.provider import run_setup


def run(infra_path: Path, force: bool = False) -> None:
    run_setup(infra_path=infra_path, force=force)
