from orchestrator.actions.test.runner import TestRunner
from orchestrator.config import resolve_path


def run(config: str) -> None:
    config_path = resolve_path(config)
    runner = TestRunner(config_path)
    runner.run_all()
