from orchestrator.actions.analyze.engine import PerformanceAnalyzer
from orchestrator.config import resolve_path


def run(
    results_dir: str,
    report_type: str = "load",
    champions: list[str] | None = None,
    force: bool = False,
) -> None:
    input_path = resolve_path(results_dir)
    analyzer = PerformanceAnalyzer(input_path, report_type, champions, force=force)
    analyzer.run()
