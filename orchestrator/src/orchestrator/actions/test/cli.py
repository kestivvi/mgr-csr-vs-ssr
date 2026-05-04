from typing import Any, Dict, Optional

from orchestrator.actions.test.runner import TestRunner
from orchestrator.config import resolve_path


def run(
    mode: str,
    path: Optional[str] = None,
    num_runs: Optional[int] = None,
    inter_run_delay: Optional[str] = None,
    duration: Optional[str] = None,
    warmup: Optional[str] = None,
    after: Optional[str] = None,
    vus: Optional[int] = None,
    rps: Optional[int] = None,
    peak_rate: Optional[int] = None,
    ramp_up: Optional[str] = None,
    sustain: Optional[str] = None,
    ramp_down: Optional[str] = None,
    start_rate: Optional[int] = None,
    peak_rate_2: Optional[int] = None,
    ramp_up_2: Optional[str] = None,
) -> None:
    """
    Core test execution logic.
    """
    overrides: Dict[str, Any] = {}
    if num_runs is not None:
        overrides["num_runs"] = num_runs
    if inter_run_delay is not None:
        overrides["inter_run_delay"] = inter_run_delay
    if duration is not None:
        overrides["duration"] = duration
    if warmup is not None:
        overrides["warmup"] = warmup
    if after is not None:
        overrides["after"] = after
    if vus is not None:
        overrides["vus"] = vus
    if rps is not None:
        overrides["rps"] = rps

    # Capacity parameters
    if peak_rate is not None:
        overrides["peak_rate"] = peak_rate
    if ramp_up is not None:
        overrides["ramp_up"] = ramp_up
    if sustain is not None:
        overrides["sustain"] = sustain
    if ramp_down is not None:
        overrides["ramp_down"] = ramp_down
    if start_rate is not None:
        overrides["start_rate"] = start_rate
    if peak_rate_2 is not None:
        overrides["peak_rate_2"] = peak_rate_2
    if ramp_up_2 is not None:
        overrides["ramp_up_2"] = ramp_up_2

    if mode == "file":
        if path is None:
            raise ValueError("Config file path must be provided for 'file' mode.")
        config_path = resolve_path(path)
        runner = TestRunner(config_path, overrides=overrides)
    else:
        # For 'load' and 'capacity', we just use the values passed from the CLI
        # as the initial config.
        config_dict = {"test_type": "capacity_k6" if mode == "capacity" else "load"}
        runner = TestRunner(None, overrides=overrides, config_dict=config_dict)

    runner.run_all()


def run_local_wrk(app_filter: Optional[str] = None, num_runs: int = 1) -> None:
    """
    Run local capacity testing with wrk.
    """
    from orchestrator.actions.test.local_wrk import run_capacity_local_wrk

    run_capacity_local_wrk(app_filter=app_filter, num_runs=num_runs)
