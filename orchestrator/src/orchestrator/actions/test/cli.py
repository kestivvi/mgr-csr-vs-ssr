from typing import Any, Dict, Optional
from orchestrator.actions.test.runner import TestRunner
from orchestrator.config import resolve_path

def run(
    mode: str,
    path: Optional[str] = None,
    num_runs: Optional[int] = None,
    duration: Optional[str] = None,
    vus: Optional[int] = None,
    rps: Optional[int] = None,
    peak_rate: Optional[int] = None,
    ramp_up: Optional[str] = None,
    sustain: Optional[str] = None,
    ramp_down: Optional[str] = None,
    start_rate: Optional[int] = None,
) -> None:
    """
    Core test execution logic. 
    """
    overrides: Dict[str, Any] = {}
    if num_runs is not None:
        overrides["num_runs"] = num_runs
    if duration is not None:
        overrides["duration"] = duration
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
